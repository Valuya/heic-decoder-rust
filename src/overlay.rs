//! Overlay (`iovl`) derived images, composed the way libheif composes them.
//!
//! The canvas fill colour is RGB, so libheif (`ImageItem_Overlay::
//! decode_overlay_image`, `HeifPixelImage::overlay`) composes in RGB, and so
//! does this module:
//!
//! - The canvas is RGB 4:4:4 at the bit depth of the first input's first coded
//!   image, at least 8. It is filled with `canvas_fill_value` reduced to that
//!   depth by dropping low bits, and has no alpha: the fill's alpha is unused.
//! - Each input is decoded with its own transforms and alpha, converted to RGB
//!   at the canvas depth, and drawn in reference order at its offset, clipped to
//!   the canvas. Without alpha it is copied; with alpha it is blended as
//!   `(input * a + canvas * (max - a)) / max`, truncated. An input deeper than
//!   the canvas is refused, as libheif refuses it.
//! - The overlay item's own transforms then apply to the composed picture.
//!
//! The result is returned as planar 4:4:4 coded with the identity matrix
//! (`matrix_coefficients` 0, full range): green in `y`, blue in `cb`, red in
//! `cr`, which the RGB converters turn back into exactly the composed RGB at
//! 8 bits. Overlays nested in overlays are not supported.

use rayon::prelude::*;

use crate::color::{ChromaSampling, ColorInfo, RgbOptions};
use crate::error::{HeifError, Result};
use crate::hevc::nal::NAL_SPS;
use crate::hevc::params::parse_sps;
use crate::isobmff::{OverlayInfo, ParsedHeif, ResolvedImage};
use crate::{DecodedImage, MAX_IMAGE_PIXELS, apply_transforms, find_and_parse};

fn sps_pixels(image: &ResolvedImage) -> Result<u64> {
    let sps = find_and_parse(image.hevc_config.nalus_of_type(NAL_SPS), NAL_SPS, parse_sps)?;
    Ok(u64::from(sps.pic_width_in_luma_samples) * u64::from(sps.pic_height_in_luma_samples))
}

/// Checks the overlay against limits before anything is decoded or
/// allocated: every coded picture of every input (its tiles, alpha and
/// prediction reference) against `max_picture`; each input grid's canvas and
/// tiles, the overlay canvas, and the pixels decoded for all placements
/// together against `max_total`. An item placed twice is decoded twice.
pub(crate) fn check_limits(overlay: &OverlayInfo, max_picture: u64, max_total: u64) -> Result<()> {
    let too_large = || HeifError::LimitExceeded("overlay: picture larger than allowed");
    if u64::from(overlay.output_width) * u64::from(overlay.output_height) > max_total {
        return Err(too_large());
    }
    let mut decoded = Vec::with_capacity(overlay.images.len());
    for image in &overlay.images {
        let mut pixels = 0u64;
        for part in std::iter::once(image).chain(image.alpha.iter().map(|alpha| alpha.image.as_ref())) {
            let mut coded = 0u64;
            for picture in part.images.iter().chain(&part.reference_image) {
                let n = sps_pixels(picture)?;
                if n > max_picture {
                    return Err(too_large());
                }
                coded = coded.saturating_add(n);
            }
            if let Some(grid) = part.grid {
                let canvas = u64::from(grid.output_width) * u64::from(grid.output_height);
                if canvas > max_total || coded > max_total {
                    return Err(too_large());
                }
                coded = coded.saturating_add(canvas);
            }
            pixels = pixels.saturating_add(coded);
        }
        decoded.push(pixels);
    }
    let mut total = 0u64;
    for input in &overlay.inputs {
        total = total.saturating_add(*decoded.get(input.image).ok_or(HeifError::MalformedBox("overlay: input out of range"))?);
    }
    if total > max_total {
        return Err(too_large());
    }
    Ok(())
}

/// The canvas bit depth: the first input's first coded image's luma depth,
/// at least 8, as libheif reports and composes it.
pub(crate) fn canvas_bit_depth(overlay: &OverlayInfo) -> Result<u32> {
    let missing = || HeifError::MalformedBox("overlay: no input images");
    let first = overlay.inputs.first().and_then(|input| overlay.images.get(input.image)).ok_or_else(missing)?;
    let coded = first.images.first().ok_or_else(missing)?;
    let sps = find_and_parse(coded.hevc_config.nalus_of_type(NAL_SPS), NAL_SPS, parse_sps)?;
    if sps.bit_depth_luma > 16 {
        return Err(HeifError::Unsupported("overlay: more than 16 bits per sample"));
    }
    Ok(sps.bit_depth_luma.max(8))
}

/// One input converted to interleaved RGB (and alpha, if it has one) at the
/// canvas depth.
struct InputPixels {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<u16>,
}

fn input_pixels(image: &DecodedImage, depth: u32, rgb: &RgbOptions) -> Result<InputPixels> {
    let channels = if image.alpha.is_some() { 4 } else { 3 };
    let data = if depth == 8 {
        if image.alpha.is_some() { image.to_rgba8_with_options(rgb)?.data } else { image.to_rgb8_with_options(rgb)?.data }.into_iter().map(u16::from).collect()
    } else {
        // 16 bits, rounded to the canvas depth.
        let max = (1u32 << depth) - 1;
        let samples = if image.alpha.is_some() { image.to_rgba16_with_options(rgb)?.data } else { image.to_rgb16_with_options(rgb)?.data };
        samples.into_iter().map(|v| ((u32::from(v) * max + 32767) / 65535) as u16).collect()
    };
    Ok(InputPixels { width: image.width, height: image.height, channels, data })
}

/// Composes the overlay `parsed` describes. With `transform`, inputs and the
/// overlay get their transforms, as libheif applies or ignores them all. `rgb`
/// converts the inputs.
pub(crate) fn compose(parsed: &ParsedHeif, transform: bool, rgb: &RgbOptions) -> Result<DecodedImage> {
    let overlay = parsed.overlay.as_ref().ok_or(HeifError::MalformedBox("overlay: not an overlay"))?;
    check_limits(overlay, MAX_IMAGE_PIXELS, MAX_IMAGE_PIXELS)?;
    let depth = canvas_bit_depth(overlay)?;
    let max = (1u32 << depth) - 1;
    let (width, height) = (overlay.output_width as usize, overlay.output_height as usize);
    let len = width.checked_mul(height).and_then(|n| n.checked_mul(3)).ok_or(HeifError::LimitExceeded("overlay: canvas size"))?;
    let [r, g, b, _] = overlay.canvas_fill.map(|v| v >> (16 - depth));
    let mut canvas = Vec::new();
    canvas.try_reserve_exact(len).map_err(|_| HeifError::LimitExceeded("overlay: canvas allocation failed"))?;
    canvas.extend(std::iter::repeat_n([r, g, b], width * height).flatten());

    let mut first_color = None;
    for input in &overlay.inputs {
        let source = overlay.images.get(input.image).ok_or(HeifError::MalformedBox("overlay: input out of range"))?;
        let image = crate::decode_item(source, transform, rgb)?;
        if image.bit_depth_luma.max(image.bit_depth_chroma) > depth {
            return Err(HeifError::Unsupported("overlay: an input deeper than the first input"));
        }
        first_color = first_color.or(Some(image.color_info));
        let pixels = input_pixels(&image, depth, rgb)?;
        drop(image);
        // The part of the input inside the canvas, in canvas coordinates.
        let (dx, dy) = (i64::from(input.x), i64::from(input.y));
        let (x0, y0) = (dx.max(0), dy.max(0));
        let (x1, y1) = ((dx + pixels.width as i64).min(width as i64), (dy + pixels.height as i64).min(height as i64));
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let (x0, x1, y0, y1) = (x0 as usize, x1 as usize, y0 as usize, y1 as usize);
        let source_x = (x0 as i64 - dx) as usize;
        canvas.par_chunks_mut(width * 3).enumerate().skip(y0).take(y1 - y0).for_each(|(y, row)| {
            let source_y = (y as i64 - dy) as usize;
            let start = (source_y * pixels.width + source_x) * pixels.channels;
            let Some(source) = pixels.data.get(start..start + (x1 - x0) * pixels.channels) else { return };
            let Some(target) = row.get_mut(x0 * 3..x1 * 3) else { return };
            if pixels.channels == 3 {
                target.copy_from_slice(source);
                return;
            }
            for (out, px) in target.chunks_exact_mut(3).zip(source.chunks_exact(4)) {
                let a = u32::from(px[3]).min(max);
                for (o, &c) in out.iter_mut().zip(&px[..3]) {
                    *o = ((u32::from(c) * a + u32::from(*o) * (max - a)) / max) as u16;
                }
            }
        });
    }

    // Primaries and transfer from the overlay (or, failing that, its inputs'
    // `colr`), else the first input's own; the samples are now RGB.
    let described = parsed.metadata.color_info.or(first_color.flatten());
    let color_info = ColorInfo {
        color_primaries: described.map_or(2, |c| c.color_primaries),
        transfer_characteristics: described.map_or(2, |c| c.transfer_characteristics),
        matrix_coefficients: 0,
        full_range: true,
    };
    let plane = |channel: usize| canvas.iter().skip(channel).step_by(3).map(|&v| i32::from(v)).collect::<Vec<_>>();
    let (red, green, blue) = (plane(0), plane(1), plane(2));
    drop(canvas);
    let image = DecodedImage {
        alpha: None,
        width,
        height,
        bit_depth_luma: depth,
        bit_depth_chroma: depth,
        chroma_format_idc: 3,
        color_info: Some(color_info),
        chroma_sampling: ChromaSampling::coded(3, None),
        y: green,
        cb: blue,
        cr: red,
        cb_width: width,
        cb_height: height,
        metadata: Default::default(),
    };
    apply_transforms(image, if transform { &parsed.transforms } else { &[] })
}
