//! Decode straight to RGB8 without assembling the planar picture.
//!
//! [`crate::decode`] builds the whole canvas as `i32` planes, then
//! [`crate::DecodedImage::to_rgb8`] converts it: ~180 MB at the peak for a
//! 12 MP iPhone photo, 3.5 GB for 200 MP. [`decode_rgb8`] decodes a grid one
//! row of tiles at a time and converts each source row as soon as the chroma
//! rows it reads are decoded, then lets them go. Its output is bit-identical
//! to `decode(..)?.to_rgb8_with_options(..)` (but for an overlay, whose inputs
//! are converted with the options before they are composed, where `decode`
//! composes them with the defaults): the container's crop, rotation
//! and mirroring are resolved into per-axis index maps with the same chroma
//! geometry [`crate::decode`] would produce, and every pixel goes through the
//! same arithmetic, in the same order, as the planar converter.
//!
//! With [`DecodeRgbOptions::max_edge`] the pixels are averaged into a
//! smaller image as they are produced, so only the output is ever held.
//!
//! Files this does not stream — a predicted image, a grid with uneven or
//! odd-sized tiles, an alpha plane to composite, an overlay — go through
//! [`crate::decode`] instead, under the same limits.

use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc;

use rayon::prelude::*;

use crate::color::{Axis, BandSample, ChromaSampling, ColorInfo, PixelRgb, RgbImage, RgbOptions, alpha8, axis};
use crate::error::{HeifError, Result};
use crate::hevc::ctu::Picture;
use crate::hevc::nal::NAL_SPS;
use crate::hevc::params::parse_sps;
use crate::isobmff::{self, ResolvedImage, Transform};
use crate::{ImageWindow, MAX_IMAGE_PIXELS, PicturePool, chroma_subsampling, conformance_window, crop_axis, decode_into, find_and_parse};

/// What to do with an alpha plane.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AlphaMode {
    /// Leave it out, as [`crate::DecodedImage::to_rgb8`] does. The alpha
    /// image is then not decoded at all.
    #[default]
    Ignore,
    /// Composite the colour over this background, using alpha scaled as
    /// [`crate::DecodedImage::to_rgba8`] scales it.
    Over([u8; 3]),
}

/// Options for [`decode_rgb8_with`].
#[derive(Debug, Clone, Copy)]
pub struct DecodeRgbOptions {
    /// Longest edge of the output. `None`, or an image that already fits,
    /// keeps the full size; otherwise the image is scaled down by averaging
    /// the full-size pixels that fall in each output pixel.
    pub max_edge: Option<usize>,
    pub alpha: AlphaMode,
    /// Colour override and chroma upsampling, as for `to_rgb8_with_options`.
    /// An overlay's inputs are converted with them before they are composed.
    pub rgb: RgbOptions,
    /// Largest coded picture (one grid tile, or a whole non-grid image with
    /// its alpha and prediction reference), in luma samples as the SPS
    /// codes it, before any cropping. Checked before anything is allocated.
    pub max_picture_pixels: u64,
    /// Largest grid: the assembled canvas, and the sum of its tiles.
    pub max_total_pixels: u64,
    /// Interpolate each chroma row once for the output rows that share it,
    /// instead of once per output row. Same pixels either way; it only
    /// applies to pictures that are not turned by a 90° rotation.
    pub reuse_chroma_rows: bool,
}

impl Default for DecodeRgbOptions {
    fn default() -> Self {
        DecodeRgbOptions {
            max_edge: None,
            alpha: AlphaMode::Ignore,
            rgb: RgbOptions::default(),
            max_picture_pixels: MAX_IMAGE_PIXELS,
            max_total_pixels: MAX_IMAGE_PIXELS,
            reuse_chroma_rows: REUSE_CHROMA_ROWS,
        }
    }
}

/// The primary image as RGB8, at full size: the same pixels as
/// `decode(bytes)?.to_rgb8()`, without holding the planar picture.
pub fn decode_rgb8(bytes: &[u8]) -> Result<RgbImage<u8>> {
    decode_rgb8_with(bytes, &DecodeRgbOptions::default())
}

/// [`decode_rgb8`] with options: a maximum size, alpha compositing, colour
/// overrides and limits.
pub fn decode_rgb8_with(bytes: &[u8], options: &DecodeRgbOptions) -> Result<RgbImage<u8>> {
    let parsed = isobmff::parse(bytes)?;
    let too_large = || HeifError::LimitExceeded("RGB: picture larger than allowed");
    let wants_alpha = matches!(options.alpha, AlphaMode::Over(_)) && parsed.alpha.is_some();

    let mut tile_pixels = Vec::with_capacity(parsed.images.len());
    for image in &parsed.images {
        let pixels = picture_pixels(image)?;
        if pixels > options.max_picture_pixels {
            return Err(too_large());
        }
        tile_pixels.push(pixels);
    }
    if let Some(grid) = parsed.grid {
        let canvas = u64::from(grid.output_width) * u64::from(grid.output_height);
        if canvas > options.max_total_pixels || tile_pixels.iter().sum::<u64>() > options.max_total_pixels {
            return Err(too_large());
        }
    }
    let alpha_pixels = match &parsed.alpha {
        Some(alpha) if wants_alpha => {
            let mut pixels = 0u64;
            for image in &alpha.image.images {
                pixels = pixels.saturating_add(picture_pixels(image)?);
            }
            if let Some(grid) = alpha.image.grid {
                pixels = pixels.saturating_add(u64::from(grid.output_width) * u64::from(grid.output_height));
            }
            pixels
        }
        _ => 0,
    };
    if alpha_pixels > options.max_picture_pixels {
        return Err(too_large());
    }

    if let Some(overlay) = &parsed.overlay {
        crate::overlay::check_limits(overlay, options.max_picture_pixels, options.max_total_pixels)?;
        return through_decode(bytes, options);
    }
    if parsed.reference_image.is_some() {
        let mut pixels = tile_pixels.iter().sum::<u64>();
        if let Some(image) = &parsed.reference_image {
            pixels = pixels.saturating_add(picture_pixels(image)?);
        }
        if pixels.saturating_add(alpha_pixels) > options.max_picture_pixels {
            return Err(too_large());
        }
        return through_decode(bytes, options);
    }
    match stream(&parsed, options)? {
        Some(rgb) => Ok(rgb),
        None => through_decode(bytes, options),
    }
}

fn picture_pixels(image: &ResolvedImage) -> Result<u64> {
    let sps = find_and_parse(image.hevc_config.nalus_of_type(NAL_SPS), NAL_SPS, parse_sps)?;
    Ok(u64::from(sps.pic_width_in_luma_samples) * u64::from(sps.pic_height_in_luma_samples))
}

/// `decode` and the planar converter, for the files [`stream`] leaves out.
fn through_decode(bytes: &[u8], options: &DecodeRgbOptions) -> Result<RgbImage<u8>> {
    let (image, rgb_options) = crate::decode_for_rgb(bytes, &options.rgb)?;
    let rgb = match options.alpha {
        AlphaMode::Over(background) if image.alpha.is_some() => {
            let rgba = image.to_rgba8_with_options(&rgb_options)?;
            let data = rgba.data.as_chunks::<4>().0.iter().flat_map(|&[r, g, b, a]| over([r, g, b], a, background)).collect();
            RgbImage { width: rgba.width, height: rgba.height, data }
        }
        _ => image.to_rgb8_with_options(&rgb_options)?,
    };
    drop(image);
    let full = (rgb.width, rgb.height);
    let out = fit_within(full, options.max_edge);
    if out == full {
        return Ok(rgb);
    }
    let maps = [AxisMap::identity(full.0, out.0, 1), AxisMap::identity(full.1, out.1, out.0)];
    let mut sink = Sink::bins(full, out, maps)?;
    for (y, row) in rgb.data.as_chunks::<3>().0.chunks(full.0).enumerate() {
        for (x, px) in row.iter().enumerate() {
            sink.put(x, y, *px);
        }
    }
    Ok(sink.finish())
}

/// `color` over `background` with 8-bit `alpha`.
fn over(color: [u8; 3], alpha: u8, background: [u8; 3]) -> [u8; 3] {
    let a = u32::from(alpha);
    let blend = |c: u8, b: u8| ((u32::from(c) * a + u32::from(b) * (255 - a) + 127) / 255) as u8;
    [blend(color[0], background[0]), blend(color[1], background[1]), blend(color[2], background[2])]
}

/// Width and height scaled down so the longest edge is at most `max_edge`.
fn fit_within((width, height): (usize, usize), max_edge: Option<usize>) -> (usize, usize) {
    let longest = width.max(height);
    match max_edge {
        Some(max_edge) if longest > max_edge => {
            let scale = |side: usize| ((side * max_edge + longest / 2) / longest).max(1);
            (scale(width), scale(height))
        }
        _ => (width, height),
    }
}

/// Where transformed coordinates come from along one source axis:
/// `source = offset ± t`, on the source's x (`from_y == false`) or y axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Affine {
    from_y: bool,
    flip: bool,
    offset: i64,
}

impl Affine {
    fn eval(self, t: usize) -> i64 {
        if self.flip { self.offset - t as i64 } else { self.offset + t as i64 }
    }
    /// The transformed coordinate source coordinate `s` lands on, if it is
    /// inside `0..len`.
    fn invert(self, s: usize, len: usize) -> Option<usize> {
        let t = if self.flip { self.offset - s as i64 } else { s as i64 - self.offset };
        (0..len as i64).contains(&t).then_some(t as usize)
    }
    /// `self(c - t)`: the axis flipped within `0..=c`.
    fn reflected(self, c: usize) -> Self {
        let c = c as i64;
        if self.flip { Affine { flip: false, offset: self.offset - c, ..self } } else { Affine { flip: true, offset: self.offset + c, ..self } }
    }
    /// `self(t + by)`: the axis shifted by a crop.
    fn shifted(self, by: usize) -> Self {
        let by = by as i64;
        Affine { offset: if self.flip { self.offset - by } else { self.offset + by }, ..self }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    /// One 90° anticlockwise turn.
    Turn,
    FlipX,
    FlipY,
}

/// A plane's transformed raster: its size, and for each transformed axis
/// (`[x, y]`) where its coordinates come from in the source plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlaneMap {
    width: usize,
    height: usize,
    axes: [Affine; 2],
}

impl PlaneMap {
    fn new(width: usize, height: usize, ops: &[Op], crop: Option<(usize, usize, usize, usize)>) -> Self {
        let mut axes = [Affine { from_y: false, flip: false, offset: 0 }, Affine { from_y: true, flip: false, offset: 0 }];
        let (mut w, mut h) = (width, height);
        for &op in ops {
            match op {
                // rotate90_ccw_plane: dst(x, y) = src(w - 1 - y, x).
                Op::Turn => {
                    axes = [axes[1], axes[0].reflected(w.saturating_sub(1))];
                    (w, h) = (h, w);
                }
                Op::FlipX => axes[0] = axes[0].reflected(w.saturating_sub(1)),
                Op::FlipY => axes[1] = axes[1].reflected(h.saturating_sub(1)),
            }
        }
        if let Some((left, top, cw, ch)) = crop {
            axes = [axes[0].shifted(left), axes[1].shifted(top)];
            (w, h) = (cw, ch);
        }
        PlaneMap { width: w, height: h, axes }
    }

    /// The transformed axis the source's y axis (`true`) or x axis feeds.
    fn axis_of(&self, from_y: bool) -> usize {
        usize::from(self.axes[1].from_y == from_y)
    }

    fn len(&self, axis: usize) -> usize {
        if axis == 0 { self.width } else { self.height }
    }
}

/// The planar picture `decode` would hand to the converter, as geometry:
/// what `apply_transforms` and `extract_window` do to its size and chroma
/// siting, resolved into index maps instead of moved samples.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    luma: PlaneMap,
    chroma: PlaneMap,
    sampling: ChromaSampling,
}

fn geometry(width: usize, height: usize, cb_width: usize, cb_height: usize, format: u32, coded: ChromaSampling, transforms: &[Transform]) -> Result<Geometry> {
    let (mut sub_w, mut sub_h) = chroma_subsampling(format);
    let (mut w, mut h, mut cw, mut ch) = (width, height, cb_width, cb_height);
    let mut window = (0usize, 0usize, width, height);
    let mut sampling = coded;
    let mut ops = Vec::new();
    // As apply_transforms, step by step.
    for &transform in transforms {
        match transform {
            Transform::CleanAperture(aperture) => {
                let (dx, ww) = crop_axis(window.2, aperture.width, aperture.horizontal_offset)?;
                let (dy, wh) = crop_axis(window.3, aperture.height, aperture.vertical_offset)?;
                window = (window.0 + dx, window.1 + dy, ww, wh);
                continue;
            }
            Transform::Rotate90Ccw(turns) => {
                for _ in 0..turns % 4 {
                    window = (window.1, w - window.0 - window.2, window.3, window.2);
                    let s = sampling;
                    sampling = ChromaSampling {
                        sub_x: s.sub_y,
                        sub_y: s.sub_x,
                        origin_x: s.origin_y,
                        origin_y: (w - 1) as f32 - s.origin_x - cw.saturating_sub(1) as f32 * s.sub_x as f32,
                    };
                    if format == 0 {
                        sampling = ChromaSampling::coded(0, None);
                    }
                    ops.push(Op::Turn);
                    (w, h, cw, ch) = (h, w, ch, cw);
                    std::mem::swap(&mut sub_w, &mut sub_h);
                }
                continue;
            }
            Transform::MirrorHorizontal => {
                window.0 = w - window.0 - window.2;
                sampling.origin_x = (w - 1) as f32 - sampling.origin_x - cw.saturating_sub(1) as f32 * sampling.sub_x as f32;
                ops.push(Op::FlipX);
            }
            Transform::MirrorVertical => {
                window.1 = h - window.1 - window.3;
                sampling.origin_y = (h - 1) as f32 - sampling.origin_y - ch.saturating_sub(1) as f32 * sampling.sub_y as f32;
                ops.push(Op::FlipY);
            }
        }
        if format == 0 {
            sampling = ChromaSampling::coded(0, None);
        }
    }
    let (luma_crop, chroma_crop) = if window == (0, 0, w, h) {
        (None, None)
    } else {
        let (left, top, ww, wh) = window;
        let chroma = if format == 0 { (0, 0, 0, 0) } else { (left / sub_w, top / sub_h, ww.div_ceil(sub_w), wh.div_ceil(sub_h)) };
        sampling.origin_x -= (left % sub_w) as f32;
        sampling.origin_y -= (top % sub_h) as f32;
        (Some(window), Some(chroma))
    };
    Ok(Geometry { luma: PlaneMap::new(width, height, &ops, luma_crop), chroma: PlaneMap::new(cb_width, cb_height, &ops, chroma_crop), sampling })
}

/// The checks the planar converter makes on the picture it is given; a
/// picture failing them goes through `decode`, which then fails the same way.
fn converter_accepts(g: &Geometry, format: u32) -> bool {
    let s = g.sampling;
    let monochrome = format == 0;
    let valid_subsampling = match format {
        0 => (s.sub_x, s.sub_y) == (1, 1),
        1 => (s.sub_x, s.sub_y) == (2, 2),
        2 => matches!((s.sub_x, s.sub_y), (2, 1) | (1, 2)),
        3 => (s.sub_x, s.sub_y) == (1, 1),
        _ => false,
    };
    let shape = if monochrome {
        (g.chroma.width, g.chroma.height) == (0, 0)
    } else {
        g.chroma.width == g.luma.width.div_ceil(s.sub_x as usize)
            && g.chroma.height == g.luma.height.div_ceil(s.sub_y as usize)
            && g.chroma.width > 0
            && g.chroma.height > 0
    };
    g.luma.width > 0
        && g.luma.height > 0
        && (g.luma.width as u64) * (g.luma.height as u64) <= MAX_IMAGE_PIXELS
        && valid_subsampling
        && shape
        && s.origin_x.is_finite()
        && s.origin_y.is_finite()
        && s.origin_x.abs() <= 2.0
        && s.origin_y.abs() <= 2.0
}

/// One source axis, resolved: for each source coordinate, where it lands
/// on the transformed axis it feeds and which chroma samples it reads there.
struct SourceAxis {
    /// The transformed axis this source axis feeds (0 = x).
    axis: usize,
    entries: Vec<Option<Entry>>,
}

#[derive(Clone, Copy)]
struct Entry {
    /// Transformed coordinate.
    t: usize,
    /// Source chroma indices (along this same source axis) of the two
    /// samples the converter interpolates, and the weight of the second.
    lo: usize,
    hi: usize,
    weight: f32,
}

fn source_axis(len: usize, from_y: bool, g: &Geometry, tables: &[Vec<Axis>; 2]) -> SourceAxis {
    let axis = g.luma.axis_of(from_y);
    let luma = g.luma.axes[axis];
    let chroma = g.chroma.axes[axis];
    let tlen = g.luma.len(axis);
    let entries = (0..len)
        .map(|s| {
            let t = luma.invert(s, tlen)?;
            let Some(a) = tables[axis].get(t) else { return Some(Entry { t, lo: 0, hi: 0, weight: 0.0 }) };
            Some(Entry { t, lo: chroma.eval(a.lo) as usize, hi: chroma.eval(a.hi) as usize, weight: a.weight })
        })
        .collect();
    SourceAxis { axis, entries }
}

/// Where converted pixels go: straight into the full-size image, or
/// averaged into a smaller one.
enum Sink {
    Full { width: usize, data: Vec<u8> },
    Bins(Bins),
}

/// Where converted source pixels land in the transformed picture, for
/// writing a batch of rows in parallel: each output row (or row of bins)
/// is owned by one task, and gathers its pixels from one source row, or
/// one source column when the picture is turned.
struct Scatter {
    turned: bool,
    /// Transformed coordinate of each source row and source column.
    row_t: Vec<Option<usize>>,
    col_t: Vec<Option<usize>>,
    /// For each transformed row: the source row (unturned) or source
    /// column (turned) it comes from.
    source_of_row: Vec<Option<usize>>,
    /// Unturned, the source columns kept land on one run of transformed
    /// columns: `(first source column, first transformed column, length,
    /// reversed)`. A row then copies as one slice.
    span: Option<(usize, usize, usize, bool)>,
}

impl Scatter {
    fn new(rows: &SourceAxis, columns: &SourceAxis, transformed_height: usize) -> Self {
        let turned = columns.axis == 1;
        let row_t: Vec<Option<usize>> = rows.entries.iter().map(|e| e.map(|e| e.t)).collect();
        let col_t: Vec<Option<usize>> = columns.entries.iter().map(|e| e.map(|e| e.t)).collect();
        let mut source_of_row = vec![None; transformed_height];
        for (s, t) in (if turned { &col_t } else { &row_t }).iter().enumerate() {
            if let Some(t) = t {
                source_of_row[*t] = Some(s);
            }
        }
        let span = Self::span(&col_t);
        Scatter { turned, row_t, col_t, source_of_row, span }
    }

    fn span(col_t: &[Option<usize>]) -> Option<(usize, usize, usize, bool)> {
        let start = col_t.iter().position(Option::is_some)?;
        let len = col_t[start..].iter().take_while(|t| t.is_some()).count();
        if col_t[start + len..].iter().any(Option::is_some) {
            return None;
        }
        let ts: Vec<usize> = col_t[start..start + len].iter().flatten().copied().collect();
        let reversed = len > 1 && ts[1] < ts[0];
        let ok = ts.windows(2).all(|w| if reversed { w[1] + 1 == w[0] } else { w[0] + 1 == w[1] });
        ok.then(|| (start, if reversed { ts[len - 1] } else { ts[0] }, len, reversed))
    }

    /// The transformed rows a batch of source rows writes to: all of them
    /// when turned (each holds a source column), else the batch's own.
    fn rows_touched(&self, first: usize, count: usize, height: usize) -> std::ops::Range<usize> {
        if self.turned {
            return 0..height;
        }
        let ts = self.row_t[first..first + count].iter().flatten();
        match (ts.clone().min(), ts.max()) {
            (Some(&lo), Some(&hi)) => lo..hi + 1,
            _ => 0..0,
        }
    }

    /// The pixels of transformed row `ty` that the batch of source rows
    /// `first..first + converted.len()` holds, as `(tx, rgb)`.
    fn row_pixels<'a>(&'a self, ty: usize, first: usize, converted: &'a [Vec<[u8; 3]>]) -> impl Iterator<Item = (usize, [u8; 3])> + 'a {
        let source = self.source_of_row[ty];
        let unturned = (!self.turned)
            .then(|| source.and_then(|y| y.checked_sub(first)).and_then(|i| converted.get(i)))
            .flatten()
            .into_iter()
            .flat_map(move |row| self.col_t.iter().zip(row).filter_map(|(t, rgb)| Some(((*t)?, *rgb))));
        let turned = self
            .turned
            .then_some(source)
            .flatten()
            .into_iter()
            .flat_map(move |x| converted.iter().enumerate().filter_map(move |(i, row)| Some((self.row_t[first + i]?, *row.get(x)?))));
        unturned.chain(turned)
    }
}

/// An alpha plane to composite over a background, at transformed size.
struct AlphaOver {
    samples: Vec<i32>,
    width: usize,
    max: u32,
    background: [u8; 3],
}

impl AlphaOver {
    #[inline(always)]
    fn apply(&self, tx: usize, ty: usize, rgb: [u8; 3]) -> [u8; 3] {
        over(rgb, alpha8(self.samples[ty * self.width + tx] as u32, self.max), self.background)
    }
}

impl Sink {
    fn full(width: usize, height: usize) -> Result<Self> {
        let len = width.checked_mul(height).and_then(|p| p.checked_mul(3)).ok_or(HeifError::LimitExceeded("RGB: output size"))?;
        // Probe that the allocation can succeed, then take it zeroed from the
        // allocator (fresh pages for a picture, no pass writing zeros).
        Vec::<u8>::new().try_reserve_exact(len).map_err(|_| HeifError::LimitExceeded("RGB: output allocation failed"))?;
        let data = if *ZERO_FILL {
            let mut data = Vec::with_capacity(len);
            data.resize(len, 0);
            data
        } else {
            vec![0; len]
        };
        Ok(Sink::Full { width, data })
    }

    fn bins(full: (usize, usize), out: (usize, usize), maps: [AxisMap; 2]) -> Result<Self> {
        Ok(Sink::Bins(Bins::new(full, out, maps)?))
    }

    #[inline(always)]
    fn put(&mut self, x: usize, y: usize, rgb: [u8; 3]) {
        match self {
            Sink::Full { width, data } => {
                let i = (y * *width + x) * 3;
                data[i..i + 3].copy_from_slice(&rgb);
            }
            Sink::Bins(bins) => bins.add(x, y, rgb),
        }
    }

    /// A batch of converted source rows, written in parallel over output
    /// rows (or rows of bins), each owned by one task.
    fn put_batch(&mut self, scatter: &Scatter, first: usize, converted: &[Vec<[u8; 3]>], alpha: Option<&AlphaOver>) {
        let pixel = |tx: usize, ty: usize, rgb: [u8; 3]| match alpha {
            Some(a) => a.apply(tx, ty, rgb),
            None => rgb,
        };
        match self {
            Sink::Full { width, data } => {
                let row = *width * 3;
                let touched = scatter.rows_touched(first, converted.len(), data.len() / row.max(1));
                // Turned, a batch whose rows land on one run of transformed
                // columns: each transformed row gathers one source column.
                let run = Scatter::span(&scatter.row_t[first..first + converted.len()]).filter(|&(start, _, len, _)| start == 0 && len == converted.len());
                // Turned, blocks of eight transformed rows reading eight consecutive
                // source columns: transposed eight source rows at a time.
                if let (true, None, Some((_, t0, len, reversed)), false) = (scatter.turned, alpha, run, *NO_TURN_BLOCKS) {
                    let start = touched.start;
                    data[touched.start * row..touched.end * row].par_chunks_mut(row * 8).enumerate().for_each(|(b, block)| {
                        let ty0 = start + 8 * b;
                        let x0 = scatter.source_of_row.get(ty0).copied().flatten();
                        let consecutive = block.len() == row * 8
                            && x0.is_some_and(|x0| (0..8).all(|j| scatter.source_of_row.get(ty0 + j).copied().flatten() == Some(x0 + j)));
                        let mut rows: Vec<&mut [[u8; 3]]> = block.chunks_mut(row).map(|r| r.as_chunks_mut::<3>().0).collect();
                        let mut done = 0;
                        if let (true, Some(x0)) = (consecutive, x0)
                            && let Ok(mut eight) = <[&mut [[u8; 3]]; 8]>::try_from(rows.iter_mut().map(|r| &mut r[t0..t0 + len]).collect::<Vec<_>>())
                        {
                            done = crate::color::turn_rgb_8rows(&mut eight, converted, x0, reversed);
                        }
                        // the rest (rows not in a whole block, source rows past the last block of eight)
                        for (j, out) in rows.iter_mut().enumerate() {
                            let Some(x) = scatter.source_of_row.get(ty0 + j).copied().flatten() else { continue };
                            let Some(dst) = out.get_mut(t0..t0 + len) else { continue };
                            let skip = if consecutive { done } else { 0 };
                            for (k, src) in converted.iter().enumerate().skip(skip) {
                                let t = if reversed { len - 1 - k } else { k };
                                if let (Some(d), Some(s)) = (dst.get_mut(t), src.get(x)) {
                                    *d = *s;
                                }
                            }
                        }
                    });
                    return;
                }
                data[touched.start * row..touched.end * row].par_chunks_mut(row).zip(touched).for_each(|(out, ty)| {
                    if let (true, None, Some((_, t0, len, reversed))) = (scatter.turned, alpha, run) {
                        let Some(x) = scatter.source_of_row[ty] else { return };
                        let Some(dst) = out.as_chunks_mut::<3>().0.get_mut(t0..t0 + len) else { return };
                        let gather = |(d, src): (&mut [u8; 3], &Vec<[u8; 3]>)| {
                            if let Some(s) = src.get(x) {
                                *d = *s;
                            }
                        };
                        if reversed {
                            dst.iter_mut().rev().zip(converted).for_each(gather);
                        } else {
                            dst.iter_mut().zip(converted).for_each(gather);
                        }
                        return;
                    }
                    // An unturned row whose columns land on one run: a copy.
                    if let (false, None, Some((x0, t0, len, reversed))) = (scatter.turned, alpha, scatter.span) {
                        let Some(src) = scatter.source_of_row[ty].and_then(|y| y.checked_sub(first)).and_then(|i| converted.get(i)) else { return };
                        let (Some(src), Some(dst)) = (src.get(x0..x0 + len), out.as_chunks_mut::<3>().0.get_mut(t0..t0 + len)) else { return };
                        if reversed {
                            dst.iter_mut().zip(src.iter().rev()).for_each(|(d, s)| *d = *s);
                        } else {
                            dst.copy_from_slice(src);
                        }
                        return;
                    }
                    for (tx, rgb) in scatter.row_pixels(ty, first, converted) {
                        out[tx * 3..tx * 3 + 3].copy_from_slice(&pixel(tx, ty, rgb));
                    }
                });
            }
            Sink::Bins(bins) => {
                let [x_map, y_map] = &bins.maps;
                let stride = bins.out.0 * 3;
                // Transformed rows of each row of bins: consecutive, in order.
                let mut starts = Vec::with_capacity(y_map.counts.len() + 1);
                let mut t = 0;
                for &n in &y_map.counts {
                    starts.push(t);
                    t += n as usize;
                }
                starts.push(t);
                let fill = |br: usize, add: &mut dyn FnMut(usize, [u32; 3])| {
                    for ty in starts[br]..starts[br + 1] {
                        let mut run: Option<(usize, [u32; 3])> = None;
                        for (tx, rgb) in scatter.row_pixels(ty, first, converted) {
                            let rgb = pixel(tx, ty, rgb);
                            let bin = x_map.parts[tx];
                            match &mut run {
                                Some((b, sum)) if *b == bin => sum.iter_mut().zip(rgb).for_each(|(s, c)| *s += u32::from(c)),
                                _ => {
                                    if let Some((b, sum)) = run {
                                        add(b, sum);
                                    }
                                    run = Some((bin, rgb.map(u32::from)));
                                }
                            }
                        }
                        if let Some((b, sum)) = run {
                            add(b, sum);
                        }
                    }
                };
                // Rows of bins the batch reaches.
                let touched = scatter.rows_touched(first, converted.len(), t);
                let bin_row = |ty: usize| y_map.parts[ty] / bins.out.0.max(1);
                let rows = if touched.is_empty() { 0..0 } else { bin_row(touched.start)..bin_row(touched.end - 1) + 1 };
                match &mut bins.sums {
                    Sums::Narrow(sums) => sums[rows.start * stride..rows.end * stride].par_chunks_mut(stride).zip(rows).for_each(|(row, br)| {
                        // A run is part of one bin, so it fits whatever the bin fits.
                        fill(br, &mut |b, sum| row[b * 3..b * 3 + 3].iter_mut().zip(sum).for_each(|(s, c)| *s += c as u16));
                    }),
                    Sums::Wide(sums) => sums[rows.start * stride..rows.end * stride].par_chunks_mut(stride).zip(rows).for_each(|(row, br)| {
                        fill(br, &mut |b, sum| row[b * 3..b * 3 + 3].iter_mut().zip(sum).for_each(|(s, c)| *s += c));
                    }),
                }
            }
        }
    }

    fn finish(self) -> RgbImage<u8> {
        match self {
            Sink::Full { width, data } => RgbImage { width, height: data.len() / 3 / width.max(1), data },
            Sink::Bins(bins) => bins.finish(),
        }
    }
}

/// Output bin of each transformed coordinate along one axis, already
/// scaled for its axis (`bin` for x, `bin * out_width` for y).
struct AxisMap {
    parts: Vec<usize>,
    counts: Vec<u32>,
}

impl AxisMap {
    fn identity(full: usize, out: usize, stride: usize) -> Self {
        let mut counts = vec![0u32; out];
        let parts = (0..full)
            .map(|t| {
                let bin = t * out / full;
                counts[bin] += 1;
                bin * stride
            })
            .collect();
        AxisMap { parts, counts }
    }
}

/// Sums per output channel: `u16` while a bin's total cannot overflow it,
/// else `u32`.
enum Sums {
    Narrow(Vec<u16>),
    Wide(Vec<u32>),
}

/// Box-filter accumulator over the transformed image.
struct Bins {
    out: (usize, usize),
    maps: [AxisMap; 2],
    sums: Sums,
}

impl Bins {
    fn new(_full: (usize, usize), out: (usize, usize), maps: [AxisMap; 2]) -> Result<Self> {
        let channels = out.0.checked_mul(out.1).and_then(|p| p.checked_mul(3)).ok_or(HeifError::LimitExceeded("RGB: output size"))?;
        let widest = |map: &AxisMap| map.counts.iter().copied().max().unwrap_or(0);
        let area = u64::from(widest(&maps[0])) * u64::from(widest(&maps[1]));
        let failed = |_| HeifError::LimitExceeded("RGB: output allocation failed");
        let sums = if area * 255 <= u64::from(u16::MAX) {
            let mut sums = Vec::new();
            sums.try_reserve_exact(channels).map_err(failed)?;
            sums.resize(channels, 0);
            Sums::Narrow(sums)
        } else {
            let mut sums = Vec::new();
            sums.try_reserve_exact(channels).map_err(failed)?;
            sums.resize(channels, 0);
            Sums::Wide(sums)
        };
        Ok(Bins { out, maps, sums })
    }

    #[inline(always)]
    fn add(&mut self, x: usize, y: usize, rgb: [u8; 3]) {
        let i = (self.maps[0].parts[x] + self.maps[1].parts[y]) * 3;
        match &mut self.sums {
            Sums::Narrow(s) => s[i..i + 3].iter_mut().zip(rgb).for_each(|(s, c)| *s += u16::from(c)),
            Sums::Wide(s) => s[i..i + 3].iter_mut().zip(rgb).for_each(|(s, c)| *s += u32::from(c)),
        }
    }

    fn finish(self) -> RgbImage<u8> {
        let (width, height) = self.out;
        let mut data = Vec::with_capacity(width * height * 3);
        for (y, &row) in self.maps[1].counts.iter().enumerate() {
            for (x, &column) in self.maps[0].counts.iter().enumerate() {
                let n = row * column;
                let i = (y * width + x) * 3;
                for c in 0..3 {
                    let sum = match &self.sums {
                        Sums::Narrow(s) => u32::from(s[i + c]),
                        Sums::Wide(s) => s[i + c],
                    };
                    // Every bin covers at least one pixel; white if one did not.
                    data.push((sum + n / 2).checked_div(n).map_or(255, |m| m as u8));
                }
            }
        }
        RgbImage { width, height, data }
    }
}

/// What every tile must share with the first, and what the converter
/// takes from it.
#[derive(Clone, Copy)]
struct TileShape {
    width: usize,
    height: usize,
    format: u32,
    bit_depth_luma: u32,
    bit_depth_chroma: u32,
}

/// A decoded tile: its picture as decoded, and where in it the image
/// `decode` would crop it to lies.
struct Tile {
    pic: Picture,
    shape: TileShape,
    /// Luma origin of the SPS conformance window.
    left: usize,
    top: usize,
    sampling: ChromaSampling,
    color_info: Option<ColorInfo>,
}

/// One row of tiles laid out as source rows, as `blit_tiles` places them on
/// the canvas: zeros where no tile reaches. Samples are `u8` for 8-bit
/// pictures and `u16` for deeper ones (bit depths are at most 16).
struct Band<S> {
    luma: Vec<Vec<S>>,
    cb: Vec<Vec<S>>,
    cr: Vec<Vec<S>>,
}

/// A band sample type that rows of a decoded plane copy into.
trait PlaneRow: BandSample {
    fn copy(plane: &crate::hevc::ctu::Plane, x: usize, y: usize, dst: &mut [Self]);
}

impl PlaneRow for u8 {
    fn copy(plane: &crate::hevc::ctu::Plane, x: usize, y: usize, dst: &mut [u8]) {
        plane.copy_row_narrow(x, y, dst);
    }
}

impl PlaneRow for u16 {
    fn copy(plane: &crate::hevc::ctu::Plane, x: usize, y: usize, dst: &mut [u16]) {
        plane.copy_row(x, y, dst);
    }
}

/// The canvas layout bands are cut from.
#[derive(Clone, Copy)]
struct Layout {
    tile: TileShape,
    width: usize,
    height: usize,
    cb_width: usize,
    sub_w: usize,
    sub_h: usize,
}

impl Layout {
    /// Band `index`'s rows, reusing `spare` rows where there are (every
    /// sample of a band row is placed from a tile, so reused rows need no clearing).
    fn band<S: BandSample>(&self, index: usize, spare: [&mut Vec<Vec<S>>; 3]) -> Band<S> {
        let rows = self.tile.height.min(self.height.saturating_sub(index * self.tile.height));
        let c_rows = if self.tile.format == 0 { 0 } else { rows.div_ceil(self.sub_h) };
        let [sl, sb, sr] = spare;
        let take = |spare: &mut Vec<Vec<S>>, n: usize, width: usize| -> Vec<Vec<S>> {
            (0..n)
                .map(|_| match spare.pop() {
                    Some(mut r) => {
                        r.resize(width, S::default());
                        r
                    }
                    None => vec![S::default(); width],
                })
                .collect()
        };
        Band { luma: take(sl, rows, self.width), cb: take(sb, c_rows, self.cb_width), cr: take(sr, c_rows, self.cb_width) }
    }

    /// Copies the tile in column `column` into `band`, straight from the
    /// decoded picture's planes.
    fn place<S: PlaneRow>(&self, band: &mut Band<S>, column: usize, tile: &Tile) {
        let x0 = column * self.tile.width;
        let w = self.tile.width.min(self.width.saturating_sub(x0));
        if w == 0 {
            return;
        }
        for (r, row) in band.luma.iter_mut().enumerate() {
            S::copy(&tile.pic.y, tile.left, tile.top + r, &mut row[x0..x0 + w]);
        }
        let (cx, cw) = (x0 / self.sub_w, w.div_ceil(self.sub_w));
        let (left, top) = (tile.left / self.sub_w, tile.top / self.sub_h);
        for (r, (row_cb, row_cr)) in band.cb.iter_mut().zip(band.cr.iter_mut()).enumerate() {
            S::copy(&tile.pic.cb, left, top + r, &mut row_cb[cx..cx + cw]);
            S::copy(&tile.pic.cr, left, top + r, &mut row_cr[cx..cx + cw]);
        }
    }
}

/// Source rows held for conversion, `first..first + rows.len()`.
struct Rows<S> {
    first: usize,
    rows: VecDeque<Vec<S>>,
    /// Rows dropped below the converted ones, reused for the next bands.
    spare: Vec<Vec<S>>,
}

impl<S> Rows<S> {
    fn new() -> Self {
        Rows { first: 0, rows: VecDeque::new(), spare: Vec::new() }
    }
    fn end(&self) -> usize {
        self.first + self.rows.len()
    }
    fn get(&self, row: usize) -> &[S] {
        row.checked_sub(self.first).and_then(|i| self.rows.get(i)).map_or(&[], Vec::as_slice)
    }
    fn drop_below(&mut self, row: usize) {
        while self.first < row && !self.rows.is_empty() {
            if let Some(r) = self.rows.pop_front() {
                self.spare.push(r);
            }
            self.first += 1;
        }
    }
}

/// Decodes a tile into a picture from `pool`, and resolves the crop `decode`
/// would make (failing as it would).
fn decode_tile(image: &ResolvedImage, pool: &PicturePool) -> Result<Tile> {
    let (sps, pic) = decode_into(image, None, pool.take())?;
    let format = sps.chroma_format_idc;
    let mut sampling = ChromaSampling::coded(format, sps.chroma_location);
    let window = conformance_window(&sps)?;
    let ImageWindow { left, top, width, height } =
        window.unwrap_or(ImageWindow { left: 0, top: 0, width: sps.pic_width_in_luma_samples as usize, height: sps.pic_height_in_luma_samples as usize });
    if window.is_some() {
        let (sub_w, sub_h) = chroma_subsampling(format);
        sampling.origin_x -= (left % sub_w) as f32;
        sampling.origin_y -= (top % sub_h) as f32;
    }
    let shape = TileShape { width, height, format, bit_depth_luma: sps.bit_depth_luma, bit_depth_chroma: sps.bit_depth_chroma };
    Ok(Tile { pic, shape, left, top, sampling, color_info: sps.color_info })
}

/// Rows converted per parallel batch.
const ROWS_PER_BATCH: usize = 64;

/// Consecutive rows one task converts: neighbouring rows read the same
/// chroma rows, which the task then interpolates only once.
const ROWS_PER_TASK: usize = 8;

/// [`DecodeRgbOptions::reuse_chroma_rows`] by default.
const REUSE_CHROMA_ROWS: bool = true;

/// Stream-decodes `parsed` to RGB, or `None` when the file needs `decode`'s
/// general path.
fn stream(parsed: &isobmff::ParsedHeif, options: &DecodeRgbOptions) -> Result<Option<RgbImage<u8>>> {
    let (columns, rows) = match parsed.grid {
        Some(grid) => {
            if parsed.images.len() != (grid.rows as usize) * (grid.columns as usize) {
                return Err(HeifError::MalformedBox("grid image count does not match rows*columns"));
            }
            (grid.columns as usize, grid.rows as usize)
        }
        None if parsed.images.len() == 1 => (1, 1),
        None => return Ok(None),
    };
    let images = &parsed.images;
    // Each thread decodes its tiles into the pictures of the tiles placed
    // before them.
    let pool = PicturePool::default();
    // Called from inside rayon's pool, blocking on a channel could starve
    // a one-thread pool; with one thread, a decoder thread beside the
    // caller only adds switching. Decode a few tiles at a time there.
    if rayon::current_thread_index().is_some() || rayon::current_num_threads() == 1 || images.len() == 1 {
        let threads = rayon::current_num_threads();
        return convert(parsed, options, columns, rows, &pool, &mut Chunked { images, pool: &pool, threads, pending: BTreeMap::new() });
    }
    rayon::in_place_scope(|scope| {
        let (sender, receiver) = mpsc::channel();
        let window = TILES_AHEAD_PER_THREAD * rayon::current_num_threads();
        let mut feed = Ahead { scope, images, pool: &pool, window, sender, receiver, spawned: 0, pending: BTreeMap::new() };
        feed.spawn_until(window);
        convert(parsed, options, columns, rows, &pool, &mut feed)
    })
}

/// Tiles decoded ahead of the one being placed, per thread: enough that
/// no thread waits at a row boundary, few enough that what is held stays
/// a couple of tiles per thread.
const TILES_AHEAD_PER_THREAD: usize = 2;

/// Where [`convert`] gets its decoded tiles, in raster order.
trait Tiles {
    fn tile(&mut self, index: usize) -> Result<Tile>;
}

/// Tiles decoded on rayon's pool, a window of them beyond the one being
/// placed, while the calling thread places and converts rows: the threads
/// never wait for a row to finish, and conversion overlaps decoding.
struct Ahead<'s, 'p> {
    scope: &'s rayon::Scope<'p>,
    images: &'p [ResolvedImage<'p>],
    pool: &'p PicturePool,
    window: usize,
    sender: mpsc::Sender<(usize, Result<Tile>)>,
    receiver: mpsc::Receiver<(usize, Result<Tile>)>,
    spawned: usize,
    /// Tiles that arrived before their turn.
    pending: BTreeMap<usize, Result<Tile>>,
}

impl<'p> Ahead<'_, 'p> {
    /// Starts decoding every tile below index `end`.
    fn spawn_until(&mut self, end: usize) {
        let end = end.min(self.images.len());
        while self.spawned < end {
            let (index, image, pool, sender) = (self.spawned, &self.images[self.spawned], self.pool, self.sender.clone());
            self.scope.spawn(move |_| {
                // The receiver is gone only if the caller already failed.
                let _ = sender.send((index, decode_tile(image, pool)));
            });
            self.spawned += 1;
        }
    }
}

impl Tiles for Ahead<'_, '_> {
    fn tile(&mut self, index: usize) -> Result<Tile> {
        self.spawn_until(index + 1 + self.window);
        loop {
            if let Some(tile) = self.pending.remove(&index) {
                return tile;
            }
            let (arrived, tile) = self.receiver.recv().map_err(|_| HeifError::MalformedHevc("RGB: tile decoder stopped"))?;
            self.pending.insert(arrived, tile);
        }
    }
}

/// Tiles decoded a few at a time (one per thread) by the caller: with one
/// thread, or when the caller is itself on rayon's pool.
struct Chunked<'p> {
    images: &'p [ResolvedImage<'p>],
    pool: &'p PicturePool,
    threads: usize,
    pending: BTreeMap<usize, Tile>,
}

impl Tiles for Chunked<'_> {
    fn tile(&mut self, index: usize) -> Result<Tile> {
        if let Some(tile) = self.pending.remove(&index) {
            return Ok(tile);
        }
        let end = (index + self.threads.max(1)).min(self.images.len());
        let pool = self.pool;
        let decoded: Vec<Tile> = self.images[index..end].par_iter().map(|image| decode_tile(image, pool)).collect::<Result<_>>()?;
        let mut decoded = decoded.into_iter();
        let first = decoded.next().ok_or(HeifError::MalformedBox("grid has no tiles"))?;
        self.pending.extend((index + 1..end).zip(decoded));
        Ok(first)
    }
}

/// [`stream`]'s conversion, fed tiles in raster order by `tiles`; their
/// pictures go back to `pool` once placed.
fn convert(
    parsed: &isobmff::ParsedHeif,
    options: &DecodeRgbOptions,
    columns: usize,
    rows: usize,
    pool: &PicturePool,
    tiles: &mut dyn Tiles,
) -> Result<Option<RgbImage<u8>>> {
    // The first tile tells the layout; it is placed like the others.
    let first = tiles.tile(0)?;
    let shape = first.shape;
    let (width, height) = match parsed.grid {
        Some(grid) => (grid.output_width as usize, grid.output_height as usize),
        None => (shape.width, shape.height),
    };
    let (sub_w, sub_h) = chroma_subsampling(shape.format);
    // Tiles on chroma boundaries stack into whole chroma rows, as the grid
    // assembly lays them out.
    if !shape.width.is_multiple_of(sub_w) || !shape.height.is_multiple_of(sub_h) {
        return Ok(None);
    }
    let (cb_width, cb_height) = if shape.format == 0 { (0, 0) } else { (width.div_ceil(sub_w), height.div_ceil(sub_h)) };
    let g = geometry(width, height, cb_width, cb_height, shape.format, first.sampling, &parsed.transforms)?;
    if !converter_accepts(&g, shape.format) {
        return Ok(None);
    }
    let info = options.rgb.color.or(parsed.metadata.color_info).or(first.color_info).unwrap_or_default();
    let pixel = PixelRgb::new(info, shape.format, g.sampling, shape.bit_depth_luma, shape.bit_depth_chroma)?;
    let layout = Layout { tile: shape, width, height, cb_width, sub_w, sub_h };

    // The converter's chroma tables over the transformed picture.
    let tables = if pixel.monochrome() {
        [Vec::new(), Vec::new()]
    } else {
        [
            axis(g.luma.width, g.chroma.width, g.sampling.sub_x, g.sampling.origin_x, options.rgb.upsampling)?,
            axis(g.luma.height, g.chroma.height, g.sampling.sub_y, g.sampling.origin_y, options.rgb.upsampling)?,
        ]
    };
    let columns_axis = source_axis(width, false, &g, &tables);
    let rows_axis = source_axis(height, true, &g, &tables);
    let turned = columns_axis.axis == 1;
    let column_arrays = Columns::new(&columns_axis);
    // Nearest-neighbour chroma (or 4:4:4) gives every interpolation a
    // weight of 0, and `lo * (1 - 0) + hi * 0` is exactly `lo`.
    let flat = column_arrays.weight.iter().chain(rows_axis.entries.iter().flatten().map(|e| &e.weight)).all(|&w| w == 0.0);

    // Chroma rows each source row reads, and the lowest any later row reads.
    let needs: Vec<Option<(usize, usize)>> =
        if pixel.monochrome() { vec![None; height] } else { rows_axis.entries.iter().map(|e| e.map(|e| (e.lo.min(e.hi), e.lo.max(e.hi)))).collect() };
    let mut keep_from = vec![usize::MAX; height + 1];
    for y in (0..height).rev() {
        keep_from[y] = keep_from[y + 1].min(needs[y].map_or(usize::MAX, |n| n.0));
    }

    let alpha = match (options.alpha, &parsed.alpha) {
        (AlphaMode::Over(background), Some(auxiliary)) => {
            let plane = crate::decode_pixels(&auxiliary.image, true)?;
            if (plane.width, plane.height) != (g.luma.width, g.luma.height) {
                return Err(HeifError::Unsupported("alpha: display dimensions do not match colour image"));
            }
            let max = (1u32 << plane.bit_depth_luma) - 1;
            Some(AlphaOver { samples: plane.y, width: plane.width, max, background })
        }
        _ => None,
    };

    let full = (g.luma.width, g.luma.height);
    let scatter = Scatter::new(&rows_axis, &columns_axis, full.1);
    let out = fit_within(full, options.max_edge);
    let mut sink = if out == full {
        Sink::full(full.0, full.1)?
    } else {
        Sink::bins(full, out, [AxisMap::identity(full.0, out.0, 1), AxisMap::identity(full.1, out.1, out.0)])?
    };

    let plan = Plan { layout, shape, rows, columns, needs: &needs, keep_from: &keep_from, height, flat, turned, options, alpha: alpha.as_ref(), scatter: &scatter, column_arrays: &column_arrays, rows_axis: &rows_axis, pixel: &pixel };
    let converted = if shape.bit_depth_luma <= 8 && shape.bit_depth_chroma <= 8 && !*WIDE_BANDS {
        convert_bands::<u8>(&plan, first, tiles, pool, &mut sink)?
    } else {
        convert_bands::<u16>(&plan, first, tiles, pool, &mut sink)?
    };
    if converted.is_none() {
        return Ok(None);
    }
    Ok(Some(sink.finish()))
}

/// What [`convert_bands`] needs from [`convert`]'s setup.
struct Plan<'a> {
    layout: Layout,
    shape: TileShape,
    rows: usize,
    columns: usize,
    needs: &'a [Option<(usize, usize)>],
    keep_from: &'a [usize],
    height: usize,
    flat: bool,
    turned: bool,
    options: &'a DecodeRgbOptions,
    alpha: Option<&'a AlphaOver>,
    scatter: &'a Scatter,
    column_arrays: &'a Columns,
    rows_axis: &'a SourceAxis,
    pixel: &'a PixelRgb,
}

/// [`convert`]'s bands: tiles placed in rows of samples `S`, rows converted
/// into `sink` as soon as their chroma is there. `None` when a tile does not
/// fit the layout (the file needs `decode`'s general path).
fn convert_bands<S: PlaneRow>(plan: &Plan, first: Tile, tiles: &mut dyn Tiles, pool: &PicturePool, sink: &mut Sink) -> Result<Option<()>> {
    let Plan { layout, shape, rows, columns, needs, keep_from, height, flat, turned, options, alpha, scatter, column_arrays, rows_axis, pixel } = *plan;
    let (mut luma, mut cb, mut cr) = (Rows::<S>::new(), Rows::<S>::new(), Rows::<S>::new());
    let mut first = Some(first);
    let mut next_y = 0;
    for band_index in 0..rows {
        // Each tile is placed in the band as soon as it is its turn.
        let mut band = layout.band(band_index, [&mut luma.spare, &mut cb.spare, &mut cr.spare]);
        for column in 0..columns {
            let tile = match first.take() {
                Some(tile) => tile,
                None => tiles.tile(band_index * columns + column)?,
            };
            let s = tile.shape;
            if (s.format, s.bit_depth_luma, s.bit_depth_chroma) != (shape.format, shape.bit_depth_luma, shape.bit_depth_chroma) {
                return Err(HeifError::Unsupported("decode: grid tiles with inconsistent bit depth/chroma format are not supported"));
            }
            if (s.width, s.height) != (shape.width, shape.height) {
                return Ok(None);
            }
            layout.place(&mut band, column, &tile);
            pool.give(tile.pic);
        }
        luma.rows.extend(band.luma);
        cb.rows.extend(band.cb);
        cr.rows.extend(band.cr);
        let finished = band_index + 1 == rows;

        // Every source row whose chroma is all here, in order, in batches.
        let mut ready = next_y;
        while ready < luma.end() && needs[ready].is_none_or(|(_, hi)| finished || hi < cb.end()) {
            ready += 1;
        }
        for batch in (next_y..ready).step_by(ROWS_PER_BATCH) {
            let batch_end = (batch + ROWS_PER_BATCH).min(ready);
            // Upright, no interpolation, whole rows kept in order: each source row
            // is converted straight into its output row.
            if let (true, false, None, Some((0, 0, len, false)), Sink::Full { width, data }, false) =
                (flat, turned, alpha, scatter.span, &mut *sink, *NO_IN_PLACE)
                && len == *width
            {
                let row = *width * 3;
                let touched = scatter.rows_touched(batch, batch_end - batch, data.len() / row.max(1));
                let start = touched.start;
                data[touched.start * row..touched.end * row].par_chunks_mut(row).enumerate().for_each(|(i, out)| {
                    let Some(y) = scatter.source_of_row[start + i].filter(|y| (batch..batch_end).contains(y)) else { return };
                    let mut out = crate::color::InPlace { row: out.as_chunks_mut::<3>().0, at: 0 };
                    flat_row_to(&mut out, y, &luma, &cb, &cr, column_arrays, rows_axis, pixel);
                });
                continue;
            }
            let converted: Vec<Vec<[u8; 3]>> = if flat {
                // No interpolation anywhere: each pixel's chroma is one sample.
                (batch..batch_end).into_par_iter().map(|y| flat_row(y, &luma, &cb, &cr, column_arrays, rows_axis, pixel)).collect()
            } else if turned || !options.reuse_chroma_rows {
                // A turned picture's rows share no interpolated chroma.
                (batch..batch_end).into_par_iter().map(|y| convert_row(y, &luma, &cb, &cr, column_arrays, rows_axis, turned, pixel)).collect()
            } else {
                let tasks: Vec<usize> = (batch..batch_end).step_by(ROWS_PER_TASK).collect();
                tasks
                    .into_par_iter()
                    .flat_map_iter(|start| {
                        let mut rows = RowConverter::new(&luma, &cb, &cr, column_arrays, rows_axis, pixel);
                        (start..(start + ROWS_PER_TASK).min(batch_end)).map(move |y| rows.convert(y))
                    })
                    .collect()
            };
            sink.put_batch(scatter, batch, &converted, alpha);
        }
        next_y = ready;
        luma.drop_below(next_y);
        let keep = keep_from[next_y.min(height)];
        cb.drop_below(keep.min(cb.end()));
        cr.drop_below(keep.min(cr.end()));
    }
    if next_y != height {
        return Err(HeifError::MalformedHevc("RGB: rows left unconverted"));
    }
    Ok(Some(()))
}

/// The source columns as flat arrays, for the row loops: the chroma
/// columns each reads and their weight (0 for a column a crop drops).
struct Columns {
    lo: Vec<usize>,
    hi: Vec<usize>,
    weight: Vec<f32>,
    /// `Some((lo[0], phase, last))` when column `x` reads chroma
    /// `min(lo[0] + (x + phase) / 2, last)` (4:2:0, nearest), `phase` 0 or 1.
    halves: Option<(usize, usize, usize)>,
    /// `Some(lo[0])` when column `x` blends chroma `lo[0] + x / 2` and the next
    /// one with weight 0 (even `x`) or 0.5 (odd `x`), but for the last columns,
    /// clamped to the last chroma sample (4:2:0, bilinear, co-sited).
    bilinear_halves: Option<(usize, usize)>,
}

impl Columns {
    fn new(axis: &SourceAxis) -> Self {
        let get = |f: fn(&Entry) -> usize| axis.entries.iter().map(|e| e.as_ref().map_or(0, f)).collect();
        let lo: Vec<usize> = get(|e| e.lo);
        let halves = lo.first().copied().filter(|_| axis.entries.iter().all(Option::is_some)).and_then(|lo0| {
            // the last columns may be clamped to the last chroma sample
            let last = lo.iter().copied().max().unwrap_or(0);
            (0..2).find(|&phase| lo.iter().enumerate().all(|(x, &l)| l == (lo0 + (x + phase) / 2).min(last))).map(|phase| (lo0, phase, last))
        });
        let (hi, weight): (Vec<usize>, Vec<f32>) = (get(|e| e.hi), axis.entries.iter().map(|e| e.map_or(0.0, |e| e.weight)).collect());
        let last_hi = hi.iter().copied().max().unwrap_or(0);
        let bilinear_halves = lo.first().copied().filter(|_| axis.entries.iter().all(Option::is_some)).map(|lo0| (lo0, last_hi)).filter(|&(lo0, last)| {
            lo.iter().zip(&hi).zip(&weight).enumerate().all(|(x, ((&l, &h), &w))| {
                let (el, eh) = ((lo0 + x / 2).min(last), (lo0 + x / 2 + 1).min(last));
                l == el && h == eh && w == if x % 2 == 1 && el != eh { 0.5 } else { 0.0 }
            })
        });
        Columns { lo, hi, weight, halves, bilinear_halves }
    }
}

/// A chroma row interpolated along the columns: its index, Cb and Cr.
type Along = Option<(usize, Vec<f32>, Vec<f32>)>;

/// Source rows through the converter's arithmetic, laid out by rows: the
/// same expressions as `ChromaRows::slot`/`blend` and `Matrix::row`,
/// evaluated once per value. Every index comes from the geometry, not the
/// data, and lies inside the rows held; a miss would read 0, not panic.
/// Consecutive rows share the chroma rows they interpolate and the buffers.
struct RowConverter<'a, S> {
    luma: &'a Rows<S>,
    cb: &'a Rows<S>,
    cr: &'a Rows<S>,
    columns: &'a Columns,
    rows: &'a SourceAxis,
    pixel: &'a PixelRgb,
    /// Chroma rows interpolated along the columns, Cb and Cr, by chroma row
    /// index.
    along: [Along; 2],
    cb_row: Vec<f32>,
    cr_row: Vec<f32>,
    luma_row: Vec<i32>,
}

impl<'a, S: BandSample> RowConverter<'a, S> {
    fn new(luma: &'a Rows<S>, cb: &'a Rows<S>, cr: &'a Rows<S>, columns: &'a Columns, rows: &'a SourceAxis, pixel: &'a PixelRgb) -> Self {
        RowConverter { luma, cb, cr, columns, rows, pixel, along: [None, None], cb_row: Vec::new(), cr_row: Vec::new(), luma_row: Vec::new() }
    }

    /// Source row `y`, converted.
    fn convert(&mut self, y: usize) -> Vec<[u8; 3]> {
        let samples = self.luma.get(y);
        let Some(ry) = self.rows.entries[y] else { return Vec::new() };
        let mut out = vec![[0u8; 3]; samples.len()];
        let pixel = self.pixel;
        if pixel.monochrome() {
            for (o, &s) in out.iter_mut().zip(samples) {
                *o = pixel.rgb8(s.value(), 0.0, 0.0);
            }
            return out;
        }
        let columns = self.columns;
        let (cb_row, cr_row) = (&mut self.cb_row, &mut self.cr_row);
        cb_row.clear();
        cr_row.clear();
        // Each chroma row interpolated to this row's width once, then blended
        // for every pixel of the row; then the row through the planar
        // converter's own matrix loop.
        let lo = along(&mut self.along, ry.lo, ry.hi, self.cb, self.cr, columns);
        let hi = along(&mut self.along, ry.hi, ry.lo, self.cb, self.cr, columns);
        let (top, bottom) = (self.along[lo].as_ref().map(|a| (&a.1, &a.2)), self.along[hi].as_ref().map(|a| (&a.1, &a.2)));
        if let (Some((cb_top, cr_top)), Some((cb_bottom, cr_bottom))) = (top, bottom) {
            let blend = |t: &[f32], b: &[f32], to: &mut Vec<f32>| to.extend(t.iter().zip(b).map(|(&t, &b)| PixelRgb::lerp_rows(t, b, ry.weight)));
            blend(cb_top, cb_bottom, cb_row);
            blend(cr_top, cr_bottom, cr_row);
        }
        self.luma_row.clear();
        self.luma_row.extend(samples.iter().map(|&s| s.value()));
        pixel.row8(&mut out, &self.luma_row, cb_row, cr_row);
        out
    }
}

/// One source row through the converter's arithmetic, laid out by rows:
/// the same expressions as `ChromaRows::slot`/`blend` and `Matrix::row`,
/// evaluated once per value. Every index comes from the geometry, not the
/// data, and lies inside the rows held; a miss would read 0, not panic.
#[allow(clippy::too_many_arguments)]
fn convert_row<S: BandSample>(y: usize, luma: &Rows<S>, cb: &Rows<S>, cr: &Rows<S>, columns: &Columns, rows: &SourceAxis, turned: bool, pixel: &PixelRgb) -> Vec<[u8; 3]> {
    let samples = luma.get(y);
    let Some(ry) = rows.entries[y] else { return Vec::new() };
    let mut out = vec![[0u8; 3]; samples.len()];
    if pixel.monochrome() {
        for (o, &s) in out.iter_mut().zip(samples) {
            *o = pixel.rgb8(s.value(), 0.0, 0.0);
        }
        return out;
    }
    let at = |row: &[S], i: usize| row.get(i).map_or(0, |s| s.value());
    let (cb_lo, cb_hi, cr_lo, cr_hi) = (cb.get(ry.lo), cb.get(ry.hi), cr.get(ry.lo), cr.get(ry.hi));
    // Chroma blended for every pixel of the row, then the row through the
    // planar converter's own matrix loop.
    let (cb_row, cr_row): (Vec<f32>, Vec<f32>) = if turned {
        // The transformed picture's chroma rows run along source columns:
        // its x axis comes from this source row, so the "horizontal"
        // interpolation is between this row's two chroma rows, the same
        // for every source chroma column. Then the vertical one, per pixel.
        let across =
            |lo: &[S], hi: &[S]| -> Vec<f32> { (0..lo.len().max(hi.len())).map(|c| PixelRgb::lerp_samples(at(lo, c), at(hi, c), ry.weight)).collect() };
        let (h_cb, h_cr) = (across(cb_lo, cb_hi), across(cr_lo, cr_hi));
        let pick = |h: &[f32], i: usize| h.get(i).copied().unwrap_or(0.0);
        let blend = |h: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(columns.lo.len());
            if let Some((lo0, last)) = columns.bilinear_halves {
                crate::color::blend_halves(&mut out, h.get(lo0..=last).unwrap_or(&[]), columns.lo.len());
            }
            let done = out.len();
            out.extend(
                columns.lo.iter().zip(&columns.hi).zip(&columns.weight).skip(done).map(|((&lo, &hi), &w)| PixelRgb::lerp_rows(pick(h, lo), pick(h, hi), w)),
            );
            out
        };
        (blend(&h_cb), blend(&h_cr))
    } else {
        // Each chroma row interpolated to this row's width, then blended.
        let along = |row: &[S]| -> Vec<f32> {
            columns.lo.iter().zip(&columns.hi).zip(&columns.weight).map(|((&lo, &hi), &w)| PixelRgb::lerp_samples(at(row, lo), at(row, hi), w)).collect()
        };
        let blend = |top: Vec<f32>, bottom: Vec<f32>| -> Vec<f32> { top.iter().zip(&bottom).map(|(&t, &b)| PixelRgb::lerp_rows(t, b, ry.weight)).collect() };
        (blend(along(cb_lo), along(cb_hi)), blend(along(cr_lo), along(cr_hi)))
    };
    let luma_row: Vec<i32> = samples.iter().map(|&s| s.value()).collect();
    pixel.row8(&mut out, &luma_row, &cb_row, &cr_row);
    out
}

/// A source row when no chroma sample is interpolated (every weight 0):
/// the arithmetic of `convert_row` with each interpolation reduced to the
/// sample it keeps, exactly, as `x * 1.0 + y * 0.0 == x` for the
/// non-negative integers samples are. Upright or turned, that sample is
/// chroma row `ry.lo` at column `columns.lo[x]`.
fn flat_row<S: BandSample>(y: usize, luma: &Rows<S>, cb: &Rows<S>, cr: &Rows<S>, columns: &Columns, rows: &SourceAxis, pixel: &PixelRgb) -> Vec<[u8; 3]> {
    let mut out = Vec::new();
    if rows.entries[y].is_some() {
        out.reserve_exact(luma.get(y).len());
        flat_row_to(&mut out, y, luma, cb, cr, columns, rows, pixel);
    }
    out
}

/// [`flat_row`] written to `out`.
#[allow(clippy::too_many_arguments)]
fn flat_row_to<O: crate::color::RowOut, S: BandSample>(out: &mut O, y: usize, luma: &Rows<S>, cb: &Rows<S>, cr: &Rows<S>, columns: &Columns, rows: &SourceAxis, pixel: &PixelRgb) {
    let samples = luma.get(y);
    let Some(ry) = rows.entries[y] else { return };
    if pixel.monochrome() {
        for &s in samples {
            out.push(pixel.rgb8(s.value(), 0.0, 0.0));
        }
        return;
    }
    if let Some((lo0, phase, last)) = columns.halves
        && !*NO_FUSED
    {
        let (cb_row, cr_row) = (cb.get(ry.lo), cr.get(ry.lo));
        let chroma = |x: usize| {
            let c = columns.lo.get(x).copied().unwrap_or(0);
            (at(cb_row, c) as f32, at(cr_row, c) as f32)
        };
        // With phase 1, pixel 0 has its chroma sample alone; pairs start at pixel 1.
        let head = phase.min(samples.len());
        for (x, &s) in samples.iter().enumerate().take(head) {
            let (c_b, c_r) = chroma(x);
            out.push(pixel.rgb8(s.value(), c_b, c_r));
        }
        let c0 = lo0 + phase;
        let (cb_k, cr_k) = (cb_row.get(c0..=last).unwrap_or(&[]), cr_row.get(c0..=last).unwrap_or(&[]));
        let done = head + pixel.row8_420_nearest(out, &samples[head..], cb_k, cr_k);
        for (x, &s) in samples.iter().enumerate().skip(done) {
            let (c_b, c_r) = chroma(x);
            out.push(pixel.rgb8(s.value(), c_b, c_r));
        }
        return;
    }
    let pick = |row: &[S]| -> Vec<f32> { columns.lo.iter().map(|&lo| at(row, lo) as f32).collect() };
    let (cb_row, cr_row) = (pick(cb.get(ry.lo)), pick(cr.get(ry.lo)));
    let luma_row: Vec<i32> = samples.iter().map(|&s| s.value()).collect();
    pixel.row8(out.next(samples.len()), &luma_row, &cb_row, &cr_row);
}

/// A chroma sample as `i32`, 0 past the row's end.
fn at<S: BandSample>(row: &[S], i: usize) -> i32 {
    row.get(i).map_or(0, |s| s.value())
}

/// The slot of `slots` holding chroma row `index` interpolated along the
/// columns. A missing row replaces one other than `keep`, the row the same
/// output row also reads, and otherwise the lower one.
fn along<S: BandSample>(slots: &mut [Along; 2], index: usize, keep: usize, cb: &Rows<S>, cr: &Rows<S>, columns: &Columns) -> usize {
    if let Some(slot) = slots.iter().position(|s| s.as_ref().is_some_and(|s| s.0 == index)) {
        return slot;
    }
    let key = |slot: &Along| slot.as_ref().map(|s| s.0);
    let slot = match (key(&slots[0]), key(&slots[1])) {
        (None, _) => 0,
        (_, None) => 1,
        (Some(a), _) if a == keep && index != keep => 1,
        (_, Some(b)) if b == keep && index != keep => 0,
        (Some(a), Some(b)) => usize::from(b < a),
    };
    let (_, cb_along, cr_along) = slots[slot].get_or_insert_with(|| (index, Vec::new(), Vec::new()));
    let interpolate = |row: &[S], to: &mut Vec<f32>| {
        to.clear();
        to.extend(columns.lo.iter().zip(&columns.hi).zip(&columns.weight).map(|((&lo, &hi), &w)| PixelRgb::lerp_samples(at(row, lo), at(row, hi), w)));
    };
    interpolate(cb.get(index), cb_along);
    interpolate(cr.get(index), cr_along);
    if let Some(s) = slots[slot].as_mut() {
        s.0 = index;
    }
    slot
}

/// `HEIC_NO_TURN_BLOCKS=1`: gather turned rows pixel by pixel (A/B runs).
static NO_TURN_BLOCKS: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("HEIC_NO_TURN_BLOCKS").as_deref() == Ok("1"));
/// `HEIC_NO_FUSED=1`: nearest 4:2:0 rows through the general row loop (A/B runs).
static NO_FUSED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("HEIC_NO_FUSED").as_deref() == Ok("1"));
/// `HEIC_ZERO_FILL=1`: the output written with zeros first, as before (A/B runs).
static ZERO_FILL: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("HEIC_ZERO_FILL").as_deref() == Ok("1"));
/// `HEIC_NO_IN_PLACE=1`: upright rows converted to their own buffers, then copied (A/B runs).
static NO_IN_PLACE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("HEIC_NO_IN_PLACE").as_deref() == Ok("1"));
/// `HEIC_WIDE_BANDS=1`: 8-bit pictures held in `u16` band rows, as before (A/B runs).
static WIDE_BANDS: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("HEIC_WIDE_BANDS").as_deref() == Ok("1"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_maps_follow_turns_flips_and_crops() {
        // 4×2, one anticlockwise turn: transformed (0, 0) is source (3, 0).
        let m = PlaneMap::new(4, 2, &[Op::Turn], None);
        assert_eq!((m.width, m.height), (2, 4));
        let src = |m: &PlaneMap, x: usize, y: usize| {
            let (a, b) = (m.axes[0].eval(x), m.axes[1].eval(y));
            if m.axes[0].from_y { (b, a) } else { (a, b) }
        };
        assert_eq!(src(&m, 0, 0), (3, 0));
        assert_eq!(src(&m, 1, 0), (3, 1));
        assert_eq!(src(&m, 0, 3), (0, 0));
        let m = PlaneMap::new(4, 2, &[Op::FlipX, Op::FlipY], Some((1, 0, 2, 2)));
        assert_eq!(src(&m, 0, 0), (2, 1));
        assert_eq!(m.axes[0].invert(2, 2), Some(0));
        assert_eq!(m.axes[0].invert(0, 2), None);
    }

    #[test]
    fn fits_within_and_never_scales_up() {
        assert_eq!(fit_within((4032, 3024), Some(1568)), (1568, 1176));
        assert_eq!(fit_within((800, 600), Some(1568)), (800, 600));
        assert_eq!(fit_within((800, 600), None), (800, 600));
        assert_eq!(fit_within((100_000, 10), Some(1568)), (1568, 1));
    }

    #[test]
    fn averaging_bins_cover_the_whole_output() {
        let (full, out) = ((5, 3), fit_within((5, 3), Some(2)));
        assert_eq!(out, (2, 1));
        let mut bins = Bins::new(full, out, [AxisMap::identity(5, 2, 1), AxisMap::identity(3, 1, 2)]).unwrap();
        for y in 0..3 {
            for x in 0..5 {
                bins.add(x, y, [if x < 3 { 0 } else { 200 }, 10, 255]);
            }
        }
        assert_eq!(bins.finish().data, [0, 10, 255, 200, 10, 255]);
    }
}
