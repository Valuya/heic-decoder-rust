//! Overlay (`iovl`) images, checked against their inputs decoded on their own
//! and composed here as libheif composes them. The fixtures reuse the coded
//! images of this repository's HM-checked fixtures; see
//! fixtures/overlay/README.md. With `HEIF_CONFORMANCE_DIR` set, `--ignored`
//! also checks the Nokia conformance overlays C015 to C021.
use heic_decoder::{
    ChromaUpsampling, ColorInfo, DecodeOptions, DecodeRgbOptions, DecodedImage, HeifError, RgbOptions, decode, decode_rgb8, decode_rgb8_with, decode_with,
    probe,
};
use std::{env, fs, path::PathBuf};

const OVERLAY: &[u8] = include_bytes!("fixtures/overlay/overlay.heic");
const WIDE: &[u8] = include_bytes!("fixtures/overlay/overlay-wide.heic");
const PATTERN8: &[u8] = include_bytes!("fixtures/422/pattern8.heic");
const LOSSLESS444: &[u8] = include_bytes!("fixtures/alpha/lossless444.heic");
const MONO8: &[u8] = include_bytes!("fixtures/alpha/mono8.heic");
const LOSSLESS420: &[u8] = include_bytes!("fixtures/alpha/lossless420.heic");

/// Interleaved RGB or RGBA samples at the canvas depth.
#[derive(Clone, Debug, PartialEq)]
struct Pixels {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<u32>,
}

impl Pixels {
    fn at(&self, x: usize, y: usize) -> &[u32] {
        &self.data[(y * self.width + x) * self.channels..][..self.channels]
    }
    fn remap(&self, width: usize, height: usize, source: impl Fn(usize, usize) -> (usize, usize)) -> Pixels {
        let mut data = Vec::with_capacity(width * height * self.channels);
        for y in 0..height {
            for x in 0..width {
                let (sx, sy) = source(x, y);
                data.extend_from_slice(self.at(sx, sy));
            }
        }
        Pixels { width, height, channels: self.channels, data }
    }
    /// One 90° anticlockwise turn, as `irot` 1.
    fn turned(&self) -> Pixels {
        self.remap(self.height, self.width, |x, y| (self.width - 1 - y, x))
    }
    fn mirrored(&self) -> Pixels {
        self.remap(self.width, self.height, |x, y| (self.width - 1 - x, y))
    }
    fn with_alpha(&self, alpha: &[u32]) -> Pixels {
        let data = self.data.chunks(3).zip(alpha).flat_map(|(rgb, &a)| [rgb[0], rgb[1], rgb[2], a]).collect();
        Pixels { channels: 4, data, ..*self }
    }
}

/// RGB at `depth` bits, through the public converters.
fn rgb(image: &DecodedImage, depth: u32, options: &RgbOptions) -> Pixels {
    let data = if depth == 8 {
        image.to_rgb8_with_options(options).unwrap().data.into_iter().map(u32::from).collect()
    } else {
        let max = (1u32 << depth) - 1;
        image.to_rgb16_with_options(options).unwrap().data.into_iter().map(|v| (u32::from(v) * max + 32767) / 65535).collect()
    };
    Pixels { width: image.width, height: image.height, channels: 3, data }
}

/// libheif's composition: fill reduced to `depth` bits, inputs clipped and
/// copied, or blended by their alpha with truncation.
fn compose(fill: [u16; 3], width: usize, height: usize, depth: u32, inputs: &[(&Pixels, i64, i64)]) -> Pixels {
    let max = (1u32 << depth) - 1;
    let mut canvas =
        Pixels { width, height, channels: 3, data: [0, 1, 2].repeat(width * height).into_iter().map(|c| u32::from(fill[c] >> (16 - depth))).collect() };
    for &(input, dx, dy) in inputs {
        for y in 0..height as i64 {
            for x in 0..width as i64 {
                let (sx, sy) = (x - dx, y - dy);
                if sx < 0 || sy < 0 || sx >= input.width as i64 || sy >= input.height as i64 {
                    continue;
                }
                let source = input.at(sx as usize, sy as usize).to_vec();
                let target = &mut canvas.data[(y as usize * width + x as usize) * 3..][..3];
                for c in 0..3 {
                    target[c] = if input.channels == 4 { (source[c] * source[3] + target[c] * (max - source[3])) / max } else { source[c] };
                }
            }
        }
    }
    canvas
}

/// The composed picture `decode` returns: green, blue, red in Y, Cb, Cr.
fn planes(image: &DecodedImage) -> Pixels {
    let data = (0..image.width * image.height).flat_map(|i| [image.cr[i], image.y[i], image.cb[i]]).map(|v| v as u32).collect();
    Pixels { width: image.width, height: image.height, channels: 3, data }
}

/// `overlay.heic` composed from its inputs, before the overlay's own turns.
fn expected_overlay(transform: bool, options: &RgbOptions) -> Pixels {
    let pattern = rgb(&decode(PATTERN8).unwrap(), 8, options);
    let lossless = rgb(&decode(LOSSLESS444).unwrap(), 8, options);
    let turned = if transform { lossless.turned().mirrored() } else { lossless };
    let mono = rgb(&decode(MONO8).unwrap(), 8, options);
    // Item 4 is the 4:2:2 picture with the monochrome picture as 8-bit alpha.
    let alpha: Vec<u32> = decode(MONO8).unwrap().y.iter().map(|&v| v as u32).collect();
    let translucent = pattern.with_alpha(&alpha);
    compose([0x1234, 0xABCD, 0x8000], 100, 80, 8, &[(&pattern, -10, -6), (&turned, 50, 30), (&mono, 20, 60), (&translucent, 30, 10), (&pattern, 200, 0)])
}

#[test]
fn overlay_is_its_inputs_composed_in_rgb_then_turned() {
    let image = decode(OVERLAY).unwrap();
    assert_eq!((image.width, image.height, image.cb_width, image.cb_height), (80, 100, 80, 100));
    assert_eq!((image.chroma_format_idc, image.bit_depth_luma, image.bit_depth_chroma), (3, 8, 8));
    let color = image.color_info.unwrap();
    assert_eq!((color.matrix_coefficients, color.full_range), (0, true));
    assert!(image.alpha.is_none());
    let turned = expected_overlay(true, &RgbOptions::default()).turned().turned().turned();
    assert_eq!(planes(&image), turned);
    let rgb8 = image.to_rgb8().unwrap();
    assert_eq!(rgb8.data.iter().map(|&v| u32::from(v)).collect::<Vec<_>>(), turned.data);
}

#[test]
fn transforms_off_leaves_inputs_and_overlay_untransformed() {
    let image = decode_with(OVERLAY, &DecodeOptions { apply_transforms: false }).unwrap();
    assert_eq!((image.width, image.height), (100, 80));
    assert_eq!(planes(&image), expected_overlay(false, &RgbOptions::default()));
}

#[test]
fn canvas_takes_the_first_inputs_bit_depth() {
    // 32-bit descriptor fields; a 10-bit 4:2:0 input first, then the 8-bit
    // 4:2:2 one, converted up to 10 bits.
    let image = decode(WIDE).unwrap();
    assert_eq!((image.width, image.height, image.bit_depth_luma, image.chroma_format_idc), (96, 72, 10, 3));
    let options = RgbOptions::default();
    let (deep, pattern) = (rgb(&decode(LOSSLESS420).unwrap(), 10, &options), rgb(&decode(PATTERN8).unwrap(), 10, &options));
    assert_eq!(planes(&image), compose([0xFFFF, 0x0000, 0x4321], 96, 72, 10, &[(&deep, 40, 8), (&pattern, -20, -30)]));
    let info = probe(WIDE).unwrap();
    assert_eq!((info.width, info.height, info.bit_depth_luma, info.bit_depth_chroma, info.chroma_format_idc), (96, 72, 10, 10, 3));
}

#[test]
fn probe_reports_the_canvas_and_its_transforms() {
    let info = probe(OVERLAY).unwrap();
    assert_eq!((info.width, info.height, info.coded_width, info.coded_height), (80, 100, 100, 80));
    assert_eq!((info.bit_depth_luma, info.chroma_format_idc, info.alpha_bit_depth), (8, 3, None));
}

#[test]
fn rgb_decoding_matches_the_planar_path_and_respects_limits() {
    let full = decode(OVERLAY).unwrap().to_rgb8().unwrap();
    assert_eq!(decode_rgb8(OVERLAY).unwrap(), full);
    let small = decode_rgb8_with(OVERLAY, &DecodeRgbOptions { max_edge: Some(40), ..Default::default() }).unwrap();
    assert_eq!((small.width, small.height), (32, 40));

    // A colour override applies to the inputs, not to the composed RGB.
    let rgb = RgbOptions {
        color: Some(ColorInfo { color_primaries: 1, transfer_characteristics: 1, matrix_coefficients: 1, full_range: true }),
        ..Default::default()
    };
    let overridden = decode_rgb8_with(OVERLAY, &DecodeRgbOptions { rgb, ..Default::default() }).unwrap();
    let want = expected_overlay(true, &rgb).turned().turned().turned();
    assert_eq!(overridden.data.iter().map(|&v| u32::from(v)).collect::<Vec<_>>(), want.data);
    assert_ne!(overridden, full);
    let rgb = RgbOptions { upsampling: ChromaUpsampling::Nearest, ..Default::default() };
    let nearest = decode_rgb8_with(OVERLAY, &DecodeRgbOptions { rgb, ..Default::default() }).unwrap();
    assert_eq!(nearest.data.iter().map(|&v| u32::from(v)).collect::<Vec<_>>(), expected_overlay(true, &rgb).turned().turned().turned().data);
    assert_ne!(nearest, full);

    // Canvas 100x80; five placements of 64x64 pictures plus one 64x64 alpha.
    let limited =
        |max_picture_pixels, max_total_pixels| decode_rgb8_with(OVERLAY, &DecodeRgbOptions { max_picture_pixels, max_total_pixels, ..Default::default() });
    assert!(limited(4096, 6 * 4096).is_ok());
    assert!(matches!(limited(4095, 6 * 4096), Err(HeifError::LimitExceeded(_))));
    assert!(matches!(limited(4096, 6 * 4096 - 1), Err(HeifError::LimitExceeded(_))));
    assert!(matches!(limited(1 << 20, 7999), Err(HeifError::LimitExceeded(_))));
}

#[test]
fn malformed_descriptors_are_errors() {
    let at = OVERLAY.windows(10).position(|w| w == [0, 0, 0x12, 0x34, 0xAB, 0xCD, 0x80, 0x00, 0xFF, 0xFF]).unwrap();
    let with = |offset: usize, bytes: &[u8]| {
        let mut file = OVERLAY.to_vec();
        file[at + offset..at + offset + bytes.len()].copy_from_slice(bytes);
        file
    };
    assert!(matches!(decode(&with(0, &[1])), Err(HeifError::Unsupported(_))));
    assert!(matches!(decode(&with(10, &[0, 0])), Err(HeifError::MalformedBox(_))));
    // 32-bit fields make the descriptor too short for five offsets.
    assert!(decode(&with(1, &[1])).is_err());
    // Any single changed descriptor byte decodes or fails, without panicking.
    for byte in 0..34 {
        for value in [0x00, 0x7F, 0x80, 0xFF] {
            let _ = decode(&with(byte, &[value]));
            let _ = decode_rgb8(&with(byte, &[value]));
        }
    }
}

#[test]
#[ignore = "requires HEIF_CONFORMANCE_DIR pointing to Nokia conformance_files"]
fn nokia_overlays_are_their_inputs_composed() {
    let root = PathBuf::from(env::var_os("HEIF_CONFORMANCE_DIR").expect("set HEIF_CONFORMANCE_DIR"));
    // The overlay is not the primary item in these files: point `pitm` at it.
    let select = |data: &[u8], item: u16| {
        let mut data = data.to_vec();
        let at = data.windows(4).position(|w| w == b"pitm").unwrap() + 8;
        data[at..at + 2].copy_from_slice(&item.to_be_bytes());
        data
    };
    type Case = (&'static str, u16, [u16; 3], (usize, usize), &'static [(u16, i64, i64)]);
    let cases: [Case; 7] = [
        ("C015.heic", 1006, [0, 0, 0], (1440, 960), &[(1002, 0, 0), (1005, 160, 240)]),
        ("C016.heic", 1003, [0, 0, 0], (1440, 960), &[(1002, 80, 120)]),
        ("C017.heic", 1006, [0xFFFF; 3], (1440, 960), &[(1005, 0, 0), (1002, 640, 360)]),
        ("C018.heic", 1006, [0, 0, 0], (1440, 960), &[(1005, 0, 0), (1002, 640, 360)]),
        ("C019.heic", 1006, [0xFFFF; 3], (1440, 960), &[(1005, 0, 0), (1002, -320, -180)]),
        ("C020.heic", 1006, [0xFFFF; 3], (1440, 960), &[(1005, 0, 0), (1002, -640, -360)]),
        ("C021.heic", 1003, [0xFFFF; 3], (640, 360), &[(1002, 0, 0), (1002, -640, -360)]),
    ];
    for (name, overlay, fill, (width, height), placements) in cases {
        let data = fs::read(root.join(name)).unwrap();
        let image = decode(&select(&data, overlay)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((image.width, image.height, image.chroma_format_idc), (width, height, 3), "{name}");
        let inputs: Vec<(Pixels, i64, i64)> =
            placements.iter().map(|&(item, x, y)| (rgb(&decode(&select(&data, item)).unwrap(), 8, &RgbOptions::default()), x, y)).collect();
        let inputs: Vec<_> = inputs.iter().map(|(pixels, x, y)| (pixels, *x, *y)).collect();
        assert!(planes(&image) == compose(fill, width, height, 8, &inputs), "{name}");
        assert_eq!(decode_rgb8(&select(&data, overlay)).unwrap(), image.to_rgb8().unwrap(), "{name}");
    }
}
