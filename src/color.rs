//! Optional conversion from native planar samples to interleaved RGB.
//!
//! The matrix/range equations and chroma locations follow ITU-T H.273.
//! Output retains the source transfer function and primaries: this is not
//! an ICC transform, sRGB conversion, linearisation or HDR tone mapping.
use crate::{DecodedImage, HeifError, MAX_IMAGE_PIXELS, Result};
use rayon::prelude::*;

/// Coding-independent colour information from `colr/nclx` or SPS VUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorInfo {
    pub color_primaries: u16,
    pub transfer_characteristics: u16,
    pub matrix_coefficients: u16,
    pub full_range: bool,
}

impl Default for ColorInfo {
    /// Unlabelled images use BT.601 limited range, as libheif does (and
    /// MIAF's default matrix). Primaries and transfer remain unspecified;
    /// this does not declare sRGB.
    fn default() -> Self {
        Self { color_primaries: 2, transfer_characteristics: 2, matrix_coefficients: 6, full_range: false }
    }
}

/// Position of the returned chroma raster relative to the luma raster.
/// Positions are in luma pixels, with luma sample (0,0) at the origin.
/// Crops can move the first chroma sample outside the luma output window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChromaSampling {
    pub sub_x: u32,
    pub sub_y: u32,
    pub origin_x: f32,
    pub origin_y: f32,
}

impl ChromaSampling {
    pub(crate) fn coded(format: u32, location: Option<u32>) -> Self {
        let (sub_x, sub_y) = match format {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        // H.265's inferred 4:2:0 location is type 0 (left, vertically centred).
        let (origin_x, origin_y) = if format == 1 {
            match location.unwrap_or(0) {
                1 => (0.5, 0.5),
                2 => (0.0, 0.0),
                3 => (0.5, 0.0),
                4 => (0.0, 1.0),
                5 => (0.5, 1.0),
                _ => (0.0, 0.5),
            }
        } else {
            (0.0, 0.0)
        };
        Self { sub_x, sub_y, origin_x, origin_y }
    }
}

/// Chroma reconstruction filter, with edge samples extended at boundaries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChromaUpsampling {
    Nearest,
    #[default]
    Bilinear,
}

/// Optional overrides for applications with external colour information.
#[derive(Debug, Clone, Copy, Default)]
pub struct RgbOptions {
    /// Override the file's matrix and range; `None` uses file information.
    pub color: Option<ColorInfo>,
    pub upsampling: ChromaUpsampling,
}

/// Tightly packed RGB triples, row-major. `u16` values span 0..=65535,
/// preserving higher-depth input without an intermediate 8-bit conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage<T> {
    pub width: usize,
    pub height: usize,
    pub data: Vec<T>,
}

/// Tightly packed RGBA. Alpha spans the entire destination range, independently
/// of the colour range. `premultiplied` preserves the container's `prem` flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage<T> {
    pub width: usize,
    pub height: usize,
    pub data: Vec<T>,
    pub premultiplied: bool,
}

impl DecodedImage {
    /// Convert to RGB8 using file colour information and bilinear chroma.
    /// Alpha is omitted; premultiplied colour is not unpremultiplied.
    pub fn to_rgb8(&self) -> Result<RgbImage<u8>> {
        self.to_rgb8_with_options(&RgbOptions::default())
    }
    /// Convert to RGB16 using file colour information and bilinear chroma.
    /// Alpha is omitted; premultiplied colour is not unpremultiplied.
    pub fn to_rgb16(&self) -> Result<RgbImage<u16>> {
        self.to_rgb16_with_options(&RgbOptions::default())
    }
    pub fn to_rgb8_with_options(&self, options: &RgbOptions) -> Result<RgbImage<u8>> {
        convert::<_, 3>(self, options)
    }
    pub fn to_rgb16_with_options(&self, options: &RgbOptions) -> Result<RgbImage<u16>> {
        convert::<_, 3>(self, options)
    }
    /// Convert to RGBA8, using an opaque alpha channel when none is present.
    pub fn to_rgba8(&self) -> Result<RgbaImage<u8>> {
        self.to_rgba8_with_options(&RgbOptions::default())
    }
    /// Convert to RGBA16 without an intermediate 8-bit colour or alpha plane.
    pub fn to_rgba16(&self) -> Result<RgbaImage<u16>> {
        self.to_rgba16_with_options(&RgbOptions::default())
    }
    pub fn to_rgba8_with_options(&self, options: &RgbOptions) -> Result<RgbaImage<u8>> {
        convert_rgba(self, options)
    }
    pub fn to_rgba16_with_options(&self, options: &RgbOptions) -> Result<RgbaImage<u16>> {
        convert_rgba(self, options)
    }
}

trait RgbSample: Copy + Default + Send + Sync {
    fn quantize(value: f32) -> Self;
    fn alpha(sample: u32, max: u32) -> Self;
}
impl RgbSample for u8 {
    fn alpha(sample: u32, max: u32) -> Self {
        ((u64::from(sample) * 255 + u64::from(max / 2)) / u64::from(max)) as u8
    }
    fn quantize(value: f32) -> Self {
        if HAS_ROUND {
            // `as` saturates: below zero and NaN become 0, above 255 becomes 255.
            (value * 255.0).round() as u8
        } else {
            quantize_without_round(value, 255.0, 255) as u8
        }
    }
}
impl RgbSample for u16 {
    fn alpha(sample: u32, max: u32) -> Self {
        ((u64::from(sample) * 65535 + u64::from(max / 2)) / u64::from(max)) as u16
    }
    fn quantize(value: f32) -> Self {
        if HAS_ROUND { (value * 65535.0).round() as u16 } else { quantize_without_round(value, 65535.0, 65535) as u16 }
    }
}

/// Whether `f32::round` is an instruction here. Baseline x86-64 (SSE2) has
/// none, so `round` is a `roundf` call per channel and the loop does not
/// vectorize; there [`quantize_without_round`] is used instead.
const HAS_ROUND: bool = cfg!(any(target_arch = "aarch64", target_feature = "sse4.1"));

/// `value * scale` rounded half away from zero and clamped to `0..=max`,
/// NaN to 0: the same as `(value * scale).round() as u8` (or `u16`) for every
/// `f32`, all of which were compared, on ARM and on x86-64. Neither `round`
/// nor a float-to-integer cast vectorizes with SSE2 only. Instead, adding 2^23
/// rounds the clamped value to an integer (ties to even), which the sum's
/// mantissa then holds; a tie rounded down is moved up.
#[inline(always)]
fn quantize_without_round(value: f32, scale: f32, max: u32) -> u32 {
    const TWO_23: f32 = 8_388_608.0;
    let clamped = (value * scale).max(0.0).min(max as f32);
    let shifted = clamped + TWO_23;
    let nearest = shifted.to_bits() - TWO_23.to_bits();
    nearest + u32::from(clamped - (shifted - TWO_23) == 0.5)
}

pub(crate) struct Axis {
    pub(crate) lo: usize,
    pub(crate) hi: usize,
    pub(crate) weight: f32,
}
pub(crate) fn axis(count: usize, samples: usize, sub: u32, origin: f32, filter: ChromaUpsampling) -> Result<Vec<Axis>> {
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(|_| HeifError::LimitExceeded("RGB: sampling allocation failed"))?;
    result.extend((0..count).map(|pixel| {
        // Use f64 for coordinates: f32 rounds large raster indices and
        // could otherwise round the last valid index up to `samples`.
        let position = ((pixel as f64 - f64::from(origin)) / f64::from(sub)).clamp(0.0, (samples - 1) as f64);
        if filter == ChromaUpsampling::Nearest {
            let index = position.round() as usize;
            Axis { lo: index, hi: index, weight: 0.0 }
        } else {
            let lo = position.floor() as usize;
            Axis { lo, hi: (lo + 1).min(samples - 1), weight: (position - lo as f64) as f32 }
        }
    }));
    Ok(result)
}

struct Matrix {
    identity: bool,
    y_offset: f32,
    y_scale: f32,
    c_offset: f32,
    c_scale: f32,
    red_cr: f32,
    blue_cb: f32,
    green_cb: f32,
    green_cr: f32,
}
impl Matrix {
    fn new(info: ColorInfo, y_depth: u32, c_depth: u32) -> Result<Self> {
        let (kr, kb) = match info.matrix_coefficients {
            0 => (0.0, 0.0),
            1 => (0.2126, 0.0722),
            4 => (0.30, 0.11),
            // Unspecified (2) as libheif reads it: BT.601.
            2 | 5 | 6 => (0.299, 0.114),
            7 => (0.212, 0.087),
            9 => (0.2627, 0.0593),
            _ => return Err(HeifError::Unsupported("RGB: unsupported matrix coefficients")),
        };
        let y_shift = (1u32 << (y_depth - 8)) as f32;
        let c_shift = (1u32 << (c_depth - 8)) as f32;
        let (y_offset, y_scale, c_offset, c_scale) = if info.full_range {
            (0.0, 1.0 / ((1u32 << y_depth) - 1) as f32, (1u32 << (c_depth - 1)) as f32, 1.0 / ((1u32 << c_depth) - 1) as f32)
        } else {
            (16.0 * y_shift, 1.0 / (219.0 * y_shift), 128.0 * c_shift, 1.0 / (224.0 * c_shift))
        };
        let identity = info.matrix_coefficients == 0;
        // Identity coding stores G, B, R in the nominal Y, Cb, Cr planes, so
        // Cb and Cr are scaled like luma, at their own depth.
        let (c_offset, c_scale) = match (identity, info.full_range) {
            (false, _) => (c_offset, c_scale),
            (true, true) => (0.0, 1.0 / ((1u32 << c_depth) - 1) as f32),
            (true, false) => (16.0 * c_shift, 1.0 / (219.0 * c_shift)),
        };
        let kg = 1.0 - kr - kb;
        Ok(Self {
            identity,
            y_offset,
            y_scale,
            c_offset,
            c_scale,
            red_cr: 2.0 * (1.0 - kr),
            blue_cb: 2.0 * (1.0 - kb),
            green_cb: -2.0 * kb * (1.0 - kb) / kg,
            green_cr: -2.0 * kr * (1.0 - kr) / kg,
        })
    }
    /// Converts one row from luma and chroma at full width. The identity
    /// test sits outside the loops so that each one vectorizes.
    fn row<T: RgbSample, const CHANNELS: usize>(&self, pixels: &mut [[T; CHANNELS]], luma: &[i32], cb: &[f32], cr: &[f32]) {
        if self.identity {
            for (((rgb, &y), &cb), &cr) in pixels.iter_mut().zip(luma).zip(cb).zip(cr) {
                rgb[0] = T::quantize((cr - self.c_offset) * self.c_scale);
                rgb[1] = T::quantize((y as f32 - self.y_offset) * self.y_scale);
                rgb[2] = T::quantize((cb - self.c_offset) * self.c_scale);
            }
        } else {
            for (((rgb, &y), &cb), &cr) in pixels.iter_mut().zip(luma).zip(cb).zip(cr) {
                let y = (y as f32 - self.y_offset) * self.y_scale;
                let cb = (cb - self.c_offset) * self.c_scale;
                let cr = (cr - self.c_offset) * self.c_scale;
                rgb[0] = T::quantize(y + self.red_cr * cr);
                rgb[1] = T::quantize(y + self.green_cb * cb + self.green_cr * cr);
                rgb[2] = T::quantize(y + self.blue_cb * cb);
            }
        }
    }
}

/// [`convert`]'s arithmetic for one pixel at a time, for decoders that
/// produce RGB without assembling the planes first. Each step is the same
/// `f32` expression, in the same order, so the result is bit-identical.
pub(crate) struct PixelRgb {
    matrix: Matrix,
    monochrome: bool,
}

impl PixelRgb {
    /// As [`convert`] sets up its matrix for this colour information,
    /// format and depths.
    pub(crate) fn new(info: ColorInfo, chroma_format_idc: u32, sampling: ChromaSampling, y_depth: u32, c_depth: u32) -> Result<Self> {
        let monochrome = chroma_format_idc == 0;
        let matrix = Matrix::new(if monochrome { ColorInfo { matrix_coefficients: 1, ..info } } else { info }, y_depth, c_depth)?;
        if matrix.identity && (sampling.sub_x != 1 || sampling.sub_y != 1) {
            return Err(HeifError::Unsupported("RGB: identity matrix requires 4:4:4"));
        }
        Ok(PixelRgb { matrix, monochrome })
    }

    /// `ChromaRows::slot`'s horizontal interpolation of one chroma row.
    #[inline(always)]
    pub(crate) fn lerp_samples(lo: i32, hi: i32, weight: f32) -> f32 {
        lo as f32 * (1.0 - weight) + hi as f32 * weight
    }

    /// `ChromaRows::blend`'s vertical interpolation of two such rows.
    #[inline(always)]
    pub(crate) fn lerp_rows(top: f32, bottom: f32, weight: f32) -> f32 {
        top * (1.0 - weight) + bottom * weight
    }

    /// One pixel, as `Matrix::row` (or the monochrome branch) writes it.
    #[inline(always)]
    pub(crate) fn rgb8(&self, y: i32, cb: f32, cr: f32) -> [u8; 3] {
        let m = &self.matrix;
        if self.monochrome {
            return [u8::quantize((y as f32 - m.y_offset) * m.y_scale); 3];
        }
        if m.identity {
            return [
                u8::quantize((cr - m.c_offset) * m.c_scale),
                u8::quantize((y as f32 - m.y_offset) * m.y_scale),
                u8::quantize((cb - m.c_offset) * m.c_scale),
            ];
        }
        let y = (y as f32 - m.y_offset) * m.y_scale;
        let cb = (cb - m.c_offset) * m.c_scale;
        let cr = (cr - m.c_offset) * m.c_scale;
        [u8::quantize(y + m.red_cr * cr), u8::quantize(y + m.green_cb * cb + m.green_cr * cr), u8::quantize(y + m.blue_cb * cb)]
    }

    pub(crate) fn monochrome(&self) -> bool {
        self.monochrome
    }

    /// A whole row through `Matrix::row` itself, the loop the planar
    /// converter vectorizes; not for monochrome (see [`PixelRgb::rgb8`]).
    pub(crate) fn row8(&self, pixels: &mut [[u8; 3]], luma: &[i32], cb: &[f32], cr: &[f32]) {
        let mut done = 0;
        #[cfg(target_arch = "x86_64")]
        if !self.matrix.identity && *AVX2 {
            if let Some(simd) = pulp::x86::V3::try_new() {
                let n = pixels.len().min(luma.len()).min(cb.len()).min(cr.len());
                done = simd.vectorize(Row8Avx2 { simd, m: &self.matrix, pixels: &mut pixels[..n], luma: &luma[..n], cb: &cb[..n], cr: &cr[..n] });
            }
        }
        if done < pixels.len() {
            let n = pixels.len().min(luma.len()).min(cb.len()).min(cr.len()).max(done);
            self.matrix.row::<u8, 3>(&mut pixels[done..n], &luma[done..n], &cb[done..n], &cr[done..n]);
        }
    }
}

/// A sample type the streaming converter holds rows in: `u8` for 8-bit
/// pictures (half the memory), `u16` for deeper ones.
pub(crate) trait BandSample: Copy + Default + Send + Sync + 'static {
    fn value(self) -> i32;
    /// Sixteen samples as `u16` lanes.
    #[cfg(target_arch = "x86_64")]
    fn lanes16(simd: pulp::x86::V3, s: &[Self; 16]) -> core::arch::x86_64::__m256i;
    /// Eight samples as `u16` lanes.
    #[cfg(target_arch = "x86_64")]
    fn lanes8(simd: pulp::x86::V3, s: &[Self; 8]) -> core::arch::x86_64::__m128i;
}

impl BandSample for u8 {
    #[inline(always)]
    fn value(self) -> i32 {
        i32::from(self)
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn lanes16(simd: pulp::x86::V3, s: &[u8; 16]) -> core::arch::x86_64::__m256i {
        simd.avx2._mm256_cvtepu8_epi16(pulp::bytemuck::cast(*s))
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn lanes8(simd: pulp::x86::V3, s: &[u8; 8]) -> core::arch::x86_64::__m128i {
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(s);
        simd.sse4_1._mm_cvtepu8_epi16(pulp::bytemuck::cast(b))
    }
}

impl BandSample for u16 {
    #[inline(always)]
    fn value(self) -> i32 {
        i32::from(self)
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn lanes16(_simd: pulp::x86::V3, s: &[u16; 16]) -> core::arch::x86_64::__m256i {
        pulp::bytemuck::cast(*s)
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn lanes8(_simd: pulp::x86::V3, s: &[u16; 8]) -> core::arch::x86_64::__m128i {
        pulp::bytemuck::cast(*s)
    }
}

/// Where a converted row goes: appended to a `Vec`, or written in place
/// into an output row.
pub(crate) trait RowOut {
    fn push(&mut self, pixel: [u8; 3]);
    fn push16(&mut self, block: &[[u8; 3]; 16]);
    /// The next `n` pixels, to be written by the caller.
    fn next(&mut self, n: usize) -> &mut [[u8; 3]];
}

impl RowOut for Vec<[u8; 3]> {
    #[inline(always)]
    fn push(&mut self, pixel: [u8; 3]) {
        Vec::push(self, pixel);
    }
    #[inline(always)]
    fn push16(&mut self, block: &[[u8; 3]; 16]) {
        self.extend_from_slice(block);
    }
    fn next(&mut self, n: usize) -> &mut [[u8; 3]] {
        let start = self.len();
        self.resize(start + n, [0; 3]);
        &mut self[start..]
    }
}

/// An output row written in place, from its start.
pub(crate) struct InPlace<'a> {
    pub(crate) row: &'a mut [[u8; 3]],
    pub(crate) at: usize,
}

impl RowOut for InPlace<'_> {
    #[inline(always)]
    fn push(&mut self, pixel: [u8; 3]) {
        if let Some(d) = self.row.get_mut(self.at) {
            *d = pixel;
        }
        self.at += 1;
    }
    #[inline(always)]
    fn push16(&mut self, block: &[[u8; 3]; 16]) {
        if let Some(d) = self.row.get_mut(self.at..self.at + 16) {
            d.copy_from_slice(block);
        }
        self.at += 16;
    }
    fn next(&mut self, n: usize) -> &mut [[u8; 3]] {
        let start = self.at.min(self.row.len());
        self.at += n;
        let end = self.at.min(self.row.len());
        &mut self.row[start..end]
    }
}

impl PixelRgb {
    /// A 4:2:0 row with nearest chroma straight from the samples: pixel `x`
    /// takes chroma sample `x / 2`. Appends whole blocks of 16 pixels to
    /// `out` with the AVX2 kernel and returns how many it did (0 without AVX2,
    /// for identity or monochrome); the caller converts the rest.
    pub(crate) fn row8_420_nearest<O: RowOut, S: BandSample>(&self, out: &mut O, luma: &[S], cb: &[S], cr: &[S]) -> usize {
        #[cfg(target_arch = "x86_64")]
        if !self.matrix.identity && !self.monochrome && *AVX2 {
            if let Some(simd) = pulp::x86::V3::try_new() {
                return simd.vectorize(Row420Avx2 { simd, m: &self.matrix, out, luma, cb, cr });
            }
        }
        let _ = (out, luma, cb, cr);
        0
    }
}

#[cfg(target_arch = "x86_64")]
struct Row420Avx2<'a, O, S> {
    simd: pulp::x86::V3,
    m: &'a Matrix,
    out: &'a mut O,
    luma: &'a [S],
    cb: &'a [S],
    cr: &'a [S],
}

#[cfg(target_arch = "x86_64")]
impl<O: RowOut, S: BandSample> pulp::NullaryFnOnce for Row420Avx2<'_, O, S> {
    type Output = usize;

    #[inline(always)]
    fn call(self) -> usize {
        use core::arch::x86_64::__m128i;
        use pulp::bytemuck::cast;
        let Row420Avx2 { simd, m, out, luma, cb, cr } = self;
        let avx = simd.avx;
        let k = Consts {
            y_offset: avx._mm256_set1_ps(m.y_offset), y_scale: avx._mm256_set1_ps(m.y_scale),
            c_offset: avx._mm256_set1_ps(m.c_offset), c_scale: avx._mm256_set1_ps(m.c_scale),
            red_cr: avx._mm256_set1_ps(m.red_cr), green_cb: avx._mm256_set1_ps(m.green_cb),
            green_cr: avx._mm256_set1_ps(m.green_cr), blue_cb: avx._mm256_set1_ps(m.blue_cb),
        };
        let masks: [__m128i; 9] = cast(INTERLEAVE);
        let (ys, _) = luma.as_chunks::<16>();
        let (cbs, _) = cb.as_chunks::<8>();
        let (crs, _) = cr.as_chunks::<8>();
        let mut done = 0;
        if *FAST_RGB {
            // Opt-in, libheif's precision: 8-bit fixed-point coefficients,
            // out = clip((ky * (Y - yo) + k * (C - co) + 128) >> 8). For full-range
            // video ky is 256 and this is libheif's Y + ((k * C' + 128) >> 8).
            let avx2 = simd.avx2;
            let f = |v: f32| (v * 255.0 * 256.0).round() as i32;
            let ky = avx._mm256_set1_epi32(f(m.y_scale));
            let (kr, kgb, kgr, kb) = (
                avx._mm256_set1_epi32(f(m.red_cr * m.c_scale)),
                avx._mm256_set1_epi32(f(m.green_cb * m.c_scale)),
                avx._mm256_set1_epi32(f(m.green_cr * m.c_scale)),
                avx._mm256_set1_epi32(f(m.blue_cb * m.c_scale)),
            );
            let (yo, co) = (avx._mm256_set1_epi32(m.y_offset.round() as i32), avx._mm256_set1_epi32(m.c_offset.round() as i32));
            let (rnd, low16) = (avx._mm256_set1_epi32(128), avx._mm256_set1_epi32(0xFFFF));
            for ((y, cb), cr) in ys.iter().zip(cbs).zip(crs) {
                let yv = S::lanes16(simd, y);
                let ye = avx2._mm256_mullo_epi32(avx2._mm256_sub_epi32(avx2._mm256_and_si256(yv, low16), yo), ky);
                let yo_ = avx2._mm256_mullo_epi32(avx2._mm256_sub_epi32(avx2._mm256_srli_epi32::<16>(yv), yo), ky);
                let cbi = avx2._mm256_sub_epi32(avx2._mm256_cvtepu16_epi32(S::lanes8(simd, cb)), co);
                let cri = avx2._mm256_sub_epi32(avx2._mm256_cvtepu16_epi32(S::lanes8(simd, cr)), co);
                let tr = avx2._mm256_add_epi32(avx2._mm256_mullo_epi32(kr, cri), rnd);
                let tg = avx2._mm256_add_epi32(avx2._mm256_add_epi32(avx2._mm256_mullo_epi32(kgb, cbi), avx2._mm256_mullo_epi32(kgr, cri)), rnd);
                let tb = avx2._mm256_add_epi32(avx2._mm256_mullo_epi32(kb, cbi), rnd);
                let r = to_bytes_even_odd(simd, avx2._mm256_srai_epi32::<8>(avx2._mm256_add_epi32(ye, tr)), avx2._mm256_srai_epi32::<8>(avx2._mm256_add_epi32(yo_, tr)));
                let g = to_bytes_even_odd(simd, avx2._mm256_srai_epi32::<8>(avx2._mm256_add_epi32(ye, tg)), avx2._mm256_srai_epi32::<8>(avx2._mm256_add_epi32(yo_, tg)));
                let b = to_bytes_even_odd(simd, avx2._mm256_srai_epi32::<8>(avx2._mm256_add_epi32(ye, tb)), avx2._mm256_srai_epi32::<8>(avx2._mm256_add_epi32(yo_, tb)));
                let block = cast::<[__m128i; 3], [[u8; 3]; 16]>([
                    interleave(simd, r, g, b, &masks, 0),
                    interleave(simd, r, g, b, &masks, 1),
                    interleave(simd, r, g, b, &masks, 2),
                ]);
                out.push16(&block);
                done += 16;
            }
            return done;
        }
        if *EVEN_ODD {
            // Even and odd pixels apart: each pair shares its chroma sample, so the
            // chroma terms are computed once, on eight lanes. Same f32 operations,
            // in the same order, for every pixel.
            let (avx2, low16) = (simd.avx2, avx._mm256_set1_epi32(0xFFFF));
            for ((y, cb), cr) in ys.iter().zip(cbs).zip(crs) {
                let yv = S::lanes16(simd, y);
                let scaled = |v| avx._mm256_mul_ps(avx._mm256_sub_ps(avx._mm256_cvtepi32_ps(v), k.y_offset), k.y_scale);
                let (ye, yo) = (scaled(avx2._mm256_and_si256(yv, low16)), scaled(avx2._mm256_srli_epi32::<16>(yv)));
                let chroma = |c: &[S; 8]| {
                    let c = avx._mm256_cvtepi32_ps(avx2._mm256_cvtepu16_epi32(S::lanes8(simd, c)));
                    avx._mm256_mul_ps(avx._mm256_sub_ps(c, k.c_offset), k.c_scale)
                };
                let (cbf, crf) = (chroma(cb), chroma(cr));
                let (tr, tgb, tgr, tb) = (
                    avx._mm256_mul_ps(k.red_cr, crf),
                    avx._mm256_mul_ps(k.green_cb, cbf),
                    avx._mm256_mul_ps(k.green_cr, crf),
                    avx._mm256_mul_ps(k.blue_cb, cbf),
                );
                let r = to_bytes_even_odd(simd, quantize8_lanes(simd, avx._mm256_add_ps(ye, tr)), quantize8_lanes(simd, avx._mm256_add_ps(yo, tr)));
                let g = to_bytes_even_odd(
                    simd,
                    quantize8_lanes(simd, avx._mm256_add_ps(avx._mm256_add_ps(ye, tgb), tgr)),
                    quantize8_lanes(simd, avx._mm256_add_ps(avx._mm256_add_ps(yo, tgb), tgr)),
                );
                let b = to_bytes_even_odd(simd, quantize8_lanes(simd, avx._mm256_add_ps(ye, tb)), quantize8_lanes(simd, avx._mm256_add_ps(yo, tb)));
                let block = cast::<[__m128i; 3], [[u8; 3]; 16]>([
                    interleave(simd, r, g, b, &masks, 0),
                    interleave(simd, r, g, b, &masks, 1),
                    interleave(simd, r, g, b, &masks, 2),
                ]);
                out.push16(&block);
                done += 16;
            }
            return done;
        }
        for ((y, cb), cr) in ys.iter().zip(cbs).zip(crs) {
            let [y0, y1]: [__m128i; 2] = cast(S::lanes16(simd, y));
            let (y0, y1) = (simd.avx2._mm256_cvtepu16_epi32(cast(y0)), simd.avx2._mm256_cvtepu16_epi32(cast(y1)));
            let (cb0, cb1) = doubled(simd, S::lanes8(simd, cb));
            let (cr0, cr1) = doubled(simd, S::lanes8(simd, cr));
            let lo = rgb_lanes(simd, &k, y0, cb0, cr0);
            let hi = rgb_lanes(simd, &k, y1, cb1, cr1);
            let (r, g, b) = (to_bytes(simd, lo[0], hi[0]), to_bytes(simd, lo[1], hi[1]), to_bytes(simd, lo[2], hi[2]));
            let block = cast::<[__m128i; 3], [[u8; 3]; 16]>([
                interleave(simd, r, g, b, &masks, 0),
                interleave(simd, r, g, b, &masks, 1),
                interleave(simd, r, g, b, &masks, 2),
            ]);
            out.push16(&block);
            done += 16;
        }
        done
    }
}

/// Even and odd pixels' 0..=255 lanes as 16 bytes in pixel order: the 32-bit
/// unpacks give pixels 0-3 and 8-11, 4-7 and 12-15, which the packs put in order.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn to_bytes_even_odd(simd: pulp::x86::V3, even: core::arch::x86_64::__m256i, odd: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m128i {
    let avx2 = simd.avx2;
    let p16 = avx2._mm256_packus_epi32(avx2._mm256_unpacklo_epi32(even, odd), avx2._mm256_unpackhi_epi32(even, odd));
    simd.sse2._mm_packus_epi16(simd.avx._mm256_castsi256_si128(p16), avx2._mm256_extracti128_si256::<1>(p16))
}

/// `HEIC_NO_EVEN_ODD=1`: the kernel with doubled chroma lanes (A/B runs).
#[cfg(target_arch = "x86_64")]
static EVEN_ODD: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("HEIC_NO_EVEN_ODD").as_deref() != Ok("1"));

/// Turned output: `dst[j][t(k)] = src[k][x0 + j]` for the eight rows `dst`
/// and the source rows `src`, eight at a time as 8x8 blocks of RGB pixels
/// (each widened to 32 bits by a byte shuffle, transposed, packed back);
/// `t(k)` is `k`, or `len - 1 - k` when `reversed`. Returns how many source
/// rows it did (whole blocks of eight); the caller does the rest. `false`
/// cases (no AVX2, rows too short) do nothing.
pub(crate) fn turn_rgb_8rows(dst: &mut [&mut [[u8; 3]]; 8], src: &[Vec<[u8; 3]>], x0: usize, reversed: bool) -> usize {
    #[cfg(target_arch = "x86_64")]
    if *AVX2 {
        if let Some(simd) = pulp::x86::V3::try_new() {
            return simd.vectorize(TurnRgb { simd, dst, src, x0, reversed });
        }
    }
    let _ = (dst, src, x0, reversed);
    0
}

#[cfg(target_arch = "x86_64")]
struct TurnRgb<'a, 'b> {
    simd: pulp::x86::V3,
    dst: &'a mut [&'b mut [[u8; 3]]; 8],
    src: &'a [Vec<[u8; 3]>],
    x0: usize,
    reversed: bool,
}

#[cfg(target_arch = "x86_64")]
impl pulp::NullaryFnOnce for TurnRgb<'_, '_> {
    type Output = usize;

    #[inline(always)]
    fn call(self) -> usize {
        use core::arch::x86_64::{__m128i, __m256i};
        use pulp::bytemuck::{cast, cast_slice};
        let TurnRgb { simd, dst, src, x0, reversed } = self;
        let (avx, avx2) = (simd.avx, simd.avx2);
        let len = src.len();
        if dst.iter().any(|d| d.len() < len) || src.iter().any(|r| r.len() < x0 + 8) {
            return 0;
        }
        // lane 0: pixels 0-3 from bytes 0..12; lane 1: pixels 4-7, at 4..16 of a load from byte 8
        let widen: __m256i = cast([0i8, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1, 4, 5, 6, -1, 7, 8, 9, -1, 10, 11, 12, -1, 13, 14, 15, -1]);
        let narrow: __m256i = cast([0i8, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1, 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1]);
        let flip = avx._mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        let mut k0 = 0;
        while k0 + 8 <= len {
            let mut r = [avx._mm256_setzero_si256(); 8];
            for (i, r) in r.iter_mut().enumerate() {
                let bytes: &[u8] = cast_slice(&src[k0 + i][x0..x0 + 8]);
                let lo: [u8; 16] = bytes[0..16].try_into().unwrap_or([0; 16]);
                let hi: [u8; 16] = bytes[8..24].try_into().unwrap_or([0; 16]);
                let both = avx._mm256_set_m128i(cast::<[u8; 16], __m128i>(hi), cast::<[u8; 16], __m128i>(lo));
                *r = avx2._mm256_shuffle_epi8(both, widen);
            }
            let t = [
                avx2._mm256_unpacklo_epi32(r[0], r[1]), avx2._mm256_unpackhi_epi32(r[0], r[1]),
                avx2._mm256_unpacklo_epi32(r[2], r[3]), avx2._mm256_unpackhi_epi32(r[2], r[3]),
                avx2._mm256_unpacklo_epi32(r[4], r[5]), avx2._mm256_unpackhi_epi32(r[4], r[5]),
                avx2._mm256_unpacklo_epi32(r[6], r[7]), avx2._mm256_unpackhi_epi32(r[6], r[7]),
            ];
            let u = [
                avx2._mm256_unpacklo_epi64(t[0], t[2]), avx2._mm256_unpackhi_epi64(t[0], t[2]),
                avx2._mm256_unpacklo_epi64(t[1], t[3]), avx2._mm256_unpackhi_epi64(t[1], t[3]),
                avx2._mm256_unpacklo_epi64(t[4], t[6]), avx2._mm256_unpackhi_epi64(t[4], t[6]),
                avx2._mm256_unpacklo_epi64(t[5], t[7]), avx2._mm256_unpackhi_epi64(t[5], t[7]),
            ];
            let cols = [
                avx2._mm256_permute2x128_si256::<0x20>(u[0], u[4]), avx2._mm256_permute2x128_si256::<0x20>(u[1], u[5]),
                avx2._mm256_permute2x128_si256::<0x20>(u[2], u[6]), avx2._mm256_permute2x128_si256::<0x20>(u[3], u[7]),
                avx2._mm256_permute2x128_si256::<0x31>(u[0], u[4]), avx2._mm256_permute2x128_si256::<0x31>(u[1], u[5]),
                avx2._mm256_permute2x128_si256::<0x31>(u[2], u[6]), avx2._mm256_permute2x128_si256::<0x31>(u[3], u[7]),
            ];
            let at = if reversed { len - k0 - 8 } else { k0 };
            for (j, col) in cols.into_iter().enumerate() {
                let col = if reversed { avx2._mm256_permutevar8x32_epi32(col, flip) } else { col };
                let packed: [u8; 32] = cast(avx2._mm256_shuffle_epi8(col, narrow));
                let out: &mut [u8] = pulp::bytemuck::cast_slice_mut(&mut dst[j][at..at + 8]);
                out[0..12].copy_from_slice(&packed[0..12]);
                out[12..24].copy_from_slice(&packed[16..28]);
            }
            k0 += 8;
        }
        k0
    }
}

/// Eight chroma samples, each repeated for two pixels, as two vectors of
/// eight `f32` lanes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn doubled(simd: pulp::x86::V3, c: core::arch::x86_64::__m128i) -> (core::arch::x86_64::__m256, core::arch::x86_64::__m256) {
    let (sse2, avx, avx2) = (simd.sse2, simd.avx, simd.avx2);
    let lo = avx2._mm256_cvtepu16_epi32(sse2._mm_unpacklo_epi16(c, c));
    let hi = avx2._mm256_cvtepu16_epi32(sse2._mm_unpackhi_epi16(c, c));
    (avx._mm256_cvtepi32_ps(lo), avx._mm256_cvtepi32_ps(hi))
}

/// The co-sited bilinear blend of chroma `h` (from its first column) to
/// `width` pixels, 16 at a time with AVX2, as `PixelRgb::lerp_rows` computes
/// it: pixel `2j` is `h[j] * 1 + h[j + 1] * 0`, pixel `2j + 1` is
/// `h[j] * 0.5 + h[j + 1] * 0.5`. Appends only blocks whose samples are all
/// inside `h` (none without AVX2); the caller does the rest.
pub(crate) fn blend_halves(out: &mut Vec<f32>, h: &[f32], width: usize) {
    #[cfg(target_arch = "x86_64")]
    if *AVX2 {
        if let Some(simd) = pulp::x86::V3::try_new() {
            simd.vectorize(BlendHalves { simd, out, h, width });
        }
    }
    let _ = (out, h, width);
}

#[cfg(target_arch = "x86_64")]
struct BlendHalves<'a> {
    simd: pulp::x86::V3,
    out: &'a mut Vec<f32>,
    h: &'a [f32],
    width: usize,
}

#[cfg(target_arch = "x86_64")]
impl pulp::NullaryFnOnce for BlendHalves<'_> {
    type Output = ();

    #[inline(always)]
    fn call(self) {
        use core::arch::x86_64::__m256;
        use pulp::bytemuck::cast;
        let BlendHalves { simd, out, h, width } = self;
        let avx = simd.avx;
        let (one, zero, half) = (avx._mm256_set1_ps(1.0), avx._mm256_set1_ps(0.0), avx._mm256_set1_ps(0.5));
        let mut j = 0;
        // block j..j+8 reads h[j..j+9] and writes pixels 2j..2j+16
        while 2 * j + 16 <= width && j + 9 <= h.len() {
            let a: [f32; 8] = h[j..j + 8].try_into().unwrap_or([0.0; 8]);
            let b: [f32; 8] = h[j + 1..j + 9].try_into().unwrap_or([0.0; 8]);
            let (a, b): (__m256, __m256) = (cast(a), cast(b));
            let even = avx._mm256_add_ps(avx._mm256_mul_ps(a, one), avx._mm256_mul_ps(b, zero));
            let odd = avx._mm256_add_ps(avx._mm256_mul_ps(a, half), avx._mm256_mul_ps(b, half));
            let (lo, hi) = (avx._mm256_unpacklo_ps(even, odd), avx._mm256_unpackhi_ps(even, odd));
            let first: [f32; 8] = cast(avx._mm256_permute2f128_ps::<0x20>(lo, hi));
            let second: [f32; 8] = cast(avx._mm256_permute2f128_ps::<0x31>(lo, hi));
            out.extend_from_slice(&first);
            out.extend_from_slice(&second);
            j += 8;
        }
    }
}

/// Whether the AVX2 row kernel runs; `HEIC_NO_SIMD=1` turns it off (for A/B runs).
#[cfg(target_arch = "x86_64")]
static AVX2: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("HEIC_NO_SIMD").as_deref() != Ok("1"));

/// Byte shuffles interleaving 16 R, G and B samples into 48 RGB bytes:
/// entry `3 * k + c` places channel `c` into output block `k`.
#[cfg(target_arch = "x86_64")]
const INTERLEAVE: [[i8; 16]; 9] = [
    [0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5],
    [-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1],
    [-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1],
    [-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1],
    [5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10],
    [-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1],
    [-1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1],
    [-1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1],
    [10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15],
];

/// `Matrix::row::<u8, 3>` (not identity), 16 pixels at a time in AVX2: the same
/// `f32` expressions in the same order, [`quantize_without_round`]'s exact
/// rounding, then packed to bytes and interleaved with byte shuffles. Returns
/// how many pixels it converted; the caller does the rest.
#[cfg(target_arch = "x86_64")]
struct Row8Avx2<'a> {
    simd: pulp::x86::V3,
    m: &'a Matrix,
    pixels: &'a mut [[u8; 3]],
    luma: &'a [i32],
    cb: &'a [f32],
    cr: &'a [f32],
}

#[cfg(target_arch = "x86_64")]
impl pulp::NullaryFnOnce for Row8Avx2<'_> {
    type Output = usize;

    #[inline(always)]
    fn call(self) -> usize {
        use core::arch::x86_64::__m128i;
        use pulp::bytemuck::cast;
        let Row8Avx2 { simd, m, pixels, luma, cb, cr } = self;
        let avx = simd.avx;
        let k = Consts {
            y_offset: avx._mm256_set1_ps(m.y_offset), y_scale: avx._mm256_set1_ps(m.y_scale),
            c_offset: avx._mm256_set1_ps(m.c_offset), c_scale: avx._mm256_set1_ps(m.c_scale),
            red_cr: avx._mm256_set1_ps(m.red_cr), green_cb: avx._mm256_set1_ps(m.green_cb),
            green_cr: avx._mm256_set1_ps(m.green_cr), blue_cb: avx._mm256_set1_ps(m.blue_cb),
        };
        let masks: [__m128i; 9] = cast(INTERLEAVE);
        let (px, _) = pixels.as_chunks_mut::<16>();
        let (ys, _) = luma.as_chunks::<8>();
        let (cbs, _) = cb.as_chunks::<8>();
        let (crs, _) = cr.as_chunks::<8>();
        let mut done = 0;
        for (i, out) in px.iter_mut().enumerate() {
            let (Some(y0), Some(y1), Some(cb0), Some(cb1), Some(cr0), Some(cr1)) =
                (ys.get(2 * i), ys.get(2 * i + 1), cbs.get(2 * i), cbs.get(2 * i + 1), crs.get(2 * i), crs.get(2 * i + 1))
            else {
                break;
            };
            let lo = rgb_lanes(simd, &k, cast(*y0), cast(*cb0), cast(*cr0));
            let hi = rgb_lanes(simd, &k, cast(*y1), cast(*cb1), cast(*cr1));
            let (r, g, b) = (to_bytes(simd, lo[0], hi[0]), to_bytes(simd, lo[1], hi[1]), to_bytes(simd, lo[2], hi[2]));
            *out = cast::<[__m128i; 3], [[u8; 3]; 16]>([
                interleave(simd, r, g, b, &masks, 0),
                interleave(simd, r, g, b, &masks, 1),
                interleave(simd, r, g, b, &masks, 2),
            ]);
            done += 16;
        }
        done
    }
}

/// Two vectors of eight 0..=255 lanes as 16 bytes, in order.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn to_bytes(simd: pulp::x86::V3, lo: core::arch::x86_64::__m256i, hi: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m128i {
    let p16 = simd.avx2._mm256_permute4x64_epi64::<0xD8>(simd.avx2._mm256_packus_epi32(lo, hi));
    simd.sse2._mm_packus_epi16(simd.avx._mm256_castsi256_si128(p16), simd.avx2._mm256_extracti128_si256::<1>(p16))
}

/// Output block `j` (16 of the 48 bytes) of 16 interleaved RGB pixels.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn interleave(
    simd: pulp::x86::V3, r: core::arch::x86_64::__m128i, g: core::arch::x86_64::__m128i, b: core::arch::x86_64::__m128i,
    masks: &[core::arch::x86_64::__m128i; 9], j: usize,
) -> core::arch::x86_64::__m128i {
    let (sse2, ssse3) = (simd.sse2, simd.ssse3);
    sse2._mm_or_si128(sse2._mm_or_si128(ssse3._mm_shuffle_epi8(r, masks[3 * j]), ssse3._mm_shuffle_epi8(g, masks[3 * j + 1])), ssse3._mm_shuffle_epi8(b, masks[3 * j + 2]))
}

#[cfg(target_arch = "x86_64")]
struct Consts {
    y_offset: core::arch::x86_64::__m256,
    y_scale: core::arch::x86_64::__m256,
    c_offset: core::arch::x86_64::__m256,
    c_scale: core::arch::x86_64::__m256,
    red_cr: core::arch::x86_64::__m256,
    green_cb: core::arch::x86_64::__m256,
    green_cr: core::arch::x86_64::__m256,
    blue_cb: core::arch::x86_64::__m256,
}

/// Eight pixels' R, G and B as 0..=255 in `i32` lanes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn rgb_lanes(
    simd: pulp::x86::V3, k: &Consts, y: core::arch::x86_64::__m256i, cb: core::arch::x86_64::__m256, cr: core::arch::x86_64::__m256,
) -> [core::arch::x86_64::__m256i; 3] {
    let avx = simd.avx;
    let y = avx._mm256_mul_ps(avx._mm256_sub_ps(avx._mm256_cvtepi32_ps(y), k.y_offset), k.y_scale);
    let cb = avx._mm256_mul_ps(avx._mm256_sub_ps(cb, k.c_offset), k.c_scale);
    let cr = avx._mm256_mul_ps(avx._mm256_sub_ps(cr, k.c_offset), k.c_scale);
    [
        quantize8_lanes(simd, avx._mm256_add_ps(y, avx._mm256_mul_ps(k.red_cr, cr))),
        quantize8_lanes(simd, avx._mm256_add_ps(avx._mm256_add_ps(y, avx._mm256_mul_ps(k.green_cb, cb)), avx._mm256_mul_ps(k.green_cr, cr))),
        quantize8_lanes(simd, avx._mm256_add_ps(y, avx._mm256_mul_ps(k.blue_cb, cb))),
    ]
}

/// [`quantize_without_round`] to 0..=255 in eight lanes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn quantize8_lanes(simd: pulp::x86::V3, v: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256i {
    const TWO_23: f32 = 8_388_608.0;
    let (avx, avx2) = (simd.avx, simd.avx2);
    let (zero, full, two_23, half) = (avx._mm256_set1_ps(0.0), avx._mm256_set1_ps(255.0), avx._mm256_set1_ps(TWO_23), avx._mm256_set1_ps(0.5));
    let clamped = avx._mm256_min_ps(avx._mm256_max_ps(avx._mm256_mul_ps(v, full), zero), full);
    let shifted = avx._mm256_add_ps(clamped, two_23);
    let nearest = avx2._mm256_sub_epi32(avx._mm256_castps_si256(shifted), avx._mm256_set1_epi32(TWO_23.to_bits() as i32));
    let tie = avx._mm256_castps_si256(avx._mm256_cmp_ps::<0>(avx._mm256_sub_ps(clamped, avx._mm256_sub_ps(shifted, two_23)), half));
    // a tie is all ones (-1): subtracting it adds one
    avx2._mm256_sub_epi32(nearest, tie)
}

/// An alpha sample as [`DecodedImage::to_rgba8`] scales it to 0..=255.
pub(crate) fn alpha8(sample: u32, max: u32) -> u8 {
    u8::alpha(sample, max)
}

/// Chroma rows interpolated horizontally to the output width, Cb and Cr,
/// keyed by chroma row. Each output row blends two of them, and neighbouring
/// output rows share theirs, so a band of rows interpolates each chroma row
/// once rather than once for every output row that reads it.
struct ChromaRows {
    keys: [Option<usize>; 3],
    cb: Vec<f32>,
    cr: Vec<f32>,
    /// One output row's chroma, blended from two of the rows above.
    blended: [Vec<f32>; 2],
}
impl ChromaRows {
    fn new(width: usize) -> Self {
        ChromaRows { keys: [None; 3], cb: vec![0.0; 3 * width], cr: vec![0.0; 3 * width], blended: [vec![0.0; width], vec![0.0; width]] }
    }
    /// The slot holding chroma row `row`, interpolating it into a slot other
    /// than `keep`'s if no slot holds it yet. `None` if a sample of the row
    /// exceeds `max`.
    fn slot(&mut self, row: usize, keep: usize, image: &DecodedImage, xs: &[Axis], max: u32) -> Option<usize> {
        if let Some(slot) = self.keys.iter().position(|&key| key == Some(row)) {
            return Some(slot);
        }
        let slot =
            (0..3).filter(|&slot| self.keys[slot] != Some(keep)).max_by_key(|&slot| self.keys[slot].map_or(usize::MAX, |key| key.abs_diff(row))).unwrap_or(0);
        let width = xs.len();
        for (plane, rows) in [(&image.cb, &mut self.cb), (&image.cr, &mut self.cr)] {
            let source = &plane[row * image.cb_width..][..image.cb_width];
            if !in_range(source, max) {
                return None;
            }
            for (o, ax) in rows[slot * width..][..width].iter_mut().zip(xs) {
                *o = source[ax.lo] as f32 * (1.0 - ax.weight) + source[ax.hi] as f32 * ax.weight;
            }
        }
        self.keys[slot] = Some(row);
        Some(slot)
    }
    /// Blends the rows in slots `top` and `bottom`: `top * (1 - weight) + bottom * weight`.
    fn blend(&mut self, top: usize, bottom: usize, weight: f32) -> (&[f32], &[f32]) {
        let width = self.blended[0].len();
        let [cb, cr] = &mut self.blended;
        for (rows, out) in [(&self.cb, &mut *cb), (&self.cr, &mut *cr)] {
            let (t, b) = (&rows[top * width..][..width], &rows[bottom * width..][..width]);
            for ((o, &t), &b) in out.iter_mut().zip(t).zip(b) {
                *o = t * (1.0 - weight) + b * weight;
            }
        }
        (cb, cr)
    }
}

/// Whether every sample lies in `0..=max`; written as a maximum, without an
/// early exit, so that it vectorizes.
fn in_range(samples: &[i32], max: u32) -> bool {
    samples.iter().map(|&v| v as u32).max().is_none_or(|v| v <= max)
}

/// Output rows per parallel task; a band reuses its interpolated chroma rows.
const ROWS_PER_BAND: usize = 16;

fn convert<T: RgbSample, const CHANNELS: usize>(image: &DecodedImage, options: &RgbOptions) -> Result<RgbImage<T>> {
    let _timer = crate::timing::start(crate::timing::Stage::Color);
    let bad = || HeifError::MalformedHevc("RGB: invalid plane geometry or sampling");
    let monochrome = image.chroma_format_idc == 0;
    if image.width == 0 || image.height == 0 || (!monochrome && (image.cb_width == 0 || image.cb_height == 0)) {
        return Err(bad());
    }
    let pixels = image.width.checked_mul(image.height).ok_or_else(bad)?;
    if pixels as u64 > MAX_IMAGE_PIXELS {
        return Err(HeifError::LimitExceeded("RGB: picture larger than MAX_IMAGE_PIXELS"));
    }
    let cpixels = image.cb_width.checked_mul(image.cb_height).ok_or_else(bad)?;
    let s = image.chroma_sampling;
    let valid_subsampling = match image.chroma_format_idc {
        0 => (s.sub_x, s.sub_y) == (1, 1),
        1 => (s.sub_x, s.sub_y) == (2, 2),
        2 => matches!((s.sub_x, s.sub_y), (2, 1) | (1, 2)),
        3 => (s.sub_x, s.sub_y) == (1, 1),
        _ => false,
    };
    let chroma_shape_valid = if monochrome {
        image.cb_width == 0 && image.cb_height == 0
    } else {
        image.cb_width == image.width.div_ceil(s.sub_x.max(1) as usize) && image.cb_height == image.height.div_ceil(s.sub_y.max(1) as usize)
    };
    if !matches!(s.sub_x, 1 | 2)
        || !matches!(s.sub_y, 1 | 2)
        || !s.origin_x.is_finite()
        || !s.origin_y.is_finite()
        || !valid_subsampling
        || s.origin_x.abs() > 2.0
        || s.origin_y.abs() > 2.0
        || !chroma_shape_valid
        || image.y.len() != pixels
        || image.cb.len() != cpixels
        || image.cr.len() != cpixels
        || !(8..=16).contains(&image.bit_depth_luma)
        || !(8..=16).contains(&image.bit_depth_chroma)
    {
        return Err(bad());
    }
    let info = options.color.or(image.color_info).unwrap_or_default();
    let matrix = Matrix::new(if monochrome { ColorInfo { matrix_coefficients: 1, ..info } } else { info }, image.bit_depth_luma, image.bit_depth_chroma)?;
    if matrix.identity && (s.sub_x != 1 || s.sub_y != 1) {
        return Err(HeifError::Unsupported("RGB: identity matrix requires 4:4:4"));
    }
    let xs = if monochrome { Vec::new() } else { axis(image.width, image.cb_width, s.sub_x, s.origin_x, options.upsampling)? };
    let ys = if monochrome { Vec::new() } else { axis(image.height, image.cb_height, s.sub_y, s.origin_y, options.upsampling)? };
    let length = pixels.checked_mul(CHANNELS).ok_or_else(bad)?;
    let mut data = Vec::new();
    data.try_reserve_exact(length).map_err(|_| HeifError::LimitExceeded("RGB: output allocation failed"))?;
    data.resize(length, T::default());
    // Samples are range-checked as the rows are read: luma row by row, and
    // each chroma row as it is interpolated. A chroma row no output row reads
    // is checked here.
    let out_of_range = || HeifError::MalformedHevc("RGB: sample outside coded range");
    let (luma_max, chroma_max) = ((1u32 << image.bit_depth_luma) - 1, (1u32 << image.bit_depth_chroma) - 1);
    let mut read = vec![false; if monochrome { 0 } else { image.cb_height }];
    for ay in &ys {
        read[ay.lo] = true;
        read[ay.hi] = true;
    }
    for (row, _) in read.iter().enumerate().filter(|(_, read)| !**read) {
        for plane in [&image.cb, &image.cr] {
            if !in_range(&plane[row * image.cb_width..][..image.cb_width], chroma_max) {
                return Err(out_of_range());
            }
        }
    }
    let width = image.width;
    let row_length = width * CHANNELS;
    // The arithmetic, and its order, is the same as interpolating chroma
    // pixel by pixel, so the output is too. Alpha comes last.
    data.par_chunks_mut(row_length * ROWS_PER_BAND).enumerate().try_for_each_init(
        || ChromaRows::new(if monochrome { 0 } else { width }),
        |chroma, (band, rows)| {
            for (i, output) in rows.chunks_mut(row_length).enumerate() {
                let row = band * ROWS_PER_BAND + i;
                let pixels = output.as_chunks_mut::<CHANNELS>().0;
                let luma = &image.y[row * width..][..width];
                if !in_range(luma, luma_max) {
                    return Err(out_of_range());
                }
                if monochrome {
                    for (rgb, &y) in pixels.iter_mut().zip(luma) {
                        let value = T::quantize((y as f32 - matrix.y_offset) * matrix.y_scale);
                        rgb[..3].fill(value);
                    }
                } else {
                    let ay = &ys[row];
                    let top = chroma.slot(ay.lo, ay.hi, image, &xs, chroma_max).ok_or_else(out_of_range)?;
                    let bottom = chroma.slot(ay.hi, ay.lo, image, &xs, chroma_max).ok_or_else(out_of_range)?;
                    let (cb, cr) = chroma.blend(top, bottom, ay.weight);
                    matrix.row(pixels, luma, cb, cr);
                }
                if CHANNELS == 4 {
                    match &image.alpha {
                        Some(alpha) => {
                            let max = (1u32 << alpha.bit_depth) - 1;
                            for (rgb, &a) in pixels.iter_mut().zip(&alpha.samples[row * width..][..width]) {
                                rgb[3] = T::alpha(a as u32, max);
                            }
                        }
                        None => pixels.iter_mut().for_each(|rgb| rgb[3] = T::alpha(1, 1)),
                    }
                }
            }
            Ok(())
        },
    )?;
    Ok(RgbImage { width: image.width, height: image.height, data })
}

fn convert_rgba<T: RgbSample>(image: &DecodedImage, options: &RgbOptions) -> Result<RgbaImage<T>> {
    if let Some(alpha) = &image.alpha {
        let pixels = image.width.checked_mul(image.height).ok_or(HeifError::MalformedHevc("RGBA: dimensions overflow"))?;
        if !(8..=16).contains(&alpha.bit_depth) || alpha.samples.len() != pixels {
            return Err(HeifError::MalformedHevc("RGBA: invalid alpha geometry or bit depth"));
        }
        let max = (1i32 << alpha.bit_depth) - 1;
        if alpha.samples.iter().any(|&v| v < 0 || v > max) {
            return Err(HeifError::MalformedHevc("RGBA: alpha outside coded range"));
        }
    }
    let rgb = convert::<T, 4>(image, options)?;
    Ok(RgbaImage { width: rgb.width, height: rgb.height, data: rgb.data, premultiplied: image.alpha.as_ref().is_some_and(|alpha| alpha.premultiplied) })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rgba_uses_full_range_alpha_and_preserves_opaque_rgb() {
        let mut image = planar(vec![16, 128, 235], vec![128; 3], vec![128; 3], 8, info(1, false));
        let opaque = image.to_rgba16().unwrap();
        assert!(opaque.data.as_chunks::<4>().0.iter().all(|pixel| pixel[3] == 65535));
        image.alpha = Some(crate::AlphaPlane { bit_depth: 10, samples: vec![0, 513, 1023], premultiplied: true });
        let rgba = image.to_rgba16().unwrap();
        assert_eq!(rgba.data.as_chunks::<4>().0.iter().map(|p| p[3]).collect::<Vec<_>>(), [0, 32864, 65535]);
        assert!(rgba.premultiplied);
        assert_eq!(rgba.data.as_chunks::<4>().0.iter().flat_map(|p| p[..3].iter().copied()).collect::<Vec<_>>(), image.to_rgb16().unwrap().data);
        image.alpha.as_mut().unwrap().bit_depth = 16;
        image.alpha.as_mut().unwrap().samples = vec![0, 32769, 65535];
        assert_eq!(image.to_rgba16().unwrap().data.as_chunks::<4>().0.iter().map(|p| p[3]).collect::<Vec<_>>(), [0, 32769, 65535]);
    }

    #[test]
    fn invalid_alpha_is_rejected_before_rgba_conversion() {
        let mut image = planar(vec![16], vec![128], vec![128], 8, info(1, false));
        for (depth, samples) in [(0, vec![0]), (17, vec![0]), (8, vec![]), (8, vec![-1]), (8, vec![256])] {
            image.alpha = Some(crate::AlphaPlane { bit_depth: depth, samples, premultiplied: false });
            assert!(image.to_rgba8().is_err());
            assert!(image.to_rgba16().is_err());
            assert!(image.to_rgb8().is_ok(), "explicit RGB conversion omits alpha");
        }
    }
    fn planar(y: Vec<i32>, cb: Vec<i32>, cr: Vec<i32>, depth: u32, color: ColorInfo) -> DecodedImage {
        DecodedImage {
            alpha: None,
            width: y.len(),
            height: 1,
            cb_width: cb.len(),
            cb_height: 1,
            bit_depth_luma: depth,
            bit_depth_chroma: depth,
            chroma_format_idc: 3,
            color_info: Some(color),
            chroma_sampling: ChromaSampling::coded(3, None),
            y,
            cb,
            cr,
            metadata: Default::default(),
        }
    }
    fn info(matrix_coefficients: u16, full_range: bool) -> ColorInfo {
        ColorInfo { matrix_coefficients, full_range, ..Default::default() }
    }
    #[test]
    fn limited_and_full_range_black_white_at_every_depth() {
        for depth in [8, 10, 12, 16] {
            let shift = 1 << (depth - 8);
            let mid = 1 << (depth - 1);
            for full in [false, true] {
                let (black, white) = if full { (0, (1 << depth) - 1) } else { (16 * shift, 235 * shift) };
                let image = planar(vec![black, white], vec![mid; 2], vec![mid; 2], depth, info(1, full));
                assert_eq!(image.to_rgb8().unwrap().data, [0, 0, 0, 255, 255, 255], "depth={depth}, full={full}");
                assert_eq!(image.to_rgb16().unwrap().data, [0, 0, 0, 65535, 65535, 65535]);
            }
        }
    }
    #[test]
    fn matrix_codes_produce_distinct_known_colour_vectors() {
        for (matrix, want) in [(5, [255, 22, 21]), (6, [255, 22, 21]), (1, [255, 46, 18]), (9, [255, 32, 16])] {
            let image = planar(vec![100], vec![90], vec![240], 8, info(matrix, false));
            assert_eq!(image.to_rgb8().unwrap().data, want, "matrix={matrix}");
        }
    }
    #[test]
    fn identity_matrix_has_gbr_order_and_independent_channel_depths() {
        let mut image = planar(vec![128], vec![256], vec![512], 8, info(0, true));
        image.bit_depth_chroma = 10;
        assert_eq!(image.to_rgb8().unwrap().data, [128, 128, 64]);
        let image = planar(vec![16], vec![235], vec![126], 8, info(0, false));
        assert_eq!(image.to_rgb8().unwrap().data, [128, 0, 255]);
    }
    #[test]
    fn sixteen_bit_conversion_preserves_ten_bit_precision() {
        let image = planar(vec![65], vec![321], vec![1000], 10, info(0, true));
        let rgb = image.to_rgb16().unwrap();
        let expected: Vec<_> = [1000u32, 65, 321].iter().map(|v| ((v * 65535 + 511) / 1023) as u16).collect();
        assert_eq!(rgb.data, expected);
        assert!(rgb.data.iter().any(|v| v % 257 != 0));
    }
    #[test]
    fn unlabelled_colour_defaults_and_explicit_override_are_separate() {
        let mut image = planar(vec![100], vec![90], vec![240], 8, info(6, false));
        let options = RgbOptions { color: Some(info(1, false)), ..Default::default() };
        assert_eq!(image.to_rgb8_with_options(&options).unwrap().data, [255, 46, 18]);
        assert_eq!(image.to_rgb8().unwrap().data, [255, 22, 21]);
        // Unlabelled and unspecified both read as BT.601, as in libheif.
        image.color_info = None;
        assert_eq!(image.to_rgb8().unwrap().data, [255, 22, 21]);
        image.color_info = Some(info(2, false));
        assert_eq!(image.to_rgb8().unwrap().data, [255, 22, 21]);
    }
    #[test]
    fn bilinear_upsampling_uses_chroma_locations_and_nearest_is_optional() {
        let mut image = planar(vec![64; 4], vec![128, 160], vec![128; 2], 8, info(1, true));
        image.chroma_format_idc = 2;
        image.chroma_sampling = ChromaSampling::coded(2, None);
        let blue: Vec<_> = image.to_rgb8().unwrap().data.as_chunks::<3>().0.iter().map(|rgb| rgb[2]).collect();
        assert_eq!(blue, [64, 94, 123, 123]);
        image.chroma_sampling.origin_x = 0.5;
        let blue: Vec<_> = image.to_rgb8().unwrap().data.as_chunks::<3>().0.iter().map(|rgb| rgb[2]).collect();
        assert_eq!(blue, [64, 79, 109, 123]);
        let options = RgbOptions { upsampling: ChromaUpsampling::Nearest, ..Default::default() };
        let blue: Vec<_> = image.to_rgb8_with_options(&options).unwrap().data.as_chunks::<3>().0.iter().map(|rgb| rgb[2]).collect();
        assert_eq!(blue, [64, 64, 123, 123]);
    }
    #[test]
    fn rotated_and_mirrored_422_converts_like_transformed_rgb() {
        let image = || DecodedImage {
            alpha: None,
            width: 4,
            height: 3,
            cb_width: 2,
            cb_height: 3,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            chroma_format_idc: 2,
            color_info: Some(info(1, true)),
            chroma_sampling: ChromaSampling::coded(2, None),
            y: vec![100; 12],
            cb: vec![128, 150, 140, 160, 160, 180],
            cr: vec![128; 6],
            metadata: Default::default(),
        };
        let source = image().to_rgb8().unwrap();
        for transform in [crate::isobmff::Transform::Rotate90Ccw(1), crate::isobmff::Transform::MirrorHorizontal, crate::isobmff::Transform::MirrorVertical] {
            let transformed = crate::apply_transforms(image(), &[transform]).unwrap();
            let rgb = transformed.to_rgb8().unwrap();
            for y in 0..rgb.height {
                for x in 0..rgb.width {
                    let (sx, sy) = match transform {
                        crate::isobmff::Transform::Rotate90Ccw(_) => (3 - y, x),
                        crate::isobmff::Transform::MirrorHorizontal => (3 - x, y),
                        _ => (x, 2 - y),
                    };
                    assert_eq!(
                        &rgb.data[(y * rgb.width + x) * 3..(y * rgb.width + x + 1) * 3],
                        &source.data[(sy * 4 + sx) * 3..(sy * 4 + sx + 1) * 3],
                        "{transform:?}, ({x},{y})"
                    );
                }
            }
        }
    }
    #[test]
    fn odd_crop_tracks_chroma_phase_interior_samples() {
        let image = DecodedImage {
            alpha: None,
            width: 8,
            height: 2,
            cb_width: 4,
            cb_height: 2,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            chroma_format_idc: 2,
            color_info: Some(info(1, true)),
            chroma_sampling: ChromaSampling::coded(2, None),
            y: vec![100; 16],
            cb: vec![128, 140, 160, 180, 128, 140, 160, 180],
            cr: vec![128; 8],
            metadata: Default::default(),
        };
        let before = image.to_rgb8().unwrap();
        let window = crate::ImageWindow { left: 1, top: 0, width: 6, height: 2 };
        let cropped = crate::extract_window(image, window, 2, 1).unwrap();
        assert_eq!(cropped.chroma_sampling.origin_x, -1.0);
        let rgb = cropped.to_rgb8().unwrap();
        // Samples outside the returned native crop are edge-extended; the
        // interior must still use the original chroma positions.
        assert_eq!(&rgb.data[..12], &before.data[3..15]);
    }
    #[test]
    fn invalid_shapes_ranges_sampling_and_matrices_return_errors() {
        let original = || planar(vec![100], vec![128], vec![128], 8, info(1, false));
        let mut image = original();
        image.cb.clear();
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.height = usize::MAX;
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.bit_depth_luma = 32;
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.y[0] = -1;
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.cb[0] = 256;
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.chroma_sampling.sub_x = 0;
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.chroma_sampling.origin_y = f32::NAN;
        assert!(image.to_rgb8().is_err());
        let mut image = original();
        image.color_info = Some(info(10, false));
        assert!(matches!(image.to_rgb8(), Err(HeifError::Unsupported(_))));
        let mut image = original();
        image.chroma_format_idc = 2;
        image.chroma_sampling = ChromaSampling::coded(2, None);
        image.color_info = Some(info(0, true));
        assert!(image.to_rgb8().is_err());
    }
    #[test]
    fn quantize_matches_rounding_at_the_edges() {
        // Every f32 was compared once; these are the cases that could differ.
        let below_half = f32::from_bits(0.5f32.to_bits() - 1);
        for value in [
            0.0,
            -0.0,
            0.5,
            below_half,
            1.5,
            2.5,
            254.5,
            255.0,
            255.49998,
            255.5,
            -0.5,
            -0.49999997,
            1e10,
            -1e10,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MAX,
            f32::MIN_POSITIVE,
            16_777_217.0,
            2_147_483_648.0,
        ] {
            for scale in [1.0f32, 1.0 / 255.0, 1.0 / 65535.0] {
                let v = value * scale;
                assert_eq!(quantize_without_round(v, 255.0, 255), u32::from((v * 255.0).round() as u8), "{value} * {scale} to u8");
                assert_eq!(quantize_without_round(v, 65535.0, 65535), u32::from((v * 65535.0).round() as u16), "{value} * {scale} to u16");
            }
        }
    }
    #[test]
    fn a_chroma_row_no_pixel_reads_is_still_range_checked() {
        // With the chroma origin two rows up and nearest sampling, every
        // output row reads chroma row 1, none reads row 0.
        let image = |cb0: i32| DecodedImage {
            width: 2,
            height: 4,
            cb_width: 1,
            cb_height: 2,
            chroma_format_idc: 1,
            chroma_sampling: ChromaSampling { sub_x: 2, sub_y: 2, origin_x: 0.0, origin_y: -2.0 },
            y: vec![100; 8],
            cb: vec![cb0, 128],
            cr: vec![128; 2],
            ..planar(vec![100], vec![128], vec![128], 8, info(1, false))
        };
        let nearest = RgbOptions { upsampling: ChromaUpsampling::Nearest, ..Default::default() };
        assert!(image(255).to_rgb8_with_options(&nearest).is_ok());
        assert!(image(256).to_rgb8_with_options(&nearest).is_err());
        assert!(image(-1).to_rgb8_with_options(&nearest).is_err());
    }
}

/// PoC switch: `HEIC_FAST_RGB=1` converts nearest 4:2:0 rows with libheif's
/// 8-bit fixed-point precision (not bit-identical to the exact f32 formula).
#[cfg(target_arch = "x86_64")]
static FAST_RGB: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("HEIC_FAST_RGB").as_deref() == Ok("1"));
