//! Dequantization (clause 8.6.3) and inverse transform (clause 8.6.4):
//! the 32-point integer DCT-II basis (smaller sizes are subsampled from
//! it, per the spec's transform-nesting property) and the 4x4 DST-VII
//! used for luma intra 4x4 blocks.
//!
//! The DCT runs as partial butterflies (as in HM), on the box of a
//! block's nonzero coefficients only; the tests check it against the
//! direct matrix products, bit for bit.
//!
//! The matrices and two-pass shift/rounding structure here were checked
//! against `libde265`'s `fallback-dct.cc` (LGPL-3.0) rather than typed
//! from memory, given how easy a single wrong entry in a 32x32 table
//! would be to miss. See this crate's `README.md`.

use crate::error::{HeifError, Result};

/// `levelScale[qP % 6]`, clause 8.6.3.
const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

/// Table 8-10's irregular middle section (`qPi` 30..=42) of the
/// `qPi` → `QpC` chroma QP mapping (clause 8.6.1), checked against
/// `libde265`/`imazen-heic` rather than typed from memory.
const CHROMA_QP_TABLE: [i32; 13] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37];

/// `qPi` → `QpC` (clause 8.6.1): identity below 30, `Table 8-10` in the
/// middle, `qPi - 6` above 43. Only valid for `ChromaArrayType == 1`
/// (4:2:0); 4:2:2/4:4:4 instead use `min(qPi, 51)` directly at the call
/// site.
pub fn chroma_qp_mapping(qp_i: i32) -> i32 {
    if qp_i < 30 {
        qp_i
    } else if qp_i >= 43 {
        qp_i - 6
    } else {
        CHROMA_QP_TABLE.get((qp_i - 30) as usize).copied().unwrap_or(qp_i)
    }
}

/// Dequantizes one coefficient with a flat scaling list (`m = 16`).
/// `qp` is `Qp'Y`/`Qp'Cb`/`Qp'Cr` (already offset-applied).
pub fn dequant(level: i32, qp: i32, bit_depth: u32, log2_size: u32) -> Result<i32> {
    dequant_with_scale(level, qp, bit_depth, log2_size, 16)
}

/// Dequantization with the selected scaling-matrix coefficient (8.6.3).
/// `qp` includes the component's bit-depth offset, i.e. Qp' rather than Qp.
pub fn dequant_with_scale(level: i32, qp: i32, bit_depth: u32, log2_size: u32, weight: i32) -> Result<i32> {
    Dequant::new(qp, bit_depth, log2_size)?.apply(level, weight)
}

/// The dequantization (8.6.3) of one transform block, with its parameters
/// checked once rather than per coefficient.
#[derive(Debug, Clone, Copy)]
pub struct Dequant {
    fact: i64,
    shift: u32,
}

impl Dequant {
    /// `qp` includes the component's bit-depth offset, i.e. Qp' rather than Qp.
    pub fn new(qp: i32, bit_depth: u32, log2_size: u32) -> Result<Self> {
        if !(8..=16).contains(&bit_depth) || !(2..=5).contains(&log2_size) || !(0..=51 + 6 * (bit_depth as i32 - 8)).contains(&qp) {
            return Err(HeifError::MalformedHevc("dequant: parameter out of range"));
        }
        Ok(Dequant { fact: LEVEL_SCALE[(qp % 6) as usize] << (qp / 6), shift: bit_depth + log2_size - 5 })
    }

    /// One coefficient, with the selected scaling-matrix entry `weight`.
    #[inline]
    pub fn apply(self, level: i32, weight: i32) -> Result<i32> {
        if !(0..=255).contains(&weight) {
            return Err(HeifError::MalformedHevc("dequant: parameter out of range"));
        }
        Ok(self.apply_checked(level, weight))
    }

    /// [`Dequant::apply`] with a weight already known to be in 0..=255
    /// (a [`super::scaling::ScalingWeights`] entry, or the flat 16).
    #[inline(always)]
    pub fn apply_checked(self, level: i32, weight: i32) -> i32 {
        let v = (i64::from(level) * i64::from(weight) * self.fact + (1i64 << (self.shift - 1))) >> self.shift;
        v.clamp(-32768, 32767) as i32
    }
}

/// Row 0 and column 0 of the 32-point DCT-II integer basis repeat across
/// every even subsample, by construction (DCT-II nesting): `mat[0][*] ==
/// 64` for every valid `fact`. Smaller transforms read `MAT_DCT[fact*j][i]`
/// with `fact = 32 / nT`.
#[rustfmt::skip]
pub(crate) const MAT_DCT: [[i32; 32]; 32] = [
    [64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64],
    [90,90,88,85,82,78,73,67,61,54,46,38,31,22,13,4,-4,-13,-22,-31,-38,-46,-54,-61,-67,-73,-78,-82,-85,-88,-90,-90],
    [90,87,80,70,57,43,25,9,-9,-25,-43,-57,-70,-80,-87,-90,-90,-87,-80,-70,-57,-43,-25,-9,9,25,43,57,70,80,87,90],
    [90,82,67,46,22,-4,-31,-54,-73,-85,-90,-88,-78,-61,-38,-13,13,38,61,78,88,90,85,73,54,31,4,-22,-46,-67,-82,-90],
    [89,75,50,18,-18,-50,-75,-89,-89,-75,-50,-18,18,50,75,89,89,75,50,18,-18,-50,-75,-89,-89,-75,-50,-18,18,50,75,89],
    [88,67,31,-13,-54,-82,-90,-78,-46,-4,38,73,90,85,61,22,-22,-61,-85,-90,-73,-38,4,46,78,90,82,54,13,-31,-67,-88],
    [87,57,9,-43,-80,-90,-70,-25,25,70,90,80,43,-9,-57,-87,-87,-57,-9,43,80,90,70,25,-25,-70,-90,-80,-43,9,57,87],
    [85,46,-13,-67,-90,-73,-22,38,82,88,54,-4,-61,-90,-78,-31,31,78,90,61,4,-54,-88,-82,-38,22,73,90,67,13,-46,-85],
    [83,36,-36,-83,-83,-36,36,83,83,36,-36,-83,-83,-36,36,83,83,36,-36,-83,-83,-36,36,83,83,36,-36,-83,-83,-36,36,83],
    [82,22,-54,-90,-61,13,78,85,31,-46,-90,-67,4,73,88,38,-38,-88,-73,-4,67,90,46,-31,-85,-78,-13,61,90,54,-22,-82],
    [80,9,-70,-87,-25,57,90,43,-43,-90,-57,25,87,70,-9,-80,-80,-9,70,87,25,-57,-90,-43,43,90,57,-25,-87,-70,9,80],
    [78,-4,-82,-73,13,85,67,-22,-88,-61,31,90,54,-38,-90,-46,46,90,38,-54,-90,-31,61,88,22,-67,-85,-13,73,82,4,-78],
    [75,-18,-89,-50,50,89,18,-75,-75,18,89,50,-50,-89,-18,75,75,-18,-89,-50,50,89,18,-75,-75,18,89,50,-50,-89,-18,75],
    [73,-31,-90,-22,78,67,-38,-90,-13,82,61,-46,-88,-4,85,54,-54,-85,4,88,46,-61,-82,13,90,38,-67,-78,22,90,31,-73],
    [70,-43,-87,9,90,25,-80,-57,57,80,-25,-90,-9,87,43,-70,-70,43,87,-9,-90,-25,80,57,-57,-80,25,90,9,-87,-43,70],
    [67,-54,-78,38,85,-22,-90,4,90,13,-88,-31,82,46,-73,-61,61,73,-46,-82,31,88,-13,-90,-4,90,22,-85,-38,78,54,-67],
    [64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64,64,-64,-64,64],
    [61,-73,-46,82,31,-88,-13,90,-4,-90,22,85,-38,-78,54,67,-67,-54,78,38,-85,-22,90,4,-90,13,88,-31,-82,46,73,-61],
    [57,-80,-25,90,-9,-87,43,70,-70,-43,87,9,-90,25,80,-57,-57,80,25,-90,9,87,-43,-70,70,43,-87,-9,90,-25,-80,57],
    [54,-85,-4,88,-46,-61,82,13,-90,38,67,-78,-22,90,-31,-73,73,31,-90,22,78,-67,-38,90,-13,-82,61,46,-88,4,85,-54],
    [50,-89,18,75,-75,-18,89,-50,-50,89,-18,-75,75,18,-89,50,50,-89,18,75,-75,-18,89,-50,-50,89,-18,-75,75,18,-89,50],
    [46,-90,38,54,-90,31,61,-88,22,67,-85,13,73,-82,4,78,-78,-4,82,-73,-13,85,-67,-22,88,-61,-31,90,-54,-38,90,-46],
    [43,-90,57,25,-87,70,9,-80,80,-9,-70,87,-25,-57,90,-43,-43,90,-57,-25,87,-70,-9,80,-80,9,70,-87,25,57,-90,43],
    [38,-88,73,-4,-67,90,-46,-31,85,-78,13,61,-90,54,22,-82,82,-22,-54,90,-61,-13,78,-85,31,46,-90,67,4,-73,88,-38],
    [36,-83,83,-36,-36,83,-83,36,36,-83,83,-36,-36,83,-83,36,36,-83,83,-36,-36,83,-83,36,36,-83,83,-36,-36,83,-83,36],
    [31,-78,90,-61,4,54,-88,82,-38,-22,73,-90,67,-13,-46,85,-85,46,13,-67,90,-73,22,38,-82,88,-54,-4,61,-90,78,-31],
    [25,-70,90,-80,43,9,-57,87,-87,57,-9,-43,80,-90,70,-25,-25,70,-90,80,-43,-9,57,-87,87,-57,9,43,-80,90,-70,25],
    [22,-61,85,-90,73,-38,-4,46,-78,90,-82,54,-13,-31,67,-88,88,-67,31,13,-54,82,-90,78,-46,4,38,-73,90,-85,61,-22],
    [18,-50,75,-89,89,-75,50,-18,-18,50,-75,89,-89,75,-50,18,18,-50,75,-89,89,-75,50,-18,-18,50,-75,89,-89,75,-50,18],
    [13,-38,61,-78,88,-90,85,-73,54,-31,4,22,-46,67,-82,90,-90,82,-67,46,-22,-4,31,-54,73,-85,90,-88,78,-61,38,-13],
    [9,-25,43,-57,70,-80,87,-90,90,-87,80,-70,57,-43,25,-9,-9,25,-43,57,-70,80,-87,90,-90,87,-80,70,-57,43,-25,9],
    [4,-13,22,-31,38,-46,54,-61,67,-73,78,-82,85,-88,90,-90,90,-90,88,-85,82,-78,73,-67,61,-54,46,-38,31,-22,13,-4],
];

/// Residual for a transform-skipped block (clause 8.6.4.2): the
/// dequantized coefficients stand directly for the spatial residual,
/// scaled by a fixed shift instead of the NxN inverse transform.
/// Cross-checked against `libde265::fallback-dct.cc`'s
/// `transform_skip_residual_fallback` rather than typed from memory.
pub fn transform_skip_residual(coeffs: &[i32], nt: usize, bit_depth: u32, out: &mut [i32]) -> Result<()> {
    if coeffs.len() != nt * nt || out.len() != nt * nt {
        return Err(HeifError::MalformedHevc("transform_skip_residual: coefficient buffer size mismatch"));
    }
    let ts_shift = 5 + log2_usize(nt) as i64;
    let bd_shift = (20i64 - bit_depth as i64).max(0);
    let rnd: i64 = if bd_shift > 0 { 1 << (bd_shift - 1) } else { 0 };
    for (o, &c) in out.iter_mut().zip(coeffs) {
        let scaled = (i64::from(c)) << ts_shift;
        *o = i32::try_from((scaled + rnd) >> bd_shift).map_err(|_| HeifError::MalformedHevc("transform_skip_residual: overflow"))?;
    }
    Ok(())
}

fn log2_usize(n: usize) -> u32 {
    n.trailing_zeros()
}

/// The odd basis functions `1, 3, 5, ...` of the `2 * H`-point DCT-II over
/// its first `H` samples (the rest follow by antisymmetry), as `i16` so
/// that products with 16-bit coefficients can use 16-bit multiplies.
const fn odd_basis<const H: usize>() -> [[i16; H]; H] {
    let mut m = [[0; H]; H];
    let mut t = 0;
    while t < H {
        let mut i = 0;
        while i < H {
            m[t][i] = MAT_DCT[16 / H * (2 * t + 1)][i] as i16;
            i += 1;
        }
        t += 1;
    }
    m
}

const ODD_16: [[i16; 8]; 8] = odd_basis();
const ODD_32: [[i16; 16]; 16] = odd_basis();

/// `Clip3(coeffMin, coeffMax, v)` with the 16-bit coefficient range (clause 8.6.4.2).
fn clip_coeff(v: i32) -> i32 {
    v.clamp(-32768, 32767)
}

/// Where a transform block's nonzero coefficients can be: every
/// coefficient at column `x >= width` or row `y >= height` is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoeffBox {
    pub width: usize,
    pub height: usize,
}

impl CoeffBox {
    /// The whole `nt x nt` block.
    pub fn full(nt: usize) -> Self {
        CoeffBox { width: nt, height: nt }
    }

    /// Only the DC coefficient can be nonzero.
    pub fn dc_only(self) -> bool {
        self.width <= 1 && self.height <= 1
    }
}

/// The columns and rows of an `nt x nt` block that
/// [`inverse_transform_within`] reads for coefficients in `bounds`: the
/// rows up to the last nonzero one, and the columns up to the last nonzero
/// one rounded up to what the vertical pass handles at once. Everything
/// else may hold garbage.
pub fn transform_reads(nt: usize, bounds: CoeffBox) -> (usize, usize) {
    if nt <= 4 {
        return (nt, nt);
    }
    let width = bounds.width.clamp(1, nt);
    // The 32-point transform reads one column at a time, the others the
    // first 4, 8 or 16 columns at once.
    (if nt == 32 { width } else { width.next_power_of_two().max(4) }, bounds.height.clamp(1, nt))
}

/// Inverse transform for an `nT x nT` block of dequantized coefficients
/// (row-major, `coeffs[y*nT+x]`), producing the spatial residual
/// (row-major `i32`). `use_dst` selects the 4x4 DST-VII (luma intra 4x4
/// only); every other case uses the DCT-II. `out` needs no clearing
/// beforehand, so the caller can reuse it across blocks.
pub fn inverse_transform(coeffs: &[i32], nt: usize, bit_depth: u32, use_dst: bool, out: &mut [i32]) -> Result<()> {
    if coeffs.len() != nt * nt || out.len() != nt * nt {
        return Err(HeifError::MalformedHevc("inverse_transform: coefficient buffer size mismatch"));
    }
    // Both stages take inputs clipped to 16 bits (dequantization already
    // guarantees it for decoded blocks).
    let mut clipped = [0i32; 32 * 32];
    let clipped = clipped.get_mut(..nt * nt).ok_or(HeifError::Unsupported("inverse_transform: size must be 4/8/16/32"))?;
    for (c, &v) in clipped.iter_mut().zip(coeffs) {
        *c = clip_coeff(v);
    }
    inverse_transform_within(clipped, nt, bit_depth, use_dst, CoeffBox::full(nt), out)
}

/// [`inverse_transform`] of a block whose nonzero coefficients all lie in
/// `bounds`, reading only the coefficients [`transform_reads`] names: the
/// others may hold anything. The result is the full transform's, sample
/// for sample. The coefficients must be within 16 bits, as dequantization
/// clips them.
pub fn inverse_transform_within(coeffs: &[i32], nt: usize, bit_depth: u32, use_dst: bool, bounds: CoeffBox, out: &mut [i32]) -> Result<()> {
    if coeffs.len() != nt * nt || out.len() != nt * nt {
        return Err(HeifError::MalformedHevc("inverse_transform: coefficient buffer size mismatch"));
    }
    let bd_shift = match 20u32.checked_sub(bit_depth) {
        Some(shift) if shift >= 1 => shift,
        _ => return Err(HeifError::Unsupported("inverse_transform: bit depth too high")),
    };
    #[cfg(target_arch = "x86_64")]
    if !use_dst && super::transform_avx2::inverse_dct(coeffs, nt, bounds, bd_shift, out) {
        return Ok(());
    }
    let (columns, _) = transform_reads(nt, bounds);
    match (nt, use_dst, columns) {
        (4, true, _) => two_stage::<Dst4, 4, 4>(coeffs, bounds, bd_shift, out),
        (_, true, _) => return Err(HeifError::MalformedHevc("inverse_transform: DST-VII only valid for 4x4")),
        (4, false, _) => two_stage::<Dct4, 4, 4>(coeffs, bounds, bd_shift, out),
        (8, false, 4) => two_stage::<Dct8, 8, 4>(coeffs, bounds, bd_shift, out),
        (8, false, _) => two_stage::<Dct8, 8, 8>(coeffs, bounds, bd_shift, out),
        (16, false, 4) => two_stage::<Dct16, 16, 4>(coeffs, bounds, bd_shift, out),
        (16, false, 8) => two_stage::<Dct16, 16, 8>(coeffs, bounds, bd_shift, out),
        (16, false, _) => two_stage::<Dct16, 16, 16>(coeffs, bounds, bd_shift, out),
        (32, false, _) => two_stage_by_lines::<Dct32, 32>(coeffs, bounds, bd_shift, out),
        _ => return Err(HeifError::Unsupported("inverse_transform: size must be 4/8/16/32")),
    }
    Ok(())
}

/// [`inverse_transform`] of a DCT block whose only nonzero coefficient is
/// the DC. Row 0 of every DCT basis is 64,
/// so both stages reduce to one product each and the residual is flat:
/// the same arithmetic as [`two_stage`], done once instead of per sample.
/// Not for the 4x4 DST, whose first basis row is not constant.
pub fn inverse_dct_dc(dc: i32, bit_depth: u32, out: &mut [i32]) -> Result<()> {
    let bd_shift = match 20u32.checked_sub(bit_depth) {
        Some(shift) if shift >= 1 => shift,
        _ => return Err(HeifError::Unsupported("inverse_transform: bit depth too high")),
    };
    let g = clip_coeff((64 * clip_coeff(dc) + 64) >> 7);
    out.fill((64 * g + (1 << (bd_shift - 1))) >> bd_shift);
    Ok(())
}

/// One `N`-point inverse transform of `L` independent inputs side by side:
/// `out[i][l] = sum_t m[t][i] * x[t * pitch][l]`, where the rows past the
/// end of `x` are zero, and so are the rows `t >= n` (`n` only saves work).
///
/// The DCT kernels are partial butterflies: basis function `t` is
/// symmetric about the middle of the block for even `t` and
/// antisymmetric for odd `t`, so outputs `i` and `N - 1 - i` share the
/// even rows' sum `E[i]` (the `N / 2`-point transform of those rows) and
/// the odd rows' sum `O[i]` with opposite signs: `E[i] + O[i]` and
/// `E[i] - O[i]`. That only regroups the same integer products, so the
/// result is exactly the matrix product's; nothing overflows `i32` (see
/// [`two_stage`]).
trait Kernel<const N: usize> {
    fn apply<T: Sample, const L: usize>(x: &[[T; L]], pitch: usize, n: usize, out: &mut [[i32; L]; N]);
}

/// A transform input: a dequantized coefficient, or an intermediate value
/// between the two stages, kept as `i16` so that the compiler can multiply
/// with 16-bit instructions.
trait Sample: Copy + Default + Into<i32> {}

impl Sample for i16 {}

impl Sample for i32 {}

/// Row `t` of `x` as `i32`, zeros past its end.
#[inline(always)]
fn row<T: Sample, const L: usize>(x: &[[T; L]], t: usize) -> [i32; L] {
    x.get(t).copied().unwrap_or([T::default(); L]).map(Into::into)
}

/// The 4-point DCT of the rows `0, step, 2 * step, 3 * step` of `x`: rows
/// 0 and 2 of its basis are `64 * (1, 1, 1, 1)` and `64 * (1, -1, -1, 1)`,
/// rows 1 and 3 `(83, 36, -36, -83)` and `(36, -83, 83, -36)`.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // one lane of eight arrays
fn dct4<T: Sample, const L: usize>(x: &[[T; L]], step: usize) -> [[i32; L]; 4] {
    let [s0, s1, s2, s3] = [0, 1, 2, 3].map(|t| row(x, t * step));
    let mut out = [[0; L]; 4];
    for l in 0..L {
        let (e0, e1) = (64 * (s0[l] + s2[l]), 64 * (s0[l] - s2[l]));
        let (o0, o1) = (83 * s1[l] + 36 * s3[l], 36 * s1[l] - 83 * s3[l]);
        out[0][l] = e0 + o0;
        out[1][l] = e1 + o1;
        out[2][l] = e1 - o1;
        out[3][l] = e0 - o0;
    }
    out
}

/// The odd half of the 8-point DCT of the rows `0, step, 2 * step, ...`
/// of `x`: its basis functions 1, 3, 5 and 7 over the first four samples.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // one lane of eight arrays
fn dct8_odd<T: Sample, const L: usize>(x: &[[T; L]], step: usize) -> [[i32; L]; 4] {
    let [s1, s3, s5, s7] = [1, 3, 5, 7].map(|t| row(x, t * step));
    let mut o = [[0; L]; 4];
    for l in 0..L {
        o[0][l] = 89 * s1[l] + 75 * s3[l] + 50 * s5[l] + 18 * s7[l];
        o[1][l] = 75 * s1[l] - 18 * s3[l] - 89 * s5[l] - 50 * s7[l];
        o[2][l] = 50 * s1[l] - 89 * s3[l] + 18 * s5[l] + 75 * s7[l];
        o[3][l] = 18 * s1[l] - 50 * s3[l] + 75 * s5[l] - 89 * s7[l];
    }
    o
}

/// The odd part of a `2 * H`-point DCT with odd basis functions `m` (see
/// [`odd_basis`]) on the rows `0, step, 2 * step, ...` of `x`, the first
/// `n` of them: `o[i] = sum m[t][i] * x[(2 * t + 1) * step]` for `i < H`.
#[inline(always)]
fn odd_part<T: Sample, const L: usize, const H: usize>(m: &[[i16; H]; H], x: &[[T; L]], step: usize, n: usize) -> [[i32; L]; H] {
    let mut o = [[0i32; L]; H];
    // A loop over the rows present, without an exit per row, so the
    // compiler keeps it a loop and vectorises its body.
    for (basis, row) in m.iter().zip(x.iter().step_by(step).take(n).skip(1).step_by(2)) {
        for (acc, &weight) in o.iter_mut().zip(basis) {
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += i32::from(weight) * v.into();
            }
        }
    }
    o
}

/// `out = [e[0] + o[0], ..., e[H-1] + o[H-1], e[H-1] - o[H-1], ..., e[0] - o[0]]`.
#[inline(always)]
fn butterfly<const L: usize>(e: &[[i32; L]], o: &[[i32; L]], out: &mut [[i32; L]]) {
    let (low, high) = out.split_at_mut(e.len());
    for (((low, high), e), o) in low.iter_mut().zip(high.iter_mut().rev()).zip(e).zip(o) {
        for (((low, high), &e), &o) in low.iter_mut().zip(high.iter_mut()).zip(e).zip(o) {
            let sum = e + o;
            *low = sum;
            // Rather than `e - o`: the compiler (LLVM 22, plain x86-64)
            // vectorises neither half when both are computed from `e` and `o`.
            *high = sum - 2 * o;
        }
    }
}

/// The 4-point DST-VII, written out with HM's shared terms.
struct Dst4;

impl Kernel<4> for Dst4 {
    #[inline(always)]
    #[allow(clippy::needless_range_loop)] // one lane of eight arrays
    fn apply<T: Sample, const L: usize>(x: &[[T; L]], pitch: usize, _: usize, out: &mut [[i32; L]; 4]) {
        let [s0, s1, s2, s3] = [0, 1, 2, 3].map(|t| row(x, t * pitch));
        for l in 0..L {
            let (c0, c1, c2, c3) = (s0[l] + s2[l], s2[l] + s3[l], s0[l] - s3[l], 74 * s1[l]);
            out[0][l] = 29 * c0 + 55 * c1 + c3;
            out[1][l] = 55 * c2 - 29 * c1 + c3;
            out[2][l] = 74 * (s0[l] - s2[l] + s3[l]);
            out[3][l] = 55 * c0 + 29 * c2 - c3;
        }
    }
}

struct Dct4;

impl Kernel<4> for Dct4 {
    #[inline(always)]
    fn apply<T: Sample, const L: usize>(x: &[[T; L]], pitch: usize, _: usize, out: &mut [[i32; L]; 4]) {
        *out = dct4(x, pitch);
    }
}

/// The 8-point DCT: the 4-point one on the even rows, and the odd half.
struct Dct8;

impl Kernel<8> for Dct8 {
    #[inline(always)]
    fn apply<T: Sample, const L: usize>(x: &[[T; L]], pitch: usize, _: usize, out: &mut [[i32; L]; 8]) {
        butterfly(&dct4(x, 2 * pitch), &dct8_odd(x, pitch), out);
    }
}

/// The 16-point DCT: the 8-point one on the even rows, the odd rows from the basis.
struct Dct16;

impl Kernel<16> for Dct16 {
    #[inline(always)]
    fn apply<T: Sample, const L: usize>(x: &[[T; L]], pitch: usize, n: usize, out: &mut [[i32; L]; 16]) {
        let mut even = [[0; L]; 8];
        butterfly(&dct4(x, 4 * pitch), &dct8_odd(x, 2 * pitch), &mut even);
        butterfly(&even, &odd_part(&ODD_16, x, pitch, n), out);
    }
}

/// The 32-point DCT: the 16-point one on the even rows, the odd rows from the basis.
struct Dct32;

impl Kernel<32> for Dct32 {
    #[inline(always)]
    fn apply<T: Sample, const L: usize>(x: &[[T; L]], pitch: usize, n: usize, out: &mut [[i32; L]; 32]) {
        let (mut even8, mut even16) = ([[0; L]; 8], [[0; L]; 16]);
        butterfly(&dct4(x, 8 * pitch), &dct8_odd(x, 4 * pitch), &mut even8);
        butterfly(&even8, &odd_part(&ODD_16, x, 2 * pitch, n.div_ceil(2)), &mut even16);
        butterfly(&even16, &odd_part(&ODD_32, x, pitch, n), out);
    }
}

/// The separable two-stage transform of clause 8.6.4.2 with kernel `K`,
/// for coefficients inside `bounds`: stage 1 on `L` columns side by side,
/// stage 2 one row at a time with its outputs side by side. `coeffs` and
/// `out` hold exactly `N * N` values; only the first `bounds.height` rows
/// (all of them for `N == 4`) of the first `L` columns of `coeffs` are
/// read, `L` at least `bounds.width`.
///
/// Everything fits `i32`: both stages take inputs clipped to 16 bits, and
/// a sum of 32 products of a 16-bit value with a basis entry of at most
/// 90 stays below 2^27, as does any partial sum of it.
fn two_stage<K: Kernel<N>, const N: usize, const L: usize>(coeffs: &[i32], bounds: CoeffBox, bd_shift: u32, out: &mut [i32]) {
    // Stage 1 (vertical): coefficient row t starts at chunk t * N / L, and
    // the rows past the box are left out. Columns past `bounds.width` come
    // out zero, so stage 2 skips them.
    let rows = if N == 4 { N } else { bounds.height };
    let x = coeffs.get(..rows * N).unwrap_or(coeffs).as_chunks::<L>().0;
    let mut sums = [[0i32; L]; N];
    K::apply(x, N / L, rows, &mut sums);
    let mut g = [[0i16; L]; N];
    for (g, &v) in g.as_flattened_mut().iter_mut().zip(sums.as_flattened()) {
        *g = clip_coeff((v + 64) >> 7) as i16;
    }

    // Stage 2 (horizontal), one row at a time, straight into `out`.
    let rnd = 1i32 << (bd_shift - 1);
    for (dst, g_row) in out.as_chunks_mut::<N>().0.iter_mut().zip(&g) {
        if let Ok(dst) = <&mut [[i32; 1]; N]>::try_from(dst.as_chunks_mut::<1>().0) {
            K::apply(g_row.as_chunks::<1>().0, 1, bounds.width, dst);
        }
        for o in dst.iter_mut() {
            *o = (*o + rnd) >> bd_shift;
        }
    }
}

/// [`two_stage`] one line at a time in both stages: each column, then
/// each row, with its `N` outputs side by side. For the 32-point DCT, whose
/// columns side by side would not fit in registers. Reads only the
/// coefficients inside `bounds`.
fn two_stage_by_lines<K: Kernel<N>, const N: usize>(coeffs: &[i32], bounds: CoeffBox, bd_shift: u32, out: &mut [i32]) {
    let (width, height) = (bounds.width.min(N), bounds.height.min(N));
    // Stage 1 (vertical): column c of the intermediate block, kept as row c.
    let mut columns = [[0i16; N]; N];
    for (c, column) in columns.iter_mut().enumerate().take(width) {
        let mut x = [[0i16; 1]; N];
        for (x, &v) in x.iter_mut().zip(coeffs.iter().skip(c).step_by(N)).take(height) {
            *x = [v as i16];
        }
        let mut sums = [[0i32; 1]; N];
        K::apply(&x, 1, height, &mut sums);
        for (g, [v]) in column.iter_mut().zip(sums) {
            *g = clip_coeff((v + 64) >> 7) as i16;
        }
    }

    // Stage 2 (horizontal): row y of the intermediate block is entry y of
    // the first `width` columns.
    let rnd = 1i32 << (bd_shift - 1);
    for (y, dst) in out.as_chunks_mut::<N>().0.iter_mut().enumerate() {
        let mut x = [[0i16; 1]; N];
        for (x, column) in x.iter_mut().zip(&columns).take(width) {
            *x = [column.get(y).copied().unwrap_or(0)];
        }
        if let Ok(dst) = <&mut [[i32; 1]; N]>::try_from(dst.as_chunks_mut::<1>().0) {
            K::apply(&x, 1, width, dst);
        }
        for o in dst.iter_mut() {
            *o = (*o + rnd) >> bd_shift;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4x4 DST-VII integer basis used for luma intra 4x4 (clause 8.6.4.1).
    const MAT_DST_4: [[i32; 4]; 4] = [[29, 55, 74, 84], [74, 74, 0, -74], [84, -29, -74, 55], [55, -84, 74, -29]];

    /// The `n`-point DCT-II basis: rows `0, 32/n, 2*32/n, ...` of [`MAT_DCT`], first `n` columns.
    const fn dct_matrix<const N: usize>() -> [[i32; N]; N] {
        let mut m = [[0; N]; N];
        let mut j = 0;
        while j < N {
            let mut i = 0;
            while i < N {
                m[j][i] = MAT_DCT[32 / N * j][i];
                i += 1;
            }
            j += 1;
        }
        m
    }

    const MAT_DCT_4: [[i32; 4]; 4] = dct_matrix();
    const MAT_DCT_8: [[i32; 8]; 8] = dct_matrix();
    const MAT_DCT_16: [[i32; 16]; 16] = dct_matrix();
    const MAT_DCT_32: [[i32; 32]; 32] = dct_matrix();

    /// The transform as it was before the butterflies: the direct matrix
    /// products, kept to check the new one against, bit for bit.
    ///
    /// The separable two-stage transform of clause 8.6.4.2 with basis `m`
    /// (`m[j][i]`: basis function `j` at sample `i`). `coeffs` and `out` hold
    /// exactly `N * N` values.
    ///
    /// Everything fits `i32`: both stages take inputs clipped to 16 bits (the
    /// input here too, which dequantization already guarantees), and a sum of
    /// 32 products of a 16-bit value with a basis entry of at most 90 stays
    /// below 2^27. The loops run over whole fixed-size rows, so the compiler
    /// can vectorise them; all-zero coefficient rows, most of a typical
    /// block, are skipped.
    fn two_stage_matrix<const N: usize>(coeffs: &[i32], m: &[[i32; N]; N], bd_shift: u32, out: &mut [i32]) {
        // Stage 1 (vertical): g[i][c] = sum_j m[j][i] * d[j][c], accumulated
        // one coefficient row j at a time so the inner loop runs along c.
        let mut g = [[0i32; N]; N];
        for (row, basis) in coeffs.as_chunks::<N>().0.iter().zip(m) {
            // Small blocks: computing a zero row costs less than the
            // mispredicted branch that would skip it.
            if N >= 16 && row.iter().all(|&d| d == 0) {
                continue;
            }
            let d = row.map(clip_coeff);
            for (g_row, &weight) in g.iter_mut().zip(basis) {
                for (acc, &dc) in g_row.iter_mut().zip(&d) {
                    *acc += weight * dc;
                }
            }
        }
        for g_row in &mut g {
            for v in g_row.iter_mut() {
                *v = clip_coeff((*v + 64) >> 7);
            }
        }

        // Stage 2 (horizontal): out[y][i] = sum_j m[j][i] * g[y][j].
        let rnd = 1i32 << (bd_shift - 1);
        for (dst, g_row) in out.as_chunks_mut::<N>().0.iter_mut().zip(&g) {
            let mut acc = [0i32; N];
            for (&gj, basis) in g_row.iter().zip(m) {
                if N >= 16 && gj == 0 {
                    continue;
                }
                for (a, &weight) in acc.iter_mut().zip(basis) {
                    *a += weight * gj;
                }
            }
            for (o, a) in dst.iter_mut().zip(acc) {
                *o = (a + rnd) >> bd_shift;
            }
        }
    }

    /// [`two_stage_matrix`] on a whole block.
    fn matrix_transform(coeffs: &[i32], nt: usize, bit_depth: u32, use_dst: bool) -> Vec<i32> {
        let mut out = vec![99; nt * nt];
        let shift = 20 - bit_depth;
        match (nt, use_dst) {
            (4, true) => two_stage_matrix(coeffs, &MAT_DST_4, shift, &mut out),
            (4, false) => two_stage_matrix(coeffs, &MAT_DCT_4, shift, &mut out),
            (8, false) => two_stage_matrix(coeffs, &MAT_DCT_8, shift, &mut out),
            (16, false) => two_stage_matrix(coeffs, &MAT_DCT_16, shift, &mut out),
            (32, false) => two_stage_matrix(coeffs, &MAT_DCT_32, shift, &mut out),
            _ => panic!("no such transform"),
        }
        out
    }

    #[test]
    fn the_butterflies_give_the_matrix_products_bit_for_bit() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut blocks = 0;
        for (nt, use_dst) in [(4, true), (4, false), (8, false), (16, false), (32, false)] {
            for bit_depth in 8..=16 {
                for round in 0..400 {
                    // The nonzero coefficients' box: anything from the DC
                    // alone to the whole block, then some lopsided ones.
                    let (width, height) = match round % 4 {
                        0 => (1 + next() as usize % nt, 1 + next() as usize % nt),
                        1 => (1 + next() as usize % 4.min(nt), 1 + next() as usize % nt),
                        2 => (1 + next() as usize % nt, 1 + next() as usize % 4.min(nt)),
                        _ => (nt, nt),
                    };
                    let bounds = CoeffBox { width, height };
                    // Sparse, dense, extreme, small: values the decoder can
                    // produce, 16-bit, with the corner of the box set so
                    // the box is tight.
                    let density = [1, 2, 4, 64][next() as usize % 4];
                    let mut coeffs = vec![0i32; nt * nt];
                    for y in 0..height {
                        for x in 0..width {
                            if next() % density == 0 || (x + 1, y + 1) == (width, height) {
                                coeffs[y * nt + x] = match next() % 6 {
                                    0 => 32767,
                                    1 => -32768,
                                    2 => (next() % 65536) as i32 - 32768,
                                    3 => (next() % 33) as i32 - 16,
                                    4 => (next() % 2049) as i32 - 1024,
                                    _ => [1, -1][next() as usize % 2],
                                };
                            }
                        }
                    }
                    let expected = matrix_transform(&coeffs, nt, bit_depth, use_dst);
                    assert_eq!(idct(&coeffs, nt, bit_depth, use_dst).unwrap(), expected, "nt={nt} dst={use_dst} bd={bit_depth} {bounds:?}");

                    // The bounded transform must not read past what it says
                    // it reads: garbage everywhere else, as in reused scratch.
                    let (columns, rows) = transform_reads(nt, bounds);
                    let mut scratch: Vec<i32> = (0..nt * nt).map(|_| next() as i32).collect();
                    for y in 0..rows {
                        for x in 0..columns {
                            scratch[y * nt + x] = coeffs[y * nt + x];
                        }
                    }
                    let mut out = vec![99; nt * nt];
                    inverse_transform_within(&scratch, nt, bit_depth, use_dst, bounds, &mut out).unwrap();
                    assert_eq!(out, expected, "within: nt={nt} dst={use_dst} bd={bit_depth} {bounds:?}");
                    blocks += 1;
                }
            }
        }
        assert_eq!(blocks, 5 * 9 * 400);
    }

    #[test]
    fn out_of_range_coefficients_are_clipped_as_before() {
        let mut seed = 0x0123_4567_89ab_cdefu64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for (nt, use_dst) in [(4, true), (4, false), (8, false), (16, false), (32, false)] {
            for bit_depth in [8, 12, 16] {
                for _ in 0..50 {
                    let coeffs: Vec<i32> = (0..nt * nt).map(|_| [32768, -32769, i32::MAX, i32::MIN, 0, next() as i32][next() as usize % 6]).collect();
                    assert_eq!(idct(&coeffs, nt, bit_depth, use_dst).unwrap(), matrix_transform(&coeffs, nt, bit_depth, use_dst));
                }
            }
        }
    }

    #[test]
    fn the_read_region_covers_the_box() {
        for nt in [4, 8, 16, 32] {
            for width in 1..=nt {
                for height in 1..=nt {
                    let (columns, rows) = transform_reads(nt, CoeffBox { width, height });
                    assert!(columns >= width && rows >= height && columns <= nt && rows <= nt);
                }
            }
        }
    }

    #[test]
    fn a_zero_level_dequantizes_to_zero_whatever_the_parameters() {
        for bit_depth in 8..=16 {
            for log2_size in 2..=5 {
                for qp in 0..=51 + 6 * (bit_depth as i32 - 8) {
                    let scale = Dequant::new(qp, bit_depth, log2_size).unwrap();
                    for weight in [0, 1, 16, 255] {
                        assert_eq!(scale.apply_checked(0, weight), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn the_dc_shortcut_matches_the_full_transform() {
        for nt in [4usize, 8, 16, 32] {
            for bit_depth in [8, 10, 12] {
                for dc in [-32768, -32767, -4097, -1000, -65, -64, -63, -1, 0, 1, 63, 64, 65, 999, 4096, 32767] {
                    let mut coeffs = vec![0; nt * nt];
                    coeffs[0] = dc;
                    let mut full = vec![7; nt * nt];
                    inverse_transform(&coeffs, nt, bit_depth, false, &mut full).unwrap();
                    let mut fast = vec![7; nt * nt];
                    inverse_dct_dc(dc, bit_depth, &mut fast).unwrap();
                    assert_eq!(fast, full, "nt {nt} depth {bit_depth} dc {dc}");
                }
            }
        }
    }

    // Buffers start out full of garbage, as reused decoder scratch would.
    fn idct(coeffs: &[i32], nt: usize, bit_depth: u32, use_dst: bool) -> Result<Vec<i32>> {
        let mut out = vec![99; coeffs.len()];
        inverse_transform(coeffs, nt, bit_depth, use_dst, &mut out).map(|()| out)
    }

    fn transform_skip(coeffs: &[i32], nt: usize, bit_depth: u32) -> Result<Vec<i32>> {
        let mut out = vec![99; coeffs.len()];
        transform_skip_residual(coeffs, nt, bit_depth, &mut out).map(|()| out)
    }

    #[test]
    fn dequant_uses_matrix_weight_before_rounding() {
        // Qp'=0, 8-bit 8x8: shift=6, levelScale=40. A weight of 17
        // gives (3*17*40+32)>>6 = 32, versus 30 with the flat matrix.
        assert_eq!(dequant_with_scale(3, 0, 8, 3, 17).unwrap(), 32);
        assert_eq!(dequant_with_scale(3, 0, 8, 3, 16).unwrap(), 30);
        assert_eq!(dequant_with_scale(-3, 0, 8, 3, 17).unwrap(), -32);
        assert_eq!(dequant_with_scale(3, 0, 8, 3, 0).unwrap(), 0);
        // For 10-bit QpY=0, Qp'Y=12 compensates the extra two shift bits.
        assert_eq!(dequant_with_scale(3, 12, 10, 3, 17).unwrap(), 32);
    }

    #[test]
    fn a_zero_level_dequantizes_to_zero() {
        // What lets the block loop skip zeros: the rounding offset is below
        // one step of the shift, for every parameter set.
        for bit_depth in 8..=16 {
            for log2_size in 2..=5 {
                for qp in 0..=51 + 6 * (bit_depth as i32 - 8) {
                    for weight in [0, 1, 16, 255] {
                        assert_eq!(dequant_with_scale(0, qp, bit_depth, log2_size, weight).unwrap(), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn dc_only_dct_is_flat() {
        // Hand-derived (see module history / PR description): for an
        // 8-bit block with only the DC coefficient (value 128) set,
        // every output sample must equal exactly 1, because row 0 and
        // column 0 of the DCT-II basis are constant (64), collapsing
        // the whole 2-D separable transform to two scalar passes.
        for &nt in &[4usize, 8, 16, 32] {
            let mut coeffs = vec![0i32; nt * nt];
            coeffs[0] = 128;
            let out = idct(&coeffs, nt, 8, false).expect("DC-only transform should succeed");
            assert!(out.iter().all(|&v| v == 1), "nt={nt} out={out:?}");
        }
    }

    #[test]
    fn all_zero_coeffs_is_all_zero_residual() {
        let coeffs = vec![0i32; 64];
        let out = idct(&coeffs, 8, 8, false).expect("all-zero transform should succeed");
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn transform_skip_residual_matches_hand_computed_values() {
        // 8-bit 4x4: tsShift = 5 + log2(4) = 7, bdShift = 20 - 8 = 12,
        // rnd = 1<<11 = 2048 (cross-checked against
        // `libde265::transform_skip_residual_fallback`, see module docs).
        // coeff=100: (100<<7 + 2048)>>12 = (12800+2048)>>12 = 14848>>12 = 3.
        // coeff=-100: (-12800+2048)>>12 = -10752>>12 = -3 (floor division).
        // coeff=0 stays 0 regardless of rounding.
        let coeffs = [100, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let out = transform_skip(&coeffs, 4, 8).expect("transform_skip_residual should succeed");
        assert_eq!(out[0], 3);
        assert_eq!(out[1], -3);
        assert!(out[2..].iter().all(|&v| v == 0));
    }

    #[test]
    fn dequant_known_value() {
        // qp=32 -> qp%6=2 (levelScale=51), qp/6=5. bit_depth=8, log2_size=3 (8x8) -> bdShift=8+3-5-4=2.
        // fact = 51<<5 = 1632. offset=1<<1=2. level=10: (10*1632+2)>>2 = 16322>>2=4080.
        let v = dequant(10, 32, 8, 3).expect("dequant should succeed");
        assert_eq!(v, 4080);
    }

    /// The two stages written out directly from clause 8.6.4.2 in `i64`,
    /// indexing the 32-point basis as the spec does.
    fn reference(coeffs: &[i32], nt: usize, bit_depth: u32, use_dst: bool) -> Vec<i32> {
        let m = |j: usize, i: usize| i64::from(if use_dst { MAT_DST_4[j][i] } else { MAT_DCT[32 / nt * j][i] });
        let mut g = vec![0i64; nt * nt];
        for c in 0..nt {
            for i in 0..nt {
                let sum: i64 = (0..nt).map(|j| m(j, i) * i64::from(coeffs[j * nt + c])).sum();
                g[i * nt + c] = ((sum + 64) >> 7).clamp(-32768, 32767);
            }
        }
        let shift = 20 - bit_depth;
        (0..nt * nt)
            .map(|k| {
                let (y, i) = (k / nt, k % nt);
                let sum: i64 = (0..nt).map(|j| m(j, i) * g[y * nt + j]).sum();
                ((sum + (1 << (shift - 1))) >> shift) as i32
            })
            .collect()
    }

    #[test]
    fn matches_the_spec_formula_on_random_blocks() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for (nt, use_dst) in [(4, true), (4, false), (8, false), (16, false), (32, false)] {
            for bit_depth in [8, 10, 12] {
                for _ in 0..200 {
                    // Mostly zeros, like real blocks, with full-range values in between.
                    let coeffs: Vec<i32> = (0..nt * nt)
                        .map(|_| {
                            let r = next();
                            if r % 4 == 0 { (r >> 16) as i32 % 32768 } else { 0 }
                        })
                        .collect();
                    assert_eq!(
                        idct(&coeffs, nt, bit_depth, use_dst).unwrap(),
                        reference(&coeffs, nt, bit_depth, use_dst),
                        "nt={nt} dst={use_dst} bd={bit_depth}"
                    );
                }
            }
        }
    }

    #[test]
    fn extreme_coefficients_do_not_overflow() {
        // Valid streams never exceed 16 bits here (dequantization clips);
        // larger values are clipped on input instead of overflowing `i32`.
        for nt in [4, 8, 16, 32] {
            for v in [32767, -32768, i32::MAX, i32::MIN] {
                let coeffs = vec![v; nt * nt];
                let clipped = vec![v.clamp(-32768, 32767); nt * nt];
                assert_eq!(idct(&coeffs, nt, 8, false).unwrap(), reference(&clipped, nt, 8, false));
            }
        }
    }

    #[test]
    fn rejects_wrong_block_size() {
        assert!(idct(&[0; 9], 3, 8, false).is_err());
    }
}
