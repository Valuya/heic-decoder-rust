//! The 8-, 16- and 32-point inverse DCT in AVX2 (safe, through pulp), as two
//! matrix stages of 16-bit products summed in 32 bits (`pmaddwd`). Stage 1
//! interleaves pairs of coefficient rows once and multiplies them by pairs of
//! basis entries; stage 2 broadcasts pairs of intermediate samples against
//! pairs of basis rows, so neither stage transposes. Exactly the scalar
//! arithmetic: both stages take 16-bit inputs, the sums fit `i32`, stage 1
//! rounds, shifts by 7 and clips to 16 bits (a saturating pack), stage 2 rounds
//! and shifts by `bd_shift`.

use core::arch::x86_64::__m256i;
use pulp::bytemuck::cast;

use super::transform::{CoeffBox, MAT_DCT};

/// The `n`-point basis, `m[t][i]` for frequency `t` and sample `i`, padded with zeros to 32 x 32.
const fn basis(n: usize) -> [[i16; 32]; 32] {
    let mut m = [[0i16; 32]; 32];
    let mut t = 0;
    while t < n {
        let mut i = 0;
        while i < n {
            m[t][i] = MAT_DCT[32 / n * t][i] as i16;
            i += 1;
        }
        t += 1;
    }
    m
}

static BASIS: [[[i16; 32]; 32]; 3] = [basis(8), basis(16), basis(32)];

/// Whether the kernel runs; `HEIC_NO_SIMD_IDCT=1` turns it off (for A/B runs).
static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("HEIC_NO_SIMD_IDCT").as_deref() != Ok("1"));

/// The inverse DCT of an `nt x nt` block (8, 16 or 32) whose nonzero
/// coefficients lie in `bounds`, into `out`; `false` (nothing written) where
/// AVX2 is missing or the size is not handled.
pub(crate) fn inverse_dct(coeffs: &[i32], nt: usize, bounds: CoeffBox, bd_shift: u32, out: &mut [i32]) -> bool {
    if !*ENABLED || !matches!(nt, 8 | 16 | 32) || coeffs.len() < nt * nt || out.len() < nt * nt {
        return false;
    }
    let Some(simd) = pulp::x86::V3::try_new() else { return false };
    simd.vectorize(Idct { simd, coeffs, nt, width: bounds.width.clamp(1, nt), height: bounds.height.clamp(1, nt), bd_shift, out });
    true
}

struct Idct<'a> {
    simd: pulp::x86::V3,
    coeffs: &'a [i32],
    nt: usize,
    width: usize,
    height: usize,
    bd_shift: u32,
    out: &'a mut [i32],
}

/// 16 lanes of row `r` of a padded 32 x 32 block, from column `16 * k`.
#[inline(always)]
fn lanes(block: &[[i16; 32]; 32], r: usize, k: usize) -> __m256i {
    let (chunks, _) = block[r & 31].as_chunks::<16>();
    cast(chunks[k & 1])
}

/// Two 16-bit values as one 32-bit lane, `lo` first, for `pmaddwd`.
#[inline(always)]
fn pair(lo: i16, hi: i16) -> i32 {
    (i32::from(hi) << 16) | i32::from(lo as u16)
}

impl pulp::NullaryFnOnce for Idct<'_> {
    type Output = ();

    #[inline(always)]
    fn call(self) {
        let Idct { simd, coeffs, nt, width, height, bd_shift, out } = self;
        let (avx, avx2) = (simd.avx, simd.avx2);
        let m = &BASIS[match nt {
            8 => 0,
            16 => 1,
            _ => 2,
        }];

        // The coefficients inside the box as 16-bit samples (they are within 16 bits).
        let mut c = [[0i16; 32]; 32];
        for (row, src) in c.iter_mut().zip(coeffs.chunks_exact(nt)).take(height) {
            for (d, &s) in row.iter_mut().zip(src).take(width) {
                *d = s as i16;
            }
        }

        // Stage 1 (vertical): g[i][x] = clip16((sum_t m[t][i] c[t][x] + 64) >> 7).
        let chunks = width.div_ceil(16);
        let pairs = height.div_ceil(2);
        let mut rows = [[[avx._mm256_setzero_si256(); 2]; 2]; 16];
        for (p, row) in rows.iter_mut().enumerate().take(pairs) {
            for (k, half) in row.iter_mut().enumerate().take(chunks) {
                let (a, b) = (lanes(&c, 2 * p, k), lanes(&c, 2 * p + 1, k));
                *half = [avx2._mm256_unpacklo_epi16(a, b), avx2._mm256_unpackhi_epi16(a, b)];
            }
        }
        let round1 = avx._mm256_set1_epi32(64);
        let mut g = [[0i16; 32]; 32];
        for (i, g_row) in g.iter_mut().enumerate().take(nt) {
            let mut w = [avx._mm256_setzero_si256(); 16];
            for (p, w) in w.iter_mut().enumerate().take(pairs) {
                *w = avx._mm256_set1_epi32(pair(m[(2 * p) & 31][i], m[(2 * p + 1) & 31][i]));
            }
            let (g_chunks, _) = g_row.as_chunks_mut::<16>();
            for (k, dst) in g_chunks.iter_mut().enumerate().take(chunks) {
                let (mut lo, mut hi) = (avx._mm256_setzero_si256(), avx._mm256_setzero_si256());
                for (row, &w) in rows.iter().zip(&w).take(pairs) {
                    lo = avx2._mm256_add_epi32(lo, avx2._mm256_madd_epi16(row[k][0], w));
                    hi = avx2._mm256_add_epi32(hi, avx2._mm256_madd_epi16(row[k][1], w));
                }
                lo = avx2._mm256_srai_epi32::<7>(avx2._mm256_add_epi32(lo, round1));
                hi = avx2._mm256_srai_epi32::<7>(avx2._mm256_add_epi32(hi, round1));
                *dst = cast(avx2._mm256_packs_epi32(lo, hi));
            }
        }

        // Stage 2 (horizontal): out[y][x] = (sum_t g[y][t] m[t][x] + rnd) >> bd_shift.
        let x_chunks = nt.div_ceil(16);
        let t_pairs = width.div_ceil(2);
        let mut basis_rows = [[[avx._mm256_setzero_si256(); 2]; 2]; 16];
        for (p, row) in basis_rows.iter_mut().enumerate().take(t_pairs) {
            for (k, half) in row.iter_mut().enumerate().take(x_chunks) {
                let (a, b) = (lanes(m, 2 * p, k), lanes(m, 2 * p + 1, k));
                *half = [avx2._mm256_unpacklo_epi16(a, b), avx2._mm256_unpackhi_epi16(a, b)];
            }
        }
        let rnd = avx._mm256_set1_epi32(1 << (bd_shift - 1));
        let shift = simd.sse2._mm_cvtsi32_si128(bd_shift as i32);
        for (y, dst) in out.chunks_exact_mut(nt).enumerate().take(nt) {
            let g_row = &g[y];
            let mut w = [avx._mm256_setzero_si256(); 16];
            for (p, w) in w.iter_mut().enumerate().take(t_pairs) {
                *w = avx._mm256_set1_epi32(pair(g_row[(2 * p) & 31], g_row[(2 * p + 1) & 31]));
            }
            let (out8, _) = dst.as_chunks_mut::<8>();
            for k in 0..x_chunks {
                let (mut lo, mut hi) = (avx._mm256_setzero_si256(), avx._mm256_setzero_si256());
                for (row, &w) in basis_rows.iter().zip(&w).take(t_pairs) {
                    lo = avx2._mm256_add_epi32(lo, avx2._mm256_madd_epi16(row[k][0], w));
                    hi = avx2._mm256_add_epi32(hi, avx2._mm256_madd_epi16(row[k][1], w));
                }
                lo = avx2._mm256_sra_epi32(avx2._mm256_add_epi32(lo, rnd), shift);
                hi = avx2._mm256_sra_epi32(avx2._mm256_add_epi32(hi, rnd), shift);
                // the unpacks split each 128-bit half: lo holds x 0-3 and 8-11, hi 4-7 and 12-15
                if let Some(o) = out8.get_mut(2 * k) {
                    *o = cast(avx2._mm256_permute2x128_si256::<0x20>(lo, hi));
                }
                if let Some(o) = out8.get_mut(2 * k + 1) {
                    *o = cast(avx2._mm256_permute2x128_si256::<0x31>(lo, hi));
                }
            }
        }
    }
}
