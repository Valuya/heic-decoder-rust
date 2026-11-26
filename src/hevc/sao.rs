//! Sample Adaptive Offset filter (clause 8.7.3), applied to the whole
//! picture right after deblocking, per-component (luma/Cb/Cr) and
//! per-CTB using the parameters `Decoder::decode_sao` parsed.
//!
//! Edge offset depends on unmodified *neighbour* samples — including
//! ones in another CTB that might itself get an edit earlier in this
//! same pass — so every CTB reads unfiltered samples and only ever
//! writes to the live plane. This mirrors `imazen/heic`'s own SAO
//! module, which keeps the same `src`/`dst` split for edge offset (and
//! applies band offset in place) for the same reason. The unfiltered
//! samples are not a copy of the whole plane: CTB rows are filtered top
//! to bottom, and a CTB row only reads its own rows and the one on each
//! side, so the pass keeps a copy of just those, taken before the row is
//! filtered (the row above it from the copy made for the previous CTB
//! row, which has filtered it since).

use super::ctu::{BypassMap, Picture, Sample, Samples, SaoCtbParams};
use super::params::Sps;
use crate::error::Result;

/// `(dx0, dy0, dx1, dy1)` of the two neighbours compared for each of
/// the 4 `sao_eo_class` values (clause 8.7.3.2's class-to-direction
/// mapping).
const EO_OFFSETS: [(i32, i32, i32, i32); 4] = [
    (-1, 0, 1, 0),  // 0: horizontal
    (0, -1, 0, 1),  // 1: vertical
    (-1, -1, 1, 1), // 2: 135° diagonal
    (1, -1, -1, 1), // 3: 45° diagonal
];

/// Unfiltered samples of rows `first..`, `width` apart.
#[derive(Clone, Copy)]
struct Unfiltered<'a, T> {
    rows: &'a [T],
    first: i64,
    width: usize,
}

impl<T: Sample> Unfiltered<'_, T> {
    /// Row `y` of the copy, whole.
    #[inline(always)]
    fn row(&self, y: i64) -> Option<&[T]> {
        let row = usize::try_from(y - self.first).ok()?;
        self.rows.get(row * self.width..(row + 1) * self.width)
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_band<T: Sample>(
    src: Unfiltered<T>,
    dst: &mut [T],
    width: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    band_position: u8,
    offsets: [i32; 4],
    bit_depth: u32,
) {
    let max_val = (1i32 << bit_depth) - 1;
    let band_shift = bit_depth.saturating_sub(5);
    let mut band_table = [0i32; 32];
    for (k, &o) in offsets.iter().enumerate() {
        if let Some(slot) = band_table.get_mut((band_position as usize + k) & 31) {
            *slot = o;
        }
    }
    // Every sample of the CTB is written, offset 0 or not: no branch on the
    // band, and row slices instead of a lookup per sample.
    for y in y0..y0 + h {
        let (Some(row), Some(out)) = (src.row(y as i64).and_then(|r| r.get(x0..x0 + w)), dst.get_mut(y * width + x0..y * width + x0 + w)) else { continue };
        for (d, &v) in out.iter_mut().zip(row) {
            let sample: i32 = v.into();
            let offset = band_table[(sample.clamp(0, max_val) >> band_shift) as usize & 31];
            *d = T::narrow((sample + offset).clamp(0, max_val));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_edge<T: Sample>(
    src: Unfiltered<T>,
    dst: &mut [T],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    eo_class: u8,
    offsets: [i32; 4],
    bit_depth: u32,
) {
    let max_val = (1i32 << bit_depth) - 1;
    let (dx0, dy0, dx1, dy1) = EO_OFFSETS[eo_class as usize & 3];
    // Categories 1,2 (table index 0,1) add a positive offset; 3,4
    // (index 2,3) a negative one; "equal to both neighbours" (edge_idx
    // == 2, i.e. sample on a local plateau) is never offset at all.
    let table: [i32; 5] = [offsets[0], offsets[1], 0, -offsets[2], -offsets[3]];

    // The columns whose two neighbours are in the plane, once per CTB; then
    // each row from three row slices, every sample written (an offset of 0
    // writes the sample back), no branch on the edge class.
    let lo = x0.max(usize::from(dx0 < 0 || dx1 < 0));
    let hi = (x0 + w).min(width - usize::from(dx0 > 0 || dx1 > 0));
    if lo >= hi {
        return;
    }
    let sign = |a: i32, b: i32| i32::from(a > b) - i32::from(a < b);
    for y in y0..y0 + h {
        let (ny0, ny1) = (y as i64 + dy0 as i64, y as i64 + dy1 as i64);
        if ny0 < 0 || ny0 as usize >= height || ny1 < 0 || ny1 as usize >= height {
            continue;
        }
        let (Some(row), Some(r0), Some(r1)) = (src.row(y as i64), src.row(ny0), src.row(ny1)) else { continue };
        let n = hi - lo;
        let (s0, s1) = ((lo as i64 + dx0 as i64) as usize, (lo as i64 + dx1 as i64) as usize);
        let (Some(cur), Some(a), Some(b), Some(out)) = (row.get(lo..hi), r0.get(s0..s0 + n), r1.get(s1..s1 + n), dst.get_mut(y * width + lo..y * width + hi))
        else {
            continue;
        };
        for (((d, &v), &n0), &n1) in out.iter_mut().zip(cur).zip(a).zip(b) {
            let (sample, n0, n1): (i32, i32, i32) = (v.into(), n0.into(), n1.into());
            let offset = table[(2 + sign(sample, n0) + sign(sample, n1)) as usize];
            *d = T::narrow((sample + offset).clamp(0, max_val));
        }
    }
}

/// Where a component's CTBs are and which of its samples SAO leaves alone.
#[derive(Clone, Copy)]
struct Component<'a> {
    sao: &'a [SaoCtbParams],
    pic_width_ctbs: usize,
    pic_height_ctbs: usize,
    cidx: usize,
    width: usize,
    height: usize,
    ctb_width: usize,
    ctb_height: usize,
    bit_depth: u32,
    /// The bypass map and this component's subsampling, when any CU is
    /// coded with transquant bypass: those samples keep their values.
    bypass: Option<(&'a BypassMap<'a>, usize, usize)>,
}

/// Runs one component's whole-picture SAO pass in place, one CTB row at a
/// time: copies the rows the CTB row reads into `rows` (the one above from
/// the previous CTB row's copy), then applies every CTB's (already-parsed)
/// band/edge offset over its own area (clipped to the plane's true bounds —
/// the last row/column of CTBs is typically only partially used), reading
/// the copy and writing `plane`.
fn apply_component<T: Sample>(c: &Component, plane: &mut [T], rows: &mut Vec<T>) {
    let width = c.width;
    let window = c.ctb_height + 2;
    rows.clear();
    rows.resize(window * width, T::default());
    // Whether `rows` holds the previous CTB row (it was filtered).
    let mut previous = false;
    for ctb_y in 0..c.pic_height_ctbs {
        let y0 = ctb_y * c.ctb_height;
        if y0 >= c.height {
            continue;
        }
        let h = c.ctb_height.min(c.height - y0);
        let filtered = |ctb_x: usize| {
            let params = c.sao.get(ctb_y * c.pic_width_ctbs + ctb_x)?;
            let type_idx = *params.type_idx.get(c.cidx)?;
            (type_idx != 0 && ctb_x * c.ctb_width < width).then_some((params, type_idx))
        };
        if !(0..c.pic_width_ctbs).any(|ctb_x| filtered(ctb_x).is_some()) {
            previous = false;
            continue;
        }
        // Row `y0 - 1 + k` of the plane in row `k` of `rows`.
        if y0 > 0 {
            if previous {
                rows.copy_within(c.ctb_height * width..(c.ctb_height + 1) * width, 0);
            } else {
                rows[..width].copy_from_slice(&plane[(y0 - 1) * width..y0 * width]);
            }
        }
        let end = (y0 + h + 1).min(c.height);
        rows[width..(1 + end - y0) * width].copy_from_slice(&plane[y0 * width..end * width]);
        let src = Unfiltered { rows, first: y0 as i64 - 1, width };
        for ctb_x in 0..c.pic_width_ctbs {
            let Some((params, type_idx)) = filtered(ctb_x) else { continue };
            let x0 = ctb_x * c.ctb_width;
            let w = c.ctb_width.min(width - x0);
            let extra = params.extra.get(c.cidx).copied().unwrap_or(0);
            let offsets = params.offsets.get(c.cidx).copied().unwrap_or([0; 4]);
            if type_idx == 1 {
                apply_band(src, plane, width, x0, y0, w, h, extra, offsets, c.bit_depth);
            } else {
                apply_edge(src, plane, width, c.height, x0, y0, w, h, extra, offsets, c.bit_depth);
            }
        }
        if let Some((bypass, sub_w, sub_h)) = c.bypass {
            for y in y0..y0 + h {
                for x in 0..width {
                    if bypass.at((x * sub_w) as i64, (y * sub_h) as i64) {
                        plane[y * width + x] = rows[(y - y0 + 1) * width + x];
                    }
                }
            }
        }
        previous = true;
    }
}

/// Applies the SAO filter to the whole picture in-place.
pub fn apply_sao(pic: &mut Picture, sps: &Sps, sao_luma: bool, sao_chroma: bool) -> Result<()> {
    let ctb_pixel = 1usize << pic.ctb_log2;
    let (pic_width_ctbs, pic_height_ctbs) = (pic.pic_width_ctbs(), pic.pic_height_ctbs());
    let sao = pic.sao_params().to_vec();
    let restore_bypass = pic.has_bypass();
    // A component no CTB applies SAO to is left as it is: encoders often
    // signal SAO for the slice and then use it nowhere.
    let used = |cidx: usize| sao.iter().any(|params| params.type_idx[cidx] != 0);
    let mut filter = |cidx: usize, ctb_width: usize, ctb_height: usize, sub_w: usize, sub_h: usize| {
        let (plane, rows, bypass) = pic.sao_parts(cidx);
        let c = Component {
            sao: &sao,
            pic_width_ctbs,
            pic_height_ctbs,
            cidx,
            width: plane.width,
            height: plane.height,
            ctb_width,
            ctb_height,
            bit_depth: plane.bit_depth,
            bypass: restore_bypass.then_some((&bypass, sub_w, sub_h)),
        };
        match plane.samples_mut() {
            Samples::Narrow(samples) => apply_component(&c, samples, &mut rows.narrow),
            Samples::Wide(samples) => apply_component(&c, samples, &mut rows.wide),
        }
    };

    if sao_luma && used(0) {
        filter(0, ctb_pixel, ctb_pixel, 1, 1);
    }
    if sao_chroma && sps.chroma_format_idc != 0 {
        let (sub_w, sub_h) = (sps.sub_width_c() as usize, sps.sub_height_c() as usize);
        let (chroma_ctb_width, chroma_ctb_height) = (ctb_pixel / sub_w, ctb_pixel / sub_h);
        for cidx in [1, 2] {
            if used(cidx) {
                filter(cidx, chroma_ctb_width, chroma_ctb_height, sub_w, sub_h);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn whole(rows: &[u8], width: usize) -> Unfiltered<'_, u8> {
        Unfiltered { rows, first: 0, width }
    }

    #[test]
    fn bypass_samples_keep_their_values_while_neighbours_receive_sao() {
        let mut pic = Picture::new(8, 4, 2, 2, 8, 8, 3, 4);
        for y in 0..4 {
            for x in 0..8 {
                pic.y.set_sample(x, y, 10);
            }
        }
        pic.mark_bypass(0, 0, 4, true);
        let mut params = SaoCtbParams::default();
        params.type_idx[0] = 1;
        params.extra[0] = 1;
        params.offsets[0] = [3, 0, 0, 0];
        let sao = [params];
        let (plane, rows, bypass) = pic.sao_parts(0);
        let c = Component {
            sao: &sao,
            pic_width_ctbs: 1,
            pic_height_ctbs: 1,
            cidx: 0,
            width: 8,
            height: 4,
            ctb_width: 16,
            ctb_height: 16,
            bit_depth: 8,
            bypass: Some((&bypass, 1, 1)),
        };
        let Samples::Narrow(samples) = plane.samples_mut() else { panic!("an 8-bit plane holds bytes") };
        apply_component(&c, samples, &mut rows.narrow);
        for y in 0..4 {
            for x in 0..8 {
                assert_eq!(pic.y.sample(x, y), Some(if x < 4 { 10 } else { 13 }));
            }
        }
    }

    /// Filtering CTB row by CTB row from copies of the rows each reads gives
    /// what filtering every CTB from one copy of the whole plane gives.
    #[test]
    fn row_by_row_matches_a_whole_plane_copy() {
        let (width, height, ctb) = (20usize, 19usize, 8usize);
        let (w_ctbs, h_ctbs) = (width.div_ceil(ctb), height.div_ceil(ctb));
        let mut seed = 0x2545_f491_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for round in 0..64 {
            let plane: Vec<u8> = (0..width * height).map(|_| 100 + (next() % 9) as u8).collect();
            let sao: Vec<SaoCtbParams> = (0..w_ctbs * h_ctbs)
                .map(|_| {
                    let mut p = SaoCtbParams::default();
                    // Some CTB rows entirely unfiltered, the others mixed.
                    p.type_idx[0] = if round % 3 == 0 { (next() % 2) as u8 * 2 } else { (next() % 3) as u8 };
                    p.extra[0] = (next() % if p.type_idx[0] == 1 { 32 } else { 4 }) as u8;
                    p.offsets[0] = [1 + (next() % 3) as i32, 1, 1, 2 + (next() % 3) as i32];
                    p
                })
                .collect();
            let mut expected = plane.clone();
            for ctb_y in 0..h_ctbs {
                for ctb_x in 0..w_ctbs {
                    let p = sao[ctb_y * w_ctbs + ctb_x];
                    let (x0, y0) = (ctb_x * ctb, ctb_y * ctb);
                    let (w, h) = (ctb.min(width - x0), ctb.min(height - y0));
                    match p.type_idx[0] {
                        1 => apply_band(whole(&plane, width), &mut expected, width, x0, y0, w, h, p.extra[0], p.offsets[0], 8),
                        2 => apply_edge(whole(&plane, width), &mut expected, width, height, x0, y0, w, h, p.extra[0], p.offsets[0], 8),
                        _ => {}
                    }
                }
            }
            let mut actual = plane.clone();
            let c = Component {
                sao: &sao,
                pic_width_ctbs: w_ctbs,
                pic_height_ctbs: h_ctbs,
                cidx: 0,
                width,
                height,
                ctb_width: ctb,
                ctb_height: ctb,
                bit_depth: 8,
                bypass: None,
            };
            apply_component(&c, &mut actual, &mut Vec::new());
            assert_eq!(actual, expected, "round {round}");
        }
    }

    /// 8-bit band offset: `band = sample >> 3`, so samples 10/40/70/100
    /// land in bands 1/5/8/12. With `band_position = 1`, only bands
    /// 1..=4 (offsets[0..4]) are non-zero.
    #[test]
    fn band_offset_applies_to_matching_band_only() {
        let snapshot = [10u8, 40, 70, 100];
        let mut dst = snapshot;
        apply_band(whole(&snapshot, 4), &mut dst, 4, 0, 0, 4, 1, 1, [5, -3, 7, -9], 8);
        assert_eq!(dst, [15, 40, 70, 100]);
    }

    /// Band 0 never maps to an offset slot a 4-offset `sao_band_position`
    /// can reach when `band_position` keeps it out of range, so an
    /// out-of-band sample is left untouched even with all-nonzero
    /// offsets available elsewhere.
    #[test]
    fn band_offset_leaves_out_of_range_bands_untouched() {
        let snapshot = [0u8]; // band 0
        let mut dst = snapshot;
        apply_band(whole(&snapshot, 1), &mut dst, 1, 0, 0, 1, 1, 10, [1, 2, 3, 4], 8); // bands 10..=13 get the offsets
        assert_eq!(dst, [0]);
    }

    /// Clause 8.7.3.2: `edge_idx = 2 + sign(sample-n0) + sign(sample-n1)`.
    /// Sample 50 flanked by 60 and 70 (both higher) is a local minimum
    /// (`edge_idx == 0`), which the table maps to the *positive*
    /// `offsets[0]`. The two boundary samples have an out-of-picture
    /// neighbour and are left untouched.
    #[test]
    fn edge_offset_local_minimum_gets_first_category() {
        let snapshot = [60u8, 50, 70];
        let mut dst = snapshot;
        apply_edge(whole(&snapshot, 3), &mut dst, 3, 1, 0, 0, 3, 1, 0, [3, 99, 99, 7], 8);
        assert_eq!(dst, [60, 53, 70]);
    }

    /// Sample 70 flanked by 60 and 50 (both lower) is a local maximum
    /// (`edge_idx == 4`), which the table maps to the *negative*
    /// `-offsets[3]`.
    #[test]
    fn edge_offset_local_maximum_gets_last_category() {
        let snapshot = [60u8, 70, 50];
        let mut dst = snapshot;
        apply_edge(whole(&snapshot, 3), &mut dst, 3, 1, 0, 0, 3, 1, 0, [99, 99, 99, 4], 8);
        assert_eq!(dst, [60, 66, 50]);
    }
}
