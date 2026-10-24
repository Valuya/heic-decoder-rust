# Valuya fork of heic-decoder

A fork of [Multipad-cyber/heic-decoder-rust](https://github.com/Multipad-cyber/heic-decoder-rust)
at `a1afbf4` (0.1.1), used by [Thalamas](https://github.com/Valuya/thalamas)
to convert the HEIC photos parents and children upload. `main` is upstream
plus the changes below, each also offered upstream as its own pull request.
Once they are merged, Thalamas goes back to the crates.io release and this
fork has no reason to exist.

| Change | Branch | Upstream PR |
|---|---|---|
| Skip zero-length NAL units (Nokia `C034.heic` failed to decode) | `fix/skip-zero-length-nal-units` | [#1](https://github.com/Multipad-cyber/heic-decoder-rust/pull/1) |
| Read unlabelled colour as BT.601, as libheif does | `feat/bt601-for-unlabelled-colour` | [#2](https://github.com/Multipad-cyber/heic-decoder-rust/pull/2) |
| `decode_rgb8` / `decode_rgb8_with`: RGB bit-identical to `decode` + `to_rgb8` without the planar picture, optionally downscaled (12 MP: 151 → 53 MB full size, 35 MB at 1568 px) | `feat/decode-rgb8-fit` | [#3](https://github.com/Multipad-cyber/heic-decoder-rust/pull/3) |
| Branch-free CABAC decision and batched bypass bins (−16 to −26 % cycles, bit-identical) | `perf/branch-free-entropy-decoding` | [#4](https://github.com/Multipad-cyber/heic-decoder-rust/pull/4) |
| Overlay (`iovl`) images, composed in RGB as libheif composes them | `feat/overlay-images` | not yet offered; built on this fork's `main` |
| Compact sample planes (`u8`/`u16`), pooled tile pictures, SAO by CTB row: memory nearly flat across threads | `perf/compact-planes` | not yet offered; built on this fork's `main` |
| Inverse transforms as partial butterflies, bounded by the coded coefficients (−2.5 to −17 % cycles) | `perf/butterfly-transforms` | not yet offered; built on this fork's `main` |
| `decode_rgb8`: tiles decoded ahead on the pool, output rows written in parallel, rows through `Matrix::row` (4 threads ~2× faster) | `perf/tile-pipeline` | not yet offered; built on this fork's `main` |
| Deblocking filters a segment's four lines at once, rows read as slices (−1.2 % cycles) | `perf/deblock-lanes` | not yet offered; built on this fork's `main` |
| `decode_rgb8`: a rotated picture's output rows gathered with one loop each (−7.9 % cycles on rotated iPhone photos) | `perf/turned-scatter` | not yet offered; built on this fork's `main` |
| Intra prediction row by row, border read once (−1.7 % cycles) | `perf/intra-rows` | not yet offered; built on this fork's `main` |

Cycle figures: one thread on a P-core, the 10 sample photos, `perf stat`
in pairs of alternating order against the `main` before the change; every
change is bit-identical on 353 files (conformance, fuzz seeds, photos).

Each branch is upstream `a1afbf4` plus that change alone, and passes
`cargo test` and the conformance tests on its own. CI
(`.github/workflows/ci.yml`) runs rustfmt, clippy, the tests and the Nokia
conformance tests on every push.

Licence unchanged: MIT OR Apache-2.0.
