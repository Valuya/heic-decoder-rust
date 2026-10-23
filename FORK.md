# Valuya fork of heic-decoder

A fork of [Multipad-cyber/heic-decoder-rust](https://github.com/Multipad-cyber/heic-decoder-rust)
at `a1afbf4` (0.1.1), used by [Thalamas](https://github.com/Valuya/thalamas)
to convert the HEIC photos parents and children upload. `main` is upstream
plus the changes below, each also offered upstream as its own pull request.
Once they are merged, Thalamas goes back to the crates.io release and this
fork has no reason to exist.

| Change | Branch | Upstream PR |
|---|---|---|
| Skip zero-length NAL units (Nokia `C034.heic` failed to decode) | `fix/skip-zero-length-nal-units` | PR_NAL |
| Read unlabelled colour as BT.601, as libheif does | `feat/bt601-for-unlabelled-colour` | PR_BT601 |
| `decode_rgb8_fit`: a downscaled RGB image in bounded memory (24 MB for a 12 MP photo, 20 MB for 200 MP) | `feat/decode-rgb8-fit` | PR_FIT |
| Branch-free CABAC decision and batched bypass bins (−16 to −26 % cycles, bit-identical) | `perf/branch-free-entropy-decoding` | PR_PERF |

Each branch is upstream `a1afbf4` plus that change alone, and passes
`cargo test` and the conformance tests on its own. CI
(`.github/workflows/ci.yml`) runs rustfmt, clippy, the tests and the Nokia
conformance tests on every push.

Licence unchanged: MIT OR Apache-2.0.
