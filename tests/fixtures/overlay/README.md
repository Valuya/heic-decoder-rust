# Overlay (`iovl`) fixtures

Assembled from this repository's own fixtures: the coded images are copied out
of `422/pattern8.heic` (8-bit 4:2:2), `alpha/lossless444.heic` (8-bit 4:4:4),
`alpha/mono8.heic` (8-bit monochrome) and `alpha/lossless420.heic` (10-bit
4:2:0), which HM encoded from original synthetic pixels. The fixtures are
covered by this repository's MIT OR Apache-2.0 licence; no third-party image is
bundled. Reproduce, without an encoder:

```sh
python3 scripts/generate_overlay_fixtures.py            # writes the two files
python3 scripts/generate_overlay_fixtures.py --libheif  # also checks them with pillow-heif
```

- `overlay.heic`: 100×80 canvas, 16-bit fields, fill RGBA `1234 abcd 8000 ffff`,
  turned three times anticlockwise (`irot` 3) to 80×100. Inputs in order: the
  4:2:2 picture at (−10, −6), cut by the top-left corner; the 4:4:4 picture,
  itself turned and mirrored, at (50, 30), cut by the bottom-right corner; the
  monochrome picture at (20, 60); the 4:2:2 picture again as another item, with
  the monochrome picture as its 8-bit alpha, at (30, 10); the first item again,
  entirely off the canvas, at (200, 0).
- `overlay-wide.heic`: 96×72 canvas, 32-bit fields, fill `ffff 0000 4321 0000`.
  The 10-bit picture first at (40, 8), so the canvas is 10-bit, then the 8-bit
  4:2:2 picture at (−20, −30), converted up to 10 bits.

`tests/overlay.rs` checks the decoded overlay against its inputs decoded on
their own and composed as libheif composes them. libheif 1.23.6 (pillow-heif
1.8.1) gives `overlay.heic` exactly as its own inputs composed that way. For
`overlay-wide.heic` it does too, within 3 levels of 1023 where its 8-bit input
lies: libheif converts that input to 10-bit RGB directly, which the generator,
seeing it through pillow-heif as 8-bit, can only approximate.
