#!/usr/bin/env python3
"""Assemble the overlay (`iovl`) fixtures from this repository's own fixtures.

Usage: python3 scripts/generate_overlay_fixtures.py [--libheif]

No encoder is needed: the coded images are copied out of the HM-encoded
fixtures in tests/fixtures/422 and tests/fixtures/alpha. With --libheif (needs
pillow-heif and numpy) each overlay is decoded by libheif and checked against
libheif's own inputs composed as tests/overlay.rs composes ours.
"""
import argparse
import io
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / 'tests' / 'fixtures'
OUT = FIXTURES / 'overlay'


def u16(value):
    return struct.pack('>H', value)


def u32(value):
    return struct.pack('>I', value)


def box(kind, body):
    return u32(len(body) + 8) + kind.encode('ascii') + body


def full(kind, body, version=0, flags=0):
    return box(kind, bytes([version]) + flags.to_bytes(3, 'big') + body)


def coded(path):
    """The hvcC box and coded sample of a one-item fixture."""
    data = (FIXTURES / path).read_bytes()
    at = data.index(b'hvcC') - 4
    return data[at:at + int.from_bytes(data[at:at + 4], 'big')], data[data.index(b'mdat') + 4:]


def descriptor(fill, width, height, offsets, wide):
    field = (lambda v: struct.pack('>i', v)) if wide else (lambda v: struct.pack('>h', v))
    size = u32 if wide else u16
    return bytes([0, int(wide)]) + b''.join(u16(v) for v in fill) + size(width) + size(height) + b''.join(field(x) + field(y) for x, y in offsets)


def container(destination, items, properties, refs, primary):
    """items: (id, type, payload, property indices, hidden)."""
    associations = b''.join(u16(i) + bytes([len(props), *[0x80 | p for p in props]]) for i, _, _, props, _ in items)
    iprp = box('iprp', box('ipco', b''.join(properties)) + full('ipma', u32(len(items)) + associations))
    iinf = full('iinf', u16(len(items)) + b''.join(full('infe', u16(i) + u16(0) + kind + b'\0', 2, int(hidden)) for i, kind, _, _, hidden in items))
    iref = full('iref', b''.join(box(kind, u16(source) + u16(len(targets)) + b''.join(u16(t) for t in targets)) for kind, source, targets in refs))
    ftyp = box('ftyp', b'heic' + u32(0) + b'mif1heic')

    def meta(offset):
        locations = b''
        for i, _, payload, _, _ in items:
            locations += u16(i) + u16(0) + u16(1) + u32(offset) + u32(len(payload))
            offset += len(payload)
        iloc = full('iloc', bytes([0x44, 0]) + u16(len(items)) + locations)
        hdlr = full('hdlr', u32(0) + b'pict' + bytes(12) + b'\0')
        return full('meta', hdlr + full('pitm', u16(primary)) + iinf + iloc + iprp + iref)

    destination.write_bytes(ftyp + meta(len(ftyp) + len(meta(0)) + 8) + box('mdat', b''.join(p for _, _, p, _, _ in items)))


def ispe(width, height):
    return full('ispe', u32(width) + u32(height))


# (name, canvas fill RGBA, canvas size, 32-bit fields, overlay transforms, placements of item ids at offsets)
CASES = {
    # 8-bit canvas: a 4:2:2 image cut by the top-left corner; a rotated and
    # mirrored 4:4:4 image cut by the bottom-right corner; a monochrome one; the
    # 4:2:2 picture again, as another item, with an 8-bit alpha plane; the first
    # item a second time, entirely off the canvas. The overlay is then turned.
    'overlay': dict(fill=(0x1234, 0xABCD, 0x8000, 0xFFFF), size=(100, 80), wide=False, turn=3,
                    placements=[(1, (-10, -6)), (2, (50, 30)), (3, (20, 60)), (4, (30, 10)), (1, (200, 0))]),
    # 32-bit fields; the first input is 10-bit, so the canvas is, and the 8-bit
    # 4:2:2 input is converted up to it.
    'overlay-wide': dict(fill=(0xFFFF, 0x0000, 0x4321, 0x0000), size=(96, 72), wide=True, turn=0,
                         placements=[(6, (40, 8)), (1, (-20, -30))]),
}


def build(name, case):
    pattern8 = coded('422/pattern8.heic')
    lossless444 = coded('alpha/lossless444.heic')
    mono8 = coded('alpha/mono8.heic')
    lossless420 = coded('alpha/lossless420.heic')
    width, height = case['size']
    properties = [
        pattern8[0], ispe(64, 64),  # 1, 2
        lossless444[0], box('irot', b'\1'), box('imir', b'\1'),  # 3, 4, 5
        mono8[0],  # 6
        full('auxC', b'urn:mpeg:hevc:2015:auxid:1\0'),  # 7
        lossless420[0],  # 8
        ispe(width, height),  # 9
        box('irot', bytes([case['turn']])),  # 10
    ]
    ids = [item for item, _ in case['placements']]
    overlay = descriptor(case['fill'], width, height, [offset for _, offset in case['placements']], case['wide'])
    items = [
        (1, b'hvc1', pattern8[1], [1, 2], False),
        (2, b'hvc1', lossless444[1], [3, 2, 4, 5], False),
        (3, b'hvc1', mono8[1], [6, 2], False),
        (4, b'hvc1', pattern8[1], [1, 2], False),
        (5, b'hvc1', mono8[1], [6, 2, 7], True),
        (6, b'hvc1', lossless420[1], [8, 2], False),
        (7, b'iovl', overlay, [9] + ([10] if case['turn'] else []), False),
    ]
    used = set(ids) | ({5} if 4 in ids else set())
    items = [item for item in items if item[0] in used or item[0] == 7]
    refs = [('dimg', 7, ids)] + ([('auxl', 5, [4])] if 4 in ids else [])
    container(OUT / f'{name}.heic', items, properties, refs, 7)


def check_with_libheif(name, case):
    import numpy as np
    import pillow_heif

    data = (OUT / f'{name}.heic').read_bytes()
    pitm = data.index(b'pitm') + 8

    def item(item_id, depth):
        patched = data[:pitm] + u16(item_id) + data[pitm + 2:]
        heif = pillow_heif.open_heif(io.BytesIO(patched), convert_hdr_to_8bit=False)
        array = np.asarray(heif).astype(np.int64)
        if array.ndim == 2:  # monochrome
            array = np.repeat(array[..., None], 3, axis=2)
        # pillow-heif scales deeper images to 16 bits. An 8-bit input in a
        # 10-bit canvas is only approximated here: libheif converts it to
        # 10-bit RGB directly, which pillow-heif cannot be asked for.
        return array >> (16 - depth) if heif.info['bit_depth'] > 8 else array << (depth - 8)

    first = 6 if name == 'overlay-wide' else 1
    depth = 10 if first == 6 else 8
    overlay = item(7, depth)
    width, height = case['size']
    canvas = np.empty((height, width, 3), np.int64)
    canvas[:] = [v >> (16 - depth) for v in case['fill'][:3]]
    for item_id, (dx, dy) in case['placements']:
        image = item(item_id, depth)
        ih, iw = image.shape[:2]
        x0, y0, x1, y1 = max(dx, 0), max(dy, 0), min(dx + iw, width), min(dy + ih, height)
        if x1 <= x0 or y1 <= y0:
            continue
        source = image[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        if source.shape[2] == 4:
            alpha = source[..., 3:]
            canvas[y0:y1, x0:x1] = (source[..., :3] * alpha + canvas[y0:y1, x0:x1] * ((1 << depth) - 1 - alpha)) // ((1 << depth) - 1)
        else:
            canvas[y0:y1, x0:x1] = source[..., :3]
    canvas = np.rot90(canvas, case['turn'])
    difference = np.abs(canvas - overlay[..., :3])
    return dict(libheif=pillow_heif.libheif_version(), shape=list(overlay.shape), max_difference_to_composed_inputs=int(difference.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libheif', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    manifest = []
    for name, case in CASES.items():
        build(name, case)
        entry = dict(name=name, canvas=list(case['size']), fill=list(case['fill']), wide=case['wide'], turns=case['turn'], placements=case['placements'])
        if args.libheif:
            entry['libheif_check'] = check_with_libheif(name, case)
        manifest.append(entry)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
