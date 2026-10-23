#!/usr/bin/env python3
"""Pack a Zoomify JPEG tile folder into a BigTIFF-based ZIF file.

This experimental revision attempts to match the observed layout of a
Zoomify Unlimited-generated ZIF: full-to-smallest linked IFDs, smallest-to-
largest JPEG payload, 2-byte alignment of individual JPEG tiles, and tile
indexes after the image data. It also matches the observed TIFF tags.

NOT verified against Zoomify Unlimited's proprietary loader. Test first.


No third-party Python packages are required. JPEG data is copied byte-for-byte.
Reference: https://zif.photo/ and https://github.com/lovasoa/ZIF/blob/master/SPECIFICATION.md

Usage:
    python zoomify_to_zif_zoomify_layout.py "C:\\images\\slide_zoomify" "C:\\images\\slide.zif"
    python zoomify_to_zif_zoomify_layout.py "C:\\images\\slide_zoomify"  # automatic filename
    python zoomify_to_zif_zoomify_layout.py --batch "C:\\images" "C:\\output"
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

DESCRIPTION = b"Created by Python Zoomify ZIF Converter\x00"  # 40 bytes
TILE_RE = re.compile(r"^(\d+)-(\d+)-(\d+)\.(jpg|jpeg)$", re.I)
JPEG_SOFS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def align8(n):
    return (n + 7) & ~7


def jpeg_info(path):
    """Read JPEG frame dimensions/channels and subsampling, without decoding pixels."""
    with path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError(f"Not a JPEG file: {path}")
        while True:
            c = f.read(1)
            if not c:
                raise ValueError(f"JPEG has no Start Of Frame marker: {path}")
            if c != b"\xff":
                raise ValueError(f"Invalid JPEG marker: {path}")
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if not marker:
                raise ValueError(f"Truncated JPEG: {path}")
            code = marker[0]
            if code in (0x00, 0xD8) or 0xD0 <= code <= 0xD9 or code == 0x01:
                if code == 0xD9:
                    raise ValueError(f"JPEG has no frame: {path}")
                continue
            raw = f.read(2)
            if len(raw) != 2:
                raise ValueError(f"Truncated JPEG segment: {path}")
            segment_len = struct.unpack(">H", raw)[0]
            if segment_len < 2:
                raise ValueError(f"Invalid JPEG segment size: {path}")
            if code in JPEG_SOFS:
                data = f.read(segment_len - 2)
                if len(data) < 6:
                    raise ValueError(f"Truncated JPEG frame: {path}")
                bits, h, w, components = struct.unpack(">BHHB", data[:6])
                if bits != 8 or components not in (1, 3):
                    raise ValueError(f"Only 8-bit grayscale/RGB JPEG tiles are supported: {path}")
                if len(data) < 6 + components * 3:
                    raise ValueError(f"Truncated JPEG component metadata: {path}")
                # Luma H/V sampling factors, e.g. 0x22 means 4:2:0.
                sampling = data[7] if components == 3 else None
                if components == 3 and (data[10] != 0x11 or data[13] != 0x11):
                    raise ValueError(f"Unsupported chroma subsampling: {path}")
                if components == 3 and sampling not in (0x11, 0x22):
                    raise ValueError(
                        f"JPEG subsampling {sampling:#04x} in {path} is not ZIF Baseline "
                        "(requires 4:4:4 or 4:2:0). Regenerate tiles with a compatible JPEG setting."
                    )
                subsampling = (1, 1) if sampling == 0x11 else (2, 2)
                return (w, h, components, subsampling)
            f.seek(segment_len - 2, os.SEEK_CUR)


def read_properties(root):
    path = root / "ImageProperties.xml"
    if not path.is_file():
        raise ValueError(f"Missing {path}")
    node = ET.parse(path).getroot()
    attrs = {k.upper(): v for k, v in node.attrib.items()}
    try:
        width = int(attrs["WIDTH"])
        height = int(attrs["HEIGHT"])
        size = int(attrs["TILESIZE"])
        declared_tiles = int(attrs.get("NUMTILES", "0"))
    except (ValueError, KeyError) as err:
        raise ValueError(f"Invalid ImageProperties.xml: {err}") from err
    if width < 1 or height < 1 or size < 16 or size % 16:
        raise ValueError("Invalid image dimensions or tile size (tile size must be divisible by 16)")
    if width > 0xFFFFFFFF or height > 0xFFFFFFFF:
        raise ValueError("This script requires image width and height to fit in 32-bit TIFF tags")
    return width, height, size, declared_tiles


@dataclass
class Level:
    number: int       # Zoomify level: 0 is smallest
    width: int
    height: int
    tile_size: int
    paths: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    offsets: list = field(default_factory=list)
    ifd_pos: int = 0
    description_pos: int = 0
    offsets_pos: int = 0
    counts_pos: int = 0
    channels: int = 0
    subsampling: tuple = (2, 2)

    @property
    def columns(self):
        return (self.width + self.tile_size - 1) // self.tile_size

    @property
    def rows(self):
        return (self.height + self.tile_size - 1) // self.tile_size


def analyze(root):
    width, height, tile_size, declared = read_properties(root)
    # Index tiles by their *actual* filenames, rather than guessing their
    # TileGroup from the cumulative tile number. Some copied/exported Zoomify
    # pyramids have otherwise valid files in different TileGroup directories.
    tiles = {}
    group_folders = []
    unexpected = []
    for folder in root.iterdir():
        if not folder.is_dir() or not re.fullmatch(r"TileGroup\d+", folder.name, re.I):
            continue
        group_folders.append(folder)
        for path in folder.iterdir():
            if not path.is_file():
                continue
            match = TILE_RE.fullmatch(path.name)
            if not match:
                if len(unexpected) < 8:
                    unexpected.append(f"{folder.name}/{path.name}")
                continue
            key = tuple(int(value) for value in match.group(1, 2, 3))
            if key in tiles:
                raise ValueError(
                    f"Duplicate tile {key} found at {tiles[key]} and {path}. "
                    "Cannot choose between two different JPEGs."
                )
            tiles[key] = path

    if not tiles:
        seen = ", ".join(unexpected) if unexpected else "(none)"
        raise ValueError(
            "No JPEG Zoomify tiles found in TileGroup folders. "
            f"Found {len(group_folders)} TileGroup folders; sample contents: {seen}. "
            "This converter requires JPEG (.jpg/.jpeg) Zoomify tiles."
        )
    level_nums = {level for level, x, y in tiles}
    nlevels = max(level_nums) + 1
    if level_nums != set(range(nlevels)):
        sample = ", ".join(f"{p.parent.name}/{p.name}" for _, p in sorted(tiles.items())[:8])
        raise ValueError(
            f"Zoomify level numbers are not continuous: {sorted(level_nums)}. "
            f"Expected levels 0..{nlevels - 1}. Sample files: {sample}. "
            "The tile pyramid may be incomplete, or it may not use standard Zoomify filenames."
        )

    # Zoomify levels are named from smallest (0) to full resolution (N-1).
    # libvips rounds up at each /2, as specified for ZIF pyramids.
    resolutions = [(width, height)]
    for _ in range(nlevels - 1):
        w, h = resolutions[-1]
        if w == h == 1:
            raise ValueError("More pyramid levels than image dimensions allow")
        resolutions.append(((w + 1) // 2, (h + 1) // 2))
    levels = [Level(nlevels - i - 1, w, h, tile_size)
              for i, (w, h) in enumerate(resolutions)]
    if max(levels[-1].width, levels[-1].height) > tile_size:
        raise ValueError("Zoomify pyramid is incomplete (no smallest one-tile level)")

    global_index = 0
    nonstandard_locations = 0
    for level in reversed(levels):
        for y in range(level.rows):
            for x in range(level.columns):
                key = (level.number, x, y)
                path = tiles.get(key)
                if path is None:
                    # Help diagnose a genuinely missing tile or a different
                    # naming convention instead of implying it must be in a
                    # particular TileGroup folder.
                    sample = ", ".join(
                        f"{p.parent.name}/{p.name}" for _, p in
                        sorted(tiles.items())[:8]
                    )
                    group0 = root / "TileGroup0"
                    contents = (", ".join(p.name for p in
                                sorted(group0.iterdir())[:8])
                                if group0.is_dir() else "TileGroup0 not present")
                    raise ValueError(
                        f"Missing required tile {level.number}-{x}-{y}.jpg "
                        f"(searched {len(tiles):,} tile filenames in "
                        f"{len(group_folders)} TileGroup folders).\n"
                        f"  First discovered tiles: {sample}\n"
                        f"  TileGroup0 contents: {contents}\n"
                        "  If the source was created with libvips, check "
                        "whether dzsave finished successfully and whether "
                        "the directory contains the complete image pyramid."
                    )
                expected_group = f"TileGroup{global_index // 256}"
                if path.parent.name.lower() != expected_group.lower():
                    nonstandard_locations += 1
                size = path.stat().st_size
                if size <= 4:
                    raise ValueError(f"Empty or damaged tile: {path}")
                level.paths.append(path)
                level.sizes.append(size)
                global_index += 1
        _, _, level.channels, level.subsampling = jpeg_info(level.paths[0])
    if global_index != len(tiles):
        raise ValueError(
            f"Found {len(tiles):,} JPEG tiles but expected {global_index:,}. "
            "Some tiles have unexpected coordinates or extra levels."
        )
    if declared and declared != global_index:
        raise ValueError(f"ImageProperties.xml says {declared} tiles, but computed {global_index}")
    if nonstandard_locations:
        print(f"Note: {nonstandard_locations:,} tiles are outside their usual "
              "TileGroup; found them by filename instead.")
    return levels


def u16(*values):
    return struct.pack("<" + "H" * len(values), *values)


def u32(value):
    return struct.pack("<I", value)


def u64(value):
    return struct.pack("<Q", value)


def ifd_entries(level):
    """Use the tag set/data types observed in Zoomify Unlimited output."""
    n = len(level.paths)
    entries = [
        (254, 4, 1, u32(0)),  # NewSubfileType: official reference uses 0 for all IFDs
        (256, 4, 1, u32(level.width)),
        (257, 4, 1, u32(level.height)),
        (258, 3, level.channels, u16(*([8] * level.channels))),
        (259, 3, 1, u16(7)),  # independent JPEG tiles
        (262, 3, 1, u16(6 if level.channels == 3 else 1)),
        (270, 2, len(DESCRIPTION), u64(level.description_pos)),
        (274, 3, 1, u16(1)),  # Orientation top-left
        (277, 3, 1, u16(level.channels)),
        (284, 3, 1, u16(1)),
        (322, 3, 1, u16(level.tile_size)),  # SHORT instead of LONG
        (323, 3, 1, u16(level.tile_size)),
        (324, 16, n, u64(level.offsets_pos if n > 1 else level.offsets[0])),
        (325, 4, n, (u64(level.counts_pos) if n > 2 else
                     b"".join(u32(v) for v in level.sizes))),
        (32997, 4, 1, u32(1)),  # ImageDepth=1, matching Zoomify reference
    ]
    if level.channels == 3:
        entries.append((530, 3, 2, u16(*level.subsampling)))
    return sorted(entries)


def write_u64_array(f, values):
    # Avoid creating a potentially massive Python tuple or writing one number at a time.
    for start in range(0, len(values), 4096):
        chunk = values[start:start + 4096]
        f.write(struct.pack("<" + "Q" * len(chunk), *chunk))


def write_u32_array(f, values):
    for start in range(0, len(values), 4096):
        chunk = values[start:start + 4096]
        f.write(struct.pack("<" + "I" * len(chunk), *chunk))


def pack(root, output, overwrite=False):
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists (use --overwrite to replace): {output}")
    levels = analyze(root)
    total_tiles = sum(len(level.paths) for level in levels)
    print(f"{root.name}: {levels[0].width} x {levels[0].height}, "
          f"{len(levels)} levels, {total_tiles:,} JPEG tiles")

    # Match the reference file: reserve IFDs only at the start.
    # Each RGB level has 16 tags, so 8 + 16*20 + 8 = 336 bytes per IFD.
    position = 16
    for level in levels:
        level.ifd_pos = position
        entry_count = 16 if level.channels == 3 else 15
        position += 8 + 20 * entry_count + 8

    if position & 1:
        raise RuntimeError("IFD block not word-aligned")
    start_of_tiles = position

    # JPEG tiles are stored smallest-resolution-first (reverse IFD order).
    # Unlike the old script, pad EACH JPEG with one byte when necessary.
    # The reference Zoomify tile starts all have even offsets.
    for level in reversed(levels):
        for size in level.sizes:
            level.offsets.append(position)
            position += size
            if position & 1:
                position += 1

    metadata_start = align8(position)
    position = metadata_start
    for level in levels:
        level.description_pos = position
        position += len(DESCRIPTION)
        position = align8(position)
        n = len(level.paths)
        if any(size > 0xFFFFFFFF for size in level.sizes):
            raise ValueError("Tile byte count too large for 32-bit Zoomify convention")
        if n > 1:
            level.offsets_pos = position
            position += 8 * n
        if n > 2:
            level.counts_pos = position
            position += 4 * n
        position = align8(position)
    expected_file_size = position

    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(output.name + ".part")
    if part.exists():
        raise FileExistsError(f"Temporary output already exists: {part}")
    try:
        with part.open("xb") as f:
            f.write(b"II" + u16(43) + u16(8) + u16(0) + u64(levels[0].ifd_pos))
            for i, level in enumerate(levels):
                assert f.tell() == level.ifd_pos
                entries = ifd_entries(level)
                f.write(u64(len(entries)))
                for tag, typ, count, value in entries:
                    f.write(struct.pack("<HHQ", tag, typ, count))
                    f.write(value.ljust(8, b"\0"))
                f.write(u64(levels[i + 1].ifd_pos if i + 1 < len(levels) else 0))

            if f.tell() != start_of_tiles:
                raise RuntimeError("IFD size prediction was incorrect")

            for level in reversed(levels):
                print(f"  Level {level.number}: {level.width} x {level.height}; "
                      f"{len(level.paths):,} tiles", flush=True)
                for idx, (path, size, offset) in enumerate(zip(level.paths, level.sizes, level.offsets)):
                    x = idx % level.columns
                    y = idx // level.columns
                    if f.tell() != offset:
                        raise RuntimeError("Incorrect tile offset during write")
                    actual_w, actual_h, channels, subs = jpeg_info(path)
                    wanted_w = min(level.tile_size, level.width - x * level.tile_size)
                    wanted_h = min(level.tile_size, level.height - y * level.tile_size)
                    if (actual_w, actual_h) != (wanted_w, wanted_h):
                        raise ValueError(f"Incorrect tile dimensions in {path}: "
                                         f"{actual_w}x{actual_h}, expected {wanted_w}x{wanted_h}")
                    if channels != level.channels or subs != level.subsampling:
                        raise ValueError(f"Inconsistent JPEG channels/subsampling in {path}")
                    if path.stat().st_size != size:
                        raise RuntimeError(f"Tile changed during conversion: {path}")
                    with path.open("rb") as tile_file:
                        shutil.copyfileobj(tile_file, f, length=1024 * 1024)
                    if f.tell() != offset + size:
                        raise RuntimeError(f"Tile changed during conversion: {path}")
                    if size & 1:
                        f.write(b"\0")  # even-byte tile alignment

            f.write(b"\0" * (metadata_start - f.tell()))
            for level in levels:
                assert f.tell() == level.description_pos
                f.write(DESCRIPTION)
                f.write(b"\0" * (align8(f.tell()) - f.tell()))
                if len(level.paths) > 1:
                    assert f.tell() == level.offsets_pos
                    write_u64_array(f, level.offsets)
                if len(level.paths) > 2:
                    assert f.tell() == level.counts_pos
                    write_u32_array(f, level.sizes)
                f.write(b"\0" * (align8(f.tell()) - f.tell()))
            if f.tell() != expected_file_size:
                raise RuntimeError("Incorrect final file size")
        os.replace(part, output)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    print(f"Created: {output} ({output.stat().st_size / (1024 * 1024):,.1f} MiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", help="Zoomify image folder, or parent folder with --batch")
    parser.add_argument("output", nargs="?", help="Output .zif file, or output folder with --batch")
    parser.add_argument("--batch", action="store_true", help="Convert every immediate subfolder containing ImageProperties.xml")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing .zif")
    args = parser.parse_args()

    if args.input is None:
        args.input = input("Path to Zoomify tile folder: ").strip().strip('"')
        if not args.input:
            parser.error("No input folder given")
    source = Path(args.input).expanduser()
    if args.batch:
        if not source.is_dir():
            parser.error(f"Not a directory: {source}")
        outfolder = Path(args.output).expanduser() if args.output else source
        candidates = [p for p in sorted(source.iterdir()) if (p / "ImageProperties.xml").is_file()]
        if not candidates:
            parser.error("No Zoomify folders containing ImageProperties.xml found")
        failed = 0
        for folder in candidates:
            try:
                pack(folder, outfolder / (folder.name + ".zif"), args.overwrite)
            except (OSError, ValueError, RuntimeError) as ex:
                print(f"ERROR [{folder.name}]: {ex}", file=sys.stderr)
                failed += 1
        return 1 if failed else 0
    else:
        output = Path(args.output).expanduser() if args.output else source.with_name(source.name + ".zif")
        try:
            pack(source, output, args.overwrite)
        except (OSError, ValueError, RuntimeError) as ex:
            print(f"ERROR: {ex}", file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
