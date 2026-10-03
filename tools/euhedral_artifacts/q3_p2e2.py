"""Losslessly transcode the Q3 tensors of a Q3 artifact to the P2E2 layout (the compressed Q3 artifact).

P2E2 ("row-split-p2e2-v1") stores the same Q3G64_F16S values in less memory: the signed 3-bit codes
are entropy coded as a fixed 2-bit primary plane plus a short payload stream, and the FP16 scale
plane is kept byte for byte. docs/COMPRESSED_Q3.md specifies the layout. Every transcoded tensor is
decoded back and compared byte for byte with its source before the output is published.
"""

from __future__ import annotations

import os
from pathlib import Path
import struct
import tempfile
from typing import Any

import numpy as np

from euhedral_artifacts.edrl import (FORMAT_Q3, HEADER_FORMAT, LAYOUT_P2E2, LAYOUT_ROW_SPLIT, align_up,
                                     encode_descriptors, fail, read_table)

SLICE = 1024
PAYLOAD_PAD_WORDS = 80
ROW_CHUNK_CODES = 1 << 25
BIG_VALUES = np.array([-3, -2, 2, 3], dtype=np.int8)


def row_split_q3_size(rows: int, k: int) -> int:
    groups = align_up(k, 128) // 64
    return align_up(rows * groups * 24, 256) + rows * groups * 2


def p2e2_offsets(rows: int, k: int) -> tuple[int, int, int]:
    """Byte offsets of the row-base, scale and payload planes."""
    row_base = align_up(rows * k // 4, 256)
    scales = align_up(row_base + rows * 4, 256)
    payload = align_up(scales + rows * (k // 64) * 2, 256)
    return row_base, scales, payload


def p2e2_size(rows: int, k: int, units: int) -> int:
    return p2e2_offsets(rows, k)[2] + 4 * ((units + 15) // 16 + PAYLOAD_PAD_WORDS)


def eligible(shape: tuple[int, ...], fmt: int, layout: int) -> bool:
    return fmt == FORMAT_Q3 and layout == LAYOUT_ROW_SPLIT and len(shape) == 2 and shape[1] % SLICE == 0


def unpack_q3(codes_plane: np.ndarray, rows: int, k: int) -> np.ndarray:
    """Row-split Q3 code bytes [rows * k/64 * 24] -> signed codes int8 [rows, k]."""
    groups = k // 64
    bits = np.unpackbits(codes_plane.reshape(rows, groups, 24), axis=2, bitorder="little")
    fields = bits.reshape(rows, groups, 64, 3)
    unsigned = (fields[..., 0] | (fields[..., 1] << 1) | (fields[..., 2] << 2)).astype(np.int8)
    return np.where(unsigned >= 4, unsigned - 8, unsigned).astype(np.int8).reshape(rows, k)


def pack_q3(codes: np.ndarray) -> bytes:
    """Signed codes int8 [rows, k] -> row-split Q3 code bytes (LSB-first 3-bit stream per group)."""
    rows, k = codes.shape
    unsigned = (codes.astype(np.int16) & 7).astype(np.uint8).reshape(rows, k // 64, 64, 1)
    bits = (unsigned >> np.arange(3, dtype=np.uint8)) & 1
    return np.packbits(bits.reshape(rows, k // 64, 192), axis=2, bitorder="little").tobytes()


def pack_units(values: np.ndarray, per_word: int, width: int) -> np.ndarray:
    """Pack small unsigned values LSB-first, `per_word` to a little-endian u32 word."""
    words = (len(values) + per_word - 1) // per_word
    padded = np.zeros(words * per_word, dtype=np.uint32)
    padded[: len(values)] = values
    shifts = (width * np.arange(per_word, dtype=np.uint32))
    return (padded.reshape(words, per_word) << shifts).sum(axis=1, dtype=np.uint64).astype(np.uint32)


def unpack_units(words: np.ndarray, count: int, per_word: int, width: int) -> np.ndarray:
    shifts = (width * np.arange(per_word, dtype=np.uint32))
    values = (words[:, None] >> shifts) & ((1 << width) - 1)
    return values.ravel()[:count].astype(np.uint8)


def encode(source: bytes | np.ndarray, rows: int, k: int) -> bytes:
    """Row-split Q3 tensor bytes -> P2E2 tensor bytes."""
    data = np.frombuffer(source, dtype=np.uint8) if isinstance(source, (bytes, bytearray)) else source
    if k % SLICE != 0:
        fail(f"P2E2 requires K to be a multiple of {SLICE}, got {k}")
    if len(data) != row_split_q3_size(rows, k):
        fail("source size does not match the Q3 row-split geometry")
    groups = k // 64
    scale_offset = align_up(rows * groups * 24, 256)
    primary = np.empty((rows, k // 16), dtype=np.uint32)
    row_base = np.empty(rows, dtype=np.uint32)
    payload_parts: list[np.ndarray] = []
    units = 0
    chunk = max(1, ROW_CHUNK_CODES // k)
    shifts = 2 * np.arange(16, dtype=np.uint32)
    for begin in range(0, rows, chunk):
        end = min(rows, begin + chunk)
        codes = unpack_q3(data[begin * groups * 24:end * groups * 24], end - begin, k)
        if (codes == -4).any():
            fail("code -4 cannot be represented; the Q3 converter never emits it")
        symbols = np.full(codes.shape, 3, dtype=np.uint32)
        symbols[codes == -1] = 0
        symbols[codes == 0] = 1
        symbols[codes == 1] = 2
        primary[begin:end] = (symbols.reshape(end - begin, k // 16, 16) << shifts).sum(axis=2, dtype=np.uint64)
        big = symbols == 3
        counts = big.sum(axis=1)
        row_base[begin:end] = units + np.concatenate([[0], np.cumsum(counts)[:-1]])
        values = codes[big]
        payload_parts.append(np.searchsorted(BIG_VALUES, values).astype(np.uint8))
        units += int(counts.sum())
        if units >= 1 << 32:
            fail("payload exceeds the 32-bit unit index")
    payload_values = np.concatenate(payload_parts) if payload_parts else np.zeros(0, np.uint8)
    payload = pack_units(payload_values, 16, 2)
    base_offset, out_scale_offset, payload_offset = p2e2_offsets(rows, k)
    out = np.zeros(p2e2_size(rows, k, units), dtype=np.uint8)
    out[: primary.nbytes] = primary.astype("<u4").view(np.uint8).ravel()
    out[base_offset:base_offset + rows * 4] = row_base.astype("<u4").view(np.uint8)
    out[out_scale_offset:out_scale_offset + rows * groups * 2] = data[scale_offset:scale_offset + rows * groups * 2]
    out[payload_offset:payload_offset + payload.nbytes] = payload.astype("<u4").view(np.uint8)
    return out.tobytes()


def decode(tensor: bytes | np.ndarray, rows: int, k: int) -> bytes:
    """P2E2 tensor bytes -> row-split Q3 tensor bytes (independent of the encoder's bookkeeping)."""
    data = np.frombuffer(tensor, dtype=np.uint8) if isinstance(tensor, (bytes, bytearray)) else tensor
    base_offset, scale_offset, payload_offset = p2e2_offsets(rows, k)
    if len(data) < payload_offset + 4 * PAYLOAD_PAD_WORDS or (len(data) - payload_offset) % 4:
        fail("P2E2 tensor is too small for its shape")
    groups = k // 64
    primary = data[: rows * k // 4].view("<u4").reshape(rows, k // 16)
    row_base = data[base_offset:base_offset + rows * 4].view("<u4")
    payload = data[payload_offset:].view("<u4")
    capacity = (len(payload) - PAYLOAD_PAD_WORDS) * 16
    out = np.zeros(row_split_q3_size(rows, k), dtype=np.uint8)
    chunk = max(1, ROW_CHUNK_CODES // k)
    shifts = 2 * np.arange(16, dtype=np.uint32)
    for begin in range(0, rows, chunk):
        end = min(rows, begin + chunk)
        symbols = ((primary[begin:end, :, None] >> shifts) & 3).reshape(end - begin, k).astype(np.int8)
        codes = symbols - 1
        big = symbols == 3
        counts = big.sum(axis=1)
        first = int(row_base[begin])
        expected = first + np.concatenate([[0], np.cumsum(counts)[:-1]])
        if not np.array_equal(row_base[begin:end], expected):
            fail("row base plane does not match the primary plane")
        total = int(counts.sum())
        if first + total > capacity:
            fail("payload plane is shorter than the primary plane requires")
        units = unpack_units(payload[first // 16:(first + total + 15) // 16 + 1], first % 16 + total, 16, 2)
        codes[big] = BIG_VALUES[units[first % 16:]]
        out[begin * groups * 24:end * groups * 24] = np.frombuffer(pack_q3(codes), dtype=np.uint8)
    out_scale = align_up(rows * groups * 24, 256)
    out[out_scale:out_scale + rows * groups * 2] = data[scale_offset:scale_offset + rows * groups * 2]
    return out.tobytes()


def transcode(input_path: Path, output_path: Path, force: bool = False) -> dict[str, Any]:
    """Writes the P2E2 artifact of the Q3 artifact `input_path`; returns its manifest fields."""
    if output_path.exists() and not force:
        fail(f"output already exists; pass --force: {output_path}")
    with input_path.open("rb") as source:
        header, metadata, objects = read_table(source)
        data_offset = struct.unpack(HEADER_FORMAT, header)[7]
        planned = []
        cursor = data_offset
        for obj in objects:
            out = dict(obj)
            if eligible(obj["shape"], obj["format"], obj["layout"]):
                out["layout"] = LAYOUT_P2E2
                out["bytes"] = None
            planned.append(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        stats = {"tensors": 0, "source_bytes": 0, "p2e2_bytes": 0}
        try:
            with temporary.open("w+b") as sink:
                sink.write(b"\0" * data_offset)
                for index, (obj, out) in enumerate(zip(objects, planned), start=1):
                    source.seek(obj["offset"])
                    payload = source.read(obj["bytes"])
                    if len(payload) != obj["bytes"]:
                        fail(f"payload of {obj['name']} is truncated")
                    if out["bytes"] is None:
                        rows, k = obj["shape"]
                        payload_out = encode(payload, rows, k)
                        if decode(payload_out, rows, k) != payload:
                            fail(f"P2E2 round trip of {obj['name']} is not exact")
                        stats["tensors"] += 1
                        stats["source_bytes"] += len(payload)
                        stats["p2e2_bytes"] += len(payload_out)
                        payload = payload_out
                        out["bytes"] = len(payload)
                    out["offset"] = align_up(cursor, 256)
                    sink.seek(out["offset"])
                    sink.write(payload)
                    cursor = out["offset"] + len(payload)
                    if index % 32 == 0 or index == len(objects):
                        print(f"transcoded {index}/{len(objects)} {obj['name']}", flush=True)
                sink.truncate(cursor)
                table = encode_descriptors(planned)
                if len(table) != data_offset - struct.unpack(HEADER_FORMAT, header)[4]:
                    fail("tensor table size changed")
                sink.seek(0)
                sink.write(header)
                sink.write(metadata)
                sink.write(table)
                sink.flush()
                os.fsync(sink.fileno())
            os.replace(temporary, output_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return {
        "object_count": len(objects),
        "p2e2_tensor_count": stats["tensors"],
        "p2e2_source_bytes": stats["source_bytes"],
        "p2e2_bytes": stats["p2e2_bytes"],
        "payload_bytes": sum(out["bytes"] for out in planned),
        "file_bytes": cursor,
        "round_trip": "exact",
    }
