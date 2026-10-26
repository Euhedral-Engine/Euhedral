"""Writes tiny.edrl, the small artifact the synthetic tests parse.

Regenerate from anywhere with:

    python3 rust/crates/edrl/tests/fixtures/make_fixture.py

The container comes from tools/euhedral_artifacts/edrl.py and tokenizer_section.py (the writers the converter uses),
so the checked-in bytes are what the Python side produces. The payload sizes below are written out here from the
layout rules, not taken from the Rust crate, so the tests compare two independent implementations.

tiny.edrl holds one tensor of each stored layout, the four tensors of a trellis object, a metadata region that
ends in a draft depth of 4, and a tokenizer section. Each payload is filled with its tensor's 1-based index, except
the trellis `.segments` payload, which is the little-endian int32 values 128 and 256.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "tools"))

from euhedral_artifacts import tokenizer_section  # noqa: E402
from euhedral_artifacts.edrl import (HEADER_SIZE, MAGIC, VERSION, ObjectPlan, align_up, assign_offsets,  # noqa: E402
                                     encode_table, utf8)

TOKENIZER_FILES = (
    '{"version":"1.0"}',
    '{"chat_template":null}',
    "{{ messages[0].content }} é",
    '{"bos_token_id":1,"eos_token_id":2,"pad_token_id":3}',
)


def metadata() -> bytes:
    """A four-layer configuration (full attention on layer 3) ending in the MTP draft depth."""
    out = bytearray()
    out += struct.pack(">12i", 1000, 256, 4, 8, 2, 32, 512, 4, 8, 16, 16, 4)
    out += struct.pack(">3d", 1e-6, 1e7, 0.25)
    out += struct.pack(">i", 4096)
    activation = utf8("silu", "hidden activation")
    out += struct.pack(">i", len(activation)) + activation
    layer_types = [1, 1, 1, 0]
    out += struct.pack(">i", len(layer_types)) + b"".join(struct.pack(">i", t) for t in layer_types)
    out += struct.pack(">4i", 0, 0, 0, 0)
    out += struct.pack(">BB", 0, 1)
    out += struct.pack(">i", 1)  # MTP layer count
    out += struct.pack(">i", 4)  # draft depth
    return bytes(out)


def grouped_q3(n: int, k: int) -> int:
    k_pad = align_up(k, 128)
    groups = k_pad // 64
    base = align_up(n * groups * 24, 256)
    return base + n * groups * 2  # no high plane for Q3


def nvfp4(n: int, k: int, sd4: bool) -> int:
    k_pad = align_up(k, 128)
    scales = align_up(n * k_pad // 2, 256)
    row_scales = k_pad // (32 if sd4 else 16)
    trailer = align_up(scales + n * row_scales, 256)
    return trailer + (16 if sd4 else 0) + 4


def p2e2_minimum(n: int, k: int) -> tuple[int, int]:
    """(smallest valid payload size, offset where the payload plane starts)."""
    row_base = align_up(n * k // 4, 256)
    scales = align_up(row_base + n * 4, 256)
    payload = align_up(scales + n * k // 32, 256)
    return payload + 4 * 80, payload


def build() -> bytes:
    contiguous, k128, p2e2, sd4, tile16 = ("contiguous-le-v1", "row-split-k128-v1", "row-split-p2e2-v1",
                                           "row-split-k128-sd4-v1", "trellis-tile16-v1")
    p2e2_min, _ = p2e2_minimum(2, 1024)
    segments = struct.pack("<2i", 128, 256)
    specs = [
        ("embed", (4, 8), "BF16", "BF16", contiguous, 4 * 8 * 2),
        ("norm", (8,), "BF16", "FP32", contiguous, 8 * 4),
        ("ids", (3,), "INT32", "I32", contiguous, 3 * 4),
        ("q3", (2, 128), "BF16", "Q3G64_F16S", k128, grouped_q3(2, 128)),
        ("nvfp4", (2, 128), "BF16", "NVFP4", k128, nvfp4(2, 128, False)),
        ("nvfp4.sd4", (2, 128), "BF16", "NVFP4", sd4, nvfp4(2, 128, True)),
        ("p2e2", (2, 1024), "BF16", "Q3G64_F16S", p2e2, p2e2_min + 4 * 20),
        ("trellis", (8, 16, 24), "BF16", "TRELLIS", tile16, 8 * 16 * 24 * 2),
        ("trellis.suh", (2, 128), "FP16", "FP16", contiguous, 2 * 128 * 2),
        ("trellis.svh", (256,), "FP16", "FP16", contiguous, 256 * 2),
        ("trellis.segments", (2,), "INT32", "I32", contiguous, len(segments)),
    ]
    payloads = []
    plans = []
    for index, (name, shape, dtype, fmt, layout, size) in enumerate(specs, start=1):
        payloads.append(segments if name == "trellis.segments" else bytes([index]) * size)
        plans.append(ObjectPlan(name, shape, dtype, fmt, layout, size, writer=None))

    meta = metadata()
    data_base = HEADER_SIZE + len(meta) + len(encode_table(plans))
    file_size = assign_offsets(plans, data_base)
    table = encode_table(plans)
    header = struct.pack(">iiqqqiiq", MAGIC, VERSION, HEADER_SIZE, len(meta), HEADER_SIZE + len(meta), len(plans), 0,
                         data_base)
    artifact = bytearray(file_size)
    artifact[:HEADER_SIZE] = header
    artifact[HEADER_SIZE:HEADER_SIZE + len(meta)] = meta
    artifact[HEADER_SIZE + len(meta):data_base] = table
    for p, payload in zip(plans, payloads, strict=True):
        artifact[p.offset:p.offset + p.byte_size] = payload
    return bytes(artifact) + tokenizer_section.encode(TOKENIZER_FILES)


if __name__ == "__main__":
    out = HERE / "tiny.edrl"
    out.write_bytes(build())
    print(f"wrote {out} ({out.stat().st_size} bytes)")
