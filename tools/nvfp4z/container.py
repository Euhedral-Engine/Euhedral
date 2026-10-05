"""The .nvfp4z container and safetensors parsing (no GPU, NumPy only).

    magic "NVFP4Z\\0\\1" | u64 header length | JSON header, space padded to a multiple of 64 | payload

The payload offsets in the header are relative to the payload start, which is 64-byte aligned in the file. The header
embeds the source safetensors header verbatim and the source file's SHA-256, so decompression rebuilds the exact file
and checks it.
"""

from __future__ import annotations

import json
import struct

import numpy as np

MAGIC = b"NVFP4Z\x00\x01"
BF16_MIN_ELEMENTS = 1 << 16   # smaller BF16 tensors are stored as they are
BLOCK = 16
ALIGN = 4096    # the header block is a multiple of this, so the payload starts on a page boundary (direct I/O)


def fail(message: str):
    raise ValueError(message)


def parse_safetensors(image) -> tuple[int, bytes, dict]:
    """(data start, raw header bytes, header dict without __metadata__) of a safetensors file image (bytes-like)."""
    raw = np.frombuffer(image, dtype=np.uint8, count=8)
    size = struct.unpack("<Q", raw.tobytes())[0]
    if size > len(image) - 8:
        fail("safetensors header is truncated")
    header_bytes = bytes(image[8:8 + size])
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as error:
        fail(f"safetensors header is not JSON: {error}")
    return 8 + size, header_bytes, {k: v for k, v in header.items() if k != "__metadata__"}


def packed_partner(name: str, header: dict) -> str | None:
    """The scale tensor of an NVFP4 code tensor, by the naming of the common checkpoint layouts:
    `X_packed` + `X_scale` (compressed-tensors, this repo's converters) and `X` + `X_scale` (ModelOpt)."""
    entry = header[name]
    if entry["dtype"] != "U8" or len(entry["shape"]) < 2 or entry["shape"][-1] % (BLOCK // 2):
        return None
    candidates = [name[:-len("_packed")] + "_scale"] if name.endswith("_packed") else []
    candidates.append(name + "_scale")
    want = list(entry["shape"][:-1]) + [entry["shape"][-1] // (BLOCK // 2)]
    for candidate in candidates:
        other = header.get(candidate)
        if other is not None and other["dtype"] == "F8_E4M3" and list(other["shape"]) == want:
            return candidate
    return None


def plan(header: dict) -> list[dict]:
    """The tensors of a safetensors header in file order, each with its kind:
    nvfp4 (a code tensor, coded with its scale tensor), paired (that scale tensor), bf16 (high bytes coded) or raw."""
    items = sorted(header.items(), key=lambda item: item[1]["data_offsets"][0])
    partners = {}
    for name, _ in items:
        partner = packed_partner(name, header)
        if partner is not None and partner not in partners.values():
            partners[name] = partner
    paired = set(partners.values())
    out = []
    for name, entry in items:
        start, end = entry["data_offsets"]
        count = int(np.prod(entry["shape"], dtype=np.int64)) if entry["shape"] else 1
        item = {"name": name, "start": start, "end": end, "dtype": entry["dtype"], "shape": entry["shape"]}
        if name in partners:
            item.update(kind="nvfp4", partner=partners[name], blocks=count * 2 // BLOCK,
                        groups=entry["shape"][0] if len(entry["shape"]) >= 3 else 1)
        elif name in paired:
            item.update(kind="paired")
        elif entry["dtype"] == "BF16" and count >= BF16_MIN_ELEMENTS:
            item.update(kind="bf16", count=count)
        else:
            item.update(kind="raw")
        out.append(item)
    return out


def pack_header(meta: dict) -> bytes:
    body = json.dumps(meta, separators=(",", ":")).encode()
    body += b" " * (-(len(MAGIC) + 8 + len(body)) % ALIGN)
    return MAGIC + struct.pack("<Q", len(body)) + body


def read_header(prefix: bytes) -> tuple[dict, int]:
    """(meta, payload offset) from the first bytes of a container file (at least the header block)."""
    if prefix[:len(MAGIC)] != MAGIC:
        fail("not an nvfp4z container (bad magic or an unsupported version)")
    size = struct.unpack("<Q", prefix[len(MAGIC):len(MAGIC) + 8])[0]
    end = len(MAGIC) + 8 + size
    if len(prefix) < end:
        fail("container header is truncated")
    return json.loads(prefix[len(MAGIC) + 8:end]), end


def header_block_size(prefix: bytes) -> int:
    """Size of the header block, from the first 16 bytes of a container file."""
    if prefix[:len(MAGIC)] != MAGIC:
        fail("not an nvfp4z container (bad magic or an unsupported version)")
    return len(MAGIC) + 8 + struct.unpack("<Q", prefix[len(MAGIC):len(MAGIC) + 8])[0]
