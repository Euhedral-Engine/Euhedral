"""EDRL v2 container: header, metadata, tensor table and aligned payloads."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Any, Callable, Iterable, NoReturn

MAGIC = 0x5157454E
VERSION = 2
HEADER_SIZE = 48
HEADER_FORMAT = ">iiqqqiiq"

DTYPE_ORDINAL = {"BF16": 0, "INT32": 6}
FORMAT_ORDINAL = {
    "BF16": 0,
    "FP32": 2,
    "I32": 3,
    "Q3G64_F16S": 4,
    "Q4G64_F16S": 5,
    "Q5G64_F16S": 6,
    "NVFP4": 7,
}
LAYOUT_ORDINAL = {
    "contiguous-le-v1": 0,
    "row-split-k128-v1": 1,
    "row-split-p2e2-v1": 2,
    "row-split-k128-sd4-v1": 3,
}
FORMAT_Q3 = FORMAT_ORDINAL["Q3G64_F16S"]
LAYOUT_ROW_SPLIT = LAYOUT_ORDINAL["row-split-k128-v1"]
LAYOUT_P2E2 = LAYOUT_ORDINAL["row-split-p2e2-v1"]


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def checked_product(values: Iterable[int], label: str) -> int:
    result = 1
    for value in values:
        if type(value) is not int or value <= 0:
            fail(f"{label} contains a non-positive dimension")
        result *= value
    return result


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def utf8(value: str, field: str) -> bytes:
    if not isinstance(value, str) or not value:
        fail(f"{field} must be a non-empty string")
    encoded = value.encode("utf-8", "strict")
    if len(encoded) > 1 << 20:
        fail(f"{field} is too long")
    return encoded


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ObjectPlan:
    name: str
    shape: tuple[int, ...]
    source_dtype: str
    format_name: str
    layout: str
    byte_size: int
    writer: Callable[[Any, int], None]
    offset: int = 0


def encode_table(plans: list[ObjectPlan]) -> bytes:
    result = bytearray()
    for plan in plans:
        name = utf8(plan.name, "tensor name")
        result.extend(struct.pack(">i", len(name)))
        result.extend(name)
        result.extend(struct.pack(">i", len(plan.shape)))
        for dimension in plan.shape:
            result.extend(struct.pack(">q", dimension))
        result.extend(struct.pack(">iiiqq", DTYPE_ORDINAL[plan.source_dtype], FORMAT_ORDINAL[plan.format_name],
                                  LAYOUT_ORDINAL[plan.layout], plan.offset, plan.byte_size))
    return bytes(result)


def assign_offsets(plans: list[ObjectPlan], data_base: int) -> int:
    cursor = data_base
    for plan in plans:
        plan.offset = align_up(cursor, 256)
        cursor = plan.offset + plan.byte_size
    return cursor


def read_table(handle) -> tuple[bytes, bytes, list[dict[str, Any]]]:
    """(header bytes, metadata bytes, tensor descriptors) of an EDRL v2 artifact."""
    header = handle.read(HEADER_SIZE)
    if len(header) != HEADER_SIZE:
        fail("input is not an EDRL v2 artifact")
    magic, version, metadata_offset, metadata_size, table_offset, count, reserved, data_offset = struct.unpack(
        HEADER_FORMAT, header)
    if magic != MAGIC or version != VERSION or reserved != 0:
        fail("input is not an EDRL v2 artifact")
    handle.seek(metadata_offset)
    metadata = handle.read(metadata_size)
    handle.seek(table_offset)
    objects = []
    for _ in range(count):
        name_size = struct.unpack(">i", handle.read(4))[0]
        name = handle.read(name_size).decode("utf-8")
        rank = struct.unpack(">i", handle.read(4))[0]
        shape = struct.unpack(f">{rank}q", handle.read(8 * rank))
        dtype, fmt, layout, offset, size = struct.unpack(">iiiqq", handle.read(28))
        objects.append({"name": name, "shape": tuple(shape), "dtype": dtype, "format": fmt, "layout": layout,
                        "offset": offset, "bytes": size})
    if handle.tell() != data_offset:
        fail("unexpected gap between the tensor table and the data")
    return header, metadata, objects


def encode_descriptors(objects: list[dict[str, Any]]) -> bytes:
    """The tensor table of descriptors in read_table's form."""
    result = bytearray()
    for obj in objects:
        name = obj["name"].encode("utf-8")
        result += struct.pack(">i", len(name)) + name + struct.pack(">i", len(obj["shape"]))
        result += struct.pack(f">{len(obj['shape'])}q", *obj["shape"])
        result += struct.pack(">iiiqq", obj["dtype"], obj["format"], obj["layout"], obj["offset"], obj["bytes"])
    return bytes(result)
