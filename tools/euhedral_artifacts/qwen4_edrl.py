"""EDRL version 3: the architecture-tagged container for `qwen4_exp` (docs/FLASH_NEXT_ARTIFACT.md).

    header (64 bytes) | metadata | tables | zero padding to 4096 | payloads

All header, metadata and table integers are big-endian. The tables carry every object's CRC-32 (IEEE, as in zlib)
and the file is only valid once its header is written, which is the last write of a conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import struct
import time
from typing import Any, Callable
import zlib

import numpy as np

from euhedral_artifacts.edrl import MAGIC, align_up, fail, sha256
from euhedral_artifacts.nvfp4 import nvfp4_offsets
from euhedral_artifacts.qwen4_source import (BANK_ALIGNMENT, GROUP_ORDINAL, GROUPS, NGRAM_ROW_LAYOUT, NVFP4_BLOCK,
                                             BankPlan, Plan, SourceCheckpoint, TensorPlan, assign_offsets, build_plan,
                                             ngram_row_bytes)

VERSION = 3
HEADER_SIZE = 64
HEADER_FORMAT = ">iiiiqqqqqq"
ARCHITECTURE_QWEN4_EXP = 1
DATA_ALIGNMENT = 4096

DTYPE_ORDINAL = {"BF16": 0}
FORMAT_ORDINAL = {"BF16": 0, "NVFP4": 7}
LAYOUT_ORDINAL = {"contiguous-le-v1": 0, "row-split-k128-v1": 1, "row-interleaved-nvfp4-v1": 4}

TYPE_INT, TYPE_FLOAT, TYPE_BOOL, TYPE_STRING, TYPE_INT_ARRAY, TYPE_STRING_ARRAY = range(6)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8", "strict")
    return struct.pack(">i", len(encoded)) + encoded


def encode_metadata(metadata: dict[str, Any]) -> bytes:
    out = bytearray(struct.pack(">i", len(metadata)))
    for key in sorted(metadata):
        value = metadata[key]
        out += _string(key)
        if isinstance(value, bool):
            out += bytes([TYPE_BOOL, 1 if value else 0])
        elif isinstance(value, int):
            out += bytes([TYPE_INT]) + struct.pack(">q", value)
        elif isinstance(value, float):
            out += bytes([TYPE_FLOAT]) + struct.pack(">d", value)
        elif isinstance(value, str):
            out += bytes([TYPE_STRING]) + _string(value)
        elif isinstance(value, list) and all(isinstance(v, str) for v in value) and value:
            out += bytes([TYPE_STRING_ARRAY]) + struct.pack(">i", len(value)) + b"".join(_string(v) for v in value)
        elif isinstance(value, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            out += bytes([TYPE_INT_ARRAY]) + struct.pack(">i", len(value)) + struct.pack(f">{len(value)}q", *value)
        else:
            fail(f"metadata {key} has an unsupported value {value!r}")
    return bytes(out)


def decode_metadata(data: bytes) -> dict[str, Any]:
    cursor = 0

    def take(size: int) -> bytes:
        nonlocal cursor
        if cursor + size > len(data):
            fail("metadata is truncated")
        chunk = data[cursor:cursor + size]
        cursor += size
        return chunk

    def string() -> str:
        return take(struct.unpack(">i", take(4))[0]).decode("utf-8")

    result: dict[str, Any] = {}
    for _ in range(struct.unpack(">i", take(4))[0]):
        key = string()
        kind = take(1)[0]
        if kind == TYPE_INT:
            result[key] = struct.unpack(">q", take(8))[0]
        elif kind == TYPE_FLOAT:
            result[key] = struct.unpack(">d", take(8))[0]
        elif kind == TYPE_BOOL:
            result[key] = bool(take(1)[0])
        elif kind == TYPE_STRING:
            result[key] = string()
        elif kind == TYPE_INT_ARRAY:
            count = struct.unpack(">i", take(4))[0]
            result[key] = list(struct.unpack(f">{count}q", take(8 * count)))
        elif kind == TYPE_STRING_ARRAY:
            result[key] = [string() for _ in range(struct.unpack(">i", take(4))[0])]
        else:
            fail(f"metadata {key} has unknown type {kind}")
    if cursor != len(data):
        fail("metadata has trailing bytes")
    return result


def encode_tables(plan: Plan) -> bytes:
    out = bytearray(struct.pack(">i", len(plan.tensors)))
    for tensor in plan.tensors:
        name = tensor.name.encode("utf-8")
        out += struct.pack(">i", len(name)) + name + struct.pack(">i", len(tensor.shape))
        out += struct.pack(f">{len(tensor.shape)}q", *tensor.shape)
        out += struct.pack(">iiiiqqI", DTYPE_ORDINAL[tensor.dtype], FORMAT_ORDINAL[tensor.fmt],
                           LAYOUT_ORDINAL[tensor.layout], GROUP_ORDINAL[tensor.group], tensor.offset, tensor.size,
                           tensor.crc)
    out += struct.pack(">i", len(plan.banks))
    for bank in plan.banks:
        name = bank.name.encode("utf-8")
        out += struct.pack(">i", len(name)) + name
        out += struct.pack(">iiii", GROUP_ORDINAL[bank.group], bank.layer, bank.expert_count, len(bank.projections))
        for projection in bank.projections:
            pname = projection.name.encode("utf-8")
            out += struct.pack(">i", len(pname)) + pname + struct.pack(">i", 2)
            out += struct.pack(">qq", *projection.shape)
            out += struct.pack(">iiiqq", DTYPE_ORDINAL["BF16"], FORMAT_ORDINAL["NVFP4"],
                               LAYOUT_ORDINAL["row-split-k128-v1"], projection.offset, projection.size)
        crcs = bank.crcs or [0] * bank.expert_count
        for expert in range(bank.expert_count):
            out += struct.pack(">qqI", bank.offset + expert * bank.record_bytes, bank.record_bytes, crcs[expert])
    return bytes(out)


def encode_header(metadata_size: int, tables_size: int, data_offset: int, file_size: int) -> bytes:
    tables_offset = HEADER_SIZE + metadata_size
    return struct.pack(HEADER_FORMAT, MAGIC, VERSION, ARCHITECTURE_QWEN4_EXP, 0, HEADER_SIZE, metadata_size,
                       tables_offset, tables_size, data_offset, file_size)


# --- rendering objects from the checkpoint ----------------------------------------------------------------------


def _u8(source: SourceCheckpoint, name: str) -> np.ndarray:
    return np.frombuffer(source.view(name), dtype=np.uint8)


def render_tensor(source: SourceCheckpoint, tensor: TensorPlan) -> np.ndarray:
    """The payload bytes of a tensor object, as a flat uint8 array."""
    if tensor.kind == "direct":
        return _u8(source, tensor.parts["raw"])
    rows, k = tensor.shape
    packed = _u8(source, tensor.parts["packed"]).reshape(rows, k // 2)
    scales = _u8(source, tensor.parts["scale"]).reshape(rows, k // NVFP4_BLOCK)
    global_scale = _u8(source, tensor.parts["scale_2"])
    out = np.zeros(tensor.size, dtype=np.uint8)
    if tensor.kind == "nvfp4":
        scale_offset, global_offset, size = nvfp4_offsets((rows, k))
        out[:rows * k // 2] = packed.reshape(-1)
        out[scale_offset:scale_offset + rows * k // NVFP4_BLOCK] = scales.reshape(-1)
        out[global_offset:global_offset + 4] = global_scale
        return out
    row_bytes = ngram_row_bytes(k)
    body = out[:rows * row_bytes].reshape(rows, row_bytes)
    body[:, :k // 2] = packed
    body[:, k // 2:] = scales
    trailer = align_up(rows * row_bytes, 256)
    out[trailer:trailer + 4] = global_scale
    return out


def render_bank(source: SourceCheckpoint, bank: BankPlan, low: int, high: int) -> np.ndarray:
    """Expert records `low`..`high` of a bank, as a (high - low, record_bytes) uint8 array."""
    out = np.zeros((high - low, bank.record_bytes), dtype=np.uint8)
    for projection in bank.projections:
        parts = bank.parts[projection.name]
        rows, k = projection.shape
        scale_offset, global_offset, size = nvfp4_offsets((rows, k))
        packed = _u8(source, parts["packed"]).reshape(bank.expert_count, rows * k // 2)[low:high]
        scales = _u8(source, parts["scale"]).reshape(bank.expert_count, rows * k // NVFP4_BLOCK)[low:high]
        global_scale = _u8(source, parts["scale_2"]).reshape(bank.expert_count, 4)[low:high]
        base = projection.offset
        out[:, base:base + rows * k // 2] = packed
        out[:, base + scale_offset:base + scale_offset + rows * k // NVFP4_BLOCK] = scales
        out[:, base + global_offset:base + global_offset + 4] = global_scale
    return out


# --- converting -------------------------------------------------------------------------------------------------

EXPERT_CHUNK = 64

_STATE: dict[str, Any] = {}


def _pwrite_all(fd: int, buffer, offset: int) -> None:
    view = memoryview(buffer).cast("B")
    written = 0
    while written < len(view):
        count = os.pwrite(fd, view[written:written + (1 << 30)], offset + written)
        written += count


def _work(item: tuple[str, int, int, int]) -> tuple[str, int, int, int, list[int]]:
    kind, index, low, high = item
    plan: Plan = _STATE["plan"]
    source: SourceCheckpoint = _STATE["source"]
    fd = os.open(_STATE["path"], os.O_WRONLY)
    try:
        if kind == "tensor":
            tensor = plan.tensors[index]
            data = render_tensor(source, tensor)
            _pwrite_all(fd, data, tensor.offset)
            return kind, index, low, high, [zlib.crc32(data)]
        bank = plan.banks[index]
        records = render_bank(source, bank, low, high)
        _pwrite_all(fd, records, bank.offset + low * bank.record_bytes)
        return kind, index, low, high, [zlib.crc32(records[e]) for e in range(high - low)]
    finally:
        os.close(fd)


def work_items(plan: Plan) -> list[tuple[str, int, int, int]]:
    items: list[tuple[str, int, int, int]] = []
    for index, bank in enumerate(plan.banks):
        for low in range(0, bank.expert_count, EXPERT_CHUNK):
            items.append(("bank", index, low, min(low + EXPERT_CHUNK, bank.expert_count)))
    for index in range(len(plan.tensors)):
        items.append(("tensor", index, 0, 0))
    items.sort(key=lambda item: -(plan.tensors[item[1]].size if item[0] == "tensor"
                                  else (item[3] - item[2]) * plan.banks[item[1]].record_bytes))
    return items


def plan_fingerprint(plan: Plan, metadata_bytes: bytes) -> str:
    digest = hashlib.sha256(metadata_bytes)
    for tensor in plan.tensors:
        digest.update(f"{tensor.name}|{tensor.offset}|{tensor.size}|{tensor.layout}".encode())
    for bank in plan.banks:
        digest.update(f"{bank.name}|{bank.offset}|{bank.record_bytes}|{bank.expert_count}".encode())
    return digest.hexdigest()


def prepare(source: SourceCheckpoint) -> tuple[Plan, bytes, int, int]:
    plan = build_plan(source)
    metadata = encode_metadata(plan.metadata)
    tables_size = len(encode_tables(plan))
    data_offset = align_up(HEADER_SIZE + len(metadata) + tables_size, DATA_ALIGNMENT)
    file_size = assign_offsets(plan, data_offset)
    return plan, metadata, data_offset, file_size


def convert(checkpoint: Path, output: Path, jobs: int = 4, force: bool = False,
            log: Callable[[str], None] = lambda message: print(message, flush=True)) -> dict[str, Any]:
    """Writes the artifact for the NVFP4 checkpoint directory. Restartable: an interrupted conversion leaves
    `<output>.partial` and `<output>.progress`, and running it again writes only what is missing."""
    if output.exists() and not force:
        fail(f"output already exists; pass --force: {output}")
    source = SourceCheckpoint(checkpoint)
    plan, metadata, data_offset, file_size = prepare(source)
    fingerprint = plan_fingerprint(plan, metadata)
    partial = output.with_name(output.name + ".partial")
    progress = output.with_name(output.name + ".progress")
    done: dict[tuple[str, int, int, int], list[int]] = {}
    if partial.exists() and progress.exists() and partial.stat().st_size == file_size:
        lines = progress.read_text().splitlines()
        if lines and json.loads(lines[0]).get("fingerprint") == fingerprint:
            for line in lines[1:]:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    break  # a torn last line: that item is rewritten
                done[(entry["kind"], entry["index"], entry["low"], entry["high"])] = entry["crcs"]
            log(f"resuming: {len(done)} work items already written")
        else:
            done.clear()
    if not done:
        output.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("wb") as handle:
            handle.truncate(file_size)
        progress.write_text(json.dumps({"fingerprint": fingerprint}) + "\n")
    items = work_items(plan)
    todo = [item for item in items if item not in done]
    log(f"{len(plan.tensors)} tensors and {len(plan.banks)} expert banks; {file_size:,} bytes; "
        f"{len(todo)} of {len(items)} work items to write with {jobs} workers")
    _STATE.update(plan=plan, source=source, path=str(partial))
    started = time.time()
    written_bytes = 0
    with progress.open("a") as journal:
        def record(result) -> None:
            nonlocal written_bytes
            kind, index, low, high, crcs = result
            done[(kind, index, low, high)] = crcs
            journal.write(json.dumps({"kind": kind, "index": index, "low": low, "high": high, "crcs": crcs}) + "\n")
            journal.flush()
            written_bytes += (plan.tensors[index].size if kind == "tensor"
                              else (high - low) * plan.banks[index].record_bytes)
            if len(done) % 50 == 0 or len(done) == len(items):
                rate = written_bytes / max(time.time() - started, 1e-9) / 1e6
                log(f"{len(done)}/{len(items)} work items, {written_bytes / 1e9:.1f} GB written at {rate:.0f} MB/s")

        if jobs <= 1:
            for item in todo:
                record(_work(item))
        else:
            with multiprocessing.get_context("fork").Pool(jobs) as pool:
                for result in pool.imap_unordered(_work, todo):
                    record(result)
    for index, tensor in enumerate(plan.tensors):
        tensor.crc = done[("tensor", index, 0, 0)][0]
    for index, bank in enumerate(plan.banks):
        bank.crcs = []
        for low in range(0, bank.expert_count, EXPERT_CHUNK):
            bank.crcs += done[("bank", index, low, min(low + EXPERT_CHUNK, bank.expert_count))]
    tables = encode_tables(plan)
    with partial.open("r+b") as handle:
        handle.seek(HEADER_SIZE)
        handle.write(metadata)
        handle.write(tables)
        handle.flush()
        os.fsync(handle.fileno())
        handle.seek(0)
        handle.write(encode_header(len(metadata), len(tables), data_offset, file_size))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, output)
    progress.unlink()
    manifest = describe(output)
    manifest["source_checkpoint"] = str(checkpoint)
    manifest["sha256"] = sha256(output)
    Path(f"{output}.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    source.close()
    return manifest


def describe(path: Path) -> dict[str, Any]:
    """Object counts, bytes per group and layout counts of an artifact file, from its own tables."""
    artifact = read_artifact(path)
    by_group: dict[str, int] = {name: 0 for name in GROUPS}
    formats: dict[str, int] = {}
    layouts: dict[str, int] = {}
    ordinal_format = {v: k for k, v in FORMAT_ORDINAL.items()}
    ordinal_layout = {v: k for k, v in LAYOUT_ORDINAL.items()}
    for tensor in artifact["tensors"]:
        by_group[GROUPS[tensor["group"]]] += tensor["bytes"]
        formats[ordinal_format[tensor["format"]]] = formats.get(ordinal_format[tensor["format"]], 0) + 1
        layouts[ordinal_layout[tensor["layout"]]] = layouts.get(ordinal_layout[tensor["layout"]], 0) + 1
    expert_records = 0
    for bank in artifact["banks"]:
        by_group[GROUPS[bank["group"]]] += sum(entry[1] for entry in bank["index"])
        expert_records += bank["expert_count"]
        formats["NVFP4 (expert records)"] = formats.get("NVFP4 (expert records)", 0) + bank["expert_count"]
        layouts["expert-record-v1"] = layouts.get("expert-record-v1", 0) + bank["expert_count"]
    sizes = path.stat().st_size
    fixed = sum(by_group[name] for name in GROUPS if name not in ("routed-expert", "shared-expert", "ngram", "mtp",
                                                                   "vision"))
    return {
        "version": VERSION,
        "architecture": "qwen4_exp",
        "artifact_bytes": sizes,
        "object_count": len(artifact["tensors"]) + len(artifact["banks"]),
        "tensor_count": len(artifact["tensors"]),
        "expert_bank_count": len(artifact["banks"]),
        "expert_record_count": expert_records,
        "fixed_text_bytes": fixed,
        "routed_expert_bytes": by_group["routed-expert"],
        "shared_expert_bytes": by_group["shared-expert"],
        "ngram_bytes": by_group["ngram"],
        "mtp_bytes": by_group["mtp"],
        "vision_bytes": by_group["vision"],
        "bytes_by_group": by_group,
        "format_counts": formats,
        "layout_counts": layouts,
        "data_offset": artifact["header"]["data_offset"],
    }


# --- reading (verification, tests) ------------------------------------------------------------------------------


def read_artifact(path: Path) -> dict[str, Any]:
    """Parses the header, metadata and tables of a version 3 artifact."""
    with path.open("rb") as handle:
        header = handle.read(HEADER_SIZE)
        magic, version, architecture, reserved, meta_off, meta_size, table_off, table_size, data_off, file_size = (
            struct.unpack(HEADER_FORMAT, header))
        if magic != MAGIC or version != VERSION or reserved != 0:
            fail("not an EDRL version 3 artifact")
        if file_size != path.stat().st_size:
            fail("file size does not match the header")
        handle.seek(meta_off)
        metadata = decode_metadata(handle.read(meta_size))
        handle.seek(table_off)
        data = handle.read(table_size)
    cursor = 0

    def take(size: int) -> bytes:
        nonlocal cursor
        chunk = data[cursor:cursor + size]
        if len(chunk) != size:
            fail("tables are truncated")
        cursor += size
        return chunk

    def integer(fmt: str):
        values = struct.unpack(fmt, take(struct.calcsize(fmt)))
        return values if len(values) > 1 else values[0]

    tensors = []
    for _ in range(integer(">i")):
        name = take(integer(">i")).decode("utf-8")
        rank = integer(">i")
        shape = tuple(struct.unpack(f">{rank}q", take(8 * rank)))
        dtype, fmt, layout, group, offset, size, crc = integer(">iiiiqqI")
        tensors.append({"name": name, "shape": shape, "dtype": dtype, "format": fmt, "layout": layout,
                        "group": group, "offset": offset, "bytes": size, "crc": crc})
    banks = []
    for _ in range(integer(">i")):
        name = take(integer(">i")).decode("utf-8")
        group, layer, expert_count, projection_count = integer(">iiii")
        projections = []
        for _ in range(projection_count):
            pname = take(integer(">i")).decode("utf-8")
            rank = integer(">i")
            shape = tuple(struct.unpack(f">{rank}q", take(8 * rank)))
            dtype, fmt, layout, offset, size = integer(">iiiqq")
            projections.append({"name": pname, "shape": shape, "dtype": dtype, "format": fmt, "layout": layout,
                                "offset": offset, "bytes": size})
        index = [integer(">qqI") for _ in range(expert_count)]
        banks.append({"name": name, "group": group, "layer": layer, "expert_count": expert_count,
                      "projections": projections, "index": index})
    if cursor != len(data):
        fail("tables have trailing bytes")
    return {"header": {"architecture": architecture, "metadata_offset": meta_off, "tables_offset": table_off,
                       "tables_size": table_size, "data_offset": data_off, "file_size": file_size},
            "metadata": metadata, "tensors": tensors, "banks": banks}


def _verify_range(args: tuple[str, int, int, int, str]) -> tuple[str, int]:
    path, offset, size, crc, label = args
    fd = os.open(path, os.O_RDONLY)
    try:
        value = 0
        position = 0
        while position < size:
            chunk = os.pread(fd, min(1 << 26, size - position), offset + position)
            if not chunk:
                fail(f"{label} is truncated")
            value = zlib.crc32(chunk, value)
            position += len(chunk)
    finally:
        os.close(fd)
    if value != crc:
        fail(f"{label}: CRC-32 {value:08x} does not match the table's {crc:08x}")
    return label, size


def verify(path: Path, checkpoint: Path | None = None, jobs: int = 4,
           log: Callable[[str], None] = lambda message: print(message, flush=True)) -> dict[str, Any]:
    """Checks every written range of the artifact against the tables' CRC-32s, the ranges against each other (no
    overlap, inside the data section), and with `checkpoint` every byte of every object against the checkpoint."""
    artifact = read_artifact(path)
    header = artifact["header"]
    ranges: list[tuple[int, int, str]] = []
    jobs_list: list[tuple[str, int, int, int, str]] = []
    for tensor in artifact["tensors"]:
        ranges.append((tensor["offset"], tensor["bytes"], tensor["name"]))
        jobs_list.append((str(path), tensor["offset"], tensor["bytes"], tensor["crc"], tensor["name"]))
    for bank in artifact["banks"]:
        for expert, (offset, size, crc) in enumerate(bank["index"]):
            ranges.append((offset, size, f"{bank['name']}#{expert}"))
            jobs_list.append((str(path), offset, size, crc, f"{bank['name']}#{expert}"))
    ranges.sort()
    previous_end = header["data_offset"]
    for offset, size, label in ranges:
        if offset < previous_end:
            fail(f"{label} overlaps the previous object or precedes the data section")
        previous_end = offset + size
    if previous_end > header["file_size"]:
        fail("objects extend beyond the file")
    log(f"{len(ranges)} ranges are inside the file and do not overlap; checking CRC-32s")
    checked = 0
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        for _, size in pool.imap_unordered(_verify_range, jobs_list, chunksize=16):
            checked += size
    log(f"CRC-32 of {len(jobs_list)} ranges ({checked:,} bytes) match")
    result = {"ranges": len(ranges), "crc_bytes": checked}
    if checkpoint is not None:
        source = SourceCheckpoint(checkpoint)
        plan, metadata, data_offset, file_size = prepare(source)
        by_name = {tensor.name: tensor for tensor in plan.tensors}
        if data_offset != header["data_offset"] or file_size != header["file_size"]:
            fail("the checkpoint's plan does not match the artifact's geometry")
        _STATE.update(plan=plan, source=source, path=str(path))
        compared = 0
        fd = os.open(path, os.O_RDONLY)
        try:
            for tensor in artifact["tensors"]:
                planned = by_name[tensor["name"]]
                if planned.offset != tensor["offset"] or planned.size != tensor["bytes"]:
                    fail(f"{tensor['name']} is not where the checkpoint's plan puts it")
                expected = render_tensor(source, planned)
                actual = os.pread(fd, planned.size, planned.offset) if planned.size < (1 << 30) else None
                if actual is None:
                    actual = b"".join(os.pread(fd, min(1 << 28, planned.size - p), planned.offset + p)
                                      for p in range(0, planned.size, 1 << 28))
                if actual != bytes(expected):
                    fail(f"{tensor['name']} differs from the checkpoint")
                compared += planned.size
            for index, bank in enumerate(plan.banks):
                for low in range(0, bank.expert_count, EXPERT_CHUNK):
                    high = min(low + EXPERT_CHUNK, bank.expert_count)
                    expected = render_bank(source, bank, low, high)
                    actual = os.pread(fd, (high - low) * bank.record_bytes, bank.offset + low * bank.record_bytes)
                    if actual != expected.tobytes():
                        fail(f"{bank.name} experts {low}..{high} differ from the checkpoint")
                    compared += len(actual)
        finally:
            os.close(fd)
        source.close()
        log(f"{compared:,} bytes equal the checkpoint's")
        result["compared_bytes"] = compared
    return result
