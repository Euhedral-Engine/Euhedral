"""The Hugging Face safetensors checkpoint and row-matrix views of its tensors."""

from __future__ import annotations

from dataclasses import dataclass
import json
import mmap
from pathlib import Path
import struct
from typing import Any, Callable

import numpy as np

from euhedral_artifacts.edrl import checked_product, fail

VOCAB_SIZE = 248320
HIDDEN = 5120


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON {path}: {error}")
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def bf16_to_float32(words: np.ndarray) -> np.ndarray:
    bits = words.astype(np.uint32, copy=False) << 16
    return bits.view(np.float32)


def read_safetensors_header(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                fail(f"safetensors header is truncated: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            raw = handle.read(header_size)
    except OSError as error:
        fail(f"cannot read safetensors header {path}: {error}")
    if len(raw) != header_size:
        fail(f"safetensors header is truncated: {path}")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"invalid safetensors header {path}: {error}")
    if not isinstance(header, dict):
        fail(f"safetensors header is not an object: {path}")
    return 8 + header_size, {name: value for name, value in header.items() if name != "__metadata__"}


@dataclass(frozen=True)
class SourceRef:
    shard: str
    offset: int
    shape: tuple[int, ...]


class SourceStore:
    def __init__(self, model: Path):
        index = read_json(model / "model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            fail("model.safetensors.index.json has no weight_map")
        self.model = model
        self.refs: dict[str, SourceRef] = {}
        self.files: dict[str, Any] = {}
        self.maps: dict[str, mmap.mmap] = {}
        self.headers: dict[str, dict[str, Any]] = {}
        for shard_name in sorted(set(weight_map.values())):
            if not isinstance(shard_name, str):
                fail("safetensors shard name is not a string")
            shard_path = model / shard_name
            header_base, header = read_safetensors_header(shard_path)
            self.headers[shard_name] = header
            file = shard_path.open("rb")
            self.files[shard_name] = file
            self.maps[shard_name] = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            for name, entry in header.items():
                if name not in weight_map or weight_map[name] != shard_name:
                    fail(f"source index/header mismatch for {name}")
                if not isinstance(entry, dict) or entry.get("dtype") != "BF16":
                    fail(f"source tensor {name} is not BF16")
                shape = entry.get("shape")
                offsets = entry.get("data_offsets")
                if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                    fail(f"source tensor {name} has malformed metadata")
                shape_tuple = tuple(int(item) for item in shape)
                source_bytes = int(offsets[1]) - int(offsets[0])
                if source_bytes != checked_product(shape_tuple, name) * 2:
                    fail(f"source tensor {name} byte size conflicts with shape")
                self.refs[name] = SourceRef(
                    shard_name, header_base + int(offsets[0]), shape_tuple
                )
        if set(self.refs) != set(weight_map):
            fail("source index does not match safetensors headers")

    def close(self) -> None:
        for value in self.maps.values():
            value.close()
        for value in self.files.values():
            value.close()

    def __enter__(self) -> "SourceStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def ref(self, name: str, shape: tuple[int, ...] | None = None) -> SourceRef:
        try:
            value = self.refs[name]
        except KeyError:
            fail(f"source tensor is missing: {name}")
        if shape is not None and value.shape != shape:
            fail(f"source tensor {name} has shape {value.shape}, expected {shape}")
        return value

    def words(self, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        ref = self.ref(name, shape)
        count = checked_product(ref.shape, name)
        return np.frombuffer(self.maps[ref.shard], dtype="<u2", count=count, offset=ref.offset).reshape(ref.shape)

    def raw(self, name: str, shape: tuple[int, ...] | None = None) -> bytes:
        ref = self.ref(name, shape)
        return bytes(self.maps[ref.shard][ref.offset : ref.offset + checked_product(ref.shape, name) * 2])

    def float_rows(self, name: str, begin: int, end: int, shape: tuple[int, int]) -> np.ndarray:
        words = self.words(name, shape)[begin:end]
        return np.ascontiguousarray(bf16_to_float32(words), dtype=np.float32)

    def float_indices(self, name: str, indices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        words = self.words(name, shape)[indices]
        return np.ascontiguousarray(bf16_to_float32(words), dtype=np.float32)


@dataclass
class MatrixSource:
    shape: tuple[int, int]
    read_rows: Callable[[int, int], np.ndarray]


def source_matrix(store: SourceStore, name: str, shape: tuple[int, int]) -> MatrixSource:
    store.ref(name, shape)
    return MatrixSource(shape, lambda begin, end: store.float_rows(name, begin, end, shape))



def slice_matrix(source: MatrixSource, begin: int, end: int) -> MatrixSource:
    if begin < 0 or end <= begin or end > source.shape[0]:
        fail("invalid matrix row slice")
    return MatrixSource(
        (end - begin, source.shape[1]),
        lambda row_begin, row_end: source.read_rows(begin + row_begin, begin + row_end),
    )


def concat_matrix(*sources: MatrixSource) -> MatrixSource:
    if not sources or len({source.shape[1] for source in sources}) != 1:
        fail("matrix concatenation requires a common K dimension")
    shape = (sum(source.shape[0] for source in sources), sources[0].shape[1])

    def read_rows(begin: int, end: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        cursor = 0
        for source in sources:
            source_end = cursor + source.shape[0]
            if begin < source_end and end > cursor:
                local_begin = max(begin, cursor) - cursor
                local_end = min(end, source_end) - cursor
                parts.append(source.read_rows(local_begin, local_end))
            cursor = source_end
        result = np.concatenate(parts, axis=0)
        if result.shape != (end - begin, shape[1]):
            fail("matrix concatenation produced an unexpected shape")
        return result

    return MatrixSource(shape, read_rows)


def head_part(store: SourceStore, name: str, gate: bool) -> MatrixSource:
    shape = (24 * 256, HIDDEN)
    source_shape = (24 * 512, HIDDEN)
    store.ref(name, source_shape)
    base = 256 if gate else 0

    def read_rows(begin: int, end: int) -> np.ndarray:
        logical = np.arange(begin, end, dtype=np.int64)
        heads = logical // 256
        within = logical % 256
        source_rows = heads * 512 + base + within
        return store.float_indices(name, source_rows, source_shape)

    return MatrixSource(shape, read_rows)


def gather_matrix(store: SourceStore, name: str, rows: np.ndarray) -> MatrixSource:
    source_shape = (VOCAB_SIZE, HIDDEN)
    store.ref(name, source_shape)
    rows = np.asarray(rows, dtype=np.int64)
    return MatrixSource(
        (int(rows.size), HIDDEN),
        lambda begin, end: store.float_indices(name, rows[begin:end], source_shape),
    )
