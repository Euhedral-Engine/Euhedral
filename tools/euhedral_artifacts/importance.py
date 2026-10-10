"""Calibrated rounding: per-column importance from a llama.cpp importance matrix (imatrix GGUF).

An importance matrix holds, for every linear layer's input, the mean square of each input column over a
calibration text (`<tensor>.in_sum2` divided by `<tensor>.counts`). A linear's output error under a weight
error E is then approximately sum over columns j of importance_j * |E_j|^2, so calibrated quantizers choose
each group's scale to minimize that weighted error rather than the plain one.

Objects the matrix does not cover (the token embedding, the LM head, the draft head and the MTP layer) are
still searched, with uniform weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import struct

import numpy as np

from euhedral_artifacts.edrl import fail

GGUF_MAGIC = b"GGUF"
GGML_TYPE_F32 = 0
# GGUF metadata value types: struct formats of the scalars; 8 is a string and 9 an array.
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


@dataclass(frozen=True)
class Calibration:
    """How one object is rounded: by scale search, weighting each column's squared error by `importance`
    (length K), or uniformly when it is None. `lowest_code`, when set, is the most negative integer code the
    search may use (the format's own range otherwise)."""
    importance: np.ndarray | None = None
    lowest_code: int | None = None

    def weights(self, k_pad: int) -> np.ndarray:
        """Column weights padded to `k_pad` columns (padding columns hold zeros and weigh nothing)."""
        weights = np.zeros(k_pad, dtype=np.float32)
        if self.importance is None:
            weights[:] = 1.0
        else:
            weights[:len(self.importance)] = self.importance
        return weights


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.position = data, 0

    def take(self, size: int) -> bytes:
        if self.position + size > len(self.data):
            fail("importance matrix is truncated")
        chunk = self.data[self.position:self.position + size]
        self.position += size
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8")

    def value(self, kind: int):
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            item_kind, count = self.u32(), self.u64()
            return [self.value(item_kind) for _ in range(count)]
        if kind not in _SCALARS:
            fail(f"importance matrix has an unknown metadata type {kind}")
        return struct.unpack(_SCALARS[kind], self.take(struct.calcsize(_SCALARS[kind])))[0]


def read_gguf_f32(path: Path) -> dict[str, np.ndarray]:
    """The F32 tensors of the GGUF file `path`, flattened, by name."""
    data = path.read_bytes()
    reader = _Reader(data)
    if reader.take(4) != GGUF_MAGIC:
        fail(f"{path} is not a GGUF file")
    if reader.u32() not in (2, 3):
        fail(f"{path} has an unsupported GGUF version")
    tensor_count, metadata_count = reader.u64(), reader.u64()
    alignment = 32
    for _ in range(metadata_count):
        key = reader.string()
        value = reader.value(reader.u32())
        if key == "general.alignment":
            alignment = int(value)
    infos = []
    for _ in range(tensor_count):
        name = reader.string()
        dims = [reader.u64() for _ in range(reader.u32())]
        kind, offset = reader.u32(), reader.u64()
        infos.append((name, int(np.prod(dims)), kind, offset))
    base = (reader.position + alignment - 1) // alignment * alignment
    tensors = {}
    for name, count, kind, offset in infos:
        if kind != GGML_TYPE_F32:
            fail(f"importance matrix tensor {name} is not F32")
        begin = base + offset
        if begin + count * 4 > len(data):
            fail(f"importance matrix tensor {name} is truncated")
        tensors[name] = np.frombuffer(data, dtype="<f4", count=count, offset=begin).astype(np.float32)
    return tensors


# Artifact object -> the llama.cpp tensor whose input it shares. Fused objects read one input, so any of
# their parts' entries serves.
_SOURCES = (
    (re.compile(r"text/layers/(\d+)/attention/(query_key|gate_value)"), "blk.{}.attn_q.weight"),
    (re.compile(r"text/layers/(\d+)/attention/output"), "blk.{}.attn_output.weight"),
    (re.compile(r"text/layers/(\d+)/gdn/(query_key|value_z)"), "blk.{}.attn_qkv.weight"),
    (re.compile(r"text/layers/(\d+)/gdn/output"), "blk.{}.ssm_out.weight"),
    (re.compile(r"text/layers/(\d+)/mlp/gate_up"), "blk.{}.ffn_gate.weight"),
    (re.compile(r"text/layers/(\d+)/mlp/down"), "blk.{}.ffn_down.weight"),
)


class ImportanceMatrix:
    """Per-object calibration from a llama.cpp importance matrix."""

    def __init__(self, path: Path):
        tensors = read_gguf_f32(path)
        self.path = path
        self.columns: dict[str, np.ndarray] = {}
        for name, sums in tensors.items():
            if not name.endswith(".in_sum2"):
                continue
            stem = name[:-len(".in_sum2")]
            counts = tensors.get(stem + ".counts")
            if counts is None or counts.size != 1 or not counts[0] > 0:
                fail(f"importance matrix has no usable count for {stem}")
            if not np.isfinite(sums).all() or (sums < 0).any():
                fail(f"importance matrix entry {stem} is not finite and non-negative")
            self.columns[stem] = sums / counts[0]
        if not self.columns:
            fail(f"{path} holds no importance entries")

    def calibration(self, name: str, k: int) -> Calibration:
        """The calibration of artifact object `name`, whose rows have `k` columns."""
        for pattern, source in _SOURCES:
            match = pattern.fullmatch(name)
            if match:
                stem = source.format(match.group(1))
                if stem not in self.columns:
                    fail(f"importance matrix has no entry {stem} for {name}")
                importance = self.columns[stem]
                if importance.size != k:
                    fail(f"importance entry {stem} has {importance.size} columns, {name} has {k}")
                return Calibration(importance)
        return Calibration()
