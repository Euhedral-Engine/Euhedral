"""Trellis (EXL3) checkpoints as engine objects.

The quantization module writes an exllamav3 checkpoint: per quantized linear `<name>.trellis` (int16 words
[k/16, n/16, w]), `<name>.suh` (f16 [k]), `<name>.svh` (f16 [n]) and `<name>.mul1` (i32 scalar). The engine's
objects fuse the rows of several linears (the same fusion as Qwen38Objects in the quantization module), and each
source linear has its own `suh`, so a TRELLIS object is stored as four tensors:

    <name>            BF16 / TRELLIS / trellis-tile16-v1   [k/16, n/16, w]  tile columns of its pieces, in row order
    <name>.suh        FP16 / FP16 / contiguous             [s, k]           one row per segment
    <name>.svh        FP16 / FP16 / contiguous             [n]              object row order
    <name>.segments   INT32 / I32 / contiguous             [s]              exclusive row ends, multiples of 128

A segment is a maximal run of object rows that come from one source linear. Every piece (rows of one source
linear) starts and ends on a multiple of 128 rows, and every piece of an object has the same tile width.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Callable

import numpy as np

from euhedral_artifacts.codes import state_file
from euhedral_artifacts.edrl import ObjectPlan, fail
from euhedral_artifacts.recipes import LAYOUT_TRELLIS, TRELLIS_FORMAT
from euhedral_artifacts.sources import read_json, read_safetensors_header

EXL3_VERSION = "1.6.1"
MUL1 = 0x83DCD12D
BLOCK = 128
MAX_SEGMENTS = 8
HEAD_NAMES = ("text/output_head", "text/draft_head")

TEXT = "model.language_model."
MTP = "mtp.layers.0."


@dataclass(frozen=True)
class Dims:
    """The text model's dimensions that fix the shapes of the source linears (Swift-1.5 by default)."""
    hidden: int = 5120
    intermediate: int = 17408
    layers: int = 64
    heads: int = 24
    kv_heads: int = 4
    head_dim: int = 256
    key_heads: int = 16
    key_dim: int = 128
    value_heads: int = 48
    value_dim: int = 128
    full_attention: frozenset[int] = frozenset(range(3, 64, 4))


def words_valid(w: int) -> bool:
    """Whether `w` words per tile is a stored width: 16*K for K in 1..8, or 16*K + 8 for K in 1..7."""
    if w < 16 or w > 128:
        return False
    return w % 16 == 0 or (w % 16 == 8 and w <= 120)


# A piece is rows [first, first + rows) of the source linear `linear` (tensor name without `.weight`).
Piece = tuple[str, int, int]


def attention_half(prefix: str, dims: Dims, gate: bool) -> list[Piece]:
    """The query (or gate) rows of a per-head interleaved q_proj: head h owns rows h*2*head_dim + [0, head_dim)
    for the query and + [head_dim, 2*head_dim) for the gate."""
    return [(prefix + "self_attn.q_proj", h * 2 * dims.head_dim + (dims.head_dim if gate else 0), dims.head_dim)
            for h in range(dims.heads)]


def linear_shapes(dims: Dims) -> dict[str, tuple[int, int]]:
    """[n, k] of each linear role."""
    q = dims.heads * dims.head_dim
    kv = dims.kv_heads * dims.head_dim
    qk = 2 * dims.key_heads * dims.key_dim
    v = dims.value_heads * dims.value_dim
    return {
        "q_proj": (2 * q, dims.hidden), "k_proj": (kv, dims.hidden), "v_proj": (kv, dims.hidden),
        "o_proj": (dims.hidden, q), "in_proj_qkv": (qk + v, dims.hidden), "in_proj_z": (v, dims.hidden),
        "out_proj": (dims.hidden, v), "gate_proj": (dims.intermediate, dims.hidden),
        "up_proj": (dims.intermediate, dims.hidden), "down_proj": (dims.hidden, dims.intermediate),
        "mtp.fc": (dims.hidden, 2 * dims.hidden),
    }


def linear_shape(linear: str, dims: Dims) -> tuple[int, int]:
    role = "mtp.fc" if linear == "mtp.fc" else linear.rsplit(".", 1)[1]
    return linear_shapes(dims)[role]


def layer_objects(dims: Dims, index: int) -> list[str]:
    o = f"text/layers/{index}/"
    if index in dims.full_attention:
        return [o + "attention/query_key", o + "attention/gate_value", o + "attention/output", o + "mlp/gate_up",
                o + "mlp/down"]
    return [o + "gdn/query_key", o + "gdn/value_z", o + "gdn/output", o + "mlp/gate_up", o + "mlp/down"]


MTP_OBJECTS = ("mtp/input_projection", "mtp/layer/attention/query_key_gate_value", "mtp/layer/attention/output",
               "mtp/layer/mlp/gate_up", "mtp/layer/mlp/down")


def object_pieces(name: str, dims: Dims = Dims()) -> list[Piece]:
    """The rows of source linears that make up object `name`, in object row order."""
    if name == "mtp/input_projection":
        return [("mtp.fc", 0, dims.hidden)]
    if name.startswith("mtp/layer/"):
        return role_pieces(name, MTP, name[len("mtp/layer/"):], dims)
    match = re.fullmatch(r"text/layers/(\d+)/(.+)", name)
    if match is None:
        fail(f"{name} is not a trellis object")
    return role_pieces(name, f"{TEXT}layers.{match.group(1)}.", match.group(2), dims)


def role_pieces(name: str, p: str, role: str, dims: Dims) -> list[Piece]:
    shapes = linear_shapes(dims)

    def whole(linear: str) -> Piece:
        return (linear, 0, linear_shape(linear, dims)[0])

    qk = 2 * dims.key_heads * dims.key_dim
    v = dims.value_heads * dims.value_dim
    if role == "attention/query_key":
        return attention_half(p, dims, False) + [whole(p + "self_attn.k_proj")]
    if role == "attention/gate_value":
        return attention_half(p, dims, True) + [whole(p + "self_attn.v_proj")]
    if role == "attention/query_key_gate_value":
        return (attention_half(p, dims, False) + [whole(p + "self_attn.k_proj")]
                + attention_half(p, dims, True) + [whole(p + "self_attn.v_proj")])
    if role == "attention/output":
        return [whole(p + "self_attn.o_proj")]
    if role == "gdn/query_key":
        return [(p + "linear_attn.in_proj_qkv", 0, qk)]
    if role == "gdn/value_z":
        return [(p + "linear_attn.in_proj_qkv", qk, v), whole(p + "linear_attn.in_proj_z")]
    if role == "gdn/output":
        return [whole(p + "linear_attn.out_proj")]
    if role == "mlp/gate_up":
        return [whole(p + "mlp.gate_proj"), whole(p + "mlp.up_proj")]
    if role == "mlp/down":
        return [whole(p + "mlp.down_proj")]
    fail(f"unknown trellis object {name} (roles {sorted(shapes)})")


@dataclass(frozen=True)
class Stored:
    """A tensor of the checkpoint: where it sits and what it is."""
    path: Path
    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]

    def read(self, begin: int = 0, size: int | None = None) -> bytes:
        size = self.nbytes - begin if size is None else size
        fd = os.open(self.path, os.O_RDONLY)
        try:
            parts, done = [], 0
            while done < size:
                part = os.pread(fd, size - done, self.offset + begin + done)
                if not part:
                    fail(f"{self.path} is truncated")
                parts.append(part)
                done += len(part)
            return b"".join(parts)
        finally:
            os.close(fd)


DTYPES = {"I16": ("<i2", 2), "F16": ("<f2", 2), "I32": ("<i4", 4)}


@dataclass(frozen=True)
class ObjectLayout:
    """One object as the checkpoint's linears fill it."""
    name: str
    pieces: tuple[Piece, ...]
    n: int
    k: int
    w: int
    ends: tuple[int, ...]
    suh_linears: tuple[str, ...]

    @property
    def payload_shape(self) -> tuple[int, int, int]:
        return (self.k // 16, self.n // 16, self.w)


class TrellisSource:
    """The EXL3 checkpoint in `directory`, checked against the format the engine accepts."""

    def __init__(self, directory: Path, dims: Dims = Dims()):
        self.directory = directory
        self.dims = dims
        config = read_json(directory / "config.json").get("quantization_config")
        if not isinstance(config, dict):
            fail(f"{directory}/config.json has no quantization_config: not an EXL3 checkpoint")
        self.quantization_config = config
        self.check_config(config)
        index_path = directory / "model.safetensors.index.json"
        weight_map = read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            fail(f"{index_path} has no weight_map")
        self.tensors: dict[str, Stored] = {}
        for shard in sorted(set(weight_map.values())):
            base, header = read_safetensors_header(directory / shard)
            for name, entry in header.items():
                if weight_map.get(name) != shard:
                    continue
                dtype, shape, offsets = entry.get("dtype"), tuple(entry.get("shape", ())), entry.get("data_offsets")
                if dtype not in DTYPES:
                    continue
                size = DTYPES[dtype][1] * int(np.prod(shape, dtype=np.int64))
                if not isinstance(offsets, list) or len(offsets) != 2 or offsets[1] - offsets[0] != size:
                    fail(f"{name} in {shard} has data offsets that conflict with its shape")
                self.tensors[name] = Stored(directory / shard, base + int(offsets[0]), size, dtype, shape)
        self.module_bits = self.stored_bits(directory)
        self.has_mtp = "mtp_bits" in config
        self.layouts: dict[str, ObjectLayout] = {}
        for name in self.object_names():
            self.layouts[name] = self.layout(name)

    @staticmethod
    def check_config(config: dict) -> None:
        for key, expected in (("quant_method", "exl3"), ("version", EXL3_VERSION), ("codebook", "mul1"),
                              ("out_scales", "always")):
            if config.get(key) != expected:
                fail(f"quantization_config {key} is {config.get(key)!r}, the engine reads {expected!r}")

    def stored_bits(self, directory: Path) -> dict[str, float]:
        """bits_per_weight of each module in quantization_config.json's tensor_storage, when the file is there."""
        path = directory / "quantization_config.json"
        if not path.is_file():
            return {}
        storage = read_json(path).get("tensor_storage", {})
        return {module: float(entry["bits_per_weight"]) for module, entry in storage.items()
                if isinstance(entry, dict) and "bits_per_weight" in entry}

    @property
    def bits(self) -> float:
        return float(self.quantization_config["bits"])

    def object_names(self) -> list[str]:
        names = [name for layer in range(self.dims.layers) for name in layer_objects(self.dims, layer)]
        return names + (list(MTP_OBJECTS) if self.has_mtp else [])

    def carries(self, name: str) -> bool:
        return name in self.layouts

    def tensor(self, name: str, dtype: str, shape: tuple[int, ...] | None = None) -> Stored:
        found = self.tensors.get(name)
        if found is None:
            fail(f"EXL3 checkpoint {self.directory} has no tensor {name}")
        if found.dtype != dtype:
            fail(f"{name} is {found.dtype}, expected {dtype}")
        if shape is not None and found.shape != shape:
            fail(f"{name} has shape {found.shape}, expected {shape}")
        return found

    def check_linear(self, linear: str) -> tuple[Stored, int]:
        """Validates the four tensors of source linear `linear`; returns its trellis tensor and tile width."""
        n, k = linear_shape(linear, self.dims)
        trellis = self.tensors.get(linear + ".trellis")
        if trellis is None:
            fail(f"EXL3 checkpoint {self.directory} has no tensor {linear}.trellis")
        if trellis.dtype != "I16" or len(trellis.shape) != 3:
            fail(f"{linear}.trellis is {trellis.dtype} {trellis.shape}, expected rank-3 I16")
        w = trellis.shape[2]
        if trellis.shape != (k // 16, n // 16, w):
            fail(f"{linear}.trellis has shape {trellis.shape}, the linear [n={n}, k={k}] needs [{k // 16}, {n // 16}, w]")
        if not words_valid(w):
            fail(f"{linear}.trellis stores {w} words per tile, which is not a supported width")
        recorded = self.module_bits.get(linear)
        if recorded is not None and round(recorded * 16) != w:
            fail(f"{linear}: tensor_storage says {recorded} bits per weight, the tensor stores {w} words per tile")
        self.tensor(linear + ".suh", "F16", (k,))
        self.tensor(linear + ".svh", "F16", (n,))
        mul1 = self.tensor(linear + ".mul1", "I32", ())
        if int(np.frombuffer(mul1.read(), dtype="<i4")[0]) & 0xFFFFFFFF != MUL1:
            fail(f"{linear}.mul1 is not {MUL1:#x}")
        return trellis, w

    def layout(self, name: str) -> ObjectLayout:
        pieces = object_pieces(name, self.dims)
        widths: dict[str, int] = {}
        n = 0
        for linear, first, rows in pieces:
            if first % BLOCK or rows % BLOCK:
                fail(f"{name}: rows [{first}, {first + rows}) of {linear} are not aligned to {BLOCK} rows")
            if linear not in widths:
                widths[linear] = self.check_linear(linear)[1]
            n += rows
        distinct = sorted(set(widths.values()))
        if len(distinct) != 1:
            detail = ", ".join(f"{linear} {w / 16:g}" for linear, w in widths.items())
            fail(f"{name}: its pieces have different bit widths ({detail}); a trellis object has one width")
        ends: list[int] = []
        suh_linears: list[str] = []
        cursor = 0
        for linear, _, rows in pieces:
            cursor += rows
            if suh_linears and suh_linears[-1] == linear:
                ends[-1] = cursor
            else:
                suh_linears.append(linear)
                ends.append(cursor)
        if len(ends) > MAX_SEGMENTS:
            fail(f"{name} has {len(ends)} segments, at most {MAX_SEGMENTS} are supported")
        k = linear_shape(pieces[0][0], self.dims)[1]
        return ObjectLayout(name, tuple(pieces), n, k, distinct[0], tuple(ends), tuple(suh_linears))

    def check_sources(self, shape_of: Callable[[str], tuple[int, ...]]) -> None:
        """Fails unless every linear has the shape the BF16 source gives its `.weight`."""
        linears = {linear for layout in self.layouts.values() for linear in layout.suh_linears}
        for linear in sorted(linears):
            expected = tuple(shape_of(linear + ".weight"))
            actual = linear_shape(linear, self.dims)
            if expected != actual:
                fail(f"{linear}: the BF16 source is {expected}, the EXL3 checkpoint's linear is {actual}")

    def check_files(self) -> None:
        """Fails unless every shard holds the bytes its header promises."""
        for stored in self.tensors.values():
            if stored.path.stat().st_size < stored.offset + stored.nbytes:
                fail(f"{stored.path} is truncated")

    # ---- plans ----

    def add_plans(self, plans: list[ObjectPlan], name: str) -> None:
        """Adds the object `name` and its three companions."""
        layout = self.layouts[name]
        n, k, w = layout.n, layout.k, layout.w
        s = len(layout.ends)
        plans.append(ObjectPlan(name, layout.payload_shape, "BF16", TRELLIS_FORMAT, LAYOUT_TRELLIS,
                                (k // 16) * (n // 16) * w * 2,
                                lambda output, offset, source=self, spec=layout: source.write_payload(output, offset,
                                                                                                   spec)))
        plans.append(ObjectPlan(name + ".suh", (s, k), "FP16", "FP16", "contiguous-le-v1", s * k * 2,
                                lambda output, offset, source=self, spec=layout: source.write_suh(output, offset,
                                                                                                 spec)))
        plans.append(ObjectPlan(name + ".svh", (n,), "FP16", "FP16", "contiguous-le-v1", n * 2,
                                lambda output, offset, source=self, spec=layout: source.write_svh(output, offset,
                                                                                                 spec)))
        plans.append(ObjectPlan(name + ".segments", (s,), "INT32", "I32", "contiguous-le-v1", s * 4,
                                lambda output, offset, ends=layout.ends: write_at(output, offset,
                                                                                  np.asarray(ends, "<i4").tobytes())))

    def payload(self, layout: ObjectLayout) -> np.ndarray:
        """int16 [k/16, n/16, w]: the tile columns of each piece, side by side in object row order."""
        out = np.empty(layout.payload_shape, dtype="<i2")
        loaded: dict[str, np.ndarray] = {}
        column = 0
        for linear, first, rows in layout.pieces:
            if linear not in loaded:
                stored = self.tensors[linear + ".trellis"]
                loaded[linear] = np.frombuffer(stored.read(), dtype="<i2").reshape(stored.shape)
            tiles = rows // 16
            out[:, column:column + tiles, :] = loaded[linear][:, first // 16:first // 16 + tiles, :]
            column += tiles
        return out

    def write_payload(self, output, offset: int, layout: ObjectLayout) -> None:
        write_at(output, offset, self.payload(layout).tobytes())

    def write_suh(self, output, offset: int, layout: ObjectLayout) -> None:
        rows = [self.tensors[linear + ".suh"].read() for linear in layout.suh_linears]
        write_at(output, offset, b"".join(rows))

    def write_svh(self, output, offset: int, layout: ObjectLayout) -> None:
        parts = []
        for linear, first, rows in layout.pieces:
            parts.append(self.tensors[linear + ".svh"].read(2 * first, 2 * rows))
        write_at(output, offset, b"".join(parts))


def write_at(output, offset: int, data: bytes) -> None:
    output.seek(offset)
    output.write(data)


def head_names(output_head: str | None) -> list[str]:
    """The heads whose codes the converter takes from --codes-from: a BF16 LM head is stored as is."""
    return [name for name in HEAD_NAMES if not (name == "text/output_head" and output_head == "BF16")]


def require_head_codes(codes_from: Path | None, output_head: str | None) -> None:
    """Fails naming each head without a codes file in `codes_from` (calibrate --format trellis writes them)."""
    wanted = head_names(output_head)
    if codes_from is None:
        fail("missing head codes: pass --codes-from with the directory calibrate --format trellis wrote "
             f"({', '.join(wanted)})")
    missing = [f"{name} ({state_file(codes_from, name).name})" for name in wanted
               if not state_file(codes_from, name).is_file()]
    if missing:
        fail(f"missing head codes in {codes_from}: {', '.join(missing)}")

