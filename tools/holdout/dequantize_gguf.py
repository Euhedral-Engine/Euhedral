#!/usr/bin/env python3
"""Write an artifact's dequantized weights into a copy of its checkpoint's BF16 GGUF (weight-only attribution).

Usage: dequantize_gguf.py ARTIFACT.edrl TEMPLATE-BF16.gguf OUT.gguf [--keep REGEX]

Scoring OUT.gguf with `llama-perplexity --kl-divergence` measures the artifact's weight rounding alone: llama.cpp runs
it with its own KV cache and arithmetic, so the engine's NVFP4 KV cache, FP4 prefill activations and relaxed kernels
are left out (docs/QUALITY.md). TEMPLATE must be the BF16 GGUF of the checkpoint the artifact was converted from
(llama.cpp's convert_hf_to_gguf.py --outtype bf16): it is copied, and the data of every tensor the artifact quantizes
is overwritten in place with the artifact's values rounded to BF16 (metadata, norms and other tensors stay as they
are). Fused objects are split into llama.cpp's tensors with llama.cpp's head order. Every written tensor is checked
against the template: its relative RMS difference is the rounding error (a few to tens of percent), and above
MAX_RELATIVE_ERROR the mapping is wrong and the tool stops. The MTP layer and the draft head are not written.
--keep leaves the llama.cpp tensors whose names match REGEX at their BF16 values: the per-family sweep restores one
family at a time and reads its share of the KL from the difference.
Needs NumPy. Every file is read with ordinary reads, never memory-mapped.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from euhedral_artifacts import edrl, nvfp4, q3_p2e2  # noqa: E402
from euhedral_artifacts.importance import gguf_tensors  # noqa: E402
from euhedral_artifacts.edrl import align_up  # noqa: E402

MAX_RELATIVE_ERROR = 0.6
GGML_TYPE_BF16 = 30
ROW_CHUNK = 16384
# Qwen3.8-27B geometry (config.json text_config).
HEADS, HEAD_DIM = 24, 256
K_HEADS, V_HEADS, V_HEAD_DIM = 16, 48, 128
INTERMEDIATE = 17408


def formats() -> dict[int, str]:
    return {ordinal: name for name, ordinal in edrl.FORMAT_ORDINAL.items()}


def layouts() -> dict[int, str]:
    return {ordinal: name for name, ordinal in edrl.LAYOUT_ORDINAL.items()}


def signed(values: np.ndarray, bits: int) -> np.ndarray:
    values = values.astype(np.int16)
    return np.where(values >= 1 << (bits - 1), values - (1 << bits), values)


def grouped_rows(data: np.ndarray, fmt: str, rows: int, k: int, begin: int, end: int) -> np.ndarray:
    """Rows begin..end of a row-split Q3/Q4/Q5 G64 tensor as float32 [end - begin, k]."""
    bits = {"Q3G64_F16S": 3, "Q4G64_F16S": 4, "Q5G64_F16S": 5}[fmt]
    k_pad = align_up(k, 128)
    groups = k_pad // 64
    base_group = 24 if bits == 3 else 32
    high_group = 8 if bits == 5 else 0
    high_offset = align_up(rows * groups * base_group, 256)
    scale_offset = high_offset + align_up(rows * groups * high_group, 256)
    n = end - begin
    base = data[begin * groups * base_group:end * groups * base_group]
    if bits == 3:
        codes = q3_p2e2.unpack_q3(base, n, k_pad).astype(np.int16)
    else:
        nibbles = base.reshape(n, k_pad // 2)
        low = np.empty((n, k_pad), np.int16)
        low[:, 0::2], low[:, 1::2] = nibbles & 15, nibbles >> 4
        if bits == 4:
            codes = signed(low, 4)
        else:
            high = data[high_offset + begin * groups * 8:high_offset + end * groups * 8].reshape(n, groups, 8)
            fifth = np.unpackbits(high, axis=2, bitorder="little").reshape(n, k_pad).astype(np.int16)
            codes = signed(low | fifth << 4, 5)
    scales = data[scale_offset + begin * groups * 2:scale_offset + end * groups * 2].view("<f2").reshape(n, groups)
    values = codes.reshape(n, groups, 64).astype(np.float32) * scales.astype(np.float32)[..., None]
    return values.reshape(n, k_pad)[:, :k]


def nvfp4_rows(data: np.ndarray, layout: str, rows: int, k: int, begin: int, end: int) -> np.ndarray:
    k_pad = align_up(k, 128)
    packed = data[begin * k_pad // 2:end * k_pad // 2].reshape(end - begin, k_pad // 2)
    if layout == "row-split-k128-sd4-v1":
        index_offset, table_offset, _ = nvfp4.nvfp4_sd4_offsets((rows, k))
        table = data[table_offset:table_offset + 16]
        global_scale = data[table_offset + 16:table_offset + 20].view("<f4")[0]
        indices = data[index_offset + begin * k_pad // 32:index_offset + end * k_pad // 32].reshape(end - begin, k_pad // 32)
        scales = nvfp4.expand_nvfp4_sd4(indices, table)
    else:
        scale_offset, global_offset, _ = nvfp4.nvfp4_offsets((rows, k))
        global_scale = data[global_offset:global_offset + 4].view("<f4")[0]
        scales = data[scale_offset + begin * k_pad // 16:scale_offset + end * k_pad // 16].reshape(end - begin, k_pad // 16)
    return nvfp4.dequantize_nvfp4_rows(packed, scales, global_scale)[:, :k]


def dequantize(handle, obj: dict, fmt: str, layout: str) -> np.ndarray:
    rows, k = obj["shape"]
    handle.seek(obj["offset"])
    data = np.frombuffer(handle.read(obj["bytes"]), np.uint8)
    if layout == "row-split-p2e2-v1":
        data = np.frombuffer(q3_p2e2.decode(data, rows, k), np.uint8)
    out = np.empty((rows, k), np.float32)
    for begin in range(0, rows, ROW_CHUNK):
        end = min(rows, begin + ROW_CHUNK)
        out[begin:end] = (nvfp4_rows(data, layout, rows, k, begin, end) if fmt == "NVFP4"
                          else grouped_rows(data, fmt, rows, k, begin, end))
    return out


def reorder_v_heads(tensor: np.ndarray, axis: int) -> np.ndarray:
    """llama.cpp's _reorder_v_heads: V heads grouped by K head -> tiled, along `axis`."""
    shape = list(tensor.shape)
    per_k = V_HEADS // K_HEADS
    split = shape[:axis] + [K_HEADS, per_k, V_HEAD_DIM] + shape[axis + 1:]
    return np.swapaxes(tensor.reshape(split), axis, axis + 1).reshape(shape)


def gguf_tensors(name: str, values: np.ndarray, pending: dict) -> list[tuple[str, np.ndarray]]:
    """The llama.cpp tensors of artifact object `name` (fused attention objects wait for their partner)."""
    if name == "text/token_embedding":
        return [("token_embd.weight", values)]
    if name == "text/output_head":
        return [("output.weight", values)]
    match = re.fullmatch(r"text/layers/(\d+)/(.+)", name)
    if not match:
        return []
    layer, part = match.group(1), match.group(2)
    blk = f"blk.{layer}."
    if part == "mlp/gate_up":
        return [(blk + "ffn_gate.weight", values[:INTERMEDIATE]), (blk + "ffn_up.weight", values[INTERMEDIATE:])]
    if part == "mlp/down":
        return [(blk + "ffn_down.weight", values)]
    if part == "attention/output":
        return [(blk + "attn_output.weight", values)]
    if part == "gdn/output":
        return [(blk + "ssm_out.weight", reorder_v_heads(values, 1))]
    if part in ("attention/query_key", "attention/gate_value", "gdn/query_key", "gdn/value_z"):
        family = part.split("/")[0]
        pending.setdefault((layer, family), {})[part] = values
        both = pending[(layer, family)]
        if len(both) < 2:
            return []
        del pending[(layer, family)]
        if family == "attention":
            query_key, gate_value = both["attention/query_key"], both["attention/gate_value"]
            q = HEADS * HEAD_DIM
            query = query_key[:q].reshape(HEADS, HEAD_DIM, -1)
            gate = gate_value[:q].reshape(HEADS, HEAD_DIM, -1)
            q_proj = np.concatenate([query, gate], axis=1).reshape(2 * q, -1)
            return [(blk + "attn_q.weight", q_proj), (blk + "attn_k.weight", query_key[q:]),
                    (blk + "attn_v.weight", gate_value[q:])]
        query_key, value_z = both["gdn/query_key"], both["gdn/value_z"]
        v = V_HEADS * V_HEAD_DIM
        qkv = np.concatenate([query_key, reorder_v_heads(value_z[:v], 0)])
        return [(blk + "attn_qkv.weight", qkv), (blk + "attn_gate.weight", reorder_v_heads(value_z[v:], 0))]
    return []


def to_bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 bit patterns, to nearest even."""
    bits = np.ascontiguousarray(values, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")


def main(argv: list[str]) -> int:
    keep = None
    if "--keep" in argv:
        at = argv.index("--keep")
        keep = re.compile(argv[at + 1])
        argv = argv[:at] + argv[at + 2:]
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    artifact, template, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    tensors = gguf_tensors(template)
    print(f"copying {template} to {out}", flush=True)
    shutil.copyfile(template, out)
    names, kinds = formats(), layouts()
    pending, written = {}, 0
    with artifact.open("rb") as handle, out.open("r+b") as sink, template.open("rb") as source:
        _, _, objects = edrl.read_table(handle)
        for obj in objects:
            fmt, layout = names[obj["format"]], kinds[obj["layout"]]
            if fmt not in ("Q3G64_F16S", "Q4G64_F16S", "Q5G64_F16S", "NVFP4") or obj["name"].startswith("mtp/") \
                    or obj["name"] == "text/draft_head":
                continue
            for gguf_name, values in gguf_tensors(obj["name"], dequantize(handle, obj, fmt, layout), pending):
                if keep is not None and keep.search(gguf_name):
                    print(f"{gguf_name:32s} kept at BF16", flush=True)
                    continue
                target = tensors[gguf_name]
                if target.kind != GGML_TYPE_BF16 or target.elements != values.size:
                    raise SystemExit(f"{gguf_name}: template is type {target.kind} x {target.elements}, "
                                     f"artifact gives {values.shape}")
                source.seek(target.offset)
                original = np.frombuffer(source.read(2 * values.size), "<u2").reshape(values.shape)
                original = (original.astype(np.uint32) << 16).view(np.float32)
                error = float(np.sqrt(np.mean((values - original) ** 2) / np.mean(original.astype(np.float64) ** 2)))
                print(f"{gguf_name:32s} {fmt:11s} {layout:22s} relative RMS error {error:.4f}", flush=True)
                if error > MAX_RELATIVE_ERROR:
                    raise SystemExit(f"{gguf_name}: relative error {error:.3f}; the object-to-tensor mapping is wrong")
                sink.seek(target.offset)
                sink.write(to_bf16(values).tobytes())
                written += 1
    if pending:
        raise SystemExit(f"unpaired fused objects: {sorted(pending)}")
    print(f"wrote {written} tensors into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
