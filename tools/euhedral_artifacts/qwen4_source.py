"""The Qwen3.8-Flash-Next (`qwen4_exp`) NVFP4 checkpoint: its tensors, their runtime names and groups, and the
object plan the converter writes.

The checkpoint stores each quantized tensor as three safetensors entries (`<stem>.weight_packed`, `.weight_scale`,
`.weight_scale_2`; routed experts and n-gram shards use `<stem>_packed`, `_scale`, `_scale_2`). A runtime object is
either one tensor (BF16 as is, or NVFP4 repacked into the row-split layout of `Nvfp4Layout`), one n-gram shard
(NVFP4 rows interleaved with their scales), or an expert bank (one record per routed expert). Nothing is
quantized again: every code, scale and global scale is copied unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import mmap
from pathlib import Path
import re
import struct
from typing import Any

from euhedral_artifacts.edrl import align_up, fail
from euhedral_artifacts.nvfp4 import nvfp4_offsets

# Component groups; the ordinals are the artifact's `group` field (ComponentGroup in the Java reader).
GROUPS = (
    "fixed-text",
    "routed-expert",
    "shared-expert",
    "router",
    "gdn",
    "qsa",
    "hyper-connection",
    "token-embedding",
    "output-head",
    "norm-small-state",
    "ngram",
    "mtp",
    "vision",
    "other",
)
GROUP_ORDINAL = {name: ordinal for ordinal, name in enumerate(GROUPS)}

NVFP4_BLOCK = 16
NGRAM_ROW_LAYOUT = "row-interleaved-nvfp4-v1"
ROW_SPLIT_LAYOUT = "row-split-k128-v1"
CONTIGUOUS_LAYOUT = "contiguous-le-v1"
TENSOR_ALIGNMENT = 256
BANK_ALIGNMENT = 4096

SAFETENSORS_DTYPES = {"BF16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1, "I64": 8}


@dataclass
class SourceTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    offset: int  # absolute file offset of the payload
    size: int


class SourceCheckpoint:
    """The tensors of a (sharded) safetensors checkpoint directory, read lazily through memory maps."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.tensors: dict[str, SourceTensor] = {}
        self._maps: dict[Path, mmap.mmap] = {}
        shards = sorted(directory.glob("*.safetensors"))
        if not shards:
            fail(f"no safetensors files in {directory}")
        for shard in shards:
            with shard.open("rb") as handle:
                header_size = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_size))
            base = 8 + header_size
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                low, high = entry["data_offsets"]
                if name in self.tensors:
                    fail(f"tensor {name} appears in two shards")
                shape = tuple(entry["shape"])
                if entry["dtype"] not in SAFETENSORS_DTYPES:
                    fail(f"tensor {name} has unsupported dtype {entry['dtype']}")
                elements = 1
                for dimension in shape:
                    elements *= dimension
                if elements * SAFETENSORS_DTYPES[entry["dtype"]] != high - low:
                    fail(f"tensor {name} byte range does not match its shape")
                self.tensors[name] = SourceTensor(name, entry["dtype"], shape, shard, base + low, high - low)

    def view(self, name: str) -> memoryview:
        tensor = self.tensors[name]
        mapped = self._maps.get(tensor.path)
        if mapped is None:
            with tensor.path.open("rb") as handle:
                mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self._maps[tensor.path] = mapped
        return memoryview(mapped)[tensor.offset:tensor.offset + tensor.size]

    def close(self) -> None:
        for mapped in self._maps.values():
            try:
                mapped.close()
            except BufferError:
                pass
        self._maps.clear()


# --- name mapping ---------------------------------------------------------------------------------------------

_HC_PARTS = {
    "block_inject_weight": "block_inject",
    "hc_norm": "hc_norm",
    "input_mix_weight_down": "input_mix_down",
    "input_mix_weight_up": "input_mix_up",
}

_QUANT_SUFFIXES = (".weight_packed", ".weight_scale", ".weight_scale_2", "_packed", "_scale", "_scale_2")


def split_quantized(name: str) -> tuple[str, str] | None:
    """(stem, part) when `name` is one entry of a quantized triple; part is packed, scale or scale_2."""
    for suffix in (".weight_scale_2", ".weight_scale", ".weight_packed"):
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix[len(".weight_"):]
    for suffix in ("_scale_2", "_scale", "_packed"):
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix[1:]
    return None


def layer_object(prefix: str, local: str) -> tuple[str, str] | None:
    """(runtime name, group) of the layer-local object `local` (the part after `layers.L.`), or None."""
    simple = {
        "linear_attn.A_log": ("gdn/a_log", "norm-small-state"),
        "linear_attn.dt_bias": ("gdn/dt_bias", "norm-small-state"),
        "linear_attn.conv1d.weight": ("gdn/conv1d", "gdn"),
        "linear_attn.in_proj_a.weight": ("gdn/in_proj_a", "gdn"),
        "linear_attn.in_proj_b.weight": ("gdn/in_proj_b", "gdn"),
        "linear_attn.in_proj_qkv": ("gdn/in_proj_qkv", "gdn"),
        "linear_attn.in_proj_z": ("gdn/in_proj_z", "gdn"),
        "linear_attn.out_proj": ("gdn/out_proj", "gdn"),
        "linear_attn.norm.weight": ("gdn/norm", "norm-small-state"),
        "self_attn.q_proj": ("attention/q_proj", "qsa"),
        "self_attn.k_proj": ("attention/k_proj", "qsa"),
        "self_attn.v_proj": ("attention/v_proj", "qsa"),
        "self_attn.o_proj": ("attention/o_proj", "qsa"),
        "self_attn.q_norm.weight": ("attention/q_norm", "norm-small-state"),
        "self_attn.k_norm.weight": ("attention/k_norm", "norm-small-state"),
        "self_attn.indexer.index_qk_proj.weight": ("attention/indexer/index_qk_proj", "qsa"),
        "self_attn.indexer.q_layernorm.weight": ("attention/indexer/q_layernorm", "norm-small-state"),
        "self_attn.indexer.k_layernorm.weight": ("attention/indexer/k_layernorm", "norm-small-state"),
        "mlp.gate.weight": ("moe/router", "router"),
        "mlp.shared_expert.gate_proj": ("moe/shared_expert/gate_proj", "shared-expert"),
        "mlp.shared_expert.up_proj": ("moe/shared_expert/up_proj", "shared-expert"),
        "mlp.shared_expert.down_proj": ("moe/shared_expert/down_proj", "shared-expert"),
        "mlp.shared_expert_gate.weight": ("moe/shared_expert_gate", "shared-expert"),
        "ple.key_proj": ("ple/key_proj", "fixed-text"),
        "ple.value_proj": ("ple/value_proj", "fixed-text"),
        "ple.conv1d.weight": ("ple/conv1d", "fixed-text"),
        "ple.norm_conv.weight": ("ple/norm_conv", "norm-small-state"),
        "ple.norm_key.weight": ("ple/norm_key", "norm-small-state"),
        "ple.norm_query.weight": ("ple/norm_query", "norm-small-state"),
    }
    hit = simple.get(local)
    if hit is not None:
        return f"{prefix}/{hit[0]}", hit[1]
    match = re.fullmatch(r"(attn|mlp)_hyper_connection\.(\w+)\.weight", local)
    if match and match.group(2) in _HC_PARTS:
        return f"{prefix}/{match.group(1)}_hc/{_HC_PARTS[match.group(2)]}", "hyper-connection"
    return None


@dataclass
class Mapped:
    """How one checkpoint stem (an unquantized tensor name, or the stem of a quantized triple) is stored."""

    kind: str  # direct | nvfp4 | ngram | expert | metadata
    name: str
    group: str
    layer: int = -1
    detail: str = ""


def map_stem(stem: str, quantized: bool) -> Mapped | None:
    """Maps a checkpoint stem to its runtime object. `stem` is the full name for an unquantized tensor and
    the name without the quantization suffix for a quantized one."""
    if stem == "lm_head.weight":
        return Mapped("direct", "text/output_head", "output-head")
    if stem == "model.language_model.embed_tokens.weight":
        return Mapped("direct", "text/token_embedding", "token-embedding")
    match = re.fullmatch(r"model\.language_model\.hyper_connection_mixer\.(\w+)\.weight", stem)
    if match and match.group(1) in _HC_PARTS:
        return Mapped("direct", f"text/hyper_connection_mixer/{_HC_PARTS[match.group(1)]}", "hyper-connection")
    match = re.fullmatch(r"model\.visual\.(.+)", stem)
    if match:
        return Mapped("direct", "vision/" + match.group(1).replace(".", "/"), "vision")
    match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", stem)
    if match:
        layer, local = int(match.group(1)), match.group(2)
        prefix = f"text/layers/{layer}"
        if local == "mlp.experts.gate_up_proj" or local == "mlp.experts.down_proj":
            return Mapped("expert", f"{prefix}/moe/experts", "routed-expert", layer, local.rsplit(".", 1)[1])
        shard = re.fullmatch(r"ple\.ple_embedding\.ngram_embedding\.shard_(\d+)", local)
        if shard:
            return Mapped("ngram", f"{prefix}/ple/ngram/shard_{int(shard.group(1)):03d}", "ngram", layer,
                          str(int(shard.group(1))))
        if local in ("ple.ple_embedding.layer_multipliers", "ple.ple_embedding.ngram_heads_offsets",
                     "ple.ple_embedding.ngram_heads_vocab_sizes"):
            return Mapped("metadata", local.rsplit(".", 1)[1], "ngram", layer)
        found = layer_object(prefix, local)
        if found is not None:
            return Mapped("nvfp4" if quantized else "direct", found[0], found[1], layer)
        return None
    match = re.fullmatch(r"mtp\.(.+)", stem)
    if match:
        local = match.group(1)
        simple = {
            "fc_embedding": "mtp/fc_embedding",
            "fc_hidden": "mtp/fc_hidden",
            "pre_fc_norm_embedding.weight": "mtp/pre_fc_norm_embedding",
            "pre_fc_norm_hidden.weight": "mtp/pre_fc_norm_hidden",
        }
        if local in simple:
            return Mapped("nvfp4" if quantized else "direct", simple[local], "mtp")
        hc = re.fullmatch(r"hyper_connection_mixer\.(\w+)\.weight", local)
        if hc and hc.group(1) in _HC_PARTS:
            return Mapped("direct", f"mtp/hyper_connection_mixer/{_HC_PARTS[hc.group(1)]}", "mtp")
        layered = re.fullmatch(r"layers\.(\d+)\.(.+)", local)
        if layered:
            layer, inner = int(layered.group(1)), layered.group(2)
            prefix = f"mtp/layers/{layer}"
            if inner in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                return Mapped("expert", f"{prefix}/moe/experts", "mtp", layer, inner.rsplit(".", 1)[1])
            found = layer_object(prefix, inner)
            if found is not None:
                return Mapped("nvfp4" if quantized else "direct", found[0], "mtp", layer)
        return None
    return None


# --- the plan ---------------------------------------------------------------------------------------------------


@dataclass
class TensorPlan:
    name: str
    group: str
    kind: str  # direct | nvfp4 | ngram
    shape: tuple[int, ...]
    dtype: str  # source metadata dtype
    fmt: str
    layout: str
    size: int
    parts: dict[str, str]  # packed/scale/scale_2 or raw -> checkpoint tensor name
    offset: int = 0
    crc: int = 0


@dataclass
class ProjectionPlan:
    name: str
    shape: tuple[int, int]
    offset: int  # inside an expert record
    size: int


@dataclass
class BankPlan:
    name: str
    group: str
    layer: int
    expert_count: int
    projections: list[ProjectionPlan]
    record_bytes: int
    parts: dict[str, dict[str, str]]  # projection -> packed/scale/scale_2 -> checkpoint name
    offset: int = 0
    crcs: list[int] = field(default_factory=list)


@dataclass
class Plan:
    metadata: dict[str, Any]
    tensors: list[TensorPlan]
    banks: list[BankPlan]


def nvfp4_size(rows: int, k: int) -> int:
    if k % 128 != 0:
        fail(f"row-split NVFP4 needs K divisible by 128, got {k}")
    return nvfp4_offsets((rows, k))[2]


def ngram_row_bytes(k: int) -> int:
    return k // 2 + k // NVFP4_BLOCK


def ngram_size(rows: int, k: int) -> int:
    return align_up(rows * ngram_row_bytes(k), TENSOR_ALIGNMENT) + 4


def expert_record(projection_shapes: dict[str, tuple[int, int]]) -> tuple[list[ProjectionPlan], int]:
    """Projections of one expert record in the order gate_up, down (each 256-aligned) and the 4096-aligned stride."""
    offset = 0
    projections = []
    for name in ("gate_up", "down"):
        rows, k = projection_shapes[name]
        size = nvfp4_size(rows, k)
        projections.append(ProjectionPlan(name, (rows, k), offset, size))
        offset = align_up(offset + size, TENSOR_ALIGNMENT)
    return projections, align_up(offset, BANK_ALIGNMENT)


def build_metadata(config: dict[str, Any], ngram_tables: dict[str, list[int]], ple_layers: list[int],
                   shard_rows: int) -> dict[str, Any]:
    text = config["text_config"]
    vision = config["vision_config"]
    rope = text["rope_parameters"]
    mtp = text["mtp"]
    metadata: dict[str, Any] = {
        "architecture": "qwen4_exp",
        "source.model_type": config["model_type"],
        "text.vocab_size": text["vocab_size"],
        "text.hidden_size": text["hidden_size"],
        "text.num_hidden_layers": text["num_hidden_layers"],
        "text.max_position_embeddings": text["max_position_embeddings"],
        "text.rms_norm_eps": float(text["rms_norm_eps"]),
        "text.hidden_act": text["hidden_act"],
        "text.tie_word_embeddings": bool(text["tie_word_embeddings"]),
        "text.bos_token_id": text["bos_token_id"],
        "text.eos_token_id": text["eos_token_id"],
        "text.layer_types": list(text["layer_types"]),
        "attention.num_heads": text["num_attention_heads"],
        "attention.num_kv_heads": text["num_key_value_heads"],
        "attention.head_dim": text["head_dim"],
        "attention.partial_rotary_factor": float(text["partial_rotary_factor"]),
        "attention.rope_theta": float(rope["rope_theta"]),
        "attention.mrope_section": list(rope["mrope_section"]),
        "attention.mrope_interleaved": bool(rope["mrope_interleaved"]),
        "attention.output_gate": text["output_gate_type"],
        "attention.full_attention_interval": text["full_attention_interval"],
        "gdn.num_key_heads": text["linear_num_key_heads"],
        "gdn.num_value_heads": text["linear_num_value_heads"],
        "gdn.key_head_dim": text["linear_key_head_dim"],
        "gdn.value_head_dim": text["linear_value_head_dim"],
        "gdn.conv_kernel_dim": text["linear_conv_kernel_dim"],
        "gdn.ssm_dtype": text["mamba_ssm_dtype"],
        "qsa.indexer_budget": text["indexer_budget"],
        "qsa.compress_ratio": text["indexer_compress_ratio"],
        "qsa.head_dim": text["indexer_head_dim"],
        "qsa.kv_heads": text["indexer_kv_heads"],
        "qsa.num_heads": text["indexer_n_heads"],
        "moe.num_experts": text["num_experts"],
        "moe.experts_per_token": text["num_experts_per_tok"],
        "moe.intermediate_size": text["moe_intermediate_size"],
        "moe.shared_expert_intermediate_size": text["shared_expert_intermediate_size"],
        "hc.count": text["hc_count"],
        "hc.lowrank": text["hc_lowrank"],
        "ngram.size": text["ngram_size"],
        "ngram.vocab_size_base": text["ngram_vocab_size_base"],
        "ngram.heads_per_ngram": text["heads_per_ngram"],
        "ngram.vocab_divisible_by": text["make_ngram_vocab_size_divisible_by"],
        "ngram.split_parts": text["split_ngram_parts"],
        "ngram.shard_rows": shard_rows,
        "ngram.layer_multipliers": ngram_tables["layer_multipliers"],
        "ngram.heads_offsets": ngram_tables["ngram_heads_offsets"],
        "ngram.heads_vocab_sizes": ngram_tables["ngram_heads_vocab_sizes"],
        "ple.layers": ple_layers,
        "ple.embed_dim": text["ple_embed_dim"],
        "ple.conv_kernel_size": text["ple_conv_kernel_size"],
        "mtp.num_layers": text["mtp_num_hidden_layers"],
        "mtp.hybrid": bool(mtp["hybrid"]),
        "mtp.layer_types": list(mtp["layer_types"]),
        "mtp.rope_theta": float(mtp["rope_theta"]),
        "mtp.use_dedicated_embeddings": bool(text["mtp_use_dedicated_embeddings"]),
        "mtp.use_hidden_state_from_layer": (-1 if mtp["mtp_use_hidden_state_from_layer"] is None
                                            else int(mtp["mtp_use_hidden_state_from_layer"])),
        "vision.depth": vision["depth"],
        "vision.hidden_size": vision["hidden_size"],
        "vision.intermediate_size": vision["intermediate_size"],
        "vision.num_heads": vision["num_heads"],
        "vision.in_channels": vision["in_channels"],
        "vision.patch_size": vision["patch_size"],
        "vision.spatial_merge_size": vision["spatial_merge_size"],
        "vision.temporal_patch_size": vision["temporal_patch_size"],
        "vision.out_hidden_size": vision["out_hidden_size"],
        "vision.num_position_embeddings": vision["num_position_embeddings"],
        "vision.hidden_act": vision["hidden_act"],
        "vision.deepstack_visual_indexes": list(vision["deepstack_visual_indexes"]),
        "vision.image_token_id": config["image_token_id"],
        "vision.video_token_id": config["video_token_id"],
        "vision.start_token_id": config["vision_start_token_id"],
        "vision.end_token_id": config["vision_end_token_id"],
        "quant.format": "nvfp4",
        "quant.block": NVFP4_BLOCK,
        "quant.scale_dtype": "e4m3",
    }
    return metadata


def build_plan(source: SourceCheckpoint) -> Plan:
    """Classifies every checkpoint tensor and returns the objects to write. Rejects a checkpoint that is not the
    NVFP4 `qwen4_exp` layout this converter knows."""
    config = source.config
    if config.get("model_type") != "qwen4_exp" or config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        fail("source is not a Qwen4ExpForConditionalGeneration (qwen4_exp) checkpoint")
    text = config["text_config"]
    tensors: dict[str, TensorPlan] = {}
    triples: dict[str, dict[str, str]] = {}
    mapped_triples: dict[str, Mapped] = {}
    banks: dict[str, dict[str, dict[str, str]]] = {}
    bank_info: dict[str, Mapped] = {}
    ngram_tables: dict[str, list[int]] = {}
    ple_layers: set[int] = set()
    unmapped: list[str] = []

    for name, tensor in source.tensors.items():
        split = split_quantized(name)
        if split is not None and tensor.dtype in ("U8", "F8_E4M3", "F32"):
            stem, part = split
            mapping = map_stem(stem, True)
            if mapping is None:
                unmapped.append(name)
                continue
            if mapping.kind == "expert":
                banks.setdefault(mapping.name, {}).setdefault(mapping.detail, {})[part] = name
                bank_info[mapping.name] = mapping
            else:
                triples.setdefault(stem, {})[part] = name
                mapped_triples[stem] = mapping
            continue
        mapping = map_stem(name, False)
        if mapping is None:
            unmapped.append(name)
            continue
        if mapping.kind == "metadata":
            if tensor.dtype != "I64":
                fail(f"{name} must be I64")
            values = struct.unpack(f"<{tensor.size // 8}q", bytes(source.view(name)))
            ngram_tables[mapping.name] = list(values)
            ple_layers.add(mapping.layer)
            continue
        if tensor.dtype != "BF16":
            fail(f"direct tensor {name} has dtype {tensor.dtype}; expected BF16")
        elements = 1
        for dimension in tensor.shape:
            elements *= dimension
        if mapping.name in tensors:
            fail(f"two checkpoint tensors map to {mapping.name}")
        tensors[mapping.name] = TensorPlan(mapping.name, mapping.group, "direct", tensor.shape, "BF16", "BF16",
                                           CONTIGUOUS_LAYOUT, elements * 2, {"raw": name})
    if unmapped:
        fail(f"{len(unmapped)} checkpoint tensors have no runtime object, for example {unmapped[:5]}")

    shard_rows_seen: set[int] = set()
    for stem, parts in triples.items():
        mapping = mapped_triples[stem]
        if set(parts) != {"packed", "scale", "scale_2"}:
            fail(f"quantized tensor {stem} has parts {sorted(parts)}")
        packed, scale, global_scale = (source.tensors[parts[part]] for part in ("packed", "scale", "scale_2"))
        if len(packed.shape) != 2 or packed.dtype != "U8" or scale.dtype != "F8_E4M3" or global_scale.dtype != "F32":
            fail(f"quantized tensor {stem} has unexpected dtypes or rank")
        rows, packed_k = packed.shape
        k = packed_k * 2
        if scale.shape != (rows, k // NVFP4_BLOCK) or global_scale.shape != (1,):
            fail(f"quantized tensor {stem} has inconsistent scale shapes {scale.shape} {global_scale.shape}")
        if mapping.kind == "ngram":
            shard_rows_seen.add(rows)
            ple_layers.add(mapping.layer)
            tensors[mapping.name] = TensorPlan(mapping.name, mapping.group, "ngram", (rows, k), "BF16", "NVFP4",
                                               NGRAM_ROW_LAYOUT, ngram_size(rows, k), parts)
        else:
            tensors[mapping.name] = TensorPlan(mapping.name, mapping.group, "nvfp4", (rows, k), "BF16", "NVFP4",
                                               ROW_SPLIT_LAYOUT, nvfp4_size(rows, k), parts)
    if len(shard_rows_seen) > 1:
        fail(f"n-gram shards have different row counts: {sorted(shard_rows_seen)}")

    bank_plans: list[BankPlan] = []
    for name, projections in banks.items():
        info = bank_info[name]
        if set(projections) != {"gate_up_proj", "down_proj"}:
            fail(f"expert bank {name} has projections {sorted(projections)}")
        shapes: dict[str, tuple[int, int]] = {}
        expert_count = None
        for projection, parts in projections.items():
            if set(parts) != {"packed", "scale", "scale_2"}:
                fail(f"expert bank {name} {projection} has parts {sorted(parts)}")
            packed, scale, global_scale = (source.tensors[parts[part]] for part in ("packed", "scale", "scale_2"))
            if len(packed.shape) != 3 or packed.dtype != "U8":
                fail(f"expert bank {name} {projection} packed codes must be a rank-3 U8 tensor")
            experts, rows, packed_k = packed.shape
            k = packed_k * 2
            if (scale.shape != (experts, rows, k // NVFP4_BLOCK) or global_scale.shape != (experts,)
                    or scale.dtype != "F8_E4M3" or global_scale.dtype != "F32"):
                fail(f"expert bank {name} {projection} has inconsistent scale shapes")
            if expert_count not in (None, experts):
                fail(f"expert bank {name} projections disagree on the expert count")
            expert_count = experts
            shapes[projection.removesuffix("_proj")] = (rows, k)
        projection_plans, record_bytes = expert_record(shapes)
        bank_plans.append(BankPlan(name, info.group, info.layer, expert_count, projection_plans, record_bytes,
                                   {key.removesuffix("_proj"): value for key, value in projections.items()}))

    ple_sorted = sorted(ple_layers)
    configured = text["ple_layer_ids"]
    # The checkpoint numbers the layers that hold the n-gram tables from 0; its config lists them from 1.
    if [layer + 1 for layer in ple_sorted] != sorted(configured):
        fail(f"n-gram tables are on layers {ple_sorted} but the config's ple_layer_ids are {configured}")
    for key in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
        if key not in ngram_tables:
            fail(f"checkpoint has no {key}")
    shard_rows = shard_rows_seen.pop() if shard_rows_seen else 0
    metadata = build_metadata(config, ngram_tables, ple_sorted, shard_rows)

    def order(plan: TensorPlan) -> tuple[int, int, str]:
        section = {"vision": 4, "mtp": 3, "ngram": 2}.get(plan.group, 0)
        layer = int(re.search(r"layers/(\d+)/", plan.name).group(1)) if re.search(r"layers/(\d+)/", plan.name) else -1
        return section, layer, plan.name

    ordered = sorted(tensors.values(), key=order)
    bank_plans.sort(key=lambda bank: (bank.group == "mtp", bank.layer))
    return Plan(metadata, ordered, bank_plans)


def assign_offsets(plan: Plan, data_offset: int) -> int:
    """Gives every object its file offset (fixed objects first, then banks, n-gram shards, MTP, vision) and returns the
    file size. Tensors are 256-aligned; banks and n-gram shards start on 4096."""
    cursor = data_offset
    fixed = [t for t in plan.tensors if t.group not in ("ngram", "mtp", "vision")]
    ngram = [t for t in plan.tensors if t.group == "ngram"]
    mtp = [t for t in plan.tensors if t.group == "mtp"]
    vision = [t for t in plan.tensors if t.group == "vision"]
    for tensor in fixed:
        tensor.offset = align_up(cursor, TENSOR_ALIGNMENT)
        cursor = tensor.offset + tensor.size
    for bank in [b for b in plan.banks if b.group != "mtp"]:
        bank.offset = align_up(cursor, BANK_ALIGNMENT)
        cursor = bank.offset + bank.record_bytes * bank.expert_count
    for tensor in ngram:
        tensor.offset = align_up(cursor, BANK_ALIGNMENT)
        cursor = tensor.offset + tensor.size
    for tensor in mtp:
        tensor.offset = align_up(cursor, TENSOR_ALIGNMENT)
        cursor = tensor.offset + tensor.size
    for bank in [b for b in plan.banks if b.group == "mtp"]:
        bank.offset = align_up(cursor, BANK_ALIGNMENT)
        cursor = bank.offset + bank.record_bytes * bank.expert_count
    for tensor in vision:
        tensor.offset = align_up(cursor, TENSOR_ALIGNMENT)
        cursor = tensor.offset + tensor.size
    return cursor
