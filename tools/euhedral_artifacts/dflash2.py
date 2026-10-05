"""The DFlash2 drafter (z-lab/Qwen3.8-27B-DFlash2) as artifact objects under `dflash2/`.

An artifact that holds them drafts with DFlash2 instead of MTP. `dflash2/config` (I32) records the drafter's
configuration, so the engine reads no Hugging Face file:

    0 version (1)          6 head dim             12 selector rank          18.. target layers tapped
    1 draft layers         7 block size           13 sliding window
    2 hidden size          8 mask token           14 vocabulary
    3 intermediate size    9 conv kernel          15 RMSNorm epsilon (FP32 bits)
    4 attention heads     10 conv group size      16 RoPE theta (FP32 bits)
    5 KV heads            11 selector top-k       17 tap count

Objects (BF16, contiguous, unless the projections are quantized):

    dflash2/fc                                  [hidden, taps * hidden]  feature fusion
    dflash2/hidden_norm, dflash2/final_norm     [hidden]
    dflash2/layers/N/input_norm, post_attention_norm                     [hidden]
    dflash2/layers/N/attention/query            [heads * head_dim, hidden]
    dflash2/layers/N/attention/key_value        [2 * kv_heads * head_dim, hidden]   k rows, then v rows
    dflash2/layers/N/attention/output           [hidden, heads * head_dim]
    dflash2/layers/N/attention/query_norm, key_norm                      [head_dim]
    dflash2/layers/N/mlp/gate_up                [2 * intermediate, hidden]          gate rows, then up rows
    dflash2/layers/N/mlp/down                   [hidden, intermediate]
    dflash2/layers/N/attention_conv/base, mlp_conv/base                  [2, kernel, hidden]
    dflash2/layers/N/attention_conv/projection, mlp_conv/projection      [2 * kernel * hidden / group, hidden]
    dflash2/selector/hidden_projection          [rank, hidden]
    dflash2/selector/predecessor, successor     [vocabulary, rank]

`projections="nvfp4"` stores the five projections of each layer (query, key_value, output, gate_up, down: the
checkpoint's 35 q/k/v/o/gate/up/down matrices) as plain NVFP4; everything else stays BF16. `"nvfp4-all"` also stores the
feature fusion `fc` and the convolutions' kernel projections as NVFP4.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any

import numpy as np

from euhedral_artifacts.edrl import (FORMAT_ORDINAL, HEADER_SIZE, LAYOUT_ORDINAL, MAGIC, VERSION, ObjectPlan,
                                     assign_offsets, encode_descriptors, encode_table, fail, read_table)
from euhedral_artifacts.nvfp4 import nvfp4_offsets, quantize_nvfp4_matrix
from euhedral_artifacts.sources import SourceStore, concat_matrix, read_json, source_matrix

PREFIX = "dflash2/"
CONFIG_VERSION = 1
PROJECTIONS = ("bf16", "nvfp4", "nvfp4-all")
EXPECTED = {
    "num_hidden_layers": 5,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 248320,
    "sliding_window": 2048,
}


def check_draft(config: dict[str, Any]) -> dict[str, Any]:
    """The drafter configuration, checked against what the engine implements."""
    if config.get("architectures") != ["DFlash2DraftModel"]:
        fail("draft checkpoint is not a DFlash2DraftModel")
    for key, value in EXPECTED.items():
        if config.get(key) != value:
            fail(f"draft checkpoint {key} is {config.get(key)!r}, expected {value!r}")
    if config.get("hidden_act") != "silu" or config.get("attention_bias") or config.get("tie_word_embeddings"):
        fail("draft checkpoint uses an unsupported activation, bias or tied embedding")
    if set(config.get("layer_types", [])) != {"sliding_attention"} or config.get("is_causal") is not False:
        fail("draft checkpoint layers must be non-causal sliding attention")
    rope = config.get("rope_parameters", {})
    if rope.get("rope_type") != "default":
        fail("draft checkpoint RoPE must be the default type")
    draft = config.get("dflash_config", {})
    for key in ("block_size", "mask_token_id", "conv_kernel_size", "conv_group_size", "selector_top_k",
                "selector_rank", "target_layer_ids"):
        if key not in draft:
            fail(f"draft checkpoint dflash_config has no {key}")
    if config["hidden_size"] % draft["conv_group_size"]:
        fail("conv group size must divide the hidden size")
    return config


def config_words(config: dict[str, Any]) -> np.ndarray:
    draft = config["dflash_config"]
    taps = [int(layer) for layer in draft["target_layer_ids"]]
    words = [
        CONFIG_VERSION,
        config["num_hidden_layers"],
        config["hidden_size"],
        config["intermediate_size"],
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["head_dim"],
        draft["block_size"],
        draft["mask_token_id"],
        draft["conv_kernel_size"],
        draft["conv_group_size"],
        draft["selector_top_k"],
        draft["selector_rank"],
        config["sliding_window"],
        config["vocab_size"],
        struct.unpack("<i", struct.pack("<f", float(config["rms_norm_eps"])))[0],
        struct.unpack("<i", struct.pack("<f", float(config["rope_parameters"]["rope_theta"])))[0],
        len(taps),
        *taps,
    ]
    return np.asarray(words, dtype="<i4")


def _raw(store: SourceStore, name: str, source: str, shape: tuple[int, ...]) -> ObjectPlan:
    store.ref(source, shape)
    size = int(np.prod(shape)) * 2
    return ObjectPlan(name, shape, "BF16", "BF16", "contiguous-le-v1", size,
                      lambda output, offset: (output.seek(offset), output.write(store.raw(source, shape))))


def _matrix(name: str, matrix, projections: str) -> ObjectPlan:
    if projections in ("nvfp4", "nvfp4-all"):
        return ObjectPlan(name, matrix.shape, "BF16", "NVFP4", "row-split-k128-v1", nvfp4_offsets(matrix.shape)[2],
                          lambda output, offset, source=matrix: quantize_nvfp4_matrix(output, offset, source))

    def write(output, offset, source=matrix) -> None:
        output.seek(offset)
        for begin in range(0, source.shape[0], 2048):
            rows = source.read_rows(begin, min(source.shape[0], begin + 2048)).astype(np.float32)
            output.write((rows.view(np.uint32) >> 16).astype("<u2").tobytes())

    return ObjectPlan(name, matrix.shape, "BF16", "BF16", "contiguous-le-v1", matrix.shape[0] * matrix.shape[1] * 2,
                      write)


def build_plans(store: SourceStore, config: dict[str, Any], projections: str = "bf16") -> list[ObjectPlan]:
    """The `dflash2/` objects of the drafter checkpoint in `store`."""
    if projections not in PROJECTIONS:
        fail(f"unknown DFlash2 projection format {projections!r}")
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    heads, kv_heads, head_dim = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
    draft = config["dflash_config"]
    taps = len(draft["target_layer_ids"])
    kernel, group = draft["conv_kernel_size"], draft["conv_group_size"]
    rank, vocab = draft["selector_rank"], config["vocab_size"]
    words = config_words(config)
    plans = [
        ObjectPlan(PREFIX + "config", (len(words),), "INT32", "I32", "contiguous-le-v1", 4 * len(words),
                   lambda output, offset: (output.seek(offset), output.write(words.tobytes()))),
        _matrix(PREFIX + "fc", source_matrix(store, "fc.weight", (hidden, taps * hidden)),
                "nvfp4" if projections == "nvfp4-all" else "bf16"),
        _raw(store, PREFIX + "hidden_norm", "hidden_norm.weight", (hidden,)),
        _raw(store, PREFIX + "final_norm", "norm.weight", (hidden,)),
    ]
    for layer in range(config["num_hidden_layers"]):
        source = f"layers.{layer}."
        target = f"{PREFIX}layers/{layer}/"

        def matrix(name: str, rows: int, columns: int):
            return source_matrix(store, source + name, (rows, columns))

        plans += [
            _raw(store, target + "input_norm", source + "input_layernorm.weight", (hidden,)),
            _raw(store, target + "post_attention_norm", source + "post_attention_layernorm.weight", (hidden,)),
            _matrix(target + "attention/query", matrix("self_attn.q_proj.weight", heads * head_dim, hidden),
                    projections),
            _matrix(target + "attention/key_value",
                    concat_matrix(matrix("self_attn.k_proj.weight", kv_heads * head_dim, hidden),
                                  matrix("self_attn.v_proj.weight", kv_heads * head_dim, hidden)),
                    projections),
            _matrix(target + "attention/output", matrix("self_attn.o_proj.weight", hidden, heads * head_dim),
                    projections),
            _raw(store, target + "attention/query_norm", source + "self_attn.q_norm.weight", (head_dim,)),
            _raw(store, target + "attention/key_norm", source + "self_attn.k_norm.weight", (head_dim,)),
            _matrix(target + "mlp/gate_up",
                    concat_matrix(matrix("mlp.gate_proj.weight", intermediate, hidden),
                                  matrix("mlp.up_proj.weight", intermediate, hidden)),
                    projections),
            _matrix(target + "mlp/down", matrix("mlp.down_proj.weight", hidden, intermediate), projections),
        ]
        for conv in ("attention_conv", "mlp_conv"):
            plans += [
                _raw(store, f"{target}{conv}/base", f"{source}{conv}.base_kernel", (2, kernel, hidden)),
                _matrix(f"{target}{conv}/projection",
                        matrix(f"{conv}.kernel_projection.weight", 2 * kernel * hidden // group, hidden),
                        "nvfp4" if projections == "nvfp4-all" else "bf16"),
            ]
    plans += [
        _raw(store, PREFIX + "selector/hidden_projection", "candidate_selector.hidden_projection.weight",
             (rank, hidden)),
        _raw(store, PREFIX + "selector/predecessor", "candidate_selector.predecessor_codebook", (vocab, rank)),
        _raw(store, PREFIX + "selector/successor", "candidate_selector.successor_codebook", (vocab, rank)),
    ]
    return plans


def extend(artifact: Path, draft: Path, output_path: Path, projections: str = "bf16", force: bool = False,
           check: bool = True) -> None:
    """Writes `output_path`: the artifact `artifact` (its metadata and every object, byte for byte) followed by
    the `dflash2/` objects of the drafter checkpoint directory `draft`. `check` holds the drafter to the
    published configuration (tests turn it off for small drafters)."""
    if output_path.exists() and not force:
        fail(f"output already exists; pass --force: {output_path}")
    config = read_json(draft / "config.json")
    if check:
        check_draft(config)
    with artifact.open("rb") as source:
        _, metadata, objects = read_table(source)
    if any(obj["name"].startswith(PREFIX) for obj in objects):
        fail(f"{artifact} already holds DFlash2 objects")
    with SourceStore(draft) as store:
        plans = build_plans(store, config, projections)
        # Old objects keep their order and payloads; the table grows, so every offset is reassigned.
        carried = [ObjectPlan(obj["name"], obj["shape"], "BF16", "BF16", "contiguous-le-v1", obj["bytes"],
                              lambda output, offset: None) for obj in objects]
        table_size = len(encode_descriptors(objects)) + len(encode_table(plans))
        data_base = HEADER_SIZE + len(metadata) + table_size
        file_size = assign_offsets(carried + plans, data_base)
        descriptors = [dict(obj, offset=plan.offset) for obj, plan in zip(objects, carried)]
        table = encode_descriptors(descriptors) + encode_table(plans)
        header = struct.pack(">iiqqqiiq", MAGIC, VERSION, HEADER_SIZE, len(metadata), HEADER_SIZE + len(metadata),
                             len(objects) + len(plans), 0, data_base)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with temporary.open("w+b") as output, artifact.open("rb") as source:
                output.truncate(file_size)
                output.seek(0)
                output.write(header)
                output.write(metadata)
                output.write(table)
                for obj, plan in zip(objects, carried):
                    source.seek(obj["offset"])
                    output.seek(plan.offset)
                    remaining = obj["bytes"]
                    while remaining:
                        chunk = source.read(min(remaining, 64 << 20))
                        output.write(chunk)
                        remaining -= len(chunk)
                for index, plan in enumerate(plans):
                    plan.writer(output, plan.offset)
                    if index % 16 == 0 or index == len(plans) - 1:
                        print(f"dflash2 {index + 1}/{len(plans)} {plan.name}", flush=True)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, output_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
