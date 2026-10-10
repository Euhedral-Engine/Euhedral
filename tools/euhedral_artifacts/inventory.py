"""The object inventory of the Qwen3.8-27B checkpoint and the writers that fill each object."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct
from typing import Any

import numpy as np

from euhedral_artifacts.edrl import ObjectPlan, checked_product, fail, read_table, utf8
from euhedral_artifacts.grouped import QUANT, quantize_matrix, row_split_size
from euhedral_artifacts.importance import ImportanceMatrix
from euhedral_artifacts.nvfp4 import (nvfp4_offsets, nvfp4_sd4_offsets, quantize_nvfp4_matrix,
                                      quantize_nvfp4_sd4_matrix)
from euhedral_artifacts.q3_p2e2 import P2E2_LOWEST_CODE
from euhedral_artifacts.recipes import LAYOUT_SD4, NVFP4_FORMAT, Q3_FORMAT, Recipe, storage
from euhedral_artifacts.sources import (HIDDEN, VOCAB_SIZE, MatrixSource, SourceStore, bf16_to_float32,
                                        concat_matrix, gather_matrix, head_part, read_json, slice_matrix,
                                        source_matrix)

FULL_ATTENTION_LAYERS = frozenset(range(3, 64, 4))
TOKENIZER_VOCAB_SIZE = 248077
DRAFT_ROWS = 131072
INTERMEDIATE = 17408
OBJECT_COUNT = 785


def direct_size(shape: tuple[int, ...], format_name: str) -> int:
    return checked_product(shape, "shape") * (2 if format_name == "BF16" else 4)


def payload_size(shape: tuple[int, ...], format_name: str) -> int:
    if format_name == NVFP4_FORMAT:
        return nvfp4_offsets(shape)[2]
    return row_split_size(shape, format_name) if format_name in QUANT else direct_size(shape, format_name)


def write_direct(output, base_offset: int, store: SourceStore, action: tuple[str, Any]) -> None:
    kind, value = action
    if kind == "raw":
        output.seek(base_offset)
        output.write(store.raw(value[0], value[1]))
        return
    if kind == "transpose":
        words = store.words(value[0], value[1]).reshape(value[2]).transpose(value[3]).copy()
        output.seek(base_offset)
        output.write(words.astype("<u2", copy=False).tobytes())
        return
    if kind == "fp32":
        values = bf16_to_float32(store.words(value[0], value[1])).astype("<f4", copy=False)
        output.seek(base_offset)
        output.write(np.ascontiguousarray(values).tobytes())
        return
    if kind == "i32":
        output.seek(base_offset)
        output.write(np.asarray(value, dtype="<i4").tobytes())
        return
    fail(f"unknown direct action {kind}")


def add_direct(plans: list[ObjectPlan], name: str, shape: tuple[int, ...], store: SourceStore,
               source_name: str, source_shape: tuple[int, ...], kind: str = "raw", extra: Any = None,
               format_name: str = "BF16", source_dtype: str = "BF16") -> None:
    if format_name not in ("BF16", "FP32", "I32"):
        fail(f"direct object {name} has unsupported format {format_name}")
    if kind == "i32":
        value = extra
        writer = lambda output, offset: write_direct(output, offset, store, ("i32", value))
    else:
        action_value = (
            (source_name, source_shape, (source_shape[0], source_shape[2]), extra)
            if kind == "transpose"
            else (source_name, source_shape, extra)
        )
        writer = lambda output, offset, action=(kind, action_value): write_direct(output, offset, store, action)
    plans.append(ObjectPlan(name, shape, source_dtype, format_name, "contiguous-le-v1", direct_size(shape, format_name), writer))




def add_quant(plans: list[ObjectPlan], recipe: Recipe, name: str, matrix: MatrixSource, q3_format: str,
              importance: ImportanceMatrix | None = None) -> None:
    """Adds the quantized object `name`, stored as `recipe` says; `q3_format` is its format in the q3 artifact.
    With `importance` it is rounded by calibrated scale search."""
    format_name, layout = storage(recipe, name, q3_format)
    calibration = importance.calibration(name, matrix.shape[1]) if importance is not None else None
    if calibration is not None and recipe.quantization == "q3" and recipe.compressed and format_name == Q3_FORMAT:
        # Compressed q3 stores its Q3 tensors in P2E2, which holds the codes -3..3 (docs/COMPRESSED_Q3.md).
        calibration = replace(calibration, lowest_code=P2E2_LOWEST_CODE)
    if layout == LAYOUT_SD4:
        plans.append(ObjectPlan(
            name, matrix.shape, "BF16", format_name, layout, nvfp4_sd4_offsets(matrix.shape)[2],
            lambda output, offset, source=matrix: quantize_nvfp4_sd4_matrix(output, offset, source, calibration),
        ))
    elif format_name == NVFP4_FORMAT:
        plans.append(ObjectPlan(
            name, matrix.shape, "BF16", format_name, layout, payload_size(matrix.shape, format_name),
            lambda output, offset, source=matrix: quantize_nvfp4_matrix(output, offset, source, calibration),
        ))
    elif format_name in QUANT:
        plans.append(ObjectPlan(
            name, matrix.shape, "BF16", format_name, layout, row_split_size(matrix.shape, format_name),
            lambda output, offset, source=matrix, fmt=format_name: quantize_matrix(output, offset, source, fmt,
                                                                                    calibration),
        ))
    else:
        fail(f"unknown quantized format {format_name}")


def attention_parts(store: SourceStore, prefix: str) -> tuple[MatrixSource, MatrixSource]:
    q_name = prefix + "self_attn.q_proj.weight"
    return head_part(store, q_name, False), head_part(store, q_name, True)


def build_plans(store: SourceStore, selected: np.ndarray, recipe: Recipe,
                importance: ImportanceMatrix | None = None) -> list[ObjectPlan]:
    """The 785 text and MTP objects. Formats written here are the q3 artifact's; `recipe` maps them to
    the artifact being built. With `importance`, quantized objects are rounded by calibrated scale search."""
    plans: list[ObjectPlan] = []

    def quant(name: str, matrix: MatrixSource, q3_format: str) -> None:
        add_quant(plans, recipe, name, matrix, q3_format, importance)

    quant("text/token_embedding", source_matrix(store, "model.language_model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN)), "Q3G64_F16S")
    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        add_direct(plans, object_prefix + "input_norm", (HIDDEN,), store, source_prefix + "input_layernorm.weight", (HIDDEN,))
        if layer in FULL_ATTENTION_LAYERS:
            query, gate = attention_parts(store, source_prefix)
            quant(object_prefix + "attention/query_key", concat_matrix(query, source_matrix(store, source_prefix + "self_attn.k_proj.weight", (1024, HIDDEN))), "Q4G64_F16S")
            quant(object_prefix + "attention/gate_value", concat_matrix(gate, source_matrix(store, source_prefix + "self_attn.v_proj.weight", (1024, HIDDEN))), "Q5G64_F16S")
            add_direct(plans, object_prefix + "attention/query_norm", (256,), store, source_prefix + "self_attn.q_norm.weight", (256,))
            add_direct(plans, object_prefix + "attention/key_norm", (256,), store, source_prefix + "self_attn.k_norm.weight", (256,))
            quant(object_prefix + "attention/output", source_matrix(store, source_prefix + "self_attn.o_proj.weight", (HIDDEN, 6144)), "Q3G64_F16S")
        else:
            qkv = source_matrix(store, source_prefix + "linear_attn.in_proj_qkv.weight", (10240, HIDDEN))
            add_direct(plans, object_prefix + "gdn/a_log", (48,), store, source_prefix + "linear_attn.A_log", (48,), kind="fp32", format_name="FP32")
            add_direct(plans, object_prefix + "gdn/dt_bias", (48,), store, source_prefix + "linear_attn.dt_bias", (48,), kind="fp32", format_name="FP32")
            add_direct(plans, object_prefix + "gdn/convolution", (4, 10240), store, source_prefix + "linear_attn.conv1d.weight", (10240, 1, 4), kind="transpose", extra=(1, 0))
            add_direct(plans, object_prefix + "gdn/a_projection", (48, HIDDEN), store, source_prefix + "linear_attn.in_proj_a.weight", (48, HIDDEN))
            add_direct(plans, object_prefix + "gdn/b_projection", (48, HIDDEN), store, source_prefix + "linear_attn.in_proj_b.weight", (48, HIDDEN))
            quant(object_prefix + "gdn/query_key", slice_matrix(qkv, 0, 4096), "Q4G64_F16S")
            quant(object_prefix + "gdn/value_z", concat_matrix(slice_matrix(qkv, 4096, 10240), source_matrix(store, source_prefix + "linear_attn.in_proj_z.weight", (6144, HIDDEN))), "Q5G64_F16S")
            add_direct(plans, object_prefix + "gdn/norm", (128,), store, source_prefix + "linear_attn.norm.weight", (128,))
            quant(object_prefix + "gdn/output", source_matrix(store, source_prefix + "linear_attn.out_proj.weight", (HIDDEN, 6144)), "Q3G64_F16S")
        add_direct(plans, object_prefix + "post_attention_norm", (HIDDEN,), store, source_prefix + "post_attention_layernorm.weight", (HIDDEN,))
        quant(object_prefix + "mlp/gate_up", concat_matrix(
            source_matrix(store, source_prefix + "mlp.gate_proj.weight", (INTERMEDIATE, HIDDEN)),
            source_matrix(store, source_prefix + "mlp.up_proj.weight", (INTERMEDIATE, HIDDEN))), "Q3G64_F16S")
        quant(object_prefix + "mlp/down", source_matrix(store, source_prefix + "mlp.down_proj.weight", (HIDDEN, INTERMEDIATE)), "Q3G64_F16S")
    add_direct(plans, "text/final_norm", (HIDDEN,), store, "model.language_model.norm.weight", (HIDDEN,))
    quant("text/output_head", source_matrix(store, "lm_head.weight", (VOCAB_SIZE, HIDDEN)), "Q3G64_F16S")
    quant("text/draft_head", gather_matrix(store, "lm_head.weight", selected), "Q3G64_F16S")
    add_direct(plans, "text/draft_head_token_ids", (DRAFT_ROWS,), store, "", (), kind="i32", extra=selected, format_name="I32", source_dtype="INT32")

    # The MTP layer is NVFP4 in the q3 artifact: Q3 drafts accept fewer tokens per step and catch up slower
    # on multi-row verification (docs/MTP_VERIFIER.md).
    mtp = "mtp.layers.0."
    quant("mtp/input_projection", source_matrix(store, "mtp.fc.weight", (HIDDEN, 10240)), "NVFP4")
    add_direct(plans, "mtp/embedding_norm", (HIDDEN,), store, "mtp.pre_fc_norm_embedding.weight", (HIDDEN,))
    add_direct(plans, "mtp/hidden_norm", (HIDDEN,), store, "mtp.pre_fc_norm_hidden.weight", (HIDDEN,))
    query, gate = attention_parts(store, mtp)
    add_direct(plans, "mtp/layer/input_norm", (HIDDEN,), store, mtp + "input_layernorm.weight", (HIDDEN,))
    quant("mtp/layer/attention/query_key_gate_value", concat_matrix(
        query,
        source_matrix(store, mtp + "self_attn.k_proj.weight", (1024, HIDDEN)),
        gate,
        source_matrix(store, mtp + "self_attn.v_proj.weight", (1024, HIDDEN))), "NVFP4")
    add_direct(plans, "mtp/layer/attention/query_norm", (256,), store, mtp + "self_attn.q_norm.weight", (256,))
    add_direct(plans, "mtp/layer/attention/key_norm", (256,), store, mtp + "self_attn.k_norm.weight", (256,))
    quant("mtp/layer/attention/output", source_matrix(store, mtp + "self_attn.o_proj.weight", (HIDDEN, 6144)), "NVFP4")
    add_direct(plans, "mtp/layer/post_attention_norm", (HIDDEN,), store, mtp + "post_attention_layernorm.weight", (HIDDEN,))
    quant("mtp/layer/mlp/gate_up", concat_matrix(
        source_matrix(store, mtp + "mlp.gate_proj.weight", (INTERMEDIATE, HIDDEN)),
        source_matrix(store, mtp + "mlp.up_proj.weight", (INTERMEDIATE, HIDDEN))), "NVFP4")
    quant("mtp/layer/mlp/down", source_matrix(store, mtp + "mlp.down_proj.weight", (HIDDEN, INTERMEDIATE)), "NVFP4")
    add_direct(plans, "mtp/final_norm", (HIDDEN,), store, "mtp.norm.weight", (HIDDEN,))
    if len(plans) != OBJECT_COUNT:
        fail(f"inventory produced {len(plans)} objects, expected {OBJECT_COUNT}")
    return plans


def shortlist(model: Path, ranking_path: Path) -> np.ndarray:
    counts = np.fromfile(ranking_path, dtype="<i8", count=VOCAB_SIZE)
    if counts.size != VOCAB_SIZE:
        fail("ranking file does not contain the complete vocabulary row")
    tokenizer = read_json(model / "tokenizer_config.json")
    forced = sorted(
        int(token_id)
        for token_id, value in tokenizer.get("added_tokens_decoder", {}).items()
        if isinstance(value, dict) and value.get("special", False) and 0 <= int(token_id) < TOKENIZER_VOCAB_SIZE
    )
    order = np.argsort(-counts[:TOKENIZER_VOCAB_SIZE], kind="stable")
    forced_set = set(forced)
    wanted = DRAFT_ROWS - len(forced)
    picked: list[int] = []
    for token_id in order.tolist():
        if token_id not in forced_set:
            picked.append(token_id)
            if len(picked) == wanted:
                break
    selected = np.asarray(picked + forced, dtype=np.int64)
    selected = selected[np.argsort(-counts[selected], kind="stable")]
    if selected.size != DRAFT_ROWS or np.unique(selected).size != DRAFT_ROWS:
        fail("draft shortlist is not exactly 131072 unique rows")
    return selected


def encode_metadata(config: dict[str, Any]) -> bytes:
    text = config["text_config"]
    layer_types = [0 if i in FULL_ATTENTION_LAYERS else 1 for i in range(64)]
    result = bytearray()
    values = [
        (">i", text["vocab_size"]), (">i", text["hidden_size"]), (">i", text["num_hidden_layers"]),
        (">i", text["num_attention_heads"]), (">i", text["num_key_value_heads"]), (">i", text["head_dim"]),
        (">i", text["intermediate_size"]), (">i", text["linear_num_key_heads"]), (">i", text["linear_num_value_heads"]),
        (">i", text["linear_key_head_dim"]), (">i", text["linear_value_head_dim"]), (">i", text["linear_conv_kernel_dim"]),
        (">d", text["rms_norm_eps"]), (">d", text["rope_parameters"]["rope_theta"]),
        (">d", text["partial_rotary_factor"]), (">i", text["max_position_embeddings"]),
    ]
    for fmt, value in values:
        result.extend(struct.pack(fmt, value))
    activation = utf8(text["hidden_act"], "hidden activation")
    result.extend(struct.pack(">i", len(activation)))
    result.extend(activation)
    result.extend(struct.pack(">i", len(layer_types)))
    for layer_type in layer_types:
        result.extend(struct.pack(">i", layer_type))
    result.extend(struct.pack(">iiii", 0, 0, 0, 0))
    result.extend(struct.pack(">BBi", 0, 1, 1))
    return bytes(result)


def draft_token_ids(artifact: Path) -> np.ndarray:
    """The draft-head shortlist stored in an existing artifact (text/draft_head_token_ids)."""
    with artifact.open("rb") as handle:
        _, _, objects = read_table(handle)
        for obj in objects:
            if obj["name"] == "text/draft_head_token_ids":
                if obj["shape"] != (DRAFT_ROWS,) or obj["bytes"] != DRAFT_ROWS * 4:
                    fail("draft token id object has an unexpected shape")
                handle.seek(obj["offset"])
                selected = np.frombuffer(handle.read(obj["bytes"]), dtype="<i4").astype(np.int64)
                if np.unique(selected).size != DRAFT_ROWS:
                    fail("draft token ids are not unique")
                return selected
    fail(f"{artifact} has no text/draft_head_token_ids")
