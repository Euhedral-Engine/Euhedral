"""Tests for the Flash-Next converter on a miniature checkpoint with the real tensor names and layouts."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from euhedral_artifacts import qwen4_edrl as edrl  # noqa: E402
from euhedral_artifacts.nvfp4 import nvfp4_offsets  # noqa: E402
from euhedral_artifacts.qwen4_source import SourceCheckpoint, build_plan  # noqa: E402

HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 256, 4, 2, 64
GDN_KEY_HEADS, GDN_VALUE_HEADS, GDN_DIM = 2, 4, 64
EXPERTS, MOE, LAYERS, HC, LOWRANK = 8, 128, 4, 2, 32
NGRAM_K, NGRAM_SHARDS, NGRAM_ROWS = 32, 4, 11


def write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header, offset = {}, 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for _, _, data in tensors.values():
            handle.write(data)


class Maker:
    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)
        self.tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}

    def bf16(self, name: str, *shape: int) -> None:
        words = self.rng.integers(0x3C00, 0x4000, size=shape, dtype=np.uint16)
        self.tensors[name] = ("BF16", shape, words.astype("<u2").tobytes())

    def quant(self, stem: str, rows: int, k: int, experts: int | None = None, dotted: bool = True) -> None:
        lead = (experts,) if experts else ()
        packed = self.rng.integers(0, 256, size=lead + (rows, k // 2), dtype=np.uint8)
        scales = self.rng.integers(0, 0x7F, size=lead + (rows, k // 16), dtype=np.uint8)
        global_scale = self.rng.random(size=(experts or 1,)).astype("<f4") + 0.01
        sep = ".weight_" if dotted else "_"
        self.tensors[stem + sep + "packed"] = ("U8", packed.shape, packed.tobytes())
        self.tensors[stem + sep + "scale"] = ("F8_E4M3", scales.shape, scales.tobytes())
        self.tensors[stem + sep + "scale_2"] = ("F32", global_scale.shape, global_scale.tobytes())

    def i64(self, name: str, values: list[int]) -> None:
        self.tensors[name] = ("I64", (len(values),), struct.pack(f"<{len(values)}q", *values))


def config() -> dict:
    types = ["linear_attention", "linear_attention", "full_attention", "linear_attention"]
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
        "image_token_id": 9, "video_token_id": 10, "vision_start_token_id": 11, "vision_end_token_id": 12,
        "text_config": {
            "full_attention_interval": 4, "hc_count": HC, "hc_lowrank": LOWRANK, "head_dim": HEAD_DIM,
            "heads_per_ngram": 1, "hidden_act": "silu", "hidden_size": HIDDEN, "indexer_budget": 16,
            "indexer_compress_ratio": 4, "indexer_head_dim": 64, "indexer_kv_heads": 1, "indexer_n_heads": 2,
            "layer_types": types, "linear_conv_kernel_dim": 4, "linear_key_head_dim": GDN_DIM,
            "linear_num_key_heads": GDN_KEY_HEADS, "linear_num_value_heads": GDN_VALUE_HEADS,
            "linear_value_head_dim": GDN_DIM, "make_ngram_vocab_size_divisible_by": 4, "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 4096, "moe_intermediate_size": MOE,
            "mtp": {"hybrid": True, "layer_types": ["full_attention"], "mtp_use_hidden_state_from_layer": None,
                    "num_hidden_layers": 1, "rope_theta": 10000000},
            "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False, "ngram_size": 3,
            "ngram_vocab_size_base": 40, "num_attention_heads": HEADS, "num_experts": EXPERTS,
            "num_experts_per_tok": 2, "num_hidden_layers": LAYERS, "num_key_value_heads": KV_HEADS,
            "output_gate_type": "sigmoid", "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4,
            "ple_embed_dim": HIDDEN, "ple_layer_ids": [2],
            "rms_norm_eps": 1e-6, "rope_parameters": {"mrope_interleaved": True, "mrope_section": [3, 3, 2],
                                                      "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                                                      "rope_type": "default"},
            "shared_expert_intermediate_size": MOE, "split_ngram_parts": NGRAM_SHARDS, "tie_word_embeddings": False,
            "vocab_size": 64, "bos_token_id": 1, "eos_token_id": 2,
        },
        "vision_config": {"deepstack_visual_indexes": [], "depth": 1, "hidden_act": "gelu_pytorch_tanh",
                          "hidden_size": 32, "in_channels": 3, "intermediate_size": 64, "num_heads": 2,
                          "num_position_embeddings": 16, "out_hidden_size": HIDDEN, "patch_size": 4,
                          "spatial_merge_size": 2, "temporal_patch_size": 2},
    }


def moe_layer(m: Maker, prefix: str, experts_name: str) -> None:
    m.bf16(f"{prefix}.mlp.gate.weight", EXPERTS, HIDDEN)
    m.quant(f"{prefix}.mlp.experts.gate_up_proj", 2 * MOE, HIDDEN, EXPERTS, dotted=False)
    m.quant(f"{prefix}.mlp.experts.down_proj", HIDDEN, MOE, EXPERTS, dotted=False)
    m.quant(f"{prefix}.mlp.shared_expert.gate_proj", MOE, HIDDEN)
    m.quant(f"{prefix}.mlp.shared_expert.up_proj", MOE, HIDDEN)
    m.quant(f"{prefix}.mlp.shared_expert.down_proj", HIDDEN, MOE)
    m.bf16(f"{prefix}.mlp.shared_expert_gate.weight", 1, HIDDEN)


def hyper(m: Maker, prefix: str, inject: bool = True) -> None:
    width = HC * HIDDEN
    m.bf16(f"{prefix}.hc_norm.weight", width)
    m.bf16(f"{prefix}.input_mix_weight_down.weight", LOWRANK, width)
    m.bf16(f"{prefix}.input_mix_weight_up.weight", width, LOWRANK)
    if inject:
        m.bf16(f"{prefix}.block_inject_weight.weight", HC, width)


def attention(m: Maker, prefix: str) -> None:
    m.quant(f"{prefix}.self_attn.q_proj", HEADS * HEAD_DIM * 2, HIDDEN)
    m.quant(f"{prefix}.self_attn.k_proj", KV_HEADS * HEAD_DIM, HIDDEN)
    m.quant(f"{prefix}.self_attn.v_proj", KV_HEADS * HEAD_DIM, HIDDEN)
    m.quant(f"{prefix}.self_attn.o_proj", HIDDEN, HEADS * HEAD_DIM)
    m.bf16(f"{prefix}.self_attn.q_norm.weight", HEAD_DIM)
    m.bf16(f"{prefix}.self_attn.k_norm.weight", HEAD_DIM)
    m.bf16(f"{prefix}.self_attn.indexer.index_qk_proj.weight", 3 * 64, HIDDEN)
    m.bf16(f"{prefix}.self_attn.indexer.q_layernorm.weight", 64)
    m.bf16(f"{prefix}.self_attn.indexer.k_layernorm.weight", 64)


def make_checkpoint(directory: Path, seed: int = 7) -> None:
    m = Maker(seed)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config()))
    m.bf16("lm_head.weight", 64, HIDDEN)
    m.bf16("model.language_model.embed_tokens.weight", 64, HIDDEN)
    hyper(m, "model.language_model.hyper_connection_mixer", inject=False)
    qkv = 2 * GDN_KEY_HEADS * GDN_DIM + GDN_VALUE_HEADS * GDN_DIM
    for layer in range(LAYERS):
        prefix = f"model.language_model.layers.{layer}"
        hyper(m, f"{prefix}.attn_hyper_connection")
        hyper(m, f"{prefix}.mlp_hyper_connection")
        if layer == 2:
            attention(m, prefix)
        else:
            m.bf16(f"{prefix}.linear_attn.A_log", GDN_VALUE_HEADS)
            m.bf16(f"{prefix}.linear_attn.dt_bias", GDN_VALUE_HEADS)
            m.bf16(f"{prefix}.linear_attn.conv1d.weight", qkv, 1, 4)
            m.bf16(f"{prefix}.linear_attn.in_proj_a.weight", GDN_VALUE_HEADS, HIDDEN)
            m.bf16(f"{prefix}.linear_attn.in_proj_b.weight", GDN_VALUE_HEADS, HIDDEN)
            m.quant(f"{prefix}.linear_attn.in_proj_qkv", qkv, HIDDEN)
            m.quant(f"{prefix}.linear_attn.in_proj_z", GDN_VALUE_HEADS * GDN_DIM, HIDDEN)
            m.quant(f"{prefix}.linear_attn.out_proj", HIDDEN, GDN_VALUE_HEADS * GDN_DIM)
            m.bf16(f"{prefix}.linear_attn.norm.weight", GDN_DIM)
        moe_layer(m, prefix, "experts")
    ple = "model.language_model.layers.1.ple"
    m.quant(f"{ple}.key_proj", HC * HIDDEN, HIDDEN)
    m.quant(f"{ple}.value_proj", HIDDEN, HIDDEN)
    m.bf16(f"{ple}.conv1d.weight", HC * HIDDEN, 1, 4)
    for norm in ("norm_conv", "norm_key", "norm_query"):
        m.bf16(f"{ple}.{norm}.weight", HC * HIDDEN)
    m.i64(f"{ple}.ple_embedding.layer_multipliers", [10**13 + 1, 10**13 + 2, 10**13 + 3])
    m.i64(f"{ple}.ple_embedding.ngram_heads_offsets", [0, 41])
    m.i64(f"{ple}.ple_embedding.ngram_heads_vocab_sizes", [41, 43])
    for shard in range(NGRAM_SHARDS):
        m.quant(f"{ple}.ple_embedding.ngram_embedding.shard_{shard}", NGRAM_ROWS, NGRAM_K, dotted=False)
        # n-gram shards use the dotted ".weight_*" suffix in the real checkpoint
        for part in ("packed", "scale", "scale_2"):
            m.tensors[f"{ple}.ple_embedding.ngram_embedding.shard_{shard}.weight_{part}"] = m.tensors.pop(
                f"{ple}.ple_embedding.ngram_embedding.shard_{shard}_{part}")
    m.quant("mtp.fc_embedding", HIDDEN, HIDDEN)
    m.quant("mtp.fc_hidden", HIDDEN, HIDDEN)
    m.bf16("mtp.pre_fc_norm_embedding.weight", HIDDEN)
    m.bf16("mtp.pre_fc_norm_hidden.weight", HC * HIDDEN)
    hyper(m, "mtp.hyper_connection_mixer", inject=False)
    hyper(m, "mtp.layers.0.attn_hyper_connection")
    hyper(m, "mtp.layers.0.mlp_hyper_connection")
    attention(m, "mtp.layers.0")
    moe_layer(m, "mtp.layers.0", "experts")
    m.bf16("model.visual.pos_embed.weight", 16, 32)
    m.bf16("model.visual.blocks.0.attn.qkv.bias", 96)
    names = sorted(m.tensors)
    half = len(names) // 2
    write_safetensors(directory / "model-00001-of-00002.safetensors", {n: m.tensors[n] for n in names[:half]})
    write_safetensors(directory / "model-00002-of-00002.safetensors", {n: m.tensors[n] for n in names[half:]})


class FlashNextConverterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "checkpoint"
        make_checkpoint(self.checkpoint)
        self.quiet = lambda message: None

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def convert(self, name: str = "model.edrl", jobs: int = 1) -> Path:
        output = self.root / name
        edrl.convert(self.checkpoint, output, jobs=jobs, log=self.quiet)
        return output

    def test_plan_classifies_every_tensor(self) -> None:
        source = SourceCheckpoint(self.checkpoint)
        plan = build_plan(source)
        groups = {t.group for t in plan.tensors}
        self.assertTrue({"gdn", "qsa", "router", "hyper-connection", "ngram", "mtp", "vision", "token-embedding",
                         "output-head", "shared-expert", "norm-small-state", "fixed-text"} <= groups)
        self.assertEqual(len(plan.banks), LAYERS + 1)
        self.assertEqual(plan.metadata["ple.layers"], [1])
        self.assertEqual(plan.metadata["ngram.heads_offsets"], [0, 41])
        source.close()

    def test_conversion_is_valid_and_matches_the_checkpoint(self) -> None:
        artifact = self.convert(jobs=2)
        result = edrl.verify(artifact, self.checkpoint, jobs=2, log=self.quiet)
        self.assertGreater(result["compared_bytes"], 0)
        manifest = json.loads(Path(f"{artifact}.manifest.json").read_text())
        self.assertEqual(manifest["expert_bank_count"], LAYERS + 1)
        self.assertEqual(manifest["expert_record_count"], (LAYERS + 1) * EXPERTS)
        self.assertEqual(manifest["artifact_bytes"], artifact.stat().st_size)
        self.assertEqual(len(manifest["sha256"]), 64)
        self.assertFalse(Path(f"{artifact}.partial").exists())

    def test_expert_record_decodes_like_the_checkpoint(self) -> None:
        artifact = self.convert()
        parsed = edrl.read_artifact(artifact)
        bank = next(b for b in parsed["banks"] if b["name"] == "text/layers/3/moe/experts")
        source = SourceCheckpoint(self.checkpoint)
        packed = np.frombuffer(source.view("model.language_model.layers.3.mlp.experts.down_proj_packed"), np.uint8)
        scales = np.frombuffer(source.view("model.language_model.layers.3.mlp.experts.down_proj_scale"), np.uint8)
        global_scale = np.frombuffer(source.view("model.language_model.layers.3.mlp.experts.down_proj_scale_2"),
                                     "<f4")
        down = next(p for p in bank["projections"] if p["name"] == "down")
        rows, k = down["shape"]
        scale_offset, global_offset, size = nvfp4_offsets((rows, k))
        self.assertEqual(size, down["bytes"])
        with artifact.open("rb") as handle:
            for expert in (0, 5, EXPERTS - 1):
                offset, record_bytes, _ = bank["index"][expert]
                handle.seek(offset + down["offset"])
                blob = handle.read(size)
                self.assertEqual(blob[:rows * k // 2],
                                 packed.reshape(EXPERTS, -1)[expert].tobytes())
                self.assertEqual(blob[scale_offset:scale_offset + rows * k // 16],
                                 scales.reshape(EXPERTS, -1)[expert].tobytes())
                self.assertEqual(struct.unpack("<f", blob[global_offset:global_offset + 4])[0],
                                 float(global_scale[expert]))
                self.assertEqual(record_bytes, edrl.read_artifact(artifact)["banks"][0]["index"][0][1])
        source.close()

    def test_ngram_rows_are_self_contained(self) -> None:
        artifact = self.convert()
        parsed = edrl.read_artifact(artifact)
        shard = next(t for t in parsed["tensors"] if t["name"] == "text/layers/1/ple/ngram/shard_002")
        source = SourceCheckpoint(self.checkpoint)
        base = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_2.weight_"
        packed = np.frombuffer(source.view(base + "packed"), np.uint8).reshape(NGRAM_ROWS, NGRAM_K // 2)
        scales = np.frombuffer(source.view(base + "scale"), np.uint8).reshape(NGRAM_ROWS, NGRAM_K // 16)
        row_bytes = NGRAM_K // 2 + NGRAM_K // 16
        with artifact.open("rb") as handle:
            handle.seek(shard["offset"])
            blob = handle.read(shard["bytes"])
        for row in range(NGRAM_ROWS):
            self.assertEqual(blob[row * row_bytes:row * row_bytes + NGRAM_K // 2], packed[row].tobytes())
            self.assertEqual(blob[row * row_bytes + NGRAM_K // 2:(row + 1) * row_bytes], scales[row].tobytes())
        source.close()

    def test_verify_detects_a_flipped_byte(self) -> None:
        artifact = self.convert()
        parsed = edrl.read_artifact(artifact)
        victim = parsed["banks"][1]["index"][3][0] + 100
        with artifact.open("r+b") as handle:
            handle.seek(victim)
            byte = handle.read(1)
            handle.seek(victim)
            handle.write(bytes([byte[0] ^ 0x10]))
        with self.assertRaises(ValueError):
            edrl.verify(artifact, None, jobs=1, log=self.quiet)

    def test_interrupted_conversion_resumes_to_identical_bytes(self) -> None:
        reference = self.convert("reference.edrl")
        output = self.root / "resumed.edrl"
        original = edrl._work
        calls = {"count": 0}

        def flaky(item):
            calls["count"] += 1
            if calls["count"] == 6:
                raise RuntimeError("simulated crash")
            return original(item)

        edrl._work = flaky
        try:
            with self.assertRaises(RuntimeError):
                edrl.convert(self.checkpoint, output, jobs=1, log=self.quiet)
        finally:
            edrl._work = original
        self.assertFalse(output.exists())
        self.assertTrue(Path(f"{output}.partial").exists())
        edrl.convert(self.checkpoint, output, jobs=1, log=self.quiet)
        self.assertEqual(output.read_bytes(), reference.read_bytes())

    def test_rejects_a_checkpoint_with_an_unknown_tensor(self) -> None:
        source_file = self.checkpoint / "model-00002-of-00002.safetensors"
        extra = self.checkpoint / "model-00003-of-00003.safetensors"
        write_safetensors(extra, {"model.language_model.mystery.weight": ("BF16", (2,), b"\0\0\0\0")})
        with self.assertRaises(ValueError):
            edrl.convert(self.checkpoint, self.root / "bad.edrl", jobs=1, log=self.quiet)
        self.assertTrue(source_file.exists())

    def test_rejects_an_incomplete_quantized_triple(self) -> None:
        shard = self.checkpoint / "model-00003-of-00003.safetensors"
        write_safetensors(shard, {"model.language_model.layers.0.self_attn.q_proj.weight_packed":
                                  ("U8", (2, 2), b"\0\0\0\0")})
        with self.assertRaises(ValueError):
            edrl.convert(self.checkpoint, self.root / "bad.edrl", jobs=1, log=self.quiet)


if __name__ == "__main__":
    unittest.main()
