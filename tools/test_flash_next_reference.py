"""Tests of the Flash-Next reference harness (tools/flash_next_reference/).

    PYTHONPATH=tools ~/.cache/flash-next-ref-venv/bin/python -m unittest tools/test_flash_next_reference.py

Everything runs on a tiny synthetic `Qwen4ExpTextConfig` or on random packed tensors, except `RealArtifactTest`, which
skips when the artifact or a GPU is missing.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from euhedral_artifacts.nvfp4 import dequantize_nvfp4_rows  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as upstream  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextExperts, Qwen4ExpTextModel  # noqa: E402

from flash_next_reference import kvcodec  # noqa: E402
from flash_next_reference.artifact import DEFAULT_ARTIFACT, Artifact, ExpertRecord  # noqa: E402
from flash_next_reference.instrument import (Capture, RecordingPLELayer, RecordingQSAIndexer, ReplayMismatch,  # noqa: E402
                                             stable_topk)
from flash_next_reference.model import (IdsOnlyEmbedding, LazyExperts, make_ngram_module,  # noqa: E402
                                        ngram_tables_from_metadata)
from flash_next_reference.nvfp4 import ExactNvfp4Linear, PackedNvfp4, configure_exact_math, expand_nvfp4, exact_linear  # noqa: E402
from flash_next_reference.record import Recorder, load_tensor, read_manifest  # noqa: E402

CUDA = torch.cuda.is_available()


def random_packed(rows: int, k: int, seed: int, device="cpu") -> PackedNvfp4:
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (rows, k // 2), generator=g, dtype=torch.uint8)
    scales = torch.randint(30, 80, (rows, k // 16), generator=g, dtype=torch.uint8)  # positive finite E4M3 codes
    return PackedNvfp4(codes.to(device), scales.to(device), 0.0123, (rows, k))


def tiny_config(**overrides) -> Qwen4ExpTextConfig:
    kwargs = dict(
        vocab_size=300, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=8, num_experts_per_tok=3, hc_count=4,
        hc_lowrank=16, ple_layer_ids=[2], ple_embed_dim=64, ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=128, split_ngram_parts=4, indexer_n_heads=2, indexer_kv_heads=1,
        indexer_head_dim=32, indexer_budget=16, indexer_compress_ratio=4, eos_token_id=7,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_section": [3, 2, 3], "mrope_interleaved": True}, output_gate_type="sigmoid")
    kwargs.update(overrides)
    cfg = Qwen4ExpTextConfig(**kwargs)
    cfg._attn_implementation = "eager"
    return cfg


def tiny_flash(seed: int = 0):
    """A tiny upstream model with randomly initialised weights, dressed like `FlashNext` for `Capture`."""
    torch.manual_seed(seed)
    cfg = tiny_config()
    model = Qwen4ExpTextModel(cfg).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 1:  # norms start at zero: give them some life
                parameter.add_(0.1 * torch.randn_like(parameter))
        for layer in model.layers:
            if layer.ple is not None:
                layer.ple.conv1d.weight.normal_(0, 0.3)
    layers = list(range(cfg.num_hidden_layers))
    return SimpleNamespace(model=model, cfg=cfg, by_index={n: model.layers[n] for n in layers}, full=True, layers=layers,
                           check_rows=[])


# --- NVFP4 -----------------------------------------------------------------------------------------------------------


class Nvfp4Test(unittest.TestCase):
    def test_expansion_equals_the_repos_dequantizer(self):
        packed = random_packed(48, 256, 1)
        expected = dequantize_nvfp4_rows(packed.codes.numpy(), packed.scales.numpy(), np.float32(packed.global_scale))
        self.assertTrue(np.array_equal(expected.view(np.uint32), packed.expand().numpy().view(np.uint32)))
        # even k in the LOW nibble
        codes = torch.zeros(1, 8, dtype=torch.uint8)
        codes[0, 0] = 0x42  # low nibble 2 (1.0) for k=0, high nibble 4 (2.0) for k=1
        scales = torch.full((1, 1), 0x38, dtype=torch.uint8)  # E4M3 1.0
        row = expand_nvfp4(codes, scales, 1.0)[0]
        self.assertEqual(row[:2].tolist(), [1.0, 2.0])

    def test_per_row_global_scales(self):
        packed = random_packed(5, 160, 2)
        g = torch.tensor([0.5, 1.0, 2.0, 0.25, 4.0])
        rows = expand_nvfp4(packed.codes, packed.scales, g)
        for i in range(5):
            self.assertTrue(torch.equal(rows[i], expand_nvfp4(packed.codes[i:i + 1], packed.scales[i:i + 1], float(g[i]))[0]))

    def test_exact_linear_semantics(self):
        configure_exact_math()
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        packed = random_packed(96, 256, 3)
        layer = ExactNvfp4Linear(packed)
        x = torch.randn(7, 256, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)
        out = layer(x)
        self.assertEqual(out.dtype, torch.bfloat16)
        weights = packed.expand()
        exact = (x.double() @ weights.double().T).to(torch.bfloat16)
        agree = (out == exact).float().mean().item()
        self.assertGreater(agree, 0.995)  # fp32 accumulation vs exact: equal after one bf16 rounding but for rare ulps
        manual = F.linear(x.float(), weights).to(torch.bfloat16)
        self.assertTrue(torch.equal(out, manual))
        self.assertTrue(torch.equal(exact_linear(x, weights), out))
        self.assertEqual(layer.in_features, 256)
        self.assertEqual(layer.out_features, 96)

    @unittest.skipUnless(CUDA, "needs a GPU")
    def test_cuda_expansion_equals_cpu(self):
        packed = random_packed(64, 384, 5)
        on_gpu = PackedNvfp4(packed.codes.cuda(), packed.scales.cuda(), packed.global_scale, packed.shape)
        self.assertTrue(torch.equal(packed.expand(), on_gpu.expand().cpu()))


# --- KV codec --------------------------------------------------------------------------------------------------------

MAGNITUDES = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
SCALES = np.array([(b & 7) * 2.0 ** -9 if b < 8 else (1 + (b & 7) / 8) * 2.0 ** ((b >> 3) - 7) for b in range(127)])
H256 = np.array([[(-1.0 if (i & j).bit_count() & 1 else 1.0) / 16 for j in range(256)] for i in range(256)])


def nearest_even(values, representable):
    clipped = np.clip(np.asarray(values), representable[0], representable[-1])
    distance = abs(clipped[..., None] - representable)
    minimum = distance.min(axis=-1, keepdims=True)
    rank = np.arange(len(representable)) % 2
    return np.argmin(np.where(distance == minimum, rank, 2), axis=-1)


def java_reference_row(row: np.ndarray) -> np.ndarray:
    """core/.../Nvfp4KvReference.represented: H from its definition, float64 rotation, nearest-even choices."""
    rotated = H256 @ row.astype(np.float64)
    for base in range(0, 256, 16):
        maximum = np.abs(rotated[base:base + 16]).max()
        if maximum == 0:
            continue
        scale = SCALES[nearest_even(np.clip(np.float32(maximum) / np.float32(6), 1 / 512, 448), SCALES)]
        divided = rotated[base:base + 16].astype(np.float32) / np.float32(scale)
        rotated[base:base + 16] = np.copysign(MAGNITUDES[nearest_even(np.abs(divided), MAGNITUDES)], divided) * scale
    return H256 @ rotated


class KvCodecTest(unittest.TestCase):
    def test_hadamard_equals_the_definition_and_is_self_inverse(self):
        rows = torch.randn(9, 256, generator=torch.Generator().manual_seed(1))
        forward = kvcodec.hadamard256(rows)
        self.assertLess(np.abs(forward.numpy() - rows.numpy().astype(np.float64) @ H256).max(), 1e-5)
        self.assertLess((kvcodec.hadamard256(forward) - rows).abs().max().item(), 1e-5)
        basis = kvcodec.hadamard256(torch.eye(256) * 16)
        self.assertTrue(np.array_equal(basis.numpy(), H256.astype(np.float32) * 16))

    def test_encode_ties_go_to_the_even_code_and_saturate(self):
        mids = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float32)
        positive = np.concatenate(([0., 1e-30, 1e30], np.nextafter(mids, -np.inf), mids, np.nextafter(mids, np.inf)))
        values = np.concatenate((positive, -positive)).astype(np.float32)
        expected = nearest_even(np.abs(values), MAGNITUDES).astype(np.uint8) | (np.signbit(values).astype(np.uint8) * 8)
        self.assertTrue(np.array_equal(kvcodec.e2m1_encode(torch.from_numpy(values)).numpy(), expected))

    def test_scale_clamps_and_zero_groups(self):
        rows = torch.zeros(3, 256)
        rows[1, 0] = 1e-30  # rotated max / 6 is far below 2^-9: the scale code clamps to the smallest (1)
        rows[2, 0] = 3e38   # saturates at 448 (code 126)
        codes, scales = kvcodec.quantize_rows(rows.to(torch.bfloat16))
        self.assertEqual(scales[0].tolist(), [0] * 16)
        self.assertEqual(codes[0].sum().item(), 0)
        self.assertEqual(scales[1].tolist(), [1] * 16)
        self.assertEqual(scales[2].tolist(), [126] * 16)
        represented = kvcodec.dequantize_rows(codes, scales)
        self.assertEqual(represented[0].abs().sum().item(), 0)

    def test_roundtrip_matches_the_java_reference_semantics(self):
        rng = np.random.default_rng(7)
        scale = rng.choice([0.01, 1.0, 30.0], size=(64, 1))
        rows = (rng.standard_normal((64, 256)) * scale).astype(np.float32)
        rows = torch.from_numpy(rows).to(torch.bfloat16).float().numpy()
        ours = kvcodec.roundtrip(torch.from_numpy(rows)).numpy()
        reference = np.stack([java_reference_row(r) for r in rows])
        # the device butterfly is float32 and the reference float64: they only disagree on (rare) exact ties
        close = np.abs(ours - reference) <= 1e-4 * np.abs(rows).max(axis=1, keepdims=True)
        self.assertGreater(close.mean(), 0.995)
        relative = np.linalg.norm(ours - rows) / np.linalg.norm(rows)
        self.assertLess(relative, 0.15)  # 4-bit codes
        self.assertGreater(relative, 0.01)

    def test_cache_roundtrip_keeps_dtype_and_shape(self):
        states = torch.randn(1, 2, 5, 256).to(torch.bfloat16)
        out = kvcodec.roundtrip_cache(states)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, states.shape)

    def test_pack_layout_is_144_bytes_low_nibble_even(self):
        row = torch.randn(3, 256).to(torch.bfloat16)
        packed = kvcodec.pack_rows(row)
        self.assertEqual(packed.shape, (3, 144))
        codes, scales = kvcodec.quantize_rows(row)
        self.assertTrue(torch.equal(packed[:, :128] & 15, codes[:, 0::2]))
        self.assertTrue(torch.equal(packed[:, :128] >> 4, codes[:, 1::2]))
        self.assertTrue(torch.equal(packed[:, 128:], scales))

    def test_bit_exact_against_the_cuda_codec(self):
        """Runs native/src/attention/nvfp4_kv.cuh through the native test harness (NVRTC) in a subprocess."""
        root = TOOLS.parent
        if not (root / "build/cuda-dev/linux-x64/runtime/libnvrtc.so.13").is_file():
            self.skipTest("pinned NVRTC runtime not built")
        script = r'''
import sys, contextlib, ctypes as C
sys.path.insert(0, sys.argv[1] + "/native/tests")
import numpy as np
from gpu_harness import Gpu
P, U = C.c_uint64, C.c_uint
src = b"""#include "attention/nvfp4_kv.cuh"
extern "C" __global__ void probe_row(const __nv_bfloat16* in, unsigned char* codes, unsigned char* scales, float* fwd, float* inv, unsigned int rows) {
    __shared__ float scratch[4][256];
    const unsigned int warp = threadIdx.x / 32, lane = threadIdx.x & 31, row = blockIdx.x * 4 + warp;
    if (row >= rows) return;
    nvfp4kv::quantize_row(in + row * 256, codes + row * 128, scales + row * 16, scratch[warp], lane);
    float values[8];
    for (int r = 0; r < 8; r++) values[r] = __bfloat162float(in[row * 256 + lane + 32 * r]);
    nvfp4kv::hadamard256(values, lane);
    for (int r = 0; r < 8; r++) fwd[row * 256 + lane + 32 * r] = values[r];
    nvfp4kv::hadamard256(values, lane);
    for (int r = 0; r < 8; r++) inv[row * 256 + lane + 32 * r] = values[r];
}
"""
gpu = Gpu(src)
rng = np.random.default_rng(5)
rows = (rng.standard_normal((300, 256)) * rng.choice([0.01, 1, 30], size=(300, 1))).astype(np.float32)
rows[3] = 0; rows[4, :16] = 0
bits = (rows.view(np.uint32) >> 16).astype(np.uint16)
n = len(rows)
a = gpu.upload(bits.tobytes()); codes = gpu.zeros(n * 128); scales = gpu.zeros(n * 16); fwd = gpu.zeros(n * 1024); inv = gpu.zeros(n * 1024)
gpu.launch("probe_row", (n + 3) // 4, [P(a), P(codes), P(scales), P(fwd), P(inv), U(n)], block=128)
c = np.frombuffer(gpu.download(codes, n * 128), np.uint8).reshape(n, 128)
sc = np.frombuffer(gpu.download(scales, n * 16), np.uint8).reshape(n, 16)
f = np.frombuffer(gpu.download(fwd, n * 1024), np.float32).reshape(n, 256)
i = np.frombuffer(gpu.download(inv, n * 1024), np.float32).reshape(n, 256)
gpu.close()
np.savez(sys.argv[2], bits=bits, codes=c, scales=sc, fwd=f, inv=i)
'''
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "twin.npz"
            run = subprocess.run([sys.executable, "-c", script, str(root), str(out)], capture_output=True, text=True)
            if run.returncode != 0:
                self.skipTest("CUDA twin unavailable: " + run.stderr[-300:])
            twin = np.load(out)
        x = torch.from_numpy((twin["bits"].astype(np.uint32) << 16).view(np.float32)).to(torch.bfloat16)
        forward = kvcodec.hadamard256(x.float())
        self.assertTrue(np.array_equal(forward.numpy().view(np.uint32), twin["fwd"].view(np.uint32)))
        self.assertTrue(np.array_equal(kvcodec.hadamard256(forward).numpy().view(np.uint32), twin["inv"].view(np.uint32)))
        packed = kvcodec.pack_rows(x).numpy()
        self.assertTrue(np.array_equal(packed[:, :128], twin["codes"]))
        self.assertTrue(np.array_equal(packed[:, 128:], twin["scales"]))


# --- recorder ----------------------------------------------------------------------------------------------------------


class RecorderTest(unittest.TestCase):
    def test_manifest_roundtrip_dedup_light_and_mute(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = Recorder(Path(tmp) / "case", {"k": 1})
            rec.begin("c0")
            a = torch.randn(3, 4).to(torch.bfloat16)
            rec.add("L0/out", a)
            rec.add("L1/in", a.clone())  # same bytes: one file
            rec.add("ids", torch.arange(5, dtype=torch.int64))
            rec.add("flags", torch.tensor([True, False]))
            with rec.mute():
                self.assertFalse(rec.add("hidden", a))
            rec.light = {"L0/out"}
            rec.begin("c1", detail=False)
            self.assertTrue(rec.add("L0/out", a))
            self.assertFalse(rec.add("L1/in", a))
            with self.assertRaises(RuntimeError):
                rec.add("L0/out", a)
            total = rec.finish({"extra": 2})
            manifest = read_manifest(Path(tmp) / "case")
            entries = manifest["tensors"]
            self.assertEqual(entries["c0/L0/out"]["file"], entries["c0/L1/in"]["file"])
            self.assertEqual(entries["c0/L0/out"]["dtype"], "bf16")
            self.assertEqual(entries["c0/ids"]["dtype"], "i64")
            self.assertEqual(entries["c0/flags"]["dtype"], "u8")
            self.assertEqual(manifest["metadata"]["k"], 1)
            self.assertEqual(manifest["metadata"]["extra"], 2)
            self.assertEqual(manifest["metadata"]["total_bytes"], total)
            self.assertTrue(torch.equal(load_tensor(Path(tmp) / "case", manifest, "c0/L1/in"), a))
            self.assertTrue(torch.equal(load_tensor(Path(tmp) / "case", manifest, "c0/ids"), torch.arange(5)))
            self.assertNotIn("c0/hidden", entries)
            raw = (Path(tmp) / "case" / entries["c0/L0/out"]["file"]).read_bytes()
            self.assertEqual(raw, a.view(torch.int16).numpy().tobytes())  # raw little-endian


# --- the upstream pieces -------------------------------------------------------------------------------------------------


def pytest_values(row):
    return sorted(row.tolist(), reverse=True)[:5]


class SelectionRuleTest(unittest.TestCase):
    def test_stable_topk_orders_ties_by_ascending_expert_and_flags_boundary_ties(self):
        probs = torch.tensor([[0.1, 0.3, 0.3, 0.05, 0.3, 0.1, 0.2, 0.0],   # three-way tie at the top, tie 0.1/0.1 at the boundary
                              [0.4, 0.3, 0.2, 0.05, 0.02, 0.01, 0.01, 0.01]])
        index, values, tie = stable_topk(probs, 5)
        self.assertEqual(index[0].tolist(), [1, 2, 4, 6, 0])  # descending; ties ascending; 0.1 at index 0 beats index 5
        self.assertEqual(tie.tolist(), [True, False])  # row 0: 5th (0.1) == 6th (0.1)
        self.assertEqual(values[1].tolist(), pytest_values(probs[1]))
        self.assertEqual(index[1].tolist(), [0, 1, 2, 3, 4])


class NgramTest(unittest.TestCase):
    def setUp(self):
        self.cfg = tiny_config()

    def metadata(self):
        cfg = self.cfg
        heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram
        sizes = [upstream._find_nth_prime_after(cfg.ngram_vocab_size_base - 1, i + 1) for i in range(heads)]
        offsets = [sum(sizes[:i]) for i in range(heads)]
        multipliers = upstream._build_layer_multipliers(cfg.vocab_size, cfg.ngram_size, 0, cfg.seed).tolist()
        return {"ngram.layer_multipliers": multipliers, "ngram.heads_offsets": offsets, "ngram.heads_vocab_sizes": sizes}

    def test_tables_must_equal_upstreams_constructor_helpers(self):
        md = self.metadata()
        ngram_tables_from_metadata(md, self.cfg)
        bad = dict(md, **{"ngram.layer_multipliers": [1, 2, 3]})
        with self.assertRaises(AssertionError):
            ngram_tables_from_metadata(bad, self.cfg)
        bad = dict(md, **{"ngram.heads_offsets": md["ngram.heads_offsets"][:-1] + [5]})
        with self.assertRaises(AssertionError):
            ngram_tables_from_metadata(bad, self.cfg)

    def test_ids_match_a_scalar_implementation_and_the_cache_protocol(self):
        md = self.metadata()
        module = make_ngram_module(self.cfg, md, IdsOnlyEmbedding(), layer_idx=1)
        eos = self.cfg.eos_token_id
        tokens = [5, 9, eos, 11, 12, 13, eos, eos, 14, 2, 3]
        ids = torch.tensor(tokens)[None]
        one_shot = module(ids, None)[0]
        mult, sizes, offsets = md["ngram.layer_multipliers"], md["ngram.heads_vocab_sizes"], md["ngram.heads_offsets"]

        def previous(position, shift):  # token `shift` back inside the same EOS-delimited segment, else EOS
            start = 0
            for i in range(position - 1, -1, -1):
                if tokens[i] == eos:
                    start = i + 1
                    break
            return tokens[position - shift] if position - shift >= start and position - shift >= 0 else eos

        for position in range(len(tokens)):
            for ngram in (2, 3):
                mixed = tokens[position] * mult[0]
                for shift in range(1, ngram):
                    mixed ^= previous(position, shift) * mult[shift]
                for head in range(8):
                    column = (ngram - 2) * 8 + head
                    self.assertEqual(int(one_shot[position, column]), mixed % sizes[column] + offsets[column],
                                     (position, ngram, head))
        for chunks in ([4, 7], [1] * 11, [2, 1, 8], [11]):
            cache = DynamicCache(config=self.cfg)
            parts, start = [], 0
            for rows in chunks:
                parts.append(module(ids[:, start:start + rows], cache)[0])
                start += rows
                context = cache.layers[1].conv_states[2]
                self.assertEqual(context.shape, (1, 2))
                self.assertEqual(context[0].tolist(), tokens[max(0, start - 2):start] if start >= 2 else [eos] + tokens[:start])
            self.assertTrue(torch.equal(torch.cat(parts), one_shot), chunks)


class ExpertsTest(unittest.TestCase):
    class FakeArtifact:
        def __init__(self, hidden, inner, experts):
            self.records = {e: ExpertRecord(random_packed(2 * inner, hidden, 10 + e), random_packed(hidden, inner, 50 + e))
                            for e in range(experts)}

        def expert(self, layer, expert, device):
            return self.records[expert]

    def test_lazy_experts_equal_upstreams_loop_on_the_same_expanded_weights(self):
        torch.manual_seed(0)
        cfg = tiny_config()
        art = self.FakeArtifact(cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts)
        lazy = LazyExperts(cfg, art, 0, "cpu")
        dense = Qwen4ExpTextExperts(cfg)
        with torch.no_grad():
            dense.gate_up_proj.copy_(torch.stack([art.records[e].gate_up.expand() for e in range(cfg.num_experts)]))
            dense.down_proj.copy_(torch.stack([art.records[e].down.expand() for e in range(cfg.num_experts)]))
        hidden = torch.randn(6, cfg.hidden_size) * 0.5
        scores = torch.rand(6, cfg.num_experts)
        weights, index = torch.topk(torch.softmax(scores, -1), cfg.num_experts_per_tok)
        events = []
        lazy.observer = lambda e, t, o, w: events.append((e, t, o, w))
        with torch.no_grad():
            ours = lazy(hidden, index, weights)
            reference = dense(hidden, index, weights)
        self.assertTrue(torch.equal(ours, reference))  # fp32: exact_linear == F.linear, same loop, same accumulation order
        hit = events[-1][1]
        self.assertEqual(hit, sorted(set(index.flatten().tolist())))
        per_expert = {e: (t, o, w) for e, t, o, w in events[:-1]}
        self.assertEqual(sorted(per_expert), hit)
        for e, (tokens, out, weighted) in per_expert.items():
            positions, rows = torch.where(index.T == e)  # upstream's order: top-k position first, then token
            self.assertEqual(tokens.tolist(), rows.tolist())
            self.assertTrue(torch.equal(weighted, out * weights[rows, positions, None]))

    def test_bf16_accumulation_rounds_each_term_in_ascending_expert_order(self):
        torch.manual_seed(1)
        cfg = tiny_config()
        art = self.FakeArtifact(cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts)
        lazy = LazyExperts(cfg, art, 0, "cpu")
        hidden = (torch.randn(4, cfg.hidden_size) * 0.5).to(torch.bfloat16)
        weights, index = torch.topk(torch.softmax(torch.rand(4, cfg.num_experts), -1), 3)
        weights = weights.to(torch.bfloat16)
        events = []
        lazy.observer = lambda e, t, o, w: events.append((e, t, o, w))
        with torch.no_grad():
            out = lazy(hidden, index, weights)
        manual = torch.zeros_like(hidden)
        for e, tokens, _, weighted in events[:-1]:
            manual.index_add_(0, tokens, weighted)
        self.assertTrue(torch.equal(out, manual))


class TinyModelTest(unittest.TestCase):
    """The recording machinery on a tiny upstream model: recorders replay bit-identically; chunking is invariant."""

    @classmethod
    def setUpClass(cls):
        configure_exact_math()

    def run_model(self, flash, ids, chunks, capture_layers=(0, 1, 2, 3), tmp=None, kv_format="bf16"):
        rec = Recorder(Path(tmp) / "case")
        cap = Capture(flash, rec, capture_layers, kv_format)
        cache = DynamicCache(config=flash.cfg)
        cap.attach_cache(cache)
        outputs, start = [], 0
        try:
            with torch.no_grad():
                for k, rows in enumerate(chunks):
                    cap.begin_chunk(f"c{k}", rows)
                    chunk = ids[:, start:start + rows]
                    rec.add("tokens", chunk[0])
                    outputs.append(flash.model(input_ids=chunk, past_key_values=cache, use_cache=True).last_hidden_state)
                    start += rows
        finally:
            stats = cap.stats_metadata()
            cap.close()
        rec.finish()
        return torch.cat(outputs, 1), rec, stats, cache

    def test_recorders_replay_upstream_bit_identically_and_chunking_is_invariant(self):
        flash = tiny_flash()
        eos = flash.cfg.eos_token_id
        ids = torch.randint(0, 300, (1, 23))
        ids[0, 4] = eos
        ids[0, 12] = eos
        with torch.no_grad():
            reference = flash.model(input_ids=ids, use_cache=False).last_hidden_state
        with tempfile.TemporaryDirectory() as tmp:
            whole, _, stats_whole, _ = self.run_model(flash, ids, [23], tmp=tmp + "/a")
            chunked, rec, stats, cache = self.run_model(flash, ids, [10, 1, 1, 2, 9], tmp=tmp + "/b")
            self.assertGreater(stats["replays_checked"]["ple"], 0)
            self.assertGreater(stats["replays_checked"]["indexer"], 0)
            self.assertLess((whole - reference).abs().max().item(), 1e-6)  # recording changes nothing
            self.assertLess((chunked - whole).abs().max().item(), 5e-5)  # fp32 chunk invariance
            manifest = read_manifest(Path(tmp) / "b" / "case")
            names = set(manifest["tensors"])
            for required in ("c0/embedding", "c0/streams0", "c0/L0/in", "c4/L3/out", "c4/final_mix",
                             "c0/L3/attn/block_ids", "c4/L3/attn/token_ids", "c4/L3/attn/index_scores",
                             "c0/L1/ple/gate_raw", "c0/L1/ple/ngram_context", "c0/L0/gdn/core_out",
                             "c0/L0/gdn/recurrent_state", "c0/L3/moe/topk_weights", "c0/L2/mlp_hc/injection",
                             "c3/L3/attn/index_block_keys", "c4/L1/ple/conv_state"):
                self.assertIn(required, names)
            d = Path(tmp) / "b" / "case"
            # the recorded cache state equals the live cache
            self.assertTrue(torch.equal(load_tensor(d, manifest, "c4/L3/attn/kv_k"), cache.layers[3].keys))
            self.assertTrue(torch.equal(load_tensor(d, manifest, "c4/L1/ple/ngram_context"),
                                        cache.layers[1].conv_states[2]))
            # budget semantics: 23 tokens -> 5 complete blocks > 4 selected
            block_ids = load_tensor(d, manifest, "c4/L3/attn/block_ids")
            last = block_ids[-1]
            self.assertEqual(int((last >= 0).sum()), flash.cfg.indexer_budget // flash.cfg.indexer_compress_ratio)
            token_ids = load_tensor(d, manifest, "c4/L3/attn/token_ids")
            self.assertEqual(token_ids.shape[-1], flash.cfg.indexer_budget + flash.cfg.indexer_compress_ratio - 1)
            scores = load_tensor(d, manifest, "c4/L3/attn/index_scores")
            self.assertTrue(torch.isinf(scores).any())  # -inf padding beyond a row's own complete blocks
            # topk recorded in torch.topk order: the selected scores are the largest ones
            row = scores[-1][torch.isfinite(scores[-1])]
            self.assertEqual(sorted(row.topk(4).values.tolist()), sorted(row[last[last >= 0].long()].tolist()))
            sorted_ids = load_tensor(d, manifest, "c4/L3/attn/block_ids_sorted")
            self.assertEqual(sorted_ids[-1][:4].tolist(), sorted(last[:4].tolist()))

    def test_ple_recording_forward_assertion_fires_on_a_difference(self):
        flash = tiny_flash(1)
        ids = torch.randint(0, 300, (1, 6))

        class Broken(RecordingPLELayer):
            def _recording_forward(self, *args, **kwargs):
                return super()._recording_forward(*args, **kwargs) * 1.001

        flash.by_index[1].ple.__class__ = Broken
        with tempfile.TemporaryDirectory() as tmp:
            rec = Recorder(Path(tmp) / "case")
            cap = Capture(flash, rec, (1,))
            flash.by_index[1].ple.__class__ = Broken  # `Capture` re-assigned the class; break it again
            cap.begin_chunk("c0", 6)
            with self.assertRaises(ReplayMismatch):
                with torch.no_grad():
                    flash.model(input_ids=ids, past_key_values=DynamicCache(config=flash.cfg), use_cache=True)
            cap.close()

    def test_indexer_recording_forward_assertion_fires_on_a_difference(self):
        flash = tiny_flash(2)
        ids = torch.randint(0, 300, (1, 9))

        class Broken(RecordingQSAIndexer):
            def _recording_forward(self, *args, **kwargs):
                mask = super()._recording_forward(*args, **kwargs)
                return mask.clone().fill_(True) if mask.dtype == torch.bool else mask * 0

        with tempfile.TemporaryDirectory() as tmp:
            rec = Recorder(Path(tmp) / "case")
            cap = Capture(flash, rec, (3,))
            flash.by_index[3].self_attn.indexer.__class__ = Broken
            cap.begin_chunk("c0", 9)
            with self.assertRaises(ReplayMismatch):
                with torch.no_grad():
                    flash.model(input_ids=ids, past_key_values=DynamicCache(config=flash.cfg), use_cache=True)
            cap.close()

    def test_nvfp4_kv_format_applies_the_codec_to_cached_keys_and_values(self):
        torch.manual_seed(5)
        cfg = tiny_config(head_dim=256, num_attention_heads=2, num_key_value_heads=2, indexer_head_dim=32,
                          rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.125,
                                           "mrope_section": [6, 5, 5], "mrope_interleaved": True})
        model = Qwen4ExpTextModel(cfg).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.dim() == 1:
                    parameter.add_(0.1 * torch.randn_like(parameter))
        layers = list(range(cfg.num_hidden_layers))
        flash = SimpleNamespace(model=model, cfg=cfg, by_index={n: model.layers[n] for n in layers}, full=True,
                                layers=layers, check_rows=[])
        ids = torch.randint(0, 300, (1, 14))
        with tempfile.TemporaryDirectory() as tmp:
            plain, rec_plain, _, cache_plain = self.run_model(flash, ids, [14], (3,), tmp + "/a", "bf16")
            coded, rec, _, cache = self.run_model(flash, ids, [9, 5], (3,), tmp + "/b", "nvfp4")
            d = Path(tmp) / "b" / "case"
            manifest = read_manifest(d)
            k_first = load_tensor(d, manifest, "c0/L3/attn/k")  # [rows, heads, 256], before the codec
            kv_first = load_tensor(d, manifest, "c0/L3/attn/kv_k")  # [1, heads, rows, 256], what the cache holds
            self.assertTrue(torch.equal(kv_first[0].transpose(0, 1), kvcodec.roundtrip_cache(k_first[None].transpose(1, 2))[0].transpose(0, 1)))
            self.assertGreater((coded - plain).abs().max().item(), 1e-4)  # the codec is visible downstream
            self.assertTrue(torch.equal(cache.layers[3].keys[:, :, :9], kv_first))
            # the cache holds H^T dequant(quant(H k)): within the 4-bit error of the unquantized keys
            reference_keys = cache_plain.layers[3].keys
            relative = (cache.layers[3].keys - reference_keys).norm() / reference_keys.norm()
            self.assertLess(relative.item(), 0.25)

    def test_gdn_and_router_statistics_are_collected_for_every_layer(self):
        flash = tiny_flash(4)
        ids = torch.randint(0, 300, (1, 12))
        with tempfile.TemporaryDirectory() as tmp:
            _, _, stats, _ = self.run_model(flash, ids, [12], capture_layers=(), tmp=tmp)
        self.assertEqual(stats["router_tie_stats"]["router_calls"], 4)
        self.assertEqual(stats["router_tie_stats"]["rows"], 48)
        self.assertEqual(stats["router_tie_stats"]["topk_nondeterministic_rows"], 0)


# --- the real artifact ---------------------------------------------------------------------------------------------------


@unittest.skipUnless(DEFAULT_ARTIFACT.exists(), "the Flash-Next artifact is not present")
class RealArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.art = Artifact(DEFAULT_ARTIFACT)

    def test_config_matches_hf_config_and_ngram_tables(self):
        from flash_next_reference.model import assert_config_matches, load_hf_text_config, make_config
        if not Path("/mnt/shared/qwen38-flash-next/nvfp4/config.json").exists():
            self.skipTest("HF config missing")
        cfg = make_config(load_hf_text_config())
        assert_config_matches(cfg, self.art.metadata)
        ngram_tables_from_metadata(self.art.metadata, cfg)

    def test_crc_of_a_tensor_an_expert_and_a_shard(self):
        self.art.verify_tensor("text/layers/0/gdn/in_proj_qkv")
        self.art.verify_tensor("text/layers/1/ple/ngram/shard_127")
        self.art.verify_expert(0, 0)
        self.art.verify_expert(47, 511)

    def test_expert_record_expansion_equals_the_repos_dequantizer(self):
        from euhedral_artifacts.nvfp4 import nvfp4_offsets
        record = self.art.expert(17, 200, "cpu")
        raw = self.art.expert_bytes(17, 200)
        for projection, packed in zip(self.art.bank(17)["projections"], (record.gate_up, record.down)):
            rows, k = projection["shape"]
            scale_offset, global_offset, _ = nvfp4_offsets((rows, k))
            base = projection["offset"]
            codes = np.array(raw[base: base + rows * k // 2]).reshape(rows, k // 2)
            scales = np.array(raw[base + scale_offset: base + scale_offset + rows * k // 16]).reshape(rows, k // 16)
            g = np.frombuffer(raw[base + global_offset: base + global_offset + 4].tobytes(), "<f4")[0]
            self.assertTrue(np.array_equal(dequantize_nvfp4_rows(codes, scales, np.float32(g)), packed.expand().numpy()))

    def test_ngram_row_gather_equals_the_dequantizer_across_a_shard_boundary(self):
        shard_rows = self.art.metadata["ngram.shard_rows"]
        rows = np.array([0, 1, shard_rows - 1, shard_rows, shard_rows + 7, 127 * shard_rows + 5, 320001445], dtype=np.int64)
        got = self.art.ngram_rows(1, rows, "cpu")
        shards = self.art.ngram_shards(1)
        for out, row in zip(got, rows):
            entry = shards[int(row // shard_rows)]
            local = int(row % shard_rows)
            body = self.art.view(entry["offset"], entry["shape"][0] * 90)
            line = np.array(body[local * 90:(local + 1) * 90])
            trailer = (entry["shape"][0] * 90 + 255) // 256 * 256
            g = np.frombuffer(self.art.view(entry["offset"] + trailer, 4).tobytes(), "<f4")[0]
            expected = dequantize_nvfp4_rows(line[None, :80], line[None, 80:], np.float32(g))[0]
            self.assertTrue(torch.equal(torch.from_numpy(expected).to(torch.bfloat16), out), int(row))

    @unittest.skipUnless(CUDA, "needs a GPU")
    def test_isolated_layer_case_is_deterministic(self):
        from flash_next_reference.cases import run_case
        with tempfile.TemporaryDirectory() as tmp:
            run_case("layer_moe", DEFAULT_ARTIFACT, Path(tmp) / "a")
            run_case("layer_moe", DEFAULT_ARTIFACT, Path(tmp) / "b")
            ma, mb = read_manifest(Path(tmp) / "a/layer_moe"), read_manifest(Path(tmp) / "b/layer_moe")
            self.assertEqual(ma["tensors"].keys(), mb["tensors"].keys())
            for name, entry in ma["tensors"].items():
                self.assertEqual((Path(tmp) / "a/layer_moe" / entry["file"]).read_bytes(),
                                 (Path(tmp) / "b/layer_moe" / mb["tensors"][name]["file"]).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
