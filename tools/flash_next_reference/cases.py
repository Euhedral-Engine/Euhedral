"""Fixture cases: model-level (all 48 layers, real weights), layer-isolated, and pure-function ones."""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.qwen4_exp import modeling_qwen4_exp as upstream

from . import UPSTREAM_REVISION
from .artifact import DEFAULT_ARTIFACT, DEFAULT_HF_CONFIG, DEFAULT_TOKENIZER, Artifact
from .instrument import Capture
from .kvcodec import HEAD_DIM
from .model import (FlashNext, IdsOnlyEmbedding, assert_config_matches, load_hf_text_config, make_config,
                    make_ngram_module, ngram_tables_from_metadata)
from .nvfp4 import configure_exact_math
from .record import Recorder

EOS = 248044
TEXT = ("The quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant mountains, casting long "
        "golden shadows across the quiet valley where a small river winds its way toward the sea. In the village below, "
        "lanterns were being lit one by one as families gathered for the evening meal and the children listened to old "
        "stories about travelers who had crossed the great desert, carrying water, salt and letters for people they "
        "would never meet again. By morning the road was empty once more and only the wind remained to read the "
        "footprints left in the dust.")

LIGHT_QSA = {"tokens", "L{n}/in", "L{n}/out", "L{n}/attn/index_raw_k", "L{n}/attn/block_ids", "L{n}/attn/token_ids",
             "L{n}/attn/core", "L{n}/attn/out"}


def text_tokens(count: int, path: Path = DEFAULT_TOKENIZER) -> list[int]:
    from tokenizers import Tokenizer

    ids = Tokenizer.from_file(str(path)).encode(TEXT).ids
    if len(ids) < count:
        raise RuntimeError(f"the fixed text has {len(ids)} tokens, {count} needed")
    return ids[:count]


def random_tokens(count: int, seed: int, eos_at=()) -> list[int]:
    rng = random.Random(seed)
    ids = [rng.randrange(0, EOS) for _ in range(count)]
    for position in eos_at:
        ids[position] = EOS
    return ids


@dataclass
class Case:
    name: str
    kind: str  # model | layer | pure
    description: str
    chunks: list[int] = field(default_factory=list)
    decode: int = 0
    capture_layers: tuple[int, ...] = ()
    layer: int | None = None


CASES: dict[str, Case] = {c.name: c for c in [
    Case("short", "model", "20-token prompt in one chunk, then 3 greedy decode steps", [20], 3, (0, 1, 2, 3, 47)),
    Case("eos", "model", "40 tokens with EOS inside (positions 5, 13, 20, 21, 39), chunks [13, 1, 26]", [13, 1, 26], 0, (0, 1, 3)),
    Case("chunks_a", "model", "the same 70-token text prefilled as [70] (model-level outputs only)", [70]),
    Case("chunks_b", "model", "the same 70-token text prefilled as [32, 32, 6]", [32, 32, 6]),
    Case("chunks_c", "model", "the same 70-token text prefilled as [1] * 8 + [62]", [1] * 8 + [62]),
    Case("decode", "model", "40-token prefill, then 12 greedy decode steps, logits every step", [40], 12),
    Case("layer_gdn", "layer", "layer 0 (GDN + MoE) on synthetic streams, chunks [2,1,3,1,5,16,1,1]",
         [2, 1, 3, 1, 5, 16, 1, 1], 0, (0,), 0),
    Case("layer_ple", "layer", "layer 1 (PLE + GDN + MoE), real token ids with EOS, chunks [9,1,1,5,16]",
         [9, 1, 1, 5, 16], 0, (1,), 1),
    Case("layer_qsa_short", "layer", "layer 3 (QSA + MoE), chunks [1,2,1,1,3,4] then 6 decode rows",
         [1, 2, 1, 1, 3, 4, 1, 1, 1, 1, 1, 1], 0, (3,), 3),
    Case("layer_qsa_long", "layer", "layer 3, 2700 tokens in chunks of 512 (budget exceeded), then 8 decode rows",
         [512] * 5 + [140] + [1] * 8, 0, (3,), 3),
    Case("layer_moe", "layer", "layer 0 MoE only: 1, 8 and 64 input rows incl. rows that route alike",
         [1, 8, 64], 0, (0,), 0),
    Case("ngram_ids", "pure", "n-gram id construction: one shot vs cache protocol, multiplier tables"),
]}


# --- shared pieces ---------------------------------------------------------------------------------------------------


def run_metadata(case: Case, extra: dict) -> dict:
    meta = {
        "case": case.name,
        "description": case.description,
        "upstream_revision": UPSTREAM_REVISION,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "attn_implementation": "eager",
        "experts": "upstream loop (LazyExperts), ExactNvfp4 products",
        "tf32": False,
    }
    meta.update(extra)
    return meta


def embedding_sigma(art: Artifact, cfg) -> float:
    """Std of the real token embedding over a deterministic sample of 2048 rows."""
    rows = np.arange(0, cfg.vocab_size, cfg.vocab_size // 2048)[:2048]
    sample = art.bf16_rows("text/token_embedding", rows).float()
    return float(sample.std())


def synthetic_streams(rows: int, seed: int, sigma: float, device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, 10240, generator=generator) * sigma).to(torch.bfloat16).to(device)


def build(art_path, device, layers=None, expert_cache_bytes=768 << 20) -> tuple[Artifact, FlashNext]:
    configure_exact_math()
    art = Artifact(art_path, expert_cache_bytes=expert_cache_bytes)
    cfg = make_config(load_hf_text_config())
    return art, FlashNext(art, cfg, device, layers=layers)


def _gpu_peak() -> float:
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


# --- model cases -----------------------------------------------------------------------------------------------------


def model_inputs(case: Case) -> list[int]:
    total = sum(case.chunks)
    ids = text_tokens(total)
    if case.name == "eos":
        for position in (5, 13, 20, 21, 39):
            ids[position] = EOS
    return ids


def run_model_case(case: Case, art_path, out_root: Path, device="cuda", capture_layers=None, logit_rows=1,
                   kv_format="bf16", log=print) -> dict:
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    art, fn = build(art_path, device)
    loaded = time.time()
    layers = case.capture_layers if capture_layers is None else tuple(capture_layers)
    tokens = model_inputs(case)
    rec = Recorder(out_root / case.name)
    cap = Capture(fn, rec, layers, kv_format)
    cache = DynamicCache(config=fn.cfg)
    cap.attach_cache(cache)
    pieces, position, step = [], 0, 0
    produced: list[int] = []
    with torch.no_grad():
        def forward(ids: torch.Tensor):
            nonlocal step
            rows = ids.numel()
            cap.begin_chunk(f"c{step}", rows)
            rec.add("tokens", ids)
            hidden = fn.model(input_ids=ids[None], past_key_values=cache, use_cache=True).last_hidden_state
            count = min(rows, max(1, logit_rows))
            logits = fn.logits(hidden[:, -count:])
            rec.add("logits", logits[0])
            rec.add("logits_rows", torch.arange(rows - count, rows, dtype=torch.int32))
            pieces.append({"step": step, "rows": rows, "seconds": round(time.time() - tick, 3)})
            step += 1
            return logits[0, -1]

        for rows in case.chunks:
            tick = time.time()
            ids = torch.tensor(tokens[position:position + rows], dtype=torch.long, device=device)
            last = forward(ids)
            position += rows
        for _ in range(case.decode):
            tick = time.time()
            token = int(last.argmax())
            produced.append(token)
            last = forward(torch.tensor([token], dtype=torch.long, device=device))
    cap_meta = cap.stats_metadata()
    cap.close()
    total = rec.finish(run_metadata(case, {
        "chunks": case.chunks, "decode_steps": case.decode, "prompt_tokens": tokens, "decoded_tokens": produced,
        "capture_layers": list(layers), "kv_format": kv_format, "logit_rows": logit_rows, "chunk_timing": pieces,
        "load_seconds": round(loaded - started, 1), "total_seconds": round(time.time() - started, 1),
        "gpu_peak_gib": round(_gpu_peak(), 2), "expert_reads": art.expert_reads, "expert_cache_hits": art.expert_hits,
        **cap_meta, "config_checks": len(fn.check_rows)}))
    return {"bytes": total, "seconds": time.time() - started, "gpu_peak_gib": _gpu_peak(), **cap_meta}


# --- layer-isolated cases --------------------------------------------------------------------------------------------


def run_layer_case(case: Case, art_path, out_root: Path, device="cuda", kv_format="bf16", log=print) -> dict:
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    n = case.layer
    art, fn = build(art_path, device, layers=[n])
    sigma = embedding_sigma(art, fn.cfg)
    rec = Recorder(out_root / case.name)
    cap = Capture(fn, rec, (n,), kv_format, model_level=False)
    layer = fn.layer(n)
    total_tokens = sum(case.chunks)
    if case.name == "layer_ple":
        ids_all = text_tokens(total_tokens)
        for position in (3, 9, 10, 20, 31):
            ids_all[position] = EOS
    else:
        ids_all = random_tokens(total_tokens, seed=11)
    timing = []
    holder: dict = {}
    if case.name == "layer_moe":
        run_moe_case(case, fn, cap, rec, sigma, device)
    else:
        layer.register_forward_pre_hook(lambda m, a, kw: ((holder["x"],) + tuple(a[1:]), kw), with_kwargs=True, prepend=True)
        cache = DynamicCache(config=fn.cfg)
        cap.attach_cache(cache)
        long_case = case.name == "layer_qsa_long"
        if long_case:
            rec.light = {name.format(n=n) for name in LIGHT_QSA}
        prefill_chunks = 6 if long_case else len(case.chunks)
        position = 0
        with torch.no_grad():
            for k, rows in enumerate(case.chunks):
                tick = time.time()
                if long_case:
                    detail = k in (4, 5, 6, len(case.chunks) - 1)
                else:
                    detail = True
                cap.begin_chunk(f"c{k}", rows, detail)
                ids = torch.tensor(ids_all[position:position + rows], dtype=torch.long, device=device)
                rec.add("tokens", ids)
                holder["x"] = synthetic_streams(rows, 1000 + k, sigma, device)[None]
                fn.model(inputs_embeds=torch.zeros(1, rows, fn.cfg.hidden_size, dtype=torch.bfloat16, device=device),
                         ple_input_ids=ids[None], past_key_values=cache, use_cache=True)
                position += rows
                timing.append({"step": k, "rows": rows, "seconds": round(time.time() - tick, 3)})
    cap_meta = cap.stats_metadata()
    cap.close()
    total = rec.finish(run_metadata(case, {
        "chunks": case.chunks, "layer": n, "synthetic_sigma": sigma, "stream_seed_base": 1000, "tokens": ids_all,
        "kv_format": kv_format, "chunk_timing": timing, "total_seconds": round(time.time() - started, 1),
        "gpu_peak_gib": round(_gpu_peak(), 2), **cap_meta}))
    return {"bytes": total, "seconds": time.time() - started, "gpu_peak_gib": _gpu_peak(), **cap_meta}


def run_moe_case(case: Case, fn: FlashNext, cap: Capture, rec: Recorder, sigma: float, device) -> None:
    n = case.layer
    layer = fn.layer(n)
    with torch.no_grad():
        for k, rows in enumerate(case.chunks):
            cap.begin_chunk(f"c{k}", rows)
            unique = {1: 1, 8: 5, 64: 40}[rows]
            streams = synthetic_streams(unique, 2000 + k, sigma, device)
            with rec.mute():
                mixed = layer.mlp_hyper_connection(streams[None])[0][0]
            x = torch.empty(rows, mixed.shape[-1], dtype=mixed.dtype, device=device)
            x[:unique] = mixed
            if rows == 8:  # rows 5, 6 repeat row 0 exactly, row 7 repeats row 2 scaled
                x[5], x[6], x[7] = mixed[0], mixed[0], (mixed[2].float() * 0.97).to(mixed.dtype)
            if rows == 64:  # 40..47 exact copies of 0..7, 48..55 copies of 8..15, 56..63 scaled copies of 0..7
                x[40:48], x[48:56] = mixed[0:8], mixed[8:16]
                x[56:64] = (mixed[0:8].float() * 0.97).to(mixed.dtype)
            layer.mlp(x[None])


# --- pure n-gram id fixtures ---------------------------------------------------------------------------------------------


def run_ngram_ids(case: Case, art_path, out_root: Path, device="cpu", log=print) -> dict:
    started = time.time()
    art = Artifact(art_path)
    cfg = make_config(load_hf_text_config())
    assert_config_matches(cfg, art.metadata)
    rec = Recorder(out_root / case.name)
    module = make_ngram_module(cfg, art.metadata, IdsOnlyEmbedding(), layer_idx=1, ple_layer_index=0, device=device)
    multipliers, offsets, sizes = ngram_tables_from_metadata(art.metadata, cfg)
    rec.begin("")
    rec.add("multipliers", multipliers)
    rec.add("head_offsets", offsets)
    rec.add("head_vocab_sizes", sizes)
    samples = [0, 1, 2, 3, 1234, 248320, (1 << 63) - 1, (1 << 64) - 1]
    signed = lambda v: v - (1 << 64) if v >= (1 << 63) else v
    rec.add("splitmix_in", torch.tensor([signed(v) for v in samples], dtype=torch.int64))
    rec.add("splitmix_out", torch.tensor([signed(upstream._splitmix64(v)) for v in samples], dtype=torch.int64))
    streams = [
        (random_tokens(40, 21, eos_at=(5, 6, 20, 39)), [1, 1, 2, 5, 13, 18]),
        (random_tokens(24, 22, eos_at=(0, 1, 2, 12)), [1] * 24),
        (random_tokens(33, 23, eos_at=(7, 8, 9, 32)), [3, 30]),
        (random_tokens(17, 24, eos_at=(16,)), [2, 15]),
        (random_tokens(9, 25), [9]),
    ]
    described = []
    with torch.no_grad():
        for index, (tokens, chunks) in enumerate(streams):
            ids = torch.tensor(tokens, dtype=torch.long, device=device)[None]
            rec.begin(f"s{index}")
            rec.add("tokens", ids[0])
            oneshot = module(ids, None)[0]
            rec.begin(f"s{index}/oneshot")
            rec.add("ids", oneshot)
            cache = DynamicCache(config=cfg)
            position, pieces = 0, []
            for k, rows in enumerate(chunks):
                rec.begin(f"s{index}/c{k}")
                part = module(ids[:, position:position + rows], cache)[0]
                rec.add("ids", part)
                rec.add("ngram_context", cache.layers[1].conv_states[2])
                pieces.append(part)
                position += rows
            if not torch.equal(torch.cat(pieces), oneshot):
                raise AssertionError(f"stream {index}: chunked n-gram ids differ from the one-shot ids")
            described.append({"tokens": len(tokens), "chunks": chunks, "eos_positions": [i for i, t in enumerate(tokens) if t == EOS]})
    total = rec.finish(run_metadata(case, {"streams": described, "total_seconds": round(time.time() - started, 1)}))
    return {"bytes": total, "seconds": time.time() - started, "gpu_peak_gib": 0.0}


def run_case(name: str, art_path=DEFAULT_ARTIFACT, out_root: Path = Path("/home/brandon/fixtures/flash-next"),
             device="cuda", capture_layers=None, logit_rows=1, kv_format="bf16", log=print) -> dict:
    case = CASES[name]
    try:
        if case.kind == "model":
            return run_model_case(case, art_path, Path(out_root), device, capture_layers, logit_rows, kv_format, log)
        if case.kind == "layer":
            return run_layer_case(case, art_path, Path(out_root), device, kv_format, log)
        return run_ngram_ids(case, art_path, Path(out_root), "cpu", log)
    finally:  # modules, hooks and caches form cycles: release the GPU before the next case builds its model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def fp32_chunk_check(art_path=DEFAULT_ARTIFACT, layers=(0, 1, 3), device="cuda", log=print) -> list[dict]:
    """Chunk invariance of real layers in fp32: BF16 tensors upcast, NVFP4 expanded exactly, synthetic 70-row streams
    prefilled as [70], [32, 32, 6] and [1] * 8 + [62]. Differences are implementation error, not BF16 noise."""
    results = []
    for n in layers:
        art, fn = build(art_path, device, layers=[n])
        fn.model.float()
        for module in fn.model.layers[0].modules():
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.dtype == torch.bfloat16:
                module.weight.data = weight.data.float()
        sigma = embedding_sigma(art, fn.cfg)
        streams = synthetic_streams(70, 5, sigma, device).float()[None]
        ids = torch.tensor(text_tokens(70), device=device)[None]
        holder: dict = {}
        fn.layer(n).register_forward_pre_hook(lambda m, a, kw: ((holder["x"],) + tuple(a[1:]), kw), with_kwargs=True,
                                              prepend=True)
        outputs = {}
        for name, chunks in (("a", [70]), ("b", [32, 32, 6]), ("c", [1] * 8 + [62])):
            cache, position, parts = DynamicCache(config=fn.cfg), 0, []
            for rows in chunks:
                holder["x"] = streams[:, position:position + rows]
                with torch.no_grad():
                    parts.append(fn.model(inputs_embeds=torch.zeros(1, rows, fn.cfg.hidden_size, device=device),
                                          ple_input_ids=ids[:, position:position + rows], past_key_values=cache,
                                          use_cache=True).last_hidden_state)
                position += rows
            outputs[name] = torch.cat(parts, 1)
        for name in "bc":
            diff = outputs["a"] - outputs[name]
            row = {"layer": n, "chunks": name, "max_abs": diff.abs().max().item(),
                   "rel_l2": (diff.norm() / outputs["a"].norm()).item()}
            results.append(row)
            log(f"layer {n} fp32, [70] vs {name}: max|d| {row['max_abs']:.2e}, relative L2 {row['rel_l2']:.2e}")
        del fn, art
        torch.cuda.empty_cache()
    return results
