"""Command line of tools/flash_next_reference.py."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import time

import torch

from . import UPSTREAM_REVISION
from .artifact import DEFAULT_ARTIFACT, DEFAULT_TOKENIZER, Artifact
from .cases import CASES, build, fp32_chunk_check, run_case, text_tokens
from .model import assert_config_matches, load_hf_text_config, make_config, ngram_tables_from_metadata


def cmd_list(_args) -> int:
    width = max(len(n) for n in CASES)
    for case in CASES.values():
        print(f"{case.name:<{width}}  [{case.kind}]  {case.description}")
    return 0


def cmd_check(args) -> int:
    import numpy as np
    from euhedral_artifacts.nvfp4 import dequantize_nvfp4_rows
    from .nvfp4 import PackedNvfp4

    art = Artifact(args.artifact)
    cfg = make_config(load_hf_text_config(args.hf_config))
    for name, hf, meta in assert_config_matches(cfg, art.metadata):
        shown = f"{len(hf)} entries ({sum(1 for t in hf if t == 'full_attention')} full_attention)" if isinstance(hf, list) and len(hf) > 8 else hf
        print(f"  ok  {name}: {shown}")
    ngram_tables_from_metadata(art.metadata, cfg)
    print("  ok  n-gram multipliers, head offsets and vocabulary sizes equal upstream's constructor helpers")
    names = ["text/layers/0/gdn/in_proj_qkv", "text/layers/3/attention/q_proj", "text/layers/1/ple/key_proj",
             "text/layers/47/moe/shared_expert/down_proj", "text/layers/10/moe/router", "text/layers/1/ple/ngram/shard_127"]
    if args.full:
        names += ["text/token_embedding", "text/output_head"]
    for name in names:
        started = time.time()
        art.verify_tensor(name)
        print(f"  crc32 ok  {name}  {art.entry(name)['bytes']:,} bytes  {time.time() - started:.1f}s")
    for layer, expert in ((0, 0), (0, 511), (17, 200), (47, 511)):
        art.verify_expert(layer, expert)
        print(f"  crc32 ok  expert record layer {layer} #{expert}")
    # expansion against the repo's own dequantizer
    entry = art.entry("text/layers/0/gdn/in_proj_z")
    packed = art.nvfp4("text/layers/0/gdn/in_proj_z", "cpu")
    reference = dequantize_nvfp4_rows(packed.codes.numpy(), packed.scales.numpy(), np.float32(packed.global_scale))
    if not np.array_equal(reference, packed.expand().numpy()):
        raise SystemExit("torch expansion differs from dequantize_nvfp4_rows")
    print("  ok  torch NVFP4 expansion equals euhedral_artifacts.nvfp4.dequantize_nvfp4_rows (in_proj_z)")
    print(f"artifact {art.path}  {art.header['file_size']:,} bytes  {len(art.tensors)} objects, "
          f"{len(art.banks)} text expert banks of {art.banks[0]['expert_count']} records, "
          f"{len(art.ngram_shards(1))} n-gram shards of {art.metadata['ngram.shard_rows']:,} rows")
    return 0


def cmd_fixtures(args) -> int:
    names = list(CASES) if args.case == "all" else [args.case]
    for name in names:
        if name not in CASES:
            raise SystemExit(f"unknown case {name}; `list-cases` lists them")
        started = time.time()
        capture = None if args.capture_layers is None else [int(x) for x in args.capture_layers.split(",") if x]
        result = run_case(name, args.artifact, Path(args.out), args.device, capture, args.logit_rows, args.kv_format)
        print(f"{name}: {result['bytes'] / 1e6:.1f} MB, {time.time() - started:.1f} s, "
              f"GPU peak {result['gpu_peak_gib']:.2f} GiB", flush=True)
    return 0


def cmd_fp32(args) -> int:
    fp32_chunk_check(args.artifact, [int(x) for x in args.layers.split(",")], args.device)
    return 0


def cmd_generate(args) -> int:
    art, fn = build(args.artifact, args.device)
    from transformers.cache_utils import DynamicCache
    tokenizer = None
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    except Exception:  # noqa: BLE001
        pass
    if args.prompt_ids:
        ids = [int(x) for x in args.prompt_ids]
    elif args.prompt is not None:
        if tokenizer is None:
            raise SystemExit("--prompt needs the `tokenizers` package")
        ids = tokenizer.encode(args.prompt).ids
    else:
        raise SystemExit("give --prompt-ids or --prompt")
    cache = DynamicCache(config=fn.cfg)
    print("prompt ids:", ids)
    produced = []
    with torch.no_grad():
        feed = torch.tensor(ids, dtype=torch.long, device=args.device)
        for step in range(args.steps):
            hidden = fn.model(input_ids=feed[None], past_key_values=cache, use_cache=True).last_hidden_state
            logits = fn.logits(hidden[:, -1])[0]
            top = torch.topk(logits.float(), 5)
            token = int(top.indices[0])
            produced.append(token)
            text = lambda t: repr(tokenizer.decode([t])) if tokenizer else ""
            print(f"step {step}: token {token} {text(token)}  top5 " + ", ".join(
                f"{int(i)}:{float(v):.3f}{text(int(i))}" for v, i in zip(top.values, top.indices)), flush=True)
            feed = torch.tensor([token], dtype=torch.long, device=args.device)
    print("generated ids:", produced)
    if tokenizer:
        print("text:", repr(tokenizer.decode(produced)))
    return 0


def cmd_greedy(args) -> int:
    """Greedy continuations of several prompts, with every step's top-5 logits, as JSON (the engine's end-to-end gate)."""
    import json
    from tokenizers import Tokenizer
    from transformers.cache_utils import DynamicCache
    from .instrument import Capture
    from .record import Recorder
    art, fn = build(args.artifact, args.device)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    results = []
    for text in prompts:
        ids = tokenizer.encode(text).ids
        cache = DynamicCache(config=fn.cfg)
        Capture(fn, Recorder(Path(tempfile.mkdtemp())), (), args.kv_format, model_level=True).attach_cache(cache)
        steps = []
        with torch.no_grad():
            feed = torch.tensor(ids, dtype=torch.long, device=args.device)
            for _ in range(args.steps):
                hidden = fn.model(input_ids=feed[None], past_key_values=cache, use_cache=True).last_hidden_state
                logits = fn.logits(hidden[:, -1])[0]
                top = torch.topk(logits.float(), 5)
                token = int(top.indices[0])
                steps.append({"token": token, "top5": [int(i) for i in top.indices],
                              "top5_logits": [float(v) for v in top.values]})
                feed = torch.tensor([token], dtype=torch.long, device=args.device)
        results.append({"prompt": text, "prompt_ids": ids, "steps": steps,
                        "text": tokenizer.decode([s["token"] for s in steps])})
        print(f"{text[:40]!r}: {results[-1]['text'][:60]!r}", flush=True)
    args.out.write_text(json.dumps({"upstream_revision": UPSTREAM_REVISION, "kv_format": args.kv_format,
                                    "results": results}, indent=1) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Python reference harness for Qwen3.8-Flash-Next")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
        p.add_argument("--device", default="cuda")

    p = sub.add_parser("fixtures", help="write one case's (or `all`) fixtures to OUT/<case>/")
    common(p)
    p.add_argument("--out", type=Path, required=True, help="fixture root; the case goes in OUT/<case>")
    p.add_argument("--case", required=True)
    p.add_argument("--kv-format", choices=("bf16", "nvfp4"), default="bf16")
    p.add_argument("--capture-layers", help="comma-separated layers (model cases), overrides the case default")
    p.add_argument("--logit-rows", type=int, default=1)
    p.set_defaults(func=cmd_fixtures)
    p = sub.add_parser("list-cases")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("check-artifact", help="config/metadata cross-check and CRC-32 of a sample of objects")
    common(p)
    p.add_argument("--hf-config", type=Path, default=Path("/mnt/shared/qwen38-flash-next/nvfp4/config.json"))
    p.add_argument("--full", action="store_true", help="also CRC the 1.2 GB embedding and head")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("fp32-chunk-check", help="chunk invariance of real layers in fp32 (implementation error only)")
    common(p)
    p.add_argument("--layers", default="0,1,3")
    p.set_defaults(func=cmd_fp32)
    p = sub.add_parser("generate", help="greedy reference generation (model-level only)")
    common(p)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--prompt-ids", nargs="+")
    p.add_argument("--prompt")
    p.set_defaults(func=cmd_generate)
    p = sub.add_parser("greedy", help="greedy continuations of several prompts with every step's top-5 (JSON)")
    common(p)
    p.add_argument("--prompts", type=Path, required=True, help="JSON list of prompt strings")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--kv-format", choices=("bf16", "nvfp4"), default="nvfp4")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_greedy)
    args = parser.parse_args(argv)
    return args.func(args)
