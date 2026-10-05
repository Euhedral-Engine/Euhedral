#!/usr/bin/env python3
"""DFlash2 reference fixtures from the upstream implementation (z-lab/dflash).

Runs the published `DFlash2DraftModel` (upstream `dflash/model.py`, unmodified) over fixed inputs and writes every
intermediate the engine's drafter must reproduce: the feature projection, each draft layer (both dynamic convolutions,
the attention projections, the cached context keys and values after normalization and RoPE), the final hidden rows,
the full-vocabulary logits, the top-16 candidates, the selector's transition scores and the 7-token proposal.

Inputs, one of:
  --inputs DIR      taps.bin (BF16 [rows, 5 * 5120], the target's hidden rows after layers 5, 19, 33, 47, 61) and
                    inputs.json {"start": P, "anchor": token, "context_rows": rows}; written by the engine's
                    DFlash2 fixture test, so the reference conditions on the engine's own target.
  --synthetic SEED  deterministic Gaussian taps (scaled like real hidden rows) for numerics only.

The block is `anchor + 7 mask tokens` at positions P .. P + 7, and the context rows sit at P - rows .. P - 1, which is
how upstream `dflash_generate` calls the draft model after a prefill of P tokens.

`--embedding` replaces the target's BF16 embedding rows with the rows the engine embedded (the engine's dump), and
`--lm-head-artifact` the output head with the artifact's NVFP4 head expanded exactly (FP32 weights, FP32 products, one
BF16 rounding of the logits, as the engine's kernels compute them), so a comparison isolates the drafter's own
arithmetic.

Output: DIR/manifest.json lists every tensor (dtype, shape, file); each tensor is raw little-endian bytes.
Needs PyTorch, transformers 5.15 and safetensors (upstream's pins), plus the upstream source tree (--dflash-src).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import torch

TAP_LAYERS = (5, 19, 33, 47, 61)


def read_safetensor(directory: Path, name: str, rows: list[int] | None = None) -> torch.Tensor:
    """One tensor (or some of its rows) from a safetensors directory, without loading the rest."""
    for path in sorted(directory.glob("*.safetensors")):
        with open(path, "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
            if name not in header:
                continue
            entry = header[name]
            if entry["dtype"] != "BF16":
                raise SystemExit(f"{name}: expected BF16, found {entry['dtype']}")
            shape = entry["shape"]
            begin, _ = entry["data_offsets"]
            row_bytes = 2
            for extent in shape[1:]:
                row_bytes *= extent
            base = 8 + length + begin
            if rows is None:
                handle.seek(base)
                data = handle.read(row_bytes * shape[0])
                return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape)
            out = []
            for row in rows:
                handle.seek(base + row * row_bytes)
                out.append(torch.frombuffer(bytearray(handle.read(row_bytes)), dtype=torch.bfloat16))
            return torch.stack(out).reshape([len(rows), *shape[1:]])
    raise SystemExit(f"{name} not found in {directory}")


def read_raw(path: Path, shape: list[int]) -> torch.Tensor:
    data = bytearray(path.read_bytes())
    tensor = torch.frombuffer(data, dtype=torch.bfloat16)
    expected = 1
    for extent in shape:
        expected *= extent
    if tensor.numel() != expected:
        raise SystemExit(f"{path}: {tensor.numel()} BF16 values, expected {shape}")
    return tensor.reshape(shape)


def artifact_head(artifact: Path, vocabulary: int, hidden: int) -> torch.Tensor:
    """The artifact's NVFP4 `text/output_head` (plain or SD4 scales) expanded to FP32."""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from euhedral_artifacts import edrl, nvfp4

    with artifact.open("rb") as handle:
        _, _, objects = edrl.read_table(handle)
        entry = next(obj for obj in objects if obj["name"] == "text/output_head")
        if entry["format"] != edrl.FORMAT_ORDINAL["NVFP4"] or tuple(entry["shape"]) != (vocabulary, hidden):
            raise SystemExit("the artifact's output head is not an NVFP4 [vocab, hidden] tensor")
        handle.seek(entry["offset"])
        payload = np.frombuffer(handle.read(entry["bytes"]), dtype=np.uint8)
    codes = payload[: vocabulary * hidden // 2].reshape(vocabulary, hidden // 2)
    if entry["layout"] == edrl.LAYOUT_ORDINAL["row-split-k128-sd4-v1"]:
        index_offset, table_offset, _ = nvfp4.nvfp4_sd4_offsets((vocabulary, hidden))
        indices = payload[index_offset: index_offset + vocabulary * hidden // 32].reshape(vocabulary, hidden // 32)
        table = payload[table_offset: table_offset + 16]
        scales = nvfp4.expand_nvfp4_sd4(indices, table)
        global_scale = np.frombuffer(payload[table_offset + 16: table_offset + 20].tobytes(), "<f4")[0]
    else:
        scale_offset, global_offset, _ = nvfp4.nvfp4_offsets((vocabulary, hidden))
        scales = payload[scale_offset: scale_offset + vocabulary * hidden // 16].reshape(vocabulary, hidden // 16)
        global_scale = np.frombuffer(payload[global_offset: global_offset + 4].tobytes(), "<f4")[0]
    rows = []
    for begin in range(0, vocabulary, 16384):
        end = min(vocabulary, begin + 16384)
        rows.append(torch.from_numpy(nvfp4.dequantize_nvfp4_rows(codes[begin:end], scales[begin:end], global_scale)))
    return torch.cat(rows)


class Recorder:
    """Collects named tensors in call order and writes them with a manifest."""

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.tensors:
            raise RuntimeError(f"tensor {name} recorded twice")
        self.tensors[name] = tensor.detach().to("cpu").contiguous()

    def write(self, directory: Path, metadata: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"metadata": metadata, "tensors": {}}
        for name, tensor in self.tensors.items():
            file = name.replace("/", ".") + ".bin"
            dtype = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int64: "i64", torch.int32: "i32"}[tensor.dtype]
            if tensor.dtype == torch.int64:
                tensor = tensor.to(torch.int32)
                dtype = "i32"
            raw = tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor
            (directory / file).write_bytes(raw.numpy().tobytes())
            manifest["tensors"][name] = {"dtype": dtype, "shape": list(tensor.shape), "file": file}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def instrument(model, recorder: Recorder) -> None:
    """Records each draft layer's intermediates through hooks and wrapped convolution methods."""
    def hook(name):
        def record(_module, _inputs, output):
            recorder.add(name, output)
        return record

    model.fc.register_forward_hook(hook("fc"))
    model.hidden_norm.register_forward_hook(hook("context"))
    model.norm.register_forward_hook(hook("final_hidden"))
    for index, layer in enumerate(model.layers):
        prefix = f"layer{index}"
        layer.input_layernorm.register_forward_hook(hook(f"{prefix}/input_norm"))
        layer.post_attention_layernorm.register_forward_hook(hook(f"{prefix}/post_attention_norm"))
        attention = layer.self_attn
        attention.q_proj.register_forward_hook(hook(f"{prefix}/q"))
        calls = {"k": 0, "v": 0}

        def keyed(kind, prefix=prefix, calls=calls):
            def record(_module, _inputs, output):
                part = "context" if calls[kind] == 0 else "block"
                calls[kind] += 1
                recorder.add(f"{prefix}/{kind}_{part}", output)
            return record

        attention.k_proj.register_forward_hook(keyed("k"))
        attention.v_proj.register_forward_hook(keyed("v"))
        attention.o_proj.register_forward_pre_hook(
            lambda _module, inputs, prefix=prefix: recorder.add(f"{prefix}/attention", inputs[0]))
        attention.o_proj.register_forward_hook(hook(f"{prefix}/o"))
        layer.mlp.register_forward_hook(hook(f"{prefix}/mlp"))
        layer.register_forward_hook(hook(f"{prefix}/out"))
        for conv_name in ("attention_conv", "mlp_conv"):
            conv = getattr(layer, conv_name)
            conv.kernel_projection.register_forward_hook(hook(f"{prefix}/{conv_name}/dynamic"))
            prepare, finish = conv.prepare, conv.finish

            def wrapped_prepare(hidden, prepare=prepare, name=f"{prefix}/{conv_name}"):
                convolved, kernel = prepare(hidden)
                recorder.add(f"{name}/prepared", convolved)
                return convolved, kernel

            def wrapped_finish(hidden, dynamic, finish=finish, name=f"{prefix}/{conv_name}"):
                result = finish(hidden, dynamic)
                recorder.add(f"{name}/finished", result)
                return result

            conv.prepare, conv.finish = wrapped_prepare, wrapped_finish


def select_with_scores(selector, hidden, logits, anchor_ids, recorder: Recorder):
    """Upstream `CandidateSelector.select` at temperature 0, recording each step's transition scores."""
    unary, candidates = torch.topk(logits, selector.top_k, dim=-1, sorted=False)
    recorder.add("topk_values", unary)
    recorder.add("topk_indices", candidates)
    projected = selector.hidden_projection(hidden)
    recorder.add("selector_hidden", projected)
    predecessor = anchor_ids
    path, scores_rows = [], []
    for position in range(projected.shape[1]):
        scores = unary[:, position] + torch.einsum(
            "br,bkr->bk",
            selector.predecessor_codebook(predecessor) * projected[:, position],
            selector.successor_codebook(candidates[:, position]),
        )
        scores_rows.append(scores)
        index = torch.argmax(scores, dim=-1)
        predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
        path.append(predecessor)
    recorder.add("selector_scores", torch.stack(scores_rows, dim=1))
    proposal = torch.stack(path, dim=1)
    recorder.add("proposal", proposal)
    return proposal


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", type=Path, default=Path("/mnt/shared/Qwen3.8-27B-DFlash2"))
    parser.add_argument("--target", type=Path, default=Path("/mnt/shared/qwen38-quant/source/qwen"))
    parser.add_argument("--dflash-src", type=Path, default=Path.home() / "src/third_party/dflash")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", type=Path)
    source.add_argument("--synthetic", type=int, metavar="SEED")
    parser.add_argument("--context-rows", type=int, default=24, help="synthetic: context rows")
    parser.add_argument("--start", type=int, default=24, help="synthetic: anchor position")
    parser.add_argument("--anchor", type=int, default=785, help="synthetic: anchor token")
    parser.add_argument("--embedding", type=Path, help="BF16 [8, 5120] noise embedding rows to use instead")
    parser.add_argument("--lm-head-artifact", type=Path, help="use this artifact's NVFP4 output head instead")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.dflash_src))
    from dflash.model import DFlash2DraftModel  # noqa: E402  (upstream, unmodified)

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = DFlash2DraftModel.from_pretrained(str(args.draft), dtype=dtype).to(args.device).eval()
    config = model.config
    hidden = config.hidden_size
    taps = len(model.target_layer_ids)
    if tuple(model.target_layer_ids) != TAP_LAYERS:
        raise SystemExit(f"unexpected target layers {model.target_layer_ids}")

    if args.inputs is not None:
        meta = json.loads((args.inputs / "inputs.json").read_text())
        start, anchor, rows = int(meta["start"]), int(meta["anchor"]), int(meta["context_rows"])
        target_hidden = read_raw(args.inputs / "taps.bin", [rows, taps * hidden])
    else:
        start, anchor, rows = args.start, args.anchor, args.context_rows
        generator = torch.Generator().manual_seed(args.synthetic)
        target_hidden = (torch.randn(rows, taps * hidden, generator=generator) * 0.75).to(torch.bfloat16)
    if rows > start:
        raise SystemExit("context rows cannot precede position 0")

    block_size = model.block_size
    block = [anchor] + [model.mask_token_id] * (block_size - 1)
    if args.embedding is not None:
        noise = read_raw(args.embedding, [block_size, hidden])
    else:
        noise = read_safetensor(args.target, "model.language_model.embed_tokens.weight", block)
    if args.lm_head_artifact is not None:
        weight = artifact_head(args.lm_head_artifact, config.vocab_size, hidden)
        head = torch.nn.Linear(hidden, config.vocab_size, bias=False, dtype=torch.float32, device=args.device)
        with torch.no_grad():
            head.weight.copy_(weight)
        del weight
        exact = head

        def engine_head(rows):
            return exact(rows.float()).to(dtype)

        head = engine_head
    else:
        head_weight = read_safetensor(args.target, "lm_head.weight")
        head = torch.nn.Linear(hidden, config.vocab_size, bias=False, dtype=dtype, device=args.device)
        with torch.no_grad():
            head.weight.copy_(head_weight.to(dtype))
        del head_weight

    recorder = Recorder()
    recorder.add("taps", target_hidden)
    recorder.add("noise_embedding", noise)
    instrument(model, recorder)
    from transformers import DynamicCache  # noqa: E402

    cache = DynamicCache(config=config)
    positions = torch.arange(start - rows, start + block_size, device=args.device)[None]
    with torch.inference_mode():
        draft_hidden = model(
            target_hidden=target_hidden[None].to(args.device, dtype),
            noise_embedding=noise[None].to(args.device, dtype),
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
        )
        for index in range(config.num_hidden_layers):
            layer = cache.layers[index]
            # [rows, KV heads * head dim]: context rows, then the block's.
            recorder.add(f"layer{index}/cache_keys", layer.keys[0].transpose(0, 1).flatten(1))
            recorder.add(f"layer{index}/cache_values", layer.values[0].transpose(0, 1).flatten(1))
        proposal_hidden = draft_hidden[:, 1 - block_size:, :]
        logits = model.compute_logits(proposal_hidden, head)
        recorder.add("logits", logits[0])
        proposal = select_with_scores(
            model.candidate_selector, proposal_hidden, logits, torch.tensor([anchor], device=args.device), recorder)
    # Remove the batch dimension everywhere for the consumers.
    for name, tensor in list(recorder.tensors.items()):
        if tensor.dim() > 1 and tensor.shape[0] == 1:
            recorder.tensors[name] = tensor[0]
    recorder.write(args.out, {
        "start": start,
        "anchor": anchor,
        "context_rows": rows,
        "block": block,
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "draft": str(args.draft),
        "embedding": str(args.embedding) if args.embedding else "target BF16",
        "lm_head": str(args.lm_head_artifact) if args.lm_head_artifact else "target BF16",
    })
    print("proposal", proposal[0].tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
