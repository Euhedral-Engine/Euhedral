"""GPTQ calibration of an artifact recipe: pre-rounded codes for the converter's --codes-from entry.

Usage (in tools/): python -m gptq.calibrate --model CHECKPOINT --quantization nvfp4|q3 [--compressed]
                       --tokens TOKENS.npy --draft-ids-from ARTIFACT --out CODES_DIR [--sequences N] [--batch N]

TOKENS.npy holds calibration sequences (gptq/calibration.py). The hidden states of all sequences stay in pinned host
memory and stream through the GPU a batch at a time; all arithmetic runs on the GPU. Then, one decoder layer at a time (qwen38.Layer, read from the checkpoint): the sequences run through the layer in
batches while the Hessian X^T X of every quantized linear's input accumulates on the GPU; every quantized object of
the layer is rounded by GPTQ in the format and layout the recipe stores it (compressed q3's limits included) and saved
to CODES_DIR; the decoded weights replace the layer's, and the sequences run through it again, so the next layer is
calibrated on the quantized model's activations. The MTP layer is then rounded the same way on the rows the engine
feeds it (the quantized model's normed hidden at p with the embedding of token p + 1); the LM head is rounded under the
Hessian of the final norm's output and the draft head under that of the MTP output, which it reads. --act-order rounds
columns by decreasing diag(H). The token embedding (a lookup) is left to the converter.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from euhedral_artifacts import device
from euhedral_artifacts.codes import state_file
from euhedral_artifacts.importance import Calibration
from euhedral_artifacts.inventory import build_plans, draft_token_ids
from euhedral_artifacts.nvfp4 import E2M1_VALUES, E4M3_VALUES
from euhedral_artifacts.recipes import Recipe
from gptq.gptq import column_order, gptq, inverse_cholesky, nvfp4_tensor_scales
from gptq.qwen38 import HIDDEN, LAYERS, Checkpoint, Layer, Mtp, linear, rms_norm

HEAD_ROWS = 16384
ROUNDING_THREADS = 3
CHECKPOINT_EVERY = 8


def checkpoint_key(args, sequences: int) -> dict:
    """What a hidden-state checkpoint depends on: a resumed run must match it."""
    return {"tokens": str(Path(args.tokens).resolve()), "sequences": sequences, "quantization": args.quantization,
            "compressed": args.compressed, "act_order": args.act_order, "damp": args.damp, "batch": args.batch}


def save_hidden(out: Path, hidden: torch.Tensor, layers_done: int, key: dict) -> None:
    """The hidden states after `layers_done` layers (pinned host memory) as hidden-N.bin; hidden.json names the
    newest complete one, and older ones are deleted after it does."""
    name = f"hidden-{layers_done}.bin"
    with open(out / (name + ".partial"), "wb") as handle:
        handle.write(memoryview(hidden.view(torch.int16).numpy()).cast("B"))
    os.replace(out / (name + ".partial"), out / name)
    (out / "hidden.json.partial").write_text(json.dumps({**key, "layers_done": layers_done, "file": name}))
    os.replace(out / "hidden.json.partial", out / "hidden.json")
    for old in out.glob("hidden-*.bin"):
        if old.name != name:
            old.unlink()


def drop_hidden(out: Path) -> None:
    for path in [*out.glob("hidden-*.bin*"), out / "hidden.json"]:
        path.unlink(missing_ok=True)


def load_hidden(out: Path, hidden: torch.Tensor, key: dict) -> int:
    """Layers done by a matching checkpoint, its hidden states read into `hidden`; 0 without one."""
    meta = out / "hidden.json"
    if not meta.is_file():
        return 0
    saved = json.loads(meta.read_text())
    layers_done, name = saved.pop("layers_done"), saved.pop("file")
    if saved != key or not (out / name).is_file():
        return 0
    with open(out / name, "rb") as handle:
        handle.readinto(memoryview(hidden.view(torch.int16).numpy()).cast("B"))
    return layers_done


def calibration(self, name: str, k: int) -> Calibration:
        return Calibration()


def log(start: float, message: str) -> None:
    print(f"[{time.time() - start:7.0f} s] {message}", flush=True)


def round_object(name, weight, hessian, plan, out, damp, start, rows=None, act_order=False):
    """GPTQ of one object (in row slices of `rows` when given), saved to `out`; returns the decoded weights (BF16)."""
    shape = tuple(plan.shape)
    lowest = plan.source[1].lowest_code if plan.source[1] is not None else None
    if act_order:
        order = column_order(hessian)
        permuted = hessian[order]
        permuted = permuted[:, order]  # one 1.2 GB copy at a time for mlp/down
        upper = inverse_cholesky(permuted, damp, consume=True)
        del permuted
    else:
        order = torch.arange(hessian.shape[0], device=hessian.device)
        upper = inverse_cholesky(hessian, damp)
    tensor = nvfp4_tensor_scales(weight, hessian, plan.layout) if plan.format_name == "NVFP4" else None
    rows = rows or weight.shape[0]
    states, decoded = [], torch.empty_like(weight, dtype=torch.bfloat16)
    for begin in range(0, weight.shape[0], rows):
        end = min(weight.shape[0], begin + rows)
        state, values = gptq(weight[begin:end], hessian, plan.format_name, plan.layout, (end - begin, shape[1]),
                             lowest, damp, upper, tensor, order)
        states.append(state)
        decoded[begin:end] = values.to(torch.bfloat16)
        del values
    first = states[0]
    first.shape = shape
    if len(states) > 1:
        first.codes = np.concatenate([s.codes for s in states])
        first.scales = np.concatenate([s.scales for s in states])
    first.save(state_file(out, name))
    difference = total = 0.0
    for begin in range(0, weight.shape[0], rows):
        w = weight[begin:begin + rows].float()
        difference += float(((w - decoded[begin:begin + rows].float()) ** 2).sum())
        total += float((w ** 2).sum())
    error = (difference / total) ** 0.5
    log(start, f"{name} {plan.format_name}/{plan.layout} relative error {error:.4f}")
    return decoded


def accumulate(hessian: torch.Tensor, x: torch.Tensor) -> None:
    """hessian += x^T x for BF16 rows x [n, K]: BF16 tensor-core products, exact, accumulated in FP32 in place."""
    x = x.to(torch.bfloat16)
    torch.addmm(hessian, x.T, x, out_dtype=torch.float32, out=hessian)


class Observer:
    """The Hessians of a layer's linears, accumulated by `observe(name, x)`; linears that read the same input rows in
    a batch (query_key and gate_value or value_z) share one Hessian instead of computing it twice."""

    def __init__(self):
        self.hessians: dict[str, torch.Tensor] = {}
        self.alias: dict[str, str] = {}
        self.last: tuple[int, tuple[int, ...], str] | None = None

    def __call__(self, name: str, x: torch.Tensor) -> None:
        key = (x.data_ptr(), tuple(x.shape))
        if self.last is not None and self.last[:2] == key and name not in self.hessians:
            self.alias[name] = self.last[2]
            return
        if name in self.alias:
            return
        if name not in self.hessians:
            self.hessians[name] = torch.zeros(x.shape[1], x.shape[1], device="cuda")
        accumulate(self.hessians[name], x)
        self.last = (*key, name)

    def result(self) -> dict[str, torch.Tensor]:
        return {**self.hessians, **{name: self.hessians[source] for name, source in self.alias.items()}}


def batches(hidden: torch.Tensor, batch: int, step, write: bool = False) -> None:
    """step(rows on the GPU) for each batch of the pinned `hidden`, the next batch copied in on a side stream while the
    current one computes; with `write`, the result goes back into `hidden` without stopping the GPU."""
    copier = torch.cuda.Stream()
    compute = torch.cuda.current_stream()

    def load(begin):
        with torch.cuda.stream(copier):
            rows = hidden[begin:begin + batch].to("cuda", non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(copier)
        return rows, ready

    starts = list(range(0, hidden.shape[0], batch))
    pending = load(starts[0])
    for index, begin in enumerate(starts):
        rows, ready = pending
        compute.wait_event(ready)
        rows.record_stream(compute)
        if index + 1 < len(starts):
            # The next batch is never one this loop writes back before it (write-back goes to this batch only).
            pending = load(starts[index + 1])
        result = step(rows)
        if write:
            hidden[begin:begin + batch].copy_(result, non_blocking=True)
    torch.cuda.synchronize()


def round_objects(weights: dict[str, torch.Tensor], hessians: dict[str, torch.Tensor], plans, args, start) -> None:
    """Rounds every object of `weights` under its Hessian at once, each on its own thread and CUDA stream (one object's
    column loop launches kernels from one thread and leaves the GPU idle between them); replaces each with its decoded
    weights. The objects are independent, so the result is the sequential one."""
    device.table("e4m3", E4M3_VALUES), device.table("e2m1", E2M1_VALUES)  # shared tables, made before the threads
    inverse_cholesky(torch.eye(2, device="cuda"), 0.0)  # torch.linalg initializes lazily, and not thread-safely
    torch.cuda.synchronize()

    def one(name: str) -> torch.Tensor:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            decoded = round_object(name, weights[name], hessians[name], plans[name], args.out, args.damp, start,
                                   act_order=args.act_order)
        stream.synchronize()
        hessians[name] = None  # released as each object finishes (shared Hessians when both have)
        return decoded

    # Largest first, at most ROUNDING_THREADS at once: five objects together peaked above a 16 GB card.
    names = sorted(weights, key=lambda name: weights[name].numel() + hessians[name].numel(), reverse=True)
    with concurrent.futures.ThreadPoolExecutor(ROUNDING_THREADS) as pool:
        for name, decoded in zip(names, pool.map(one, names)):
            weights[name] = decoded
    hessians.clear()


def mtp_pass(checkpoint, hidden, ids, final_norm, head_hessian, plans, args, start) -> torch.Tensor:
    """Accumulates the LM head's Hessian from the base final norm's output, rounds the MTP layer on the rows the engine
    feeds it (docs/MTP_CONTRACT.md: base normed hidden at p with the embedding of token p + 1, at position p), and
    returns the draft head's Hessian, from the MTP output it reads. `hidden` is overwritten with the MTP rows."""
    mtp = Mtp(checkpoint)
    embedding = checkpoint.tensor("model.language_model.embed_tokens.weight")
    sequences, batch = hidden.shape[0], args.batch

    def stem(begin):
        normed = rms_norm(hidden[begin:begin + batch].cuda(), final_norm, checkpoint.eps)
        return normed, mtp.stem(embedding[ids[begin:begin + batch, 1:].cuda()], normed[:, :-1])

    stem_hessian = torch.zeros(2 * HIDDEN, 2 * HIDDEN, device="cuda")
    for begin in range(0, sequences, batch):
        normed, x = stem(begin)
        accumulate(head_hessian, normed.reshape(-1, HIDDEN))
        accumulate(stem_hessian, x.reshape(-1, 2 * HIDDEN))
    projection = round_object("mtp/input_projection", mtp.input_projection, stem_hessian, plans["mtp/input_projection"],
                              args.out, args.damp, start, act_order=args.act_order)
    del stem_hessian
    for begin in range(0, sequences, batch):
        _, x = stem(begin)
        hidden[begin:begin + batch, :-1] = linear(x, projection).cpu()
    del embedding, projection
    torch.cuda.empty_cache()
    rows = hidden[:, :-1]
    observe = Observer()
    for begin in range(0, sequences, batch):
        mtp.layer.forward(rows[begin:begin + batch].cuda(), observe)
    round_objects(mtp.layer.weights, observe.result(), plans, args, start)
    draft_hessian = torch.zeros(HIDDEN, HIDDEN, device="cuda")
    for begin in range(0, sequences, batch):
        m = rms_norm(mtp.layer.forward(rows[begin:begin + batch].cuda()), mtp.final_norm, checkpoint.eps)
        accumulate(draft_hessian, m.reshape(-1, HIDDEN))
    log(start, "MTP layer done")
    return draft_hessian


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--quantization", required=True, choices=("q3", "nvfp4"))
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--draft-ids-from", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sequences", type=int, default=0, help="use the first N sequences (default: all)")
    parser.add_argument("--batch", type=int, default=8, help="sequences per forward batch")
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--layers", type=int, default=LAYERS, help="calibrate the first N layers only (a smoke run)")
    parser.add_argument("--act-order", action="store_true", help="round columns by decreasing diag(H)")
    parser.add_argument("--skip-mtp", action="store_true",
                        help="leave the MTP layer to the converter; the draft head reads the base final norm's Hessian")
    args = parser.parse_args(argv)
    device.set_device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.model)
    plans = {p.name: p for p in build_plans(checkpoint.store, draft_token_ids(args.draft_ids_from),
                                            Recipe(args.quantization, args.compressed), _Uniform())
             if p.source is not None}
    tokens = np.load(args.tokens)
    if args.sequences:
        tokens = tokens[:args.sequences]
    sequences, length = tokens.shape
    with torch.inference_mode():
        hidden = torch.empty(sequences, length, HIDDEN, dtype=torch.bfloat16, pin_memory=True)
        ids = torch.from_numpy(tokens.astype(np.int64))
        key = checkpoint_key(args, sequences)
        first = load_hidden(args.out, hidden, key)
        if first:
            log(start, f"resumed after layer {first - 1} from a checkpoint in {args.out}")
        else:
            embedding = checkpoint.tensor("model.language_model.embed_tokens.weight")
            for begin in range(0, sequences, args.batch):
                hidden[begin:begin + args.batch] = embedding[ids[begin:begin + args.batch].cuda()].cpu()
            del embedding
            torch.cuda.empty_cache()
            log(start, f"{sequences} sequences of {length} tokens embedded")
        for index in range(first, args.layers):
            layer = Layer(checkpoint, index)
            observe = Observer()

            batches(hidden, args.batch, lambda rows: layer.forward(rows, observe))
            round_objects(layer.weights, observe.result(), plans, args, start)
            batches(hidden, args.batch, layer.forward, write=True)
            del layer, observe
            torch.cuda.empty_cache()
            log(start, f"layer {index} done, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
            torch.cuda.reset_peak_memory_stats()
            if (index + 1) % CHECKPOINT_EVERY == 0 and index + 1 < args.layers:
                save_hidden(args.out, hidden, index + 1, key)  # a crash resumes here (the codes are saved per object)
        drop_hidden(args.out)
        final_norm = checkpoint.tensor("model.language_model.norm.weight")
        head_hessian = torch.zeros(HIDDEN, HIDDEN, device="cuda")
        draft_hessian = head_hessian
        if args.skip_mtp:
            for begin in range(0, sequences, args.batch):
                accumulate(head_hessian, rms_norm(hidden[begin:begin + args.batch].cuda(), final_norm,
                                                  checkpoint.eps).reshape(-1, HIDDEN))
        else:
            draft_hessian = mtp_pass(checkpoint, hidden, ids, final_norm, head_hessian, plans, args, start)
        del hidden
        torch.cuda.empty_cache()
        head = checkpoint.tensor("lm_head.weight")
        round_object("text/output_head", head, head_hessian, plans["text/output_head"], args.out, args.damp, start,
                     HEAD_ROWS, args.act_order)
        selected = torch.from_numpy(draft_token_ids(args.draft_ids_from)).cuda()
        round_object("text/draft_head", head[selected], draft_hessian, plans["text/draft_head"], args.out, args.damp,
                     start, HEAD_ROWS, args.act_order)
    log(start, "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
