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
import time
from pathlib import Path

import numpy as np
import torch

from euhedral_artifacts import device
from euhedral_artifacts.codes import state_file
from euhedral_artifacts.importance import Calibration
from euhedral_artifacts.inventory import build_plans, draft_token_ids
from euhedral_artifacts.recipes import Recipe
from gptq.gptq import column_order, gptq, inverse_cholesky, nvfp4_tensor_scales
from gptq.qwen38 import HIDDEN, LAYERS, Checkpoint, Layer, Mtp, linear, rms_norm

HEAD_ROWS = 16384


class _Uniform:
    """An importance stub: plans carry the recipe's calibration (lowest code) without an importance matrix."""

    def calibration(self, name: str, k: int) -> Calibration:
        return Calibration()


def log(start: float, message: str) -> None:
    print(f"[{time.time() - start:7.0f} s] {message}", flush=True)


def round_object(name, weight, hessian, plan, out, damp, start, rows=None, act_order=False):
    """GPTQ of one object (in row slices of `rows` when given), saved to `out`; returns the decoded weights (BF16)."""
    shape = tuple(plan.shape)
    lowest = plan.source[1].lowest_code if plan.source[1] is not None else None
    order = column_order(hessian) if act_order else torch.arange(hessian.shape[0], device=hessian.device)
    upper = inverse_cholesky(hessian[order][:, order], damp)
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
        flat = normed.reshape(-1, HIDDEN).float()
        head_hessian.addmm_(flat.T, flat)
        flat = x.reshape(-1, 2 * HIDDEN).float()
        stem_hessian.addmm_(flat.T, flat)
    projection = round_object("mtp/input_projection", mtp.input_projection, stem_hessian, plans["mtp/input_projection"],
                              args.out, args.damp, start, act_order=args.act_order)
    del stem_hessian
    for begin in range(0, sequences, batch):
        _, x = stem(begin)
        hidden[begin:begin + batch, :-1] = linear(x, projection).cpu()
    del embedding, projection
    torch.cuda.empty_cache()
    rows = hidden[:, :-1]
    hessians: dict[str, torch.Tensor] = {}

    def observe(name: str, x: torch.Tensor) -> None:
        xf = x.float()
        if name not in hessians:
            hessians[name] = torch.zeros(xf.shape[1], xf.shape[1], device="cuda")
        hessians[name].addmm_(xf.T, xf)

    for begin in range(0, sequences, batch):
        mtp.layer.forward(rows[begin:begin + batch].cuda(), observe)
    for name in list(mtp.layer.weights):
        mtp.layer.weights[name] = round_object(name, mtp.layer.weights[name], hessians.pop(name), plans[name], args.out,
                                               args.damp, start, act_order=args.act_order)
    draft_hessian = torch.zeros(HIDDEN, HIDDEN, device="cuda")
    for begin in range(0, sequences, batch):
        m = rms_norm(mtp.layer.forward(rows[begin:begin + batch].cuda()), mtp.final_norm, checkpoint.eps)
        flat = m.reshape(-1, HIDDEN).float()
        draft_hessian.addmm_(flat.T, flat)
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
        embedding = checkpoint.tensor("model.language_model.embed_tokens.weight")
        hidden = torch.empty(sequences, length, HIDDEN, dtype=torch.bfloat16, pin_memory=True)
        ids = torch.from_numpy(tokens.astype(np.int64))
        for begin in range(0, sequences, args.batch):
            hidden[begin:begin + args.batch] = embedding[ids[begin:begin + args.batch].cuda()].cpu()
        del embedding
        torch.cuda.empty_cache()
        log(start, f"{sequences} sequences of {length} tokens embedded")
        for index in range(args.layers):
            layer = Layer(checkpoint, index)
            hessians: dict[str, torch.Tensor] = {}

            def observe(name: str, x: torch.Tensor) -> None:
                xf = x.float()
                if name not in hessians:
                    hessians[name] = torch.zeros(xf.shape[1], xf.shape[1], device="cuda")
                hessians[name].addmm_(xf.T, xf)

            for begin in range(0, sequences, args.batch):
                layer.forward(hidden[begin:begin + args.batch].cuda(non_blocking=True), observe)
            for name in list(layer.weights):
                decoded = round_object(name, layer.weights[name], hessians.pop(name), plans[name], args.out,
                                       args.damp, start, act_order=args.act_order)
                layer.weights[name] = decoded
            for begin in range(0, sequences, args.batch):
                hidden[begin:begin + args.batch] = layer.forward(hidden[begin:begin + args.batch].cuda(non_blocking=True)).cpu()
            del layer, hessians
            torch.cuda.empty_cache()
            log(start, f"layer {index} done")
        final_norm = checkpoint.tensor("model.language_model.norm.weight")
        head_hessian = torch.zeros(HIDDEN, HIDDEN, device="cuda")
        draft_hessian = head_hessian
        if args.skip_mtp:
            for begin in range(0, sequences, args.batch):
                x = rms_norm(hidden[begin:begin + args.batch].cuda(), final_norm, checkpoint.eps).reshape(-1, HIDDEN).float()
                head_hessian.addmm_(x.T, x)
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
