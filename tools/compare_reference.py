#!/usr/bin/env python3
"""Compare teacher-forced logits reports (TeacherForcedQualityCudaIntegrationTest) with a BF16 reference.

Usage: compare_reference.py REFERENCE.kld CANDIDATE.bin [CANDIDATE.bin ...]

REFERENCE.kld is the file `llama-perplexity --kl-divergence-base` writes for the BF16 checkpoint (llama.cpp, CPU,
one chunk of N tokens): it holds the reference log-probabilities of the second half of the chunk, quantized to 16
bits over the 16 nats below each position's maximum. Each candidate report must cover the same tokens: prefix
N / 2 + 1 and N / 2 - 1 steps, so that step k forces the token the reference's record k predicts.

For each candidate: mean negative log-likelihood of the forced tokens, the paired difference to the reference with
its standard error, the mean KL divergence KL(reference || candidate), and the top-1 agreement. Needs NumPy.
"""
from __future__ import annotations

import sys

import numpy as np


def read_reference(path: str):
    with open(path, "rb") as handle:
        if handle.read(8) != b"_logits_":
            raise SystemExit(f"{path} is not a llama.cpp logits file")
        n_ctx, n_vocab, n_chunk = np.fromfile(handle, dtype="<i4", count=3)
        tokens = np.fromfile(handle, dtype="<i4", count=int(n_ctx) * int(n_chunk))
        first = int(n_ctx) // 2
        records = int(n_ctx) - 1 - first
        width = 2 * ((int(n_vocab) + 1) // 2) + 4
        data = np.fromfile(handle, dtype="<u2", count=records * width).reshape(records, width)
    header = data[:, :4].copy().view("<f4")
    scale, minimum = header[:, 0], header[:, 1]
    return int(n_ctx), int(n_vocab), tokens[: int(n_ctx)], first, data[:, 4 : 4 + int(n_vocab)], scale, minimum


def read_report(path: str):
    with open(path, "rb") as handle:
        steps, vocabulary = np.frombuffer(handle.read(8), "<i4")
        record = 4 + 2 * int(vocabulary)
        data = np.fromfile(handle, dtype=np.uint8, count=int(steps) * record).reshape(int(steps), record)
    return data[:, :4].copy().view("<i4")[:, 0], data[:, 4:]


def log_softmax(rows: np.ndarray) -> np.ndarray:
    logits = (rows.copy().view("<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    n_ctx, n_vocab, tokens, first, codes, scale, minimum = read_reference(argv[0])
    steps = n_ctx - 1 - first
    targets = tokens[first + 1 : first + 1 + steps]
    print(f"{steps} forced tokens (positions {first + 1}..{n_ctx - 1}); reference {argv[0]}")
    print(f"{'report':50s} {'mean NLL':>9s} {'ppl':>8s} {'dNLL':>9s} {'+-SE':>8s} {'KL(ref||c)':>11s} {'top-1':>7s}")
    chunk = 128
    reference_nll = np.empty(steps)
    for begin in range(0, steps, chunk):
        end = min(steps, begin + chunk)
        p = scale[begin:end, None] * codes[begin:end].astype(np.float64) + minimum[begin:end, None]
        reference_nll[begin:end] = -p[np.arange(end - begin), targets[begin:end]]
    print(f"{argv[0]:50s} {reference_nll.mean():9.5f} {np.exp(reference_nll.mean()):8.4f}")
    for path in argv[1:]:
        forced, report = read_report(path)
        if len(forced) != steps or not np.array_equal(forced, targets):
            raise SystemExit(f"{path} forced different tokens (steps {len(forced)}, expected {steps})")
        nll, kl, agree = np.empty(steps), np.empty(steps), np.empty(steps, dtype=bool)
        for begin in range(0, steps, chunk):
            end = min(steps, begin + chunk)
            p = scale[begin:end, None] * codes[begin:end].astype(np.float64) + minimum[begin:end, None]
            q = log_softmax(report[begin:end])
            nll[begin:end] = -q[np.arange(end - begin), targets[begin:end]]
            kl[begin:end] = (np.exp(p) * (p - q)).sum(axis=1)
            agree[begin:end] = p.argmax(axis=1) == q.argmax(axis=1)
        difference = nll - reference_nll
        error = difference.std(ddof=1) / np.sqrt(steps)
        print(f"{path:50s} {nll.mean():9.5f} {np.exp(nll.mean()):8.4f} {difference.mean():+9.5f} "
              f"{error:8.5f} {kl.mean():11.6f} {agree.mean():7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
