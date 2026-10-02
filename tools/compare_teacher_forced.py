#!/usr/bin/env python3
"""Compare teacher-forced logits reports (TeacherForcedQualityCudaIntegrationTest).

Usage: compare_teacher_forced.py REFERENCE.bin CANDIDATE.bin [CANDIDATE.bin ...]

For each report: mean negative log-likelihood of the forced tokens (perplexity). For each candidate
against the reference, on the same tokens: the paired NLL difference with its standard error, the mean
KL divergence KL(reference || candidate), and how often their top-1 tokens agree. Needs NumPy.
"""
from __future__ import annotations

import sys

import numpy as np


def read(path: str):
    with open(path, "rb") as handle:
        steps, vocabulary = np.frombuffer(handle.read(8), "<i4")
        record = 4 + 2 * int(vocabulary)
        data = np.fromfile(handle, dtype=np.uint8, count=int(steps) * record).reshape(int(steps), record)
    tokens = data[:, :4].copy().view("<i4")[:, 0]
    return tokens, data[:, 4:]


def log_softmax(rows: np.ndarray) -> np.ndarray:
    logits = (rows.copy().view("<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    reference_tokens, reference = read(argv[0])
    candidates = [(path, *read(path)) for path in argv[1:]]
    steps = len(reference_tokens)
    for path, tokens, _ in candidates:
        if len(tokens) != steps or not np.array_equal(tokens, reference_tokens):
            raise SystemExit(f"{path} forced different tokens")
    chunk = 128
    nll = {path: np.empty(steps) for path in argv}
    kl = {path: np.empty(steps) for path, _, _ in candidates}
    agree = {path: np.empty(steps, dtype=bool) for path, _, _ in candidates}
    rows = np.arange(steps)
    for begin in range(0, steps, chunk):
        end = min(steps, begin + chunk)
        p = log_softmax(reference[begin:end])
        nll[argv[0]][begin:end] = -p[np.arange(end - begin), reference_tokens[begin:end]]
        for path, tokens, data in candidates:
            q = log_softmax(data[begin:end])
            nll[path][begin:end] = -q[np.arange(end - begin), tokens[begin:end]]
            kl[path][begin:end] = (np.exp(p) * (p - q)).sum(axis=1)
            agree[path][begin:end] = p.argmax(axis=1) == q.argmax(axis=1)
    del rows
    print(f"{steps} forced tokens; reference {argv[0]}")
    print(f"{'report':50s} {'mean NLL':>9s} {'ppl':>8s} {'dNLL':>9s} {'+-SE':>8s} {'KL(ref||c)':>11s} {'top-1':>7s}")
    print(f"{argv[0]:50s} {nll[argv[0]].mean():9.5f} {np.exp(nll[argv[0]].mean()):8.4f}")
    for path, _, _ in candidates:
        difference = nll[path] - nll[argv[0]]
        error = difference.std(ddof=1) / np.sqrt(steps)
        print(f"{path:50s} {nll[path].mean():9.5f} {np.exp(nll[path].mean()):8.4f} {difference.mean():+9.5f} "
              f"{error:8.5f} {kl[path].mean():11.6f} {agree[path].mean():7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
