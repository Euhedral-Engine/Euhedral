#!/usr/bin/env python3
"""Compare teacher-forced logits reports (TeacherForcedQualityCudaIntegrationTest) with a BF16 reference.

Usage: compare_reference.py [--detail] REFERENCE.kld CANDIDATE [CANDIDATE ...]

REFERENCE.kld is the file `llama-perplexity --kl-divergence-base` writes for the BF16 checkpoint: for each chunk of
N tokens it holds the reference log-probabilities of the chunk's second half, quantized to 16 bits over the 16 nats
below each position's maximum. A CANDIDATE is a report file covering one chunk, or the REPORT prefix given to the test
with `euhedral.quality.tokens`, whose `REPORT-chunkNNN.bin` files cover every chunk. Each chunk is forced with prefix
N / 2 + 1 and N / 2 - 1 steps, so that step k forces the token the reference's record k predicts.

For each candidate: mean negative log-likelihood of the forced tokens, the paired difference to the reference with
its standard error, the mean KL divergence KL(reference || candidate) with its standard error, and the top-1
agreement. Over several chunks the standard errors are chunk-level (tokens within a document are correlated); over
one chunk they are token-level. --detail adds the median, 95th, 99th percentile and maximum KL, the top-1 standard
error, and the mean, RMS, 1st and 99th percentile of delta-p (the candidate's probability of the forced token minus
the reference's). Needs NumPy.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROWS = 128


class Reference:
    def __init__(self, path: str):
        with open(path, "rb") as handle:
            if handle.read(8) != b"_logits_":
                raise SystemExit(f"{path} is not a llama.cpp logits file")
            self.n_ctx, self.n_vocab, self.chunks = (int(v) for v in np.fromfile(handle, dtype="<i4", count=3))
        self.tokens = np.memmap(path, dtype="<i4", mode="r", offset=20, shape=(self.chunks, self.n_ctx))
        self.first = self.n_ctx // 2
        self.records = self.n_ctx - 1 - self.first
        width = 2 * ((self.n_vocab + 1) // 2) + 4
        self.data = np.memmap(path, dtype="<u2", mode="r", offset=20 + 4 * self.chunks * self.n_ctx,
                              shape=(self.chunks, self.records, width))

    def targets(self, chunk: int) -> np.ndarray:
        return np.asarray(self.tokens[chunk, self.first + 1:self.first + 1 + self.records])

    def log_probabilities(self, chunk: int, begin: int, end: int) -> np.ndarray:
        rows = np.asarray(self.data[chunk, begin:end])
        header = rows[:, :4].copy().view("<f4")
        return header[:, 0, None] * rows[:, 4:4 + self.n_vocab].astype(np.float64) + header[:, 1, None]


def read_report(path: str):
    with open(path, "rb") as handle:
        steps, vocabulary = np.frombuffer(handle.read(8), "<i4")
    record = 4 + 2 * int(vocabulary)
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(int(steps), record))
    return np.asarray(data[:, :4]).copy().view("<i4")[:, 0], data


def log_softmax(rows: np.ndarray) -> np.ndarray:
    logits = (np.ascontiguousarray(rows).view("<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def reference_nll(reference: Reference, chunk: int) -> np.ndarray:
    targets, out = reference.targets(chunk), np.empty(reference.records)
    for begin in range(0, reference.records, ROWS):
        end = min(reference.records, begin + ROWS)
        p = reference.log_probabilities(chunk, begin, end)
        out[begin:end] = -p[np.arange(end - begin), targets[begin:end]]
    return out


def score(reference: Reference, chunk: int, path: str) -> dict[str, np.ndarray]:
    """Per-token NLL, KL, top-1 agreement and delta-p of the report `path` for `chunk`."""
    targets = reference.targets(chunk)
    forced, report = read_report(path)
    if len(forced) != reference.records or not np.array_equal(forced, targets):
        raise SystemExit(f"{path} forced different tokens (steps {len(forced)}, expected {reference.records})")
    n = reference.records
    out = {key: np.empty(n) for key in ("nll", "kl", "delta_p")}
    out["agree"] = np.empty(n, dtype=bool)
    for begin in range(0, n, ROWS):
        end = min(n, begin + ROWS)
        index = np.arange(end - begin)
        p = reference.log_probabilities(chunk, begin, end)
        q = log_softmax(report[begin:end, 4:])
        out["nll"][begin:end] = -q[index, targets[begin:end]]
        out["kl"][begin:end] = (np.exp(p) * (p - q)).sum(axis=1)
        out["agree"][begin:end] = p.argmax(axis=1) == q.argmax(axis=1)
        out["delta_p"][begin:end] = np.exp(q[index, targets[begin:end]]) - np.exp(p[index, targets[begin:end]])
    return out


def chunk_files(candidate: str, chunks: int) -> list[str]:
    if os.path.isfile(candidate):
        return [candidate]
    files = sorted(glob.glob(f"{glob.escape(candidate)}-chunk[0-9][0-9][0-9].bin"))
    if len(files) != chunks:
        raise SystemExit(f"{candidate}: found {len(files)} chunk reports, the reference has {chunks} chunks")
    return files


def standard_error(per_chunk: np.ndarray, per_token: np.ndarray) -> float:
    if len(per_chunk) > 1:
        return float(per_chunk.std(ddof=1) / np.sqrt(len(per_chunk)))
    return float(per_token.std(ddof=1) / np.sqrt(len(per_token)))


def main(argv: list[str]) -> int:
    detail = "--detail" in argv
    argv = [a for a in argv if a != "--detail"]
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    reference = Reference(argv[0])
    chunks = reference.chunks
    reference_by_chunk = [reference_nll(reference, c) for c in range(chunks)]
    reference_all = np.concatenate(reference_by_chunk)
    print(f"{chunks} chunk(s) of {reference.n_ctx}, {reference.records} forced tokens each "
          f"(positions {reference.first + 1}..{reference.n_ctx - 1}); reference {argv[0]}")
    header = f"{'report':50s} {'mean NLL':>9s} {'ppl':>8s} {'dNLL':>9s} {'+-SE':>8s} {'KL(ref||c)':>11s} {'+-SE':>9s} {'top-1':>7s}"
    if detail:
        header += f" {'KL med':>8s} {'KL p95':>8s} {'KL p99':>8s} {'KL max':>8s} {'top1 SE':>8s} {'dp mean':>8s} {'dp RMS':>8s} {'dp p1':>8s} {'dp p99':>8s}"
    print(header)
    print(f"{argv[0]:50s} {reference_all.mean():9.5f} {np.exp(reference_all.mean()):8.4f}")
    for candidate in argv[1:]:
        files = chunk_files(candidate, chunks)
        if len(files) == 1 and chunks != 1:
            raise SystemExit(f"{candidate} is one report; the reference has {chunks} chunks")
        scored = [score(reference, c, f) for c, f in enumerate(files)]
        tokens = {key: np.concatenate([s[key] for s in scored]) for key in scored[0]}
        difference = tokens["nll"] - reference_all
        chunk_difference = np.array([s["nll"].mean() - r.mean() for s, r in zip(scored, reference_by_chunk)])
        chunk_kl = np.array([s["kl"].mean() for s in scored])
        line = (f"{candidate:50s} {tokens['nll'].mean():9.5f} {np.exp(tokens['nll'].mean()):8.4f} "
                f"{difference.mean():+9.5f} {standard_error(chunk_difference, difference):8.5f} "
                f"{tokens['kl'].mean():11.6f} {standard_error(chunk_kl, tokens['kl']):9.6f} {tokens['agree'].mean():7.4f}")
        if detail:
            kl, dp, agree = tokens["kl"], tokens["delta_p"], tokens["agree"]
            chunk_agree = np.array([s["agree"].mean() for s in scored])
            top1_error = standard_error(chunk_agree, agree.astype(np.float64))
            line += (f" {np.median(kl):8.5f} {np.percentile(kl, 95):8.5f} {np.percentile(kl, 99):8.5f} {kl.max():8.4f}"
                     f" {top1_error:8.5f} {dp.mean():+8.5f} {np.sqrt((dp ** 2).mean()):8.5f}"
                     f" {np.percentile(dp, 1):+8.5f} {np.percentile(dp, 99):+8.5f}")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
