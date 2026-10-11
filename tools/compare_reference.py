#!/usr/bin/env python3
"""Compare teacher-forced logits reports (TeacherForcedQualityCudaIntegrationTest) with a BF16 reference.

Usage: compare_reference.py [--detail] [--cpu] REFERENCE.kld CANDIDATE [CANDIDATE ...]

REFERENCE.kld is the file `llama-perplexity --kl-divergence-base` writes for the BF16 checkpoint: for each chunk of
N tokens it holds the reference log-probabilities of the chunk's second half, quantized to 16 bits over the 16 nats
below each position's maximum. A CANDIDATE is a report file covering one chunk, or the REPORT prefix given to the test
with `euhedral.quality.tokens`, whose `REPORT-chunkNNN.bin` files cover every chunk. Each chunk is forced with prefix
N / 2 + 1 and N / 2 - 1 steps, so that step k forces the token the reference's record k predicts.

For each candidate: mean negative log-likelihood of the forced tokens, the paired difference to the reference with
its standard error, the mean KL divergence KL(reference || candidate) over the reference entries above its 16-nat
floor (as llama.cpp computes it) with its standard error, and the top-1 agreement. Over several chunks the standard errors are chunk-level (tokens within a document are correlated); over
one chunk they are token-level. --detail adds the median, 95th, 99th percentile and maximum KL, the top-1 standard
error, and the mean, RMS, 1st and 99th percentile of delta-p (the candidate's probability of the forced token minus
the reference's). Needs NumPy; the per-token arithmetic runs on the GPU when PyTorch with CUDA is installed (--cpu
keeps it on the CPU).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROWS = 128
# Reference entries at or below this log-probability are clamped and left out of the KL (llama.cpp's rule).
FLOOR = -16.0


def pread(fd: int, size: int, offset: int) -> bytes:
    """Positioned read of exactly `size` bytes. The files are read, never memory-mapped: page faults on large
    mappings of the shared (exFAT) drive have hung the kernel."""
    parts, done = [], 0
    while done < size:
        part = os.pread(fd, size - done, offset + done)
        if not part:
            raise SystemExit("file is truncated")
        parts.append(part)
        done += len(part)
    return b"".join(parts)


class Reference:
    def __init__(self, path: str):
        self.handle = open(path, "rb")
        fd = self.handle.fileno()
        if pread(fd, 8, 0) != b"_logits_":
            raise SystemExit(f"{path} is not a llama.cpp logits file")
        self.n_ctx, self.n_vocab, self.chunks = (int(v) for v in np.frombuffer(pread(fd, 12, 8), "<i4"))
        self.tokens = np.frombuffer(pread(fd, 4 * self.chunks * self.n_ctx, 20), "<i4").reshape(self.chunks, self.n_ctx)
        self.first = self.n_ctx // 2
        self.records = self.n_ctx - 1 - self.first
        self.width = 2 * ((self.n_vocab + 1) // 2) + 4
        self.data_offset = 20 + 4 * self.chunks * self.n_ctx

    def targets(self, chunk: int) -> np.ndarray:
        return self.tokens[chunk, self.first + 1:self.first + 1 + self.records]

    def rows(self, chunk: int, begin: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """(per-row float32 scale and minimum, uint16 codes) of records begin..end of `chunk`."""
        offset = self.data_offset + 2 * self.width * (chunk * self.records + begin)
        rows = np.frombuffer(pread(self.handle.fileno(), 2 * self.width * (end - begin), offset), "<u2")
        rows = rows.reshape(end - begin, self.width)
        return rows[:, :4].copy().view("<f4"), rows[:, 4:4 + self.n_vocab]

    def log_probabilities(self, chunk: int, begin: int, end: int) -> np.ndarray:
        header, codes = self.rows(chunk, begin, end)
        return header[:, 0, None] * codes.astype(np.float64) + header[:, 1, None]


class Report:
    """A teacher-forced report: int32 steps and vocabulary, then per step the int32 token and the BF16 logits row."""

    def __init__(self, path: str):
        self.handle = open(path, "rb")
        steps, self.vocabulary = (int(v) for v in np.frombuffer(pread(self.handle.fileno(), 8, 0), "<i4"))
        self.record = 4 + 2 * self.vocabulary
        self.steps = steps

    def rows(self, begin: int, end: int) -> np.ndarray:
        data = pread(self.handle.fileno(), self.record * (end - begin), 8 + self.record * begin)
        return np.frombuffer(data, np.uint8).reshape(end - begin, self.record)

    def forced(self) -> np.ndarray:
        out = np.empty(self.steps, np.int32)
        for begin in range(0, self.steps, ROWS):
            end = min(self.steps, begin + ROWS)
            out[begin:end] = self.rows(begin, end)[:, :4].copy().view("<i4")[:, 0]
        return out


def read_report(path: str):
    """(forced tokens, Report) of a report file."""
    report = Report(path)
    return report.forced(), report


def log_softmax(rows: np.ndarray) -> np.ndarray:
    logits = (np.ascontiguousarray(rows).view("<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row KL(reference || candidate) over the entries above the reference's floor, as llama.cpp computes it. The
    reference clamps every entry more than 16 nats below its maximum to one floor value, about 2% of the probability
    mass over the vocabulary; summing those entries would charge a candidate whose tail lies below the floor for the
    clamping (0.02 nats per nat of gap, about 0.13 on real models)."""
    return np.where(p > FLOOR, np.exp(p) * (p - q), 0.0).sum(axis=1)


def reference_nll(reference: Reference, chunk: int) -> np.ndarray:
    targets, out = reference.targets(chunk), np.empty(reference.records)
    for begin in range(0, reference.records, ROWS):
        end = min(reference.records, begin + ROWS)
        p = reference.log_probabilities(chunk, begin, end)
        out[begin:end] = -p[np.arange(end - begin), targets[begin:end]]
    return out


def cuda_torch():
    """PyTorch when it has a CUDA device, else None: the per-row arithmetic over the vocabulary is most of the
    scorer's time on one CPU core, and a few milliseconds per block on the GPU."""
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def score_cuda(torch, reference: Reference, chunk: int, report: Report, targets: np.ndarray) -> dict[str, np.ndarray]:
    """score() on the GPU, in float64 like the NumPy path."""
    n, device = reference.records, torch.device("cuda")
    out = {key: np.empty(n) for key in ("nll", "kl", "delta_p")}
    out["agree"] = np.empty(n, dtype=bool)
    for begin in range(0, n, ROWS):
        end = min(n, begin + ROWS)
        header, codes = reference.rows(chunk, begin, end)
        header = torch.from_numpy(header.astype(np.float64)).to(device)
        codes = torch.from_numpy(np.ascontiguousarray(codes).view(np.int16)).to(device).to(torch.int32) & 0xFFFF
        p = header[:, 0, None] * codes.to(torch.float64) + header[:, 1, None]
        bits = torch.from_numpy(np.ascontiguousarray(report.rows(begin, end)[:, 4:]).view(np.int16)).to(device)
        q = torch.log_softmax(((bits.to(torch.int32) & 0xFFFF) << 16).view(torch.float32).to(torch.float64), dim=1)
        index = torch.arange(end - begin, device=device)
        target = torch.from_numpy(targets[begin:end].astype(np.int64)).to(device)
        q_target, p_target = q[index, target], p[index, target]
        out["nll"][begin:end] = (-q_target).cpu().numpy()
        out["kl"][begin:end] = torch.where(p > FLOOR, p.exp() * (p - q), 0.0).sum(dim=1).cpu().numpy()
        out["agree"][begin:end] = (p.argmax(dim=1) == q.argmax(dim=1)).cpu().numpy()
        out["delta_p"][begin:end] = (q_target.exp() - p_target.exp()).cpu().numpy()
    return out


def score(reference: Reference, chunk: int, path: str, torch=None) -> dict[str, np.ndarray]:
    """Per-token NLL, KL, top-1 agreement and delta-p of the report `path` for `chunk` (on the GPU when `torch` is
    given)."""
    targets = reference.targets(chunk)
    forced, report = read_report(path)
    if len(forced) != reference.records or not np.array_equal(forced, targets):
        raise SystemExit(f"{path} forced different tokens (steps {len(forced)}, expected {reference.records})")
    if torch is not None:
        return score_cuda(torch, reference, chunk, report, targets)
    n = reference.records
    out = {key: np.empty(n) for key in ("nll", "kl", "delta_p")}
    out["agree"] = np.empty(n, dtype=bool)
    for begin in range(0, n, ROWS):
        end = min(n, begin + ROWS)
        index = np.arange(end - begin)
        p = reference.log_probabilities(chunk, begin, end)
        q = log_softmax(report.rows(begin, end)[:, 4:])
        out["nll"][begin:end] = -q[index, targets[begin:end]]
        out["kl"][begin:end] = kl_divergence(p, q)
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
    torch = None if "--cpu" in argv else cuda_torch()
    argv = [a for a in argv if a not in ("--detail", "--cpu")]
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
        scored = [score(reference, c, f, torch) for c, f in enumerate(files)]
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
