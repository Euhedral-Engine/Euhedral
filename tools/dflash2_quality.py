#!/usr/bin/env python3
"""Compare two DFlash2 drafting-quality reports (DFlash2QualityCudaIntegrationTest).

Usage: dflash2_quality.py REFERENCE.jsonl CANDIDATE.jsonl

Greedy output does not depend on the drafter, so both reports must hold the same generated tokens per prompt
(end-to-end equivalence). Their verifications differ: each report's steps start where its own previous step
ended. Blocks drafted from the same anchor position in both reports see the same context and anchor, so on those
the drafters are compared directly:

  top-1 agreement     share of proposal positions where both drafters propose the same token
  top-16 overlap      mean share of each position's 16 candidates both drafters hold
  proposal agreement  share of blocks whose 7 drafts are identical

and over every step of each report: accepted drafts per verification, its histogram, and tokens per verification.
"""
from __future__ import annotations

import json
import sys


def load(path: str):
    steps, outputs = {}, {}
    for line in open(path, encoding="utf-8"):
        record = json.loads(line)
        if "tokens" in record:
            outputs[record["prompt"]] = record["tokens"]
        else:
            steps[(record["prompt"], record["position"])] = record
    return steps, outputs


def summary(steps) -> str:
    histogram = [0] * 8
    for step in steps.values():
        histogram[step["accepted"]] += 1
    count = sum(histogram)
    mean = sum(a * n for a, n in enumerate(histogram)) / count
    return f"{count} verifications, accepted {histogram}, mean accepted {mean:.3f}, tokens/verification {mean + 1:.3f}"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    (reference, reference_out), (candidate, candidate_out) = load(argv[0]), load(argv[1])
    same = all(reference_out[p] == candidate_out.get(p) for p in reference_out)
    print(f"generated tokens identical on {len(reference_out)} prompts: {same}")
    print(f"reference: {summary(reference)}")
    print(f"candidate: {summary(candidate)}")
    shared = sorted(set(reference) & set(candidate))
    top1 = overlap = whole = positions = 0
    per_position = [0] * 7
    for key in shared:
        a, b = reference[key], candidate[key]
        whole += a["proposal"] == b["proposal"]
        for p in range(7):
            match = a["proposal"][p] == b["proposal"][p]
            top1 += match
            per_position[p] += match
            positions += 1
            ra, rb = set(a["candidates"][16 * p:16 * p + 16]), set(b["candidates"][16 * p:16 * p + 16])
            overlap += len(ra & rb) / 16
    if shared:
        print(f"blocks drafted from the same anchor position: {len(shared)}")
        print(f"top-1 agreement {top1 / positions:.4f}, per position "
              + ", ".join(f"{n / len(shared):.3f}" for n in per_position))
        print(f"top-16 overlap {overlap / positions:.4f}, proposal agreement {whole / len(shared):.4f}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
