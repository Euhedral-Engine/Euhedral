#!/usr/bin/env python3
"""Compare two DFlash2 fixture directories (tools/dflash2_reference.py, or the engine's dump of the same tensors).

Usage: dflash2_compare.py REFERENCE_DIR CANDIDATE_DIR [--max-relative-rms X]

Per tensor present in both: relative RMS error ||c - r|| / ||r||, the largest absolute difference, and for integer
tensors (candidates, proposal) the share of equal entries. The top-16 candidate sets are compared as sets per position,
so their order (unsorted upstream) does not matter. Exits non-zero when a float tensor exceeds --max-relative-rms or
the proposal differs and --require-proposal is given. Needs NumPy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(directory: Path) -> dict[str, np.ndarray]:
    manifest = json.loads((directory / "manifest.json").read_text())
    out = {}
    for name, entry in manifest["tensors"].items():
        raw = (directory / entry["file"]).read_bytes()
        if entry["dtype"] == "bf16":
            values = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
        elif entry["dtype"] == "f32":
            values = np.frombuffer(raw, "<f4")
        else:
            values = np.frombuffer(raw, "<i4")
        out[name] = values.reshape(entry["shape"])
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-relative-rms", type=float, default=None)
    parser.add_argument("--require-proposal", action="store_true")
    args = parser.parse_args(argv)
    reference, candidate = load(args.reference), load(args.candidate)
    failed = False
    print(f"{'tensor':40s} {'shape':>16s} {'rel rms':>10s} {'max abs':>10s}")
    for name, expected in reference.items():
        if name not in candidate:
            continue
        actual = candidate[name]
        if actual.shape != expected.shape:
            print(f"{name:40s} shape {actual.shape} != {expected.shape}")
            failed = True
            continue
        if name == "topk_indices":
            overlap = np.mean([len(set(a) & set(e)) / len(e) for a, e in zip(actual, expected)])
            print(f"{name:40s} {str(expected.shape):>16s} set overlap {overlap:.4f}")
            continue
        if expected.dtype == np.int32:
            equal = np.mean(actual == expected)
            print(f"{name:40s} {str(expected.shape):>16s} equal {equal:.4f}  {actual.tolist() if actual.size <= 16 else ''}")
            if name == "proposal" and args.require_proposal and equal != 1.0:
                failed = True
            continue
        if name == "topk_values":
            actual, expected = np.sort(actual, axis=-1), np.sort(expected, axis=-1)
        if name == "selector_scores":
            # Scores follow the candidate order, which upstream leaves unsorted: compare each row's sorted scores.
            actual, expected = np.sort(actual, axis=-1), np.sort(expected, axis=-1)
        diff = actual.astype(np.float64) - expected.astype(np.float64)
        norm = np.linalg.norm(expected.astype(np.float64))
        relative = np.linalg.norm(diff) / norm if norm > 0 else np.linalg.norm(diff)
        print(f"{name:40s} {str(expected.shape):>16s} {relative:10.3e} {np.abs(diff).max():10.3e}")
        if args.max_relative_rms is not None and relative > args.max_relative_rms:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
