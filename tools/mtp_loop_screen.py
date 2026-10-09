#!/usr/bin/env python3
"""Screens MTP speculation policies offline on a drafting-confidence report (MtpDraftConfidenceCudaIntegrationTest).

Usage: mtp_loop_screen.py REPORT.jsonl [--policies]

The report holds, for every output position, the drafts the MTP head makes from it (up to 16) with the head's
log-probability of each, and the generated tokens (greedy decode's). A draft prefix's acceptance at a position is
its agreement with the generated tokens, so any policy that decides how many drafts a verification checks can be
replayed exactly: from position i a policy drafts n, the verification accepts a = min(agreement, n) and commits
a + 1 tokens, and the next step starts at i + a + 1.

Printed per category (chat, code, agentic, all):
  depth profile    P(draft j accepted | drafts 1..j-1 accepted) and the head's mean probability of draft j
  full acceptance  share of verifications at fixed depth 3/4/5 that accept every draft
  gate quality     AUROC of a block's summed log-probability for "the block is fully accepted" (first and second
                   block of 4), the DLoop gate's signal
  policies         tokens per verification and predicted decode tok/s at 4K/16K/32K/64K for fixed depths and for
                   looped policies: block k, threshold g, at most `cap` drafts (DLoop: draft another block while the
                   last block's summed log-probability >= g)

Predicted tok/s uses a cost model fitted to measured MTP4 steps (docs/DFLASH2.md section 6, nvfp4-compressed):
draft time per drafted token = MTP4 draft ms / 4; verification = MTP4 verify ms + ROW_MS per row beyond 5, and
WIDE_MS more from 9 rows (exact 9-16-row linears, about 1.3x one row's 20.7 ms, NVFP4_NATIVE.md). Steps past 8 rows
are estimates; the engine has never verified them.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

# Context: (MTP4 verify ms for 5 rows, MTP4 draft ms for 4 drafts, ms per extra verified row).
CONTEXTS = {"4K": (22.05, 3.78, 0.5), "16K": (23.97, 3.89, 0.6), "32K": (25.68, 4.17, 0.8), "64K": (37.86, 6.27, 1.5)}
WIDE_MS = 6.0


def load(path):
    steps, prompts = defaultdict(dict), {}
    for line in open(path, encoding="utf-8"):
        record = json.loads(line)
        if "tokens" in record:
            prompts[record["prompt"]] = record
        else:
            steps[record["prompt"]][record["position"]] = record
    sequences = []
    for index, prompt in sorted(prompts.items()):
        tokens, base = prompt["tokens"], prompt["promptTokens"]
        positions = []
        for i in range(len(tokens) - 1):
            step = steps[index].get(base + i)
            if step is None:
                raise SystemExit(f"prompt {index} has no step at output {i}")
            if step["current"] != tokens[i]:
                raise SystemExit(f"prompt {index} output {i}: step token {step['current']} != {tokens[i]}")
            drafts, logp = step["drafts"], step["logp"]
            # Drafts past the generated tokens cannot be checked: the agreement is censored there.
            known = min(len(drafts), len(tokens) - 1 - i)
            agree = 0
            while agree < known and drafts[agree] == tokens[i + 1 + agree]:
                agree += 1
            positions.append((agree, known, logp))
        sequences.append((prompt["name"], prompt["category"], positions))
    return sequences


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = (i + j + 1) / 2
        rank_sum += rank * sum(label for _, label in pairs[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def fixed(n):
    return lambda logp: n


def looped(k, g, cap):
    def drafts(logp):
        n = k
        while n + k <= cap and sum(logp[n - k : n]) >= g:
            n += k
        return min(n, cap)

    return drafts


def cumulative(threshold, cap, least=1):
    """Draft while the running sum of log-probabilities stays above the threshold (DDD-style, one token at a time)."""

    def drafts(logp):
        total, n = 0.0, 0
        while n < cap:
            total += logp[n]
            n += 1
            if n >= least and total < threshold:
                break
        return n

    return drafts


def replay(sequences, policy):
    verifications = tokens = drafted = 0
    rows = defaultdict(int)
    for _, _, positions in sequences:
        i = 0
        while i < len(positions):
            agree, known, logp = positions[i]
            n = policy(logp)
            # Near the end the step commits only the outputs left, as the decoder's budget cut does.
            committed = min(min(agree, n) + 1, len(positions) - i)
            verifications += 1
            tokens += committed
            drafted += n
            rows[n + 1] += 1
            i += committed
    return verifications, tokens, drafted, rows


def predicted(verifications, tokens, drafted, rows, context):
    verify5, draft4, row_ms = CONTEXTS[context]
    time = drafted * draft4 / 4
    for count, steps in rows.items():
        time += steps * (verify5 + row_ms * (count - 5) + (WIDE_MS if count > 8 else 0))
    return 1000 * tokens / time


def describe(sequences, label, show_policies):
    positions = [p for _, _, ps in sequences for p in ps]
    print(f"\n== {label}: {len(sequences)} prompts, {len(positions)} positions")
    line = []
    for j in range(8):
        reached = [p for p in positions if p[1] > j and p[0] >= j]
        if not reached:
            break
        accepted = sum(1 for p in reached if p[0] > j)
        confidence = sum(math.exp(p[2][j]) for p in reached) / len(reached)
        line.append(f"d{j + 1} {accepted / len(reached):.2f}/{confidence:.2f}")
    print("depth profile (accepted | prefix accepted / head probability): " + "  ".join(line))
    for n in (3, 4, 5):
        v, t, d, rows = replay(sequences, fixed(n))
        full = sum(1 for _, _, ps in sequences for p in ps if p[1] >= n and p[0] >= n)
        eligible = sum(1 for _, _, ps in sequences for p in ps if p[1] >= n)
        print(f"MTP{n}: {t / v:.2f} tokens/verification; full acceptance at a position {full / eligible:.1%}")
    first = [(sum(p[2][0:4]), p[0] >= 4) for p in positions if p[1] >= 4]
    second = [(sum(p[2][4:8]), p[0] >= 8) for p in positions if p[1] >= 8 and p[0] >= 4]
    print(
        f"gate AUROC: block 1 of 4 {auroc(*zip(*first)):.3f} ({sum(l for _, l in first) / len(first):.1%} full),"
        f" block 2 of 4 given block 1 {auroc(*zip(*second)):.3f} ({sum(l for _, l in second) / max(1, len(second)):.1%} full)"
        if second
        else f"gate AUROC: block 1 of 4 {auroc(*zip(*first)):.3f}"
    )
    candidates = [(f"MTP{n}", fixed(n)) for n in range(1, 8)]
    for k in (2, 3, 4):
        for g in (-0.1, -0.25, -0.5, -0.75, -1.0, -1.5):
            for cap in (7, 8, 12, 16):
                if cap >= 2 * k:
                    candidates.append((f"loop k{k} g{g} cap{cap}", looped(k, g, cap)))
    for threshold in (-0.5, -1.0, -1.5, -2.0, -3.0):
        for cap in (7, 16):
            for least in (1, 3):
                candidates.append((f"cumulative {threshold} cap{cap} min{least}", cumulative(threshold, cap, least)))
    results = []
    for name, policy in candidates:
        v, t, d, rows = replay(sequences, policy)
        results.append((name, t / v, d / v, {c: predicted(v, t, d, rows, c) for c in CONTEXTS}))
    base = {r[0]: r for r in results}["MTP4"]
    print(f"{'policy':34} tok/ver drafts/ver " + " ".join(f"{c:>12}" for c in CONTEXTS))

    def row(r):
        return f"{r[0]:34} {r[1]:7.2f} {r[2]:10.2f} " + " ".join(
            f"{r[3][c]:6.1f} {r[3][c] / base[3][c] - 1:+5.0%}" for c in CONTEXTS
        )

    shown = results if show_policies else [r for r in results if r[0].startswith("MTP")]
    for r in shown:
        print(row(r))
    if not show_policies:
        print("best per context:")
        for c in CONTEXTS:
            best = max(results, key=lambda r: r[3][c])
            within8 = max((r for r in results if "cap16" not in r[0] and "cap12" not in r[0]), key=lambda r: r[3][c])
            print(f"  {c}: {row(best)}")
            print(f"  {c} (<= 8 rows): {row(within8)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report")
    parser.add_argument("--policies", action="store_true", help="print every policy, not only fixed depths and the best")
    args = parser.parse_args()
    sequences = load(args.report)
    for category in sorted({c for _, c, _ in sequences}):
        describe([s for s in sequences if s[1] == category], category, args.policies)
    describe(sequences, "all", args.policies)


if __name__ == "__main__":
    main()
