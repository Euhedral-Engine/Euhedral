#!/usr/bin/env python3
"""Replays a demand recording with prefetch into the RAM tier (an upper bound: every prefetch lands in time).

When a block of layer L is planned, a predictor names layer L+1's likely experts; before L+1's block asks, up to a
budget of them that neither the device nor the tier holds are read into the tier. Reports the disk reads the demand
still makes and the reads the prefetch spent, per token.

    python3 tools/expert_prefetch_sim.py TRACE [--decode-budget N ...] [--prefill-budget N ...]
"""

import argparse
from collections import Counter, defaultdict

import expert_cache_sim as sim


class Predictor:
    """Co-occurrence of a layer's experts with the next layer's (learned as the replay goes), then the next layer's
    popularity; decode steps get `decode` reads, prefill chunks `prefill`."""

    def __init__(self, banks, decode, prefill):
        self.cooccur = [defaultdict(Counter) for _ in range(banks)]
        self.popular = [Counter() for _ in range(banks)]
        self.pending = {}
        self.last = None
        self.decode, self.prefill = decode, prefill

    def observe(self, bank, rows, keys):
        experts = [k % 4096 for k in keys]
        if self.last is not None and self.last[0] == bank - 1:
            for e in self.last[1]:
                self.cooccur[bank][e].update(experts)
        self.popular[bank].update(experts)
        self.last = (bank, experts)
        nxt = bank + 1
        if nxt >= len(self.popular):
            return
        votes = Counter()
        for e in experts:
            votes.update(self.cooccur[nxt][e])
        ordered = [e for e, _ in votes.most_common()] + [e for e, _ in self.popular[nxt].most_common()]
        seen, out = set(), []
        for e in ordered:
            if e not in seen:
                seen.add(e)
                out.append(nxt * 4096 + e)
        self.pending[nxt] = (out, self.decode if rows == 1 else self.prefill)

    def take(self, bank, rows):
        return self.pending.pop(bank, ([], 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace")
    parser.add_argument("--device-policy", default="s3fifo")
    parser.add_argument("--tier-policy", default="slfu-scan")
    parser.add_argument("--tier", type=int)
    parser.add_argument("--decode-budget", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--prefill-budget", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    trace = sim.read_trace(args.trace)
    tier = args.tier or trace.tier_slots
    tokens = defaultdict(int)
    phase = "prefill"
    for e in trace.events:
        if e[0] == "mark":
            phase = "prefill" if e[1] == 0 else "decode"
            tokens[phase] += e[3]
    print(f"{args.trace}: device {args.device_policy} {trace.device_slots}, tier {args.tier_policy} {tier}")
    for pb in args.prefill_budget:
        for db in args.decode_budget:
            predictor = Predictor(len(trace.experts), db, pb) if (db or pb) else None
            t = sim.replay(trace, args.device_policy, args.tier_policy, trace.device_slots, tier, True, predictor)
            cells = []
            for ph in ("prefill", "decode"):
                x = t[ph]
                cells.append(
                    f"{ph}: demand disk {x.disk_bytes / 1e6 / tokens[ph]:6.1f} MB/token,"
                    f" prefetch {x.prefetch_bytes / 1e6 / tokens[ph]:6.1f} MB/token"
                )
            print(f"  budget prefill {pb:4d} decode {db:2d} per layer || " + " | ".join(cells))


if __name__ == "__main__":
    main()
