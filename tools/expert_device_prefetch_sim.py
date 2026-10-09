#!/usr/bin/env python3
"""Replays a demand recording with prefetch into the DEVICE cache (an upper bound: every prefetch lands in time).

When a decode block of layer L is planned, the recording's router prediction names layer L+1's likely experts; before
L+1's block asks, up to a budget of the first `k` candidates that the device does not hold are brought into the device
cache (evicting by its policy). Reports the device's decode miss rate and the host-to-device copies per token, with and
without the prefetch, so the misses the prefetch removes can be set against the copies it spends.

    python3 tools/expert_device_prefetch_sim.py TRACE [--k N ...] [--budget N ...] [--slots N]
"""

import argparse
from collections import defaultdict

import expert_cache_sim as sim
from expert_prefetch_sim import RouterPredictor


def run(trace, k, budget, device_slots, tier_slots, pin_current):
    nxt = sim.next_uses(trace)
    device = sim.POLICIES["s3fifo"](device_slots, trace)
    tier = sim.POLICIES["slfu-scan"](tier_slots, trace) if tier_slots > 0 else None
    if tier is not None:
        sim.preload(tier, trace, tier_slots)
    predictor = RouterPredictor(trace, k, budget) if budget else None
    stats = defaultdict(int)
    prefetched = set()
    phase = "prefill"
    position = 0
    for event in trace.events:
        if event[0] == "mark":
            phase = "prefill" if event[1] == 0 else "decode"
            continue
        _, bank, rows, keys = event
        pinned = set(keys)
        if predictor is not None:
            candidates, allowed = predictor.take(bank, rows)
            if rows == 1:
                for key in candidates:
                    if allowed <= 0:
                        break
                    if device.holds(key):
                        continue
                    allowed -= 1
                    # The copy comes from the tier when it holds the record, otherwise it is a disk read too.
                    stats["prefetch_copies"] += 1
                    if tier is not None and not tier.holds(key):
                        stats["prefetch_disk"] += 1
                        tier.prefetch(key, set())
                    device.prefetch(key, pinned if pin_current else set())
                    prefetched.add(key)
            predictor.observe(bank, rows, keys)
        for level in (device, tier):
            if level is not None and hasattr(level, "visit"):
                level.visit(bank)
            if level is not None and hasattr(level, "block"):
                level.block(bank, rows)
        for key in keys:
            position += 1
            hit = device.access(key, pinned)
            if phase == "decode":
                stats["requests"] += 1
                if hit:
                    stats["hits"] += 1
                    if key in prefetched:
                        stats["prefetch_used"] += 1
                else:
                    stats["misses"] += 1
                    if tier is not None:
                        if tier.access(key, pinned):
                            stats["tier_hits"] += 1
                        else:
                            stats["disk"] += 1
            elif not hit and tier is not None:
                tier.access(key, pinned)
            prefetched.discard(key)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace")
    parser.add_argument("--k", type=int, nargs="+", default=[10])
    parser.add_argument("--budget", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6, 10])
    parser.add_argument("--slots", type=int, help="device slots (default: the recording's)")
    parser.add_argument("--tier", type=int, help="tier slots (default: the recording's)")
    parser.add_argument("--pin-current", action="store_true", help="a prefetch may not evict the current block's keys")
    args = parser.parse_args()
    trace = sim.read_trace(args.trace)
    slots = args.slots or trace.device_slots
    tier = args.tier if args.tier is not None else trace.tier_slots
    tokens = sum(e[3] for e in trace.events if e[0] == "mark" and e[1] == 1)
    print(f"{args.trace}: device {slots} slots, tier {tier} slots, {tokens} decode tokens")
    for k in args.k:
        for budget in args.budget:
            s = run(trace, k, budget, slots, tier, args.pin_current)
            print(
                f"  top-{k:2d} budget {budget:2d}: device miss {100 * s['misses'] / s['requests']:5.1f}%"
                f" ({s['misses'] / tokens:6.1f}/token), tier hits {s['tier_hits'] / tokens:6.1f}/token,"
                f" demand disk {s['disk'] / tokens:5.1f}/token, prefetch copies {s['prefetch_copies'] / tokens:5.1f}/token"
                f" (used {s['prefetch_used'] / tokens:5.1f}), prefetch disk {s['prefetch_disk'] / tokens:5.1f}/token"
            )


if __name__ == "__main__":
    main()
