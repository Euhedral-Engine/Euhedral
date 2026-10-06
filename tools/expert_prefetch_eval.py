#!/usr/bin/env python3
"""How well the experts a layer will ask for can be predicted when the previous layer's routing is known.

Replays a demand recording (see expert_cache_sim.py). For every block of layer L+1 that follows a block of layer L in
the same step, each predictor names up to K experts of layer L+1 from what is known after layer L's routing; the report
gives, per phase, the share of layer L+1's experts the prediction named (recall) and the share of named experts that
were used (precision).

Predictors:
  previous   the experts layer L+1 used in the previous step (decode) or chunk
  popular    layer L+1's most requested experts so far
  cooccur    experts of layer L+1 that followed layer L's experts most often so far (a table learned while replaying)
  mix        cooccur, topped up by previous and popular
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict

import expert_cache_sim as sim


def steps(trace):
    """Yields (phase, [(layer bank, rows, set of expert ids)] for one step through the layers)."""
    phase = "prefill"
    current = []
    last_bank = None
    for event in trace.events:
        if event[0] == "mark":
            if current:
                yield phase, current
            current, last_bank = [], None
            phase = "prefill" if event[1] == 0 else "decode"
            continue
        _, bank, rows, keys = event
        if last_bank is not None and bank <= last_bank and current:
            yield phase, current
            current = []
        current.append((bank, rows, {k % 4096 for k in keys}))
        last_bank = bank
    if current:
        yield phase, current


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", nargs="+")
    parser.add_argument("-k", type=int, nargs="+", default=[10, 20, 40])
    args = parser.parse_args()
    for path in args.trace:
        trace = sim.read_trace(path)
        banks = len(trace.experts)
        for k in args.k:
            previous: dict[int, set] = {}
            popular = [Counter() for _ in range(banks)]
            cooccur = [defaultdict(Counter) for _ in range(banks)]  # cooccur[b][e][e'] : e at b-1 then e' at b
            score = defaultdict(lambda: [0, 0, 0])  # (phase, predictor) -> [named and used, named, used]
            for phase, step in steps(trace):
                for (bank, rows, used), nxt in zip(step, step[1:]):
                    nbank, _, nused = nxt
                    if nbank != bank + 1:
                        continue
                    table = cooccur[nbank]
                    votes = Counter()
                    for e in used:
                        votes.update(table[e])
                    preds = {
                        "previous": list(previous.get(nbank, ()))[:k],
                        "popular": [e for e, _ in popular[nbank].most_common(k)],
                        "cooccur": [e for e, _ in votes.most_common(k)],
                    }
                    mix = [e for e, _ in votes.most_common(k)]
                    for e in list(previous.get(nbank, ())) + [e for e, _ in popular[nbank].most_common(k)]:
                        if len(mix) >= k:
                            break
                        if e not in mix:
                            mix.append(e)
                    preds["mix"] = mix
                    for name, named in preds.items():
                        s = score[(phase, name)]
                        s[0] += len(set(named) & nused)
                        s[1] += len(named)
                        s[2] += len(nused)
                    for e in used:
                        table[e].update(nused)
                # Learn the step's first layer too, and remember each layer's use for the next step.
                for bank, rows, used in step:
                    popular[bank].update(used)
                    previous[bank] = used
            print(f"{path} k={k}")
            for (phase, name), (hit, named, used) in sorted(score.items()):
                print(
                    f"  {phase:7s} {name:8s} recall {100 * hit / max(1, used):5.1f}%"
                    f"  precision {100 * hit / max(1, named):5.1f}%"
                )


if __name__ == "__main__":
    sys.exit(main())
