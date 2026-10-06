#!/usr/bin/env python3
"""Replays a recording of the expert cache's demand through cache policies.

The recording comes from `Qwen4DemandRecordingCudaIntegrationTest` (`EUHEDRAL_QWEN4_TRACE=FILE`). Each block of
a layer asks for its distinct experts at once; the replay asks a device level for each, then on a device miss a
RAM tier, then the disk. The experts of the block being served cannot be evicted while it is served (they are
leased). Both levels hold whole records and keep their own contents (a device eviction does not evict from the
tier), as the engine does.

    python3 tools/expert_cache_sim.py TRACE [--device N] [--tier N] [--device-policy P ...] [--tier-policy P ...]

Policies: lru, bank-lru (a quota per layer, lending unused slots, as the engine's tier), lfu (aged counts),
tinylfu (admission by aged counts over LRU, as the engine's tier with admission), wtinylfu (a 1% recency window
before an admission-filtered segmented LRU), arc, s3fifo, sieve, and two offline references: opt (Belady: evict
the record used furthest in the future) and static (the most requested records of the whole trace, fixed).
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------------------------- the recording


@dataclass
class Trace:
    device_slots: int
    tier_slots: int
    experts: list[int]
    record_bytes: list[int]
    # Each event: ("block", bank, rows, [experts]) or ("mark", kind, request, tokens).
    events: list = field(default_factory=list)

    def key(self, bank: int, expert: int) -> int:
        return bank * 4096 + expert


def read_trace(path: str) -> Trace:
    with open(path, "rb") as f:
        data = f.read()
    ints = struct.unpack(f"<{len(data) // 4}i", data[: len(data) // 4 * 4])
    if ints[0] != 0x31445845 or ints[1] != 1:
        sys.exit(f"{path} is not an expert demand recording")
    device, tier, banks = ints[2], ints[3], ints[4]
    experts = [ints[5 + 2 * b] for b in range(banks)]
    record = [ints[6 + 2 * b] for b in range(banks)]
    trace = Trace(device, tier, experts, record)
    at = 5 + 2 * banks
    while at < len(ints):
        tag = ints[at]
        if tag == 1:
            _layer, bank, rows, count = ints[at + 1 : at + 5]
            keys = [bank * 4096 + ints[at + 5 + 2 * i] for i in range(count)]
            trace.events.append(("block", bank, rows, keys))
            at += 5 + 2 * count
        elif tag == 2:
            trace.events.append(("mark",) + tuple(ints[at + 1 : at + 4]))
            at += 4
        else:
            sys.exit(f"bad record tag {tag} at {at}")
    return trace


# ---------------------------------------------------------------------------------------------- policies
#
# A policy holds at most `capacity` keys. `access(key, pinned)` returns True on a hit; on a miss the policy may
# admit the key, evicting keys not in `pinned`; a policy that admits nothing (all pinned, or admission refused)
# simply does not hold the key.


class Policy:
    name = "?"

    def __init__(self, capacity: int, trace: Trace, future=None):
        self.capacity = capacity
        self.trace = trace

    def access(self, key: int, pinned: set) -> bool:
        raise NotImplementedError


class LRU(Policy):
    name = "lru"

    def __init__(self, capacity, trace, future=None):
        super().__init__(capacity, trace)
        self.order: OrderedDict[int, None] = OrderedDict()

    def access(self, key, pinned):
        if key in self.order:
            self.order.move_to_end(key)
            return True
        if len(self.order) >= self.capacity and not self._evict(pinned):
            return False
        self.order[key] = None
        return False

    def _evict(self, pinned) -> bool:
        for victim in self.order:
            if victim not in pinned:
                del self.order[victim]
                return True
        return False


class BankLRU(Policy):
    """A quota per layer in proportion to its experts; a layer under quota takes from the layer furthest over its
    own, a layer at quota replaces its own least recent; free slots are lent to anyone (the engine's tier)."""

    name = "bank-lru"

    def __init__(self, capacity, trace, future=None):
        super().__init__(capacity, trace)
        total = sum(trace.experts)
        self.quota = [capacity * e // total for e in trace.experts]
        self.lists = [OrderedDict() for _ in trace.experts]
        self.used = [0] * len(trace.experts)
        self.size = 0

    def access(self, key, pinned):
        bank = key // 4096
        lst = self.lists[bank]
        if key in lst:
            lst.move_to_end(key)
            return True
        if self.size >= self.capacity:
            victim_bank = None
            if self.used[bank] > 0 and self.used[bank] >= self.quota[bank] and self._unpinned(bank, pinned):
                victim_bank = bank
            else:
                best = None
                for b in range(len(self.lists)):
                    over = self.used[b] - self.quota[b]
                    if (best is None or over > best) and self._unpinned(b, pinned) is not None:
                        best, victim_bank = over, b
            if victim_bank is None:
                return False
            victim = self._unpinned(victim_bank, pinned)
            del self.lists[victim_bank][victim]
            self.used[victim_bank] -= 1
            self.size -= 1
        lst[key] = None
        self.used[bank] += 1
        self.size += 1
        return False

    def _unpinned(self, bank, pinned):
        for k in self.lists[bank]:
            if k not in pinned:
                return k
        return None


class Counts:
    """Request counts halved every `period` requests (the engine's admission counts)."""

    def __init__(self, period):
        self.count = defaultdict(int)
        self.period = period
        self.seen = 0

    def add(self, key) -> int:
        before = self.count[key]
        self.count[key] = before + 1
        self.seen += 1
        if self.seen >= self.period:
            self.seen = 0
            for k in list(self.count):
                self.count[k] >>= 1
                if self.count[k] == 0:
                    del self.count[k]
        return before


class TinyLFU(Policy):
    """LRU whose victim is replaced only by a record asked for more often before (the engine's tier admission)."""

    name = "tinylfu"

    def __init__(self, capacity, trace, future=None):
        super().__init__(capacity, trace)
        self.order: OrderedDict[int, None] = OrderedDict()
        self.counts = Counts(10 * capacity)

    def access(self, key, pinned):
        asked = self.counts.add(key)
        if key in self.order:
            self.order.move_to_end(key)
            return True
        if len(self.order) >= self.capacity:
            victim = next((v for v in self.order if v not in pinned), None)
            if victim is None or self.counts.count.get(victim, 0) >= asked:
                return False
            del self.order[victim]
        self.order[key] = None
        return False


class LFU(Policy):
    """Evict the record asked for least often (aged counts), the least recent among equals."""

    name = "lfu"

    def __init__(self, capacity, trace, future=None):
        super().__init__(capacity, trace)
        self.counts = Counts(10 * capacity)
        self.held: dict[int, int] = {}  # key -> last use
        self.clock = 0

    def access(self, key, pinned):
        self.counts.add(key)
        self.clock += 1
        if key in self.held:
            self.held[key] = self.clock
            return True
        if len(self.held) >= self.capacity:
            victim = min(
                (k for k in self.held if k not in pinned),
                key=lambda k: (self.counts.count.get(k, 0), self.held[k]),
                default=None,
            )
            if victim is None:
                return False
            del self.held[victim]
        self.held[key] = self.clock
        return False


class WTinyLFU(Policy):
    """Caffeine's design: a 1% LRU window; its victims enter a segmented LRU (20% probation, 80% protected) only if
    asked for more often than the probation victim."""

    name = "wtinylfu"

    def __init__(self, capacity, trace, future=None):
        super().__init__(capacity, trace)
        self.window_cap = max(1, capacity // 100)
        main = capacity - self.window_cap
        self.protected_cap = int(main * 0.8)
        self.window: OrderedDict = OrderedDict()
        self.probation: OrderedDict = OrderedDict()
        self.protected: OrderedDict = OrderedDict()
        self.counts = Counts(10 * capacity)

    def _size(self):
        return len(self.window) + len(self.probation) + len(self.protected)

    def access(self, key, pinned):
        self.counts.add(key)
        if key in self.window:
            self.window.move_to_end(key)
            return True
        if key in self.probation:
            del self.probation[key]
            self.protected[key] = None
            while len(self.protected) > self.protected_cap:
                demoted = next((k for k in self.protected if k not in pinned), None)
                if demoted is None:
                    break
                del self.protected[demoted]
                self.probation[demoted] = None
            return True
        if key in self.protected:
            self.protected.move_to_end(key)
            return True
        self.window[key] = None
        while len(self.window) > self.window_cap:
            candidate = next((k for k in self.window if k not in pinned), None)
            if candidate is None:
                break
            del self.window[candidate]
            if self._size() < self.capacity:
                self.probation[candidate] = None
                continue
            victim = next((k for k in self.probation if k not in pinned), None)
            if victim is None:
                victim = next((k for k in self.protected if k not in pinned), None)
                if victim is None:
                    continue
                if self.counts.count.get(candidate, 0) > self.counts.count.get(victim, 0):
                    del self.protected[victim]
                    self.probation[candidate] = None
                continue
            if self.counts.count.get(candidate, 0) > self.counts.count.get(victim, 0):
                del self.probation[victim]
                self.probation[candidate] = None
        while self._size() > self.capacity:
            k = next((k for k in self.window if k not in pinned and k != key), None)
            if k is None:
                break
            del self.window[k]
        return False


class ARC(Policy):
    name = "arc"

    def __init__(self, capacity, trace, future=None):
        super().__init__(capacity, trace)
        self.t1, self.t2, self.b1, self.b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        self.p = 0

    def _replace(self, key, pinned):
        def pop(lst):
            for k in lst:
                if k not in pinned:
                    del lst[k]
                    return k
            return None

        if self.t1 and (len(self.t1) > self.p or (key in self.b2 and len(self.t1) == self.p)):
            k = pop(self.t1)
            if k is not None:
                self.b1[k] = None
                return True
        k = pop(self.t2)
        if k is not None:
            self.b2[k] = None
            return True
        k = pop(self.t1)
        if k is not None:
            self.b1[k] = None
            return True
        return False

    def access(self, key, pinned):
        c = self.capacity
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = None
            return True
        if key in self.t2:
            self.t2.move_to_end(key)
            return True
        if key in self.b1:
            self.p = min(c, self.p + max(1, len(self.b2) // max(1, len(self.b1))))
            if not self._replace(key, pinned):
                return False
            del self.b1[key]
            self.t2[key] = None
            return False
        if key in self.b2:
            self.p = max(0, self.p - max(1, len(self.b1) // max(1, len(self.b2))))
            if not self._replace(key, pinned):
                return False
            del self.b2[key]
            self.t2[key] = None
            return False
        if len(self.t1) + len(self.b1) >= c:
            if len(self.t1) < c:
                self.b1.popitem(last=False)
                if not self._replace(key, pinned):
                    return False
            else:
                k = next((k for k in self.t1 if k not in pinned), None)
                if k is None:
                    return False
                del self.t1[k]
        elif len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= c:
            if len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= 2 * c and self.b2:
                self.b2.popitem(last=False)
            if len(self.t1) + len(self.t2) >= c and not self._replace(key, pinned):
                return False
        self.t1[key] = None
        return False


class S3FIFO(Policy):
    """A 10% small FIFO; records asked for again there move to the main FIFO; a ghost FIFO remembers recent
    small-queue victims, whose return goes straight to main. Main is a FIFO with reinsertion (a frequency of up to
    3)."""

    name = "s3fifo"

    def __init__(self, capacity, trace, future=None):
        super().__init__(capacity, trace)
        self.small_cap = max(1, capacity // 10)
        self.small: deque = deque()
        self.main: deque = deque()
        self.ghost: OrderedDict = OrderedDict()
        self.freq: dict[int, int] = {}
        self.where: dict[int, str] = {}

    def _evict(self, pinned) -> bool:
        if len(self.small) >= self.small_cap:
            for _ in range(len(self.small)):
                k = self.small.popleft()
                if k in pinned:
                    self.small.append(k)
                    continue
                if self.freq[k] > 0:
                    self.main.append(k)
                    self.where[k] = "m"
                    self.freq[k] = 0
                    if len(self.main) + len(self.small) <= self.capacity:
                        return True
                    continue
                del self.where[k], self.freq[k]
                self.ghost[k] = None
                if len(self.ghost) > self.capacity:
                    self.ghost.popitem(last=False)
                return True
        for _ in range(4 * len(self.main) + 1):
            if not self.main:
                break
            k = self.main.popleft()
            if k in pinned:
                self.main.append(k)
                continue
            if self.freq[k] > 0:
                self.freq[k] -= 1
                self.main.append(k)
                continue
            del self.where[k], self.freq[k]
            return True
        return False

    def access(self, key, pinned):
        if key in self.where:
            self.freq[key] = min(3, self.freq[key] + 1)
            return True
        while len(self.small) + len(self.main) >= self.capacity:
            if not self._evict(pinned):
                return False
        if key in self.ghost:
            del self.ghost[key]
            self.main.append(key)
            self.where[key] = "m"
        else:
            self.small.append(key)
            self.where[key] = "s"
        self.freq[key] = 0
        return False


class SIEVE(Policy):
    name = "sieve"

    def __init__(self, capacity, trace, future=None):
        super().__init__(capacity, trace)
        self.queue: list[int] = []  # oldest first
        self.visited: dict[int, bool] = {}
        self.hand = None  # index into queue, moving from old to new

    def access(self, key, pinned):
        if key in self.visited:
            self.visited[key] = True
            return True
        if len(self.queue) >= self.capacity:
            if not self._evict(pinned):
                return False
        self.queue.append(key)
        self.visited[key] = False
        return False

    def _evict(self, pinned) -> bool:
        n = len(self.queue)
        i = 0 if self.hand is None or self.hand >= n else self.hand
        for _ in range(2 * n + 1):
            k = self.queue[i]
            if k in pinned or self.visited[k]:
                self.visited[k] = False if k not in pinned else self.visited[k]
                i = (i + 1) % n
                continue
            del self.queue[i]
            del self.visited[k]
            self.hand = i if i < len(self.queue) else 0
            return True
        return False


class OPT(Policy):
    """Belady: evict the held record whose next request is furthest away (offline, needs the future)."""

    name = "opt"

    def __init__(self, capacity, trace, future=None):
        super().__init__(capacity, trace)
        self.future = future  # dict: position -> next position of the same key, filled per access by the driver
        self.held: dict[int, int] = {}  # key -> next use
        self.position = 0

    def access(self, key, pinned, next_use=None):
        if key in self.held:
            self.held[key] = next_use
            return True
        if next_use == float("inf"):
            return False  # never asked again: do not admit
        if len(self.held) >= self.capacity:
            victim = max((k for k in self.held if k not in pinned), key=lambda k: self.held[k], default=None)
            if victim is None or self.held[victim] <= next_use:
                return False
            del self.held[victim]
        self.held[key] = next_use
        return False


class Static(Policy):
    """The `capacity` most requested records of the whole trace, fixed (offline)."""

    name = "static"

    def __init__(self, capacity, trace, future=None):
        super().__init__(capacity, trace)
        counts = defaultdict(int)
        for event in trace.events:
            if event[0] == "block":
                for k in event[3]:
                    counts[k] += 1
        self.held = set(sorted(counts, key=counts.get, reverse=True)[:capacity])

    def access(self, key, pinned):
        return key in self.held


class ScanLRU(Policy):
    """LRU that a prefill cannot flush: records a multi-row block (a prefill chunk) brings in enter at the cold end,
    and its hits do not promote, so a prefill's sweep of nearly every expert of a layer replaces only what earlier
    sweeps brought, while decode's records keep their places."""

    name = "scan-lru"

    def __init__(self, capacity, trace, future=None):
        super().__init__(capacity, trace)
        self.order: OrderedDict[int, None] = OrderedDict()
        self.rows = 1

    def block(self, bank, rows):
        self.rows = rows

    def access(self, key, pinned):
        scan = self.rows > 1
        if key in self.order:
            if not scan:
                self.order.move_to_end(key)
            return True
        if len(self.order) >= self.capacity:
            victim = next((v for v in self.order if v not in pinned), None)
            if victim is None:
                return False
            del self.order[victim]
        self.order[key] = None
        if scan:
            self.order.move_to_end(key, last=False)
        return False


class LayerAge(Policy):
    """Recency measured in visits to the record's own layer, not in time: the model visits its layers in a cycle, so
    a global recency order calls the next layer's records the oldest and evicts them just before they are used. The
    victim is the record whose layer has been visited most often since it was used; among equals, the record of the
    layer visited most recently, whose next visit is furthest away."""

    name = "layer-age"

    def __init__(self, capacity, trace, future=None):
        super().__init__(capacity, trace)
        self.lists = [OrderedDict() for _ in trace.experts]  # per bank, key -> visit of the bank when last used
        self.visits = [0] * len(trace.experts)
        self.visited_at = [0] * len(trace.experts)  # global clock of each bank's latest visit
        self.clock = 0
        self.size = 0
        self.current = None

    def visit(self, bank):
        self.clock += 1
        self.visits[bank] += 1
        self.visited_at[bank] = self.clock

    def access(self, key, pinned):
        bank = key // 4096
        lst = self.lists[bank]
        if key in lst:
            lst[key] = self.visits[bank]
            lst.move_to_end(key)
            return True
        if self.size >= self.capacity:
            best = None
            for b, other in enumerate(self.lists):
                victim = next((k for k in other if k not in pinned), None)
                if victim is None:
                    continue
                rank = (self.visits[b] - other[victim], self.visited_at[b])
                if best is None or rank > best[0]:
                    best = (rank, b, victim)
            if best is None:
                return False
            del self.lists[best[1]][best[2]]
            self.size -= 1
        lst[key] = self.visits[bank]
        self.size += 1
        return False


POLICIES = {
    p.name: p for p in (LRU, ScanLRU, BankLRU, LFU, TinyLFU, WTinyLFU, ARC, S3FIFO, SIEVE, OPT, Static, LayerAge)
}


# ---------------------------------------------------------------------------------------------- replay


def next_uses(trace: Trace, select=None) -> list[float]:
    """For each request in order (optionally only those `select` keeps), the position of the same key's next
    request among the same selection."""
    keys = []
    for event in trace.events:
        if event[0] == "block":
            keys.extend(event[3])
    nxt = [float("inf")] * len(keys)
    last: dict[int, int] = {}
    for i in range(len(keys) - 1, -1, -1):
        k = keys[i]
        nxt[i] = last.get(k, float("inf"))
        last[k] = i
    return nxt


@dataclass
class Tally:
    requests: int = 0
    device_hits: int = 0
    tier_hits: int = 0
    disk: int = 0
    disk_bytes: int = 0


def preload(tier: Policy, trace: Trace, slots: int):
    """The engine's startup fill: each bank's share of the slots holds its lowest experts. Each is offered to the
    policy as a request from an empty block, and policies that count requests then forget it (a preloaded record
    is a placeholder until it is asked for)."""
    total = sum(trace.experts)
    for bank, experts in enumerate(trace.experts):
        for expert in range(slots * experts // total):
            tier.access(bank * 4096 + expert, set())
    counts = getattr(tier, "counts", None)
    if counts is not None:
        counts.count.clear()
        counts.seen = 0


def replay(
    trace: Trace, device_policy: str, tier_policy: str, device_slots: int, tier_slots: int, fill: bool = False
):
    nxt = next_uses(trace)
    device = POLICIES[device_policy](device_slots, trace)
    tier = POLICIES[tier_policy](tier_slots, trace) if tier_slots > 0 else None
    if fill and tier is not None and not isinstance(tier, (OPT, Static)):
        preload(tier, trace, tier_slots)
    # The tier sees only device misses; Belady at the tier needs the next use among those, which depends on the
    # device's choices: approximate it with the next use in the whole stream.
    tallies = defaultdict(Tally)
    phase = "prefill"
    position = 0
    for event in trace.events:
        if event[0] == "mark":
            phase = "prefill" if event[1] == 0 else "decode"
            continue
        _, bank, rows, keys = event
        pinned = set(keys)
        for level in (device, tier):
            if level is not None and hasattr(level, "visit"):
                level.visit(bank)
            if level is not None and hasattr(level, "block"):
                level.block(bank, rows)
        tally = tallies[phase]
        for key in keys:
            tally.requests += 1
            n = nxt[position]
            position += 1
            hit = device.access(key, pinned, n) if isinstance(device, OPT) else device.access(key, pinned)
            if hit:
                tally.device_hits += 1
                continue
            if tier is not None:
                thit = tier.access(key, pinned, n) if isinstance(tier, OPT) else tier.access(key, pinned)
                if thit:
                    tally.tier_hits += 1
                    continue
            tally.disk += 1
            tally.disk_bytes += trace.record_bytes[bank]
    return tallies


def report(trace, device_policy, tier_policy, device_slots, tier_slots, tallies, tokens):
    cells = []
    for phase in ("prefill", "decode"):
        t = tallies.get(phase)
        if not t or not t.requests:
            continue
        cells.append(
            f"{phase}: device miss {100 * (1 - t.device_hits / t.requests):5.1f}%"
            f"  tier hit {100 * t.tier_hits / max(1, t.requests - t.device_hits):5.1f}%"
            f"  disk {100 * t.disk / t.requests:5.1f}% ({t.disk_bytes / 1e6 / max(1, tokens[phase]):6.1f} MB/token)"
        )
    print(f"device {device_policy:9s} {device_slots:5d} | tier {tier_policy:9s} {tier_slots:5d} || " + " | ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace")
    parser.add_argument("--device", type=int, help="device slots (default: the recording's)")
    parser.add_argument("--tier", type=int, help="tier slots (default: the recording's)")
    parser.add_argument("--device-policy", nargs="+", default=["lru"])
    parser.add_argument("--tier-policy", nargs="+", default=["bank-lru", "tinylfu"])
    parser.add_argument("--fill", action="store_true", help="start the tier with the engine's startup fill")
    args = parser.parse_args()
    trace = read_trace(args.trace)
    device_slots = args.device if args.device is not None else trace.device_slots
    tier_slots = args.tier if args.tier is not None else trace.tier_slots
    tokens = defaultdict(int)
    requests = defaultdict(int)
    phase = "prefill"
    for event in trace.events:
        if event[0] == "mark":
            phase = "prefill" if event[1] == 0 else "decode"
            tokens[phase] += event[3]
        else:
            requests[phase] += len(event[3])
    print(
        f"{args.trace}: {sum(requests.values())} requests ({requests['prefill']} prefill, {requests['decode']} decode),"
        f" {tokens['prefill']} prefill and {tokens['decode']} decode tokens; {sum(trace.experts)} records"
    )
    for dp in args.device_policy:
        for tp in args.tier_policy:
            tallies = replay(trace, dp, tp, device_slots, tier_slots, args.fill)
            report(trace, dp, tp, device_slots, tier_slots, tallies, tokens)


if __name__ == "__main__":
    main()
