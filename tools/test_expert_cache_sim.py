"""Checks the expert cache simulator on synthetic recordings whose answers are known."""

import random
import struct
import tempfile
import unittest
from pathlib import Path

import expert_cache_sim as sim


def write_trace(path, banks, experts, blocks, device=4, tier=0):
    """`blocks` is a list of (bank, [experts]); one prefill mark, then every block."""
    ints = [0x31445845, 1, device, tier, banks]
    for _ in range(banks):
        ints += [experts, 4096]
    ints += [2, 0, 0, len(blocks)]
    for bank, keys in blocks:
        ints += [1, bank, bank, 1, len(keys)]
        for k in keys:
            ints += [k, 1]
    Path(path).write_bytes(struct.pack(f"<{len(ints)}i", *ints))


class SimulatorTest(unittest.TestCase):
    def replay(self, blocks, device_policy, device, banks=1, experts=16):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/t.bin"
            write_trace(path, banks, experts, blocks, device=device)
            trace = sim.read_trace(path)
            return sim.replay(trace, device_policy, "lru", device, 0)["prefill"]

    def test_a_cyclic_sweep_larger_than_the_cache_defeats_lru_but_not_opt(self):
        blocks = [(0, [e]) for _ in range(20) for e in range(8)]
        lru = self.replay(blocks, "lru", 4)
        opt = self.replay(blocks, "opt", 4)
        self.assertEqual(0, lru.device_hits)
        self.assertGreater(opt.device_hits, 0.3 * opt.requests)

    def test_every_policy_hits_a_working_set_that_fits(self):
        blocks = [(0, [e]) for _ in range(20) for e in range(4)]
        for name in sim.POLICIES:
            with self.subTest(policy=name):
                t = self.replay(blocks, name, 6)
                self.assertGreaterEqual(t.device_hits, t.requests - 4 - (8 if name in ("tinylfu", "wtinylfu") else 0))

    def test_opt_is_never_worse_than_an_online_policy(self):
        rng = random.Random(3)
        hot = list(range(5))
        blocks = [(0, [rng.choice(hot) if rng.random() < 0.7 else rng.randrange(16)]) for _ in range(2000)]
        opt = self.replay(blocks, "opt", 6)
        for name in ("lru", "lfu", "tinylfu", "wtinylfu", "arc", "s3fifo", "sieve"):
            with self.subTest(policy=name):
                self.assertGreaterEqual(opt.device_hits, self.replay(blocks, name, 6).device_hits)

    def test_a_blocks_own_experts_are_never_evicted_while_it_is_served(self):
        # A block of 4 into 4 slots: every policy must keep all four (they are leased together).
        for name in ("lru", "lfu", "arc", "s3fifo", "sieve", "bank-lru"):
            with self.subTest(policy=name):
                t = self.replay([(0, [0, 1, 2, 3]), (0, [0, 1, 2, 3])], name, 4)
                self.assertEqual(4, t.device_hits, name)


if __name__ == "__main__":
    unittest.main()
