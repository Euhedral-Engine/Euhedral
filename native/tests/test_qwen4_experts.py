"""Flash-Next routed-expert kernels (native/src/qwen4/experts.cuh) against a float64 reference of the exactly expanded
NVFP4 weights, and their row-exactness: a (token, expert) pair gives the same bits whatever work items it is in.

Kernel times are in bench_qwen4_experts.py.
"""

import contextlib
import ctypes as C
from pathlib import Path
import struct
import sys
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from gpu_harness import Gpu, NVRTC, SKIP_REASON

ROOT = Path(__file__).resolve().parents[2]
UNAVAILABLE = (f"CUDA probes unavailable: {SKIP_REASON}" if NVRTC is None
               else "NumPy unavailable" if np is None else None)
converter = None
if np is not None:
    sys.path.insert(0, str(ROOT / "tools"))
    from euhedral_artifacts import nvfp4 as converter

HIDDEN, INTER = 2560, 640
GATE_UP_OFFSET, DOWN_OFFSET, RECORD = 0, 1843456, 2768896
NAMES = ("euhedral_q4_expert_gate_up_swiglu_bf16", "euhedral_q4_expert_down_bf16", "euhedral_q4_expert_combine_bf16")


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint32) << 16
    return rounded.astype(np.uint32).view(np.float32)


def bf16_bits(values):
    return (bf16(values).view(np.uint32) >> 16).astype("<u2")


def from_bits(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def tensor(rng, rows, k):
    """(NVFP4 tensor bytes, dequantized float32 weights [rows, k])."""
    weights = (rng.standard_normal((rows, k)) * rng.uniform(0.01, 1.0, (rows, 1))).astype(np.float32)
    global_scale = converter.nvfp4_global_scale(float(np.abs(weights).max()))
    packed, scales = converter.quantize_nvfp4_rows(weights, global_scale)
    scale_offset, global_offset, size = converter.nvfp4_offsets((rows, k))
    out = bytearray(size)
    out[:packed.nbytes] = packed.tobytes()
    out[scale_offset:scale_offset + scales.nbytes] = scales.tobytes()
    out[global_offset:global_offset + 4] = np.float32(global_scale).astype("<f4").tobytes()
    return bytes(out), converter.dequantize_nvfp4_rows(packed, scales, global_scale)


def record(rng):
    gate_up, dense_gate_up = tensor(rng, 2 * INTER, HIDDEN)
    down, dense_down = tensor(rng, HIDDEN, INTER)
    data = bytearray(RECORD)
    data[GATE_UP_OFFSET:GATE_UP_OFFSET + len(gate_up)] = gate_up
    data[DOWN_OFFSET:DOWN_OFFSET + len(down)] = down
    return bytes(data), dense_gate_up, dense_down


def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def reference_act(x_bits, dense_gate_up):
    """act of one expert for token rows x_bits [n, H] (BF16 bits)."""
    x = from_bits(x_bits).astype(np.float64)
    gate_up = bf16((x @ dense_gate_up.astype(np.float64).T).astype(np.float32))
    gate, up = gate_up[:, :INTER], gate_up[:, INTER:]
    return bf16(bf16(silu(gate)) * up)


def reference_weighted(act_bits, dense_down, weight_bits):
    """weighted of one expert for act rows (BF16 bits) and routing weights [n] (BF16 bits)."""
    act = from_bits(act_bits).astype(np.float64)
    y = bf16((act @ dense_down.astype(np.float64).T).astype(np.float32))
    return bf16(y * from_bits(weight_bits)[:, None])


@unittest.skipIf(UNAVAILABLE is not None, UNAVAILABLE or "")
class ExpertKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "qwen4/kernels.cu"\n', architecture="sm_120")
        cls.rng = np.random.default_rng(0xE0)
        cls.records = [record(cls.rng) for _ in range(3)]

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    # -- launches -------------------------------------------------------------------------------------------

    def upload_slots(self, stack, expert_records, copies=1):
        slab = self.gpu.upload(b"".join(r[0] for r in expert_records) * copies)
        stack.callback(self.gpu.free, slab)
        return slab

    def run_wave(self, stack, slab, slot_index, items, pairs, x_bits, tokens, repeat=1):
        """items: [(slot, begin, count)], pairs: [(token, weight_bits)]. Returns (act, weighted, device buffers)."""
        gpu = self.gpu
        slots = np.array([slab + RECORD * s for s in slot_index], dtype="<u8")
        d_slots = gpu.upload(slots.tobytes()); stack.callback(gpu.free, d_slots)
        d_items = gpu.upload(np.array([[s, b, c, 0] for s, b, c in items], dtype="<i4").tobytes())
        stack.callback(gpu.free, d_items)
        d_pairs = gpu.upload(np.array(pairs, dtype="<i4").tobytes()); stack.callback(gpu.free, d_pairs)
        d_x = gpu.upload(x_bits.astype("<u2").tobytes()); stack.callback(gpu.free, d_x)
        n = len(pairs)
        d_act = gpu.zeros(n * INTER * 2, 0x5A); stack.callback(gpu.free, d_act)
        d_weighted = gpu.zeros(n * HIDDEN * 2, 0x5A); stack.callback(gpu.free, d_weighted)
        for _ in range(repeat):
            gpu.launch(NAMES[0], (INTER // 32, len(items)),
                       [C.c_uint64(d_slots), C.c_uint64(d_items), C.c_uint64(d_pairs), C.c_uint64(d_x),
                        C.c_uint64(d_act), C.c_uint(GATE_UP_OFFSET), C.c_uint(HIDDEN), C.c_uint(INTER)], block=128)
            gpu.launch(NAMES[1], (HIDDEN // 128, len(items)),
                       [C.c_uint64(d_slots), C.c_uint64(d_items), C.c_uint64(d_pairs), C.c_uint64(d_act),
                        C.c_uint64(d_weighted), C.c_uint(DOWN_OFFSET), C.c_uint(INTER), C.c_uint(HIDDEN)], block=128)
        act = np.frombuffer(gpu.download(d_act, n * INTER * 2), "<u2").reshape(n, INTER)
        weighted = np.frombuffer(gpu.download(d_weighted, n * HIDDEN * 2), "<u2").reshape(n, HIDDEN)
        return act, weighted

    def assert_close(self, actual_bits, expected, what, ulps=1):
        actual, expected = from_bits(actual_bits), expected
        scale = np.abs(expected).max()
        error = np.abs(actual - expected)
        bound = np.abs(expected) * 2.0**-7 * ulps + scale * 1e-6
        ok = (error <= bound).all()
        self.assertTrue(ok, f"{what}: max error {error.max()} (scale {scale})")
        different = (bf16_bits(expected) != np.asarray(actual_bits)).mean()
        self.assertLess(different, 0.02, f"{what}: {different:.4f} of the elements differ")

    @staticmethod
    def groups(counts, size=16):
        """Items of up to `size` pairs for experts holding `counts` pairs each, pairs numbered consecutively."""
        items, begin = [], 0
        for slot, count in enumerate(counts):
            for g in range(0, count, size):
                items.append((slot, begin + g, min(size, count - g)))
            begin += count
        return items

    # -- tests ----------------------------------------------------------------------------------------------

    def test_matches_the_reference(self):
        rng = self.rng
        counts = [1, 5, 8, 9, 17, 3, 16, 33]
        slot_index = [0, 1, 2, 1, 0, 2, 1, 0]
        tokens = 40
        x_bits = bf16_bits(rng.standard_normal((tokens, HIDDEN)).astype(np.float32))
        pairs = []
        for count in counts:
            for token in sorted(rng.choice(tokens, size=count, replace=False)):
                pairs.append((int(token), int(bf16_bits(rng.uniform(0.01, 0.5))))) 
        with contextlib.ExitStack() as stack:
            slab = self.upload_slots(stack, self.records)
            act, weighted = self.run_wave(stack, slab, slot_index, self.groups(counts), pairs, x_bits, tokens)
        begin = 0
        for slot, count in zip(slot_index, counts):
            rows = pairs[begin:begin + count]
            _, dense_gate_up, dense_down = self.records[slot]
            expected_act = reference_act(x_bits[[t for t, _ in rows]], dense_gate_up)
            # the down projection is checked on the kernel's own act, so a rounding flip upstream does not compound
            expected_weighted = reference_weighted(
                act[begin:begin + count], dense_down, np.array([w for _, w in rows], dtype=np.uint16))
            self.assert_close(act[begin:begin + count], expected_act, f"act of pairs {begin}..")
            self.assert_close(weighted[begin:begin + count], expected_weighted, f"weighted of pairs {begin}..", ulps=2)  # a flipped y rounding and the weight product
            begin += count

    def test_pairs_are_bitwise_independent_of_their_items(self):
        rng = self.rng
        tokens = 40
        x_bits = bf16_bits(rng.standard_normal((tokens, HIDDEN)).astype(np.float32))
        count = 37
        pairs = [(int(t), int(bf16_bits(rng.uniform(0.01, 0.5)))) for t in rng.choice(tokens, count, replace=False)]
        results = {}
        with contextlib.ExitStack() as stack:
            slab = self.upload_slots(stack, self.records)
            for size in (16, 8, 5, 3, 1):
                items = self.groups([count], size)
                results[size] = self.run_wave(stack, slab, [1], items, pairs, x_bits, tokens)
            # the same pairs in reverse order, one item of 7 and the rest 16
            reversed_pairs = pairs[::-1]
            items = [(0, 0, 7)] + [(0, b, min(16, count - b)) for b in range(7, count, 16)]
            reverse = self.run_wave(stack, slab, [1], items, reversed_pairs, x_bits, tokens)
        for size in (8, 5, 3, 1):
            for i in range(2):
                self.assertTrue(np.array_equal(results[16][i], results[size][i]), f"size {size}, output {i}")
        for i in range(2):
            self.assertTrue(np.array_equal(results[16][i], reverse[i][::-1]), f"reversed, output {i}")

    def test_combine_adds_in_the_listed_order(self):
        rng = self.rng
        tokens, pairs = 6, 20
        weighted = bf16_bits(rng.standard_normal((pairs, HIDDEN)).astype(np.float32) * 4)
        # token t lists some pairs; the order of the list is the order of the additions
        lists = [[3, 0, 7], [], [19, 1], [2, 4, 5, 6, 8, 9, 10], [11], [12, 13, 14, 15, 16, 17, 18]]
        offsets = np.cumsum([0] + [len(l) for l in lists]).astype("<u4")
        flat = np.array([p for l in lists for p in l], dtype="<u4")
        start = bf16_bits(rng.standard_normal((tokens, HIDDEN)).astype(np.float32))
        for zero_first in (0, 1):
            with contextlib.ExitStack() as stack:
                d_w = self.gpu.upload(weighted.tobytes()); stack.callback(self.gpu.free, d_w)
                d_o = self.gpu.upload(offsets.tobytes()); stack.callback(self.gpu.free, d_o)
                d_p = self.gpu.upload(flat.tobytes()); stack.callback(self.gpu.free, d_p)
                d_out = self.gpu.upload(start.astype("<u2").tobytes()); stack.callback(self.gpu.free, d_out)
                self.gpu.launch(NAMES[2], ((HIDDEN // 8 + 255) // 256, tokens),
                                [C.c_uint64(d_w), C.c_uint64(d_o), C.c_uint64(d_p), C.c_uint64(d_out), C.c_uint(HIDDEN),
                                 C.c_uint(zero_first)], block=256)
                got = np.frombuffer(self.gpu.download(d_out, tokens * HIDDEN * 2), "<u2").reshape(tokens, HIDDEN)
            for t, l in enumerate(lists):
                total = np.zeros(HIDDEN, np.float32) if zero_first else from_bits(start[t]).copy()
                for p in l:
                    total = bf16(total + from_bits(weighted[p]))
                expected = bf16_bits(total) if (l or zero_first) else start[t]
                self.assertTrue(np.array_equal(got[t], expected), f"zero_first {zero_first}, token {t}")


if __name__ == "__main__":
    unittest.main()
