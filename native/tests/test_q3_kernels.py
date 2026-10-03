"""Q3G64_F16S kernels (native/src/q3, native/src/reference).

The scalar reference is checked against an FP64 dot product of the dequantized weights; the contiguous decode
kernel against the reference within one BF16 step; its multi-row twins bit for bit against one-row launches.
Scale patterns cover zero, subnormal, maximum and mixed-sign scales.
"""

import contextlib
import ctypes as C
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from gpu_harness import Gpu, NVRTC, SKIP_REASON

UNAVAILABLE = None
if NVRTC is None:
    UNAVAILABLE = f"CUDA probes unavailable: {SKIP_REASON}"
elif np is None:
    UNAVAILABLE = "NumPy unavailable"

EDGE_SCALES = [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3555, 0xB555, 0x3C00, 0x7BFF, 0xFBFF]


def align256(value):
    return (value + 255) // 256 * 256


def pack_codes(codes):
    """Signed 3-bit codes (rows, k) into G64 groups of 24 bytes, code i of a group at bit 3 i."""
    rows, k = codes.shape
    groups = k // 64
    unsigned = (codes.astype(np.int64) & 7).reshape(rows * groups, 64)
    out = np.zeros((rows * groups, 24), dtype=np.uint8)
    for i in range(64):
        bit = 3 * i
        value = unsigned[:, i]
        out[:, bit // 8] |= ((value << (bit % 8)) & 0xFF).astype(np.uint8)
        if bit % 8 > 5:
            out[:, bit // 8 + 1] |= (value >> (8 - bit % 8)).astype(np.uint8)
    return out.tobytes()


def tensor(rng, rows, k, scales=None):
    """A row-split Q3 tensor (codes, then FP16 scales at align256) and the dequantized float64 matrix."""
    groups = k // 64
    codes = rng.integers(-4, 4, size=(rows, k), dtype=np.int8)
    if scales is None:
        scales = rng.integers(0, 0x7C00, rows * groups, dtype=np.uint16) | (
            rng.integers(0, 2, rows * groups, dtype=np.uint16) << 15)
    scales = np.asarray(scales, dtype=np.uint16)
    scale_offset = align256(rows * groups * 24)
    data = bytearray(scale_offset + rows * groups * 2)
    packed = pack_codes(codes)
    data[:len(packed)] = packed
    data[scale_offset:] = scales.astype("<u2").tobytes()
    values = scales.view(np.float16).astype(np.float64).reshape(rows, groups)
    weights = codes.astype(np.float64) * np.repeat(values, 64, axis=1)
    return bytes(data), scale_offset, weights


def bf16_bits(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_floats(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


@unittest.skipIf(UNAVAILABLE is not None, UNAVAILABLE or "")
class Q3KernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "q3/kernels.cu"\n#include "reference/kernels.cu"\n')
        cls.rng = np.random.default_rng(0x0513)

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def run_kernel(self, name, data, scale_offset, activations, rows, k, outputs, grid, block=128):
        gpu = self.gpu
        with contextlib.ExitStack() as stack:
            dx = gpu.upload(activations.astype("<u2").tobytes()); stack.callback(gpu.free, dx)
            dw = gpu.upload(data); stack.callback(gpu.free, dw)
            dy = gpu.zeros(rows * outputs * 2, fill=0xA5); stack.callback(gpu.free, dy)
            gpu.launch(name, grid, [C.c_uint64(dx), C.c_uint64(dw), C.c_uint64(dy), C.c_uint(rows), C.c_uint(k),
                                    C.c_uint(outputs), C.c_uint64(scale_offset)], block=block)
            return np.frombuffer(gpu.download(dy, rows * outputs * 2), dtype="<u2").copy()

    def activations(self, rows, k):
        return bf16_bits(self.rng.standard_normal((rows, k)).astype(np.float32) * 2.0)

    def reference(self, data, scale_offset, activations, rows, k, outputs):
        return self.run_kernel("euhedral_q3_reference", data, scale_offset, activations, rows, k, outputs, rows * outputs)

    def assertWithinOneStep(self, expected_bits, actual_bits, context):
        expected, actual = bf16_floats(expected_bits), bf16_floats(actual_bits)
        step = np.abs(expected_bits.astype(np.int64) - actual_bits.astype(np.int64))
        near = np.abs(expected - actual) <= 0.01 * np.sqrt(np.mean(expected ** 2)) + 1e-3
        ok = np.isfinite(actual) & ((step <= 1) | near)
        self.assertTrue(ok.all(), f"{context}: {np.flatnonzero(~ok)[:5]}")

    def test_reference_matches_fp64_dot_product(self):
        for rows, k, outputs in [(1, 128, 3), (3, 192, 5), (2, 1024, 16)]:
            if k % 128:
                continue
            data, scale_offset, weights = tensor(self.rng, outputs, k, scales=self.rng.integers(
                0x2000, 0x4800, outputs * (k // 64), dtype=np.uint16))
            x = self.activations(rows, k)
            actual = bf16_floats(self.reference(data, scale_offset, x, rows, k, outputs)).reshape(rows, outputs)
            expected = bf16_floats(x).reshape(rows, k) @ weights.T
            scale = np.abs(expected).max() + 1e-6
            self.assertLessEqual(np.abs(actual - expected).max(), 0.01 * scale, f"rows={rows} k={k}")

    def test_contiguous_decode_stays_within_one_bf16_step_of_the_reference(self):
        for k, outputs in [(1024, 16), (2048, 48), (6144, 32)]:
            for pattern in ("random", "edge"):
                scales = None
                if pattern == "edge":
                    groups = k // 64
                    scales = np.array([EDGE_SCALES[(r + g) % len(EDGE_SCALES)] for r in range(outputs)
                                       for g in range(groups)], dtype=np.uint16)
                data, scale_offset, _ = tensor(self.rng, outputs, k, scales)
                x = self.activations(1, k)
                actual = self.run_kernel("euhedral_q3_decode_contiguous", data, scale_offset, x, 1, k, outputs, outputs // 16)
                expected = self.reference(data, scale_offset, x, 1, k, outputs)
                self.assertWithinOneStep(expected, actual, f"k={k} outputs={outputs} {pattern}")

    def test_multi_row_twins_are_bitwise_one_row_decode(self):
        k, outputs = 2048, 48
        data, scale_offset, _ = tensor(self.rng, outputs, k)
        x = self.activations(8, k)
        single = [self.run_kernel("euhedral_q3_decode_contiguous", data, scale_offset, x[r:r + 1], 1, k, outputs,
                                  outputs // 16) for r in range(8)]
        for rows in range(2, 9):
            actual = self.run_kernel(f"euhedral_q3_decode_contiguous_rows{rows}", data, scale_offset, x[:rows], rows, k,
                                     outputs, outputs // 16).reshape(rows, outputs)
            for r in range(rows):
                np.testing.assert_array_equal(actual[r], single[r], f"rows={rows} row={r}")


if __name__ == "__main__":
    unittest.main()
