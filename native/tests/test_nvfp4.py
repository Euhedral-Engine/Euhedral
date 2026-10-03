"""NVFP4 weight kernels against a float64 reference of the dequantized weights.

Tensors come from the converter's NVFP4 quantizer (tools/euhedral_artifacts/nvfp4.py).
Decode accumulates in FP32 in its own order; the scalar reference (native/src/reference) in another.
"""

import contextlib
import ctypes as C
from pathlib import Path
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


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint32) << 16
    return rounded.astype(np.uint32).view(np.float32)


def to_bf16_bytes(values):
    bits = bf16(values).view(np.uint32) >> 16
    return bits.astype("<u2").tobytes()


def from_bf16_bytes(data, shape):
    return (np.frombuffer(data, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)


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


def sd4_tensor(rng, rows, k):
    """(SD4 tensor bytes, the same weights as a plain NVFP4 tensor's bytes). Rows span a wide range of
    magnitudes, so the table's 16 entries are all in use."""
    weights = (rng.standard_normal((rows, k)) * np.exp(rng.uniform(-4.0, 0.0, (rows, 1)))).astype(np.float32)
    global_scale = converter.nvfp4_global_scale(float(np.abs(weights).max()))
    table = converter.sd4_table(converter.sd4_costs(weights, global_scale))
    packed, indices = converter.quantize_nvfp4_sd4_rows(weights, global_scale, table)
    assert len(np.unique(np.concatenate([indices & 15, indices >> 4]))) == 16, "every table entry in use"
    index_offset, table_offset, size = converter.nvfp4_sd4_offsets((rows, k))
    sd4 = bytearray(size)
    sd4[:packed.nbytes] = packed.tobytes()
    sd4[index_offset:index_offset + indices.nbytes] = indices.tobytes()
    sd4[table_offset:table_offset + 20] = table.tobytes() + np.float32(global_scale).astype("<f4").tobytes()
    scales = converter.expand_nvfp4_sd4(indices, table)
    scale_offset, global_offset, plain_size = converter.nvfp4_offsets((rows, k))
    plain = bytearray(plain_size)
    plain[:packed.nbytes] = packed.tobytes()
    plain[scale_offset:scale_offset + scales.nbytes] = scales.tobytes()
    plain[global_offset:global_offset + 4] = np.float32(global_scale).astype("<f4").tobytes()
    return bytes(sd4), bytes(plain)


@unittest.skipIf(UNAVAILABLE is not None, UNAVAILABLE or "")
class Nvfp4KernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "nvfp4/kernels.cu"\n#include "reference/kernels.cu"\n')
        cls.rng = np.random.default_rng(0xF4)

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def run_kernel(self, name, grid, x, weights, outputs, args, rows):
        with contextlib.ExitStack() as stack:
            dx = self.gpu.upload(to_bf16_bytes(x)); stack.callback(self.gpu.free, dx)
            dw = self.gpu.upload(weights); stack.callback(self.gpu.free, dw)
            dy = self.gpu.zeros(rows * outputs * 2, fill=0xA5); stack.callback(self.gpu.free, dy)
            self.gpu.launch(name, grid, [C.c_uint64(dx), C.c_uint64(dw), C.c_uint64(dy), *args])
            return from_bf16_bytes(self.gpu.download(dy, rows * outputs * 2), (rows, outputs))

    def assert_close(self, actual, expected, scale):
        error = np.abs(actual - expected)
        bound = np.abs(expected) * 2.0**-7 + scale * 1e-3
        self.assertTrue((error <= bound).all(), f"max error {error.max()} at {np.unravel_index(error.argmax(), error.shape)}")

    def test_decode_matches_the_reference(self):
        for k, n in ((1024, 16), (5120, 48), (17408, 32)):
            with self.subTest(k=k, n=n):
                weights, dense = tensor(self.rng, n, k)
                x = bf16(self.rng.standard_normal((1, k)).astype(np.float32))
                expected = (x.astype(np.float64) @ dense.astype(np.float64).T)
                actual = self.run_kernel("euhedral_nvfp4_decode", n // 16, x, weights, n, [C.c_uint(k), C.c_uint(n)], 1)
                self.assert_close(actual, expected, np.abs(expected).max())

    def test_multi_row_decode_is_bitwise_one_row_decode(self):
        k, n = 5120, 48
        weights, _ = tensor(self.rng, n, k)
        for rows in (2, 3, 4, 8):
            with self.subTest(rows=rows):
                x = bf16(self.rng.standard_normal((rows, k)).astype(np.float32))
                together = self.run_kernel(f"euhedral_nvfp4_decode_rows{rows}", n // 16, x, weights, n,
                                           [C.c_uint(k), C.c_uint(n)], rows)
                for row in range(rows):
                    alone = self.run_kernel("euhedral_nvfp4_decode", n // 16, x[row:row + 1], weights, n,
                                            [C.c_uint(k), C.c_uint(n)], 1)
                    self.assertTrue(np.array_equal(together[row].view(np.uint32), alone[0].view(np.uint32)), f"row {row}")

    def test_scalar_reference_matches_the_float64_dequantization(self):
        for rows, k, n in ((1, 1024, 16), (3, 2048, 24), (9, 5120, 40)):
            with self.subTest(rows=rows, k=k, n=n):
                weights, dense = tensor(self.rng, n, k)
                x = bf16(self.rng.standard_normal((rows, k)).astype(np.float32))
                expected = x.astype(np.float64) @ dense.astype(np.float64).T
                actual = self.run_kernel("euhedral_nvfp4_reference", rows * n, x, weights, n,
                                         [C.c_uint(rows), C.c_uint(k), C.c_uint(n)], rows)
                self.assert_close(actual, expected, np.abs(expected).max())

    def assert_bitwise(self, actual, expected):
        self.assertTrue(np.array_equal(actual.view(np.uint32), expected.view(np.uint32)),
                        f"{np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))} outputs differ")

    def test_sd4_kernels_are_bitwise_the_plain_kernels_on_the_expanded_tensor(self):
        for k, n in ((1024, 64), (5120, 128), (17408, 64)):
            sd4, plain = sd4_tensor(self.rng, n, k)
            shape = [C.c_uint(k), C.c_uint(n)]
            for rows in range(1, 9):
                with self.subTest(kernel="decode", k=k, rows=rows):
                    x = bf16(self.rng.standard_normal((rows, k)).astype(np.float32))
                    name = "euhedral_nvfp4_decode" if rows == 1 else f"euhedral_nvfp4_decode_rows{rows}"
                    self.assert_bitwise(self.run_kernel(name + "_sd4", n // 16, x, sd4, n, shape, rows),
                                        self.run_kernel(name, n // 16, x, plain, n, shape, rows))
            for rows in (1, 3):
                with self.subTest(kernel="reference", k=k, rows=rows):
                    x = bf16(self.rng.standard_normal((rows, k)).astype(np.float32))
                    args = [C.c_uint(rows), C.c_uint(k), C.c_uint(n)]
                    self.assert_bitwise(self.run_kernel("euhedral_nvfp4_reference_sd4", rows * n, x, sd4, n, args, rows),
                                        self.run_kernel("euhedral_nvfp4_reference", rows * n, x, plain, n, args, rows))


if __name__ == "__main__":
    unittest.main()
