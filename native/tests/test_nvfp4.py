"""NVFP4 weight kernels against a float64 reference of the dequantized weights.

Tensors come from the converter's NVFP4 quantizer (tools/convert_qwen_safetensors_to_compact_edrl.py).
Decode accumulates in FP32 in its own order; the tile kernels stage BF16(weight) and accumulate with
FP32 MMAs, so the reference uses BF16-rounded weights for them.
"""

import contextlib
import ctypes as C
import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from test_q3_primitives import Gpu, NVRTC, SKIP_REASON

ROOT = Path(__file__).resolve().parents[2]
UNAVAILABLE = (f"CUDA probes unavailable: {SKIP_REASON}" if NVRTC is None
               else "NumPy unavailable" if np is None else None)
converter = None
if np is not None:
    spec = importlib.util.spec_from_file_location("compact_converter", ROOT / "tools/convert_qwen_safetensors_to_compact_edrl.py")
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)


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


@unittest.skipIf(UNAVAILABLE is not None, UNAVAILABLE or "")
class Nvfp4KernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "nvfp4/kernels.cu"\n')
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

    def test_tile_kernels_match_the_reference(self):
        for name, tile in (("euhedral_nvfp4_prefill_128x64", 128), ("euhedral_nvfp4_prefill_64x64", 64)):
            for rows, k, n in ((3, 1024, 64), (130, 2048, 96), (64, 5120, 128)):
                with self.subTest(kernel=name, rows=rows, k=k, n=n):
                    weights, dense = tensor(self.rng, n, k)
                    x = bf16(self.rng.standard_normal((rows, k)).astype(np.float32))
                    expected = x.astype(np.float64) @ bf16(dense).astype(np.float64).T
                    grid = (rows + tile - 1) // tile * ((n + 63) // 64)
                    actual = self.run_kernel(name, grid, x, weights, n, [C.c_uint(rows), C.c_uint(k), C.c_uint(n)], rows)
                    self.assert_close(actual, expected, np.abs(expected).max())

    def test_gate_up_swiglu_matches_the_reference(self):
        rows, k, n = 70, 2048, 128
        weights, dense = tensor(self.rng, n, k)
        x = bf16(self.rng.standard_normal((rows, k)).astype(np.float32))
        projected = x.astype(np.float64) @ bf16(dense).astype(np.float64).T
        gate, up = bf16(projected[:, : n // 2]).astype(np.float64), bf16(projected[:, n // 2:]).astype(np.float64)
        expected = gate / (1.0 + np.exp(-gate)) * up
        for name, tile in (("euhedral_nvfp4_gate_up_swiglu_128x32", 128), ("euhedral_nvfp4_gate_up_swiglu_64x32", 64)):
            with self.subTest(kernel=name):
                grid = (rows + tile - 1) // tile * (n // 2 // 32)
                actual = self.run_kernel(name, grid, x, weights, n // 2, [C.c_uint(rows), C.c_uint(k), C.c_uint(n)], rows)
                self.assert_close(actual, expected, np.abs(expected).max())


if __name__ == "__main__":
    unittest.main()
