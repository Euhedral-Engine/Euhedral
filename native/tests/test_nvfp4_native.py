"""Native Blackwell NVFP4 kernels (native/src/nvfp4_native, docs/NVFP4_NATIVE.md) against float64.

The linear kernels are checked over the operands they actually consume (the quantized activations
read back from the activation buffer and the converter's quantized weights), so any deviation beyond
FP32 accumulation and BF16 output rounding is a layout or scale error. The quantizer is checked by its
reconstruction error: about 10% for one term, about 1% with the residual term. Runs only on sm_12x.
"""

import contextlib
import ctypes as C
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from test_q3_primitives import Gpu, NVRTC, CUDA, SKIP_REASON
from test_nvfp4 import bf16, to_bf16_bytes, from_bf16_bytes, tensor, converter

E2M1 = None if np is None else np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)
SHARED = 67584


def device_architecture():
    if CUDA is None:
        return None
    major, minor = C.c_int(), C.c_int()
    if CUDA.cuInit(0) or CUDA.cuDeviceGetAttribute(C.byref(major), 75, 0) or CUDA.cuDeviceGetAttribute(C.byref(minor), 76, 0):
        return None
    return f"sm_{major.value}{minor.value}a" if major.value == 12 else None


ARCH = None if NVRTC is None else device_architecture()
UNAVAILABLE = (f"CUDA probes unavailable: {SKIP_REASON}" if NVRTC is None else "NumPy unavailable" if np is None
               else "not an sm_12x device" if ARCH is None else None)


def e4m3(b):
    b = np.asarray(b, np.int64) & 0x7f
    e, m = (b >> 3) & 15, b & 7
    return np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7))


def layout(rows, k, terms):
    a256 = lambda v: (v + 255) // 256 * 256
    scales = a256(terms * rows * k // 2)
    globals_ = a256(scales + terms * rows * k // 16)
    return scales, globals_, globals_ + 4 * rows


def dequantize(buffer, rows, k, terms):
    so, go, _ = layout(rows, k, terms)
    planes = terms * rows
    data = np.frombuffer(buffer, np.uint8)
    codes = data[:planes * k // 2].reshape(planes, k // 2)
    values = np.empty((planes, k))
    values[:, 0::2], values[:, 1::2] = E2M1[codes & 15], E2M1[codes >> 4]
    values *= np.repeat(e4m3(data[so:so + planes * k // 16].reshape(planes, k // 16)), 16, axis=1)
    globals_ = np.frombuffer(data[go:go + 4 * rows].tobytes(), np.float32).astype(np.float64)
    return values.reshape(terms, rows, k).sum(0) * globals_[:, None]


@unittest.skipIf(UNAVAILABLE is not None, UNAVAILABLE or "")
class Nvfp4NativeKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "nvfp4_native/kernels.cu"\n', architecture=ARCH)
        cls.rng = np.random.default_rng(0xF4F4)

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def run_native(self, x, weights, rows, k, cols, terms, paired=False):
        suffix = "" if terms == 1 else "_x2"
        out_cols = cols // 2 if paired else cols
        with contextlib.ExitStack() as stack:
            dx = self.gpu.upload(to_bf16_bytes(x)); stack.callback(self.gpu.free, dx)
            size = layout(rows, k, terms)[2]
            da = self.gpu.zeros(size, fill=0xA5); stack.callback(self.gpu.free, da)
            dw = self.gpu.upload(weights); stack.callback(self.gpu.free, dw)
            dy = self.gpu.zeros(rows * out_cols * 2, fill=0xA5); stack.callback(self.gpu.free, dy)
            self.gpu.launch("euhedral_nvfp4n_quantize_rows" + suffix, rows,
                            [C.c_uint64(dx), C.c_uint64(da), C.c_uint(rows), C.c_uint(k)], block=128)
            kernel = f"euhedral_nvfp4n_gate_up_swiglu{suffix}_128x64" if paired else f"euhedral_nvfp4n_linear{suffix}_128x128"
            grid = (rows + 127) // 128 * ((cols + 127) // 128)
            self.gpu.launch(kernel, grid, [C.c_uint64(da), C.c_uint64(dw), C.c_uint64(dy), C.c_uint(rows), C.c_uint(k),
                                           C.c_uint(cols)], block=256, shared=SHARED)
            activations = self.gpu.download(da, size)
            return activations, from_bf16_bytes(self.gpu.download(dy, rows * out_cols * 2), (rows, out_cols))

    def test_quantizer_reconstruction_error(self):
        for terms, bound in ((1, 0.11), (2, 0.012)):
            with self.subTest(terms=terms):
                rows, k = 37, 5120
                x = bf16(self.rng.standard_t(3, (rows, k)).astype(np.float32) * self.rng.uniform(0.01, 10, (rows, 1)).astype(np.float32))
                weights, _ = tensor(self.rng, 128, k)
                activations, _ = self.run_native(x, weights, rows, k, 128, terms)
                error = dequantize(activations, rows, k, terms) - x
                relative = np.sqrt((error ** 2).mean(1) / (x.astype(np.float64) ** 2).mean(1))
                self.assertLess(relative.max(), bound)

    def test_linear_matches_float64_over_its_operands(self):
        for terms in (1, 2):
            for rows, k, cols in ((3, 5120, 256), (130, 2048, 200), (257, 17408, 384)):
                with self.subTest(terms=terms, rows=rows, k=k, cols=cols):
                    weights, dense = tensor(self.rng, cols, k)
                    x = bf16(self.rng.standard_normal((rows, k)).astype(np.float32))
                    activations, y = self.run_native(x, weights, rows, k, cols, terms)
                    expected = dequantize(activations, rows, k, terms) @ dense.astype(np.float64).T
                    scale = np.abs(expected).max()
                    self.assertLess(np.abs(y - expected).max(), 2.0 ** -7 * scale)

    def test_paired_gate_up_swiglu_matches_float64_over_its_operands(self):
        for terms in (1, 2):
            with self.subTest(terms=terms):
                rows, k, cols = 70, 2048, 384
                weights, dense = tensor(self.rng, cols, k)
                x = bf16(self.rng.standard_normal((rows, k)).astype(np.float32))
                activations, y = self.run_native(x, weights, rows, k, cols, terms, paired=True)
                projected = dequantize(activations, rows, k, terms) @ dense.astype(np.float64).T
                gate = bf16(projected[:, :cols // 2].astype(np.float32)).astype(np.float64)
                up = bf16(projected[:, cols // 2:].astype(np.float32)).astype(np.float64)
                expected = gate / (1.0 + np.exp(-gate)) * up
                self.assertLess(np.abs(y - expected).max(), 2.0 ** -7 * np.abs(expected).max())

    def test_skinny_linear_matches_float64_over_its_operands(self):
        for terms in (1, 2):
            for rows, k, cols, splits in ((2, 5120, 320, 1), (16, 2048, 200, 3), (29, 4096, 384, 2), (64, 2048, 128, 1)):
                with self.subTest(terms=terms, rows=rows, k=k, cols=cols, splits=splits):
                    weights, dense = tensor(self.rng, cols, k)
                    x = bf16(self.rng.standard_normal((rows, k)).astype(np.float32))
                    fragments = 1 if rows <= 16 else 2 if rows <= 32 else 4
                    stage = (terms * 16 * fragments + 64) * 160
                    shared = min(4, 101376 // stage) * stage
                    suffix = "" if terms == 1 else "_x2"
                    with contextlib.ExitStack() as stack:
                        dx = self.gpu.upload(to_bf16_bytes(x)); stack.callback(self.gpu.free, dx)
                        size = layout(rows, k, terms)[2]
                        da = self.gpu.zeros(size); stack.callback(self.gpu.free, da)
                        dw = self.gpu.upload(weights); stack.callback(self.gpu.free, dw)
                        dy = self.gpu.zeros(rows * cols * 2, fill=0xA5); stack.callback(self.gpu.free, dy)
                        dp = self.gpu.zeros(splits * rows * cols * 4); stack.callback(self.gpu.free, dp)
                        self.gpu.launch("euhedral_nvfp4n_quantize_rows" + suffix, rows,
                                        [C.c_uint64(dx), C.c_uint64(da), C.c_uint(rows), C.c_uint(k)], block=128)
                        self.gpu.launch(f"euhedral_nvfp4n_skinny{suffix}_{16 * fragments}", ((cols + 63) // 64, splits),
                                        [C.c_uint64(da), C.c_uint64(dw), C.c_uint64(dy), C.c_uint64(dp), C.c_uint(rows),
                                         C.c_uint(k), C.c_uint(cols)], block=128, shared=shared)
                        if splits > 1:
                            self.gpu.launch("euhedral_nvfp4n_skinny_finish", (rows * cols + 255) // 256,
                                            [C.c_uint64(dp), C.c_uint64(dy), C.c_uint(rows * cols), C.c_uint(splits)],
                                            block=256)
                        activations = self.gpu.download(da, size)
                        y = from_bf16_bytes(self.gpu.download(dy, rows * cols * 2), (rows, cols))
                    expected = dequantize(activations, rows, k, terms) @ dense.astype(np.float64).T
                    self.assertLess(np.abs(y - expected).max(), 2.0 ** -7 * np.abs(expected).max())


    def skinny(self, x, weights, k, cols, splits):
        """One skinny linear (one term, up to 16 rows) as the host runs it; BF16 output bytes."""
        rows = x.shape[0]
        shared = min(4, 101376 // ((16 + 64) * 160)) * (16 + 64) * 160
        with contextlib.ExitStack() as stack:
            dx = self.gpu.upload(to_bf16_bytes(x)); stack.callback(self.gpu.free, dx)
            size = layout(rows, k, 1)[2]
            da = self.gpu.zeros(size); stack.callback(self.gpu.free, da)
            dw = self.gpu.upload(weights); stack.callback(self.gpu.free, dw)
            dy = self.gpu.zeros(rows * cols * 2, fill=0xA5); stack.callback(self.gpu.free, dy)
            dp = self.gpu.zeros(splits * rows * cols * 4, fill=0x7F); stack.callback(self.gpu.free, dp)
            self.gpu.launch("euhedral_nvfp4n_quantize_rows", rows,
                            [C.c_uint64(dx), C.c_uint64(da), C.c_uint(rows), C.c_uint(k)], block=128)
            self.gpu.launch("euhedral_nvfp4n_skinny_16", ((cols + 63) // 64, splits),
                            [C.c_uint64(da), C.c_uint64(dw), C.c_uint64(dy), C.c_uint64(dp), C.c_uint(rows),
                             C.c_uint(k), C.c_uint(cols)], block=128, shared=shared)
            if splits > 1:
                self.gpu.launch("euhedral_nvfp4n_skinny_finish", (rows * cols + 255) // 256,
                                [C.c_uint64(dp), C.c_uint64(dy), C.c_uint(rows * cols), C.c_uint(splits)], block=256)
            return self.gpu.download(dy, rows * cols * 2)

    def test_skinny_rows_are_deterministic_and_independent_of_the_other_rows(self):
        """Native-numerics speculative verification relies on this: a row's output is bit for bit the
        same whether it runs alone (decode) or with up to 15 other rows (verification), and run to run."""
        k, cols = 5120, 192
        weights, _ = tensor(self.rng, cols, k)
        x = bf16(self.rng.standard_normal((4, k)).astype(np.float32))
        for splits in (1, 2, 3, 7):
            with self.subTest(splits=splits):
                together = self.skinny(x, weights, k, cols, splits)
                self.assertEqual(together, self.skinny(x, weights, k, cols, splits), "run to run")
                for row in range(4):
                    alone = self.skinny(x[row:row + 1], weights, k, cols, splits)
                    self.assertEqual(together[row * cols * 2:(row + 1) * cols * 2], alone, f"row {row}")


if __name__ == "__main__":
    unittest.main()
