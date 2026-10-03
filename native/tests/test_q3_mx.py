"""Q3 prefill on block-scaled E4M3 tensor cores (native/src/q3_mx, docs/PREFILL_MX.md) against float64.

The activation quantizer must reproduce BF16 inputs exactly (two E4M3 terms and a power-of-two block scale) wherever
the element is within the E4M3 range of its block's maximum, and the GEMMs must match an FP64 oracle of the exact
products (BF16 activation x Q3 code, FP16 group scale applied in FP32) up to FP32 accumulation and BF16 output
rounding. Runs only on sm_12x.
"""
import ctypes as C
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from test_q3_primitives import Gpu, NVRTC, CUDA, SKIP_REASON, P
from test_nvfp4_native import ARCH, UNAVAILABLE, e4m3
from test_attention_nvfp4 import bf16

U = C.c_uint
SHARED = 3 * (3 * 128 * 64 + 128 * 4)


def pack_q3(codes, scales):
    """codes int [N][K] in -4..3, scales fp16 [N][K/64] -> the compact weight buffer (24-byte groups, scales at a 256 offset)."""
    n, k = codes.shape
    groups = k // 64
    fields = (codes.reshape(n, groups, 64) & 7).astype(np.uint8)
    bits = ((fields[..., None] >> np.arange(3)) & 1).reshape(n, groups, 192).astype(np.uint8)
    packed = np.packbits(bits, axis=-1, bitorder="little").reshape(-1)
    offset = (n * groups * 24 + 255) // 256 * 256
    buffer = np.zeros(offset + n * groups * 2, np.uint8)
    buffer[:packed.size] = packed
    buffer[offset:] = scales.astype(np.float16).reshape(-1).view(np.uint8)
    return buffer.tobytes(), offset


def pack_q45(codes, scales, bits):
    """codes int [N][K] (Q4 -8..7, Q5 -16..15) -> nibble plane, fifth-bit plane (Q5), scales; planes 256-aligned.
    Returns (buffer, high_offset, scale_offset)."""
    n, k = codes.shape
    groups = k // 64
    align = lambda v: (v + 255) // 256 * 256
    u = (codes & ((1 << bits) - 1)).astype(np.uint8).reshape(n, groups, 64)
    nibbles = u & 15
    code_bytes = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).reshape(-1)
    code_plane = align(n * groups * 32)
    high_plane = align(n * groups * 8) if bits == 5 else 0
    buffer = np.zeros(code_plane + high_plane + n * groups * 2, np.uint8)
    buffer[:code_bytes.size] = code_bytes
    if bits == 5:
        fifth = ((u >> 4) & 1).astype(np.uint8)
        buffer[code_plane:code_plane + n * groups * 8] = np.packbits(fifth, axis=-1, bitorder="little").reshape(-1)
    buffer[code_plane + high_plane:] = scales.astype(np.float16).reshape(-1).view(np.uint8)
    return buffer.tobytes(), code_plane, code_plane + high_plane


def bf16_values(rng, shape, scale=1.0):
    bits, values = bf16(rng.standard_normal(shape).astype(np.float32) * scale)
    return bits.reshape(shape), values.reshape(shape)


@unittest.skipIf(UNAVAILABLE is not None, UNAVAILABLE or "")
class Q3MxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "q3_mx/kernels.cu"\n', architecture=ARCH)
        cls.rng = np.random.default_rng(0x3D3D)

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def quantize(self, bits, rows, width):
        gpu = self.gpu
        dx = gpu.upload(bits.astype(np.uint16).tobytes())
        dhi, dlo, dsc = gpu.zeros(rows * width), gpu.zeros(rows * width), gpu.zeros(rows * width // 32)
        blocks = (rows * (width // 32) + 255) // 256
        gpu.launch("euhedral_q3mx_quantize", blocks, [P(dx), P(dhi), P(dlo), P(dsc), U(rows), U(width)], block=256)
        hi = np.frombuffer(gpu.download(dhi, rows * width), np.uint8).reshape(rows, width)
        lo = np.frombuffer(gpu.download(dlo, rows * width), np.uint8).reshape(rows, width)
        sc = np.frombuffer(gpu.download(dsc, rows * width // 32), np.uint8).reshape(rows, width // 32)
        for p in (dx, dhi, dlo, dsc):
            gpu.free(p)
        return hi, lo, sc

    @staticmethod
    def reconstruct(hi, lo, sc):
        signed = lambda b: np.where(b & 0x80, -1.0, 1.0) * e4m3(b)
        scale = np.repeat(2.0 ** (sc.astype(np.float64) - 127), 32, axis=1)
        return (signed(hi) + signed(lo)) * scale

    def test_quantizer_reproduces_bf16_activations(self):
        rows, width = 64, 512
        for magnitude in (1e-3, 1.0, 40.0, 3e4):
            bits, values = bf16_values(self.rng, (rows, width), magnitude)
            values[3, 10] = 0.0
            values[5, 32:64] = 0.0                     # an all-zero block
            values[7, 100] = magnitude * 5e3           # a block with an outlier: its small elements lose bits, not the outlier
            bits = (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            values = (bits.astype(np.uint32) << 16).view(np.float32)          # the BF16 values actually quantized
            hi, lo, sc = self.quantize(bits, rows, width)
            got = self.reconstruct(hi, lo, sc)
            block_max = np.repeat(np.abs(values).reshape(rows, -1, 32).max(axis=2), 32, axis=1)
            exact = np.abs(values) >= block_max * 2.0 ** -9       # the lo term stays above the E4M3 subnormal grid (x / 2^e >= 1/4)
            np.testing.assert_array_equal(got[exact], values[exact].astype(np.float64), err_msg=f"magnitude {magnitude}")
            # Smaller elements keep the E4M3 subnormal grid of the block: absolute error below 2^-17 of the block maximum.
            self.assertLessEqual(np.max(np.abs(got - values) / np.maximum(block_max, 1e-30)), 2.0 ** -17)
            self.assertEqual(int(sc[5, 1]), 127)           # a zero block gets unit scale

    def run_gemm(self, paired, rows, n, k, magnitude=1.0):
        gpu, rng = self.gpu, self.rng
        codes = rng.integers(-4, 4, (n, k)).astype(np.int8)
        scales = rng.uniform(0.002, 0.02, (n, k // 64)).astype(np.float16)
        weights, scale_offset = pack_q3(codes, scales)
        bits, x = bf16_values(rng, (rows, k), magnitude)
        hi, lo, sc = self.quantize(bits, rows, k)
        dhi, dlo, dsc = gpu.upload(hi.tobytes()), gpu.upload(lo.tobytes()), gpu.upload(sc.tobytes())
        dw = gpu.upload(weights)
        width = n // 2 if paired else n
        dout = gpu.zeros(rows * width * 2, 0xAA)
        grid = ((rows + 127) // 128) * (n // 128)
        name = "euhedral_q3mx_gate_up_swiglu_128x64" if paired else "euhedral_q3mx_linear_128x128"
        gpu.launch(name, grid, [P(dhi), P(dlo), P(dsc), P(dw), P(dout), U(rows), U(n), U(k), C.c_uint64(0), C.c_uint64(scale_offset), P(0)],
                   block=384, shared=SHARED)
        got = (np.frombuffer(gpu.download(dout, rows * width * 2), np.uint16).astype(np.uint32) << 16).view(np.float32)
        got = got.reshape(rows, width).astype(np.float64)
        for p in (dhi, dlo, dsc, dw, dout):
            gpu.free(p)
        weights64 = codes.reshape(n, k // 64, 64).astype(np.float64) * scales.astype(np.float64)[:, :, None]
        y = np.einsum("mk,nk->mn", x.astype(np.float64), weights64.reshape(n, k))
        if paired:
            gate = (np.frombuffer(np.float32(y[:, :width]).tobytes(), np.uint32) & 0xFFFF0000).view(np.float32).astype(np.float64)
            return got, y, x
        return got, y, x

    def test_linear_matches_float64(self):
        for rows, n, k in [(128, 128, 128), (200, 256, 384), (37, 384, 256), (512, 128, 1024)]:
            with self.subTest(rows=rows, n=n, k=k):
                got, y, _ = self.run_gemm(False, rows, n, k)
                # FP32 accumulation of exact products, then BF16 output rounding (2^-9).
                np.testing.assert_allclose(got, y, rtol=2.0 ** -8, atol=2.0 ** -8 * np.abs(y).max() * 1e-3)
                self.assertLess(np.max(np.abs(got - y)) / np.abs(y).max(), 2.0 ** -8)

    def test_q4_and_q5_linears_match_float64(self):
        gpu, rng = self.gpu, self.rng
        for bits in (4, 5):
            for rows, n, k in [(128, 128, 128), (200, 256, 384), (37, 384, 256)]:
                with self.subTest(bits=bits, rows=rows, n=n, k=k):
                    low, high = -(1 << (bits - 1)), (1 << (bits - 1))
                    codes = rng.integers(low, high, (n, k)).astype(np.int8)
                    codes[0, :64] = low                                  # the most negative code throughout a group
                    codes[1, :64] = high - 1
                    scales = rng.uniform(0.002, 0.02, (n, k // 64)).astype(np.float16)
                    weights, high_offset, scale_offset = pack_q45(codes, scales, bits)
                    bits16, x = bf16_values(rng, (rows, k))
                    hi, lo, sc = self.quantize(bits16, rows, k)
                    dhi, dlo, dsc, dw = gpu.upload(hi.tobytes()), gpu.upload(lo.tobytes()), gpu.upload(sc.tobytes()), gpu.upload(weights)
                    dout = gpu.zeros(rows * n * 2, 0xAA)
                    gpu.launch(f"euhedral_q{bits}mx_linear_128x128", ((rows + 127) // 128) * (n // 128),
                               [P(dhi), P(dlo), P(dsc), P(dw), P(dout), U(rows), U(n), U(k), C.c_uint64(high_offset),
                                C.c_uint64(scale_offset), P(0)], block=384, shared=SHARED)
                    got = (np.frombuffer(gpu.download(dout, rows * n * 2), np.uint16).astype(np.uint32) << 16).view(np.float32)
                    got = got.reshape(rows, n).astype(np.float64)
                    for p in (dhi, dlo, dsc, dw, dout):
                        gpu.free(p)
                    w64 = (codes.reshape(n, k // 64, 64) * scales.astype(np.float64)[:, :, None]).reshape(n, k)
                    y = x.astype(np.float64) @ w64.T
                    self.assertLess(np.max(np.abs(got - y)) / np.abs(y).max(), 2.0 ** -8)

    def test_split_k_linear_matches_float64(self):
        gpu, rng = self.gpu, self.rng
        for splits, rows, n, k in [(2, 200, 256, 512), (4, 128, 128, 1024), (2, 37, 384, 768)]:
            with self.subTest(splits=splits, rows=rows, n=n, k=k):
                codes = rng.integers(-4, 4, (n, k)).astype(np.int8)
                scales = rng.uniform(0.002, 0.02, (n, k // 64)).astype(np.float16)
                weights, scale_offset = pack_q3(codes, scales)
                bits, x = bf16_values(rng, (rows, k))
                hi, lo, sc = self.quantize(bits, rows, k)
                dhi, dlo, dsc, dw = gpu.upload(hi.tobytes()), gpu.upload(lo.tobytes()), gpu.upload(sc.tobytes()), gpu.upload(weights)
                dout, dpart = gpu.zeros(rows * n * 2, 0xAA), gpu.zeros(splits * rows * n * 4, 0xAA)
                gpu.launch("euhedral_q3mx_linear_128x128", (((rows + 127) // 128) * (n // 128), splits),
                           [P(dhi), P(dlo), P(dsc), P(dw), P(dout), U(rows), U(n), U(k), C.c_uint64(0), C.c_uint64(scale_offset), P(dpart)],
                           block=384, shared=SHARED)
                gpu.launch("euhedral_q3mx_reduce", (rows * n // 4 + 255) // 256, [P(dpart), P(dout), U(rows * n), U(splits)], block=256)
                got = (np.frombuffer(gpu.download(dout, rows * n * 2), np.uint16).astype(np.uint32) << 16).view(np.float32)
                got = got.reshape(rows, n).astype(np.float64)
                for p in (dhi, dlo, dsc, dw, dout, dpart):
                    gpu.free(p)
                w64 = (codes.reshape(n, k // 64, 64) * scales.astype(np.float64)[:, :, None]).reshape(n, k)
                y = x.astype(np.float64) @ w64.T
                self.assertLess(np.max(np.abs(got - y)) / np.abs(y).max(), 2.0 ** -8)

    def test_gate_up_swiglu_matches_float64(self):
        for rows, n, k in [(128, 128, 128), (200, 256, 384), (65, 512, 256)]:
            with self.subTest(rows=rows, n=n, k=k):
                got, y, _ = self.run_gemm(True, rows, n, k)
                width = n // 2
                bf = lambda a: (np.float32(a).view(np.uint32) & 0xFFFF0000).view(np.float32).astype(np.float64)
                gate, up = bf(y[:, :width]), bf(y[:, width:])
                expected = gate / (1 + np.exp(-gate)) * up
                self.assertLess(np.max(np.abs(got - expected)) / np.abs(expected).max(), 2.0 ** -6)

    def test_activations_with_outliers_and_zero_rows(self):
        gpu, rng = self.gpu, self.rng
        rows, n, k = 129, 256, 256
        codes = rng.integers(-4, 4, (n, k)).astype(np.int8)
        scales = rng.uniform(0.002, 0.02, (n, k // 64)).astype(np.float16)
        weights, scale_offset = pack_q3(codes, scales)
        bits, x = bf16_values(rng, (rows, k))
        x[10, :] = 0.0
        x[20, 5] = 3000.0
        x[30, 64:128] *= 1e-6
        bits = (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        x = (bits.astype(np.uint32) << 16).view(np.float32)
        hi, lo, sc = self.quantize(bits, rows, k)
        dhi, dlo, dsc, dw = gpu.upload(hi.tobytes()), gpu.upload(lo.tobytes()), gpu.upload(sc.tobytes()), gpu.upload(weights)
        dout = gpu.zeros(rows * n * 2, 0xAA)
        gpu.launch("euhedral_q3mx_linear_128x128", ((rows + 127) // 128) * (n // 128),
                   [P(dhi), P(dlo), P(dsc), P(dw), P(dout), U(rows), U(n), U(k), C.c_uint64(0), C.c_uint64(scale_offset), P(0)], block=384, shared=SHARED)
        got = (np.frombuffer(gpu.download(dout, rows * n * 2), np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(rows, n)
        weights64 = (codes.reshape(n, k // 64, 64) * scales.astype(np.float64)[:, :, None]).reshape(n, k)
        y = x.astype(np.float64) @ weights64.T
        self.assertTrue(np.all(got[10] == 0))
        err = np.abs(got - y) / np.maximum(np.abs(y).max(axis=1, keepdims=True), 1e-30)
        self.assertLess(err.max(), 2.0 ** -8)


if __name__ == "__main__":
    unittest.main()
