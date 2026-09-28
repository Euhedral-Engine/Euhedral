"""Restructured Q4/Q5 kernels against the original schedule and an independent CPU model.

The production module (native/src/q45/) and the pre-restructuring reference kernels
(q45_reference.cu) are compiled into one NVRTC module. Every route must match the
reference bitwise on sentinel-filled outputs; a CPU dequantization bounds both.
"""

import contextlib
import ctypes as C
import pathlib
import random
import struct
import unittest

from test_q3_primitives import Gpu, NVRTC, SKIP_REASON, bf16_value, to_bf16

HERE = pathlib.Path(__file__).resolve().parent
SENTINEL = 0xA5


def align256(value):
    return (value + 255) & ~255


def plane_offsets(bits, width, outputs):
    groups = width // 64
    high = align256(outputs * groups * 32)
    scale = high + (align256(outputs * groups * 8) if bits == 5 else 0)
    return groups, high, scale


def make_weights(rng, bits, width, outputs, scale_bits=None):
    groups, high, scale = plane_offsets(bits, width, outputs)
    payload = bytearray(rng.randrange(256) for _ in range(scale + outputs * groups * 2))
    for i in range(outputs * groups):
        value = scale_bits(i) if scale_bits else rng.randrange(0x2C00, 0x3400) | (i & 1) << 15
        struct.pack_into("<H", payload, scale + 2 * i, value)
    return bytes(payload)


def cpu_code(payload, bits, width, outputs, out, k):
    groups, high, _ = plane_offsets(bits, width, outputs)
    g = out * groups + k // 64
    lane = k % 64
    value = (payload[g * 32 + lane // 2] >> ((lane & 1) * 4)) & 15
    if bits == 5:
        value |= ((payload[high + g * 8 + lane // 8] >> (lane & 7)) & 1) << 4
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def cpu_reference(payload, bits, values, rows, width, outputs):
    groups, _, scale_offset = plane_offsets(bits, width, outputs)
    scales = [struct.unpack_from("<e", payload, scale_offset + 2 * i)[0] for i in range(outputs * groups)]
    result = []
    for r in range(rows):
        x = [bf16_value(values[r * width + k]) for k in range(width)]
        for n in range(outputs):
            total = 0.0
            for k in range(width):
                total += x[k] * cpu_code(payload, bits, width, outputs, n, k) * scales[n * groups + k // 64]
            result.append(total)
    return result


@unittest.skipIf(NVRTC is None, f"CUDA probes unavailable: {SKIP_REASON}")
class Q45KernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = b'#include "q45/kernels.cu"\n' + (HERE / "q45_reference.cu").read_bytes()
        cls.gpu = Gpu(source)
        cls.rng = random.Random(0x0545)

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def owned(self, stack, pointer):
        stack.callback(self.gpu.free, pointer)
        return pointer

    def run_kernel(self, stack, name, grid, x, w, rows, width, outputs):
        y = self.owned(stack, self.gpu.zeros(rows * outputs * 2, fill=SENTINEL))
        self.gpu.launch(name, grid, [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y),
                                     C.c_uint(rows), C.c_uint(width), C.c_uint(outputs)])
        return self.gpu.download(y, rows * outputs * 2)

    def assert_written(self, output):
        words = struct.unpack(f"<{len(output) // 2}H", output)
        self.assertNotIn(SENTINEL * 0x101, words)

    def routes(self, bits, rows, outputs):
        decode_tiles = (outputs + 7) // 8
        prefill_tiles = (outputs + 31) // 32
        routes = [(f"euhedral_q{bits}_prefill", ((rows + 31) // 32) * prefill_tiles),
                  (f"euhedral_q{bits}_prefill_64", ((rows + 63) // 64) * prefill_tiles)]
        for tile in (1, 2, 4):
            routes.append((f"euhedral_q{bits}_decode_{tile}", ((rows + tile - 1) // tile) * decode_tiles))
        return routes

    def compare_all_routes(self, bits, rows, width, outputs, payload, values, input_offset=0):
        gpu = self.gpu
        with contextlib.ExitStack() as stack:
            raw = struct.pack(f"<{len(values)}H", *values)
            x_base = self.owned(stack, gpu.upload(b"\0" * input_offset + raw))
            x = x_base + input_offset
            w = self.owned(stack, gpu.upload(payload))
            decode_ref = self.run_kernel(stack, f"reference_q{bits}_decode",
                                         rows * ((outputs + 7) // 8), x, w, rows, width, outputs)
            prefill_ref = self.run_kernel(stack, f"reference_q{bits}_prefill",
                                          ((rows + 31) // 32) * ((outputs + 31) // 32), x, w, rows, width, outputs)
            self.assert_written(decode_ref)
            self.assert_written(prefill_ref)
            for name, grid in self.routes(bits, rows, outputs):
                with self.subTest(bits=bits, rows=rows, width=width, outputs=outputs,
                                  offset=input_offset, kernel=name):
                    actual = self.run_kernel(stack, name, grid, x, w, rows, width, outputs)
                    self.assertEqual(actual, decode_ref if "decode" in name else prefill_ref)
            return decode_ref, prefill_ref

    def test_every_route_matches_the_original_kernels_bitwise(self):
        cases = [(1, 128, 1), (2, 128, 9), (3, 256, 17), (4, 384, 33), (5, 128, 40),
                 (9, 256, 8), (31, 128, 32), (33, 256, 35), (64, 128, 64), (65, 384, 70),
                 (97, 256, 37), (130, 512, 66)]
        for bits in (4, 5):
            for rows, width, outputs in cases:
                payload = make_weights(self.rng, bits, width, outputs)
                values = [to_bf16(self.rng.uniform(-2, 2)) for _ in range(rows * width)]
                self.compare_all_routes(bits, rows, width, outputs, payload, values)

    def test_special_values_match_the_original_kernels_bitwise(self):
        special = [0x3F80, 0xBF00, 0x7FC1, 0xFFC3, 0x7F80, 0xFF80, 0x0001, 0x8000]
        scales = [0x3555, 0xB555, 0x0001, 0x8000, 0x7BFF, 0x7C00, 0x7E11, 0xFE11]
        for bits in (4, 5):
            for rows, width, outputs in [(1, 128, 9), (4, 256, 33), (65, 128, 35)]:
                payload = make_weights(self.rng, bits, width, outputs, lambda i: scales[i % len(scales)])
                values = [special[i % len(special)] for i in range(rows * width)]
                self.compare_all_routes(bits, rows, width, outputs, payload, values)

    def test_two_byte_aligned_input_takes_sequential_staging(self):
        for bits in (4, 5):
            rows, width, outputs = 67, 256, 35
            payload = make_weights(self.rng, bits, width, outputs)
            values = [to_bf16(self.rng.uniform(-2, 2)) for _ in range(rows * width)]
            aligned = self.compare_all_routes(bits, rows, width, outputs, payload, values)
            shifted = self.compare_all_routes(bits, rows, width, outputs, payload, values, input_offset=2)
            self.assertEqual(aligned, shifted)

    def test_routes_agree_with_independent_cpu_dequantization(self):
        for bits in (4, 5):
            rows, width, outputs = 5, 256, 12
            payload = make_weights(self.rng, bits, width, outputs)
            values = [to_bf16(self.rng.uniform(-2, 2)) for _ in range(rows * width)]
            decode_ref, prefill_ref = self.compare_all_routes(bits, rows, width, outputs, payload, values)
            expected = cpu_reference(payload, bits, values, rows, width, outputs)
            for label, output in (("decode", decode_ref), ("prefill", prefill_ref)):
                got = [bf16_value(v) for v in struct.unpack(f"<{rows * outputs}H", output)]
                for i, (want, actual) in enumerate(zip(expected, got)):
                    with self.subTest(bits=bits, route=label, index=i):
                        self.assertLessEqual(abs(actual - want), 1e-2 * max(1.0, abs(want)))


@unittest.skipIf(NVRTC is None, f"CUDA probes unavailable: {SKIP_REASON}")
class ScalarQuantizedKernelTest(unittest.TestCase):
    """The scalar fallback strides over outputs when the grid is capped."""

    def test_capped_grid_matches_one_block_per_output_bitwise(self):
        rng = random.Random(0x5CA1)
        gpu = Gpu((HERE.parent / "src/qwen_layer_linear.cu").read_bytes())
        try:
            for bits in (4, 5):
                rows, width, outputs = 7, 256, 37
                payload = make_weights(rng, bits, width, outputs)
                values = [to_bf16(rng.uniform(-2, 2)) for _ in range(rows * width)]
                x = gpu.upload(struct.pack(f"<{len(values)}H", *values))
                w_base = gpu.upload(b"\0\0" + payload)
                results = {}
                try:
                    for grid in (rows * outputs, 5, 1):
                        y = gpu.zeros(rows * outputs * 2, fill=SENTINEL)
                        try:
                            gpu.launch("euhedral_linear_quantized_bf16", grid,
                                       [C.c_uint64(x), C.c_uint64(w_base + 2), C.c_uint64(y), C.c_uint(rows),
                                        C.c_uint(width), C.c_uint(outputs), C.c_uint(bits)])
                            results[grid] = gpu.download(y, rows * outputs * 2)
                        finally:
                            gpu.free(y)
                finally:
                    gpu.free(w_base)
                    gpu.free(x)
                full = results[rows * outputs]
                self.assertNotIn(SENTINEL * 0x101, struct.unpack(f"<{rows * outputs}H", full))
                self.assertEqual(results[5], full)
                self.assertEqual(results[1], full)
                expected = cpu_reference(payload, bits, values, rows, width, outputs)
                got = [bf16_value(v) for v in struct.unpack(f"<{rows * outputs}H", full)]
                for want, actual in zip(expected, got):
                    self.assertLessEqual(abs(actual - want), 1e-2 * max(1.0, abs(want)))
        finally:
            gpu.close()


if __name__ == "__main__":
    unittest.main()
