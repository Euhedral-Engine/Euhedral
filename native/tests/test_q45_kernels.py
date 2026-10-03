"""Q4/Q5 kernels (native/src/q45, native/src/reference).

The scalar reference is checked against an independent CPU dequantization; the contiguous decode kernels against
the reference within one BF16 step; their multi-row twins bit for bit against one-row launches.
"""

import contextlib
import ctypes as C
import random
import struct
import unittest

from gpu_harness import Gpu, NVRTC, SKIP_REASON, bf16_value, to_bf16

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
        cls.gpu = Gpu(b'#include "q45/kernels.cu"\n#include "reference/kernels.cu"\n')
        cls.rng = random.Random(0x0545)

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def owned(self, stack, pointer):
        stack.callback(self.gpu.free, pointer)
        return pointer

    def run_kernel(self, stack, name, grid, x, w, rows, width, outputs, bits=None):
        y = self.owned(stack, self.gpu.zeros(rows * outputs * 2, fill=SENTINEL))
        arguments = [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y), C.c_uint(rows), C.c_uint(width), C.c_uint(outputs)]
        if bits is not None:
            arguments.append(C.c_uint(bits))
        self.gpu.launch(name, grid, arguments)
        return self.gpu.download(y, rows * outputs * 2)

    def reference(self, stack, x, w, rows, width, outputs, bits):
        return self.run_kernel(stack, "euhedral_q45_reference", min(rows * outputs, 65535), x, w, rows, width, outputs, bits)

    def assert_within_one_step(self, expected, actual):
        words_e = struct.unpack(f"<{len(expected) // 2}H", expected)
        words_a = struct.unpack(f"<{len(actual) // 2}H", actual)
        scale = max(abs(bf16_value(w)) for w in words_e) + 1e-6
        for index, (e, a) in enumerate(zip(words_e, words_a)):
            near = abs(bf16_value(e) - bf16_value(a)) <= 0.01 * scale
            self.assertTrue(abs(e - a) <= 1 or near, f"index {index}: {bf16_value(e)} vs {bf16_value(a)}")

    def test_reference_matches_the_cpu_dequantization(self):
        for bits in (4, 5):
            for rows, width, outputs in [(1, 128, 8), (2, 256, 5), (3, 192 + 64, 17)]:
                payload = make_weights(self.rng, bits, width, outputs)
                values = [to_bf16(self.rng.uniform(-2, 2)) for _ in range(rows * width)]
                with contextlib.ExitStack() as stack:
                    x = self.owned(stack, self.gpu.upload(struct.pack(f"<{len(values)}H", *values)))
                    w = self.owned(stack, self.gpu.upload(payload))
                    actual = self.reference(stack, x, w, rows, width, outputs, bits)
                expected = cpu_reference(payload, bits, values, rows, width, outputs)
                words = struct.unpack(f"<{rows * outputs}H", actual)
                scale = max(abs(v) for v in expected) + 1e-6
                for index, (word, value) in enumerate(zip(words, expected)):
                    with self.subTest(bits=bits, rows=rows, index=index):
                        self.assertLessEqual(abs(bf16_value(word) - value), 0.01 * scale)

    def test_contiguous_decode_stays_within_one_bf16_step_of_the_reference(self):
        for bits in (4, 5):
            for width, outputs in [(1024, 16), (2048, 40), (5120, 104)]:
                payload = make_weights(self.rng, bits, width, outputs)
                values = [to_bf16(self.rng.uniform(-2, 2)) for _ in range(width)]
                with contextlib.ExitStack() as stack:
                    x = self.owned(stack, self.gpu.upload(struct.pack(f"<{width}H", *values)))
                    w = self.owned(stack, self.gpu.upload(payload))
                    actual = self.run_kernel(stack, f"euhedral_q{bits}_decode_contiguous", outputs // 8, x, w, 1, width, outputs)
                    expected = self.reference(stack, x, w, 1, width, outputs, bits)
                with self.subTest(bits=bits, width=width, outputs=outputs):
                    self.assert_within_one_step(expected, actual)

    def test_special_scales_stay_within_one_bf16_step_of_the_reference(self):
        width, outputs = 1024, 16
        patterns = [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3555, 0xB555, 0x7BFF]
        for bits in (4, 5):
            payload = make_weights(self.rng, bits, width, outputs, lambda i: patterns[i % len(patterns)])
            values = [to_bf16(self.rng.uniform(-2, 2)) for _ in range(width)]
            with contextlib.ExitStack() as stack:
                x = self.owned(stack, self.gpu.upload(struct.pack(f"<{width}H", *values)))
                w = self.owned(stack, self.gpu.upload(payload))
                actual = self.run_kernel(stack, f"euhedral_q{bits}_decode_contiguous", outputs // 8, x, w, 1, width, outputs)
                expected = self.reference(stack, x, w, 1, width, outputs, bits)
            with self.subTest(bits=bits):
                self.assert_within_one_step(expected, actual)

    def test_contiguous_rows_twins_are_bitwise_one_row_decode(self):
        """Each row of euhedral_q{4,5}_decode_contiguous_rowsM is bit for bit the one-row contiguous kernel's
        output for that row."""
        width, outputs = 5120, 104
        for bits in (4, 5):
            with contextlib.ExitStack() as stack:
                payload = make_weights(self.rng, bits, width, outputs,
                                       lambda i: self.rng.randrange(0x2000, 0x2c00) | (self.rng.randrange(2) << 15))
                w = self.owned(stack, self.gpu.upload(payload))
                for m in range(2, 9):
                    with self.subTest(bits=bits, m=m):
                        values = [to_bf16(self.rng.uniform(-2, 2)) for _ in range(m * width)]
                        x = self.owned(stack, self.gpu.upload(struct.pack(f"<{m * width}H", *values)))
                        together = self.run_kernel(stack, f"euhedral_q{bits}_decode_contiguous_rows{m}", outputs // 8,
                                                   x, w, m, width, outputs)
                        for t in range(m):
                            xt = self.owned(stack, self.gpu.upload(struct.pack(f"<{width}H", *values[t * width:(t + 1) * width])))
                            alone = self.run_kernel(stack, f"euhedral_q{bits}_decode_contiguous", outputs // 8,
                                                    xt, w, 1, width, outputs)
                            self.assertEqual(together[t * outputs * 2:(t + 1) * outputs * 2], alone, f"row {t}")

    def test_capped_grid_matches_one_block_per_output_bitwise(self):
        """The reference strides over outputs, so a grid smaller than rows * outputs gives the same bits."""
        bits, rows, width, outputs = 5, 3, 256, 37
        payload = make_weights(self.rng, bits, width, outputs)
        values = [to_bf16(self.rng.uniform(-2, 2)) for _ in range(rows * width)]
        with contextlib.ExitStack() as stack:
            x = self.owned(stack, self.gpu.upload(struct.pack(f"<{len(values)}H", *values)))
            w = self.owned(stack, self.gpu.upload(payload))
            full = self.run_kernel(stack, "euhedral_q45_reference", rows * outputs, x, w, rows, width, outputs, bits)
            capped = self.run_kernel(stack, "euhedral_q45_reference", 7, x, w, rows, width, outputs, bits)
        self.assertEqual(full, capped)


if __name__ == "__main__":
    unittest.main()
