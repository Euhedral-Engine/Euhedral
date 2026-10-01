"""Relaxed (BF16 hi-only) prefill kernels stay close to their exact hi + lo twins."""
import contextlib
import ctypes as C
import random
import struct
import unittest

from test_q3_primitives import Gpu, NVRTC, SKIP_REASON
from test_q45_kernels import make_weights

P, U = C.c_uint64, C.c_uint


def f32(bits):
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


@unittest.skipIf(NVRTC is None, str(SKIP_REASON))
class RelaxedPrefillTest(unittest.TestCase):
    def assert_close(self, exact, relaxed, label):
        a = [f32(v) for v in exact]
        b = [f32(v) for v in relaxed]
        self.assertTrue(all(x == x for x in b), label)
        rms = (sum(v * v for v in a) / len(a)) ** 0.5
        err = (sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)) ** 0.5
        self.assertLess(err, 1e-2 * rms, (label, err, rms))
        self.assertLess(max(abs(x - y) for x, y in zip(a, b)), 0.5 * rms, label)

    def test_q3_prefill_kernels(self):
        gpu = Gpu(b'#include "q3/kernels.cu"\n')
        rng = random.Random(1409)
        try:
            for rows, width, outputs in [(33, 512, 64), (100, 1024, 96), (256, 512, 64)]:
                with contextlib.ExitStack() as stack:
                    groups = outputs * (width // 64)
                    scale = (groups * 24 + 255) & ~255
                    data = rng.randbytes(groups * 24) + bytes(scale - groups * 24) + b''.join(
                        struct.pack('<H', rng.randrange(0x1c00, 0x2800) | (rng.randrange(2) << 15)) for _ in range(groups))
                    w = gpu.upload(data); stack.callback(gpu.free, w)
                    x = gpu.upload(struct.pack(f'<{rows * width}H', *[rng.randrange(0x3c00, 0x4000) | (rng.randrange(2) << 15)
                                                                   for _ in range(rows * width)]))
                    stack.callback(gpu.free, x)
                    for name, tile in [('euhedral_q3_prefill', 32), ('euhedral_q3_prefill_s104', 32),
                                       ('euhedral_q3_prefill_64', 64), ('euhedral_q3_prefill_64_k32_cb', 64)]:
                        result = []
                        for symbol in (name + '_exact', name):
                            y = gpu.zeros(rows * outputs * 2, fill=0xa5); stack.callback(gpu.free, y)
                            gpu.launch(symbol, ((rows + tile - 1) // tile) * (outputs // 32),
                                       [P(x), P(w), P(y), U(rows), U(width), U(outputs), C.c_ulonglong(scale)])
                            result.append(struct.unpack(f'<{rows * outputs}H', gpu.download(y, rows * outputs * 2)))
                        self.assert_close(result[0], result[1], (name, rows, width, outputs))
        finally:
            gpu.close()

    def test_q45_prefill_kernels(self):
        gpu = Gpu(b'#include "q45/kernels.cu"\n')
        rng = random.Random(1423)
        scales = lambda i: rng.randrange(0x1c00, 0x2800) | (rng.randrange(2) << 15)
        try:
            for bits in (4, 5):
                for rows, width, outputs in [(33, 256, 64), (100, 512, 96)]:
                    with contextlib.ExitStack() as stack:
                        w = gpu.upload(make_weights(rng, bits, width, outputs, scales)); stack.callback(gpu.free, w)
                        x = gpu.upload(struct.pack(f'<{rows * width}H', *[rng.randrange(0x3c00, 0x4000) | (rng.randrange(2) << 15)
                                                                       for _ in range(rows * width)]))
                        stack.callback(gpu.free, x)
                        for name, tile in [(f'euhedral_q{bits}_prefill', 32), (f'euhedral_q{bits}_prefill_64', 64)]:
                            result = []
                            for symbol in (name + '_exact', name):
                                y = gpu.zeros(rows * outputs * 2, fill=0xa5); stack.callback(gpu.free, y)
                                gpu.launch(symbol, ((rows + tile - 1) // tile) * ((outputs + 31) // 32),
                                           [P(x), P(w), P(y), U(rows), U(width), U(outputs)])
                                result.append(struct.unpack(f'<{rows * outputs}H', gpu.download(y, rows * outputs * 2)))
                            self.assert_close(result[0], result[1], (name, rows, width, outputs))
            # The 128 x 64 tile engine (ffn/down.cuh with the Q4/Q5 producer of ffn/formats.cuh) stages
            # the same BF16 hi weights and accumulates K16 MMA steps in the same order as the relaxed
            # 64 x 32 kernel, so on partial row and column tiles it matches that kernel bit for bit.
            for bits in (4, 5):
                for rows, width, outputs in [(64, 256, 64), (100, 512, 96), (300, 1024, 160)]:
                    with contextlib.ExitStack() as stack:
                        w = gpu.upload(make_weights(rng, bits, width, outputs, scales)); stack.callback(gpu.free, w)
                        x = gpu.upload(struct.pack(f'<{rows * width}H', *[rng.randrange(0x3c00, 0x4000) | (rng.randrange(2) << 15)
                                                                       for _ in range(rows * width)]))
                        stack.callback(gpu.free, x)
                        def run(symbol, grid):
                            y = gpu.zeros(rows * outputs * 2, fill=0xa5); stack.callback(gpu.free, y)
                            gpu.launch(symbol, grid, [P(x), P(w), P(y), U(rows), U(width), U(outputs)])
                            return struct.unpack(f'<{rows * outputs}H', gpu.download(y, rows * outputs * 2))
                        relaxed = run(f'euhedral_q{bits}_prefill_64', ((rows + 63) // 64) * ((outputs + 31) // 32))
                        wide = run(f'euhedral_q{bits}_prefill_128x64', ((rows + 127) // 128) * ((outputs + 63) // 64))
                        self.assertNotIn(0xa5a5, wide)
                        self.assertEqual(relaxed, wide, (bits, rows, width, outputs))
            with contextlib.ExitStack() as stack:
                rows, width, q4_out, q5_out = 100, 256, 64, 96
                w4 = gpu.upload(make_weights(rng, 4, width, q4_out, scales)); stack.callback(gpu.free, w4)
                w5 = gpu.upload(make_weights(rng, 5, width, q5_out, scales)); stack.callback(gpu.free, w5)
                x = gpu.upload(struct.pack(f'<{rows * width}H', *[rng.randrange(0x3c00, 0x4000) for _ in range(rows * width)]))
                stack.callback(gpu.free, x)
                tiles = (rows + 63) // 64
                grid = tiles * (q4_out // 32) + tiles * (q5_out // 32)
                outs = []
                for symbol in ('euhedral_q45_prefill_64_grouped_exact', 'euhedral_q45_prefill_64_grouped'):
                    y4 = gpu.zeros(rows * q4_out * 2, fill=0xa5); stack.callback(gpu.free, y4)
                    y5 = gpu.zeros(rows * q5_out * 2, fill=0xa5); stack.callback(gpu.free, y5)
                    gpu.launch(symbol, grid, [P(x), P(w4), P(y4), P(w5), P(y5), U(rows), U(width), U(q4_out), U(q5_out)])
                    outs.append([struct.unpack(f'<{rows * n}H', gpu.download(b, rows * n * 2)) for b, n in ((y4, q4_out), (y5, q5_out))])
                for index in range(2):
                    self.assert_close(outs[0][index], outs[1][index], ('grouped', index))
        finally:
            gpu.close()


if __name__ == '__main__':
    unittest.main()
