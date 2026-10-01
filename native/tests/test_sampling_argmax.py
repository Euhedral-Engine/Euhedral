"""Device greedy selection against the host argmax: strict comparison in token order, so the lowest
token ID wins among equal maxima, and NaN or negative infinity never win."""
import ctypes as C
import random
import struct
import unittest

from test_q3_primitives import Gpu, NVRTC, SKIP_REASON

NEG_INF, POS_INF, NAN, NEG_ZERO = 0xFF80, 0x7F80, 0x7FC1, 0x8000


def to_float(bits):
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


def host_argmax(row):
    """TokenSampler.argmax over the exact FP32 values of the BF16 row."""
    best, best_logit = -1, float('-inf')
    for token, bits in enumerate(row):
        value = to_float(bits)
        if value > best_logit:  # NaN never compares greater
            best, best_logit = token, value
    return best


@unittest.skipIf(NVRTC is None, str(SKIP_REASON))
class SamplingArgmaxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "sampling/kernels.cu"\n')

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def select(self, row, offset_elements=0):
        data = struct.pack(f'<{offset_elements + len(row)}H', *([0] * offset_elements + row))
        logits = self.gpu.upload(data)
        result = self.gpu.zeros(8, fill=0xA5)
        try:
            self.gpu.launch('euhedral_argmax_bf16', 1,
                            [C.c_uint64(logits + 2 * offset_elements), C.c_uint(len(row)), C.c_uint64(result)],
                            block=1024)
            key = struct.unpack('<Q', self.gpu.download(result, 8))[0]
        finally:
            self.gpu.free(logits)
            self.gpu.free(result)
        return -1 if key == 0 else 0xFFFFFFFF - (key & 0xFFFFFFFF)

    def test_random_rows_match_the_host_argmax(self):
        rng = random.Random(4242)
        for count in (1, 7, 8, 9, 1023, 1024, 8191, 8192, 8193, 248320):
            for offset in (0, 1, 3):
                row = [rng.getrandbits(16) for _ in range(count)]
                # Keep a realistic share of finite logits; random bits are otherwise often NaN.
                row = [bits if (bits & 0x7F80) != 0x7F80 else bits & 0xBFFF for bits in row]
                with self.subTest(count=count, offset=offset):
                    self.assertEqual(self.select(row, offset), host_argmax(row))

    def test_ties_select_the_lowest_token(self):
        rng = random.Random(7)
        row = [rng.randrange(0x3F00, 0x4000) for _ in range(248320)]
        peak = 0x4200
        for token in (248319, 200000, 151000, 9, 8, 0):
            row[token] = peak
            with self.subTest(lowest=token):
                self.assertEqual(self.select(row), token)

    def test_special_values_follow_host_comparison(self):
        cases = {
            'nan never wins': ([NAN, 0x3F80, NAN], 1),
            'positive infinity wins': ([0x3F80, POS_INF, 0x7F00], 1),
            'negative infinity never wins over finite': ([NEG_INF, 0xC000, NEG_INF], 1),
            'signed zeros compare equal': ([NEG_ZERO, 0x0000, 0xBF80], 0),
            'positive zero first': ([0x0000, NEG_ZERO], 0),
            'all negative infinity': ([NEG_INF] * 9, -1),
            'all nan': ([NAN] * 9, -1),
            'nan and negative infinity': ([NAN, NEG_INF] * 5, -1),
        }
        for name, (row, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(host_argmax(row), expected)
                self.assertEqual(self.select(row), expected)


if __name__ == '__main__':
    unittest.main()
