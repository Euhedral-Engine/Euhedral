"""Device greedy selection against the host argmax: strict comparison in token order, so the lowest
token ID wins among equal maxima, and NaN or negative infinity never win."""
import ctypes as C
import math
import random
import struct
import unittest

from gpu_harness import Gpu, NVRTC, SKIP_REASON

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

    def select_scored(self, row, offset_elements=0):
        data = struct.pack(f'<{offset_elements + len(row)}H', *([0] * offset_elements + row))
        logits = self.gpu.upload(data)
        result = self.gpu.zeros(16, fill=0xA5)
        try:
            self.gpu.launch('euhedral_argmax_logprob_bf16', 1,
                            [C.c_uint64(logits + 2 * offset_elements), C.c_uint(len(row)), C.c_uint64(result)],
                            block=1024)
            key, log_probability = struct.unpack('<Qf', self.gpu.download(result, 12))
        finally:
            self.gpu.free(logits)
            self.gpu.free(result)
        return (-1 if key == 0 else 0xFFFFFFFF - (key & 0xFFFFFFFF)), log_probability

    def selected_token(self, key, ids):
        """euhedral_selected_token: the token a selection key chose, through shortlist row ids (or none)."""
        key_address = self.gpu.upload(struct.pack('<Q', key))
        ids_address = self.gpu.upload(struct.pack(f'<{len(ids)}i', *ids)) if ids else 0
        token = self.gpu.zeros(4, fill=0xA5)
        try:
            self.gpu.launch('euhedral_selected_token', 1,
                            [C.c_uint64(key_address), C.c_uint64(ids_address), C.c_uint64(token)], block=1)
            return struct.unpack('<i', self.gpu.download(token, 4))[0]
        finally:
            self.gpu.free(key_address)
            if ids_address:
                self.gpu.free(ids_address)
            self.gpu.free(token)

    def test_selected_token_maps_the_selected_row_through_the_head_ids(self):
        ids = [1000 + 7 * row for row in range(32)]
        row = [NEG_INF] * 32
        row[19] = 0x3F80
        for offset in (0, 3):
            data = struct.pack(f'<{offset + 32}H', *([0] * offset + row))
            logits = self.gpu.upload(data)
            result = self.gpu.zeros(8)
            try:
                self.gpu.launch('euhedral_argmax_bf16', 1,
                                [C.c_uint64(logits + 2 * offset), C.c_uint(32), C.c_uint64(result)], block=1024)
                key = struct.unpack('<Q', self.gpu.download(result, 8))[0]
            finally:
                self.gpu.free(logits)
                self.gpu.free(result)
            self.assertEqual(ids[19], self.selected_token(key, ids))
            self.assertEqual(19, self.selected_token(key, None), "without ids the row is the token")
        self.assertEqual(ids[0], self.selected_token(0, ids), "no selectable logit: row 0")

    def test_scored_selection_matches_the_host_argmax_and_log_softmax(self):
        rng = random.Random(99)
        for count in (1, 9, 1024, 8193, 131072, 248320):
            for offset in (0, 3):
                # Logits of a trained head: mostly small, one or a few large.
                row = [struct.unpack('<I', struct.pack('<f', rng.gauss(0.0, 3.0)))[0] >> 16 for _ in range(count)]
                for _ in range(rng.randrange(1, 4)):
                    row[rng.randrange(count)] = struct.unpack('<I', struct.pack('<f', rng.uniform(5.0, 30.0)))[0] >> 16
                values = [to_float(bits) for bits in row]
                peak = max(values)
                expected = -math.log(sum(math.exp(v - peak) for v in values))
                with self.subTest(count=count, offset=offset):
                    token, log_probability = self.select_scored(row, offset)
                    self.assertEqual(token, host_argmax(row))
                    self.assertAlmostEqual(log_probability, expected, delta=1e-4 + 1e-5 * abs(expected))

    def test_scored_selection_skips_unselectable_logits(self):
        token, log_probability = self.select_scored([NAN, 0x3F80, NEG_INF, 0x3F80])
        self.assertEqual(token, 1)
        self.assertAlmostEqual(log_probability, -math.log(2.0), places=6)
        token, log_probability = self.select_scored([NAN, NEG_INF] * 5)
        self.assertEqual((token, log_probability), (-1, 0.0))

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
