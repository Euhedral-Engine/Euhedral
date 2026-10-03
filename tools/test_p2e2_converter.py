from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from euhedral_artifacts import edrl, q3_p2e2 as p2e2  # noqa: E402



def row_split(codes: np.ndarray, rng: np.random.Generator) -> bytes:
    rows, k = codes.shape
    groups = k // 64
    out = bytearray(p2e2.row_split_q3_size(rows, k))
    planes = p2e2.pack_q3(codes)
    out[: len(planes)] = planes
    scale_offset = edrl.align_up(rows * groups * 24, 256)
    scales = rng.integers(0, 1 << 15, rows * groups, dtype=np.uint16).astype("<u2").tobytes()
    out[scale_offset:scale_offset + len(scales)] = scales
    return bytes(out)


class P2e2ConverterTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0x9E2)

    def realistic(self, rows, k):
        return self.rng.choice(np.arange(-3, 4, dtype=np.int8), size=(rows, k),
                               p=[0.02, 0.08, 0.23, 0.34, 0.23, 0.08, 0.02])

    def round_trip(self, codes):
        rows, k = codes.shape
        source = row_split(codes, self.rng)
        encoded = p2e2.encode(source, rows, k)
        self.assertEqual(p2e2.decode(encoded, rows, k), source)
        units = int((np.abs(codes) >= 2).sum())
        self.assertEqual(len(encoded), p2e2.p2e2_size(rows, k, units))
        return source, encoded

    def test_q3_packing_round_trips(self):
        codes = self.realistic(5, 1024)
        source = p2e2.pack_q3(codes)
        self.assertTrue(np.array_equal(p2e2.unpack_q3(np.frombuffer(source, np.uint8), 5, 1024), codes))

    def test_realistic_codes_round_trip_and_shrink(self):
        source, encoded = self.round_trip(self.realistic(37, 3072))
        self.assertLess(len(encoded), len(source) * 0.86)

    def test_extreme_rows_round_trip(self):
        k = 2048
        codes = np.stack([
            np.full(k, -3, np.int8),
            np.full(k, 3, np.int8),
            np.zeros(k, np.int8),
            np.full(k, -1, np.int8),
            np.tile(np.array([-3, -2, -1, 0, 1, 2, 3], np.int8), k // 7 + 1)[:k],
            np.where(np.arange(k) % 32 < 17, 2, 0).astype(np.int8),
        ])
        self.round_trip(codes)

    def test_rows_spanning_conversion_chunks(self):
        original = p2e2.ROW_CHUNK_CODES
        p2e2.ROW_CHUNK_CODES = 3 * 1024
        try:
            self.round_trip(self.realistic(11, 1024))
        finally:
            p2e2.ROW_CHUNK_CODES = original

    def test_primary_and_payload_bit_layout(self):
        codes = np.zeros((1, 1024), np.int8)
        codes[0, :6] = [-1, 0, 1, 2, -3, 3]
        codes[0, 17] = -2
        encoded = np.frombuffer(p2e2.encode(row_split(codes, self.rng), 1, 1024), np.uint8)
        primary = encoded[:256].view("<u4")
        self.assertEqual(primary[0] & 0xFFF, 0 | 1 << 2 | 2 << 4 | 3 << 6 | 3 << 8 | 3 << 10)
        self.assertEqual((int(primary[1]) >> 2) & 3, 3)
        _, _, payload_offset = p2e2.p2e2_offsets(1, 1024)
        first = int(encoded[payload_offset:payload_offset + 4].view("<u4")[0])
        self.assertEqual([(first >> (2 * i)) & 3 for i in range(4)], [2, 0, 3, 1])

    def test_row_base_plane_counts_preceding_big_codes(self):
        codes = np.zeros((3, 1024), np.int8)
        codes[0, :5] = 2
        codes[2, :1] = -3
        encoded = np.frombuffer(p2e2.encode(row_split(codes, self.rng), 3, 1024), np.uint8)
        base_offset, _, _ = p2e2.p2e2_offsets(3, 1024)
        self.assertEqual(list(encoded[base_offset:base_offset + 12].view("<u4")), [0, 5, 5])

    def test_minus_four_is_rejected(self):
        codes = np.zeros((1, 1024), np.int8)
        codes[0, 7] = -4
        with self.assertRaises(ValueError):
            p2e2.encode(row_split(codes, self.rng), 1, 1024)

    def test_k_must_be_a_multiple_of_the_slice(self):
        with self.assertRaises(ValueError):
            p2e2.encode(bytes(p2e2.row_split_q3_size(1, 1536)), 1, 1536)

    def test_inconsistent_row_base_is_detected(self):
        _, encoded = self.round_trip(self.realistic(4, 1024))
        corrupt = bytearray(encoded)
        base_offset, _, _ = p2e2.p2e2_offsets(4, 1024)
        corrupt[base_offset + 4] ^= 1
        with self.assertRaises(ValueError):
            p2e2.decode(bytes(corrupt), 4, 1024)

    def test_eligibility_requires_q3_row_split_and_whole_slices(self):
        self.assertTrue(p2e2.eligible((8, 5120), edrl.FORMAT_Q3, edrl.LAYOUT_ROW_SPLIT))
        self.assertFalse(p2e2.eligible((8, 5120), 6, edrl.LAYOUT_ROW_SPLIT))
        self.assertFalse(p2e2.eligible((8, 4608), edrl.FORMAT_Q3, edrl.LAYOUT_ROW_SPLIT))
        self.assertFalse(p2e2.eligible((8, 5120), edrl.FORMAT_Q3, edrl.LAYOUT_P2E2))


if __name__ == "__main__":
    unittest.main()
