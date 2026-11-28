"""P2E2 Q3 tensors: the compressed routes must reproduce the row-split routes bit for bit.

The production q3 module and the embedding module are compiled with NVRTC. Tensors are built from
code matrices with tools/euhedral_artifacts/q3_p2e2.py, the reference encoder. Each case covers
the paths a kernel takes on its own data: lanes with 17 to 31 BIG codes split every way between their two
halves (the fast path's edge), lanes with 32, slices whose payload outruns the 32 prefetched words, rows
without BIG codes, and realistic and calibrated code distributions.
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
SENTINEL = 0xA5
UNAVAILABLE = None
if NVRTC is None:
    UNAVAILABLE = f"CUDA probes unavailable: {SKIP_REASON}"
elif np is None:
    UNAVAILABLE = "NumPy unavailable"
p2e2 = None
if np is not None:
    # The reference encoder (tools/euhedral_artifacts/q3_p2e2.py) needs NumPy.
    sys.path.insert(0, str(ROOT / "tools"))
    from euhedral_artifacts import q3_p2e2 as p2e2


def realistic(rng, rows, k):
    return rng.choice(np.arange(-3, 4, dtype=np.int8), size=(rows, k),
                      p=[0.02, 0.08, 0.23, 0.34, 0.23, 0.08, 0.02])


def row_split(codes, rng):
    rows, k = codes.shape
    groups = k // 64
    out = bytearray(p2e2.row_split_q3_size(rows, k))
    planes = p2e2.pack_q3(codes)
    out[:len(planes)] = planes
    scale_offset = p2e2.align_up(rows * groups * 24, 256)
    # Finite FP16 scales of both signs across normal and subnormal exponents.
    scales = rng.integers(0, 0x7C00, rows * groups, dtype=np.uint16) | (rng.integers(0, 2, rows * groups, dtype=np.uint16) << 15)
    out[scale_offset:scale_offset + rows * groups * 2] = scales.astype("<u2").tobytes()
    return bytes(out), scale_offset


def calibrated(rng, rows, k):
    """The code distribution of a calibrated Q3 limited to -3..3: about 31% BIG codes."""
    return rng.choice(np.arange(-3, 4, dtype=np.int8), size=(rows, k),
                      p=[0.06, 0.12, 0.20, 0.24, 0.20, 0.11, 0.07])


def lane_big(rng, k, first, second, every=4):
    """A row in which every `every`-th lane has `first` BIG codes among its first 16 and `second` among its
    last 16, at random positions and values, and the other lanes none, so that each slice's payload fits the
    prefetched window and the dense lanes reach the fast path."""
    codes = rng.choice(np.array([-1, 0, 1], np.int8), size=k)
    for lane in range(0, k // 32, every):
        for half, count in ((0, first), (1, second)):
            spots = lane * 32 + half * 16 + rng.choice(16, size=count, replace=False)
            codes[spots] = rng.choice(np.array([-3, -2, 2, 3], np.int8), size=count)
    return codes


# (first half, second half) BIG counts of the dense lanes, four rows per warp. Rows 4-7 are the fast path's
# edge (31 = 16 + 15 and 15 + 16, then 30) and decode together on it; rows 8-11 put lanes with 32, which must
# take the slow path, beside fast-path rows; rows 12-19 cover totals from 17 to 30.
LANE_SPLITS = [(16, 15), (15, 16), (16, 14), (14, 16),
               (16, 16), (9, 8), (12, 12), (8, 16),
               (16, 1), (1, 16), (13, 4), (10, 10),
               (11, 11), (13, 13), (14, 14), (15, 15)]


def adversarial(rng, rows, k):
    """Rows that drive every decode path: dense BIG lanes, overflowing slices, empty rows."""
    codes = realistic(rng, rows, k)
    codes[0] = 0
    codes[1] = rng.choice(np.array([-3, -2, 2, 3], np.int8), size=k)
    if rows > 2:
        lane_dense = np.where(np.arange(k) % 32 < 17, 2, -1).astype(np.int8)
        codes[2] = lane_dense
    if rows > 3:
        codes[3, : k // 2] = 3
    for row, (first, second) in enumerate(LANE_SPLITS, start=4):
        if row < rows:
            codes[row] = lane_big(rng, k, first, second)
    return codes


@unittest.skipIf(UNAVAILABLE is not None, UNAVAILABLE or "")
class P2e2KernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "q3/kernels.cu"\n')
        cls.embedding = Gpu((ROOT / "native/src/embedding/kernels.cu").read_bytes())
        cls.rng = np.random.default_rng(0x9E2E2)

    @classmethod
    def tearDownClass(cls):
        cls.embedding.close()
        cls.gpu.close()

    def owned(self, stack, gpu, pointer):
        stack.callback(gpu.free, pointer)
        return pointer

    def decode_pair(self, codes):
        rows, k = codes.shape
        source, scale_offset = row_split(codes, self.rng)
        tensor = p2e2.encode(source, rows, k)
        x = (self.rng.standard_normal(k).astype(np.float32) * 3.0)
        xb = (x.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        with contextlib.ExitStack() as stack:
            dx = self.owned(stack, self.gpu, self.gpu.upload(xb))
            dw = self.owned(stack, self.gpu, self.gpu.upload(source))
            dc = self.owned(stack, self.gpu, self.gpu.upload(tensor))
            y0 = self.owned(stack, self.gpu, self.gpu.zeros(rows * 2, fill=SENTINEL))
            y1 = self.owned(stack, self.gpu, self.gpu.zeros(rows * 2, fill=SENTINEL))
            self.gpu.launch("euhedral_q3_decode_contiguous", rows // 16,
                            [C.c_uint64(dx), C.c_uint64(dw), C.c_uint64(y0), C.c_uint(1), C.c_uint(k),
                             C.c_uint(rows), C.c_uint64(scale_offset)])
            self.gpu.launch("euhedral_q3_p2e2_decode", rows // 16,
                            [C.c_uint64(dx), C.c_uint64(dc), C.c_uint64(y1), C.c_uint(k), C.c_uint(rows)])
            return self.gpu.download(y0, rows * 2), self.gpu.download(y1, rows * 2)

    def test_contiguous_rows_twins_are_bitwise_one_row_decode(self):
        """Row-exact speculative verification: each row of euhedral_q3_decode_contiguous_rowsM is bit for
        bit the one-row contiguous kernel's output for that row."""
        rows, k = 48, 5120
        source, scale_offset = row_split(realistic(self.rng, rows, k), self.rng)
        with contextlib.ExitStack() as stack:
            dw = self.owned(stack, self.gpu, self.gpu.upload(source))
            for m in range(2, 9):
                with self.subTest(m=m):
                    x = (self.rng.standard_normal((m, k)).astype(np.float32) * 3.0)
                    xb = (x.view(np.uint32) >> 16).astype(np.uint16)
                    dx = self.owned(stack, self.gpu, self.gpu.upload(xb.tobytes()))
                    dy = self.owned(stack, self.gpu, self.gpu.zeros(m * rows * 2, fill=SENTINEL))
                    self.gpu.launch(f"euhedral_q3_decode_contiguous_rows{m}", rows // 16,
                                    [C.c_uint64(dx), C.c_uint64(dw), C.c_uint64(dy), C.c_uint(m), C.c_uint(k),
                                     C.c_uint(rows), C.c_uint64(scale_offset)])
                    together = self.gpu.download(dy, m * rows * 2)
                    for t in range(m):
                        dxt = self.owned(stack, self.gpu, self.gpu.upload(xb[t].tobytes()))
                        dyt = self.owned(stack, self.gpu, self.gpu.zeros(rows * 2, fill=SENTINEL))
                        self.gpu.launch("euhedral_q3_decode_contiguous", rows // 16,
                                        [C.c_uint64(dxt), C.c_uint64(dw), C.c_uint64(dyt), C.c_uint(1), C.c_uint(k),
                                         C.c_uint(rows), C.c_uint64(scale_offset)])
                        self.assertEqual(together[t * rows * 2:(t + 1) * rows * 2], self.gpu.download(dyt, rows * 2),
                                         f"row {t}")

    def test_decode_matches_contiguous_bitwise(self):
        for rows, k in ((16, 1024), (48, 3072), (32, 5120)):
            for name, codes in (("realistic", realistic(self.rng, rows, k)), ("calibrated", calibrated(self.rng, rows, k)),
                                ("adversarial", adversarial(self.rng, rows, k))):
                with self.subTest(rows=rows, k=k, codes=name):
                    expected, actual = self.decode_pair(codes)
                    self.assertNotIn(b"\xa5\xa5", expected[:2])
                    self.assertEqual(actual, expected)

    def test_rows_kernels_are_bitwise_one_row_decode(self):
        """Rows 2 to 4 of the fused kernels equal the contiguous one-row kernel on the row-split tensor, on
        every decode path (dense BIG lanes, overflowing slices, empty rows)."""
        rows, k = 64, 3072
        for name, codes in (("realistic", realistic(self.rng, rows, k)), ("calibrated", calibrated(self.rng, rows, k)),
                            ("adversarial", adversarial(self.rng, rows, k))):
            source, scale_offset = row_split(codes, self.rng)
            tensor = p2e2.encode(source, rows, k)
            with contextlib.ExitStack() as stack:
                dw = self.owned(stack, self.gpu, self.gpu.upload(source))
                dc = self.owned(stack, self.gpu, self.gpu.upload(tensor))
                for m in range(2, 5):
                    with self.subTest(codes=name, m=m):
                        x = (self.rng.standard_normal((m, k)).astype(np.float32) * 3.0)
                        xb = (x.view(np.uint32) >> 16).astype(np.uint16)
                        dx = self.owned(stack, self.gpu, self.gpu.upload(xb.tobytes()))
                        dy = self.owned(stack, self.gpu, self.gpu.zeros(m * rows * 2, fill=SENTINEL))
                        self.gpu.launch(f"euhedral_q3_p2e2_decode_rows{m}", rows // 16,
                                        [C.c_uint64(dx), C.c_uint64(dc), C.c_uint64(dy), C.c_uint(k), C.c_uint(rows)])
                        together = self.gpu.download(dy, m * rows * 2)
                        for t in range(m):
                            dxt = self.owned(stack, self.gpu, self.gpu.upload(xb[t].tobytes()))
                            dyt = self.owned(stack, self.gpu, self.gpu.zeros(rows * 2, fill=SENTINEL))
                            self.gpu.launch("euhedral_q3_decode_contiguous", rows // 16,
                                            [C.c_uint64(dxt), C.c_uint64(dw), C.c_uint64(dyt), C.c_uint(1), C.c_uint(k),
                                             C.c_uint(rows), C.c_uint64(scale_offset)])
                            self.assertEqual(together[t * rows * 2:(t + 1) * rows * 2], self.gpu.download(dyt, rows * 2),
                                             f"row {t}")

    def test_expand_reproduces_the_row_split_tensor(self):
        for rows, k, name in ((5, 1024, "realistic"), (37, 2048, "adversarial"), (8, 6144, "realistic")):
            with self.subTest(rows=rows, k=k, codes=name):
                codes = realistic(self.rng, rows, k) if name == "realistic" else adversarial(self.rng, rows, k)
                source, _ = row_split(codes, self.rng)
                tensor = p2e2.encode(source, rows, k)
                with contextlib.ExitStack() as stack:
                    dc = self.owned(stack, self.gpu, self.gpu.upload(tensor))
                    out = self.owned(stack, self.gpu, self.gpu.zeros(len(source), fill=0))
                    self.gpu.launch("euhedral_q3_p2e2_expand", (rows + 3) // 4,
                                    [C.c_uint64(dc), C.c_uint64(out), C.c_uint(rows), C.c_uint(k), C.c_uint(0),
                                     C.c_uint(rows)])
                    self.assertEqual(self.gpu.download(out, len(source)), source)
                    # A row range expands to the row-split tensor of exactly those rows.
                    first, count = rows // 3, rows - rows // 3 - 1
                    part, _ = row_split(codes[first:first + count], self.rng)
                    groups = k // 64
                    part = bytearray(part)
                    scale_part = p2e2.align_up(count * groups * 24, 256)
                    scale_all = p2e2.align_up(rows * groups * 24, 256)
                    part[scale_part:scale_part + count * groups * 2] = source[scale_all + first * groups * 2:
                                                                              scale_all + (first + count) * groups * 2]
                    sub = self.owned(stack, self.gpu, self.gpu.zeros(len(part), fill=0))
                    self.gpu.launch("euhedral_q3_p2e2_expand", (count + 3) // 4,
                                    [C.c_uint64(dc), C.c_uint64(sub), C.c_uint(rows), C.c_uint(k), C.c_uint(first),
                                     C.c_uint(count)])
                    self.assertEqual(self.gpu.download(sub, len(part)), bytes(part))

    def test_embedding_matches_row_split_gather_bitwise(self):
        vocab, hidden = 41, 2048
        codes = adversarial(self.rng, vocab, hidden)
        source, scale_offset = row_split(codes, self.rng)
        tensor = p2e2.encode(source, vocab, hidden)
        ids = np.array([0, 1, 2, 3, 40, -1, 41, 7, 7, 19], dtype=np.int32)
        size = len(ids) * hidden * 2
        with contextlib.ExitStack() as stack:
            dids = self.owned(stack, self.embedding, self.embedding.upload(ids.tobytes()))
            dw = self.owned(stack, self.embedding, self.embedding.upload(source))
            dc = self.owned(stack, self.embedding, self.embedding.upload(tensor))
            y0 = self.owned(stack, self.embedding, self.embedding.zeros(size, fill=SENTINEL))
            y1 = self.owned(stack, self.embedding, self.embedding.zeros(size, fill=SENTINEL))
            self.embedding.launch("euhedral_q3_embedding", len(ids) * ((hidden // 64 + 3) // 4),
                                  [C.c_uint64(dids), C.c_uint64(dw), C.c_uint64(y0), C.c_uint(len(ids)),
                                   C.c_uint(vocab), C.c_uint(hidden), C.c_uint64(scale_offset)])
            self.embedding.launch("euhedral_q3_p2e2_embedding", (len(ids) + 3) // 4,
                                  [C.c_uint64(dids), C.c_uint64(dc), C.c_uint64(y1), C.c_uint(len(ids)),
                                   C.c_uint(vocab), C.c_uint(hidden)])
            expected, actual = self.embedding.download(y0, size), self.embedding.download(y1, size)
        self.assertEqual(actual, expected)
        row = hidden * 2
        self.assertEqual(expected[5 * row:7 * row], bytes([SENTINEL]) * 2 * row)


if __name__ == "__main__":
    unittest.main()
