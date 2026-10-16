"""Trellis (EXL3) linear kernels (native/src/trellis/) against the NumPy decoder and a float64 linear: golden tiles,
device decode, the prepass bit for bit, the tensor-core route, row exactness, the prefill route and the scalar
reference. Kernel times are printed by test_kernel_times (informational, nothing asserted)."""

import contextlib
import ctypes as C
import json
from pathlib import Path
import re
import sys
import time
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from gpu_harness import Gpu, NVRTC, SKIP_REASON

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "quantization/src/test/resources/trellis-fixture"
EXTREMES = Path(__file__).resolve().parent / "fixtures" / "trellis-activation-extremes.json"
UNAVAILABLE = (f"CUDA probes unavailable: {SKIP_REASON}" if NVRTC is None
               else "NumPy unavailable" if np is None else None)
if np is not None:
    import trellis_reference as ref

TARGET_CTAS = 140  # the default of euhedral.trellis.target-ctas
BITS = {2.5: 40, 3.0: 48, 3.5: 56, 4.0: 64}


def ptr(value):
    return C.c_uint64(value)


def i32(value):
    return C.c_int(int(value))


class Buffers(contextlib.ExitStack):
    """Device buffers freed on exit."""

    def __init__(self, gpu):
        super().__init__()
        self.gpu = gpu

    def put(self, array):
        address = self.gpu.upload(np.ascontiguousarray(array).tobytes())
        self.callback(self.gpu.free, address)
        return address

    def empty(self, size, fill=0):
        address = self.gpu.zeros(size, fill)
        self.callback(self.gpu.free, address)
        return address

    def get(self, address, dtype, shape):
        count = int(np.prod(shape))
        data = self.gpu.download(address, count * np.dtype(dtype).itemsize)
        return np.frombuffer(data, dtype).reshape(shape).copy()


def random_object(rng, k, n, w, segments=1, scale=1.0):
    trellis = rng.integers(-32768, 32768, size=(k // 16, n // 16, w), dtype=np.int16)
    suh = (rng.choice([-1.0, 1.0], size=(segments, k)) * rng.uniform(0.5, 2.0, size=(segments, k))).astype(np.float16)
    svh = (rng.uniform(0.01, 1.0, size=n) * scale).astype(np.float16)
    return trellis, suh, svh


def random_x(rng, rows, k):
    return ref.bf16_bits(rng.standard_normal((rows, k)).astype(np.float32))


class DeviceLinear:
    """One uploaded trellis linear and its three routes."""

    def __init__(self, gpu, trellis, suh, svh, ends, splits):
        self.gpu, self.ends, self.splits = gpu, list(ends), splits
        self.kt, self.nt, self.w = trellis.shape
        self.k, self.n, self.s = self.kt * 16, self.nt * 16, len(self.ends)
        self.buffers = Buffers(gpu)
        self.d_words = self.buffers.put(trellis.astype("<i2"))
        self.d_suh = self.buffers.put(suh.astype("<f2"))
        self.d_svh = self.buffers.put(svh.astype("<f2"))
        self.d_ends = self.buffers.put(np.array(self.ends, dtype="<i4"))

    def close(self):
        self.buffers.close()

    def prepass(self, buffers, d_x, rows, bf16):
        d_act = buffers.empty(self.s * rows * self.k * 2)
        self.gpu.launch("euhedral_trellis_prepass", (self.k // 128, rows, self.s),
                        [ptr(d_x), ptr(self.d_suh), ptr(d_act), i32(rows), i32(self.k), i32(self.s), i32(bf16)],
                        block=128)
        return d_act

    def mma(self, x_bits, activations=False):
        """prepass(fp16) + mma (+ epilogue): BF16 bits [rows, n] (and the FP16 activations)."""
        rows = x_bits.shape[0]
        with Buffers(self.gpu) as buffers:
            d_act = self.prepass(buffers, buffers.put(x_bits.astype("<u2")), rows, 0)
            d_partials = buffers.empty(self.splits * rows * self.n * 4 if self.splits > 1 else 4)
            d_out = buffers.empty(rows * self.n * 2)
            self.gpu.launch("euhedral_trellis_mma", (self.n // 128, self.splits),
                            [ptr(d_act), ptr(self.d_words), ptr(self.d_ends), ptr(self.d_svh), ptr(d_partials),
                             ptr(d_out), i32(rows), i32(self.k), i32(self.n), i32(self.w), i32(self.splits),
                             i32(self.s)], block=256)
            if self.splits > 1:
                self.gpu.launch("euhedral_trellis_epilogue", (self.n // 128, rows),
                                [ptr(d_partials), i32(self.splits), ptr(self.d_svh), ptr(d_out), i32(rows),
                                 i32(self.n)], block=128)
            out = buffers.get(d_out, "<u2", (rows, self.n))
            if activations:
                return out, buffers.get(d_act, "<u2", (self.s, rows, self.k))
            return out

    def reference(self, x_bits):
        rows = x_bits.shape[0]
        with Buffers(self.gpu) as buffers:
            d_x = buffers.put(x_bits.astype("<u2"))
            d_out = buffers.empty(rows * self.n * 2)
            self.gpu.launch("euhedral_trellis_reference", rows * self.n,
                            [ptr(d_x), ptr(self.d_words), ptr(self.d_suh), ptr(self.d_svh), ptr(self.d_ends),
                             i32(self.s), ptr(d_out), i32(rows), i32(self.k), i32(self.n), i32(self.w)],
                            block=128, shared=4 * self.k)
            return buffers.get(d_out, "<u2", (rows, self.n))

    def prefill(self, q4, x_bits):
        """prepass(bf16) + decode + the qwen4 tensor-core BF16 linear per segment + epilogue_bf16."""
        rows = x_bits.shape[0]
        with Buffers(self.gpu) as buffers:
            d_act = self.prepass(buffers, buffers.put(x_bits.astype("<u2")), rows, 1)
            d_weight = buffers.empty(self.n * self.k * 2)
            self.gpu.launch("euhedral_trellis_decode_bf16", self.kt * self.nt,
                            [ptr(self.d_words), ptr(d_weight), i32(self.k), i32(self.n), i32(self.w)], block=256)
            d_rotated = buffers.empty(rows * self.n * 2)
            start = 0
            for g, end in enumerate(self.ends):
                width = end - start
                q4.launch("euhedral_q4_linear_tc_rows_split_bf16", (width // 8, -(-rows // 64)),
                          [ptr(d_act + g * rows * self.k * 2), ptr(d_weight + start * self.k * 2),
                           ptr(d_rotated + rows * start * 2), C.c_uint(rows), C.c_uint(self.k), C.c_uint(width)],
                          block=128)
                start = end
            d_out = buffers.empty(rows * self.n * 2)
            self.gpu.launch("euhedral_trellis_epilogue_bf16", (self.n // 128, rows),
                            [ptr(d_rotated), ptr(self.d_ends), i32(self.s), ptr(self.d_svh), ptr(d_out), i32(rows),
                             i32(self.n)], block=128)
            return buffers.get(d_out, "<u2", (rows, self.n))


def expected(x_bits, trellis, suh, svh, ends):
    return ref.linear_f64(ref.bf16_values(x_bits), trellis, suh, svh, ends)


def assert_close(test, actual_bits, wanted, relative, floor, label=""):
    """bf16 output bits against float64: |error| <= relative * |wanted| + floor * rms(wanted)."""
    actual = ref.bf16_values(actual_bits).astype(np.float64)
    rms = np.sqrt(np.mean(wanted ** 2))
    error = np.abs(actual - wanted)
    allowed = relative * np.abs(wanted) + floor * rms
    worst = float(np.max(error / allowed))
    test.assertTrue(np.all(np.isfinite(actual)), label)
    test.assertLessEqual(worst, 1.0, f"{label}: worst error is {worst:.2f}x the allowance")


@unittest.skipIf(UNAVAILABLE is not None, UNAVAILABLE or "")
class TrellisKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "trellis/kernels.cu"\n', architecture="sm_120")
        cls.q4 = Gpu(b'#include "qwen4/kernels.cu"\n', architecture="sm_120")
        cls.rng = np.random.default_rng(0x7E11)

    @classmethod
    def tearDownClass(cls):
        cls.q4.close()
        cls.gpu.close()

    # -- decode ---------------------------------------------------------------------------------------------

    def test_golden_tiles(self):
        for bits in ("2", "3", "4", "5", "6", "2.5", "3.5"):
            packed = np.load(FIXTURE / f"tiles-{bits}-packed.npy")
            golden = np.load(FIXTURE / f"tiles-{bits}-out.npy")
            decoded = ref.decode_tiles(packed, packed.shape[1])
            np.testing.assert_array_equal(decoded.view("<u2"), golden.view("<u2"), err_msg=f"{bits} bits")

    def device_decode(self, trellis):
        kt, nt, w = trellis.shape
        with Buffers(self.gpu) as buffers:
            d_words = buffers.put(trellis.astype("<i2"))
            d_out = buffers.empty(kt * 16 * nt * 16 * 2)
            self.gpu.launch("euhedral_trellis_decode_bf16", kt * nt,
                            [ptr(d_words), ptr(d_out), i32(kt * 16), i32(nt * 16), i32(w)], block=256)
            return buffers.get(d_out, "<u2", (nt * 16, kt * 16))

    def test_device_decode_matches_numpy_bitwise(self):
        for bits in ("2", "3", "4", "5", "6", "2.5", "3.5"):
            packed = np.load(FIXTURE / f"tiles-{bits}-packed.npy")
            trellis = packed.reshape(1, 64, packed.shape[1])  # 64 tiles in one tile row: k = 16, n = 1024
            wanted = ref.bf16_bits(ref.rotated_weight(trellis).astype(np.float32).T)
            np.testing.assert_array_equal(self.device_decode(trellis), wanted, err_msg=f"golden {bits}")
        for w in (40, 48, 56, 64):
            trellis = self.rng.integers(-32768, 32768, size=(3, 5, w), dtype=np.int16)
            wanted = ref.bf16_bits(ref.rotated_weight(trellis).astype(np.float32).T)
            np.testing.assert_array_equal(self.device_decode(trellis), wanted, err_msg=f"random w={w}")

    def test_generic_widths_decode(self):
        # Every accepted width: w = 16 K (K = 1..8) and 16 K + 8 (K = 1..7).
        for w in list(range(16, 129, 16)) + list(range(24, 121, 16)):
            trellis = self.rng.integers(-32768, 32768, size=(2, 2, w), dtype=np.int16)
            wanted = ref.bf16_bits(ref.rotated_weight(trellis).astype(np.float32).T)
            np.testing.assert_array_equal(self.device_decode(trellis), wanted, err_msg=f"w={w}")

    # -- the fixture linear ---------------------------------------------------------------------------------

    def test_fixture_linear(self):
        trellis = np.load(FIXTURE / "linear-trellis.npy")
        suh = np.load(FIXTURE / "linear-suh.npy")
        svh = np.load(FIXTURE / "linear-svh.npy")
        loaded = np.load(FIXTURE / "linear-loaded.npy").astype(np.float64)  # [n, k]
        dense = ref.reconstruct_f64(trellis, suh, svh)
        # The stored suh/svh order is pinned by this agreement: swapping them or the Hadamard order breaks it.
        self.assertLess(np.abs(dense - loaded).max(), 2e-4)
        x = random_x(self.rng, 6, 1024)
        wanted = ref.bf16_values(x).astype(np.float64) @ loaded.T
        for splits in (1, 4):
            linear = DeviceLinear(self.gpu, trellis, suh.reshape(1, -1), svh, [512], splits)
            try:
                assert_close(self, linear.mma(x), wanted, 0.006, 0.006, f"fixture S={splits}")
            finally:
                linear.close()

    # -- prepass --------------------------------------------------------------------------------------------

    def test_prepass_is_bitwise_the_float32_butterfly(self):
        k, rows = 384, 5
        for s in (1, 2, 4):
            x = ref.bf16_bits((self.rng.standard_normal((rows, k)) * 30).astype(np.float32))
            suh = (self.rng.choice([-1.0, 1.0], size=(s, k)) * self.rng.uniform(0.25, 3.0, size=(s, k))).astype(np.float16)
            for bf16 in (0, 1):
                with Buffers(self.gpu) as buffers:
                    d_out = buffers.empty(s * rows * k * 2)
                    self.gpu.launch("euhedral_trellis_prepass", (k // 128, rows, s),
                                    [ptr(buffers.put(x)), ptr(buffers.put(suh)), ptr(d_out), i32(rows), i32(k),
                                     i32(s), i32(bf16)], block=128)
                    got = buffers.get(d_out, "<u2", (s, rows, k))
                for g in range(s):
                    product = ref.bf16_values(x) * suh[g].astype(np.float32)
                    rotated = ref.hadamard128_f32(product)
                    wanted = ref.bf16_bits(rotated) if bf16 else ref.fragment_order(rotated.astype(np.float16).view("<u2"))
                    np.testing.assert_array_equal(got[g], wanted, err_msg=f"s={s} segment {g} bf16={bf16}")

    # -- the tensor-core route ------------------------------------------------------------------------------

    def check_mma(self, k, n, ends, bits, splits_list):
        w = BITS[bits]
        trellis, suh, svh = random_object(self.rng, k, n, w, len(ends))
        x = random_x(self.rng, 5, k)
        wanted = expected(x, trellis, suh, svh, ends)
        for splits in splits_list:
            linear = DeviceLinear(self.gpu, trellis, suh, svh, ends, splits)
            try:
                assert_close(self, linear.mma(x), wanted, 0.004, 0.004, f"k={k} n={n} bits={bits} S={splits}")
            finally:
                linear.close()

    def test_mma_against_float64(self):
        for bits in BITS:
            self.check_mma(5120, 256, [256], bits, [1, ref.splits(5120, 256, TARGET_CTAS)])
            self.check_mma(17408, 128, [128], bits, [1, ref.splits(17408, 128, TARGET_CTAS)])

    def test_mma_with_segments_against_float64(self):
        for bits in BITS:
            self.check_mma(5120, 7168, [6144, 7168], bits, [1, ref.splits(5120, 7168, TARGET_CTAS)])

    def test_mma_one_row_wide(self):
        # S > 1 splits odd tile counts unevenly: 17408 / 16 = 1088 tiles over 7 splits.
        self.check_mma(17408, 256, [128, 256], 3.0, [7])

    # -- row exactness --------------------------------------------------------------------------------------

    def test_rows_are_bitwise_independent(self):
        k, n, w = 5120, 256, 48
        trellis, suh, svh = random_object(self.rng, k, n, w)
        counts = list(range(1, 34)) + [64, 65]
        x = random_x(self.rng, max(counts), k)
        for splits in (1, ref.splits(k, n, TARGET_CTAS)):
            linear = DeviceLinear(self.gpu, trellis, suh, svh, [n], splits)
            try:
                singles = np.concatenate([linear.mma(x[r:r + 1]) for r in range(max(counts))])
                for rows in counts:
                    np.testing.assert_array_equal(linear.mma(x[:rows]), singles[:rows], err_msg=f"rows={rows} S={splits}")
            finally:
                linear.close()

    def test_rows_of_mixed_magnitudes_are_bitwise_independent(self):
        # Every m16 tile mixes rows from 1e-3 to 3e3: a row's bits must not depend on the other rows of its tile.
        k, n, w = 5120, 256, 48
        trellis, suh, svh = random_object(self.rng, k, n, w)
        rows = 32
        scale = np.geomspace(1e-3, 3e3, rows)[self.rng.permutation(rows)]
        x = ref.bf16_bits((self.rng.standard_normal((rows, k)) * scale[:, None]).astype(np.float32))
        for splits in (1, ref.splits(k, n, TARGET_CTAS)):
            linear = DeviceLinear(self.gpu, trellis, suh, svh, [n], splits)
            try:
                singles = np.concatenate([linear.mma(x[r:r + 1]) for r in range(rows)])
                np.testing.assert_array_equal(linear.mma(x), singles, err_msg=f"S={splits}")
            finally:
                linear.close()

    def test_rows_are_independent_with_segments(self):
        k, n, w = 1024, 512, 56
        trellis, suh, svh = random_object(self.rng, k, n, w, 3)
        x = random_x(self.rng, 40, k)
        linear = DeviceLinear(self.gpu, trellis, suh, svh, [128, 384, 512], 2)
        try:
            singles = np.concatenate([linear.mma(x[r:r + 1]) for r in range(40)])
            np.testing.assert_array_equal(linear.mma(x), singles)
        finally:
            linear.close()

    # -- prefill route and reference ------------------------------------------------------------------------

    def test_prefill_route_against_float64(self):
        for bits in (3.0, 3.5):
            k, ends = 2048, [128, 384, 640]
            trellis, suh, svh = random_object(self.rng, k, 640, BITS[bits], len(ends))
            x = random_x(self.rng, 70, k)
            linear = DeviceLinear(self.gpu, trellis, suh, svh, ends, 1)
            try:
                assert_close(self, linear.prefill(self.q4, x), expected(x, trellis, suh, svh, ends), 0.02, 0.02,
                             f"prefill bits={bits}")
            finally:
                linear.close()

    def test_reference_against_float64(self):
        for bits, ends in ((3.0, [128, 256]), (2.5, [256]), (4.0, [128, 256])):
            k, n = 1024, 256
            trellis, suh, svh = random_object(self.rng, k, n, BITS[bits], len(ends))
            x = random_x(self.rng, 3, k)
            linear = DeviceLinear(self.gpu, trellis, suh, svh, ends, 1)
            try:
                assert_close(self, linear.reference(x), expected(x, trellis, suh, svh, ends), 0.004, 1e-4,
                             f"reference bits={bits}")
            finally:
                linear.close()

    def test_reference_agrees_with_the_tensor_core_route(self):
        k, n = 1024, 256
        trellis, suh, svh = random_object(self.rng, k, n, 48, 2)
        x = random_x(self.rng, 4, k)
        linear = DeviceLinear(self.gpu, trellis, suh, svh, [128, 256], 2)
        try:
            assert_close(self, linear.mma(x), ref.bf16_values(linear.reference(x)).astype(np.float64), 0.006, 0.006,
                         "mma vs reference")
        finally:
            linear.close()

    # -- FP16 range -----------------------------------------------------------------------------------------

    def prepass_extreme(self, max_x, max_suh, k=1024):
        """Every 128-block holds sign-aligned values of magnitude max_x * max_suh (aligned with a Hadamard row), the
        worst case of the butterfly; the FP16 prepass output must be finite."""
        rows = 4
        signs = np.empty((rows, k))
        for r in range(rows):
            for b in range(k // 128):
                j = int(self.rng.integers(0, 128))
                signs[r, b * 128:(b + 1) * 128] = [(-1) ** bin(i & j).count("1") for i in range(128)]
        x = ref.bf16_bits(signs * max_x)
        suh = np.full((1, k), max_suh, dtype=np.float16)
        with Buffers(self.gpu) as buffers:
            d_out = buffers.empty(rows * k * 2)
            self.gpu.launch("euhedral_trellis_prepass", (k // 128, rows, 1),
                            [ptr(buffers.put(x)), ptr(buffers.put(suh)), ptr(d_out), i32(rows), i32(k), i32(1), i32(0)],
                            block=128)
            out = buffers.get(d_out, "<u2", (rows, k)).view(np.float16)
        self.assertTrue(np.all(np.isfinite(out)), f"max|x|={max_x} max|suh|={max_suh}")
        self.assertLess(np.abs(out.astype(np.float32)).max(), 65504.0)

    def test_fp16_range_synthetic_worst_case(self):
        self.prepass_extreme(4096.0, 1.0)  # 4096 in every channel with suh = 1: 11.31 * 4096 = 46341

    def test_fp16_range_cliff(self):
        # A block of equal-signed inputs reaches sqrt(128) * max|x|: finite below 65504 / sqrt(128), inf above.
        limit = 65504.0 / np.sqrt(128.0)
        self.prepass_extreme(limit * 0.99, 1.0)
        with self.assertRaises(AssertionError):
            self.prepass_extreme(limit * 1.02, 1.0)

    def test_fp16_range_recorded_extremes(self):
        if not EXTREMES.is_file():
            return  # recorded from a running engine; the synthetic case above covers the shape until then
        # Recorded by TrellisActivationRangeCudaIntegrationTest: the largest prepass output each linear met on real
        # activations, which must fit FP16 with room to spare.
        cases = json.loads(EXTREMES.read_text())["cases"]
        self.assertTrue(cases)
        for case in cases:
            self.assertLess(float(case["max_abs_prepass"]), 65504.0 / 2, case["linear"])

    # -- table ----------------------------------------------------------------------------------------------

    def test_every_table_entry_is_a_kernel_of_the_module(self):
        source = (ROOT / "native/src/host/linear_kernels.c").read_text()
        names = re.findall(r'\{"(euhedral_trellis_\w+)", \d+(?: \* \d+)?, "trellis/kernels.cu"\}', source)
        self.assertEqual(7, len(names))
        for name in names:
            function = C.c_void_p()
            self.assertEqual(0, self.gpu.function(C.byref(function), self.gpu.module, name.encode()), name)

    def test_kernel_times(self):
        for k, n, ends in ((5120, 7168, [6144, 7168]), (5120, 256, [256]), (17408, 5120, [5120])):
            for bits in (3.0, 3.5):
                trellis, suh, svh = random_object(self.rng, k, n, BITS[bits], len(ends))
                splits = ref.splits(k, n, TARGET_CTAS)
                linear = DeviceLinear(self.gpu, trellis, suh, svh, ends, splits)
                try:
                    for rows in (1, 4, 32):
                        x = random_x(self.rng, rows, k)
                        with Buffers(self.gpu) as buffers:
                            d_act = linear.prepass(buffers, buffers.put(x), rows, 0)
                            d_partials = buffers.empty(splits * rows * n * 4 if splits > 1 else 4)
                            d_out = buffers.empty(rows * n * 2)
                            arguments = [ptr(d_act), ptr(linear.d_words), ptr(linear.d_ends), ptr(linear.d_svh),
                                         ptr(d_partials), ptr(d_out), i32(rows), i32(k), i32(n), i32(linear.w),
                                         i32(splits), i32(len(ends))]
                            repeats = 50
                            for _ in range(5):
                                self.gpu.launch("euhedral_trellis_mma", (n // 128, splits), arguments, block=256)
                            begin = time.perf_counter()
                            for _ in range(repeats):
                                self.gpu.launch("euhedral_trellis_mma", (n // 128, splits), arguments,
                                                synchronize=False, block=256)
                            self.gpu.sync()
                            mma = (time.perf_counter() - begin) / repeats * 1e6
                        print(f"mma k={k} n={n} bits={bits} rows={rows} S={splits}: {mma:.1f} us", file=sys.stderr)
                finally:
                    linear.close()


if __name__ == "__main__":
    unittest.main()
