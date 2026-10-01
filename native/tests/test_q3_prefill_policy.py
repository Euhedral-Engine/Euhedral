"""Q3 AUTO prefill shape policy, exercised without CUDA or a device."""
import ctypes
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

HEADER = pathlib.Path(__file__).resolve().parents[1] / 'src'
SOURCE = '''#include "q3_prefill_policy.h"
int decode_wide(unsigned rows, unsigned width, unsigned outputs) {
    return euhedral_q3_decode_wide_shape(rows, width, outputs);
}
int route(int mode, unsigned rows, unsigned width, unsigned outputs) {
    return euhedral_q3_wide_prefill(mode, rows, width, outputs);
}
int wmma64(int mode, unsigned rows, unsigned width, unsigned outputs) {
    return euhedral_q3_wmma_prefill64(mode, rows, width, outputs);
}
int select(int mode, unsigned rows, unsigned width, unsigned outputs, int has64, int has_wmma) {
    return (int)euhedral_q3_select_prefill(mode, rows, width, outputs, has64, has_wmma);
}
int use_k32_cb(int mode, unsigned rows, unsigned width, unsigned outputs) {
#ifdef EUHEDRAL_Q3_HAS_K32_COMPACT_B_POLICY
    return euhedral_q3_use_k32_compact_b_prefill(mode, rows, width, outputs);
#else
    (void)mode; (void)rows; (void)width; (void)outputs;
    return 0;
#endif
}
int select_with_k32_cb(int mode, unsigned rows, unsigned width, unsigned outputs,
        int has_cb, int has64, int has_wmma) {
#ifdef EUHEDRAL_Q3_HAS_K32_COMPACT_B_POLICY
    return (int)euhedral_q3_select_prefill_with_k32_compact_b(
            mode, rows, width, outputs, has_cb, has64, has_wmma);
#else
    (void)has_cb;
    return (int)euhedral_q3_select_prefill(mode, rows, width, outputs, has64, has_wmma);
#endif
}
int select_with_cb_input_alignment(int mode, unsigned rows, unsigned width, unsigned outputs,
        int has_cb, int input_aligned_16, int has64, int has_wmma) {
#ifdef EUHEDRAL_Q3_HAS_CB_ALIGNMENT_POLICY
    return (int)euhedral_q3_select_prefill_for_input(
            mode, rows, width, outputs, has_cb, input_aligned_16, has64, has_wmma);
#else
    (void)mode; (void)rows; (void)width; (void)outputs;
    (void)has_cb; (void)input_aligned_16; (void)has64; (void)has_wmma;
    return -1;
#endif
}
int select_s104(int mode, unsigned rows, unsigned width, unsigned outputs, int has_s104) {
    return (int)euhedral_q3_select_prefill_s104(mode, rows, width, outputs, 1, 1, 1, 1, has_s104);
}
int prefill_tile_rows(int selected) {
#ifdef EUHEDRAL_Q3_HAS_K32_PREFILL_TILE_POLICY
    return (int)euhedral_q3_prefill_tile_rows((enum euhedral_q3_prefill_kernel)selected);
#else
    (void)selected;
    return -1;
#endif
}
'''


class Q3PrefillPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.name != 'posix' or shutil.which('cc') is None:
            raise unittest.SkipTest('host C compiler unavailable for policy probe')
        cls.temp = tempfile.TemporaryDirectory(prefix='q3-prefill-policy-')
        library = pathlib.Path(cls.temp.name) / 'policy.so'
        subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                        '-I', str(HEADER), '-x', 'c', '-', '-o', str(library)],
                       input=SOURCE, text=True, capture_output=True, check=True)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.route.argtypes = (ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
        cls.lib.route.restype = ctypes.c_int
        cls.lib.wmma64.argtypes = (ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
        cls.lib.wmma64.restype = ctypes.c_int
        cls.lib.select.argtypes = (ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                   ctypes.c_int, ctypes.c_int)
        cls.lib.select.restype = ctypes.c_int
        cls.lib.use_k32_cb.argtypes = (ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
        cls.lib.use_k32_cb.restype = ctypes.c_int
        cls.lib.select_with_k32_cb.argtypes = (ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                                ctypes.c_int, ctypes.c_int, ctypes.c_int)
        cls.lib.select_with_k32_cb.restype = ctypes.c_int
        cls.lib.select_with_cb_input_alignment.argtypes = (ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                                                            ctypes.c_uint, ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int)
        cls.lib.select_with_cb_input_alignment.restype = ctypes.c_int
        cls.lib.select_s104.argtypes = (ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                        ctypes.c_int)
        cls.lib.select_s104.restype = ctypes.c_int
        cls.lib.prefill_tile_rows.argtypes = (ctypes.c_int,)
        cls.lib.prefill_tile_rows.restype = ctypes.c_int
        cls.lib.decode_wide.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
        cls.lib.decode_wide.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        del cls.lib
        cls.temp.cleanup()

    def test_wide_decode_requires_one_row_whole_scale_vectors_and_eight_columns(self):
        wide = self.lib.decode_wide
        for width, outputs in [(5120, 34816), (17408, 5120), (6144, 5120), (5120, 248320), (512, 8), (32768, 8)]:
            self.assertEqual(1, wide(1, width, outputs), (width, outputs))
        for rows, width, outputs in [(2, 5120, 8), (0, 5120, 8), (1, 384, 8), (1, 640, 8), (1, 33280, 8),
                                     (1, 0, 8), (1, 5120, 12), (1, 5120, 0)]:
            self.assertEqual(0, wide(rows, width, outputs), (rows, width, outputs))

    def test_mixer_wins_only_after_integrated_crossover(self):
        for rows in (1, 32, 63, 64, 65, 95, 96, 97, 112, 127, 128, 129, 256, 511):
            with self.subTest(rows=rows):
                self.assertEqual(0, self.lib.route(2, rows, 6144, 5120))
        for rows in (512, 513, 1024, 2048, 8192):
            with self.subTest(rows=rows):
                self.assertEqual(1, self.lib.route(2, rows, 6144, 5120))

    def test_unchanged_projection_and_mode_routes(self):
        for width, outputs in ((5120, 34816), (17408, 5120)):
            for rows in (1, 32, 63, 64, 96, 97, 128):
                with self.subTest(width=width, rows=rows):
                    self.assertEqual(int(rows >= 64), self.lib.route(2, rows, width, outputs))
        for width, outputs in ((6144, 5120), (5120, 34816), (17408, 5120), (6144, 5121)):
            for rows in (1, 64, 96, 97, 1024):
                with self.subTest(width=width, outputs=outputs, rows=rows):
                    self.assertEqual(0, self.lib.route(0, rows, width, outputs))
                    self.assertEqual(0, self.lib.route(1, rows, width, outputs))
                    self.assertEqual(1, self.lib.route(3, rows, width, outputs))
                    if (width, outputs) == (6144, 5121):
                        self.assertEqual(0, self.lib.route(2, rows, width, outputs))

    def test_wmma_leaf_only_for_single_band_down_projection(self):
        for rows in (64,):
            self.assertEqual(1, self.lib.wmma64(2, rows, 17408, 5120))
        for rows in (1, 32, 63):  # routed to the 32-row tile, not the 64-row tile
            self.assertEqual(0, self.lib.wmma64(2, rows, 17408, 5120))
        for rows in (65, 96, 128, 1024):
            self.assertEqual(0, self.lib.wmma64(2, rows, 17408, 5120))
        for width, outputs in ((5120, 34816), (6144, 5120)):
            for rows in (64, 512, 1024):
                self.assertEqual(0, self.lib.wmma64(2, rows, width, outputs))
        for mode in (0, 1, 3):
            self.assertEqual(0, self.lib.wmma64(mode, 64, 17408, 5120))

    def test_host_kernel_selection_and_fallbacks(self):
        NONE, P32, P64, WMMA = 0, 1, 2, 3
        down = (17408, 5120)
        select = self.lib.select
        # AUTO down at 64 rows: WMMA when present, else the default tile64, else tile32.
        self.assertEqual(WMMA, select(2, 64, *down, 1, 1))
        self.assertEqual(P64, select(2, 64, *down, 1, 0))
        self.assertEqual(P32, select(2, 64, *down, 0, 1))
        self.assertEqual(P32, select(2, 63, *down, 1, 1))
        for rows in (65, 128, 1024):
            self.assertEqual(P64, select(2, rows, *down, 1, 1))
        for rows in (64, 1024):
            self.assertEqual(P64, select(2, rows, 5120, 34816, 1, 1))
        self.assertEqual(P32, select(2, 256, 6144, 5120, 1, 1))
        self.assertEqual(P64, select(2, 512, 6144, 5120, 1, 1))
        # Explicit 64-row requests never use WMMA and fail without tile64.
        self.assertEqual(P64, select(3, 64, *down, 1, 1))
        self.assertEqual(NONE, select(3, 64, *down, 0, 1))
        self.assertEqual(NONE, select(0, 64, *down, 1, 1))

    def test_s104_route_is_limited_to_measured_mixer_row_counts(self):
        P32, S104 = 1, 5
        for rows in (1, 8, 16, 32, 65, 80, 96):
            with self.subTest(rows=rows):
                self.assertEqual(S104, self.lib.select_s104(2, rows, 6144, 5120, 1))
                self.assertEqual(P32, self.lib.select_s104(2, rows, 6144, 5120, 0))
        for rows in (33, 48, 64, 97, 128, 192):
            with self.subTest(rows=rows):
                self.assertEqual(P32, self.lib.select_s104(2, rows, 6144, 5120, 1))
        for width, outputs in ((5120, 34816), (17408, 5120), (6144, 5121)):
            self.assertNotEqual(S104, self.lib.select_s104(2, 32, width, outputs, 1))
        for mode in (0, 1, 3):
            self.assertNotEqual(S104, self.lib.select_s104(mode, 32, 6144, 5120, 1))
        self.assertEqual(32, self.lib.prefill_tile_rows(S104))

    def test_compact_b_route_is_only_used_for_measured_shapes(self):
        NONE, P32, P64, WMMA, K32_CB = 0, 1, 2, 3, 4
        cases = (
            (64, 5120, 34816, P64),
            (256, 5120, 34816, P64),
            (1024, 5120, 34816, P64),
            (256, 6144, 5120, P32),
            (512, 6144, 5120, P64),
            (1024, 6144, 5120, P64),
            (64, 17408, 5120, WMMA),
            (256, 17408, 5120, P64),
            (1024, 17408, 5120, P64),
        )
        for rows, width, outputs, fallback in cases:
            with self.subTest(rows=rows, width=width, outputs=outputs):
                self.assertEqual(1, self.lib.use_k32_cb(2, rows, width, outputs))
                self.assertEqual(K32_CB, self.lib.select_with_k32_cb(
                    2, rows, width, outputs, 1, 1, 1))
                self.assertEqual(fallback, self.lib.select_with_k32_cb(
                    2, rows, width, outputs, 0, 1, 1))

        excluded = (
            (64, 6144, 5120, P32),       # measured regression: mixer M=64
            (255, 6144, 5120, P32),      # adjacent to the qualified M=256 shape
            (257, 6144, 5120, P32),
            (1023, 6144, 5120, P64),     # adjacent to the qualified M=1024 shape
            (1025, 6144, 5120, P64),
            (63, 5120, 34816, P32),      # immediately below tile64 eligibility
            (65, 5120, 34816, P64),
            (255, 5120, 34816, P64),
            (257, 5120, 34816, P64),
            (1023, 5120, 34816, P64),
            (1025, 5120, 34816, P64),
            (63, 17408, 5120, P32),
            (65, 17408, 5120, P64),
            (255, 17408, 5120, P64),
            (257, 17408, 5120, P64),
            (1023, 17408, 5120, P64),
            (1025, 17408, 5120, P64),
            (128, 5120, 34816, P64),     # unmeasured; leave AUTO unchanged
            (511, 6144, 5120, P32),      # adjacent to the qualified M=512 shape
            (513, 6144, 5120, P64),
            (256, 6144, 5121, P32),      # different operator shape
        )
        for rows, width, outputs, fallback in excluded:
            with self.subTest(excluded=(rows, width, outputs)):
                self.assertEqual(0, self.lib.use_k32_cb(2, rows, width, outputs))
                self.assertEqual(fallback, self.lib.select_with_k32_cb(
                    2, rows, width, outputs, 1, 1, 1))

        for rows, width, outputs, _ in cases:
            with self.subTest(mode=3, rows=rows, width=width):
                self.assertEqual(0, self.lib.use_k32_cb(3, rows, width, outputs))
                self.assertEqual(P64, self.lib.select_with_k32_cb(
                    3, rows, width, outputs, 1, 1, 1))
        self.assertEqual(NONE, self.lib.select_with_k32_cb(0, 256, 5120, 34816, 1, 1, 1))

    def test_cb_requires_aligned_bf16_input_pointer(self):
        K32_CB = 4
        cases = (
            (64, 5120, 34816, 2),
            (256, 5120, 34816, 2),
            (1024, 5120, 34816, 2),
            (256, 6144, 5120, 1),
            (512, 6144, 5120, 2),
            (1024, 6144, 5120, 2),
            (64, 17408, 5120, 3),
            (256, 17408, 5120, 2),
            (1024, 17408, 5120, 2),
        )
        for rows, width, outputs, fallback in cases:
            with self.subTest(rows=rows, width=width, outputs=outputs):
                choose = self.lib.select_with_cb_input_alignment
                self.assertEqual(K32_CB, choose(2, rows, width, outputs, 1, 1, 1, 1))
                self.assertEqual(fallback, choose(2, rows, width, outputs, 1, 0, 1, 1))
                self.assertEqual(fallback, choose(2, rows, width, outputs, 0, 1, 1, 1))
        self.assertEqual(2, self.lib.select_with_cb_input_alignment(
            3, 256, 5120, 34816, 1, 0, 1, 1))

    def test_missing_optional_kernel_recomputes_legacy_tile_size(self):
        K32_CB = 4
        cases = (
            (64, 5120, 34816, 64),   # qualified gate/up: legacy tile64
            (256, 5120, 34816, 64),
            (256, 6144, 5120, 32),   # qualified mixer: legacy tile32
            (512, 6144, 5120, 64),   # qualified mixer: legacy tile64
            (1024, 6144, 5120, 64),
            (64, 17408, 5120, 64),   # legacy WMMA still has a 64-row grid
            (64, 6144, 5120, 32),    # measured mixer regression
            (63, 5120, 34816, 32),   # below the production tile64 crossover
        )
        for rows, width, outputs, fallback_tile in cases:
            with self.subTest(rows=rows, width=width, outputs=outputs):
                legacy = self.lib.select(2, rows, width, outputs, 1, 1)
                without_cb = self.lib.select_with_k32_cb(
                    2, rows, width, outputs, 0, 1, 1)
                self.assertEqual(legacy, without_cb)
                self.assertEqual(fallback_tile, self.lib.prefill_tile_rows(without_cb))
                selected = self.lib.select_with_k32_cb(
                    2, rows, width, outputs, 1, 1, 1)
                expected = K32_CB if self.lib.use_k32_cb(2, rows, width, outputs) else legacy
                self.assertEqual(expected, selected)
                self.assertEqual(64 if expected in (2, 3, K32_CB) else 32,
                                 self.lib.prefill_tile_rows(selected))


if __name__ == '__main__':
    unittest.main()
