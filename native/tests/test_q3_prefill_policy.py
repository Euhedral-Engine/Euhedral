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
int route(int mode, unsigned rows, unsigned width, unsigned outputs) {
    return euhedral_q3_wide_prefill(mode, rows, width, outputs);
}
int wmma64(int mode, unsigned rows, unsigned width, unsigned outputs) {
    return euhedral_q3_wmma_prefill64(mode, rows, width, outputs);
}
int select(int mode, unsigned rows, unsigned width, unsigned outputs, int has64, int has_wmma) {
    return (int)euhedral_q3_select_prefill(mode, rows, width, outputs, has64, has_wmma);
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

    @classmethod
    def tearDownClass(cls):
        del cls.lib
        cls.temp.cleanup()

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


if __name__ == '__main__':
    unittest.main()
