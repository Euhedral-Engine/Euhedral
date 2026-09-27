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


if __name__ == '__main__':
    unittest.main()
