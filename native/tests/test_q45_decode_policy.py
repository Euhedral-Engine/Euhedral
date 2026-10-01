"""Q4/Q5 single-row wide-decode shape policy, exercised without CUDA or a device."""
import ctypes
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

HEADER = pathlib.Path(__file__).resolve().parents[1] / 'src'
SOURCE = '''#include "q45_decode_policy.h"
int wide(unsigned rows, unsigned width, unsigned outputs) {
    return euhedral_q45_decode_wide_shape(rows, width, outputs);
}
'''


class Q45DecodePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.name != 'posix' or shutil.which('cc') is None:
            raise unittest.SkipTest('host C compiler unavailable for policy probe')
        cls.temp = tempfile.TemporaryDirectory(prefix='q45-decode-policy-')
        library = pathlib.Path(cls.temp.name) / 'policy.so'
        subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                        '-I', str(HEADER), '-x', 'c', '-', '-o', str(library)],
                       input=SOURCE, text=True, capture_output=True, check=True)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.wide.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
        cls.lib.wide.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        del cls.lib
        cls.temp.cleanup()

    def test_wide_decode_only_beyond_one_wave_with_whole_vectors(self):
        wide = self.lib.wide
        # Measured model shapes: the 896- and 1536-CTA projections are wide; the 512-CTA one is not.
        for width, outputs in [(5120, 12288), (5120, 7168), (512, 4104), (8192, 4104)]:
            self.assertEqual(1, wide(1, width, outputs), (width, outputs))
        for rows, width, outputs in [(1, 5120, 4096), (2, 5120, 7168), (0, 5120, 7168), (1, 384, 7168),
                                     (1, 8704, 7168), (1, 0, 7168), (1, 5120, 7172), (1, 5120, 0)]:
            self.assertEqual(0, wide(rows, width, outputs), (rows, width, outputs))


if __name__ == '__main__':
    unittest.main()
