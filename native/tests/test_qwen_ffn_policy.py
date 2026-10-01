"""Exact FFN shape leaves and their excluded neighbors, without CUDA."""
import ctypes
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

HEADER = pathlib.Path(__file__).resolve().parents[1] / 'src'
SOURCE = '''#include "qwen_ffn_policy.h"
unsigned gate(unsigned m,unsigned k,unsigned n){return euhedral_ffn_gate_tile_rows(m,k,n);}
int down(unsigned m,unsigned k,unsigned n){return euhedral_ffn_down_wide(m,k,n);}
unsigned down_tile(unsigned m,unsigned k,unsigned n){return euhedral_ffn_down_tile_rows(m,k,n);}
int streamed(unsigned m){return euhedral_ffn_streamed_rows(m);}
'''

class FfnPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which('cc') is None:
            raise unittest.SkipTest('host C compiler unavailable')
        cls.temp = tempfile.TemporaryDirectory(prefix='ffn-policy-')
        library = pathlib.Path(cls.temp.name) / ('policy.dll' if os.name == 'nt' else 'policy.so')
        subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                        '-I', str(HEADER), '-x', 'c', '-', '-o', str(library)],
                       input=SOURCE, text=True, capture_output=True, check=True)
        cls.lib = ctypes.CDLL(str(library))
        for name in ('gate', 'down', 'down_tile'):
            getattr(cls.lib, name).argtypes = (ctypes.c_uint,) * 3
            getattr(cls.lib, name).restype = ctypes.c_uint
        cls.lib.streamed.argtypes = (ctypes.c_uint,)
        cls.lib.streamed.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        handle = cls.lib._handle
        del cls.lib
        if os.name == 'nt':
            import _ctypes
            _ctypes.FreeLibrary(handle)
        cls.temp.cleanup()

    def test_down_tile_rows(self):
        for rows in (1, 64, 255, 256, 257, 288, 320, 321, 511, 512, 513, 1023, 1024, 1025):
            expected = 64 if 256 < rows <= 320 else 128 if rows in (256, 512, 1024) else 0
            with self.subTest(rows=rows):
                self.assertEqual(expected, self.lib.down_tile(rows, 17408, 5120))
                for width, outputs in ((17409, 5120), (17408, 5184), (6144, 5120)):
                    self.assertEqual(0, self.lib.down_tile(rows, width, outputs))

    def test_exact_shapes_and_adjacent_fallbacks(self):
        for rows in (1, 63, 64, 65, 128, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025):
            with self.subTest(rows=rows):
                self.assertEqual(64 if rows == 64 else 128 if rows in (256,512,1024) else 0,
                                 self.lib.gate(rows,5120,34816))
                self.assertEqual(int(rows in (256,512,1024)), self.lib.down(rows,17408,5120))
                self.assertEqual(int(rows == 1024), self.lib.streamed(rows))
                for width, outputs in ((5121,34816),(5120,34880),(6144,5120)):
                    self.assertEqual(0,self.lib.gate(rows,width,outputs))
                for width, outputs in ((17409,5120),(17408,5184),(6144,5120)):
                    self.assertEqual(0,self.lib.down(rows,width,outputs))

if __name__ == '__main__':
    unittest.main()
