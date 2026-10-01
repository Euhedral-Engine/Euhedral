"""Opt-in CTA cluster compositions must reproduce their CTA-local Q3 tile exactly."""
import ctypes as C
import random
import struct
import unittest
from test_q3_primitives import Gpu, NVRTC, ROOT, to_bf16

COMPOSITIONS = ((1, 1), (2, 1), (4, 1), (1, 2), (1, 4), (2, 2))


@unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
class Q3ClusterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "q3/cluster_kernels.cu"\n', cpp_std=17)
        try:
            cls.baseline = Gpu(b'#include "q3/kernels.cu"\n')
        except BaseException:
            cls.gpu.close()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.baseline.close()
        cls.gpu.close()

    def test_every_composition_matches_cta_tiles_at_partial_boundaries(self):
        gpu = self.gpu
        rng = random.Random(0x51C3)
        for tile_rows, rows, width, outputs in ((32, 33, 65, 35), (64, 65, 192, 65)):
            groups = ((width + 127) // 128) * 2
            scale_offset = (outputs * groups * 24 + 255) & ~255
            packed = bytearray(rng.randbytes(scale_offset + outputs * groups * 2))
            for i in range(outputs * groups):
                struct.pack_into('<H', packed, scale_offset + 2 * i, 0x2E00 + (i % 16))
            values = [to_bf16(rng.uniform(-2, 2)) for _ in range(rows * width)]
            x = gpu.upload(struct.pack(f'<{len(values)}H', *values))
            try:
                w = gpu.upload(packed)
                try:
                    reference = self.run_kernel('euhedral_q3_prefill' + ('_64' if tile_rows == 64 else '') + '_exact',
                                                x, w, rows, width, outputs, scale_offset,
                                                ((rows + tile_rows - 1) // tile_rows) * ((outputs + 31) // 32),
                                                gpu=self.baseline)
                    for cm, cn in COMPOSITIONS:
                        label = f'euhedral_q3_cluster_{tile_rows}_{cm}x{cn}'
                        grid = (((outputs + 32 * cn - 1) // (32 * cn)) * cn,
                                ((rows + tile_rows * cm - 1) // (tile_rows * cm)) * cm)
                        result = self.run_kernel(label, x, w, rows, width, outputs, scale_offset, grid)
                        with self.subTest(tile_rows=tile_rows, rows=rows, width=width, outputs=outputs, cm=cm, cn=cn):
                            self.assertEqual(result, reference)
                            self.assertNotIn(b'\xa5\xa5', [result[i:i + 2] for i in range(0, len(result), 2)])
                finally:
                    gpu.free(w)
            finally:
                gpu.free(x)

    def run_kernel(self, name, x, w, rows, width, outputs, scale_offset, grid, gpu=None):
        gpu = gpu or self.gpu
        size = rows * outputs * 2
        y = gpu.zeros(size, 0xA5)
        try:
            gpu.launch(name, grid, [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y), C.c_uint(rows),
                                    C.c_uint(width), C.c_uint(outputs), C.c_ulonglong(scale_offset)])
            return gpu.download(y, size)
        finally:
            gpu.free(y)


if __name__ == '__main__':
    unittest.main()
