"""Direct cluster-owned Q3 tiles flow to warp leaves without full CTA copies."""
import ctypes as C
from contextlib import ExitStack, closing
import random
import struct
import unittest
from test_q3_primitives import Gpu, NVRTC, ROOT, to_bf16

SHAPES = ((1, 1), (2, 1), (4, 1), (1, 2), (1, 4), (2, 2))


@unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
class HierarchicalPrefillTest(unittest.TestCase):
    def test_direct_borrow_matches_local_and_replicated_cluster(self):
        source = ROOT / 'native/src/q3/hierarchical_kernels.cu'
        self.assertTrue(source.is_file(), 'hierarchical strategy translation unit is missing')
        with self.gpu(b'#include "q3/hierarchical_kernels.cu"\n', 17) as direct, \
             self.gpu(b'#include "q3/cluster_kernels.cu"\n', 17) as replicated, \
             self.gpu(b'#include "q3/kernels.cu"\n', 14) as local:
            rng = random.Random(0x51C3)
            for tile, rows, width, outputs in ((32, 33, 65, 35), (64, 65, 192, 65),
                                               (32, 128, 192, 128), (64, 256, 192, 128)):
                groups = ((width + 127) // 128) * 2
                offset = (outputs * groups * 24 + 255) & ~255
                packed = bytearray(rng.randbytes(offset + outputs * groups * 2))
                for i in range(outputs * groups):
                    struct.pack_into('<H', packed, offset + 2 * i, 0x2e00 + i % 16)
                with ExitStack() as buffers:
                    x = direct.upload(struct.pack(f'<{rows * width}H',
                                                   *(to_bf16(rng.uniform(-2, 2)) for _ in range(rows * width))))
                    buffers.callback(direct.free, x)
                    w = direct.upload(packed)
                    buffers.callback(direct.free, w)
                    base = 'euhedral_q3_prefill' + ('_64' if tile == 64 else '')
                    reference = self.run_one(local, base, ((rows + tile - 1)//tile) * ((outputs + 31)//32),
                                             x, w, rows, width, outputs, offset)
                    for cm, cn in SHAPES:
                        grid = (((outputs + 32*cn-1)//(32*cn))*cn,
                                ((rows + tile*cm-1)//(tile*cm))*cm)
                        old = self.run_one(replicated, f'euhedral_q3_cluster_{tile}_{cm}x{cn}', grid,
                                           x, w, rows, width, outputs, offset)
                        actual = self.run_one(direct, f'euhedral_q3_hierarchical_{tile}_{cm}x{cn}', grid,
                                              x, w, rows, width, outputs, offset)
                        with self.subTest(tile=tile, rows=rows, width=width, outputs=outputs,
                                          cm=cm, cn=cn):
                            self.assertEqual(actual, reference)
                            self.assertEqual(actual, old)
                            self.assertNotIn(b'\xa5\xa5', [actual[i:i+2] for i in range(0, len(actual), 2)])

    @unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
    def test_installed_hierarchical_source_compiles(self):
        product = ROOT / 'build/native/linux-x64/share/euhedral_cuda'
        for relative in ('hierarchical_kernels.cu', 'strategies/hierarchical.cuh'):
            with self.subTest(source=relative):
                self.assertEqual((ROOT / 'native/src/q3' / relative).read_bytes(),
                                 (product / 'q3' / relative).read_bytes())
        with self.gpu(b'#include "q3/hierarchical_kernels.cu"\n', 17, include_dir=product) as gpu:
            self.assertTrue(gpu.module.value)

    @staticmethod
    def gpu(source, cpp_std, include_dir=None):
        return closing(Gpu(source, cpp_std=cpp_std, include_dir=include_dir))

    @staticmethod
    def run_one(gpu, name, grid, x, w, rows, width, outputs, offset):
        size = rows * outputs * 2
        y = gpu.zeros(size, 0xa5)
        try:
            gpu.launch(name, grid, [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y), C.c_uint(rows),
                                    C.c_uint(width), C.c_uint(outputs), C.c_uint64(offset)])
            return gpu.download(y, size)
        finally:
            gpu.free(y)


if __name__ == '__main__':
    unittest.main()
