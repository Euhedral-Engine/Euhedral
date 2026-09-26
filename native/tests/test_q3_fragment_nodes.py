"""Sibling warp leaves share one published fragment per K generation."""
import ctypes as C
from contextlib import closing
import random
import struct
import unittest
from test_q3_primitives import Gpu, NVRTC, ROOT, to_bf16

SHAPES = ((1, 1), (2, 1), (4, 1), (1, 2), (1, 4), (2, 2))


@unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
class Q3FragmentNodeTest(unittest.TestCase):
    def test_fragment_nodes_publish_once_and_fan_out_to_siblings(self):
        source = ROOT / 'native/src/q3/fragment_kernels.cu'
        self.assertTrue(source.is_file(), 'intermediate fragment-node kernels are missing')
        with self.gpu(b'#include "q3/fragment_kernels.cu"\n', 17) as nodes, \
             self.gpu(b'#include "q3/hierarchical_kernels.cu"\n', 17) as hierarchical, \
             self.gpu(b'#include "q3/cluster_kernels.cu"\n', 17) as replicated, \
             self.gpu(b'#include "q3/kernels.cu"\n', 14) as local:
            rng = random.Random(0xF12A)
            for tile, rows, width, outputs in ((32, 33, 65, 35), (64, 65, 192, 65),
                                               (32, 128, 192, 128), (64, 256, 192, 128)):
                groups = ((width + 127) // 128) * 2
                offset = (outputs * groups * 24 + 255) & ~255
                packed = bytearray(rng.randbytes(offset + outputs * groups * 2))
                for i in range(outputs * groups):
                    struct.pack_into('<H', packed, offset + 2 * i, 0x2e00 + i % 16)
                with self.buffers(nodes, rows, width, packed) as (x, w):
                    base = 'euhedral_q3_prefill' + ('_64' if tile == 64 else '')
                    reference = self.run_one(local, base,
                        ((rows + tile-1)//tile)*((outputs+31)//32), x, w, rows, width, outputs, offset)
                    for cm, cn in SHAPES:
                        grid = (((outputs+32*cn-1)//(32*cn))*cn,
                                ((rows+tile*cm-1)//(tile*cm))*cm)
                        old = self.run_one(replicated, f'euhedral_q3_cluster_{tile}_{cm}x{cn}', grid,
                                           x, w, rows, width, outputs, offset)
                        prior = self.run_one(hierarchical, f'euhedral_q3_hierarchical_{tile}_{cm}x{cn}', grid,
                                             x, w, rows, width, outputs, offset)
                        actual, observations = self.run_node(nodes, tile, cm, cn, grid, x, w,
                                                             rows, width, outputs, offset)
                        with self.subTest(tile=tile, rows=rows, width=width, outputs=outputs, cm=cm, cn=cn):
                            self.assertEqual(actual, reference)
                            self.assertEqual(actual, old)
                            self.assertEqual(actual, prior)
                            self.assertNoDuplicateFetches(observations, grid, tile, cm, cn, width)

    @unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
    def test_installed_fragment_node_source_compiles(self):
        product = ROOT / 'build/native/linux-x64/share/euhedral_cuda'
        for relative in ('fragment_kernels.cu', 'strategies/fragments.cuh', 'strategies/hierarchical.cuh'):
            with self.subTest(source=relative):
                self.assertEqual((ROOT / 'native/src/q3' / relative).read_bytes(),
                                 (product / 'q3' / relative).read_bytes())
        with self.gpu(b'#include "q3/fragment_kernels.cu"\n', 17, include_dir=product) as gpu:
            self.assertTrue(gpu.module.value)

    @staticmethod
    def gpu(source, cpp_std, include_dir=None):
        return closing(Gpu(source, cpp_std=cpp_std, include_dir=include_dir))

    @staticmethod
    def buffers(gpu, rows, width, packed):
        from contextlib import ExitStack
        stack = ExitStack()
        rng = random.Random(0xED6E)
        x = gpu.upload(struct.pack(f'<{rows*width}H',
            *(to_bf16(rng.uniform(-2,2)) for _ in range(rows*width))))
        stack.callback(gpu.free, x)
        w = gpu.upload(packed)
        stack.callback(gpu.free, w)
        return _Buffers(stack, x, w)

    @staticmethod
    def run_one(gpu, name, grid, x, w, rows, width, outputs, offset):
        size = rows * outputs * 2
        y = gpu.zeros(size, 0xa5)
        try:
            gpu.launch(name, grid, [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y), C.c_uint(rows),
                                    C.c_uint(width), C.c_uint(outputs), C.c_uint64(offset)])
            result = gpu.download(y, size)
            if any(result[i:i+2] == b'\xa5\xa5' for i in range(0, len(result), 2)):
                raise AssertionError('output contains unwritten sentinel')
            return result
        finally:
            gpu.free(y)

    @staticmethod
    def run_node(gpu, tile, cm, cn, grid, x, w, rows, width, outputs, offset):
        size = rows * outputs * 2
        warps = 4
        ctas = grid[0] * grid[1]
        y = gpu.zeros(size, 0xa5)
        observations = gpu.zeros(ctas * (2 + 2*warps) * 8)
        try:
            gpu.launch(f'euhedral_q3_fragments_{tile}_{cm}x{cn}', grid,
                [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y), C.c_uint(rows), C.c_uint(width),
                 C.c_uint(outputs), C.c_uint64(offset), C.c_uint64(observations)])
            return gpu.download(y, size), gpu.download(observations, ctas * (2+2*warps)*8)
        finally:
            gpu.free(observations)
            gpu.free(y)

    def assertNoDuplicateFetches(self, observations, grid, tile, cm, cn, width):
        warps = 4
        values = struct.unpack(f'<{len(observations)//8}Q', observations)
        k_groups = (width + 63)//64
        a_fragments_per_branch = tile // 32
        m_branches, n_branches = 2, 2
        for cta_y in range(grid[1]):
            for cta_x in range(grid[0]):
                cta = cta_y * grid[0] + cta_x
                at = cta * (2 + 2*warps)
                expected_a = k_groups * m_branches * a_fragments_per_branch if cta_x % cn else 0
                expected_b = k_groups * n_branches if cta_y % cm else 0
                self.assertEqual(values[at:at+2], (expected_a, expected_b))
                a_views = values[at+2:at+2+2*warps:2]
                b_views = values[at+3:at+2+2*warps:2]
                for left in range(warps):
                    for right in range(warps):
                        if left // 2 == right // 2:
                            self.assertEqual(a_views[left], a_views[right], 'A node was not shared by N siblings')
                        if left % 2 == right % 2:
                            self.assertEqual(b_views[left], b_views[right], 'B node was not shared by M siblings')


class _Buffers:
    def __init__(self, stack, x, w):
        self.stack, self.x, self.w = stack, x, w
    def __enter__(self):
        return self.x, self.w
    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


if __name__ == '__main__':
    unittest.main()
