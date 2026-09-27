"""Installed Q3 headers must compile from the product beside the unchanged entry TU."""
import ctypes as C
from pathlib import Path
import unittest
from test_q3_primitives import Gpu, NVRTC

ROOT = Path(__file__).resolve().parents[2]
PRODUCT = ROOT / 'build/native/linux-x64/share/euhedral_cuda'
HEADERS = (
    'numeric.cuh', 'layout.cuh', 'primitives/packed_load.cuh',
    'primitives/activation.cuh', 'primitives/decode.cuh',
    'primitives/staging.cuh', 'primitives/mma.cuh', 'primitives/mma_leaf.cuh',
    'primitives/accumulation.cuh', 'primitives/writeback.cuh',
    'strategies/scalar.cuh', 'strategies/decode.cuh',
    'strategies/prefill.cuh',
)


class Q3ModuleTest(unittest.TestCase):
    def test_product_contains_header_graph(self):
        for relative in HEADERS:
            with self.subTest(header=relative):
                self.assertEqual((ROOT / 'native/src/q3' / relative).read_bytes(),
                                 (PRODUCT / 'q3' / relative).read_bytes())

    @unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
    def test_headers_can_be_included_directly(self):
        # Compile and load from the installed product, not the repo's source tree.
        source = b'''#include "q3/primitives/staging.cuh"
#include "q3/primitives/mma.cuh"
extern "C" __global__ void module_probe(unsigned short* x) {
'''
        source += b'''  x[0] = q3::float_to_bf16(q3::apply_scale(q3::decode_code(4,0), 1.0f));\n}\n'''
        gpu = Gpu(source, include_dir=PRODUCT)
        try:
            out = gpu.zeros(2)
            try:
                gpu.launch('module_probe', 1, [C.c_uint64(out)])
                self.assertEqual(gpu.download(out, 2), b'\x80\xc0')  # -4.0 BF16
            finally:
                gpu.free(out)
        finally:
            gpu.close()


if __name__ == '__main__':
    unittest.main()
