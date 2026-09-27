"""Q3 prefill shared-memory lifetime reuse resource contract."""
import contextlib
import ctypes as C
from pathlib import Path
import unittest

from test_q3_primitives import CUDA, NVRTC, Gpu, P, I, _bind, _check

ROOT = Path(__file__).resolve().parents[2]
PRODUCT = ROOT / 'build/native/linux-x64/share/euhedral_cuda'


@unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
class Q3SharedOverlayTest(unittest.TestCase):
    def test_packaged_kernel_source_matches(self):
        self.assertEqual((ROOT / 'native/src/q3/kernels.cu').read_bytes(),
                         (PRODUCT / 'q3/kernels.cu').read_bytes())

    def test_prefill_result_reuses_activation_shared_storage(self):
        with contextlib.ExitStack() as cleanup:
            gpu = Gpu(b'#include "q3/kernels.cu"\n', include_dir=PRODUCT)
            cleanup.callback(gpu.close)
            attribute = _bind(CUDA, 'cuFuncGetAttribute', [C.POINTER(I), I, P])
            for kernel, expected_bytes in [('euhedral_q3_prefill', 12288),
                                           ('euhedral_q3_prefill_64', 16384)]:
                with self.subTest(kernel=kernel):
                    function = P()
                    _check(gpu.function(C.byref(function), gpu.module, kernel.encode()), kernel)
                    shared = I()
                    _check(attribute(C.byref(shared), 1, function), 'static shared bytes')
                    self.assertEqual(expected_bytes, shared.value)


if __name__ == '__main__':
    unittest.main()
