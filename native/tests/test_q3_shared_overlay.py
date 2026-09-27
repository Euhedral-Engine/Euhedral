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

    def test_prefill_activates_each_shared_region_before_using_it(self):
        source = (ROOT / 'native/src/q3/strategies/prefill.cuh').read_text()
        self.assertIn('struct Activation {', source)
        self.assertIn('struct Result {', source)
        self.assertIn('PrefillShared(ActivateA) : a() {}', source)
        self.assertIn('PrefillShared(ActivateResult) : result() {}', source)
        self.assertIn('static_assert(__is_trivially_constructible(Shared)', source)
        self.assertIn('new (static_cast<void*>(&staging)) Shared(typename Shared::ActivateA{});', source)
        self.assertIn('new (static_cast<void*>(&staging)) Shared(typename Shared::ActivateResult{});', source)
        activation = source.index('new (static_cast<void*>(&staging)) Shared(typename Shared::ActivateA{});')
        activation_barrier = source.index('__syncthreads();', activation)
        a_pointer = source.index('staging.a.values', activation)
        k_loop = source.index('for (unsigned int base = 0;', a_pointer)
        final_k_barrier = source.index('__syncthreads();\n    }', k_loop)
        result = source.index('new (static_cast<void*>(&staging)) Shared(typename Shared::ActivateResult{});', final_k_barrier)
        result_barrier = source.index('__syncthreads();', result)
        result_pointer = source.index('staging.result.values', result)
        store = source.index('store_accumulators<Tile>', result_pointer)
        self.assertLess(activation, activation_barrier)
        self.assertLess(activation_barrier, a_pointer)
        self.assertLess(a_pointer, k_loop)
        self.assertLess(final_k_barrier, result)
        self.assertLess(result, result_barrier)
        self.assertLess(result_barrier, result_pointer)
        self.assertLess(result_pointer, store)
        self.assertEqual(source.count('staging.a.values'), 1)
        self.assertEqual(source.count('staging.result.values'), 1)
        kernels = (ROOT / 'native/src/q3/kernels.cu').read_text()
        self.assertNotIn('staging.a', kernels)
        self.assertNotIn('staging.result', kernels)

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
