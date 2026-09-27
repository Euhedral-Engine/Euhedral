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
        for asset in ('q3/kernels.cu', 'q3/strategies/prefill.cuh',
                      'q3/primitives/prefetch.cuh'):
            with self.subTest(asset=asset):
                self.assertEqual((ROOT / 'native/src' / asset).read_bytes(),
                                 (PRODUCT / asset).read_bytes())

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
        result = source.index('new (static_cast<void*>(&staging)) Shared(typename Shared::ActivateResult{});', k_loop)
        final_k_barrier = source.rindex('__syncthreads();', k_loop, result)
        result_barrier = source.index('__syncthreads();', result)
        result_pointer = source.index('staging.result.values', result)
        store = source.index('Leaf::store(result, acc, warp);', result_pointer)
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

    def test_prefill_layout_selects_padded_a_and_tile_specific_b(self):
        source = (ROOT / 'native/src/q3/strategies/prefill.cuh').read_text()
        kernels = (ROOT / 'native/src/q3/kernels.cu').read_text()
        self.assertIn('values[Tile::kRows * A_STRIDE]', source)
        self.assertIn('tile[(i / K_TILE) * STRIDE + i % K_TILE]',
                      (ROOT / 'native/src/q3/primitives/activation.cuh').read_text())
        self.assertIn('stage_split_pair(hi, lo, col * STRIDE + lane * 2, codes, scale);',
                      (ROOT / 'native/src/q3/primitives/staging.cuh').read_text())
        self.assertIn('Leaf::template consume<kGroup, A_STRIDE, B_STRIDE>', source)
        self.assertIn('constexpr int A_STRIDE = 80;', kernels)
        self.assertIn('constexpr int B_STRIDE = Tile::kRows == 32 ? 80 : 64;', kernels)
        self.assertIn('Tile::kCols * B_STRIDE', kernels)

    def test_prefill_compact_next_group_before_current_mma(self):
        source = (ROOT / 'native/src/q3/strategies/prefill.cuh').read_text()
        self.assertIn('CompactPrefetch<Tile> next;', source)
        self.assertIn('(reinterpret_cast<unsigned long long>(input) & 3ull) == 0ull', source)
        self.assertIn('(in_features & 1u) == 0u', source)
        self.assertIn('stage_activation_tile<Tile::kRows, kGroup, kThreads, A_STRIDE>', source)
        self.assertIn('stage_weight_tile<Tile::kCols, Tile::kWarps, B_STRIDE>', source)
        self.assertIn('prefetch_compact_tile(next, input, w,', source)
        self.assertIn('stage_prefetched_activation', source)
        self.assertIn('stage_prefetched_weights', source)
        loop = source.index('for (unsigned int base = 0;')
        stage = source.index('stage_prefetched_activation', loop)
        visible = source.index('__syncthreads();', stage)
        next_load = source.index('prefetch_compact_tile(next, input, w,', visible)
        consume = source.index('Leaf::template consume<kGroup, A_STRIDE, B_STRIDE>', next_load)
        retire = source.index('__syncthreads();', consume)
        self.assertLess(stage, visible)
        self.assertLess(visible, next_load)
        self.assertLess(next_load, consume)
        self.assertLess(consume, retire)
        self.assertNotIn('stage_activation_tile<', source[loop:retire])
        self.assertNotIn('stage_weight_tile<', source[loop:retire])
        compact = (ROOT / 'native/src/q3/primitives/prefetch.cuh').read_text()
        self.assertIn('unsigned int activation[', compact)
        self.assertIn('unsigned int words[', compact)

    def test_prefill_result_reuses_activation_shared_storage(self):
        with contextlib.ExitStack() as cleanup:
            gpu = Gpu(b'#include "q3/kernels.cu"\n', include_dir=PRODUCT)
            cleanup.callback(gpu.close)
            attribute = _bind(CUDA, 'cuFuncGetAttribute', [C.POINTER(I), I, P])
            for kernel, expected_bytes in [('euhedral_q3_prefill', 15360),
                                           ('euhedral_q3_prefill_64', 18432)]:
                with self.subTest(kernel=kernel):
                    function = P()
                    _check(gpu.function(C.byref(function), gpu.module, kernel.encode()), kernel)
                    shared = I()
                    _check(attribute(C.byref(shared), 1, function), 'static shared bytes')
                    self.assertEqual(expected_bytes, shared.value)


if __name__ == '__main__':
    unittest.main()
