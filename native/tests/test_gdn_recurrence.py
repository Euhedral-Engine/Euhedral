"""The packaged GDN recurrence must match the original shared-memory reduction tree bit for bit."""
import ctypes as C
import math
import random
import struct
import unittest
from pathlib import Path

from gpu_harness import Gpu, NVRTC, _check

ROOT = Path(__file__).resolve().parents[2]
PRODUCT = ROOT / 'build/native/linux-x64/share/euhedral_cuda'

# The reference keeps the original schedule: four full shared-memory tree reductions per row.
REFERENCE = rb'''
extern "C" __global__ void reference_gdn_recurrence_bf16(
        const __nv_bfloat16* convolved, const float* g, const float* beta,
        float* recurrentState, __nv_bfloat16* output, uint32_t rows,
        uint32_t keyHeads, uint32_t valueHeads, uint32_t keyHeadDim,
        uint32_t valueHeadDim, float outputScale) {
    const uint32_t valueRow = blockIdx.x;
    const uint32_t valueHead = valueRow / valueHeadDim;
    const uint32_t valueColumn = valueRow % valueHeadDim;
    if (valueHead >= valueHeads || blockDim.x != keyHeadDim) return;
    const uint32_t queryKeyWidth = 2 * keyHeads * keyHeadDim;
    const uint32_t convolvedWidth = queryKeyWidth + valueHeads * valueHeadDim;
    const uint32_t keyHead = valueHead / (valueHeads / keyHeads);
    const uint32_t lane = threadIdx.x;
    const uint64_t stateOffset = (static_cast<uint64_t>(valueHead) * valueHeadDim + valueColumn) * keyHeadDim;
    float stateValue = recurrentState[stateOffset + lane];
    __shared__ float scratch[128];
    for (uint32_t row = 0; row < rows; row++) {
        const uint64_t rowOffset = static_cast<uint64_t>(row) * convolvedWidth;
        const uint32_t queryBase = keyHead * keyHeadDim;
        const uint32_t keyBase = keyHeads * keyHeadDim + queryBase;
        const uint32_t valueBase = queryKeyWidth + valueHead * valueHeadDim + valueColumn;
        const float query = __bfloat162float(convolved[rowOffset + queryBase + lane]);
        const float key = __bfloat162float(convolved[rowOffset + keyBase + lane]);
        const float normalizedKey = key * rsqrtf(qwen_gdn_reduce_sum(key * key, scratch) + 1.0e-6f);
        const float normalizedQuery = query * rsqrtf(qwen_gdn_reduce_sum(query * query, scratch) + 1.0e-6f);
        const float stateKeyDot = qwen_gdn_reduce_sum(stateValue * normalizedKey, scratch);
        const float alpha = g[static_cast<uint64_t>(row) * valueHeads + valueHead];
        const float delta = beta[static_cast<uint64_t>(row) * valueHeads + valueHead]
                * (__bfloat162float(convolved[rowOffset + valueBase]) - alpha * stateKeyDot);
        stateValue = alpha * stateValue + delta * normalizedKey;
        const float outputSum = qwen_gdn_reduce_sum(stateValue * normalizedQuery, scratch);
        if (lane == 0) output[static_cast<uint64_t>(row) * valueHeads * valueHeadDim + valueRow]
                = __float2bfloat16_rn(outputSum * outputScale);
    }
    recurrentState[stateOffset + lane] = stateValue;
}
'''

KEY_HEADS, VALUE_HEADS, DIM = 2, 6, 128
WIDTH = 2 * KEY_HEADS * DIM + VALUE_HEADS * DIM
OUTPUTS = VALUE_HEADS * DIM
STATE_BYTES = VALUE_HEADS * DIM * DIM * 4


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16


def chunk(rows, seed, cancellation):
    rng = random.Random(seed)
    if cancellation:
        values = [rng.choice((1.0, -1.0)) * rng.choice((1e3, 1.0, 1e-3)) * (1 + rng.random() * 1e-2)
                  for _ in range(rows * WIDTH)]
    else:
        values = [rng.gauss(0.0, 1.0) for _ in range(rows * WIDTH)]
    convolved = struct.pack(f'<{len(values)}H', *map(bf16, values))
    g = struct.pack(f'<{rows * VALUE_HEADS}f', *[math.exp(-abs(rng.gauss(0.3, 0.3))) for _ in range(rows * VALUE_HEADS)])
    beta = struct.pack(f'<{rows * VALUE_HEADS}f', *[rng.random() for _ in range(rows * VALUE_HEADS)])
    return convolved, g, beta


@unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
class GdnRecurrenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = b'#include "gdn/kernels.cu"\n' + REFERENCE
        cls.gpu = Gpu(source, include_dir=PRODUCT)

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def launch(self, kernel, arguments):
        # The reference keeps one 128-thread CTA per value column; production uses one warp per
        # QWEN_GDN_WARP_COLUMNS (8) columns.
        grid, block = (OUTPUTS, 128) if kernel.startswith('reference') else (
                OUTPUTS // 4 if kernel.endswith('_c4_bf16') else OUTPUTS // 8, 32)
        function = C.c_void_p()
        _check(self.gpu.function(C.byref(function), self.gpu.module, kernel.encode()), kernel)
        params = (C.c_void_p * len(arguments))(*[C.cast(C.pointer(v), C.c_void_p) for v in arguments])
        _check(self.gpu.launch_kernel(function, grid, 1, 1, block, 1, 1, 0, None, params, None), kernel)
        _check(self.gpu.sync(), kernel)

    def run_chunks(self, kernel, initial, chunks, rows):
        gpu = self.gpu
        state = gpu.upload(initial)
        results = []
        try:
            for convolved, g, beta in chunks:
                inputs = [gpu.upload(convolved), gpu.upload(g), gpu.upload(beta)]
                output = gpu.zeros(rows * OUTPUTS * 2, 0xa5)
                try:
                    self.launch(kernel, [C.c_uint64(inputs[0]), C.c_uint64(inputs[1]), C.c_uint64(inputs[2]),
                                                 C.c_uint64(state), C.c_uint64(output), C.c_uint(rows),
                                                 C.c_uint(KEY_HEADS), C.c_uint(VALUE_HEADS), C.c_uint(DIM),
                                                 C.c_uint(DIM), C.c_float(DIM ** -0.5)])
                    results.append(gpu.download(output, rows * OUTPUTS * 2))
                finally:
                    gpu.free(output)
                    for pointer in inputs:
                        gpu.free(pointer)
            results.append(gpu.download(state, STATE_BYTES))
        finally:
            gpu.free(state)
        return results

    def test_matches_original_reduction_bitwise_across_chunks(self):
        rng = random.Random(7)
        initial = struct.pack(f'<{STATE_BYTES // 4}f', *[rng.gauss(0.0, 0.05) for _ in range(STATE_BYTES // 4)])
        for rows in (1, 3, 17):
            for cancellation in (False, True):
                with self.subTest(rows=rows, cancellation=cancellation):
                    chunks = [chunk(rows, 100 + rows, cancellation), chunk(rows, 200 + rows, cancellation)]
                    expected = self.run_chunks('reference_gdn_recurrence_bf16', initial, chunks, rows)
                    actual = self.run_chunks('euhedral_gdn_recurrence_bf16', initial, chunks, rows)
                    self.assertEqual(expected, actual)
                    for output in actual[:-1]:
                        self.assertNotIn(b'\xa5\xa5', [output[i:i + 2] for i in range(0, len(output), 2)])

    def test_column_owned_recurrence_stays_close_to_the_exact_kernel(self):
        # The relaxed kernel reorders each key reduction and normalize the reduced dot products; over two
        # chunks the outputs and the carried state stay within FP32 reassociation error of the exact path.
        rng = random.Random(11)
        initial = struct.pack(f'<{STATE_BYTES // 4}f', *[rng.gauss(0.0, 0.05) for _ in range(STATE_BYTES // 4)])
        def floats(data, bf16_values):
            if bf16_values:
                return [struct.unpack('<f', struct.pack('<I', v << 16))[0] for v in struct.unpack(f'<{len(data) // 2}H', data)]
            return list(struct.unpack(f'<{len(data) // 4}f', data))
        def relative_rms(expected, actual):
            num = sum((e - a) ** 2 for e, a in zip(expected, actual))
            den = sum(e * e for e in expected)
            return (num / den) ** 0.5
        for rows in (1, 3, 17):
            chunks = [chunk(rows, 300 + rows, False), chunk(rows, 400 + rows, False)]
            expected = self.run_chunks('euhedral_gdn_recurrence_bf16', initial, chunks, rows)
            for kernel in ('euhedral_gdn_recurrence_c8_bf16',):
                with self.subTest(rows=rows, kernel=kernel):
                    actual = self.run_chunks(kernel, initial, chunks, rows)
                    for e, a in zip(expected[:-1], actual[:-1]):
                        self.assertNotIn(b'\xa5\xa5', [a[i:i + 2] for i in range(0, len(a), 2)])
                        self.assertLess(relative_rms(floats(e, True), floats(a, True)), 4e-3)
                    self.assertLess(relative_rms(floats(expected[-1], False), floats(actual[-1], False)), 1e-5)


if __name__ == '__main__':
    unittest.main()
