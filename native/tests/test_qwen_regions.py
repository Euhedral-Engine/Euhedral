"""Bitwise checks for macro-prefill regions against the unfused numerical boundaries."""
import contextlib
import ctypes as C
import random
import struct
import unittest

from gpu_harness import Gpu, NVRTC, CUDA, ROOT, _check


@unittest.skipIf(NVRTC is None or CUDA is None, "CUDA/NVRTC unavailable")
class QwenRegionsTest(unittest.TestCase):
    def test_control_region_preserves_fp32_projection_control_order(self):
        source = b'#include "linear/kernels.cu"\n#include "gdn/kernels.cu"\n'
        with contextlib.ExitStack() as scope:
            gpu = Gpu(source)
            scope.callback(gpu.close)
            symbol = C.c_void_p()
            self.assertEqual(0, gpu.function(C.byref(symbol), gpu.module, b'euhedral_gdn_project_control_fp32'),
                             'combined GDN control region is unavailable')
            rng = random.Random(9921)
            for rows, width, heads in [(1, 129, 3), (7, 256, 8), (13, 129, 8), (256, 5120, 32)]:
                with contextlib.ExitStack() as case:
                    def upload(data):
                        ptr = gpu.upload(data)
                        case.callback(gpu.free, ptr)
                        return ptr
                    def alloc(size):
                        ptr = gpu.zeros(size, 0xA5)
                        case.callback(gpu.free, ptr)
                        return ptr
                    def values(n):
                        return struct.pack('<' + 'H' * n, *[
                            struct.unpack('<I', struct.pack('<f', rng.uniform(-0.25, 0.25)))[0] >> 16
                            for _ in range(n)])
                    x, wa, wb = upload(values(rows * width)), upload(values(heads * width)), upload(values(heads * width))
                    alog = upload(struct.pack('<' + 'f' * heads, *[rng.uniform(-4, 2) for _ in range(heads)]))
                    bias = upload(struct.pack('<' + 'f' * heads, *[rng.uniform(-4, 4) for _ in range(heads)]))
                    a, b, alpha, beta, fused_alpha, fb = [alloc(rows * heads * 4) for _ in range(6)]
                    for weight, out in [(wa, a), (wb, b)]:
                        gpu.launch('euhedral_linear_bf16_to_float', rows * heads,
                                   [C.c_uint64(p) for p in (x, weight, out)] +
                                   [C.c_uint(v) for v in (rows, width, heads)])
                    gpu.launch('euhedral_gdn_control_fp32', (rows * heads + 127) // 128,
                               [C.c_uint64(p) for p in (a, b, alog, bias, alpha, beta)] +
                               [C.c_uint(rows), C.c_uint(heads)])
                    gpu.launch('euhedral_gdn_project_control_fp32', rows * heads,
                               [C.c_uint64(p) for p in (x, wa, wb, alog, bias, fused_alpha, fb)] +
                               [C.c_uint(v) for v in (rows, width, heads)])
                    self.assertEqual(gpu.download(alpha, rows * heads * 4), gpu.download(fused_alpha, rows * heads * 4))
                    self.assertEqual(gpu.download(beta, rows * heads * 4), gpu.download(fb, rows * heads * 4))
                    if heads % 4 == 0:
                        # The 8-row x 4-head tile shares loads and keeps every output's order.
                        tiled_alpha, tiled_beta = alloc(rows * heads * 4), alloc(rows * heads * 4)
                        gpu.launch('euhedral_gdn_project_control_8x4_fp32', (rows + 7) // 8 * (heads // 4),
                                   [C.c_uint64(p) for p in (x, wa, wb, alog, bias, tiled_alpha, tiled_beta)] +
                                   [C.c_uint(v) for v in (rows, width, heads)])
                        self.assertEqual(gpu.download(alpha, rows * heads * 4), gpu.download(tiled_alpha, rows * heads * 4))
                        self.assertEqual(gpu.download(beta, rows * heads * 4), gpu.download(tiled_beta, rows * heads * 4))

    def test_alpha_is_exp_of_fp32_stored_g(self):
        # Old contract: control stored g in FP32 and recurrence computed expf(g). Alpha must equal that.
        source = (b'#include "linear/kernels.cu"\n#include "gdn/kernels.cu"\n'
                  b'extern "C" __global__ void reference_g(const float* a, const float* aLog, const float* dtBias,\n'
                  b'        float* g, uint32_t rows, uint32_t heads) {\n'
                  b'    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;\n'
                  b'    if (i >= static_cast<uint64_t>(rows) * heads) return;\n'
                  b'    const float shifted = a[i] + dtBias[i % heads];\n'
                  b'    const float softplus = fmaxf(shifted, 0.0f) + log1pf(expf(-fabsf(shifted)));\n'
                  b'    g[i] = -expf(aLog[i % heads]) * softplus;\n}\n'
                  b'extern "C" __global__ void reference_exp(const float* g, float* out, uint32_t count) {\n'
                  b'    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n'
                  b'    if (i < count) out[i] = expf(g[i]);\n}\n')
        rng = random.Random(4417)
        rows, heads = 512, 48
        count = rows * heads
        with contextlib.ExitStack() as scope:
            gpu = Gpu(source)
            scope.callback(gpu.close)
            def upload(data):
                ptr = gpu.upload(data)
                scope.callback(gpu.free, ptr)
                return ptr
            def alloc():
                ptr = gpu.zeros(count * 4, 0xA5)
                scope.callback(gpu.free, ptr)
                return ptr
            a = upload(struct.pack('<' + 'f' * count, *[rng.uniform(-30, 30) for _ in range(count)]))
            b = upload(struct.pack('<' + 'f' * count, *[rng.uniform(-8, 8) for _ in range(count)]))
            alog = upload(struct.pack('<' + 'f' * heads, *[rng.uniform(-6, 3) for _ in range(heads)]))
            bias = upload(struct.pack('<' + 'f' * heads, *[rng.uniform(-6, 6) for _ in range(heads)]))
            g, expected, alpha, beta = alloc(), alloc(), alloc(), alloc()
            gpu.launch('reference_g', (count + 127) // 128, [C.c_uint64(p) for p in (a, alog, bias, g)] +
                       [C.c_uint(rows), C.c_uint(heads)])
            gpu.launch('reference_exp', (count + 127) // 128, [C.c_uint64(g), C.c_uint64(expected), C.c_uint(count)])
            gpu.launch('euhedral_gdn_control_fp32', (count + 127) // 128,
                       [C.c_uint64(p) for p in (a, b, alog, bias, alpha, beta)] + [C.c_uint(rows), C.c_uint(heads)])
            self.assertEqual(gpu.download(expected, count * 4), gpu.download(alpha, count * 4),
                             'alpha differs from expf of the FP32-stored g')

    def test_residual_norm_preserves_both_bf16_boundaries(self):
        source = b'#include "elementwise/kernels.cu"\n#include "norm/kernels.cu"\n'
        with contextlib.ExitStack() as scope:
            gpu = Gpu(source)
            scope.callback(gpu.close)
            symbol = C.c_void_p()
            self.assertEqual(0, gpu.function(C.byref(symbol), gpu.module, b'euhedral_residual_rms_norm_bf16'),
                             "combined residual/norm region is unavailable")
            rng = random.Random(7823)
            for rows, width in [(1, 1), (3, 127), (7, 257), (64, 5120), (256, 5120)]:
                for special in [False, True]:
                    with contextlib.ExitStack() as case:
                        def upload(data):
                            ptr = gpu.upload(data)
                            case.callback(gpu.free, ptr)
                            return ptr
                        def alloc(size):
                            ptr = gpu.zeros(size, 0xA5)
                            case.callback(gpu.free, ptr)
                            return ptr
                        def values(n):
                            patterns = [0, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7fc1, 0xffc7, 0x7f81]
                            return struct.pack('<' + 'H' * n, *[
                                patterns[i % len(patterns)] if special and i % 131 < 9 else
                                struct.unpack('<I', struct.pack('<f', rng.uniform(-8, 8)))[0] >> 16
                                for i in range(n)])
                        count = rows * width
                        residual, delta, weights = upload(values(count)), upload(values(count)), upload(values(width))
                        reference_sum, reference_norm = alloc(count * 2), alloc(count * 2)
                        actual_sum, actual_norm = alloc(count * 2), alloc(count * 2)
                        gpu.launch('euhedral_residual_add_bf16', (count + 127) // 128,
                                   [C.c_uint64(p) for p in (residual, delta, reference_sum)] + [C.c_uint(count)])
                        gpu.launch('euhedral_rms_norm_bf16', rows,
                                   [C.c_uint64(p) for p in (reference_sum, weights, reference_norm)] +
                                   [C.c_uint(rows), C.c_uint(width), C.c_float(1e-6), C.c_float(1)])
                        gpu.launch('euhedral_residual_rms_norm_bf16', rows,
                                   [C.c_uint64(p) for p in (residual, delta, weights, actual_sum, actual_norm)] +
                                   [C.c_uint(rows), C.c_uint(width), C.c_float(1e-6)])
                        self.assertEqual(gpu.download(reference_sum, count * 2), gpu.download(actual_sum, count * 2),
                                         f'residual differs for {(rows, width, special)}')
                        self.assertEqual(gpu.download(reference_norm, count * 2), gpu.download(actual_norm, count * 2),
                                         f'normalization differs for {(rows, width, special)}')

    def test_row_owned_qk_norm_rope_matches_the_per_head_kernels_bitwise(self):
        source = b'#include "attention/kernels.cu"\n'
        with contextlib.ExitStack() as scope:
            gpu = Gpu(source)
            scope.callback(gpu.close)
            rng = random.Random(6011)
            p, u = C.c_uint64, C.c_uint
            for rows, qh, kh, rotary, start in [(1, 24, 4, 64, 0), (3, 4, 2, 64, 100003), (37, 6, 2, 256, 77),
                                                (5, 3, 1, 2, 9)]:
                with contextlib.ExitStack() as case:
                    def upload(data):
                        ptr = gpu.upload(data); case.callback(gpu.free, ptr); return ptr
                    heads, hd = qh + kh, 256
                    count = rows * heads * hd
                    data = struct.pack(f'<{count}H', *[0x3c00 + rng.randrange(1024) + (0x8000 if rng.random() < .5 else 0)
                                                        for _ in range(count)])
                    qn = upload(struct.pack(f'<{hd}H', *[0x3d00 + rng.randrange(512) for _ in range(hd)]))
                    kn = upload(struct.pack(f'<{hd}H', *[0xbd00 + rng.randrange(512) for _ in range(hd)]))
                    outputs = []
                    for name, grid in [('euhedral_attention_qk_norm_rope_bf16', rows * heads),
                                       ('euhedral_attention_qk_norm_rope_rows_bf16', rows)]:
                        qk = upload(data)  # in place, as production launches it
                        # The kernels read the quantum's start position from device memory.
                        position = upload(struct.pack('<Q', start))
                        args = [p(qk), p(qn), p(kn), p(qk), u(rows), u(qh), u(kh), u(hd), u(rotary), p(position), p(0),
                                C.c_float(1e-6), C.c_double(1e7)]
                        gpu.launch(name, grid, args, block=256)
                        outputs.append(gpu.download(qk, count * 2))
                    self.assertEqual(outputs[0], outputs[1], (rows, qh, kh, rotary, start))

    def test_relaxed_row_residual_norm_matches_the_exact_kernel_within_one_ulp(self):
        source = b'#include "elementwise/kernels.cu"\n'
        with contextlib.ExitStack() as scope:
            gpu = Gpu(source)
            scope.callback(gpu.close)
            rng = random.Random(4127)
            def ordered(bits):
                return bits - 0x10000 if bits & 0x8000 else bits
            def is_nan(bits):
                return (bits & 0x7f80) == 0x7f80 and (bits & 0x7f) != 0
            for width in [8, 1024, 5120, 8192]:
                for special in [False, True]:
                    for in_place in [False, True]:
                        with contextlib.ExitStack() as case:
                            def upload(data):
                                ptr = gpu.upload(data)
                                case.callback(gpu.free, ptr)
                                return ptr
                            def alloc(size):
                                ptr = gpu.zeros(size, 0xA5)
                                case.callback(gpu.free, ptr)
                                return ptr
                            patterns = [0, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7fc1, 0xffc7, 0x7f81]
                            def values(n):
                                return struct.pack('<' + 'H' * n, *[
                                    patterns[i % len(patterns)] if special and i % 131 < 9 else
                                    struct.unpack('<I', struct.pack('<f', rng.uniform(-8, 8)))[0] >> 16
                                    for i in range(n)])
                            residual_data = values(width)
                            exact_residual, delta, weights = upload(residual_data), upload(values(width)), upload(values(width))
                            exact_sum, exact_norm, row_norm = alloc(width * 2), alloc(width * 2), alloc(width * 2)
                            row_residual = upload(residual_data)
                            row_sum = row_residual if in_place else alloc(width * 2)
                            gpu.launch('euhedral_residual_rms_norm_bf16', 1,
                                       [C.c_uint64(p) for p in (exact_residual, delta, weights, exact_sum, exact_norm)] +
                                       [C.c_uint(1), C.c_uint(width), C.c_float(1e-6)])
                            gpu.launch('euhedral_residual_rms_norm_row_bf16', 1,
                                       [C.c_uint64(p) for p in (row_residual, delta, weights, row_sum, row_norm)] +
                                       [C.c_uint(1), C.c_uint(width), C.c_float(1e-6)],
                                       block=(width // 8 + 31) // 32 * 32)
                            case_name = (width, special, in_place)
                            self.assertEqual(gpu.download(exact_sum, width * 2), gpu.download(row_sum, width * 2),
                                             f'residual differs for {case_name}')
                            expected = struct.unpack(f'<{width}H', gpu.download(exact_norm, width * 2))
                            actual = struct.unpack(f'<{width}H', gpu.download(row_norm, width * 2))
                            for index, (e, a) in enumerate(zip(expected, actual)):
                                if is_nan(e) or is_nan(a):
                                    self.assertTrue(is_nan(e) and is_nan(a), f'NaN mismatch at {index} for {case_name}')
                                else:
                                    self.assertLessEqual(abs(ordered(e) - ordered(a)), 1,
                                                         f'more than one BF16 ulp at {index} for {case_name}')


if __name__ == '__main__':
    unittest.main()
