"""Bitwise checks for macro-prefill regions against the unfused numerical boundaries."""
import contextlib
import ctypes as C
import random
import struct
import unittest

from test_q3_primitives import Gpu, NVRTC, CUDA, ROOT, _check


@unittest.skipIf(NVRTC is None or CUDA is None, "CUDA/NVRTC unavailable")
class QwenRegionsTest(unittest.TestCase):
    def test_attention_producers_write_only_reserved_cache_rows(self):
        from test_q45_kernels import make_weights
        source = b'#include "q45/kernels.cu"\n#include "qwen_attention_ops.cu"\n'
        with contextlib.ExitStack() as scope:
            gpu = Gpu(source)
            scope.callback(gpu.close)
            symbol = C.c_void_p()
            for name in [b'euhedral_attention_qk_norm_cache_bf16', b'euhedral_attention_value_cache_bf16']:
                self.assertEqual(0, gpu.function(C.byref(symbol), gpu.module, name), 'producer cache path unavailable')
            rng = random.Random(971)
            for rows in [64, 65, 256]:
                with contextlib.ExitStack() as case:
                    def upload(data):
                        pointer = gpu.upload(data); case.callback(gpu.free, pointer); return pointer
                    def alloc(size):
                        pointer = gpu.zeros(size, 0xA5); case.callback(gpu.free, pointer); return pointer
                    p, u = C.c_uint64, C.c_uint
                    width, qh, kh, hd, start, tail = 256, 4, 2, 128, 5, 7
                    qw, kw, projected = qh * hd, kh * hd, (qh + kh) * hd
                    xbits = [0x3d00 + rng.randrange(768) + (0x8000 if i % 3 == 0 else 0) for i in range(rows * width)]
                    x = upload(struct.pack('<' + 'H' * len(xbits), *xbits))
                    qkbits = [0x3d00 + rng.randrange(768) for _ in range(rows * projected)]
                    data = struct.pack('<' + 'H' * len(qkbits), *qkbits)
                    qk, fused_qk = upload(data), upload(data)
                    qn, kn = upload(b'\x80\x3d' * hd), upload(b'\x00\xbe' * hd)
                    weight = upload(make_weights(rng, 5, width, projected))
                    normalized, gv, fused_gv = [alloc(rows * projected * 2) for _ in range(3)]
                    cache_bytes = (start + rows + tail) * kw * 2
                    key, value, fused_key, fused_value = [alloc(cache_bytes) for _ in range(4)]
                    norm_args = [p(qk), p(qn), p(kn), p(normalized), u(rows), u(qh), u(kh), u(hd), u(32), p(start), C.c_float(1e-6), C.c_double(1e6)]
                    gpu.launch('euhedral_attention_qk_norm_rope_bf16', rows * (qh + kh), norm_args)
                    gpu.launch('euhedral_q5_prefill_64_exact', ((rows + 63) // 64) * (projected // 32),
                               [p(x), p(weight), p(gv), u(rows), u(width), u(projected)])
                    gpu.launch('euhedral_attention_kv_append_bf16', (rows * kw + 127) // 128,
                               [p(normalized), p(gv), p(key), p(value), u(rows), u(qw), u(kw), p(start)])
                    norm_args[0] = p(fused_qk); norm_args[3] = p(fused_qk); norm_args.append(p(fused_key))
                    gpu.launch('euhedral_attention_qk_norm_cache_bf16', rows * (qh + kh), norm_args)
                    gpu.launch('euhedral_attention_value_cache_bf16', ((rows + 63) // 64) * (projected // 32),
                               [p(x), p(weight), p(fused_gv), u(rows), u(width), u(projected), p(fused_value), u(qw), p(start)])
                    for original, fused in [(key, fused_key), (value, fused_value)]:
                        self.assertEqual(gpu.download(original, cache_bytes), gpu.download(fused, cache_bytes),
                                         'cache prefix/tail or appended rows differ')
                    for original, fused in [(normalized, fused_qk), (gv, fused_gv)]:
                        a, b = gpu.download(original, rows * projected * 2), gpu.download(fused, rows * projected * 2)
                        for row in range(rows):
                            begin = row * projected * 2
                            self.assertEqual(a[begin:begin + qw * 2], b[begin:begin + qw * 2])

    def test_gate_up_region_keeps_bf16_round_before_swiglu(self):
        source = b'#include "qwen_ffn.cu"\n#include "q3/kernels.cu"\n#include "qwen_elementwise.cu"\n'
        with contextlib.ExitStack() as scope:
            gpu = Gpu(source)
            scope.callback(gpu.close)
            symbol = C.c_void_p()
            self.assertEqual(0, gpu.function(C.byref(symbol), gpu.module, b'euhedral_q3_gate_up_swiglu_bf16'),
                             'gate/up SwiGLU region is unavailable')
            rng = random.Random(524)
            for rows, width, outputs in [(1, 128, 32), (65, 256, 96), (65, 256, 192), (129, 256, 128), (256, 5120, 64)]:
                with contextlib.ExitStack() as case:
                    def upload(data):
                        ptr = gpu.upload(data)
                        case.callback(gpu.free, ptr)
                        return ptr
                    def alloc(size):
                        ptr = gpu.zeros(size, 0xA5)
                        case.callback(gpu.free, ptr)
                        return ptr
                    groups = outputs * (width // 64)
                    scale = (groups * 24 + 255) & ~255
                    packed = rng.randbytes(groups * 24) + bytes(scale - groups * 24)
                    packed += b'\x00\x28' * groups
                    w = upload(packed)
                    gate_up, reference, actual = alloc(rows * outputs * 2), alloc(rows * outputs), alloc(rows * outputs)
                    for special in [False, True]:
                        bits = [0x3f80, 0xbf80, 0x3f81, 0x3e00, 0, 1, 0x8000, 0xbe01]
                        if special:
                            bits += [0x7f80, 0xff80, 0x7fc1, 0xffc7, 0x7f81]
                        x = upload(struct.pack('<' + 'H' * (rows * width), *[rng.choice(bits) for _ in range(rows * width)]))
                        args = [C.c_uint64(x), C.c_uint64(w), C.c_uint64(gate_up), C.c_uint(rows),
                                C.c_uint(width), C.c_uint(outputs), C.c_uint64(scale)]
                        gpu.launch('euhedral_q3_prefill_64_k32_cb_exact', ((rows + 63) // 64) * (outputs // 32), args)
                        gpu.launch('euhedral_swiglu_bf16', (rows * (outputs // 2) + 127) // 128,
                                   [C.c_uint64(gate_up), C.c_uint64(reference), C.c_uint(rows), C.c_uint(outputs // 2)])
                        args[2] = C.c_uint64(actual)
                        gpu.launch('euhedral_q3_gate_up_swiglu_bf16', ((rows + 63) // 64) * (outputs // 32), args)
                        self.assertEqual(gpu.download(reference, rows * outputs), gpu.download(actual, rows * outputs),
                                         f'gate/up BF16 boundary differs for {(rows, width, outputs, special)}')
                        if outputs % 64 == 0:
                            for tile in (64, 128):
                                gpu.launch(f'euhedral_q3_gate_up_swiglu_{tile}x32',
                                           ((rows + tile - 1) // tile) * (outputs // 64), args)
                                self.assertEqual(gpu.download(reference, rows * outputs),
                                                 gpu.download(actual, rows * outputs),
                                                 f'paired {tile} boundary differs for {(rows, width, outputs, special)}')

    def test_control_region_preserves_fp32_projection_control_order(self):
        source = b'#include "qwen_layer_linear.cu"\n#include "qwen_gdn_ops.cu"\n'
        with contextlib.ExitStack() as scope:
            gpu = Gpu(source)
            scope.callback(gpu.close)
            symbol = C.c_void_p()
            self.assertEqual(0, gpu.function(C.byref(symbol), gpu.module, b'euhedral_gdn_project_control_fp32'),
                             'combined GDN control region is unavailable')
            rng = random.Random(9921)
            for rows, width, heads in [(1, 129, 3), (7, 256, 8), (256, 5120, 32)]:
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

    def test_alpha_is_exp_of_fp32_stored_g(self):
        # Old contract: control stored g in FP32 and recurrence computed expf(g). Alpha must equal that.
        source = (b'#include "qwen_layer_linear.cu"\n#include "qwen_gdn_ops.cu"\n'
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
        source = b'#include "qwen_elementwise.cu"\n#include "rms_norm_bf16.cu"\n'
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

    def test_relaxed_row_residual_norm_matches_the_exact_kernel_within_one_ulp(self):
        source = b'#include "qwen_elementwise.cu"\n'
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
