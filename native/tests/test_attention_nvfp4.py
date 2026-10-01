"""NVFP4 cache exact-format and attention mathematical-oracle tests.

The CPU oracle constructs H256 from its definition, rather than reproducing the
CUDA butterfly or reduction schedule. Cache rows are 128 code + 16 scale bytes.
"""
import contextlib
import ctypes as C
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from test_q3_primitives import Gpu, NVRTC, CUDA

P, U = C.c_uint64, C.c_uint
PAGE_TOKENS = 256
ROW_BYTES = 144
if np is not None:
    MAGNITUDES = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
    SCALES = np.array([(b & 7) * 2.0**-9 if b < 8 else
                       (1 + (b & 7) / 8) * 2.0**((b >> 3) - 7)
                       for b in range(127)], dtype=np.float64)
    H = np.array([[(-1.0 if (i & j).bit_count() & 1 else 1.0) / 16
                   for j in range(256)] for i in range(256)])


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded, (rounded.astype(np.uint32) << 16).view(np.float32)


def nearest_even(values, representable):
    # Saturate before measuring distances: at huge magnitudes floating-point
    # subtraction can make every candidate distance equal and select code zero.
    clipped = np.clip(np.asarray(values), representable[0], representable[-1])
    distance = abs(clipped[..., None] - representable)
    minimum = distance.min(axis=-1, keepdims=True)
    # Prefer an even code for exact midpoint ties.
    rank = np.arange(len(representable)) % 2
    return np.argmin(np.where(distance == minimum, rank, 2), axis=-1)


def pack_row(row):
    rotated = np.asarray(row, dtype=np.float64) @ H
    groups = rotated.reshape(16, 16)
    maximum = abs(groups).max(axis=1)
    raw_scale = (maximum.astype(np.float32) / np.float32(6)).astype(np.float32)
    scales = nearest_even(np.clip(raw_scale, 2.0**-9, 448), SCALES).astype(np.uint8)
    scales[maximum == 0] = 0
    represented = SCALES[scales]
    divided = np.divide(groups.astype(np.float32), represented.astype(np.float32)[:, None],
                        out=np.zeros((16, 16), np.float32), where=represented[:, None] != 0)
    codes = nearest_even(abs(divided), MAGNITUDES).astype(np.uint8)
    codes |= np.signbit(divided).astype(np.uint8) * 8
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).ravel()
    return np.concatenate([packed, scales]).tobytes()


def unpack_rows(data):
    rows = np.frombuffer(data, np.uint8).reshape(-1, ROW_BYTES)
    codes = np.empty((len(rows), 256), np.uint8)
    codes[:, ::2] = rows[:, :128] & 15
    codes[:, 1::2] = rows[:, :128] >> 4
    return MAGNITUDES[codes & 7] * np.where(codes & 8, -1, 1) * np.repeat(SCALES[rows[:, 128:]], 16, axis=1)


@unittest.skipIf(NVRTC is None or CUDA is None or np is None, "CUDA/NVRTC or NumPy unavailable")
class Nvfp4AttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "qwen_attention_ops.cu"\n' + br'''
extern "C" __global__ void probe_nvfp4_encode(const float* input, unsigned char* output, unsigned int count) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = nvfp4kv::e2m1_encode(input[i]);
}
extern "C" __global__ void probe_nvfp4_groups(const float* input, unsigned char* output, unsigned int count) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto value = nvfp4kv::quantize_group16(input + i * 16);
    for (int b = 0; b < 4; b++) {
        output[i * 9 + b] = value.codes_lo >> (b * 8);
        output[i * 9 + 4 + b] = value.codes_hi >> (b * 8);
    }
    output[i * 9 + 8] = value.scale;
}
extern "C" __global__ void probe_nvfp4_hadamard(
        const float* input, float* forward, float* inverse, unsigned int rows) {
    const unsigned int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= rows) return;
    float values[8];
#pragma unroll
    for (int d = 0; d < 8; d++) values[d] = input[row * 256 + lane + d * 32];
    nvfp4kv::hadamard256(values, lane);
#pragma unroll
    for (int d = 0; d < 8; d++) forward[row * 256 + lane + d * 32] = values[d];
    nvfp4kv::hadamard256(values, lane);
#pragma unroll
    for (int d = 0; d < 8; d++) inverse[row * 256 + lane + d * 32] = values[d];
}
extern "C" __global__ void probe_nvfp4_decode(float* fp32, unsigned short* fp16) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 16 * 256) return;
    const float value = nvfp4kv::e2m1_decode(i % 16) * nvfp4kv::e4m3_decode(i / 16);
    fp32[i] = value;
    fp16[i] = __half_as_ushort(__float2half_rn(value));
}
''')

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def owned(self, scope, ptr):
        scope.callback(self.gpu.free, ptr)
        return ptr

    def paged(self, scope, tokens, heads, fill=0xA5):
        pages = [self.owned(scope, self.gpu.zeros(PAGE_TOKENS * heads * ROW_BYTES, fill))
                 for _ in range((tokens + PAGE_TOKENS - 1) // PAGE_TOKENS)]
        table = self.owned(scope, self.gpu.upload(np.array(pages, dtype=np.uint64).tobytes()))
        return table, pages

    def download_pages(self, pages, heads):
        return b''.join(self.gpu.download(p, PAGE_TOKENS * heads * ROW_BYTES) for p in pages)

    def test_quantizer_midpoints_saturation_and_scale_rounding(self):
        mids = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float32)
        positive = np.concatenate(([0., 1e-30, 1e30], np.nextafter(mids, -np.inf), mids,
                                   np.nextafter(mids, np.inf))).astype(np.float32)
        inputs = np.concatenate((positive, -positive))
        with contextlib.ExitStack() as scope:
            source = self.owned(scope, self.gpu.upload(inputs.tobytes()))
            output = self.owned(scope, self.gpu.zeros(len(inputs)))
            self.gpu.launch('probe_nvfp4_encode', (len(inputs) + 127) // 128,
                            [P(source), P(output), U(len(inputs))])
            expected = nearest_even(abs(inputs), MAGNITUDES).astype(np.uint8)
            expected |= np.signbit(inputs).astype(np.uint8) * 8
            self.assertEqual(expected.tobytes(), self.gpu.download(output, len(inputs)))
        raw_scales = np.concatenate(([0, 1e-30, 1e30], SCALES,
                                     (SCALES[:-1] + SCALES[1:]) / 2)).astype(np.float32)
        groups = np.tile(np.linspace(-1, 1, 16, dtype=np.float32), (len(raw_scales), 1))
        groups *= (raw_scales * np.float32(6))[:, None]
        with contextlib.ExitStack() as scope:
            source = self.owned(scope, self.gpu.upload(groups.tobytes()))
            output = self.owned(scope, self.gpu.zeros(len(groups) * 9))
            self.gpu.launch('probe_nvfp4_groups', (len(groups) + 127) // 128,
                            [P(source), P(output), U(len(groups))])
            expected = []
            for group in groups:
                maximum = abs(group).max()
                if maximum == 0:
                    expected.append(bytes(9)); continue
                scale = int(nearest_even(np.clip(maximum / np.float32(6), 2.**-9, 448), SCALES))
                divided = group / np.float32(SCALES[scale])
                codes = nearest_even(abs(divided), MAGNITUDES).astype(np.uint8)
                codes |= np.signbit(divided).astype(np.uint8) * 8
                expected.append((codes[::2] | (codes[1::2] << 4)).tobytes() + bytes([scale]))
            self.assertEqual(b''.join(expected), self.gpu.download(output, len(groups) * 9))

    def test_hadamard_basis_order_normalization_and_inverse(self):
        rng = np.random.default_rng(19)
        values = np.concatenate((np.eye(256, dtype=np.float32) * 16,
                                 rng.integers(-32, 33, (7, 256)).astype(np.float32) / 32))
        size = values.nbytes
        with contextlib.ExitStack() as scope:
            source = self.owned(scope, self.gpu.upload(values.tobytes()))
            forward = self.owned(scope, self.gpu.zeros(size + 1024, 0xA5))
            inverse = self.owned(scope, self.gpu.zeros(size + 1024, 0xA5))
            self.gpu.launch('probe_nvfp4_hadamard', (len(values) + 3) // 4,
                            [P(source), P(forward), P(inverse), U(len(values))])
            actual = np.frombuffer(self.gpu.download(forward, size), np.float32).reshape(-1, 256)
            restored = np.frombuffer(self.gpu.download(inverse, size), np.float32).reshape(-1, 256)
            np.testing.assert_array_equal(values.astype(np.float64) @ H, actual)
            np.testing.assert_array_equal(values, restored)
            for pointer in [forward, inverse]:
                self.assertEqual(bytes([0xA5]) * 1024, self.gpu.download(pointer + size, 1024))

    def test_every_finite_code_scale_is_exact_without_local_storage(self):
        count = 16 * 256
        with contextlib.ExitStack() as scope:
            fp32 = self.owned(scope, self.gpu.zeros(count * 4))
            fp16 = self.owned(scope, self.gpu.zeros(count * 2))
            self.gpu.launch('probe_nvfp4_decode', (count + 127) // 128, [P(fp32), P(fp16)])
            codes = np.arange(16)
            magnitude = np.copysign(MAGNITUDES[codes & 7], np.where(codes & 8, -1., 1.))
            positive_scales = np.append(SCALES, np.nan)
            all_scales = np.concatenate((positive_scales, -positive_scales))
            expected = (all_scales[:, None] * magnitude[None, :]).ravel()
            finite = np.isfinite(expected)
            for pointer, dtype, width in [(fp32, np.float32, 4), (fp16, np.float16, 2)]:
                actual = np.frombuffer(self.gpu.download(pointer, count * width), dtype)
                self.assertEqual(expected[finite].astype(dtype).tobytes(), actual[finite].tobytes())
                self.assertTrue(np.isnan(actual[~finite]).all())
            symbol = C.c_void_p()
            self.assertEqual(0, self.gpu.function(C.byref(symbol), self.gpu.module, b'probe_nvfp4_decode'))
            attribute = CUDA.cuFuncGetAttribute
            attribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_void_p]
            attribute.restype = C.c_int
            local = C.c_int()
            self.assertEqual(0, attribute(C.byref(local), 3, symbol))  # CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
            self.assertEqual(0, local.value, 'FP4 decode must not materialize a thread-local lookup table')
            for name in [b'euhedral_attention_prefill32_nvfp4', b'euhedral_attention_prefill32_nvfp4_exact',
                         b'euhedral_attention_decode_nvfp4', b'euhedral_attention_decode_nvfp4_exact']:
                self.assertEqual(0, self.gpu.function(C.byref(symbol), self.gpu.module, name))
                self.assertEqual(0, attribute(C.byref(local), 3, symbol))
                self.assertEqual(0, local.value, name.decode() + ' must not use thread-local decode tables')

    def test_attention_matches_fp64_represented_cache_oracle(self):
        for name in [b'euhedral_attention_prefill_nvfp4', b'euhedral_attention_prefill32_nvfp4', b'euhedral_attention_decode_nvfp4',
                     b'euhedral_attention_merge_nvfp4']:
            symbol = C.c_void_p()
            self.assertEqual(0, self.gpu.function(C.byref(symbol), self.gpu.module, name), name)
        rng = np.random.default_rng(107)
        # Chunk edges, nonzero starts, page edges and long split-KV tails.
        for rows, start, heads, query_heads in [(17, 0, 4, 24), (33, 1, 4, 24), (3, 255, 2, 6),
                                                (1, 0, 4, 24), (1, 255, 2, 6), (1, 1022, 4, 24),
                                                (1, 1023, 4, 24), (1, 1024, 4, 24), (1, 65536, 1, 6)]:
            with self.subTest(rows=rows, start=start), contextlib.ExitStack() as scope:
                length = start + rows
                width = (heads + query_heads) * 256
                qbits, q = bf16(rng.normal(0, .3, (rows, width)))
                gbits, gate = bf16(rng.normal(0, .2, (rows, width)))
                # Independent legal represented cache, including both signs and scale extremes.
                count = ((length + 255) // 256) * 256 * heads
                kbytes = rng.integers(0, 256, (count, 144), dtype=np.uint8)
                vbytes = rng.integers(0, 256, (count, 144), dtype=np.uint8)
                kbytes[:, 128:] = rng.integers(8, 40, (count, 16))
                vbytes[:, 128:] = rng.integers(8, 40, (count, 16))
                upload = lambda data: self.owned(scope, self.gpu.upload(data))
                qp, gp = upload(qbits.tobytes()), upload(gbits.tobytes())
                kt, kp = self.paged(scope, length, heads)
                vt, vp = self.paged(scope, length, heads)
                for data, pages in [(kbytes, kp), (vbytes, vp)]:
                    for i, ptr in enumerate(pages):
                        chunk = data[i * 256 * heads:(i + 1) * 256 * heads].tobytes()
                        self.assertEqual(0, self.gpu.htod(ptr, C.create_string_buffer(chunk), len(chunk)))
                output = self.owned(scope, self.gpu.zeros(rows * query_heads * 256 * 2))
                args = [P(qp), P(gp), P(kt), P(vt), P(output), U(rows), U(query_heads),
                        U(heads), U(256), U(length), P(start)]
                observed = []
                def capture(label):
                    actual = (np.frombuffer(self.gpu.download(output, rows * query_heads * 512), np.uint16)
                              .astype(np.uint32) << 16).view(np.float32).reshape(rows, query_heads, 256)
                    observed.append((label, actual))
                if rows > 1:
                    self.gpu.launch('euhedral_attention_prefill_nvfp4', ((rows + 15) // 16) * query_heads, args)
                    control = self.gpu.download(output, rows * query_heads * 512)
                    capture('prefill16')
                    self.gpu.launch('euhedral_attention_prefill32_nvfp4_exact', ((rows + 31) // 32) * query_heads, args)
                    self.assertEqual(control, self.gpu.download(output, rows * query_heads * 512))
                    capture('prefill32_exact')
                    # The relaxed kernel sums each row's softmax denominator in four partial sums.
                    self.gpu.launch('euhedral_attention_prefill32_nvfp4', ((rows + 31) // 32) * query_heads, args)
                    capture('prefill32')
                    np.testing.assert_allclose(observed[-1][1], observed[-2][1], rtol=1e-2, atol=1e-4)
                else:
                    for splits in sorted({1, 3, min(64, (length + 47) // 48), min(64, length)}):
                        scratch = self.owned(scope, self.gpu.zeros(query_heads * splits * 258 * 4, 0xA5))
                        for name, grid in [('euhedral_attention_decode_nvfp4', query_heads * splits),
                                           ('euhedral_attention_decode_nvfp4_exact', query_heads * splits)]:
                            self.gpu.launch(name, grid, args + [P(scratch), U(splits)])
                            self.gpu.launch('euhedral_attention_merge_nvfp4', query_heads,
                                            [P(gp), P(output), P(scratch), U(query_heads), U(heads), U(splits)])
                            capture(f'{name}-splits-{splits}')
                keys = unpack_rows(kbytes.tobytes()).reshape(-1, heads, 256)[:length]
                values = unpack_rows(vbytes.tobytes()).reshape(-1, heads, 256)[:length]
                expected = np.empty((rows, query_heads, 256), dtype=np.float64)
                for h in range(query_heads):
                    kh = h // (query_heads // heads)
                    qr = q[:, h * 256:(h + 1) * 256].astype(np.float64) @ H
                    scores = qr @ keys[:, kh].T / 16
                    scores[np.arange(length)[None, :] > (start + np.arange(rows))[:, None]] = -np.inf
                    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
                    weights /= weights.sum(axis=1, keepdims=True)
                    expected[:, h] = ((weights @ values[:, kh]) @ H) / (1 + np.exp(-gate[:, h * 256:(h + 1) * 256]))
                for label, actual in observed:
                    with self.subTest(route=label):
                        np.testing.assert_allclose(actual, expected, rtol=.02, atol=.0005)

    def test_split_append_and_prefill_preserve_cache_and_continuation(self):
        rows, heads, query_heads = 261, 1, 6
        width = (heads + query_heads) * 256
        rng = np.random.default_rng(101)
        qbits, qvalues = bf16(rng.integers(-32, 33, (rows + 1, width)) / 32)
        gbits, gates = bf16(rng.integers(-32, 33, (rows + 1, width)) / 32)
        with contextlib.ExitStack() as scope:
            own = lambda ptr: self.owned(scope, ptr)
            q, gate = own(self.gpu.upload(qbits.tobytes())), own(self.gpu.upload(gbits.tobytes()))
            results = []
            for chunks in [[261], [253, 3, 5]]:
                kt, kp = self.paged(scope, rows, heads)
                vt, vp = self.paged(scope, rows, heads)
                out = own(self.gpu.zeros(rows * query_heads * 512, 0xA5))
                start = 0
                for count in chunks:
                    old_k, old_v = self.download_pages(kp, heads), self.download_pages(vp, heads)
                    offset = start * width * 2
                    self.gpu.launch('euhedral_attention_kv_append_nvfp4', (count * heads + 3) // 4,
                                    [P(q + offset), P(gate + offset), P(kt), P(vt), U(count),
                                     U(query_heads * 256), U(heads * 256), P(start)])
                    current_k, current_v = self.download_pages(kp, heads), self.download_pages(vp, heads)
                    prefix = start * heads * ROW_BYTES
                    self.assertEqual(old_k[:prefix], current_k[:prefix])
                    self.assertEqual(old_v[:prefix], current_v[:prefix])
                    self.gpu.launch('euhedral_attention_prefill32_nvfp4', ((count + 31) // 32) * query_heads,
                                    [P(q + offset), P(gate + offset), P(kt), P(vt),
                                     P(out + start * query_heads * 512), U(count), U(query_heads), U(heads),
                                     U(256), U(start + count), P(start)])
                    start += count
                prefill = self.gpu.download(out, rows * query_heads * 512)
                offset = rows * width * 2
                self.gpu.launch('euhedral_attention_kv_append_nvfp4', 1,
                                [P(q + offset), P(gate + offset), P(kt), P(vt), U(1),
                                 U(query_heads * 256), U(heads * 256), P(rows)])
                after_k, after_v = self.download_pages(kp, heads), self.download_pages(vp, heads)
                self.assertEqual(current_k[:rows * ROW_BYTES], after_k[:rows * ROW_BYTES])
                self.assertEqual(current_v[:rows * ROW_BYTES], after_v[:rows * ROW_BYTES])
                scratch = own(self.gpu.zeros(query_heads * 2 * 258 * 4, 0xA5))
                self.gpu.launch('euhedral_attention_decode_nvfp4', query_heads * 2,
                                [P(q + offset), P(gate + offset), P(kt), P(vt), P(out), U(1),
                                 U(query_heads), U(heads), U(256), U(rows + 1), P(rows), P(scratch), U(2)])
                self.gpu.launch('euhedral_attention_merge_nvfp4', query_heads,
                                [P(gate + offset), P(out), P(scratch), U(query_heads), U(heads), U(2)])
                decoded = self.gpu.download(out, query_heads * 512)
                actual = (np.frombuffer(decoded, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(query_heads, 256)
                keys = unpack_rows(after_k[:(rows + 1) * ROW_BYTES])
                values = unpack_rows(after_v[:(rows + 1) * ROW_BYTES])
                for head in range(query_heads):
                    query = qvalues[rows, head * 256:(head + 1) * 256].astype(np.float64) @ H
                    scores = query @ keys.T / 16
                    weights = np.exp(scores - scores.max()); weights /= weights.sum()
                    expected = (weights @ values) @ H
                    expected /= 1 + np.exp(-gates[rows, head * 256:(head + 1) * 256].astype(np.float64))
                    np.testing.assert_allclose(actual[head], expected, rtol=.02, atol=.0005)
                results.append((after_k, after_v, prefill, decoded))
            self.assertEqual(results[0][0], results[1][0])
            self.assertEqual(results[0][1], results[1][1])
            self.assertEqual(results[0][2], results[1][2])
            self.assertEqual(results[0][3], results[1][3])

    def test_append_crosses_pages_and_preserves_prefix_tail(self):
        symbol = C.c_void_p()
        self.assertEqual(0, self.gpu.function(C.byref(symbol), self.gpu.module,
                                            b'euhedral_attention_kv_append_nvfp4'),
                         'NVFP4 cache append kernel is missing')
        rng = np.random.default_rng(4102)
        heads, query_heads, start, rows = 4, 24, 253, 9
        width = (heads + query_heads) * 256
        # Dyadic represented inputs make the H256 sums exact in FP32.
        qbits, q = bf16(rng.integers(-32, 33, (rows, width)) / 32)
        vbits, v = bf16(rng.integers(-32, 33, (rows, width)) / 32)
        q[:, query_heads * 256:][0] = 0
        qbits, q = bf16(q)
        with contextlib.ExitStack() as scope:
            upload = lambda data: self.owned(scope, self.gpu.upload(data))
            qp, vp = upload(qbits.tobytes()), upload(vbits.tobytes())
            kt, kp = self.paged(scope, start + rows + 5, heads)
            vt, vpages = self.paged(scope, start + rows + 5, heads)
            self.gpu.launch('euhedral_attention_kv_append_nvfp4', (rows * heads + 3) // 4,
                            [P(qp), P(vp), P(kt), P(vt), U(rows), U(query_heads * 256),
                             U(heads * 256), P(start)])
            for source, pages in [(q, kp), (v, vpages)]:
                data = self.download_pages(pages, heads)
                expected = b''.join(pack_row(row) for row in source[:, query_heads * 256:].reshape(-1, 256))
                begin, end = start * heads * ROW_BYTES, (start + rows) * heads * ROW_BYTES
                self.assertEqual(data[:begin], bytes([0xA5]) * begin)
                self.assertEqual(data[begin:end], expected)
                self.assertEqual(data[end:], bytes([0xA5]) * (len(data) - end))


if __name__ == '__main__':
    unittest.main()
