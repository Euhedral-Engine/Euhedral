"""BF16 attention cache: exact append, the FP16 tile widening, every attention route against a float64 oracle of
causal GQA attention with the sigmoid gate, row-exact verification twins and finite outputs for extreme keys.

Cache rows are the 256 BF16 values of a head row as computed (512 bytes), with no rotation.
"""
import contextlib
import ctypes as C
import struct
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from gpu_harness import Gpu, NVRTC, CUDA

P, U = C.c_uint64, C.c_uint
PAGE_TOKENS = 256
FA2_SHARED = 2 * 2 * 32 * 264 * 2  # fa2::kSharedBytes
ROW_BYTES = 512
# Relative RMS error a route may add to the float64 oracle beyond the BF16 rounding of its output.
EXCESS = 1e-4


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded, (rounded.astype(np.uint32) << 16).view(np.float32)


def from_bf16(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def relative_rms(actual, expected):
    return np.sqrt(np.mean((actual - expected) ** 2) / np.mean(expected ** 2))


@unittest.skipIf(NVRTC is None or CUDA is None or np is None, "CUDA/NVRTC or NumPy unavailable")
class Bf16AttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "attention/kernels.cu"\n' + br'''
extern "C" __global__ void probe_bf16_widen(const unsigned* input, unsigned* output, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = bf16kv::widen2(input[i]);
}
''')

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def owned(self, scope, ptr):
        scope.callback(self.gpu.free, ptr)
        return ptr

    def position(self, scope, value):
        return self.owned(scope, self.gpu.upload(struct.pack('<Q', value)))

    def paged(self, scope, rows, heads, fill=0xA5):
        """Pages holding `rows` (tokens x heads x 256 BF16 bits, or None to leave the fill), and their table."""
        tokens = len(rows) if rows is not None else 0
        pages = []
        for page in range(max(1, (tokens + PAGE_TOKENS - 1) // PAGE_TOKENS)):
            data = np.full((PAGE_TOKENS, heads, 256), fill * 0x0101, np.uint16)
            if rows is not None:
                chunk = rows[page * PAGE_TOKENS:(page + 1) * PAGE_TOKENS]
                data[:len(chunk)] = chunk
            pages.append(self.owned(scope, self.gpu.upload(data.tobytes())))
        table = self.owned(scope, self.gpu.upload(np.array(pages, dtype=np.uint64).tobytes()))
        return table, pages

    def test_widening_is_exact_in_fp16_range_and_saturates_beyond(self):
        bits = np.arange(1 << 16, dtype=np.uint32)
        words = bits | (bits[::-1] << 16)
        with contextlib.ExitStack() as scope:
            source = self.owned(scope, self.gpu.upload(words.tobytes()))
            output = self.owned(scope, self.gpu.zeros(words.nbytes))
            self.gpu.launch('probe_bf16_widen', (len(words) + 127) // 128, [P(source), P(output), U(len(words))])
            actual = np.frombuffer(self.gpu.download(output, words.nbytes), np.uint32)
        for half, source_bits in [(actual & 0xffff, bits), (actual >> 16, bits[::-1])]:
            value = from_bf16(source_bits).astype(np.float64)
            widened = half.astype(np.uint16).view(np.float16).astype(np.float64)
            finite = np.isfinite(value)
            expected = np.clip(value[finite], -65504, 65504).astype(np.float16).astype(np.float64)
            np.testing.assert_array_equal(expected, widened[finite])
            normal = finite & (np.abs(value) >= 2.0**-14) & (np.abs(value) <= 65504)
            np.testing.assert_array_equal(value[normal], widened[normal], 'exact over the FP16 normal range')
            self.assertTrue(np.isfinite(widened).all(), 'every widened value is finite')

    def test_append_stores_rows_as_computed_across_pages(self):
        rng = np.random.default_rng(4102)
        heads, query_heads, start, rows = 4, 24, 253, 9
        width = (heads + query_heads) * 256
        qbits, _ = bf16(rng.normal(0, 3, (rows, width)))
        vbits, _ = bf16(rng.normal(0, 3, (rows, width)))
        with contextlib.ExitStack() as scope:
            qp, vp = self.owned(scope, self.gpu.upload(qbits.tobytes())), self.owned(scope, self.gpu.upload(vbits.tobytes()))
            fill = np.full((start + rows + 5, heads, 256), 0xA5A5, np.uint16)
            kt, kpages = self.paged(scope, fill, heads)
            vt, vpages = self.paged(scope, fill, heads)
            self.gpu.launch('euhedral_attention_kv_append_bf16', (rows * heads + 3) // 4,
                            [P(qp), P(vp), P(kt), P(vt), U(rows), U(query_heads * 256), U(heads * 256),
                             P(self.position(scope, start))])
            for source, pages in [(qbits, kpages), (vbits, vpages)]:
                data = b''.join(self.gpu.download(p, PAGE_TOKENS * heads * ROW_BYTES) for p in pages)
                expected = source[:, query_heads * 256:].tobytes()
                begin, end = start * heads * ROW_BYTES, (start + rows) * heads * ROW_BYTES
                self.assertEqual(bytes([0xA5]) * begin, data[:begin])
                self.assertEqual(expected, data[begin:end])
                self.assertEqual(bytes([0xA5]) * (len(data) - end), data[end:])

    def oracle(self, q, gate, keys, values, rows, start, query_heads, heads):
        """Causal GQA attention in float64 over the cached values, gated by sigmoid(gate)."""
        length = start + rows
        expected = np.empty((rows, query_heads, 256))
        for h in range(query_heads):
            kh = h // (query_heads // heads)
            scores = q[:, h * 256:(h + 1) * 256].astype(np.float64) @ keys[:length, kh].astype(np.float64).T / 16
            scores[np.arange(length)[None, :] > (start + np.arange(rows))[:, None]] = -np.inf
            weights = np.exp(scores - scores.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)
            expected[:, h] = (weights @ values[:length, kh].astype(np.float64)) \
                / (1 + np.exp(-gate[:, h * 256:(h + 1) * 256].astype(np.float64)))
        return expected

    def routes(self, scope, qp, gp, kt, vt, rows, start, query_heads, heads, splits_from=None):
        """Every kernel route's BF16 output for the problem: (label, output values)."""
        length = start + rows
        out_bytes = rows * query_heads * 512
        observed = []
        args = [P(qp), P(gp), P(kt), P(vt)]
        def capture(label, output):
            observed.append((label, from_bf16(np.frombuffer(self.gpu.download(output, out_bytes), np.uint16))
                             .reshape(rows, query_heads, 256)))
        if rows > 1:
            pre = args + [None, U(rows), U(query_heads), U(heads), U(256), U(length), P(start)]
            for name, grid, block, shared in [
                    ('euhedral_attention_prefill32_bf16', ((rows + 31) // 32) * query_heads, 128, 0),
                    ('euhedral_attention_prefill32_bf16_exact', ((rows + 31) // 32) * query_heads, 128, 0),
                    ('euhedral_attention_prefill_fa2_bf16', ((rows + 15) // 16) * heads, 32 * (query_heads // heads + 2),
                     FA2_SHARED)]:
                if 'fa2' in name and query_heads // heads > 6:
                    continue
                output = self.owned(scope, self.gpu.zeros(out_bytes, 0x5A))
                pre[4] = P(output)
                self.gpu.launch(name, grid, pre, block=block, shared=shared)
                capture(name, output)
        else:
            position = self.position(scope, start)
            for splits in sorted(splits_from or {1, 3, min(64, (length + 47) // 48), min(64, (length + 31) // 32)}):
                for name, grid, block in [('euhedral_attention_decode_bf16', query_heads * splits, 128),
                                          ('euhedral_attention_decode_bf16_exact', query_heads * splits, 128),
                                          ('euhedral_attention_decode_gqa_bf16', heads * splits, 96)]:
                    if 'gqa' in name and query_heads // heads > 8:
                        continue
                    scratch = self.owned(scope, self.gpu.zeros(query_heads * splits * 258 * 4, 0xA5))
                    output = self.owned(scope, self.gpu.zeros(out_bytes, 0x5A))
                    self.gpu.launch(name, grid, args + [P(output), U(1), U(query_heads), U(heads), U(256), P(position),
                                                        P(0), P(scratch), U(splits)], block=block)
                    self.gpu.launch('euhedral_attention_merge_bf16', query_heads,
                                    [P(gp), P(output), P(scratch), U(query_heads), U(heads), U(splits)])
                    capture(f'{name}-splits-{splits}', output)
        return observed

    def check_routes(self, rows, start, heads, query_heads, seed, scale=1.0, splits=None):
        rng = np.random.default_rng(seed)
        length = start + rows
        width = (heads + query_heads) * 256
        qbits, q = bf16(rng.normal(0, .3, (rows, width)))
        gbits, gate = bf16(rng.normal(0, .5, (rows, width)))
        kbits, keys = bf16(rng.normal(0, scale, (length, heads, 256)))
        vbits, values = bf16(rng.normal(0, 1, (length, heads, 256)))
        with contextlib.ExitStack() as scope:
            qp, gp = self.owned(scope, self.gpu.upload(qbits.tobytes())), self.owned(scope, self.gpu.upload(gbits.tobytes()))
            kt, _ = self.paged(scope, kbits, heads)
            vt, _ = self.paged(scope, vbits, heads)
            observed = self.routes(scope, qp, gp, kt, vt, rows, start, query_heads, heads, splits)
        expected = self.oracle(q, gate, keys, values, rows, start, query_heads, heads)
        floor = relative_rms(bf16(expected)[1].astype(np.float64), expected)
        self.assertTrue(observed)
        for label, actual in observed:
            with self.subTest(route=label, rows=rows, start=start, heads=heads, query_heads=query_heads):
                error = relative_rms(actual.astype(np.float64), expected)
                self.assertLessEqual(error, floor + EXCESS, f'{label}: {error:.3e} against the BF16 floor {floor:.3e}')

    def test_prefill_routes_match_the_float64_oracle(self):
        # Chunk edges, nonzero starts, page edges, query-head groups of 1 to 6 and a 4096-key cache.
        for rows, start, heads, query_heads in [(2, 0, 4, 24), (17, 0, 4, 24), (33, 1, 4, 24), (3, 255, 2, 6),
                                                (130, 31, 1, 6), (5, 3, 1, 1), (64, 1000, 2, 12), (512, 0, 4, 24),
                                                (128, 3968, 4, 24), (512, 3584, 1, 6)]:
            self.check_routes(rows, start, heads, query_heads, seed=rows * 7919 + start)

    def test_decode_routes_match_the_float64_oracle_with_split_kv(self):
        for start, heads, query_heads in [(0, 4, 24), (255, 2, 6), (1023, 4, 24), (1024, 4, 24), (2047, 4, 24),
                                          (4096, 4, 24), (9000, 2, 12)]:
            self.check_routes(1, start, heads, query_heads, seed=start + 11)
        # A 32768-key context at the host's split counts (48- and 32-key splits, at most 64).
        self.check_routes(1, 32767, 4, 24, seed=5, splits={64})

    def test_extreme_keys_stay_finite(self):
        """Keys and values beyond FP16 range: the tensor-core tiles saturate them, the FP32 kernels read them as they
        are; every output stays finite."""
        rng = np.random.default_rng(77)
        heads, query_heads = 4, 24
        width = (heads + query_heads) * 256
        for rows, start in [(1, 300), (1, 3000), (40, 0), (200, 3000)]:
            length = start + rows
            with self.subTest(rows=rows, start=start), contextlib.ExitStack() as scope:
                qbits, _ = bf16(rng.normal(0, .3, (rows, width)))
                gbits, _ = bf16(rng.normal(0, .5, (rows, width)))
                magnitude = rng.choice([1.0, 7e4, 1e30], (length, heads, 256))
                kbits, _ = bf16(rng.normal(0, 1, (length, heads, 256)) * magnitude / 256)
                vbits, _ = bf16(rng.normal(0, 1, (length, heads, 256)) * magnitude)
                qp, gp = self.owned(scope, self.gpu.upload(qbits.tobytes())), self.owned(scope, self.gpu.upload(gbits.tobytes()))
                kt, _ = self.paged(scope, kbits, heads)
                vt, _ = self.paged(scope, vbits, heads)
                observed = self.routes(scope, qp, gp, kt, vt, rows, start, query_heads, heads)
                self.assertTrue(observed)
                for label, actual in observed:
                    self.assertTrue(np.isfinite(actual).all(), label)

    def test_row_exact_decode_twins_equal_one_row_decode_bitwise(self):
        """Speculative verification over a BF16 cache: every row of the multi-row twins (per-head kernel below 2048
        keys, GQA kernel from 2048, row-split merge) is bit for bit the one-row launch at that row's position."""
        query_heads, heads = 24, 4
        rng = np.random.default_rng(0x7418)
        stride = query_heads * 64 * 258
        for start in (40, 253, 1500, 2042, 3000):
            tokens = start + 8
            with contextlib.ExitStack() as scope:
                keys, _ = self.paged(scope, bf16(rng.normal(0, 1, (tokens, heads, 256)))[0], heads)
                values, _ = self.paged(scope, bf16(rng.normal(0, 1, (tokens, heads, 256)))[0], heads)
                for m in (2, 3, 4, 5, 8):
                    with self.subTest(start=start, rows=m):
                        width = (query_heads + heads) * 256
                        qk, _ = bf16(rng.standard_normal((m, width)))
                        gate, _ = bf16(rng.standard_normal((m, width)))
                        dqk = self.owned(scope, self.gpu.upload(qk.tobytes()))
                        dgate = self.owned(scope, self.gpu.upload(gate.tobytes()))
                        scratch = self.owned(scope, self.gpu.zeros(m * stride * 4))
                        together = self.owned(scope, self.gpu.zeros(m * query_heads * 256 * 2, 0xA5))
                        first, last = start + 1, start + m
                        dstart = self.position(scope, start)
                        if first < 2048:
                            below = min(last, 2047)
                            self.gpu.launch('euhedral_attention_decode_bf16_rows',
                                            (query_heads * min(64, (below + 47) // 48), m),
                                            [P(dqk), P(keys), P(values), U(query_heads), U(heads), P(dstart),
                                             P(scratch), P(stride), U(2048)], block=128)
                        if last >= 2048:
                            self.gpu.launch('euhedral_attention_decode_gqa_bf16_rows',
                                            (heads * min(64, (last + 31) // 32), m),
                                            [P(dqk), P(keys), P(values), U(query_heads), U(heads), P(dstart),
                                             P(scratch), P(stride), U(2048), U(0)], block=96)
                        self.gpu.launch('euhedral_attention_merge_bf16_rows', (query_heads, m),
                                        [P(dgate), P(together), P(scratch), U(query_heads), U(heads), P(dstart),
                                         P(stride), U(2048)], block=128)
                        rows = self.gpu.download(together, m * query_heads * 256 * 2)
                        row_bytes = query_heads * 256 * 2
                        for r in range(m):
                            length = start + r + 1
                            gqa = length >= 2048
                            splits = min(64, (length + 31) // 32 if gqa else (length + 47) // 48)
                            rqk = self.owned(scope, self.gpu.upload(qk[r].tobytes()))
                            rgate = self.owned(scope, self.gpu.upload(gate[r].tobytes()))
                            out = self.owned(scope, self.gpu.zeros(row_bytes, 0x5A))
                            one = self.owned(scope, self.gpu.zeros(stride * 4))
                            args = [P(rqk), P(rgate), P(keys), P(values), P(out), U(1), U(query_heads), U(heads),
                                    U(256), P(dstart), P(r), P(one), U(splits)]
                            if gqa:
                                self.gpu.launch('euhedral_attention_decode_gqa_bf16', heads * splits, args, block=96)
                            else:
                                self.gpu.launch('euhedral_attention_decode_bf16', query_heads * splits, args, block=128)
                            self.gpu.launch('euhedral_attention_merge_bf16', query_heads,
                                            [P(rgate), P(out), P(one), U(query_heads), U(heads), U(splits)], block=128)
                            self.assertEqual(rows[r * row_bytes:(r + 1) * row_bytes], self.gpu.download(out, row_bytes),
                                             f"row {r} at length {length}")

    def test_kernels_use_no_thread_local_memory(self):
        attribute = CUDA.cuFuncGetAttribute
        attribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_void_p]
        attribute.restype = C.c_int
        for name in [b'euhedral_attention_prefill32_bf16', b'euhedral_attention_prefill32_bf16_exact',
                     b'euhedral_attention_prefill_fa2_bf16', b'euhedral_attention_decode_bf16',
                     b'euhedral_attention_decode_bf16_exact', b'euhedral_attention_decode_gqa_bf16',
                     b'euhedral_attention_decode_gqa_bf16_rows', b'euhedral_attention_kv_append_bf16']:
            symbol, local = C.c_void_p(), C.c_int()
            self.assertEqual(0, self.gpu.function(C.byref(symbol), self.gpu.module, name), name)
            self.assertEqual(0, attribute(C.byref(local), 3, symbol))  # CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
            self.assertEqual(0, local.value, name.decode() + ' spills to thread-local memory')


if __name__ == '__main__':
    unittest.main()
