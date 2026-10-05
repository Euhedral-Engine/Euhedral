"""DFlash2 drafter operators (src/dflash/kernels.cu) against NumPy oracles that round where the PyTorch reference
rounds. Elementwise operators whose FP32 arithmetic is exact before their BF16 rounding (the dynamic convolution, the
selector's scores, top-16) must match bit for bit; reductions (linear, norms, attention) only differ in summation
order, so they are held to a fraction of exact outputs and a small error. The linear's rows are bit for bit the same
whatever the row count."""
import ctypes as C
import math
import unittest

from gpu_harness import Gpu, NVRTC, SKIP_REASON

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def bf16(values):
    """Round FP32 values to BF16 (nearest even), returned as FP32."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint32) << 16
    return rounded.astype(np.uint32).view(np.float32)


def to_bits(values):
    return (bf16(values).view(np.uint32) >> 16).astype(np.uint16)


def from_bits(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


@unittest.skipIf(NVRTC is None or np is None, str(SKIP_REASON) if NVRTC is None else "NumPy is not installed")
class DFlash2KernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "dflash/kernels.cu"\n')
        cls.rng = np.random.default_rng(2026)

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def put(self, array):
        array = np.ascontiguousarray(array)
        if array.dtype == np.float32:
            array = to_bits(array)
        return self.gpu.upload(array.tobytes())

    def get_bf16(self, pointer, shape):
        count = int(np.prod(shape))
        return from_bits(np.frombuffer(self.gpu.download(pointer, 2 * count), dtype=np.uint16)).reshape(shape)

    def random_bf16(self, *shape, scale=1.0):
        return bf16(self.rng.standard_normal(shape).astype(np.float32) * scale)

    def linear(self, x, w):
        rows, k = x.shape
        n = w.shape[0]
        px, pw, py = self.put(x), self.put(w), self.gpu.zeros(2 * rows * n)
        split = n <= 1536
        name = ('euhedral_dflash_linear' if rows <= 16 else 'euhedral_dflash_linear_rows') + ('_split' if split else '') + '_bf16'
        columns = n // 8 if split else n // 32
        tiles = (columns, 1) if rows <= 16 else (columns, (rows + 63) // 64)
        self.gpu.launch(name, tiles, [C.c_uint64(px), C.c_uint64(pw), C.c_uint64(py), C.c_uint(rows), C.c_uint(k),
                                      C.c_uint(n)], block=128)
        out = self.get_bf16(py, (rows, n))
        for p in (px, pw, py):
            self.gpu.free(p)
        return out

    def test_linear_rounds_the_fp32_product_once_and_keeps_rows_independent_of_the_row_count(self):
        k, n = 1056, 96
        x = self.random_bf16(100, k)
        w = self.random_bf16(n, k, scale=0.05)
        full = self.linear(x, w)
        exact = x.astype(np.float64) @ w.astype(np.float64).T
        self.assertLess(np.abs(full - exact).max() / np.abs(exact).max(), 0.01)
        self.assertGreater(np.mean(full == bf16(exact.astype(np.float32))), 0.97)
        for rows in (1, 3, 8, 16, 17, 64, 65):
            np.testing.assert_array_equal(self.linear(x[:rows], w), full[:rows], err_msg=f"{rows} rows")
        # Wider than the split limit: K unsplit, rows still independent of the row count.
        w = self.random_bf16(2080, k, scale=0.05)
        full = self.linear(x, w)
        exact = x.astype(np.float64) @ w.astype(np.float64).T
        self.assertLess(np.abs(full - exact).max() / np.abs(exact).max(), 0.01)
        for rows in (1, 8, 17, 64):
            np.testing.assert_array_equal(self.linear(x[:rows], w), full[:rows], err_msg=f"{rows} rows, unsplit")

    def test_rms_norm_rounds_the_normalized_row_then_the_weighted_row(self):
        rows, width = 5, 5120
        x = self.random_bf16(rows, width, scale=3.0)
        weight = self.random_bf16(width)
        px, pw, py = self.put(x), self.put(weight), self.gpu.zeros(2 * rows * width)
        self.gpu.launch('euhedral_dflash_rms_norm_bf16', rows, [C.c_uint64(px), C.c_uint64(pw), C.c_uint64(py),
                                                                C.c_uint(width), C.c_float(1e-6)], block=256)
        out = self.get_bf16(py, (rows, width))
        inverse = (1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-6)).astype(np.float32)
        expected = bf16(weight * bf16(x * inverse))
        self.assertGreater(np.mean(out == expected), 0.995)
        self.assertLess(np.abs(out - expected).max(), 0.02 * np.abs(expected).max())

    def test_dynamic_convolution_matches_the_reference_bit_for_bit(self):
        rows, width, group, taps = 8, 512, 16, 2
        groups = width // group
        x = self.random_bf16(rows, width)
        dynamic = self.random_bf16(rows, 2 * taps * groups, scale=0.5)
        base = self.random_bf16(2, taps, width, scale=0.5)
        for part in (0, 1):
            px, pd, pb, py = self.put(x), self.put(dynamic), self.put(base), self.gpu.zeros(2 * rows * width)
            self.gpu.launch('euhedral_dflash_conv_bf16', (rows * width + 255) // 256,
                            [C.c_uint64(px), C.c_uint64(pd), C.c_uint64(pb), C.c_uint64(py), C.c_uint(rows),
                             C.c_uint(width), C.c_uint(group), C.c_uint(taps), C.c_uint(part)], block=256)
            out = self.get_bf16(py, (rows, width))
            expected = np.zeros((rows, width), dtype=np.float32)
            kernels = dynamic.reshape(rows, 2, taps, groups)[:, part]
            for o in range(taps):
                shifted = np.zeros_like(x)
                shifted[o:] = x[:rows - o]
                expected = bf16(expected + bf16(base[part, o][None, :] * shifted))
                expected = bf16(expected + np.repeat(kernels[:, o], group, axis=1) * shifted)
            np.testing.assert_array_equal(out, expected, err_msg=f"part {part}")
            for p in (px, pd, pb, py):
                self.gpu.free(p)

    @staticmethod
    def rope_reference(head, position, theta):
        dim = head.shape[-1]
        inverse = (1.0 / np.power(np.float32(theta), np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim))).astype(
            np.float32)
        angles = np.float32(position) * inverse
        angles = np.concatenate([angles, angles])
        cos, sin = bf16(np.cos(angles)), bf16(np.sin(angles))
        rotated = np.concatenate([-head[dim // 2:], head[:dim // 2]])
        return bf16(bf16(head * cos) + bf16(rotated * sin))

    @staticmethod
    def head_norm(head, weight):
        inverse = np.float32(1.0 / math.sqrt(float((head.astype(np.float64) ** 2).mean()) + 1e-6))
        return bf16(weight * bf16(head * inverse))

    def test_context_keys_are_normalized_rotated_and_written_to_their_ring_slots(self):
        rows, heads, window, start = 3, 2, 16, 13
        kv = self.random_bf16(rows, 2 * heads * 128, scale=2.0)
        norm = self.random_bf16(128)
        ring_k, ring_v = self.gpu.zeros(2 * window * heads * 128), self.gpu.zeros(2 * window * heads * 128)
        pkv, pn = self.put(kv), self.put(norm)
        position = self.gpu.upload(np.array([start], dtype=np.uint64).tobytes())
        self.gpu.launch('euhedral_dflash_context_kv_bf16', (rows, heads),
                        [C.c_uint64(pkv), C.c_uint64(pn), C.c_uint64(ring_k), C.c_uint64(ring_v), C.c_uint64(position),
                         C.c_uint(window), C.c_uint(heads), C.c_float(1e-6), C.c_float(1e7)], block=128)
        keys = self.get_bf16(ring_k, (window, heads, 128))
        values = self.get_bf16(ring_v, (window, heads, 128))
        matched = []
        for row in range(rows):
            slot = (start + row) % window
            for h in range(heads):
                expected = self.rope_reference(self.head_norm(kv[row, h * 128:(h + 1) * 128], norm), start + row, 1e7)
                matched.append(np.mean(keys[slot, h] == expected))
                self.assertLess(np.abs(keys[slot, h] - expected).max(), 0.05)
                np.testing.assert_array_equal(values[slot, h], kv[row, (heads + h) * 128:(heads + h + 1) * 128])
        self.assertGreater(np.mean(matched), 0.98)
        self.assertEqual(0.0, np.abs(keys[(start + rows) % window]).max(), "other slots untouched")

    def test_block_attention_sees_the_window_of_context_keys_and_every_block_key(self):
        # A sliding window whose oldest keys the later rows mask, a context shorter than the window, and a full
        # 2048-key window (the longest key splits).
        for rows, heads, kv_heads, window, start in ((8, 8, 2, 64, 100), (8, 8, 2, 64, 10), (8, 32, 8, 2048, 5000),
                                                     (3, 8, 2, 2048, 3000)):
            with self.subTest(window=window, start=start, rows=rows):
                self.check_block_attention(rows, heads, kv_heads, window, start)

    def check_block_attention(self, rows, heads, kv_heads, window, start):
        width = kv_heads * 128
        query = self.random_bf16(rows, heads * 128)
        block_keys = self.random_bf16(rows, width)
        kv = self.random_bf16(rows, 2 * width)
        ring_k, ring_v = self.random_bf16(window, width), self.random_bf16(window, width)
        pointers = [self.put(a) for a in (query, block_keys, kv, ring_k, ring_v)]
        out = self.gpu.zeros(2 * rows * heads * 128)
        partial = self.gpu.zeros(4 * rows * 8 * heads * 130)
        position = self.gpu.upload(np.array([start], dtype=np.uint64).tobytes())
        shared = 32 * 276 * 4 + 2 * 32 * 280 * 2 + 16 * 136 * 2 + 64 * 4
        self.gpu.launch('euhedral_dflash_attention_tc_bf16', (kv_heads, 8),
                        [C.c_uint64(p) for p in pointers] + [C.c_uint64(partial), C.c_uint64(position), C.c_uint(rows),
                                                              C.c_uint(window), C.c_uint(heads), C.c_uint(kv_heads)],
                        block=128, shared=shared)
        self.gpu.launch('euhedral_dflash_attention_merge_bf16', (rows, heads),
                        [C.c_uint64(partial), C.c_uint64(out), C.c_uint(heads)], block=128)
        got = self.get_bf16(out, (rows, heads, 128))
        group = heads // kv_heads
        for i in range(rows):
            at = start + i
            first = max(0, at + 1 - window)
            context = [p % window for p in range(first, start)]
            for h in range(heads):
                g = h // group
                keys = np.concatenate([ring_k[context, g * 128:(g + 1) * 128], block_keys[:, g * 128:(g + 1) * 128]])
                values = np.concatenate([ring_v[context, g * 128:(g + 1) * 128],
                                         kv[:, width + g * 128:width + (g + 1) * 128]])
                scores = keys.astype(np.float64) @ query[i, h * 128:(h + 1) * 128].astype(np.float64) / math.sqrt(128)
                weights = np.exp(scores - scores.max())
                expected = (weights / weights.sum()) @ values.astype(np.float64)
                np.testing.assert_allclose(got[i, h], expected, atol=0.02, rtol=0.02, err_msg=f"row {i} head {h}")
        for p in pointers + [out, partial, position]:
            self.gpu.free(p)

    def test_swiglu_rounds_the_activation_before_the_up_product(self):
        rows, inter = 4, 1024
        gate_up = self.random_bf16(rows, 2 * inter, scale=3.0)
        pg, py = self.put(gate_up), self.gpu.zeros(2 * rows * inter)
        self.gpu.launch('euhedral_dflash_swiglu_bf16', (rows * inter + 255) // 256,
                        [C.c_uint64(pg), C.c_uint64(py), C.c_uint(rows), C.c_uint(inter)], block=256)
        out = self.get_bf16(py, (rows, inter))
        gate, up = gate_up[:, :inter], gate_up[:, inter:]
        expected = bf16(bf16(gate / (np.float32(1) + np.exp(-gate))) * up)
        self.assertGreater(np.mean(out == expected), 0.999)

    def test_top16_is_ordered_by_value_then_token(self):
        rows, vocabulary = 3, 248320
        logits = self.random_bf16(rows, vocabulary, scale=4.0)
        logits[1, [5, 77, 100000]] = 50.0  # ties resolve to the lower token
        logits[2, 9] = np.nan
        pl, pv, pi = self.put(logits), self.gpu.zeros(2 * rows * 16), self.gpu.zeros(4 * rows * 16)
        splits = 64
        sv, si = self.gpu.zeros(4 * rows * splits * 16), self.gpu.zeros(4 * rows * splits * 16)
        self.gpu.launch('euhedral_dflash_topk_partial_bf16', (rows, splits),
                        [C.c_uint64(pl), C.c_uint(vocabulary), C.c_uint64(sv), C.c_uint64(si)], block=256)
        self.gpu.launch('euhedral_dflash_topk_merge_bf16', rows,
                        [C.c_uint64(sv), C.c_uint64(si), C.c_uint(splits), C.c_uint64(pv), C.c_uint64(pi)], block=256)
        values = self.get_bf16(pv, (rows, 16))
        indices = np.frombuffer(self.gpu.download(pi, 4 * rows * 16), dtype=np.int32).reshape(rows, 16)
        for row in range(rows):
            clean = np.where(np.isnan(logits[row]), -np.inf, logits[row])
            order = np.lexsort((np.arange(vocabulary), -clean))[:16]
            np.testing.assert_array_equal(indices[row], order)
            np.testing.assert_array_equal(values[row], clean[order])

    def test_selector_walks_the_path_from_the_anchor(self):
        positions, rank, vocabulary = 7, 256, 1000
        hidden = self.random_bf16(positions, rank)
        values = np.sort(self.random_bf16(positions, 16, scale=2.0), axis=1)[:, ::-1].copy()
        indices = np.stack([self.rng.choice(vocabulary, 16, replace=False) for _ in range(positions)]).astype(np.int32)
        predecessor = self.random_bf16(vocabulary, rank, scale=0.3)
        successor = self.random_bf16(vocabulary, rank, scale=0.3)
        anchor = 17
        pointers = [self.put(a) for a in (hidden, values)] + [self.gpu.upload(indices.tobytes())]
        pointers += [self.put(predecessor), self.put(successor), self.gpu.upload(np.array([anchor], np.int32).tobytes())]
        tokens, scores = self.gpu.zeros(4 * positions), self.gpu.zeros(4 * positions * 16)
        self.gpu.launch('euhedral_dflash_select_bf16', 1,
                        [C.c_uint64(p) for p in pointers] + [C.c_uint(positions), C.c_uint(rank), C.c_uint64(tokens),
                                                             C.c_uint64(scores)], block=512)
        got_tokens = np.frombuffer(self.gpu.download(tokens, 4 * positions), dtype=np.int32)
        got_scores = np.frombuffer(self.gpu.download(scores, 4 * positions * 16), dtype=np.float32).reshape(positions, 16)
        previous = anchor
        for p in range(positions):
            product = bf16(predecessor[previous] * hidden[p])
            dots = bf16((successor[indices[p]].astype(np.float64) @ product.astype(np.float64)).astype(np.float32))
            expected = bf16(values[p] + dots)
            np.testing.assert_allclose(got_scores[p], expected, atol=0.0, rtol=0.01)
            previous = int(indices[p, int(np.argmax(got_scores[p]))])
            self.assertEqual(previous, got_tokens[p])


if __name__ == "__main__":
    unittest.main()
