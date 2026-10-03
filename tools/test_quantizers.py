from pathlib import Path
import io
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from euhedral_artifacts import device, edrl, grouped, inventory, nvfp4, sources  # noqa: E402


class GroupedQuantizationTest(unittest.TestCase):
    def test_q3_group_packing_matches_little_endian_bit_stream(self):
        values = np.arange(64, dtype=np.int8) % 8 - 4
        actual, high = grouped.pack_codes(values.reshape(1, 64), 3)
        expected = bytearray(24)
        for index, value in enumerate(values):
            bit_offset = index * 3
            byte = bit_offset // 8
            shift = bit_offset % 8
            unsigned = int(value) & 0x7
            expected[byte] |= (unsigned << shift) & 0xFF
            if shift > 5:
                expected[byte + 1] |= unsigned >> (8 - shift)
        self.assertEqual(actual, bytes(expected))
        self.assertEqual(high, b"")

    def test_q5_low_plane_discards_the_high_bit(self):
        codes = np.array([[15, 16, -17, 7] + [0] * 60], dtype=np.int8)
        base, high = grouped.pack_codes(codes, 5)
        self.assertEqual(base[:2], bytes([0x0F, 0x7F]))
        self.assertEqual(high[0], 0x02)

    def test_q4_low_nibbles_are_packed_in_row_order(self):
        codes = (np.arange(64, dtype=np.int8) % 16).reshape(1, 64)
        base, high = grouped.pack_codes(codes, 4)
        expected = bytes(
            (index % 16) | (((index + 1) % 16) << 4)
            for index in range(0, 64, 2)
        )
        self.assertEqual(base, expected)
        self.assertEqual(high, b"")

    def test_scale_encoding_rounds_through_binary16_before_reciprocal(self):
        scales, reciprocal = grouped._canonical_scales(np.array([[1.0, 2.0]], dtype=np.float32), 3)
        expected = np.array([[np.float16(1.0 / 3.0), np.float16(2.0 / 3.0)]], dtype=np.float16)
        self.assertEqual(scales.dtype, np.float16)
        np.testing.assert_array_equal(scales, expected)
        np.testing.assert_array_equal(
            reciprocal,
            (1.0 / expected.astype(np.float64)).astype(np.float32),
        )

    def test_fused_matrix_construction_preserves_runtime_row_order(self):
        first = sources.MatrixSource((2, 2), lambda begin, end: np.full((end - begin, 2), 1, dtype=np.float32))
        second = sources.MatrixSource((3, 2), lambda begin, end: np.full((end - begin, 2), 2, dtype=np.float32))
        fused = sources.concat_matrix(first, second)
        np.testing.assert_array_equal(
            fused.read_rows(0, 5),
            np.array([[1, 1], [1, 1], [2, 2], [2, 2], [2, 2]], dtype=np.float32),
        )

    def test_descriptor_table_is_deterministic_and_absolute(self):
        def no_op(output, offset):
            return None

        plans = [
            edrl.ObjectPlan("a", (64, 64), "BF16", "Q3G64_F16S", "row-split-k128-v1", 1024, no_op, 4096),
            edrl.ObjectPlan("b", (4,), "BF16", "BF16", "contiguous-le-v1", 8, no_op, 5120),
        ]
        first = edrl.encode_table(plans)
        second = edrl.encode_table(plans)
        self.assertEqual(first, second)
        self.assertEqual(struct.unpack(">q", first[-16:-8])[0], 5120)
        self.assertEqual(struct.unpack(">q", first[-8:])[0], 8)

    def test_reference_payload_sizes_match_known_q3_geometry(self):
        self.assertEqual(grouped.row_split_size((248320, 5120), "Q3G64_F16S"), 516505600)
        self.assertEqual(grouped.row_split_size((7168, 5120), "Q4G64_F16S"), 19496960)
        self.assertEqual(grouped.row_split_size((7168, 5120), "Q5G64_F16S"), 24084480)

    def test_quantize_matrix_writes_all_planes_for_every_grouped_format(self):
        matrix = sources.MatrixSource(
            (2, 65),
            lambda begin, end: (
                ((np.arange((end - begin) * 65, dtype=np.float32).reshape(end - begin, 65) % 2) * 2 - 1)
                * (np.arange((end - begin) * 65, dtype=np.float32).reshape(end - begin, 65) + 1)
            ),
        )
        for format_name in ("Q3G64_F16S", "Q4G64_F16S", "Q5G64_F16S"):
            output = io.BytesIO()
            grouped.quantize_matrix(output, 0, matrix, format_name)
            payload = output.getvalue()
            self.assertEqual(len(payload), grouped.row_split_size((2, 65), format_name))
            self.assertNotEqual(payload[:32], b"\x00" * min(32, len(payload)))
            bits, group_size, _, _ = grouped.QUANT[format_name]
            groups = 128 // group_size
            base_bytes = 2 * groups * (24 if bits == 3 else 32)
            high_bytes = 2 * groups * (8 if bits == 5 else 0)
            high_offset = edrl.align_up(base_bytes, 256)
            scale_offset = high_offset + edrl.align_up(high_bytes, 256)
            scale_bytes = 2 * groups * 2
            self.assertNotEqual(payload[scale_offset:scale_offset + scale_bytes], b"\x00" * scale_bytes)
            if high_bytes:
                self.assertNotEqual(payload[high_offset:high_offset + high_bytes], b"\x00" * high_bytes)


class Nvfp4QuantizationTest(unittest.TestCase):
    def test_tables_cover_the_formats(self):
        self.assertEqual(list(nvfp4.E2M1_VALUES), [0, 0.5, 1, 1.5, 2, 3, 4, 6])
        self.assertEqual(len(nvfp4.E4M3_VALUES), 127)
        self.assertEqual(nvfp4.E4M3_VALUES[1], 2.0**-9)
        self.assertEqual(nvfp4.E4M3_VALUES[8], 2.0**-6)
        self.assertEqual(nvfp4.E4M3_VALUES[-1], 448.0)
        self.assertTrue((np.diff(nvfp4.E4M3_VALUES) > 0).all())

    def test_rounding_is_nearest_with_ties_to_even_and_saturates(self):
        values = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.26, 5.1, 7.0, 100.0, 0.0], dtype=np.float32)
        codes = nvfp4.round_to_table(values, nvfp4.E2M1_VALUES)
        self.assertEqual(list(nvfp4.E2M1_VALUES[codes]), [0, 1, 1, 2, 2, 4, 4, 0.5, 6, 6, 6, 0])
        scale = nvfp4.round_to_table(np.array([1000.0, 448.0, 2.0**-10], dtype=np.float32), nvfp4.E4M3_VALUES)
        self.assertEqual(list(nvfp4.E4M3_VALUES[scale]), [448.0, 448.0, 0.0])

    def test_representable_blocks_round_trip_exactly(self):
        rng = np.random.default_rng(4)
        global_scale = np.float32(2.0**-12)
        codes = rng.integers(0, 8, size=(3, 64))
        signs = np.where(rng.integers(0, 2, size=(3, 64)) == 1, -1.0, 1.0)
        block = nvfp4.E4M3_VALUES[rng.integers(8, 120, size=(3, 4))]
        values = (nvfp4.E2M1_VALUES[codes] * signs).reshape(3, 4, 16)
        values[..., 0] = 6.0  # each block's maximum sets its scale to block / 6 * 6
        values = (values * block[..., None] * global_scale).reshape(3, 64).astype(np.float32)
        packed, scales = nvfp4.quantize_nvfp4_rows(values, global_scale)
        np.testing.assert_array_equal(nvfp4.dequantize_nvfp4_rows(packed, scales, global_scale), values)

    def test_low_nibble_holds_the_even_column(self):
        values = np.zeros((1, 16), dtype=np.float32)
        values[0, 0], values[0, 1] = 6.0, -0.5
        packed, scales = nvfp4.quantize_nvfp4_rows(values, nvfp4.nvfp4_global_scale(6.0))
        self.assertEqual(packed[0, 0], 0x7 | (0x9 << 4))
        self.assertEqual(nvfp4.E4M3_VALUES[scales[0, 0]], 448.0)

    def test_quantization_error_is_bounded_by_half_a_step(self):
        rng = np.random.default_rng(5)
        values = rng.standard_normal((8, 256)).astype(np.float32)
        global_scale = nvfp4.nvfp4_global_scale(float(np.abs(values).max()))
        packed, scales = nvfp4.quantize_nvfp4_rows(values, global_scale)
        restored = nvfp4.dequantize_nvfp4_rows(packed, scales, global_scale)
        step = np.repeat(nvfp4.E4M3_VALUES[scales] * global_scale, 16, axis=1)
        self.assertTrue((np.abs(restored - values) <= step + 1e-6).all())

    def test_zero_tensor_has_zero_codes_and_scales(self):
        packed, scales = nvfp4.quantize_nvfp4_rows(np.zeros((2, 32), np.float32), np.float32(0))
        self.assertFalse(packed.any())
        self.assertFalse(scales.any())

    def test_row_split_nvfp4_geometry(self):
        scale_offset, global_offset, size = nvfp4.nvfp4_offsets((3, 1024))
        self.assertEqual(scale_offset, 3 * 512)
        self.assertEqual(global_offset, 3 * 512 + 256)
        self.assertEqual(size, global_offset + 4)
        self.assertEqual(inventory.payload_size((3, 1024), "NVFP4"), size)

    def test_nvfp4_sd4_geometry(self):
        index_offset, table_offset, size = nvfp4.nvfp4_sd4_offsets((3, 1024))
        self.assertEqual(index_offset, 3 * 512)
        self.assertEqual(table_offset, 3 * 512 + 256)
        self.assertEqual(size, table_offset + 16 + 4)

    def test_sd4_table_minimizes_the_total_cost(self):
        rng = np.random.default_rng(12)
        code = np.arange(nvfp4.E4M3_CODES)
        weights = np.zeros(nvfp4.E4M3_CODES)
        clients = rng.choice(np.arange(30, 100), 12, replace=False)
        weights[clients] = rng.uniform(0.1, 10.0, 12)
        costs = weights[:, None] * (code[None, :] - code[:, None]).astype(np.float64) ** 2
        saved = nvfp4.SD4_TABLE
        try:
            nvfp4.SD4_TABLE = 3
            table = nvfp4.sd4_table(costs)
        finally:
            nvfp4.SD4_TABLE = saved
        def total(entries):
            return costs[:, list(entries)].min(axis=1).sum()
        import itertools
        best = min(total(c) for c in itertools.combinations(range(25, 105), 3))
        self.assertAlmostEqual(total(table), best)
        # Sixteen entries cover twelve clients exactly.
        self.assertEqual(total(nvfp4.sd4_table(costs)), 0.0)
        self.assertTrue(set(clients) <= set(nvfp4.sd4_table(costs).tolist()))

    def test_sd4_reaches_the_largest_scales(self):
        rng = np.random.default_rng(13)
        values = (rng.standard_t(2.0, (64, 1024)) * 0.01).astype(np.float32)
        values[7, 300] = 3.0  # one outlier block four octaves above the rest
        global_scale = nvfp4.nvfp4_global_scale(float(np.abs(values).max()))
        table = nvfp4.sd4_table(nvfp4.sd4_costs(values, global_scale))
        packed, indices = nvfp4.quantize_nvfp4_sd4_rows(values, global_scale, table)
        chosen = nvfp4.expand_nvfp4_sd4(indices, table).reshape(-1).astype(np.int64)
        nearest = nvfp4.nearest_scale_codes(values.reshape(-1, 16), global_scale)
        self.assertTrue((chosen >= nearest - nvfp4.SD4_BELOW).all())
        self.assertGreaterEqual(int(table.max()), int(nearest.max()) - nvfp4.SD4_BELOW)
        restored = nvfp4.dequantize_nvfp4_rows(packed, nvfp4.expand_nvfp4_sd4(indices, table), global_scale)
        self.assertLess(abs(restored[7, 300] - 3.0), 0.5)

    def test_sd4_error_is_below_plain_nvfp4(self):
        rng = np.random.default_rng(9)
        values = (rng.standard_normal((64, 1024)) * rng.uniform(0.5, 2.0, (64, 1))).astype(np.float32)
        global_scale = nvfp4.nvfp4_global_scale(float(np.abs(values).max()))
        plain = nvfp4.dequantize_nvfp4_rows(*nvfp4.quantize_nvfp4_rows(values, global_scale), global_scale)
        table = nvfp4.sd4_table(nvfp4.sd4_costs(values, global_scale))
        packed, indices = nvfp4.quantize_nvfp4_sd4_rows(values, global_scale, table)
        compressed = nvfp4.dequantize_nvfp4_rows(packed, nvfp4.expand_nvfp4_sd4(indices, table), global_scale)
        self.assertLess(np.sum((compressed - values) ** 2), np.sum((plain - values) ** 2))

    def test_sd4_matrix_writes_codes_indices_table_and_global(self):
        rng = np.random.default_rng(10)
        values = rng.standard_normal((5, 256)).astype(np.float32)
        matrix = sources.MatrixSource((5, 256), lambda begin, end: values[begin:end])
        index_offset, table_offset, size = nvfp4.nvfp4_sd4_offsets((5, 256))
        output = io.BytesIO(bytes(size + 8))
        nvfp4.quantize_nvfp4_sd4_matrix(output, 8, matrix)
        data = np.frombuffer(output.getvalue()[8:], dtype=np.uint8)
        global_scale = nvfp4.nvfp4_global_scale(float(np.abs(values).max()))
        table = nvfp4.sd4_table(nvfp4.sd4_costs(values, global_scale))
        packed, indices = nvfp4.quantize_nvfp4_sd4_rows(values, global_scale, table)
        np.testing.assert_array_equal(data[:5 * 128], packed.reshape(-1))
        np.testing.assert_array_equal(data[index_offset:index_offset + 5 * 8], indices.reshape(-1))
        np.testing.assert_array_equal(data[table_offset:table_offset + 16], table)
        self.assertEqual(data[table_offset + 16:table_offset + 20].view("<f4")[0], global_scale)

    def test_sd4_zero_tensor(self):
        matrix = sources.MatrixSource((2, 128), lambda begin, end: np.zeros((end - begin, 128), np.float32))
        output = io.BytesIO(bytes(nvfp4.nvfp4_sd4_offsets((2, 128))[2]))
        nvfp4.quantize_nvfp4_sd4_matrix(output, 0, matrix)
        index_offset, table_offset, _ = nvfp4.nvfp4_sd4_offsets((2, 128))
        data = output.getvalue()
        self.assertFalse(any(data[:table_offset]))
        self.assertEqual(struct.unpack("<f", data[table_offset + 16:table_offset + 20])[0], 0.0)


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@unittest.skipUnless(_cuda_available(), "PyTorch with CUDA unavailable")
class CudaQuantizationMatchesCpuTest(unittest.TestCase):
    def setUp(self):
        device.DEVICE = "cuda"
        device._TABLES.clear()

    def tearDown(self):
        device.DEVICE = "cpu"
        device._TABLES.clear()

    def test_nvfp4_rows_are_byte_identical(self):
        rng = np.random.default_rng(6)
        values = (rng.standard_normal((64, 1024)) * rng.uniform(0.001, 3.0, (64, 1))).astype(np.float32)
        values[3, :16] = 0
        values[5, :32] = np.repeat(np.float32(0.25), 32)
        global_scale = nvfp4.nvfp4_global_scale(float(np.abs(values).max()))
        cpu = nvfp4.quantize_nvfp4_rows(values, global_scale)
        gpu = nvfp4.quantize_nvfp4_rows_torch(values, global_scale)
        np.testing.assert_array_equal(cpu[0], gpu[0])
        np.testing.assert_array_equal(cpu[1], gpu[1])

    def test_nvfp4_sd4_rows_are_byte_identical(self):
        rng = np.random.default_rng(11)
        values = (rng.standard_normal((64, 1024)) * rng.uniform(0.001, 3.0, (64, 1))).astype(np.float32)
        values[3, :16] = 0
        global_scale = nvfp4.nvfp4_global_scale(float(np.abs(values).max()))
        costs = nvfp4.sd4_costs(values, global_scale)
        np.testing.assert_allclose(costs, nvfp4.sd4_costs_torch(values, global_scale), rtol=1e-12)
        table = nvfp4.sd4_table(costs)
        np.testing.assert_array_equal(table, nvfp4.sd4_table(nvfp4.sd4_costs_torch(values, global_scale)))
        cpu = nvfp4.quantize_nvfp4_sd4_rows(values, global_scale, table)
        gpu = nvfp4.quantize_nvfp4_sd4_rows_torch(values, global_scale, table)
        np.testing.assert_array_equal(cpu[0], gpu[0])
        np.testing.assert_array_equal(cpu[1], gpu[1])

    def test_grouped_codes_and_packing_are_byte_identical(self):
        rng = np.random.default_rng(7)
        values = (rng.standard_normal((48, 256)) * rng.uniform(1e-6, 2.0, (48, 1))).astype(np.float32)
        values[0, :64] = 0
        for fmt in ("Q3G64_F16S", "Q4G64_F16S", "Q5G64_F16S"):
            bits, group_size, qmin, qmax = grouped.QUANT[fmt]
            with self.subTest(fmt=fmt):
                blocks = values.reshape(48, 256 // group_size, group_size)
                scales, reciprocal = grouped._canonical_scales(np.max(np.abs(blocks), axis=2), qmax)
                codes = np.clip(np.rint(blocks * reciprocal[..., None]), qmin, qmax).astype(np.int8)
                expected = grouped.pack_codes(codes.reshape(-1, group_size), bits)
                device_scales, device_codes = grouped.quantize_group_codes_torch(values, group_size, qmin, qmax)
                np.testing.assert_array_equal(device_scales.cpu().numpy(), scales)
                self.assertEqual(grouped.pack_codes_torch(device_codes, bits), expected)


if __name__ == "__main__":
    unittest.main()
