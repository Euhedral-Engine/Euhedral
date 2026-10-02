import importlib.util
import io
from pathlib import Path
import struct
import sys
import unittest

import numpy as np

MODULE_PATH = Path(__file__).with_name("convert_qwen_safetensors_to_compact_edrl.py")
spec = importlib.util.spec_from_file_location("compact_converter", MODULE_PATH)
assert spec is not None and spec.loader is not None
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)


class CompactConverterTest(unittest.TestCase):
    def test_q3_group_packing_matches_little_endian_bit_stream(self):
        values = np.arange(64, dtype=np.int8) % 8 - 4
        actual, high = converter.pack_codes(values.reshape(1, 64), 3)
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

    def test_q6_high_plane_packs_four_two_bit_values_per_byte(self):
        values = (np.arange(64, dtype=np.int8) % 64) - 32
        base, high = converter.pack_codes(values.reshape(1, 64), 6)
        expected = bytearray(16)
        for index, value in enumerate(values):
            upper = (int(value) & 0x3F) >> 4
            expected[index // 4] |= upper << ((index % 4) * 2)
        self.assertEqual(len(base), 32)
        self.assertEqual(high, bytes(expected))

    def test_q5_low_plane_discards_the_high_bit(self):
        codes = np.array([[15, 16, -17, 7] + [0] * 60], dtype=np.int8)
        base, high = converter.pack_codes(codes, 5)
        self.assertEqual(base[:2], bytes([0x0F, 0x7F]))
        self.assertEqual(high[0], 0x02)

    def test_q4_low_nibbles_are_packed_in_row_order(self):
        codes = (np.arange(64, dtype=np.int8) % 16).reshape(1, 64)
        base, high = converter.pack_codes(codes, 4)
        expected = bytes(
            (index % 16) | (((index + 1) % 16) << 4)
            for index in range(0, 64, 2)
        )
        self.assertEqual(base, expected)
        self.assertEqual(high, b"")

    def test_w8_codes_are_stored_as_signed_bytes(self):
        codes = (np.arange(64, dtype=np.int8) - 32).reshape(1, 64)
        base, high = converter.pack_codes(codes, 8)
        self.assertEqual(base, codes.view(np.uint8).tobytes())
        self.assertEqual(high, b"")

    def test_scale_encoding_rounds_through_binary16_before_reciprocal(self):
        scales, reciprocal = converter._canonical_scales(np.array([[1.0, 2.0]], dtype=np.float32), 3)
        expected = np.array([[np.float16(1.0 / 3.0), np.float16(2.0 / 3.0)]], dtype=np.float16)
        self.assertEqual(scales.dtype, np.float16)
        np.testing.assert_array_equal(scales, expected)
        np.testing.assert_array_equal(
            reciprocal,
            (1.0 / expected.astype(np.float64)).astype(np.float32),
        )

    def test_fused_matrix_construction_preserves_runtime_row_order(self):
        first = converter.MatrixSource((2, 2), lambda begin, end: np.full((end - begin, 2), 1, dtype=np.float32))
        second = converter.MatrixSource((3, 2), lambda begin, end: np.full((end - begin, 2), 2, dtype=np.float32))
        fused = converter.concat_matrix(first, second)
        np.testing.assert_array_equal(
            fused.read_rows(0, 5),
            np.array([[1, 1], [1, 1], [2, 2], [2, 2], [2, 2]], dtype=np.float32),
        )

    def test_descriptor_table_is_deterministic_and_absolute(self):
        def no_op(output, offset):
            return None

        plans = [
            converter.ObjectPlan("a", (64, 64), "BF16", "Q3G64_F16S", "row-split-k128-v1", 1024, no_op, 4096),
            converter.ObjectPlan("b", (4,), "BF16", "BF16", "contiguous-le-v1", 8, no_op, 5120),
        ]
        first = converter.encode_table(plans)
        second = converter.encode_table(plans)
        self.assertEqual(first, second)
        self.assertEqual(struct.unpack(">q", first[-16:-8])[0], 5120)
        self.assertEqual(struct.unpack(">q", first[-8:])[0], 8)

    def test_reference_payload_sizes_match_known_q3_geometry(self):
        self.assertEqual(converter.row_split_size((248320, 5120), "Q3G64_F16S"), 516505600)
        self.assertEqual(converter.row_split_size((7168, 5120), "Q4G64_F16S"), 19496960)
        self.assertEqual(converter.row_split_size((7168, 5120), "Q5G64_F16S"), 24084480)

    def test_quantize_matrix_writes_all_planes_for_every_runtime_format(self):
        matrix = converter.MatrixSource(
            (2, 65),
            lambda begin, end: (
                ((np.arange((end - begin) * 65, dtype=np.float32).reshape(end - begin, 65) % 2) * 2 - 1)
                * (np.arange((end - begin) * 65, dtype=np.float32).reshape(end - begin, 65) + 1)
            ),
        )
        for format_name in ("Q3G64_F16S", "Q4G64_F16S", "Q5G64_F16S", "Q6G64_F16S", "W8G32_F16S"):
            output = io.BytesIO()
            converter.quantize_matrix(output, 0, matrix, format_name)
            payload = output.getvalue()
            self.assertEqual(len(payload), converter.row_split_size((2, 65), format_name))
            self.assertNotEqual(payload[:32], b"\x00" * min(32, len(payload)))
            bits, group_size, _, _ = converter.QUANT[format_name]
            groups = 128 // group_size
            base_bytes = 2 * groups * (24 if bits == 3 else 32)
            high_bytes = 2 * groups * (0 if bits in (3, 4, 8) else 8 if bits == 5 else 16)
            high_offset = converter.align_up(base_bytes, 256)
            scale_offset = high_offset + converter.align_up(high_bytes, 256)
            scale_bytes = 2 * groups * 2
            self.assertNotEqual(payload[scale_offset:scale_offset + scale_bytes], b"\x00" * scale_bytes)
            if high_bytes:
                self.assertNotEqual(payload[high_offset:high_offset + high_bytes], b"\x00" * high_bytes)


class Nvfp4QuantizationTest(unittest.TestCase):
    def test_tables_cover_the_formats(self):
        self.assertEqual(list(converter.E2M1_VALUES), [0, 0.5, 1, 1.5, 2, 3, 4, 6])
        self.assertEqual(len(converter.E4M3_VALUES), 127)
        self.assertEqual(converter.E4M3_VALUES[1], 2.0**-9)
        self.assertEqual(converter.E4M3_VALUES[8], 2.0**-6)
        self.assertEqual(converter.E4M3_VALUES[-1], 448.0)
        self.assertTrue((np.diff(converter.E4M3_VALUES) > 0).all())

    def test_rounding_is_nearest_with_ties_to_even_and_saturates(self):
        values = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.26, 5.1, 7.0, 100.0, 0.0], dtype=np.float32)
        codes = converter.round_to_table(values, converter.E2M1_VALUES)
        self.assertEqual(list(converter.E2M1_VALUES[codes]), [0, 1, 1, 2, 2, 4, 4, 0.5, 6, 6, 6, 0])
        scale = converter.round_to_table(np.array([1000.0, 448.0, 2.0**-10], dtype=np.float32), converter.E4M3_VALUES)
        self.assertEqual(list(converter.E4M3_VALUES[scale]), [448.0, 448.0, 0.0])

    def test_representable_blocks_round_trip_exactly(self):
        rng = np.random.default_rng(4)
        global_scale = np.float32(2.0**-12)
        codes = rng.integers(0, 8, size=(3, 64))
        signs = np.where(rng.integers(0, 2, size=(3, 64)) == 1, -1.0, 1.0)
        block = converter.E4M3_VALUES[rng.integers(8, 120, size=(3, 4))]
        values = (converter.E2M1_VALUES[codes] * signs).reshape(3, 4, 16)
        values[..., 0] = 6.0  # each block's maximum sets its scale to block / 6 * 6
        values = (values * block[..., None] * global_scale).reshape(3, 64).astype(np.float32)
        packed, scales = converter.quantize_nvfp4_rows(values, global_scale)
        np.testing.assert_array_equal(converter.dequantize_nvfp4_rows(packed, scales, global_scale), values)

    def test_low_nibble_holds_the_even_column(self):
        values = np.zeros((1, 16), dtype=np.float32)
        values[0, 0], values[0, 1] = 6.0, -0.5
        packed, scales = converter.quantize_nvfp4_rows(values, converter.nvfp4_global_scale(6.0))
        self.assertEqual(packed[0, 0], 0x7 | (0x9 << 4))
        self.assertEqual(converter.E4M3_VALUES[scales[0, 0]], 448.0)

    def test_quantization_error_is_bounded_by_half_a_step(self):
        rng = np.random.default_rng(5)
        values = rng.standard_normal((8, 256)).astype(np.float32)
        global_scale = converter.nvfp4_global_scale(float(np.abs(values).max()))
        packed, scales = converter.quantize_nvfp4_rows(values, global_scale)
        restored = converter.dequantize_nvfp4_rows(packed, scales, global_scale)
        step = np.repeat(converter.E4M3_VALUES[scales] * global_scale, 16, axis=1)
        self.assertTrue((np.abs(restored - values) <= step + 1e-6).all())

    def test_zero_tensor_has_zero_codes_and_scales(self):
        packed, scales = converter.quantize_nvfp4_rows(np.zeros((2, 32), np.float32), np.float32(0))
        self.assertFalse(packed.any())
        self.assertFalse(scales.any())

    def test_row_split_nvfp4_geometry(self):
        scale_offset, global_offset, size = converter.nvfp4_offsets((3, 1024))
        self.assertEqual(scale_offset, 3 * 512)
        self.assertEqual(global_offset, 3 * 512 + 256)
        self.assertEqual(size, global_offset + 4)
        self.assertEqual(converter.payload_size((3, 1024), "NVFP4"), size)

if __name__ == "__main__":
    unittest.main()
