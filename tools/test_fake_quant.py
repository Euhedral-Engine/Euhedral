from pathlib import Path
import io
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from euhedral_artifacts import codes, device, grouped, nvfp4, sources  # noqa: E402
from euhedral_artifacts.importance import Calibration  # noqa: E402
from euhedral_artifacts.recipes import LAYOUT_ROW_SPLIT, LAYOUT_SD4, NVFP4_FORMAT  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

N, K = 96, 300  # K pads to 384, so the padding columns are exercised


def matrix(seed: int) -> sources.MatrixSource:
    rng = np.random.default_rng(seed)
    values = (rng.standard_normal((N, K)) * np.exp(rng.standard_normal((N, 1)))).astype(np.float32)
    values[3, :70] = 0.0  # an all-zero group and block
    values[5, 7] = 40.0  # an outlier
    return sources.MatrixSource((N, K), lambda begin, end: values[begin:end].copy())


def calibration(seed: int, lowest_code: int | None = None) -> Calibration:
    return Calibration(np.random.default_rng(seed).uniform(0.01, 4.0, K).astype(np.float32), lowest_code)


def converter_bytes(write, size: int) -> bytes:
    output = io.BytesIO(bytes(size))
    write(output, 0)
    return output.getvalue()


@unittest.skipIf(torch is None, "needs PyTorch")
class FakeQuantTest(unittest.TestCase):
    def setUp(self):
        device.set_device("cuda" if torch.cuda.is_available() else "cpu")
        from euhedral_artifacts import fake_quant
        self.fq = fake_quant

    def tearDown(self):
        device.set_device("cpu")

    def export(self, module) -> bytes:
        state = module.state()
        return converter_bytes(lambda output, offset: codes.write_state(output, offset, state), codes.state_size(state))

    def check_grouped(self, format_name: str, cal: Calibration | None):
        source = matrix(1)
        expected = converter_bytes(lambda output, offset: grouped.quantize_matrix(output, offset, source, format_name, cal),
                                   grouped.row_split_size((N, K), format_name))
        module = self.fq.fake_quant(source.read_rows(0, N), format_name, LAYOUT_ROW_SPLIT, (N, K), cal)
        self.assertEqual(self.export(module), expected)
        state = module.state()
        decoded = state.codes.astype(np.float32).reshape(N, -1, 64) * state.scales.astype(np.float32)[..., None]
        np.testing.assert_array_equal(module().detach().cpu().numpy(), decoded.reshape(N, -1))

    def test_grouped_formats_reproduce_the_converter(self):
        for format_name in grouped.QUANT:
            for cal in (None, calibration(2)):
                with self.subTest(format=format_name, calibrated=cal is not None):
                    self.check_grouped(format_name, cal)

    def test_restricted_q3_reproduces_the_converter(self):
        self.check_grouped("Q3G64_F16S", calibration(3, lowest_code=-3))

    def check_nvfp4(self, sd4: bool, cal: Calibration | None):
        source = matrix(4)
        if sd4:
            size = nvfp4.nvfp4_sd4_offsets((N, K))[2]
            expected = converter_bytes(lambda output, offset: nvfp4.quantize_nvfp4_sd4_matrix(output, offset, source, cal), size)
        else:
            size = nvfp4.nvfp4_offsets((N, K))[2]
            expected = converter_bytes(lambda output, offset: nvfp4.quantize_nvfp4_matrix(output, offset, source, cal), size)
        module = self.fq.fake_quant(source.read_rows(0, N), NVFP4_FORMAT, LAYOUT_SD4 if sd4 else LAYOUT_ROW_SPLIT, (N, K), cal)
        self.assertEqual(self.export(module), expected)
        state = module.state()
        scale_codes = state.table[state.scales] if sd4 else state.scales
        codes = state.codes
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
        decoded = nvfp4.dequantize_nvfp4_rows(packed, scale_codes, state.global_scale)
        np.testing.assert_array_equal(module().detach().cpu().numpy(), decoded)

    def test_nvfp4_layouts_reproduce_the_converter(self):
        for sd4 in (False, True):
            for cal in (None, calibration(5)):
                with self.subTest(sd4=sd4, calibrated=cal is not None):
                    self.check_nvfp4(sd4, cal)

    def test_row_slices_with_tensor_scales_reproduce_the_whole_object(self):
        source, cal = matrix(6), calibration(7)
        values = source.read_rows(0, N)
        whole = self.fq.fake_quant(values, NVFP4_FORMAT, LAYOUT_SD4, (N, K), cal).state()
        padded = lambda rows: self.fq._padded(rows, 384)  # noqa: E731
        tensor = self.fq.nvfp4_tensor_scales(lambda: [padded(values)], (N, K), cal, sd4=True)
        halves = [self.fq.Nvfp4FakeQuant.from_values(values[begin:begin + N // 2], (N, K), cal, True, tensor).state()
                  for begin in (0, N // 2)]
        np.testing.assert_array_equal(np.concatenate([h.codes for h in halves]), whole.codes)
        np.testing.assert_array_equal(np.concatenate([h.scales for h in halves]), whole.scales)
        np.testing.assert_array_equal(halves[0].table, whole.table)

    def test_gradients_reach_weights_and_scales(self):
        source = matrix(8)
        for format_name, layout in (("Q3G64_F16S", LAYOUT_ROW_SPLIT), (NVFP4_FORMAT, LAYOUT_ROW_SPLIT)):
            with self.subTest(format=format_name):
                module = self.fq.fake_quant(source.read_rows(0, N), format_name, layout, (N, K), calibration(9))
                module.weight.requires_grad_(True)
                before = module().detach().clone()
                (module() * torch.randn_like(before)).sum().backward()
                scale = module.scale if format_name != NVFP4_FORMAT else module.block_scale
                self.assertTrue(bool(module.weight.grad.abs().sum() > 0))
                self.assertTrue(bool(scale.grad.abs().sum() > 0))
                self.assertTrue(bool(torch.isfinite(scale.grad).all()))
                self.assertTrue(torch.equal(module().detach(), before))

    def test_states_round_trip_through_files(self):
        module = self.fq.fake_quant(matrix(10).read_rows(0, N), NVFP4_FORMAT, LAYOUT_SD4, (N, K), calibration(11))
        state = module.state()
        with tempfile.TemporaryDirectory() as tmp:
            path = codes.state_file(Path(tmp), "text/layers/0/mlp/down")
            state.save(path)
            loaded = codes.QuantState.load(path)
        codes.check_state(loaded, NVFP4_FORMAT, LAYOUT_SD4, (N, K))
        self.assertEqual(path.name, "text.layers.0.mlp.down.npz")
        for field in ("codes", "scales", "table"):
            np.testing.assert_array_equal(getattr(loaded, field), getattr(state, field))
        self.assertEqual(loaded.global_scale, state.global_scale)
        with self.assertRaises(ValueError):
            codes.check_state(loaded, NVFP4_FORMAT, LAYOUT_ROW_SPLIT, (N, K))


if __name__ == "__main__":
    unittest.main()
