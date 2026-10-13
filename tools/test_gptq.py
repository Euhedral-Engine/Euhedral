from pathlib import Path
import io
import sys
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

N, K = 64, 256


def written(state) -> bytes:
    output = io.BytesIO(bytes(codes.state_size(state)))
    codes.write_state(output, 0, state)
    return output.getvalue()


@unittest.skipIf(torch is None, "needs PyTorch")
class GptqTest(unittest.TestCase):
    def setUp(self):
        device.set_device("cuda" if torch.cuda.is_available() else "cpu")
        from gptq import gptq
        self.gptq = gptq
        rng = np.random.default_rng(3)
        self.values = (rng.standard_normal((N, K)) * np.exp(rng.standard_normal((N, 1)))).astype(np.float32)
        self.weight = torch.from_numpy(self.values).to(device.DEVICE)

    def tearDown(self):
        device.set_device("cpu")

    def converter(self, fmt, layout, calibration):
        source = sources.MatrixSource((N, K), lambda b, e: self.values[b:e].copy())
        if layout == LAYOUT_SD4:
            size, write = nvfp4.nvfp4_sd4_offsets((N, K))[2], nvfp4.quantize_nvfp4_sd4_matrix
        elif fmt == NVFP4_FORMAT:
            size, write = nvfp4.nvfp4_offsets((N, K))[2], nvfp4.quantize_nvfp4_matrix
        else:
            size = grouped.row_split_size((N, K), fmt)
            write = lambda o, b, m, c: grouped.quantize_matrix(o, b, m, fmt, c)  # noqa: E731
        output = io.BytesIO(bytes(size))
        write(output, 0, source, calibration)
        return output.getvalue()

    def test_a_diagonal_hessian_reproduces_the_converters_calibrated_rounding(self):
        importance = np.random.default_rng(4).uniform(0.1, 3.0, K).astype(np.float32)
        hessian = torch.diag(torch.from_numpy(importance)).to(device.DEVICE)
        for fmt, layout in (("Q3G64_F16S", LAYOUT_ROW_SPLIT), ("Q4G64_F16S", LAYOUT_ROW_SPLIT),
                            ("Q5G64_F16S", LAYOUT_ROW_SPLIT), (NVFP4_FORMAT, LAYOUT_ROW_SPLIT), (NVFP4_FORMAT, LAYOUT_SD4)):
            with self.subTest(fmt=fmt, layout=layout):
                state, _ = self.gptq.gptq(self.weight, hessian, fmt, layout, (N, K), damp=0.0)
                self.assertEqual(written(state), self.converter(fmt, layout, Calibration(importance)))

    def test_correlated_inputs_lower_the_output_error(self):
        rng = np.random.default_rng(5)
        mixing = rng.standard_normal((K, K)).astype(np.float32) * 0.3 + np.eye(K, dtype=np.float32)
        x = torch.from_numpy(rng.standard_normal((4096, K)).astype(np.float32) @ mixing).to(device.DEVICE)
        hessian = x.T @ x
        importance = torch.diag(hessian).cpu().numpy()
        for fmt, layout in (("Q3G64_F16S", LAYOUT_ROW_SPLIT), (NVFP4_FORMAT, LAYOUT_ROW_SPLIT)):
            with self.subTest(fmt=fmt):
                _, decoded = self.gptq.gptq(self.weight, hessian, fmt, layout, (N, K))
                plain, _ = self.gptq.gptq(self.weight, torch.diag(torch.diag(hessian)), fmt, layout, (N, K), damp=0.0)
                plain_decoded = decode(plain)
                gptq_error = float(((x @ (self.weight - decoded).T) ** 2).sum())
                plain_error = float(((x @ (self.weight - plain_decoded).T) ** 2).sum())
                self.assertLess(gptq_error, 0.8 * plain_error)
                self.assertTrue(np.isfinite(importance).all())

    def test_the_decoded_weights_are_the_states(self):
        rng = np.random.default_rng(6)
        x = torch.from_numpy(rng.standard_normal((1024, K)).astype(np.float32)).to(device.DEVICE)
        for fmt, layout in (("Q3G64_F16S", LAYOUT_ROW_SPLIT), (NVFP4_FORMAT, LAYOUT_SD4)):
            with self.subTest(fmt=fmt, layout=layout):
                state, decoded = self.gptq.gptq(self.weight, x.T @ x, fmt, layout, (N, K))
                self.assertTrue(torch.equal(decoded, decode(state)))


    def test_act_order_rounds_in_its_order_and_keeps_the_state_consistent(self):
        rng = np.random.default_rng(7)
        scales = np.exp(rng.standard_normal(K)).astype(np.float32)
        mixing = rng.standard_normal((K, K)).astype(np.float32) * 0.3 + np.eye(K, dtype=np.float32)
        x = torch.from_numpy((rng.standard_normal((4096, K)).astype(np.float32) @ mixing) * scales).to(device.DEVICE)
        hessian = x.T @ x
        order = self.gptq.column_order(hessian)
        for fmt, layout in (("Q3G64_F16S", LAYOUT_ROW_SPLIT), (NVFP4_FORMAT, LAYOUT_SD4)):
            with self.subTest(fmt=fmt, layout=layout):
                state, decoded = self.gptq.gptq(self.weight, hessian, fmt, layout, (N, K), order=order)
                self.assertTrue(torch.equal(decoded, decode(state)))
                plain, _ = self.gptq.gptq(self.weight, torch.diag(torch.diag(hessian)), fmt, layout, (N, K), damp=0.0)
                ordered_error = float(((x @ (self.weight - decoded).T) ** 2).sum())
                plain_error = float(((x @ (self.weight - decode(plain)).T) ** 2).sum())
                self.assertLess(ordered_error, 0.8 * plain_error)

def decode(state):
    if state.format_name == NVFP4_FORMAT:
        scale_codes = state.table[state.scales] if state.table is not None else state.scales
        packed = (state.codes[:, 0::2] | (state.codes[:, 1::2] << 4)).astype(np.uint8)
        values = nvfp4.dequantize_nvfp4_rows(packed, scale_codes, state.global_scale)
    else:
        values = (state.codes.astype(np.float32).reshape(N, -1, 64) * state.scales.astype(np.float32)[..., None]).reshape(N, K)
    return torch.from_numpy(values).to(device.DEVICE)


if __name__ == "__main__":
    unittest.main()
