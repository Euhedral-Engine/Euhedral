"""NVFP4 expansion in torch and the exact-product linear.

NVFP4 weight = e2m1(code) * (e4m3(block scale per 16 along K) * fp32 global scale), codes row-major two per byte with
the EVEN k in the LOW nibble (the association is that of `euhedral_artifacts.nvfp4.dequantize_nvfp4_rows`).
`ExactNvfp4Linear` is the engine's product: BF16 activations x exactly expanded fp32 weights, an fp32 matmul with TF32
off, one BF16 rounding of the output.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from euhedral_artifacts.nvfp4 import E2M1_VALUES, nvfp4_offsets

BLOCK = 16


def configure_exact_math() -> None:
    """True fp32 matmuls (no TF32) and no reduced-precision bf16 split-K reductions: the harness's numerics contract."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# byte -> (value of the low nibble, value of the high nibble) as fp32; the sign is bit 3 of each nibble.
_E2M1_NIBBLE = np.concatenate([E2M1_VALUES, -E2M1_VALUES]).astype(np.float32)
_BYTE_PAIRS = np.stack([_E2M1_NIBBLE[np.arange(256) & 15], _E2M1_NIBBLE[np.arange(256) >> 4]], axis=1)
_LUT_CACHE: dict[str, torch.Tensor] = {}


def _byte_lut(device) -> torch.Tensor:
    key = str(device)
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = torch.from_numpy(_BYTE_PAIRS.copy()).to(device)
    return _LUT_CACHE[key]


def expand_nvfp4(codes: torch.Tensor, scales: torch.Tensor, global_scale) -> torch.Tensor:
    """codes uint8 [rows, K/2], scales uint8 [rows, K/16] (E4M3 bits), global scale (python float, 0-d tensor, or a
    [rows, 1] tensor for per-row scales) -> fp32 [rows, K]."""
    rows, half = codes.shape
    pairs = _byte_lut(codes.device)[codes.long()]  # [rows, K/2, 2]
    values = pairs.reshape(rows, half * 2 // BLOCK, BLOCK)
    block = scales.view(torch.float8_e4m3fn).to(torch.float32)
    if isinstance(global_scale, torch.Tensor):
        scale = global_scale.to(device=block.device, dtype=torch.float32)
        block = block * (scale.reshape(-1, 1) if scale.dim() > 0 else scale)
    else:
        block = block * torch.tensor(global_scale, dtype=torch.float32, device=block.device)
    return (values * block.unsqueeze(-1)).reshape(rows, half * 2)


@dataclass
class PackedNvfp4:
    """One row-split NVFP4 tensor kept compact on its device."""

    codes: torch.Tensor  # uint8 [rows, K/2]
    scales: torch.Tensor  # uint8 [rows, K/16]
    global_scale: float
    shape: tuple[int, int]

    @staticmethod
    def parse(payload: np.ndarray, shape: tuple[int, int], device) -> "PackedNvfp4":
        """`payload`: the object's bytes (uint8, row-split-k128-v1)."""
        rows, k = shape
        if k % 128 != 0:
            raise ValueError(f"row-split NVFP4 needs K divisible by 128, got {k}")
        scale_offset, global_offset, size = nvfp4_offsets(shape)
        if payload.size != size:
            raise ValueError(f"payload is {payload.size} bytes, a {shape} NVFP4 tensor takes {size}")
        codes = torch.from_numpy(np.array(payload[: rows * k // 2])).reshape(rows, k // 2)
        scales = torch.from_numpy(np.array(payload[scale_offset: scale_offset + rows * k // BLOCK]))
        global_scale = struct.unpack("<f", payload[global_offset: global_offset + 4].tobytes())[0]
        return PackedNvfp4(codes.to(device), scales.reshape(rows, k // BLOCK).to(device), global_scale, (rows, k))

    def expand(self) -> torch.Tensor:
        return expand_nvfp4(self.codes, self.scales, self.global_scale)

    @property
    def nbytes(self) -> int:
        return self.codes.numel() + self.scales.numel() + 4


def exact_linear(x: torch.Tensor, weight_fp32: torch.Tensor) -> torch.Tensor:
    """The engine's product: fp32 products and accumulation, one rounding of the result to the activation dtype."""
    return F.linear(x.float(), weight_fp32).to(x.dtype)


class ExactNvfp4Linear(nn.Module):
    """Drop-in `nn.Linear` over a packed NVFP4 weight; the weight is expanded to fp32 per call."""

    def __init__(self, packed: PackedNvfp4):
        super().__init__()
        self.packed = packed
        self.out_features, self.in_features = packed.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return exact_linear(x, self.packed.expand())

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, nvfp4"
