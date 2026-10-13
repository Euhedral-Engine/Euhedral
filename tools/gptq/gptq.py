"""GPTQ for the artifact formats: rounding that compensates each column's error on the columns not yet rounded.

For a linear's weights W [N, K] and the Hessian H = X^T X of its inputs (X [n, K]), the columns are rounded in order;
each column's rounding error is spread over the remaining columns through the upper Cholesky factor of H^-1, so the
output error ||X (W - Q)^T|| is what the rounding minimizes, not the weight error (Frantar et al., GPTQ, 2022). Scales
are chosen per group when the group's first column is reached, by the converter's own calibrated search on the
error-updated weights, weighted by diag(H); codes are rounded as the converter rounds them, so the result is a
QuantState the converter writes unchanged (--codes-from).

Formats: Q3G64/Q4G64/Q5G64 (group 64, FP16 scales, signed), NVFP4 (block 16, E4M3 block scales, FP32 tensor scale)
and NVFP4-SD4 (block scales drawn from a 16-entry table chosen for the whole tensor). Runs on the GPU.
"""

from __future__ import annotations

import numpy as np
import torch

from euhedral_artifacts import device
from euhedral_artifacts.codes import QuantState
from euhedral_artifacts.grouped import QUANT, search_group_codes_torch
from euhedral_artifacts.nvfp4 import (E2M1_VALUES, E4M3_VALUES, NVFP4_BLOCK, SD4_TABLE, SEARCH_BELOW,
                                      nearest_scale_codes_torch, nvfp4_global_scale, round_to_table_torch,
                                      sd4_costs_torch, sd4_table, search_nvfp4_rows_torch,
                                      weighted_block_error_torch)
from euhedral_artifacts.recipes import LAYOUT_ROW_SPLIT, LAYOUT_SD4, NVFP4_FORMAT

BLOCK = 128


def inverse_cholesky(hessian: torch.Tensor, damp: float) -> torch.Tensor:
    """Upper Cholesky factor of (H + damp * mean(diag H) I)^-1, FP32. Columns no input reached get a unit diagonal."""
    h = hessian.float().clone()
    diagonal = h.diagonal()
    diagonal[diagonal == 0] = 1.0
    diagonal += damp * torch.mean(diagonal)
    torch.linalg.cholesky(h, out=h)
    h = torch.cholesky_inverse(h)
    return torch.linalg.cholesky(h, upper=True, out=h)


class _Grouped:
    def __init__(self, fmt: str, lowest_code: int | None):
        _, self.group, self.qmin, self.qmax = QUANT[fmt]
        if lowest_code is not None:
            self.qmin = max(self.qmin, lowest_code)
        self.fmt = fmt

    def choose(self, values: torch.Tensor, weights: np.ndarray):
        scales, _ = search_group_codes_torch(values.contiguous(), self.group, self.qmin, self.qmax, weights)
        decode = scales.float()  # [N, 1]
        return scales, decode

    def round(self, column: torch.Tensor, decode: torch.Tensor):
        code = torch.where(decode[:, 0] != 0, torch.round(column / decode[:, 0]), torch.zeros_like(column))
        code = code.clamp(self.qmin, self.qmax)
        return code, code * decode[:, 0]


class _Nvfp4:
    def __init__(self, global_scale: np.float32, table: np.ndarray | None, below: int):
        self.group, self.global_scale, self.table, self.below = NVFP4_BLOCK, np.float32(global_scale), table, below
        self.e4m3 = device.table("e4m3", E4M3_VALUES)
        self.e2m1 = device.table("e2m1", E2M1_VALUES)
        self.g = torch.tensor(self.global_scale, dtype=torch.float32, device=device.DEVICE)

    def choose(self, values: torch.Tensor, weights: np.ndarray):
        if self.table is not None:
            # The table entry with the least weighted error, no more than `below` codes under the nearest scale
            # (quantize_nvfp4_sd4_rows_torch, one block at a time).
            magnitudes = values.abs()
            w = torch.from_numpy(weights).to(device.DEVICE)[None, :].expand_as(values).contiguous()
            nearest = nearest_scale_codes_torch(values.contiguous(), self.global_scale)
            best = torch.zeros(values.shape[0], dtype=torch.int64, device=device.DEVICE)
            best_error = torch.full((values.shape[0],), float("inf"), device=device.DEVICE)
            for index, code in enumerate(self.table.tolist()):
                decode = (self.e4m3[code] * self.g).expand(values.shape[0])
                error = weighted_block_error_torch(magnitudes, w, decode)
                better = (error < best_error) & (code >= nearest - self.below)
                best = torch.where(better, torch.full_like(best, index), best)
                best_error = torch.where(better, error, best_error)
            code = torch.from_numpy(self.table.astype(np.int64)).to(device.DEVICE)[best]
            return best.to(torch.uint8), self.e4m3[code] * self.g
        _, codes = search_nvfp4_rows_torch(values.contiguous(), self.global_scale, weights)
        code = torch.from_numpy(codes[:, 0].astype(np.int64)).to(device.DEVICE)
        return code.to(torch.uint8), self.e4m3[code] * self.g

    def round(self, column: torch.Tensor, decode: torch.Tensor):
        scaled = torch.where(decode > 0, column.abs() / decode, torch.zeros_like(column))
        magnitude = round_to_table_torch(scaled, self.e2m1).long()
        negative = (column < 0) & (magnitude > 0)
        value = self.e2m1[magnitude] * torch.where(negative, -1.0, 1.0) * decode
        return magnitude | (negative.long() << 3), value


def nvfp4_tensor_scales(weight: torch.Tensor, hessian: torch.Tensor, layout: str,
                        chunk_rows: int = 16384) -> tuple[np.float32, np.ndarray | None]:
    """(FP32 global scale, SD4 table or None) of a whole NVFP4 object, the table chosen under diag(hessian)."""
    global_scale = nvfp4_global_scale(float(weight.abs().max()))
    if layout != LAYOUT_SD4:
        return global_scale, None
    table = np.arange(SD4_TABLE, dtype=np.uint8)
    if global_scale > 0:
        weights_np = torch.diag(hessian).float().cpu().numpy()
        costs = sum(sd4_costs_torch(weight[b:b + chunk_rows].float().contiguous(), global_scale, weights_np, SEARCH_BELOW)
                    for b in range(0, weight.shape[0], chunk_rows))
        table = sd4_table(costs)
    return global_scale, table


def gptq(weight: torch.Tensor, hessian: torch.Tensor, fmt: str, layout: str, shape: tuple[int, int],
         lowest_code: int | None = None, damp: float = 0.01, upper: torch.Tensor | None = None,
         tensor: tuple[np.float32, np.ndarray | None] | None = None) -> tuple[QuantState, torch.Tensor]:
    """(the QuantState of `weight` [N, K] rounded by GPTQ under `hessian` [K, K], the decoded weights [N, K]).
    `shape` is the object's shape (K a multiple of 128 here). `lowest_code` limits grouped codes (compressed q3).
    Rows are independent, so a large object can be rounded in row slices that share `upper` (inverse_cholesky)
    and, for NVFP4, `tensor` (nvfp4_tensor_scales of the whole object)."""
    w = weight.float().clone()
    n, k = w.shape
    importance = torch.diag(hessian).float()
    w[:, importance == 0] = 0
    if upper is None:
        upper = inverse_cholesky(hessian, damp)
    weights_np = importance.cpu().numpy().astype(np.float32)
    if fmt == NVFP4_FORMAT:
        global_scale, table = tensor if tensor is not None else nvfp4_tensor_scales(weight, hessian, layout)
        rounder = _Nvfp4(global_scale, table, SEARCH_BELOW)
        codes = torch.zeros(n, k, dtype=torch.int16, device=w.device)
        scales = torch.zeros(n, k // NVFP4_BLOCK, dtype=torch.uint8, device=w.device)
    else:
        rounder = _Grouped(fmt, lowest_code)
        codes = torch.zeros(n, k, dtype=torch.int8, device=w.device)
        scales = torch.zeros(n, k // rounder.group, dtype=torch.float16, device=w.device)
    decoded = torch.zeros_like(w)
    group = rounder.group
    decode = None
    for begin in range(0, k, BLOCK):
        end = min(k, begin + BLOCK)
        block = w[:, begin:end].clone()
        errors = torch.zeros_like(block)
        for i in range(end - begin):
            column = begin + i
            if column % group == 0:
                chosen, decode = rounder.choose(block[:, i:i + group], weights_np[column:column + group])
                scales[:, column // group] = chosen.reshape(-1).to(scales.dtype)
            code, value = rounder.round(block[:, i], decode)
            codes[:, column] = code.to(codes.dtype)
            decoded[:, column] = value
            error = (block[:, i] - value) / upper[column, column]
            block[:, i:] -= error[:, None] * upper[column, begin + i:end][None, :]
            errors[:, i] = error
        w[:, end:] -= errors @ upper[begin:end, end:]
    if fmt == NVFP4_FORMAT:
        state = QuantState(NVFP4_FORMAT, layout, shape, codes.to(torch.uint8).cpu().numpy(), scales.cpu().numpy(),
                           rounder.global_scale, rounder.table)
    else:
        state = QuantState(fmt, LAYOUT_ROW_SPLIT, shape, codes.cpu().numpy(), scales.cpu().numpy())
    return state, decoded
