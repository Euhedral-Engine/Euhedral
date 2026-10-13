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


def inverse_cholesky(hessian: torch.Tensor, damp: float, consume: bool = False) -> torch.Tensor:
    """Upper Cholesky factor of (H + damp * mean(diag H) I)^-1, FP32. Columns no input reached get a unit diagonal.
    With `consume`, an FP32 `hessian` is overwritten instead of copied (a 17408-wide H is 1.2 GB)."""
    h = hessian if consume and hessian.dtype == torch.float32 else hessian.float().clone()
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
            # (quantize_nvfp4_sd4_rows_torch); all entries at once, each row's error reduced as for one entry, and the
            # first least entry kept, as a loop over the entries with a strict comparison keeps it.
            entries = self.table.size
            n = values.shape[0]
            magnitudes = values.abs()[:, None, :].expand(n, entries, -1).reshape(-1, NVFP4_BLOCK)
            w = torch.from_numpy(weights).to(device.DEVICE)[None, :].expand(n * entries, -1).contiguous()
            nearest = nearest_scale_codes_torch(values.contiguous(), self.global_scale)
            table = torch.from_numpy(self.table.astype(np.int64)).to(device.DEVICE)
            decode = (self.e4m3[table] * self.g)[None, :].expand(n, -1).reshape(-1)
            error = weighted_block_error_torch(magnitudes.contiguous(), w, decode).view(n, entries)
            allowed = table[None, :] >= (nearest - self.below)[:, None]
            error = torch.where(allowed, error, torch.full_like(error, float("inf")))
            best = torch.argmin(error, dim=1)
            return best.to(torch.uint8), self.e4m3[table[best]] * self.g
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
                        chunk_rows: int = 4096) -> tuple[np.float32, np.ndarray | None]:
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


def column_order(hessian: torch.Tensor) -> torch.Tensor:
    """Act-order: the columns by decreasing diag(H), so the columns that matter most are rounded first, while the
    most columns are left to absorb their error."""
    return torch.argsort(torch.diag(hessian), descending=True, stable=True)


def _choose(rounder, w, block, begin, end, position, weights_np, g, scales):
    """The scale of group g, from the current values of its columns (none of them rounded yet: they are in this block
    or later)."""
    group = rounder.group
    at = position[g * group:(g + 1) * group]
    values = w[:, at].clone()
    inside = at < end
    values[:, inside] = block[:, at[inside] - begin]
    chosen, decode = rounder.choose(values, weights_np[g * group:(g + 1) * group])
    scales[:, g] = chosen.reshape(-1).to(scales.dtype)
    return decode  # the rounder's own shape: [N, 1] grouped, [N] NVFP4


def _column_loop(w, upper, rounder, columns, position, weights_np, codes, scales, decoded, group):
    """The reference loop: a few PyTorch launches per column."""
    n, k = w.shape
    decodes: list[torch.Tensor | None] = [None] * (k // group)
    for begin in range(0, k, BLOCK):
        end = min(k, begin + BLOCK)
        block = w[:, begin:end].clone()
        errors = torch.zeros_like(block)
        for i in range(end - begin):
            column = columns[begin + i]
            g = column // group
            if decodes[g] is None:
                decodes[g] = _choose(rounder, w, block, begin, end, position, weights_np, g, scales)
            code, value = rounder.round(block[:, i], decodes[g])
            codes[:, column] = code.to(codes.dtype)
            decoded[:, column] = value
            j = begin + i
            error = (block[:, i] - value) / upper[j, j]
            block[:, i:] -= error[:, None] * upper[j, j:end][None, :]
            errors[:, i] = error
        w[:, end:] -= errors @ upper[begin:end, end:]


def _fused_loop(w, upper, rounder, columns, position, weights_np, codes, scales, decoded, fmt):
    """The loop in runs: between two scale choices, one kernel launch rounds every column of the run."""
    from gptq import kernels
    n, k = w.shape
    group = rounder.group
    decodes = torch.zeros(n, k // group, dtype=torch.float32, device=w.device)
    chosen = [False] * (k // group)
    kind = kernels.NVFP4 if fmt == NVFP4_FORMAT else kernels.GROUPED
    qmin, qmax = (0.0, 0.0) if kind == kernels.NVFP4 else (float(rounder.qmin), float(rounder.qmax))
    order = torch.tensor(columns, dtype=torch.int32, device=w.device)
    group_of = order // group
    for begin in range(0, k, BLOCK):
        end = min(k, begin + BLOCK)
        block = w[:, begin:end].contiguous().clone()
        errors = torch.zeros_like(block)
        block_groups = group_of[begin:end].contiguous()
        block_columns = order[begin:end].contiguous()
        first = 0
        for i in range(end - begin):
            g = columns[begin + i] // group
            if not chosen[g]:
                if i > first:
                    kernels.segment(block, errors, upper, decodes, block_groups, block_columns, codes, decoded,
                                    begin, first, i, kind, qmin, qmax)
                    first = i
                decodes[:, g] = _choose(rounder, w, block, begin, end, position, weights_np, g, scales).reshape(-1)
                chosen[g] = True
        kernels.segment(block, errors, upper, decodes, block_groups, block_columns, codes, decoded,
                        begin, first, end - begin, kind, qmin, qmax)
        w[:, end:] -= errors @ upper[begin:end, end:]


def gptq(weight: torch.Tensor, hessian: torch.Tensor, fmt: str, layout: str, shape: tuple[int, int],
         lowest_code: int | None = None, damp: float = 0.01, upper: torch.Tensor | None = None,
         tensor: tuple[np.float32, np.ndarray | None] | None = None,
         order: torch.Tensor | None = None, fused: bool = True) -> tuple[QuantState, torch.Tensor]:
    """(the QuantState of `weight` [N, K] rounded by GPTQ under `hessian` [K, K], the decoded weights [N, K]).
    `shape` is the object's shape (K a multiple of 128 here). `lowest_code` limits grouped codes (compressed q3).
    `order` (column_order) rounds the columns in that order; each group's scale is chosen when its first column comes
    up, from the current values of all its columns. `upper` is then the factor of H permuted into that order.
    Rows are independent, so a large object can be rounded in row slices that share `upper` (inverse_cholesky)
    and, for NVFP4, `tensor` (nvfp4_tensor_scales of the whole object). `fused` rounds runs of columns with the Triton
    kernel (kernels.py), bit for bit the column loop it replaces."""
    n, k = weight.shape
    importance = torch.diag(hessian).float()
    if order is None:
        order = torch.arange(k, device=weight.device)
    permuted = hessian[order][:, order] if upper is None else None
    w = weight.float()[:, order].clone()
    w[:, importance[order] == 0] = 0
    if upper is None:
        upper = inverse_cholesky(permuted, damp)
        del permuted
    position = torch.empty_like(order)
    position[order] = torch.arange(k, device=order.device)
    columns = order.tolist()
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
    decoded = torch.zeros(n, k, dtype=torch.float32, device=w.device)
    group = rounder.group
    if fused and w.is_cuda:
        _fused_loop(w, upper, rounder, columns, position, weights_np, codes, scales, decoded, fmt)
    else:
        _column_loop(w, upper, rounder, columns, position, weights_np, codes, scales, decoded, group)
    if fmt == NVFP4_FORMAT:
        state = QuantState(NVFP4_FORMAT, layout, shape, codes.to(torch.uint8).cpu().numpy(), scales.cpu().numpy(),
                           rounder.global_scale, rounder.table)
    else:
        state = QuantState(fmt, LAYOUT_ROW_SPLIT, shape, codes.cpu().numpy(), scales.cpu().numpy())
    return state, decoded
