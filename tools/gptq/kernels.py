"""Triton kernel for GPTQ's column loop: one launch rounds a run of columns whose scales are known and carries each
column's error into the rest of the block, instead of several PyTorch launches per column.

Every operation rounds as the PyTorch loop in gptq.py does (division, multiply and subtract each rounded to nearest,
no fused multiply-add), so the codes, scales and decoded weights are bit for bit the same.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

GROUPED, NVFP4 = 0, 1
ROWS = 32


@triton.jit
def _e2m1(index):
    # E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 by index.
    return tl.where(index < 5, index.to(tl.float32) * 0.5,
                    tl.where(index == 5, 3.0, tl.where(index == 6, 4.0, 6.0)))


@triton.jit
def _segment(block_ptr, errors_ptr, upper_ptr, decode_ptr, group_ptr, column_ptr, codes_ptr, decoded_ptr,
             n, width, k, upper_row, upper_column, begin, first, last, groups, qmin, qmax,
             FORMAT: tl.constexpr, BLOCK_COLUMNS: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, BLOCK_COLUMNS)
    row_ok = rows < n
    column_ok = columns < width
    tile = tl.load(block_ptr + rows[:, None] * width + columns[None, :], mask=row_ok[:, None] & column_ok[None, :],
                   other=0.0)
    errors = tl.load(errors_ptr + rows[:, None] * width + columns[None, :],
                     mask=row_ok[:, None] & column_ok[None, :], other=0.0)
    for j in range(first, last):
        x = tl.sum(tl.where(columns[None, :] == j, tile, 0.0), axis=1)
        g = tl.load(group_ptr + j)
        decode = tl.load(decode_ptr + rows * groups + g, mask=row_ok, other=0.0)
        if FORMAT == 0:
            ratio = libdevice.div_rn(x, tl.where(decode != 0.0, decode, 1.0))
            code = tl.where(decode != 0.0, libdevice.rint(ratio), 0.0)
            code = tl.minimum(tl.maximum(code, qmin), qmax)
            value = libdevice.mul_rn(code, decode)
            stored = code.to(tl.int32)
        else:
            magnitude = tl.abs(x)
            scaled = tl.where(decode > 0.0, libdevice.div_rn(magnitude, tl.where(decode > 0.0, decode, 1.0)), 0.0)
            # round_to_table_torch over the E2M1 table: searchsorted (left), clamped to 1..7, nearest with ties to
            # the even index, saturating at 6 and flushing at 0.
            upper = ((scaled > 0.0).to(tl.int32) + (scaled > 0.5).to(tl.int32) + (scaled > 1.0).to(tl.int32)
                     + (scaled > 1.5).to(tl.int32) + (scaled > 2.0).to(tl.int32) + (scaled > 3.0).to(tl.int32)
                     + (scaled > 4.0).to(tl.int32) + (scaled > 6.0).to(tl.int32))
            upper = tl.minimum(tl.maximum(upper, 1), 7)
            lower = upper - 1
            below = libdevice.sub_rn(scaled, _e2m1(lower))
            above = libdevice.sub_rn(_e2m1(upper), scaled)
            pick = (above < below) | ((above == below) & (upper % 2 == 0))
            index = tl.where(pick, upper, lower)
            index = tl.where(scaled >= 6.0, 7, index)
            index = tl.where(scaled <= 0.0, 0, index)
            negative = (x < 0.0) & (index > 0)
            sign = tl.where(negative, -1.0, 1.0)
            value = libdevice.mul_rn(libdevice.mul_rn(_e2m1(index), sign), decode)
            stored = index | (negative.to(tl.int32) << 3)
        column = tl.load(column_ptr + j)
        tl.store(codes_ptr + rows.to(tl.int64) * k + column, stored, mask=row_ok)
        tl.store(decoded_ptr + rows.to(tl.int64) * k + column, value, mask=row_ok)
        diagonal = tl.load(upper_ptr + (begin + j).to(tl.int64) * upper_row + (begin + j).to(tl.int64) * upper_column)
        error = libdevice.div_rn(libdevice.sub_rn(x, value), diagonal)
        row_of_upper = tl.load(upper_ptr + (begin + j).to(tl.int64) * upper_row
                               + (begin + columns).to(tl.int64) * upper_column,
                               mask=column_ok & (columns >= j), other=0.0)
        updated = libdevice.sub_rn(tile, libdevice.mul_rn(error[:, None], row_of_upper[None, :]))
        tile = tl.where(columns[None, :] >= j, updated, tile)
        errors = tl.where(columns[None, :] == j, error[:, None], errors)
    tl.store(block_ptr + rows[:, None] * width + columns[None, :], tile, mask=row_ok[:, None] & column_ok[None, :])
    tl.store(errors_ptr + rows[:, None] * width + columns[None, :], errors, mask=row_ok[:, None] & column_ok[None, :])


def segment(block: torch.Tensor, errors: torch.Tensor, upper: torch.Tensor, decodes: torch.Tensor,
            groups: torch.Tensor, columns: torch.Tensor, codes: torch.Tensor, decoded: torch.Tensor,
            begin: int, first: int, last: int, fmt: int, qmin: float = 0.0, qmax: float = 0.0) -> None:
    """Rounds block columns first..last-1 (block [N, width] FP32 holding columns begin.. of the permuted weights):
    column j belongs to group groups[j] (its decode is decodes[:, groups[j]]) and to weight column columns[j];
    codes [N, K] and decoded [N, K] receive it, errors[:, j] its scaled error, and the block the propagation."""
    n, width = block.shape
    grid = (triton.cdiv(n, ROWS),)
    _segment[grid](block, errors, upper, decodes, groups, columns, codes, decoded,
                   n, width, upper.shape[0], upper.stride(0), upper.stride(1), begin, first, last, decodes.shape[1],
                   float(qmin), float(qmax),
                   FORMAT=fmt, BLOCK_COLUMNS=triton.next_power_of_2(width), ROWS=ROWS, num_warps=4)
