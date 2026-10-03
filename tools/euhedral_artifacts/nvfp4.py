"""NVFP4 and NVFP4-SD4 quantization.

NVFP4: E2M1 values in blocks of 16 along K, one E4M3 scale per block, one FP32 scale per tensor
(layout row-split-k128-v1). NVFP4-SD4 draws each tensor's block scales from a table of 16 E4M3 codes
(layout row-split-k128-sd4-v1, docs/NVFP4_COMPRESSED.md). The CUDA path (needs PyTorch) performs the
same float32/float64 operations in the same order, with IEEE division and round-half-to-even, so its
output bytes equal the CPU path's.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from euhedral_artifacts import device
from euhedral_artifacts.edrl import align_up, fail
from euhedral_artifacts.sources import MatrixSource

# NVFP4: E2M1 values in blocks of 16 along K, one E4M3 scale per block, one FP32 scale per tensor.
# A weight decodes to e2m1(code) * e4m3(block scale) * global scale; codes and scales round to
# nearest, ties to even, as the CUDA cvt instructions do.
E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E4M3_VALUES = np.array(
    [(code & 7) / 8 * 2.0**-6 if code < 8 else (1 + (code & 7) / 8) * 2.0 ** ((code >> 3) - 7) for code in range(127)],
    dtype=np.float32,
)
NVFP4_BLOCK = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
SD4_TABLE = 16
# A block may take a table scale at most SD4_BELOW E4M3 codes below its round-to-nearest scale (which
# clips its largest values; 0: never more than plain NVFP4 clips), and is costed exactly up to
# SD4_ABOVE codes above it; further above, its values all round to zero, so its cost is its energy.
# Allowing 2 codes below lowered the squared error but not the teacher-forced NLL
# (docs/NVFP4_COMPRESSED.md), and clipped the largest weights by up to 14%.
SD4_BELOW = 0
SD4_ABOVE = 40
E4M3_CODES = 127


def nvfp4_offsets(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Row-split NVFP4: (scale plane offset, global scale offset, byte size)."""
    if len(shape) != 2:
        fail("NVFP4 requires a rank-2 shape")
    n, k = shape
    groups = align_up(k, 128) // 64
    scale_offset = align_up(n * groups * 32, 256)
    global_offset = align_up(scale_offset + n * groups * 4, 256)
    return scale_offset, global_offset, global_offset + 4


def nvfp4_sd4_offsets(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Row-split NVFP4 with table-indexed scales: (index plane offset, table offset, byte size). The
    table is 16 E4M3 codes followed by the FP32 global scale."""
    if len(shape) != 2:
        fail("NVFP4 requires a rank-2 shape")
    n, k = shape
    k_pad = align_up(k, 128)
    index_offset = align_up(n * k_pad // 2, 256)
    table_offset = align_up(index_offset + n * k_pad // 32, 256)
    return index_offset, table_offset, table_offset + SD4_TABLE + 4


def round_to_table(values: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Index of the nearest table entry to each non-negative value, ties to the even index;
    values above the table saturate to its last entry."""
    upper = np.searchsorted(table, values, side="left").clip(1, len(table) - 1)
    lower = upper - 1
    below = values - table[lower]
    above = table[upper] - values
    pick_upper = (above < below) | ((above == below) & (upper % 2 == 0))
    index = np.where(pick_upper, upper, lower)
    index = np.where(values >= table[-1], len(table) - 1, index)
    return np.where(values <= table[0], 0, index).astype(np.uint8)


def nvfp4_global_scale(amax: float) -> np.float32:
    return np.float32(amax) / np.float32(E2M1_MAX * E4M3_MAX)


def quantize_nvfp4_rows(values: np.ndarray, global_scale: np.float32) -> tuple[np.ndarray, np.ndarray]:
    """values float32 [rows, K] (K a multiple of 16) -> (packed codes [rows, K/2], E4M3 scales [rows, K/16])."""
    rows, k = values.shape
    blocks = values.reshape(rows, k // NVFP4_BLOCK, NVFP4_BLOCK)
    if global_scale > 0:
        block_scale = (np.max(np.abs(blocks), axis=2) / np.float32(E2M1_MAX) / global_scale).astype(np.float32)
    else:
        block_scale = np.zeros(blocks.shape[:2], dtype=np.float32)
    scale_codes = round_to_table(block_scale, E4M3_VALUES)
    decode = (E4M3_VALUES[scale_codes] * global_scale).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(decode[..., None] > 0, np.abs(blocks) / decode[..., None], 0).astype(np.float32)
    magnitude = round_to_table(scaled, E2M1_VALUES)
    sign = ((blocks < 0) & (magnitude > 0)).astype(np.uint8) << 3
    codes = (magnitude | sign).reshape(rows, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale_codes.astype(np.uint8)


def dequantize_nvfp4_rows(packed: np.ndarray, scales: np.ndarray, global_scale: np.float32) -> np.ndarray:
    rows = packed.shape[0]
    codes = np.empty((rows, packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    values = E2M1_VALUES[codes & 7] * np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    block = np.repeat(E4M3_VALUES[scales] * global_scale, NVFP4_BLOCK, axis=1).astype(np.float32)
    return (values * block).astype(np.float32)


# NVFP4-SD4: each tensor's block scales come from a table of 16 E4M3 codes. The table minimizes the
# tensor's total squared error: sd4_costs measures, for blocks grouped by their round-to-nearest scale
# code, the error under every candidate code, and sd4_table chooses the 16 codes by dynamic programming.
# Every block then takes its least-error table scale (no more than SD4_BELOW codes below its nearest
# scale) and rounds its codes to nearest under it. The table reaches the largest scales the tensor
# needs, so outlier blocks are not clipped.
def nvfp4_codes(blocks: np.ndarray, decode: np.ndarray) -> np.ndarray:
    """E2M1 codes (magnitude | sign << 3) of `blocks` [n, 16] under absolute block scales `decode` [n]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(decode[:, None] > 0, np.abs(blocks) / decode[:, None], 0).astype(np.float32)
    magnitude = round_to_table(scaled, E2M1_VALUES)
    return magnitude | (((blocks < 0) & (magnitude > 0)).astype(np.uint8) << 3)


def block_error(blocks: np.ndarray, decode: np.ndarray) -> np.ndarray:
    """Squared error (float64) of each block quantized under `decode`."""
    codes = nvfp4_codes(blocks, decode)
    values = E2M1_VALUES[codes & 7] * np.where(codes & 8, -1.0, 1.0).astype(np.float32) * decode[:, None]
    difference = (values - blocks).astype(np.float64)
    return np.sum(difference * difference, axis=1)


def nearest_scale_codes(blocks: np.ndarray, global_scale: np.float32) -> np.ndarray:
    return round_to_table((np.max(np.abs(blocks), axis=1) / np.float32(E2M1_MAX) / global_scale).astype(np.float32),
            E4M3_VALUES).astype(np.int64)


def sd4_costs(values: np.ndarray, global_scale: np.float32) -> np.ndarray:
    """[nearest code, scale code] -> summed squared error of the blocks of `values` [rows, K] with that
    round-to-nearest scale code when quantized under that scale code (float64; inf where not allowed)."""
    blocks = values.reshape(-1, NVFP4_BLOCK)
    nearest = nearest_scale_codes(blocks, global_scale)
    costs = np.zeros(E4M3_CODES * E4M3_CODES)
    for offset in range(-SD4_BELOW, SD4_ABOVE + 1):
        code = nearest + offset
        valid = (code >= 0) & (code < E4M3_CODES)
        error = block_error(blocks[valid], (E4M3_VALUES[code[valid]] * global_scale).astype(np.float32))
        costs += np.bincount(nearest[valid] * E4M3_CODES + code[valid], weights=error, minlength=costs.size)
    energy = np.bincount(nearest, weights=np.sum(blocks.astype(np.float64) ** 2, axis=1), minlength=E4M3_CODES)
    return sd4_cost_bounds(costs.reshape(E4M3_CODES, E4M3_CODES), energy, np.bincount(nearest, minlength=E4M3_CODES))


def sd4_cost_bounds(costs: np.ndarray, energy: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Fills the codes outside each nearest code's measured window: too far below is not allowed, too
    far above costs the blocks' energy. Nearest codes without blocks cost nothing."""
    code = np.arange(E4M3_CODES)
    below = code[None, :] < code[:, None] - SD4_BELOW
    above = code[None, :] > code[:, None] + SD4_ABOVE
    costs = np.where(above, energy[:, None], costs)
    costs = np.where(below, np.inf, costs)
    return np.where(counts[:, None] > 0, costs, 0.0)


def sd4_table(costs: np.ndarray) -> np.ndarray:
    """The SD4_TABLE ascending codes minimizing sum over nearest codes o of min over entries c of
    costs[o, c]. Each o is costed at its bracketing entries (the largest at or below it and the smallest
    above it), which is the minimum over the table when costs fall to o's best scale and then rise.
    Ties go to the earliest choice."""
    n = E4M3_CODES
    code = np.arange(n)
    with np.errstate(invalid="ignore"):
        # below[b]: nearest codes under the first entry b; above[b]: at or over the last entry b.
        below = np.array([costs[:b, b].sum() for b in range(n)])
        above = np.array([costs[b:, b].sum() for b in range(n)])
        # segment[a, b]: nearest codes a <= o < b between consecutive entries a < b.
        pair = np.minimum(costs[:, :, None], costs[:, None, :])
        inside = (code[:, None, None] >= code[None, :, None]) & (code[:, None, None] < code[None, None, :])
        segment = np.where(inside, pair, 0.0).sum(axis=0)
    segment = np.where(code[:, None] < code[None, :], segment, np.inf)
    best = below.copy()
    previous = []
    for _ in range(SD4_TABLE - 1):
        total = best[:, None] + segment
        previous.append(np.argmin(total, axis=0))
        best = total[previous[-1], code]
    last = int(np.argmin(best + above))
    table = [last]
    for step in reversed(previous):
        table.append(int(step[table[-1]]))
    return np.array(sorted(table), dtype=np.uint8)


def quantize_nvfp4_sd4_rows(values: np.ndarray, global_scale: np.float32, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """values float32 [rows, K] -> (packed codes [rows, K/2], packed scale indices [rows, K/32])."""
    rows, k = values.shape
    blocks = values.reshape(-1, NVFP4_BLOCK)
    nearest = nearest_scale_codes(blocks, global_scale) if global_scale > 0 else np.zeros(len(blocks), np.int64)
    best = np.zeros(len(blocks), dtype=np.uint8)
    best_error = np.full(len(blocks), np.inf)
    for index, code in enumerate(table):
        error = block_error(blocks, np.full(len(blocks), E4M3_VALUES[code] * global_scale, dtype=np.float32))
        better = (error < best_error) & (int(code) >= nearest - SD4_BELOW)
        best[better], best_error[better] = index, error[better]
    codes = nvfp4_codes(blocks, (E4M3_VALUES[table[best]] * global_scale).astype(np.float32)).reshape(rows, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    indices = best.reshape(rows, k // NVFP4_BLOCK)
    return packed, (indices[:, 0::2] | (indices[:, 1::2] << 4)).astype(np.uint8)


def expand_nvfp4_sd4(packed_indices: np.ndarray, table: np.ndarray) -> np.ndarray:
    """E4M3 scale codes [rows, K/16] of packed scale indices [rows, K/32]."""
    rows = packed_indices.shape[0]
    indices = np.empty((rows, packed_indices.shape[1] * 2), dtype=np.uint8)
    indices[:, 0::2] = packed_indices & 15
    indices[:, 1::2] = packed_indices >> 4
    return table[indices]


def quantize_nvfp4_matrix(output, base_offset: int, matrix: MatrixSource) -> None:
    n, k = matrix.shape
    k_pad = align_up(k, 128)
    scale_offset, global_offset, _ = nvfp4_offsets(matrix.shape)
    # Small chunks: rounding allocates several index arrays per value, and many workers run at once.
    rows_per_chunk = max(1, 8 * 1024 * 1024 // max(k, 1))
    amax = 0.0
    for begin in range(0, n, rows_per_chunk):
        values = matrix.read_rows(begin, min(n, begin + rows_per_chunk))
        if not np.isfinite(values).all():
            fail("quantization source contains NaN or infinity")
        amax = max(amax, float(np.max(np.abs(values))))
    global_scale = nvfp4_global_scale(amax)
    for begin in range(0, n, rows_per_chunk):
        end = min(n, begin + rows_per_chunk)
        values = matrix.read_rows(begin, end)
        if k_pad != k:
            padded = np.zeros((end - begin, k_pad), dtype=np.float32)
            padded[:, :k] = values
            values = padded
        quantize = quantize_nvfp4_rows_torch if device.DEVICE != "cpu" else quantize_nvfp4_rows
        packed, scales = quantize(values, global_scale)
        output.seek(base_offset + begin * (k_pad // 2))
        output.write(packed.tobytes())
        output.seek(base_offset + scale_offset + begin * (k_pad // NVFP4_BLOCK))
        output.write(scales.tobytes())
    output.seek(base_offset + global_offset)
    output.write(np.float32(global_scale).astype("<f4").tobytes())


def quantize_nvfp4_sd4_matrix(output, base_offset: int, matrix: MatrixSource) -> None:
    n, k = matrix.shape
    k_pad = align_up(k, 128)
    index_offset, table_offset, _ = nvfp4_sd4_offsets(matrix.shape)
    rows_per_chunk = max(1, 8 * 1024 * 1024 // max(k, 1))

    def chunks() -> Iterable[tuple[int, np.ndarray]]:
        for begin in range(0, n, rows_per_chunk):
            end = min(n, begin + rows_per_chunk)
            values = matrix.read_rows(begin, end)
            if k_pad != k:
                padded = np.zeros((end - begin, k_pad), dtype=np.float32)
                padded[:, :k] = values
                values = padded
            yield begin, values

    amax = 0.0
    for _, values in chunks():
        if not np.isfinite(values).all():
            fail("quantization source contains NaN or infinity")
        amax = max(amax, float(np.max(np.abs(values))))
    global_scale = nvfp4_global_scale(amax)
    table = np.arange(SD4_TABLE, dtype=np.uint8)
    if global_scale > 0:
        measure = sd4_costs_torch if device.DEVICE != "cpu" else sd4_costs
        costs = sum(measure(values, global_scale) for _, values in chunks())
        table = sd4_table(costs)
    quantize = quantize_nvfp4_sd4_rows_torch if device.DEVICE != "cpu" else quantize_nvfp4_sd4_rows
    for begin, values in chunks():
        packed, indices = quantize(values, global_scale, table)
        output.seek(base_offset + begin * (k_pad // 2))
        output.write(packed.tobytes())
        output.seek(base_offset + index_offset + begin * (k_pad // 32))
        output.write(indices.tobytes())
    output.seek(base_offset + table_offset)
    output.write(table.tobytes() + np.float32(global_scale).astype("<f4").tobytes())


def round_to_table_torch(values, table):
    torch = device.torch()
    upper = torch.searchsorted(table, values.contiguous()).clamp(1, table.numel() - 1)
    lower = upper - 1
    below = values - table[lower]
    above = table[upper] - values
    pick_upper = (above < below) | ((above == below) & (upper % 2 == 0))
    index = torch.where(pick_upper, upper, lower)
    index = torch.where(values >= table[-1], torch.full_like(index, table.numel() - 1), index)
    return torch.where(values <= table[0], torch.zeros_like(index), index).to(torch.uint8)


def quantize_nvfp4_rows_torch(values: np.ndarray, global_scale: np.float32) -> tuple[np.ndarray, np.ndarray]:
    torch = device.torch()
    rows, k = values.shape
    blocks = torch.from_numpy(values).to(device.DEVICE).view(rows, k // NVFP4_BLOCK, NVFP4_BLOCK)
    e4m3, e2m1 = device.table("e4m3", E4M3_VALUES), device.table("e2m1", E2M1_VALUES)
    g = torch.tensor(global_scale, dtype=torch.float32, device=device.DEVICE)
    if global_scale > 0:
        block_scale = blocks.abs().amax(dim=2) / torch.tensor(E2M1_MAX, dtype=torch.float32, device=device.DEVICE) / g
    else:
        block_scale = torch.zeros(blocks.shape[:2], dtype=torch.float32, device=device.DEVICE)
    scale_codes = round_to_table_torch(block_scale, e4m3)
    decode = e4m3[scale_codes.long()] * g
    scaled = torch.where(decode[..., None] > 0, blocks.abs() / decode[..., None], torch.zeros_like(blocks))
    magnitude = round_to_table_torch(scaled, e2m1)
    sign = ((blocks < 0) & (magnitude > 0)).to(torch.uint8) << 3
    codes = (magnitude | sign).view(rows, k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.cpu().numpy(), scale_codes.cpu().numpy()


def nvfp4_codes_torch(blocks, decode):
    torch = device.torch()
    scaled = torch.where(decode[:, None] > 0, blocks.abs() / decode[:, None], torch.zeros_like(blocks))
    magnitude = round_to_table_torch(scaled, device.table("e2m1", E2M1_VALUES))
    return magnitude | (((blocks < 0) & (magnitude > 0)).to(torch.uint8) << 3)


def block_error_torch(blocks, decode):
    torch = device.torch()
    codes = nvfp4_codes_torch(blocks, decode).long()
    sign = torch.where((codes & 8) != 0, -1.0, 1.0).to(torch.float32)
    values = device.table("e2m1", E2M1_VALUES)[codes & 7] * sign * decode[:, None]
    difference = (values - blocks).double()
    return (difference * difference).sum(dim=1)


def nearest_scale_codes_torch(blocks, global_scale: np.float32):
    torch = device.torch()
    g = torch.tensor(global_scale, dtype=torch.float32, device=device.DEVICE)
    return round_to_table_torch(blocks.abs().amax(dim=1) / torch.tensor(E2M1_MAX, dtype=torch.float32, device=device.DEVICE) / g,
            device.table("e4m3", E4M3_VALUES)).long()


def sd4_costs_torch(values: np.ndarray, global_scale: np.float32) -> np.ndarray:
    torch = device.torch()
    blocks = torch.from_numpy(values).to(device.DEVICE).view(-1, NVFP4_BLOCK)
    e4m3 = device.table("e4m3", E4M3_VALUES)
    g = torch.tensor(global_scale, dtype=torch.float32, device=device.DEVICE)
    nearest = nearest_scale_codes_torch(blocks, global_scale)
    costs = torch.zeros(E4M3_CODES * E4M3_CODES, dtype=torch.float64, device=device.DEVICE)
    for offset in range(-SD4_BELOW, SD4_ABOVE + 1):
        code = nearest + offset
        valid = (code >= 0) & (code < E4M3_CODES)
        error = block_error_torch(blocks[valid], e4m3[code[valid]] * g)
        costs += torch.bincount(nearest[valid] * E4M3_CODES + code[valid], weights=error, minlength=costs.numel())
    energy = torch.bincount(nearest, weights=(blocks.double() ** 2).sum(dim=1), minlength=E4M3_CODES)
    counts = torch.bincount(nearest, minlength=E4M3_CODES)
    return sd4_cost_bounds(costs.view(E4M3_CODES, E4M3_CODES).cpu().numpy(), energy.cpu().numpy(), counts.cpu().numpy())


def quantize_nvfp4_sd4_rows_torch(values: np.ndarray, global_scale: np.float32, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    torch = device.torch()
    rows, k = values.shape
    blocks = torch.from_numpy(values).to(device.DEVICE).view(-1, NVFP4_BLOCK)
    e4m3 = device.table("e4m3", E4M3_VALUES)
    g = torch.tensor(global_scale, dtype=torch.float32, device=device.DEVICE)
    codes_table = torch.from_numpy(table.astype(np.int64)).to(device.DEVICE)
    nearest = nearest_scale_codes_torch(blocks, global_scale) if global_scale > 0 else torch.zeros(
            blocks.shape[0], dtype=torch.int64, device=device.DEVICE)
    best = torch.zeros(blocks.shape[0], dtype=torch.int64, device=device.DEVICE)
    best_error = torch.full((blocks.shape[0],), float("inf"), dtype=torch.float64, device=device.DEVICE)
    for index in range(len(table)):
        error = block_error_torch(blocks, (e4m3[codes_table[index]] * g).expand(blocks.shape[0]))
        better = (error < best_error) & (int(table[index]) >= nearest - SD4_BELOW)
        best = torch.where(better, torch.full_like(best, index), best)
        best_error = torch.where(better, error, best_error)
    codes = nvfp4_codes_torch(blocks, e4m3[codes_table[best]] * g).view(rows, k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    indices = best.view(rows, k // NVFP4_BLOCK).to(torch.uint8)
    return packed.cpu().numpy(), (indices[:, 0::2] | (indices[:, 1::2] << 4)).cpu().numpy()
