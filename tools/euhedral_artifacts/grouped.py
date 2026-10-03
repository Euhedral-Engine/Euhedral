"""Grouped integer quantization (Q3, Q4, Q5; layout row-split-k128-v1).

Each row of K values (K padded to 128) is split into groups of 64; a group stores an FP16 scale and
signed codes in [qmin, qmax]. Row-split layout: low-bit plane, high-bit plane (Q5 only), scale plane,
each on a 256-byte boundary. The CUDA path (needs PyTorch) produces the same bytes as the CPU path.
"""

from __future__ import annotations

import numpy as np

from euhedral_artifacts import device
from euhedral_artifacts.edrl import align_up, fail
from euhedral_artifacts.sources import MatrixSource

QUANT = {
    "Q3G64_F16S": (3, 64, -4, 3),
    "Q4G64_F16S": (4, 64, -8, 7),
    "Q5G64_F16S": (5, 64, -16, 15),
}


def row_split_size(shape: tuple[int, ...], format_name: str) -> int:
    if len(shape) != 2:
        fail(f"{format_name} requires a rank-2 shape")
    n, k = shape
    bits, group_size, _, _ = QUANT[format_name]
    k_pad = align_up(k, 128)
    groups = k_pad // group_size
    base_per_group = 24 if bits == 3 else 32
    high_per_group = 8 if bits == 5 else 0
    base = n * groups * base_per_group
    high = n * groups * high_per_group
    scale_offset = align_up(base, 256) + align_up(high, 256)
    return scale_offset + n * groups * 2


def _canonical_scales(max_abs: np.ndarray, qmax: int) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(max_abs).all():
        fail("quantization source contains NaN or infinity")
    raw_scale = (max_abs.astype(np.float64) / float(qmax)).astype(np.float32)
    scales = raw_scale.astype(np.float16)
    underflow = (scales == 0) & (max_abs > 0)
    if underflow.any():
        scales = scales.copy()
        scales[underflow] = np.float16(2.0**-24)
    reciprocal = np.zeros(max_abs.shape, dtype=np.float32)
    positive = scales > 0
    reciprocal[positive] = (1.0 / scales[positive].astype(np.float64)).astype(np.float32)
    return scales, reciprocal


def pack_codes(codes: np.ndarray, bits: int) -> tuple[bytes, bytes]:
    rows, group_size = codes.shape
    unsigned = codes.astype(np.int16) & ((1 << bits) - 1)
    if bits == 3:
        base = np.zeros((rows, group_size * 3 // 8), dtype=np.uint8)
        for index in range(group_size):
            bit = index * 3
            byte = bit // 8
            shift = bit % 8
            base[:, byte] |= ((unsigned[:, index] << shift) & 0xFF).astype(np.uint8)
            if shift > 5:
                base[:, byte + 1] |= (unsigned[:, index] >> (8 - shift)).astype(np.uint8)
        return base.tobytes(), b""
    low = unsigned & 0x0F
    base = (low[:, 0::2] | (low[:, 1::2] << 4)).astype(np.uint8)
    if bits == 4:
        return base.tobytes(), b""
    high_bits = bits - 4
    high = np.zeros((rows, group_size * high_bits // 8), dtype=np.uint8)
    upper = (unsigned >> 4) & ((1 << high_bits) - 1)
    values_per_byte = 8 // high_bits
    for index in range(group_size):
        bit = (index % values_per_byte) * high_bits
        byte = index // values_per_byte
        high[:, byte] |= ((upper[:, index] << bit) & 0xFF).astype(np.uint8)
    return base.tobytes(), high.tobytes()


def quantize_matrix(output, base_offset: int, matrix: MatrixSource, format_name: str) -> None:
    bits, group_size, qmin, qmax = QUANT[format_name]
    n, k = matrix.shape
    k_pad = align_up(k, 128)
    groups = k_pad // group_size
    base_per_group = 24 if bits == 3 else 32
    high_per_group = 8 if bits == 5 else 0
    base_row_bytes = groups * base_per_group
    high_row_bytes = groups * high_per_group
    base_bytes = n * base_row_bytes
    high_offset = align_up(base_bytes, 256)
    scale_offset = high_offset + align_up(n * high_row_bytes, 256)
    rows_per_chunk = max(1, 32 * 1024 * 1024 // max(k, 1))
    for begin in range(0, n, rows_per_chunk):
        end = min(n, begin + rows_per_chunk)
        values = matrix.read_rows(begin, end)
        if values.shape != (end - begin, k):
            fail(f"matrix source returned {values.shape}, expected {(end - begin, k)}")
        if k_pad != k:
            padded = np.zeros((end - begin, k_pad), dtype=np.float32)
            padded[:, :k] = values
            values = padded
        if device.DEVICE != "cpu":
            device_scales, flat_codes = quantize_group_codes_torch(values, group_size, qmin, qmax)
            base, high = pack_codes_torch(flat_codes, bits)
            scales = device_scales.cpu().numpy()
        else:
            grouped = values.reshape(end - begin, groups, group_size)
            max_abs = np.max(np.abs(grouped), axis=2)
            scales, reciprocal = _canonical_scales(max_abs, qmax)
            codes = np.rint(grouped * reciprocal[..., None])
            codes = np.clip(codes, qmin, qmax).astype(np.int8)
            flat_codes = codes.reshape(-1, group_size)
            base, high = pack_codes(flat_codes, bits)
        output.seek(base_offset + begin * base_row_bytes)
        output.write(base)
        if high_row_bytes:
            output.seek(base_offset + high_offset + begin * high_row_bytes)
            output.write(high)
        output.seek(base_offset + scale_offset + begin * groups * 2)
        output.write(np.ascontiguousarray(scales.astype("<f2")).tobytes())



def quantize_group_codes_torch(values: np.ndarray, group_size: int, qmin: int, qmax: int):
    """Row-split scales and codes: (FP16 scales [rows, groups], int16 codes [rows * groups, group_size])."""
    torch = device.torch()
    rows, k = values.shape
    grouped = torch.from_numpy(values).to(device.DEVICE).view(rows, k // group_size, group_size)
    max_abs = grouped.abs().amax(dim=2)
    if not bool(torch.isfinite(max_abs).all()):
        fail("quantization source contains NaN or infinity")
    scales = (max_abs.double() / float(qmax)).float().half()
    scales = torch.where((scales == 0) & (max_abs > 0), torch.full_like(scales, 2.0**-24), scales)
    reciprocal = torch.where(scales > 0, (1.0 / scales.double()).float(), torch.zeros_like(max_abs))
    codes = torch.round(grouped * reciprocal[..., None]).clamp(qmin, qmax).to(torch.int16)
    return scales, codes.view(-1, group_size)


def pack_codes_torch(codes, bits: int) -> tuple[bytes, bytes]:
    torch = device.torch()
    rows, group_size = codes.shape
    unsigned = codes & ((1 << bits) - 1)
    if bits == 3:
        base = torch.zeros((rows, group_size * 3 // 8), dtype=torch.int16, device=codes.device)
        for index in range(group_size):
            bit = index * 3
            byte, shift = bit // 8, bit % 8
            base[:, byte] |= (unsigned[:, index] << shift) & 0xFF
            if shift > 5:
                base[:, byte + 1] |= unsigned[:, index] >> (8 - shift)
        return base.to(torch.uint8).cpu().numpy().tobytes(), b""
    low = unsigned & 0x0F
    base = (low[:, 0::2] | (low[:, 1::2] << 4)).to(torch.uint8).cpu().numpy().tobytes()
    if bits == 4:
        return base, b""
    high_bits = bits - 4
    high = torch.zeros((rows, group_size * high_bits // 8), dtype=torch.int16, device=codes.device)
    upper = (unsigned >> 4) & ((1 << high_bits) - 1)
    values_per_byte = 8 // high_bits
    for index in range(group_size):
        high[:, index // values_per_byte] |= (upper[:, index] << ((index % values_per_byte) * high_bits)) & 0xFF
    return base, high.to(torch.uint8).cpu().numpy().tobytes()
