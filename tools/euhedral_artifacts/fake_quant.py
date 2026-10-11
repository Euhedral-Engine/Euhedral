"""Fake quantization: PyTorch modules that hold a quantized object's rounding state and reproduce the converter.

A module is built from an object's float32 values with the converter's own scale selection (grouped.py, nvfp4.py), so
its codes, scales and decoded weights are the converter's byte for byte. Its forward pass returns exactly the decoded
weights the artifact stores, and gradients reach the master weights and the scales through straight-through rounding:
the output is the hard decode plus a surrogate minus itself, which is zero in value. `state()` exports the codes and
scales as a QuantState (codes.py), which the converter's --codes-from entry writes into the artifact.

Formats: Q3G64/Q4G64/Q5G64 (row-split), NVFP4 (row-split) and NVFP4-SD4. Needs PyTorch.
"""

from __future__ import annotations

import numpy as np

from euhedral_artifacts import device
from euhedral_artifacts.codes import QuantState
from euhedral_artifacts.edrl import align_up, fail
from euhedral_artifacts.grouped import QUANT, _binary16_scales, quantize_group_codes_torch, search_group_codes_torch
from euhedral_artifacts.importance import Calibration
from euhedral_artifacts.nvfp4 import (E2M1_VALUES, E4M3_VALUES, NVFP4_BLOCK, RESIDENT_CHUNK_VALUES, SD4_BELOW, SD4_TABLE,
                                      SEARCH_BELOW, nvfp4_global_scale, quantize_nvfp4_rows_torch,
                                      quantize_nvfp4_sd4_rows_torch, round_to_table_torch, sd4_costs_torch, sd4_table,
                                      search_nvfp4_rows_torch)
from euhedral_artifacts.recipes import LAYOUT_ROW_SPLIT, LAYOUT_SD4, NVFP4_FORMAT


def _straight_through(hard, surrogate):
    """`hard` in value, the gradient of `surrogate`: surrogate - surrogate.detach() is exactly zero."""
    return hard.detach() + (surrogate - surrogate.detach())


def _padded(values, k_pad: int):
    torch = device.torch()
    if not torch.is_tensor(values):
        values = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    values = values.to(device.DEVICE, torch.float32)
    if values.shape[1] != k_pad:
        values = torch.nn.functional.pad(values, (0, k_pad - values.shape[1]))
    return values


def _module_base():
    return device.torch().nn.Module


class GroupedFakeQuant(_module_base()):
    """Q3G64/Q4G64/Q5G64 with FP16 group scales. `weight` (float32 [N, K_pad]) and `scale` (float32 [N, groups],
    rounded to binary16 in the forward pass) are parameters; codes are round(weight / scale) clamped to the
    format's range, by division as calibrated rounding computes them or, with `reciprocal`, by multiplication
    with the float32 reciprocal as round-to-nearest conversion does."""

    def __init__(self, format_name: str, shape: tuple[int, int], weight, scales, qmin: int, qmax: int,
                 reciprocal: bool = False):
        super().__init__()
        torch = device.torch()
        self.format_name, self.shape, self.qmin, self.qmax, self.reciprocal = format_name, tuple(shape), qmin, qmax, reciprocal
        self.group_size = QUANT[format_name][1]
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.scale = torch.nn.Parameter(scales.float())

    @classmethod
    def from_values(cls, values, format_name: str, shape: tuple[int, int], calibration: Calibration | None = None):
        """The converter's rounding of `values` (float32 [N, K], NumPy or a device tensor): calibrated scale search
        with `calibration`, else round to nearest under max-abs / qmax."""
        _, group_size, qmin, qmax = QUANT[format_name]
        if calibration is not None and calibration.lowest_code is not None:
            qmin = max(qmin, calibration.lowest_code)
        torch = device.torch()
        weight = _padded(values, align_up(shape[1], 128))
        # Groups are rounded independently; row chunks only bound device memory.
        rows = max(1, 16 * 1024 * 1024 // shape[1])
        chunks = [weight[begin:begin + rows] for begin in range(0, weight.shape[0], rows)]
        if calibration is not None:
            weights = calibration.weights(weight.shape[1])
            scales = torch.cat([search_group_codes_torch(chunk, group_size, qmin, qmax, weights)[0] for chunk in chunks])
        else:
            scales = torch.cat([quantize_group_codes_torch(chunk.cpu().numpy(), group_size, qmin, qmax)[0] for chunk in chunks])
        return cls(format_name, shape, weight, scales, qmin, qmax, reciprocal=calibration is None)

    def _grouped(self, values):
        return values.view(values.shape[0], -1, self.group_size)

    def hard(self):
        """(binary16 decode scales [N, groups, 1] as float32, codes [N, groups, 64] as float32), detached."""
        torch = device.torch()
        x = self._grouped(self.weight.detach())
        max_abs = x.abs().amax(dim=2)
        scales = _binary16_scales(self.scale.detach(), max_abs)
        decode = scales.float()[..., None]
        if self.reciprocal:
            inverse = torch.where(scales > 0, (1.0 / scales.double()).float(), torch.zeros_like(max_abs))[..., None]
            codes = torch.round(x * inverse)
        else:
            codes = torch.where(decode > 0, torch.round(x / decode), torch.zeros_like(x))
        return decode, codes.clamp(self.qmin, self.qmax)

    def forward(self):
        """The decoded weights [N, K_pad], with straight-through gradients to `weight` and `scale`."""
        torch = device.torch()
        decode, codes = self.hard()
        x = self._grouped(self.weight)
        scale = self.scale[..., None]
        quotient = x / scale
        inside = (quotient >= self.qmin) & (quotient <= self.qmax)
        surrogate = torch.where(inside, x, codes * scale)
        return _straight_through(codes * decode, surrogate).view(self.weight.shape)

    def state(self) -> QuantState:
        decode, codes = self.hard()
        scales = decode[..., 0].half()
        return QuantState(self.format_name, LAYOUT_ROW_SPLIT, self.shape, codes.to(device.torch().int8).view(
            self.weight.shape).cpu().numpy(), scales.cpu().numpy())


class Nvfp4FakeQuant(_module_base()):
    """NVFP4: E2M1 codes in blocks of 16 under an E4M3 block scale and an FP32 tensor scale. `weight` (float32
    [N, K_pad]) and `block_scale` (float32 [N, K_pad / 16], rounded to E4M3 in the forward pass) are parameters;
    the tensor scale is fixed. With `table` and `indices` (NVFP4-SD4) the block scales are table entries and fixed."""

    def __init__(self, shape: tuple[int, int], weight, block_scale, global_scale: np.float32, layout: str = LAYOUT_ROW_SPLIT,
                 table: np.ndarray | None = None, indices=None):
        super().__init__()
        torch = device.torch()
        self.shape, self.layout, self.global_scale, self.table = tuple(shape), layout, np.float32(global_scale), table
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.block_scale = torch.nn.Parameter(block_scale.float(), requires_grad=table is None)
        self.register_buffer("indices", indices)

    @classmethod
    def from_values(cls, values, shape: tuple[int, int], calibration: Calibration | None = None, sd4: bool = False,
                    tensor: tuple[np.float32, np.ndarray | None] | None = None):
        """The converter's rounding of `values` (float32 [rows, K]): NVFP4 or, with `sd4`, NVFP4-SD4; calibrated
        with `calibration`, else round to nearest. `tensor` is (global scale, SD4 table) of the whole object
        (nvfp4_tensor_scales), needed when `values` holds only some of its rows; without it `values` is the object."""
        torch = device.torch()
        weight = _padded(values, align_up(shape[1], 128))
        if tensor is None:
            tensor = nvfp4_tensor_scales(lambda: [weight], shape, calibration, sd4)
        global_scale, table = tensor
        weights = calibration.weights(weight.shape[1]) if calibration is not None else None
        e4m3 = device.table("e4m3", E4M3_VALUES)
        if sd4:
            below = SEARCH_BELOW if calibration is not None else SD4_BELOW
            packed = np.concatenate([quantize_nvfp4_sd4_rows_torch(chunk, global_scale, table, weights, below)[1]
                                     for chunk in _row_chunks(weight, shape, calibration)])
            indices = np.empty((weight.shape[0], packed.shape[1] * 2), dtype=np.uint8)
            indices[:, 0::2], indices[:, 1::2] = packed & 15, packed >> 4
            indices = torch.from_numpy(indices).to(device.DEVICE)
            codes = torch.from_numpy(table.astype(np.int64)).to(device.DEVICE)[indices.long()]
            return cls(shape, weight, e4m3[codes], global_scale, LAYOUT_SD4, table, indices)
        if calibration is not None:
            codes = np.concatenate([search_nvfp4_rows_torch(chunk, global_scale, weights)[1]
                                    for chunk in _row_chunks(weight, shape, calibration)])
        else:
            codes = np.concatenate([quantize_nvfp4_rows_torch(chunk.cpu().numpy(), global_scale)[1]
                                    for chunk in _row_chunks(weight, shape, calibration)])
        return cls(shape, weight, e4m3[torch.from_numpy(codes).to(device.DEVICE).long()], global_scale)

    def scale_codes(self):
        """E4M3 codes of the block scales [N, K_pad / 16] (long), detached."""
        if self.table is not None:
            torch = device.torch()
            return torch.from_numpy(self.table.astype(np.int64)).to(device.DEVICE)[self.indices.long()]
        return round_to_table_torch(self.block_scale.detach(), device.table("e4m3", E4M3_VALUES)).long()

    def hard(self):
        """(absolute block scales [N, K_pad / 16, 1], E2M1 magnitude codes [N, K_pad / 16, 16], signs), detached."""
        torch = device.torch()
        g = torch.tensor(self.global_scale, dtype=torch.float32, device=device.DEVICE)
        decode = (device.table("e4m3", E4M3_VALUES)[self.scale_codes()] * g)[..., None]
        blocks = self.weight.detach().view(self.weight.shape[0], -1, NVFP4_BLOCK)
        scaled = torch.where(decode > 0, blocks.abs() / decode, torch.zeros_like(blocks))
        magnitude = round_to_table_torch(scaled, device.table("e2m1", E2M1_VALUES)).long()
        sign = torch.where((blocks < 0) & (magnitude > 0), -1.0, 1.0)
        return decode, magnitude, sign

    def forward(self):
        """The decoded weights [N, K_pad], with straight-through gradients to `weight` and the block scales."""
        torch = device.torch()
        decode, magnitude, sign = self.hard()
        values = device.table("e2m1", E2M1_VALUES)[magnitude] * sign
        blocks = self.weight.view(self.weight.shape[0], -1, NVFP4_BLOCK)
        g = torch.tensor(self.global_scale, dtype=torch.float32, device=device.DEVICE)
        scale = self.block_scale[..., None] * g
        inside = blocks.abs() <= E2M1_VALUES[-1] * scale
        surrogate = torch.where(inside, blocks, values * scale)
        return _straight_through(values * decode, surrogate).view(self.weight.shape)

    def state(self) -> QuantState:
        torch = device.torch()
        _, magnitude, sign = self.hard()
        codes = (magnitude | torch.where(sign < 0, 8, 0)).to(torch.uint8).view(self.weight.shape)
        scales = self.indices if self.table is not None else self.scale_codes()
        return QuantState(NVFP4_FORMAT, self.layout, self.shape, codes.cpu().numpy(), scales.to(torch.uint8).cpu().numpy(),
                          self.global_scale, self.table)


def _row_chunks(weight, shape: tuple[int, int], calibration: Calibration | None):
    """Row slices of `weight` [rows, K_pad] as the converter's NVFP4 quantizers take them: calibrated objects in
    chunks of RESIDENT_CHUNK_VALUES values, round-to-nearest ones in chunks of 8M / K rows. Only the SD4 cost
    sums depend on the boundaries (their float64 addition order); the chunks also bound device memory."""
    rows = nvfp4_chunk_rows(shape, calibration)
    return [weight[begin:begin + rows] for begin in range(0, weight.shape[0], rows)]


def nvfp4_chunk_rows(shape: tuple[int, int], calibration: Calibration | None) -> int:
    """Rows per chunk of the converter's NVFP4 quantizers for an object of `shape`."""
    k_pad = align_up(shape[1], 128)
    return max(1, RESIDENT_CHUNK_VALUES // k_pad) if calibration is not None else max(1, 8 * 1024 * 1024 // shape[1])


def nvfp4_tensor_scales(read_all, shape: tuple[int, int], calibration: Calibration | None,
                        sd4: bool) -> tuple[np.float32, np.ndarray | None]:
    """(FP32 global scale, SD4 table or None) of an NVFP4 object, as the converter computes them. `read_all()`
    returns the object's float32 rows [N, K_pad] as device tensors in consecutive pieces (called twice), each but
    the last a multiple of nvfp4_chunk_rows rows."""
    torch = device.torch()
    amax = 0.0
    for piece in read_all():
        if not bool(torch.isfinite(piece).all()):
            fail("quantization source contains NaN or infinity")
        amax = max(amax, float(piece.abs().max()))
    global_scale = nvfp4_global_scale(amax)
    if not sd4:
        return global_scale, None
    table = np.arange(SD4_TABLE, dtype=np.uint8)
    if global_scale > 0:
        weights = calibration.weights(align_up(shape[1], 128)) if calibration is not None else None
        below = SEARCH_BELOW if calibration is not None else SD4_BELOW
        costs = sum(sd4_costs_torch(chunk, global_scale, weights, below)
                    for piece in read_all() for chunk in _row_chunks(piece, shape, calibration))
        table = sd4_table(costs)
    return global_scale, table


def fake_quant(values, format_name: str, layout: str, shape: tuple[int, int], calibration: Calibration | None = None):
    """The fake-quant module the converter's rounding of an object stored as `format_name` / `layout` gives."""
    if format_name == NVFP4_FORMAT:
        return Nvfp4FakeQuant.from_values(values, shape, calibration, sd4=layout == LAYOUT_SD4)
    if format_name in QUANT:
        return GroupedFakeQuant.from_values(values, format_name, shape, calibration)
    fail(f"no fake quantization for {format_name}")

