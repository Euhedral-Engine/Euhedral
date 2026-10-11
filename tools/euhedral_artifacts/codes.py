"""Pre-rounded codes: the stored rounding of a quantized object, saved one file per object, and packed into the
artifact's planes exactly as the converter's quantizers lay them out (the converter's --codes-from entry).
fake_quant.py exports these from trained modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from euhedral_artifacts.edrl import align_up, fail
from euhedral_artifacts.grouped import QUANT, pack_codes, row_split_size
from euhedral_artifacts.nvfp4 import NVFP4_BLOCK, SD4_TABLE, nvfp4_offsets, nvfp4_sd4_offsets
from euhedral_artifacts.recipes import LAYOUT_SD4, NVFP4_FORMAT


@dataclass
class QuantState:
    """The stored rounding of one quantized object of shape (N, K), K padded to K_pad = K rounded up to 128.

    Grouped formats: `codes` int8 [N, K_pad] signed codes, `scales` float16 [N, K_pad / 64].
    NVFP4: `codes` uint8 [N, K_pad] E2M1 codes (magnitude | sign << 3), `scales` uint8 [N, K_pad / 16] E4M3 codes,
    `global_scale` float32. NVFP4-SD4: `scales` holds uint8 indices into `table` (16 E4M3 codes) instead."""
    format_name: str
    layout: str
    shape: tuple[int, int]
    codes: np.ndarray
    scales: np.ndarray
    global_scale: np.float32 = np.float32(0)
    table: np.ndarray | None = None

    def save(self, path: Path) -> None:
        arrays = {"codes": self.codes, "scales": self.scales, "global_scale": np.float32(self.global_scale)}
        if self.table is not None:
            arrays["table"] = self.table
        np.savez(path, format_name=self.format_name, layout=self.layout, shape=np.asarray(self.shape, np.int64), **arrays)

    @staticmethod
    def load(path: Path) -> "QuantState":
        with np.load(path) as data:
            return QuantState(str(data["format_name"]), str(data["layout"]), tuple(int(v) for v in data["shape"]),
                              data["codes"], data["scales"], np.float32(data["global_scale"]),
                              data["table"] if "table" in data else None)


def state_file(directory: Path, name: str) -> Path:
    """The file of object `name` in a codes directory."""
    return directory / (name.replace("/", ".") + ".npz")


def state_size(state: QuantState) -> int:
    if state.layout == LAYOUT_SD4:
        return nvfp4_sd4_offsets(state.shape)[2]
    if state.format_name == NVFP4_FORMAT:
        return nvfp4_offsets(state.shape)[2]
    return row_split_size(state.shape, state.format_name)


def check_state(state: QuantState, format_name: str, layout: str, shape: tuple[int, ...]) -> None:
    """Fails unless `state` stores an object of `shape` as `format_name` / `layout`, with arrays of matching shapes."""
    if (state.format_name, state.layout, tuple(state.shape)) != (format_name, layout, tuple(shape)):
        fail(f"codes are {state.format_name}/{state.layout} {state.shape}, the recipe stores "
             f"{format_name}/{layout} {tuple(shape)}")
    n, k = shape
    k_pad = align_up(k, 128)
    per_scale = NVFP4_BLOCK if format_name == NVFP4_FORMAT else QUANT[format_name][1]
    if state.codes.shape != (n, k_pad) or state.scales.shape != (n, k_pad // per_scale):
        fail(f"codes {state.codes.shape} and scales {state.scales.shape} do not match shape {tuple(shape)}")
    if layout == LAYOUT_SD4 and (state.table is None or state.table.shape != (SD4_TABLE,)):
        fail("NVFP4-SD4 codes need a 16-entry scale table")


def write_state(output, base_offset: int, state: QuantState) -> None:
    """Writes `state` into its object's planes at `base_offset`, as the converter's quantizers lay them out."""
    n, k = state.shape
    k_pad = align_up(k, 128)
    if state.format_name == NVFP4_FORMAT:
        codes = state.codes.astype(np.uint8)
        output.seek(base_offset)
        output.write((codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8).tobytes())
        if state.layout == LAYOUT_SD4:
            index_offset, table_offset, _ = nvfp4_sd4_offsets(state.shape)
            indices = state.scales.astype(np.uint8)
            output.seek(base_offset + index_offset)
            output.write((indices[:, 0::2] | (indices[:, 1::2] << 4)).astype(np.uint8).tobytes())
            output.seek(base_offset + table_offset)
            output.write(state.table.astype(np.uint8).tobytes() + np.float32(state.global_scale).astype("<f4").tobytes())
        else:
            scale_offset, global_offset, _ = nvfp4_offsets(state.shape)
            output.seek(base_offset + scale_offset)
            output.write(state.scales.astype(np.uint8).tobytes())
            output.seek(base_offset + global_offset)
            output.write(np.float32(state.global_scale).astype("<f4").tobytes())
        return
    bits, group_size, _, _ = QUANT[state.format_name]
    groups = k_pad // group_size
    base, high = pack_codes(state.codes.reshape(-1, group_size), bits)
    high_offset = align_up(n * groups * (24 if bits == 3 else 32), 256)
    scale_offset = high_offset + align_up(n * groups * (8 if bits == 5 else 0), 256)
    output.seek(base_offset)
    output.write(base)
    if high:
        output.seek(base_offset + high_offset)
        output.write(high)
    output.seek(base_offset + scale_offset)
    output.write(np.ascontiguousarray(state.scales.astype("<f2")).tobytes())
