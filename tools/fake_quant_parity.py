#!/usr/bin/env python3
"""Check that the fake-quant modules (euhedral_artifacts/fake_quant.py) reproduce a converted artifact byte for byte.

Usage: fake_quant_parity.py --model CHECKPOINT_DIR --artifact ARTIFACT.edrl --quantization q3|nvfp4 [--compressed]
                            [--imatrix IMATRIX.gguf] [--objects REGEX] [--export DIR]

Rebuilds every quantized object of the artifact (those whose names match REGEX) from the checkpoint as a fake-quant
module, with the converter's rounding (calibrated with --imatrix, as the artifact was converted), and compares:
the module's exported codes and scales, packed as the converter packs them, against the artifact's bytes for that
object; and the module's forward pass against the weights those codes and scales decode to. --export saves the
modules' states as a codes directory for convert_checkpoint.py --codes-from. Large objects are built in row slices
that share the object's tensor-level scales. Needs PyTorch, preferably with CUDA. Compressed q3 stores its Q3
objects in the lossless P2E2 layout; check the q3 artifact it is transcoded from instead.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from euhedral_artifacts import device  # noqa: E402
from euhedral_artifacts.codes import QuantState, state_file, state_size, write_state  # noqa: E402
from euhedral_artifacts.edrl import FORMAT_ORDINAL, LAYOUT_ORDINAL, align_up, read_table  # noqa: E402
from euhedral_artifacts.importance import ImportanceMatrix  # noqa: E402
from euhedral_artifacts.inventory import build_plans, draft_token_ids  # noqa: E402
from euhedral_artifacts.recipes import LAYOUT_SD4, NVFP4_FORMAT, Recipe  # noqa: E402
from euhedral_artifacts.sources import SourceStore, device_rows  # noqa: E402

# Rows per module slice: about 128M weights (512 MB of float32 master weights on the device).
SLICE_VALUES = 128 * 1024 * 1024


class Bytes:
    """A writable in-memory object region for write_state."""

    def __init__(self, size: int):
        self.buffer, self.position = bytearray(size), 0

    def seek(self, position: int) -> None:
        self.position = position

    def write(self, data: bytes) -> None:
        self.buffer[self.position:self.position + len(data)] = data
        self.position += len(data)


def pread(fd: int, size: int, offset: int) -> bytes:
    parts, done = [], 0
    while done < size:
        part = os.pread(fd, size - done, offset + done)
        if not part:
            raise SystemExit("artifact is truncated")
        parts.append(part)
        done += len(part)
    return b"".join(parts)


def decoded(state: QuantState):
    """The weights `state` decodes to, on the device (float32 [rows, K_pad])."""
    from euhedral_artifacts.nvfp4 import E2M1_VALUES, E4M3_VALUES, NVFP4_BLOCK
    torch = device.torch()
    codes = torch.from_numpy(state.codes).to(device.DEVICE)
    if state.format_name == NVFP4_FORMAT:
        scale_codes = state.table[state.scales] if state.table is not None else state.scales
        block = device.table("e4m3", E4M3_VALUES)[torch.from_numpy(scale_codes).to(device.DEVICE).long()]
        block = block * torch.tensor(state.global_scale, dtype=torch.float32, device=device.DEVICE)
        sign = torch.where((codes & 8) != 0, -1.0, 1.0)
        values = device.table("e2m1", E2M1_VALUES)[(codes & 7).long()] * sign
        return (values.view(codes.shape[0], -1, NVFP4_BLOCK) * block[..., None]).view(codes.shape)
    scales = torch.from_numpy(state.scales.astype(np.float32)).to(device.DEVICE)
    return (codes.float().view(codes.shape[0], scales.shape[1], -1) * scales[..., None]).view(codes.shape)


def rebuild(plan) -> tuple[QuantState, bool]:
    """(the object's state rebuilt by fake-quant modules, whether every module's forward equals its decode)."""
    from euhedral_artifacts.fake_quant import Nvfp4FakeQuant, fake_quant, nvfp4_chunk_rows, nvfp4_tensor_scales
    torch = device.torch()
    matrix, calibration = plan.source
    n, k = plan.shape
    k_pad = align_up(k, 128)
    nvfp4 = plan.format_name == NVFP4_FORMAT
    sd4 = plan.layout == LAYOUT_SD4
    step = max(1, SLICE_VALUES // k_pad)
    if nvfp4:
        chunk = nvfp4_chunk_rows((n, k), calibration)
        step = max(chunk, step // chunk * chunk)
    slices = [(begin, min(n, begin + step)) for begin in range(0, n, step)]
    tensor = None
    if nvfp4:
        tensor = nvfp4_tensor_scales(lambda: (device_rows(matrix, b, e, k_pad) for b, e in slices), (n, k), calibration, sd4)
    states, forward_equal = [], True
    for begin, end in slices:
        values = device_rows(matrix, begin, end, k_pad)
        if nvfp4:
            module = Nvfp4FakeQuant.from_values(values, (n, k), calibration, sd4, tensor)
        else:
            module = fake_quant(values, plan.format_name, plan.layout, (n, k), calibration)
        state = module.state()
        with torch.no_grad():
            forward_equal &= bool(torch.equal(module(), decoded(state)))
        states.append(state)
        del module, values
    first = states[0]
    return QuantState(first.format_name, first.layout, (n, k), np.concatenate([s.codes for s in states]),
                      np.concatenate([s.scales for s in states]), first.global_scale, first.table), forward_equal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fake_quant_parity.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, metavar="CHECKPOINT_DIR")
    parser.add_argument("--artifact", required=True, type=Path, metavar="ARTIFACT.edrl")
    parser.add_argument("--quantization", required=True, choices=("q3", "nvfp4"))
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--imatrix", type=Path, metavar="IMATRIX.gguf")
    parser.add_argument("--objects", default=".*", metavar="REGEX", help="objects to check (default: all)")
    parser.add_argument("--export", type=Path, metavar="DIR", help="save the checked objects' states here")
    args = parser.parse_args(argv)
    recipe = Recipe(args.quantization, args.compressed)
    if recipe.quantization == "q3" and recipe.compressed:
        parser.error("compressed q3 is the q3 artifact transcoded to P2E2; check the q3 artifact")
    device.set_device(device.default_device())
    importance = ImportanceMatrix(args.imatrix) if args.imatrix is not None else None
    pattern = re.compile(args.objects)
    if args.export is not None:
        args.export.mkdir(parents=True, exist_ok=True)
    formats = {ordinal: name for name, ordinal in FORMAT_ORDINAL.items()}
    layouts = {ordinal: name for name, ordinal in LAYOUT_ORDINAL.items()}
    with args.artifact.open("rb") as handle:
        _, _, objects = read_table(handle)
    stored = {obj["name"]: obj for obj in objects}
    checked = mismatched = 0
    with SourceStore(args.model) as store, args.artifact.open("rb") as artifact:
        plans = [plan for plan in build_plans(store, draft_token_ids(args.artifact), recipe, importance)
                 if plan.source is not None and pattern.search(plan.name)]
        for index, plan in enumerate(plans):
            start = time.monotonic()
            obj = stored.get(plan.name)
            if obj is None or (formats[obj["format"]], layouts[obj["layout"]], obj["shape"]) != (
                    plan.format_name, plan.layout, tuple(plan.shape)):
                print(f"MISMATCH {plan.name}: the artifact does not store it as {plan.format_name}/{plan.layout}",
                      flush=True)
                mismatched += 1
                continue
            state, forward_equal = rebuild(plan)
            region = Bytes(state_size(state))
            write_state(region, 0, state)
            equal = len(region.buffer) == obj["bytes"] and bytes(region.buffer) == pread(artifact.fileno(), obj["bytes"],
                                                                                        obj["offset"])
            checked += 1
            if not (equal and forward_equal):
                mismatched += 1
            if args.export is not None:
                state.save(state_file(args.export, plan.name))
            verdict = "ok" if equal and forward_equal else ("BYTES DIFFER" if not equal else "FORWARD DIFFERS")
            print(f"[{index + 1}/{len(plans)}] {plan.name} {plan.format_name}/{plan.layout} {verdict} "
                  f"({time.monotonic() - start:.1f} s)", flush=True)
    print(f"{checked} objects checked, {mismatched} mismatched", flush=True)
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
