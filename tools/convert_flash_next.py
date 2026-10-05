#!/usr/bin/env python3
"""Convert the Qwen3.8-Flash-Next NVFP4 checkpoint to an Euhedral artifact (EDRL version 3).

The checkpoint is already NVFP4 (E2M1 codes, E4M3 block scales, FP32 global scales); nothing is quantized again.
Every code, scale and global scale is copied into the artifact's runtime objects: fixed tensors in the row-split
NVFP4 layout, routed experts as one contiguous record each in per-layer expert banks, n-gram embedding rows
interleaved with their scales. See docs/FLASH_NEXT_ARTIFACT.md.

    convert_flash_next.py convert CHECKPOINT_DIR ARTIFACT.edrl [--jobs N] [--force]
    convert_flash_next.py verify  ARTIFACT.edrl [--checkpoint CHECKPOINT_DIR] [--jobs N]

`convert` is restartable (ARTIFACT.edrl.partial and ARTIFACT.edrl.progress hold the work done so far), writes the
header last, and ends with ARTIFACT.edrl.manifest.json. `verify` checks every range's CRC-32 and, with
--checkpoint, every byte against the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from euhedral_artifacts.qwen4_edrl import convert, verify  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    convert_parser = commands.add_parser("convert")
    convert_parser.add_argument("checkpoint", type=Path)
    convert_parser.add_argument("artifact", type=Path)
    convert_parser.add_argument("--jobs", type=int, default=4)
    convert_parser.add_argument("--force", action="store_true")
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("artifact", type=Path)
    verify_parser.add_argument("--checkpoint", type=Path)
    verify_parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.command == "convert":
            result = convert(args.checkpoint, args.artifact, args.jobs, args.force)
        else:
            result = verify(args.artifact, args.checkpoint, args.jobs)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
