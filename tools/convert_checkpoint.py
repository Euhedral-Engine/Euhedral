#!/usr/bin/env python3
"""Convert a Qwen3.8-27B Hugging Face checkpoint to an Euhedral inference artifact.

Two choices select one of the four artifacts the engine ships:

  --quantization q3                 Q3 weights
  --quantization q3    --compressed Q3 weights, stored compressed   (lossless; same outputs as q3)
  --quantization nvfp4              NVFP4 weights
  --quantization nvfp4 --compressed NVFP4 weights, stored compressed (4.25 bits per weight)
  --quantization trellis            trellis weights of an exllamav3 checkpoint (--trellis-from DIR)

The draft head's token shortlist comes from --ranking (token frequency counts, int64 little endian,
one per vocabulary entry) or from an existing artifact's shortlist (--draft-ids-from); pass one.
Quantization runs on the GPU when PyTorch finds one, otherwise on the CPU. See tools/README.md.

--imatrix FILE rounds by calibrated scale search: every group's scale minimizes its squared error weighted
by the importance of each column (a llama.cpp importance matrix, the mean square of each input column over
a calibration text). The formats and layouts are unchanged.

--codes-from DIR writes the quantized objects DIR holds pre-rounded codes for (one file per object, written by
euhedral_artifacts/fake_quant.py or a training tool) instead of rounding them; the other objects are rounded as
usual. A file must match the object's format, layout and shape in the chosen artifact.

--trellis-from DIR (required with, and only valid for, --quantization trellis) packs the linears of the EXL3
checkpoint calibrate --format trellis wrote into DIR: the engine's objects, each with its input scales, output
scales and segments. Everything else (embedding, norms, GDN a/b, the draft-head shortlist) is built from --model.
The LM head and the draft head are not part of the checkpoint: --codes-from takes the codes calibrate wrote to
DIR/heads, and the conversion refuses when one is missing.

--dflash2 DIR adds the DFlash2 drafter checkpoint (z-lab/Qwen3.8-27B-DFlash2), so the artifact drafts with
DFlash2 instead of MTP. With --extend ARTIFACT it is added to an existing artifact, whose objects are copied
byte for byte, instead of converting the target checkpoint again (--model is then not needed).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from euhedral_artifacts import device  # noqa: E402
from euhedral_artifacts.dflash2 import PROJECTIONS  # noqa: E402
from euhedral_artifacts.pipeline import convert, extend  # noqa: E402
from euhedral_artifacts.recipes import EMBEDDING_FORMATS, HEAD_FORMATS, QUANTIZATIONS, Recipe  # noqa: E402

CUDA_JOBS = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="convert_checkpoint.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, metavar="CHECKPOINT_DIR",
                        help="Hugging Face checkpoint directory (config.json, tokenizer_config.json, safetensors)")
    parser.add_argument("--out", required=True, type=Path, metavar="ARTIFACT.edrl", help="artifact file to write")
    parser.add_argument("--quantization", choices=QUANTIZATIONS,
                        help="weight quantization of the artifact")
    parser.add_argument("--compressed", action="store_true",
                        help="store the weights compressed: q3 is transcoded losslessly, nvfp4 is quantized "
                             "with 4.25 bits per weight")
    parser.add_argument("--draft-ids-from", type=Path, metavar="ARTIFACT",
                        help="reuse the draft-head token shortlist of this existing artifact")
    parser.add_argument("--ranking", type=Path, metavar="FILE",
                        help="token frequency counts (int64 little endian per vocabulary entry) to choose the "
                             "draft-head shortlist")
    parser.add_argument("--device", metavar="DEV", help="cuda or cpu (default: cuda when PyTorch has a CUDA device)")
    parser.add_argument("--jobs", type=int, metavar="N",
                        help=f"objects quantized in parallel worker processes (default: {CUDA_JOBS} on cuda, "
                             "one per CPU on cpu)")
    parser.add_argument("--dflash2", type=Path, metavar="DRAFT_DIR",
                        help="add this DFlash2 drafter checkpoint (config.json, model.safetensors)")
    parser.add_argument("--dflash2-projections", choices=PROJECTIONS, default="bf16",
                        help="format of the drafter's attention and MLP projections (default: bf16)")
    parser.add_argument("--extend", type=Path, metavar="ARTIFACT",
                        help="add the --dflash2 drafter to this existing artifact instead of converting --model")
    parser.add_argument("--imatrix", type=Path, metavar="IMATRIX.gguf",
                        help="llama.cpp importance matrix: choose every group's scale by a search weighted by "
                             "column importance instead of from its largest value (needs PyTorch)")
    parser.add_argument("--codes-from", type=Path, metavar="DIR",
                        help="take the codes and scales of the objects DIR holds files for (see above)")
    parser.add_argument("--trellis-from", type=Path, metavar="EXL3_DIR",
                        help="exllamav3 checkpoint to take the linears from (with --quantization trellis)")
    parser.add_argument("--output-head", choices=HEAD_FORMATS, metavar="FORMAT",
                        help="the LM head's format in place of the recipe's (" + ", ".join(HEAD_FORMATS) + ")")
    parser.add_argument("--token-embedding", choices=EMBEDDING_FORMATS, metavar="FORMAT",
                        help="the token embedding's format in place of Q3 (" + ", ".join(EMBEDDING_FORMATS) + ")")
    parser.add_argument("--force", action="store_true", help="replace an existing artifact")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.extend is not None:
        if args.dflash2 is None:
            parser.error("--extend needs --dflash2")
        name = args.device or device.default_device()
        device.set_device(name)
        try:
            manifest = extend(args.extend, args.dflash2, args.out, args.dflash2_projections, args.force)
        except (OSError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
        return 0
    if args.model is None or args.quantization is None:
        parser.error("--model and --quantization are required")
    if (args.ranking is None) == (args.draft_ids_from is None):
        parser.error("pass exactly one of --ranking and --draft-ids-from")
    if (args.quantization == "trellis") != (args.trellis_from is not None):
        parser.error("--trellis-from is required with, and only valid for, --quantization trellis")
    name = args.device or device.default_device()
    if name not in ("cpu", "cuda"):
        parser.error("--device must be cuda or cpu")
    jobs = args.jobs if args.jobs is not None else (CUDA_JOBS if name == "cuda" else os.cpu_count() or 1)
    if jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.imatrix is not None:
        try:
            device.torch()
        except ImportError:
            parser.error("--imatrix needs PyTorch")
    device.set_device(name)
    trellis = {"trellis_from": args.trellis_from} if args.trellis_from is not None else {}
    try:
        manifest = convert(args.model, args.out, Recipe(args.quantization, args.compressed, args.output_head, args.token_embedding), args.ranking,
                           args.draft_ids_from, jobs, args.force, args.dflash2, args.dflash2_projections,
                           args.imatrix, args.codes_from, **trellis)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
