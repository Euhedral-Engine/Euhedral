#!/usr/bin/env python3
"""Lossless compression of NVFP4 safetensors checkpoints.

  nvfp4z.py compress   SRC DST   SRC is a checkpoint directory (or one .safetensors file); DST gets one .nvfp4z per shard
  nvfp4z.py decompress SRC DST   rebuilds the original shards, byte for byte, and checks each against its recorded SHA-256
  nvfp4z.py info       SRC       sizes and the compression of each file

NVFP4 tensors (4-bit codes with an E4M3 scale per 16 weights) and BF16 tensors are entropy coded on the GPU; every other
tensor and every non-safetensors file is stored as it is. See docs/NVFP4_LOSSLESS.md and tools/README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from nvfp4z import codec, pipeline  # noqa: E402


def gigabytes(value: float) -> str:
    return f"{value / 1e9:.2f} GB"


def collect(source: Path, target: Path, suffix_in: str, suffix_out: str, force: bool) -> tuple[list, list]:
    """(jobs for the files with suffix_in, other files to copy)."""
    if source.is_file():
        files = [source]
        base = source.parent
    else:
        files = sorted(path for path in source.rglob("*") if path.is_file())
        base = source
    jobs, copies = [], []
    for path in files:
        relative = path.relative_to(base)
        if path.name.endswith(suffix_in):
            out = target / relative.with_name(relative.name[:len(relative.name) - len(suffix_in)] + suffix_out)
            if force or not out.exists():
                jobs.append(pipeline.Job(path, out))
        elif source.is_dir():
            out = target / relative
            if force or not out.exists():
                copies.append((path, out))
    return jobs, copies


def run(args, compressing: bool) -> int:
    source, target = Path(args.source), Path(args.target)
    suffix_in, suffix_out = (".safetensors", ".safetensors" + pipeline.EXTENSION) if compressing else (".safetensors" + pipeline.EXTENSION, ".safetensors")
    jobs, copies = collect(source, target, suffix_in, suffix_out, args.force)
    for _, out in [(None, job.target) for job in jobs] + copies:
        out.parent.mkdir(parents=True, exist_ok=True)
    for path, out in copies:
        shutil.copyfile(path, out)
    totals = [0, 0]
    started = time.perf_counter()

    def report(result: pipeline.Result) -> None:
        totals[0] += result.bytes_in
        totals[1] += result.bytes_out
        print(f"{result.name}: {gigabytes(result.bytes_in)} -> {gigabytes(result.bytes_out)} "
              f"({result.bytes_in / max(1, result.bytes_out):.4f}x) {result.seconds:.1f}s", flush=True)

    options = dict(readers=args.readers, writers=args.writers, gpus=args.gpu_threads, pinned_gb=args.pinned_gb, fsync=not args.no_fsync, direct_io=args.direct_io, on_result=report)
    if compressing:
        pipeline.compress_files(jobs, verify=not args.no_verify, segment_blocks=args.segment_blocks, **options)
    else:
        pipeline.decompress_files(jobs, **options)
    elapsed = time.perf_counter() - started
    if jobs:
        original, stored = (totals[0], totals[1]) if compressing else (totals[1], totals[0])
        print(f"{len(jobs)} files: {gigabytes(totals[0])} -> {gigabytes(totals[1])} in {elapsed:.1f}s "
              f"({totals[0] / elapsed / 1e9:.2f} GB/s in, {totals[1] / elapsed / 1e9:.2f} GB/s out); "
              f"ratio {original / max(1, stored):.4f}x")
    else:
        print("nothing to do")
    return 0


def info(args) -> int:
    source = Path(args.source)
    files = [source] if source.is_file() else sorted(source.rglob("*" + pipeline.EXTENSION))
    total_in = total_out = 0
    for path in files:
        meta, block = pipeline.read_container_meta(path)
        size = path.stat().st_size
        total_in += meta["source"]["size"]
        total_out += size
        kinds = {}
        for record in meta["tensors"]:
            kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
        print(f"{path.name}: {gigabytes(meta['source']['size'])} -> {gigabytes(size)} "
              f"({meta['source']['size'] / size:.4f}x); tensors {kinds}")
    if len(files) > 1:
        print(f"total: {gigabytes(total_in)} -> {gigabytes(total_out)} ({total_in / max(1, total_out):.4f}x)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("compress", "decompress"):
        command = sub.add_parser(name)
        command.add_argument("source")
        command.add_argument("target")
        command.add_argument("--readers", type=int, default=3, help="threads reading shards (default 3)")
        command.add_argument("--gpu-threads", type=int, default=2, help="threads driving the GPU, one stream each (default 2)")
        command.add_argument("--writers", type=int, default=4, help="threads writing shards (default 4)")
        command.add_argument("--pinned-gb", type=float, default=None,
                             help="page-locked host memory for shard buffers in GiB (default: a quarter of RAM, at most 16)")
        command.add_argument("--direct-io", choices=("auto", "on", "off"), default="auto",
                             help="write outputs without the page cache (auto: on exFAT, where it is faster)")
        command.add_argument("--no-fsync", action="store_true", help="do not flush each output file to storage")
        command.add_argument("--force", action="store_true", help="replace existing outputs (default: skip them)")
        if name == "compress":
            command.add_argument("--no-verify", action="store_true", help="skip the GPU round-trip check of every segment")
            command.add_argument("--segment-blocks", type=int, default=codec.SEGMENT_BLOCKS,
                                 help="blocks of 16 weights per independently coded segment (default %(default)s)")
    show = sub.add_parser("info")
    show.add_argument("source")
    args = parser.parse_args(argv)
    if args.command == "info":
        return info(args)
    return run(args, args.command == "compress")


if __name__ == "__main__":
    raise SystemExit(main())
