"""Scores candidate artifacts on the frozen evaluation set and prints aggregate and per-domain numbers only.

Usage (from the repository root): python3 tools/t5/run.py STORE --baseline NAME NAME=ARTIFACT...
                                  [--engine java|rust] [--gradle ./gradlew] [--rust-bench PATH]
                                  [--cuda-library PATH] [--resamples 10000]

The runner refuses a store whose files differ from STORE/frozen/manifest.json, scores every artifact with
`quantization score --values` (`--engine java`, the default) or `euhedral-bench score --values` (`--engine rust`, with
`--rust-bench` naming the binary) into a private directory under STORE/runs, prints the report, and deletes the per-token
values: nothing per token, per chunk or per prompt leaves the runner.

Per domain and overall: mean KL(reference || candidate) over scored tokens, top-1 agreement, KL p95 and p99, dNLL
(candidate NLL minus reference NLL), and the paired KL difference against the baseline with a 95% interval from a
bootstrap over chunks (each chunk is one conversation or document; resampled within each domain).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import shutil
import struct
import subprocess
import sys
from pathlib import Path

FIELDS = ("nll", "reference_nll", "kl", "agree", "probability_difference")
BOOTSTRAP_SEED = 20261017


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def read_values(path: Path) -> list[list[tuple[float, ...]]]:
    """Per chunk, per scored token, the scorer's values (FIELDS)."""
    data = path.read_bytes()
    chunks, steps, fields = struct.unpack_from("<3i", data)
    if fields != len(FIELDS):
        raise ValueError(f"{path}: {fields} values per token, expected {len(FIELDS)}")
    flat = struct.unpack_from(f"<{chunks * steps * fields}d", data, 12)
    return [[flat[(c * steps + s) * fields:(c * steps + s + 1) * fields] for s in range(steps)] for c in range(chunks)]


def score_command(args: argparse.Namespace, name: str, artifact: str, private: Path) -> list[str]:
    """The command that scores one artifact with the chosen engine; both write DIR/NAME.values and the same summary."""
    reference = args.store / "frozen" / "reference.kld"
    if args.engine == "rust":
        if args.rust_bench is None:
            raise SystemExit("error: --engine rust needs --rust-bench, the path of the euhedral-bench binary")
        command = [str(args.rust_bench), "score", "--artifact", artifact, "--results", str(private / "score.txt"),
                   "--reference", str(reference), "--name", name, "--values", str(private)]
        if args.cuda_library is not None:
            command += ["--cuda-library", str(args.cuda_library)]
        return command
    return [args.gradle, "--no-daemon", "-q", ":quantization:run",
            f"--args=score --artifact {artifact} --results {private / 'score.txt'} "
            f"--reference {reference} --name {name} --values {private}"]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary(chunks: list[list[tuple[float, ...]]]) -> dict:
    tokens = [t for c in chunks for t in c]
    kl = [t[2] for t in tokens]
    return {"kl": sum(kl) / len(kl), "top1": sum(t[3] for t in tokens) / len(tokens),
            "p95": percentile(kl, 0.95), "p99": percentile(kl, 0.99),
            "dnll": sum(t[0] - t[1] for t in tokens) / len(tokens)}


def chunk_kl(chunk: list[tuple[float, ...]]) -> float:
    return sum(t[2] for t in chunk) / len(chunk)


def paired(candidate: list[float], baseline: list[float], strata: list[str], resamples: int) -> dict:
    """Mean per-chunk KL difference and its 95% bootstrap interval, chunks resampled within each stratum."""
    differences = [c - b for c, b in zip(candidate, baseline)]
    groups: dict[str, list[int]] = {}
    for index, stratum in enumerate(strata):
        groups.setdefault(stratum, []).append(index)
    rng = random.Random(BOOTSTRAP_SEED)
    means = []
    for _ in range(resamples):
        drawn = [differences[rng.choice(members)] for members in groups.values() for _ in members]
        means.append(sum(drawn) / len(drawn))
    base = sum(baseline) / len(baseline)
    mean = sum(differences) / len(differences)
    return {"difference": mean, "low": percentile(means, 0.025), "high": percentile(means, 0.975),
            "relative": mean / base if base else float("nan")}


def report(scored: dict[str, list[list[tuple[float, ...]]]], domains: list[str], baseline: str,
           resamples: int) -> str:
    names = list(scored)
    order = list(dict.fromkeys(domains))
    lines = []
    for scope in ["overall", *order]:
        members = [i for i, d in enumerate(domains) if scope == "overall" or d == scope]
        lines.append(f"{scope} ({len(members)} chunks)")
        lines.append(f"  {'artifact':<24} {'KL':>9} {'top-1':>7} {'KL p95':>8} {'KL p99':>8} {'dNLL':>9}   "
                     f"KL vs {baseline} (95% interval)")
        base_kl = [chunk_kl(scored[baseline][i]) for i in members]
        for name in names:
            chunks = [scored[name][i] for i in members]
            s = summary(chunks)
            line = (f"  {name:<24} {s['kl']:>9.5f} {s['top1']:>7.4f} {s['p95']:>8.4f} {s['p99']:>8.4f} "
                    f"{s['dnll']:>+9.5f}")
            if name != baseline:
                p = paired([chunk_kl(c) for c in chunks], base_kl, [domains[i] for i in members], resamples)
                line += (f"   {p['difference']:+.5f} ({p['low']:+.5f} .. {p['high']:+.5f}), "
                         f"{100 * p['relative']:+.1f}%")
            lines.append(line)
        lines.append("")
    return "\n".join(lines)


def check_store(store: Path) -> None:
    manifest = json.loads((store / "frozen" / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = store / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise SystemExit(f"error: {relative} differs from the frozen manifest (version {manifest['version']})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store", type=Path)
    parser.add_argument("--baseline", required=True, help="the NAME the others are compared with")
    parser.add_argument("artifacts", nargs="+", help="NAME=ARTIFACT")
    parser.add_argument("--engine", choices=("java", "rust"), default="java",
                        help="the engine that scores the candidates (default java)")
    parser.add_argument("--gradle", default="./gradlew", help="the Gradle wrapper of --engine java")
    parser.add_argument("--rust-bench", type=Path, help="the euhedral-bench binary of --engine rust")
    parser.add_argument("--cuda-library", type=Path, help="the CUDA library of --engine rust, when not at its default")
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args(argv)
    artifacts = dict(item.split("=", 1) for item in args.artifacts)
    if args.baseline not in artifacts:
        raise SystemExit("error: the baseline must be one of the artifacts")
    check_store(args.store)
    started = datetime.datetime.now().astimezone()
    runs = args.store / "runs"
    private = runs / f"{started:%Y%m%d-%H%M%S}.values"
    private.mkdir(parents=True)
    try:
        scored = {}
        for name, path in artifacts.items():
            subprocess.run(score_command(args, name, path, private),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            scored[name] = read_values(private / f"{name}.values")
        domains = [json.loads(line)["domain"]
                   for line in (args.store / "frozen" / "chunks.jsonl").read_text().splitlines()]
        text = report(scored, domains, args.baseline, args.resamples)
    finally:
        shutil.rmtree(private)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
