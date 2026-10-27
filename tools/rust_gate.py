#!/usr/bin/env python3
"""Run the Java and the Rust benchmark on the same configuration, alternately, and compare what they produced.

Usage:
  tools/rust_gate.py --java-config CONFIG.json --rust-config CONFIG.json --forks N [--memory-max 24G]
                     [--out-dir DIR] [--speculation config|off] [--dry-run]

Each fork runs one process of each engine; the order alternates per fork (Java first on odd forks, Rust first on even
ones), so slow drift of the machine falls on both. Every process goes through the shared GPU launcher, which takes the
exclusive GPU lock, caps host memory in its own scope and kills the job at the time limit:

  gpu-run -m <memory-max> -t <timeout> -- <command>

(--gpu-run names the launcher; it is looked up on PATH by default. The lock belongs to the launcher.)

The Java arm is the benchmark's documented launch (mise.toml, task benchmark-smoke):
  ./gradlew :benchmark:run --args='run CONFIG --fork-id ID'
run with --no-daemon, so that the benchmark's JVM is a child of the launcher's scope and the cap holds for it (a daemon runs
outside the scope). The Rust arm is the release binary `euhedral-bench run CONFIG --fork-id ID`.

Both arms read a derived copy of their configuration, written next to the results:
- the prefix cache is off (`prefixCacheBytes` 0), because the Rust engine always prefills the whole prompt, so a Java
  prefix cache would answer every repeated prompt from stored state and the arms would run different work;
- the output is a per-arm, per-fork JSONL file, overwritten;
- at least one warmup iteration per scenario runs before the measured ones (the first frame of a fresh process compiles
  the kernels it uses, in either engine);
- `tokenizer` is dropped from the Java copy (its configuration rejects unknown fields; the Rust one ignores it);
- with --speculation off, `speculation` is "off" in both copies (one token per quantum).

The report prints, per measured row, the output hash of each engine and whether they are identical (greedy decoding
must produce the same tokens), and the absolute medians per engine and scenario over the successful measured rows.

Exit code: 0 when every row of both arms succeeded and every pair of output hashes is identical, 1 when a row failed or
a hash differs, 2 when an arm could not run.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GPU_RUN = "gpu-run"
ARMS = ("java", "rust")


def derived_config(path: Path, arm: str, output: Path, speculation: str) -> dict:
    config = json.loads(path.read_text())
    config["prefixCacheBytes"] = 0
    config["output"] = str(output)
    config["overwrite"] = True
    config["append"] = False
    config["warmup"] = max(int(config.get("warmup", 1)), 1)
    if speculation == "off":
        config["speculation"] = "off"
    if arm == "java":
        config.pop("tokenizer", None)
    return config


def rust_binary(build: bool) -> Path:
    env = dict(os.environ)
    env.pop("RUSTUP_TOOLCHAIN", None)  # the repository's rust-toolchain.toml chooses the compiler
    if build:
        subprocess.run(
            ["cargo", "build", "--release", "-p", "euhedral-inference-bench"], cwd=REPO / "rust", env=env, check=True
        )
    metadata = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=REPO / "rust",
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    binary = Path(json.loads(metadata)["target_directory"]) / "release" / "euhedral-bench"
    if not binary.is_file():
        sys.exit(f"{binary} is missing: build it with `cargo build --release -p euhedral-inference-bench` in rust/")
    return binary


def arm_command(arm: str, config: Path, fork: int, binary: Path) -> list[str]:
    if arm == "java":
        arguments = f"run {shlex.quote(str(config))} --fork-id {fork}"
        return ["./gradlew", "--no-daemon", "--console=plain", "-q", ":benchmark:run", f"--args={arguments}"]
    return [str(binary), "run", str(config), "--fork-id", str(fork)]


def scoped(command: list[str], gpu_run: str, memory_max: str, timeout: str) -> list[str]:
    return [gpu_run, "-m", memory_max, "-t", timeout, "--", *command]


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def median(values: list[float]) -> str:
    return f"{statistics.median(values):.1f}" if values else "-"


def report(out_dir: Path, forks: int) -> int:
    rows: dict[str, dict[int, list[dict]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for fork in range(1, forks + 1):
            rows[arm][fork] = [r for r in read_rows(out_dir / f"{arm}-fork{fork}.jsonl") if not r["warmup"]]

    lines: list[str] = []
    worst = 0
    lines.append("output hashes (sha256 of the comma-joined token ids), measured rows")
    lines.append(f"{'scenario':<24} {'fork':>4} {'iter':>4}  {'java':<14} {'rust':<14} {'status j/r':<22} identical")
    identical = different = missing = 0
    for fork in range(1, forks + 1):
        by_key = {
            arm: {(r["scenario"]["name"], r["iteration"]): r for r in rows[arm][fork]} for arm in ARMS
        }
        for key in sorted(set(by_key["java"]) | set(by_key["rust"])):
            java, rust = by_key["java"].get(key), by_key["rust"].get(key)
            hashes = [(r["work"].get("outputSha256") if r else None) for r in (java, rust)]
            statuses = "/".join((r["status"] if r else "absent") for r in (java, rust))
            if None in hashes:
                verdict = "n/a"
                missing += 1
            elif hashes[0] == hashes[1]:
                verdict = "yes"
                identical += 1
            else:
                verdict = "NO"
                different += 1
            lines.append(
                f"{key[0]:<24} {fork:>4} {key[1]:>4}  {(hashes[0] or '-')[:12]:<14} {(hashes[1] or '-')[:12]:<14} "
                f"{statuses:<22} {verdict}"
            )
            if any(r is None or r["status"] != "success" for r in (java, rust)):
                worst = max(worst, 1)
    lines.append(f"identical {identical}, different {different}, not comparable {missing}")
    if different or missing:
        worst = max(worst, 1)

    lines.append("")
    lines.append("absolute medians over the successful measured rows of all forks")
    header = f"{'engine':<6} {'scenario':<24} {'rows':>4} {'prefill t/s':>12} {'ttft ms':>9} {'decode t/s':>11} {'e2e ms':>9}"
    lines.append(header)
    for arm in ARMS:
        every = [r for fork in rows[arm].values() for r in fork]
        for name in dict.fromkeys(r["scenario"]["name"] for r in every):
            ok = [r for r in every if r["scenario"]["name"] == name and r["status"] == "success"]

            def of(get):
                return [v for v in (get(r) for r in ok) if v is not None]

            lines.append(
                f"{arm:<6} {name:<24} {len(ok):>4} "
                f"{median(of(lambda r: r['throughput']['prefillTokensPerSecond'])):>12} "
                f"{median(of(lambda r: r['timings']['timeToFirstToken'] / 1e6 if r['timings']['timeToFirstToken'] else None)):>9} "
                f"{median(of(lambda r: r['throughput']['decodeTokensPerSecond'])):>11} "
                f"{median(of(lambda r: r['timings']['endToEnd'] / 1e6)):>9}"
            )
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (out_dir / "summary.txt").write_text(text)
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--java-config", type=Path, required=True)
    parser.add_argument("--rust-config", type=Path, required=True)
    parser.add_argument("--forks", type=int, default=1)
    parser.add_argument("--memory-max", default="24G", help="host memory cap of each arm (never above 32G)")
    parser.add_argument("--timeout", default="30m", help="time limit of each arm, as `timeout` reads it")
    parser.add_argument("--gpu-run", default=GPU_RUN, help="the GPU launcher: lock, memory cap, time limit")
    parser.add_argument("--out-dir", type=Path, help="default: benchmark-results/rust-gate-<time>")
    parser.add_argument("--speculation", choices=("config", "off"), default="config")
    parser.add_argument("--no-build", action="store_true", help="use the release binary as it is")
    parser.add_argument("--dry-run", action="store_true", help="write the configurations, print the commands, run nothing")
    args = parser.parse_args()
    if args.forks < 1:
        parser.error("--forks must be at least 1")

    out_dir = (args.out_dir or REPO / "benchmark-results" / time.strftime("rust-gate-%Y%m%dT%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = Path("euhedral-bench") if args.dry_run else rust_binary(build=not args.no_build)
    sources = {"java": args.java_config, "rust": args.rust_config}

    failed = False
    for fork in range(1, args.forks + 1):
        # Java first on odd forks, Rust first on even ones.
        for arm in (ARMS if fork % 2 == 1 else ARMS[::-1]):
            output = out_dir / f"{arm}-fork{fork}.jsonl"
            config = out_dir / f"{arm}-fork{fork}.json"
            config.write_text(json.dumps(derived_config(sources[arm], arm, output, args.speculation), indent=2) + "\n")
            command = scoped(arm_command(arm, config, fork, binary), args.gpu_run, args.memory_max, args.timeout)
            if args.dry_run:
                print(shlex.join(command))
                continue
            log = out_dir / f"{arm}-fork{fork}.log"
            print(f"fork {fork} {arm}: running (log {log})", flush=True)
            started = time.monotonic()
            with log.open("w") as handle:
                result = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)  # gpu-run bounds the wait and the run
            took = time.monotonic() - started
            # 1: some measured row failed or was ineligible (the rows are still written and compared below).
            print(f"fork {fork} {arm}: exit {result.returncode} after {took:.0f} s", flush=True)
            if result.returncode not in (0, 1):
                failed = True
                print(log.read_text()[-2000:], file=sys.stderr)
    if args.dry_run:
        return 0
    worst = report(out_dir, args.forks)
    return 2 if failed else worst


if __name__ == "__main__":
    sys.exit(main())
