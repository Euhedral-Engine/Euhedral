#!/usr/bin/env python3
"""Run the Java and the Rust benchmark on the same configuration, alternately, and compare what they produced.

Usage:
  tools/rust_gate.py --java-config CONFIG.json --rust-config CONFIG.json --forks N [--memory-max 24G] [--timeout 30m]
                     [--gpu-run PATH] [--out-dir DIR] [--speculation config|off] [--no-build] [--dry-run]

Each fork runs one process of each engine; the order alternates per fork (Java first on odd forks, Rust first on even
ones), so slow drift of the machine falls on both. Every process goes through the shared GPU launcher, which takes the
exclusive GPU lock, caps host memory in its own scope and kills the job at the time limit:

  gpu-run -m <memory-max> -t <timeout> -- <command>

(--gpu-run names the launcher; it is looked up on PATH by default. The lock belongs to the launcher.)

The Java arm is the benchmark's start script (`:benchmark:installDist`, built once before the forks, outside the lock;
BENCHMARKING.md documents it), which `exec`s a single JVM, so the memory cap and the time limit bind the benchmark itself
and Gradle never runs under the GPU lock. The Rust arm is the release binary `euhedral-bench run CONFIG --fork-id ID`
(built once before the forks).

Both arms read a derived copy of their configuration, written next to the results:
- the prefix cache is off (`prefixCacheBytes` 0), because the Rust engine always prefills the whole prompt, so a Java
  prefix cache would answer every repeated prompt from stored state and the arms would run different work;
- `kvDeviceMinimumTokens` is stated in both copies (the configuration's value, else 65536), so no engine default decides it;
- the output is a per-arm, per-fork JSONL file, removed before the arm starts, so a file that is there afterwards is this
  run's;
- at least one warmup iteration per scenario runs before the measured ones (the first frame of a fresh process compiles
  the kernels it uses, in either engine);
- `tokenizer` is dropped from the Java copy (its configuration rejects unknown fields; the Rust one ignores it);
- with --speculation off, `speculation` is "off" in both copies (one token per quantum).

The report, per measured row, compares the prompt (`promptSha256` and `promptTokens`), the output
(`outputSha256`, the SHA-256 of the comma-joined token ids), and the accepted-drafts histogram when both rows carry one.
A row without output (prefill) is compared on its prompt only and is counted as "no output", never as an output pass.
It then prints absolute figures per engine: pooled medians over all forks, and the median of each fork, with the number of
forks in which the Rust figure is at least the Java one. Decode tok/s is computed here for both engines over the same
window, first token found to last token found: `decodeSampledTokens / (timeToLastToken - timeToFirstToken)`; each
engine's own `decodeTokensPerSecond` is not used. There is no end-to-end column: Java's end-to-end includes a commit-only
quantum that the Rust engine does not run.

Exit code: 0 when every row of both arms succeeded and every comparison is identical, 1 when a row failed or something
differs or is missing, 2 when an arm could not run or the arguments are wrong.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GPU_RUN = "gpu-run"
ARMS = ("java", "rust")


JAVA_START_SCRIPT = REPO / "benchmark" / "build" / "install" / "euhedral-inference-benchmark" / "bin" / "euhedral-inference-benchmark"
DEFAULT_KV_DEVICE_MINIMUM_TOKENS = 65536
MAX_MEMORY_BYTES = 32 << 30


def derived_config(path: Path, arm: str, output: Path, speculation: str) -> dict:
    config = json.loads(path.read_text())
    config["prefixCacheBytes"] = 0
    config["kvDeviceMinimumTokens"] = config.get("kvDeviceMinimumTokens", DEFAULT_KV_DEVICE_MINIMUM_TOKENS)
    config["output"] = str(output)
    config["overwrite"] = True
    config["append"] = False
    config["warmup"] = max(int(config.get("warmup", 1)), 1)
    if speculation == "off":
        config["speculation"] = "off"
    if arm == "java":
        config.pop("tokenizer", None)
    return config


def cargo_env() -> dict:
    env = dict(os.environ)
    env.pop("RUSTUP_TOOLCHAIN", None)  # the repository's rust-toolchain.toml chooses the compiler
    return env


def rust_binary(build: bool) -> Path:
    env = cargo_env()
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


def java_start_script(build: bool) -> Path:
    if build:
        subprocess.run(["./gradlew", "--console=plain", "-q", ":benchmark:installDist"], cwd=REPO, check=True)
    if not JAVA_START_SCRIPT.is_file():
        sys.exit(f"{JAVA_START_SCRIPT} is missing: run `./gradlew :benchmark:installDist`")
    return JAVA_START_SCRIPT


def arm_command(arm: str, config: Path, fork: int, launchers: dict[str, Path]) -> list[str]:
    return [str(launchers[arm]), "run", str(config), "--fork-id", str(fork)]


def scoped(command: list[str], gpu_run: str, memory_max: str, timeout: str) -> list[str]:
    return [gpu_run, "-m", memory_max, "-t", timeout, "--", *command]


def memory_bytes(text: str) -> int:
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    text = text.strip().upper().removesuffix("B")
    try:
        return int(text[:-1]) * units[text[-1]] if text[-1:] in units else int(text)
    except (ValueError, IndexError):
        raise ValueError(f"cannot read {text!r} as a size (use 24G, 512M or bytes)") from None


def read_rows(path: Path) -> tuple[list[dict], int]:
    """The rows of a result file, and the number of lines that are not whole rows (a killed arm leaves one)."""
    if not path.is_file():
        return [], 0
    rows, bad = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return rows, bad


def median_of(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def show(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def decode_tps(row: dict) -> float | None:
    """Decode tokens per second over the window both engines can give: first token found to last token found."""
    timings, work = row.get("timings"), row["work"]
    sampled = work.get("decodeSampledTokens")
    if not timings or not sampled or timings.get("timeToLastToken") is None or timings.get("timeToFirstToken") is None:
        return None
    window = timings["timeToLastToken"] - timings["timeToFirstToken"]
    return sampled * 1e9 / window if window > 0 else None


def prefill_tps(row: dict) -> float | None:
    return (row.get("throughput") or {}).get("prefillTokensPerSecond")


def ttft_ms(row: dict) -> float | None:
    value = (row.get("timings") or {}).get("timeToFirstToken")
    return None if value is None else value / 1e6


METRICS = (("prefill t/s", prefill_tps), ("ttft ms", ttft_ms), ("decode t/s (first-to-last)", decode_tps))


def histogram(row: dict):
    speculative = row["work"].get("speculative")
    return None if not speculative else speculative.get("acceptedDrafts")


def compare(java: dict | None, rust: dict | None) -> tuple[str, str, str, str]:
    """(prompt, output, histogram, verdict) of one pair of rows. A verdict is `ok`, `DIFFERENT` or `missing`."""
    if java is None or rust is None:
        return "-", "-", "-", "missing"
    same_prompt = (
        java["scenario"]["promptSha256"] == rust["scenario"]["promptSha256"]
        and java["work"].get("promptTokens") == rust["work"].get("promptTokens")
    )
    prompt = "same" if same_prompt else "DIFFERENT"
    generated = (java["work"].get("generatedTokens") or 0, rust["work"].get("generatedTokens") or 0)
    if generated == (0, 0):
        output = "no output"
    elif java["work"].get("outputSha256") == rust["work"].get("outputSha256") and generated[0] == generated[1]:
        output = "same"
    else:
        output = "DIFFERENT"
    hists = (histogram(java), histogram(rust))
    if hists[0] is not None and hists[1] is not None:
        drafts = "same" if hists[0] == hists[1] else "DIFFERENT"
    else:
        drafts = "-"
    bad = "DIFFERENT" in (prompt, output, drafts)
    return prompt, output, drafts, "DIFFERENT" if bad else "ok"


def report(out_dir: Path, forks: int, failed_arms: set[tuple[str, int]]) -> int:
    rows: dict[str, dict[int, list[dict]]] = {arm: {} for arm in ARMS}
    bad_lines = 0
    for arm in ARMS:
        for fork in range(1, forks + 1):
            read, bad = read_rows(out_dir / f"{arm}-fork{fork}.jsonl")
            rows[arm][fork] = [r for r in read if not r["warmup"]]
            bad_lines += bad

    lines: list[str] = []
    worst = 1 if bad_lines or failed_arms else 0
    if bad_lines:
        lines.append(f"{bad_lines} result line(s) are not whole rows (an arm was killed); the run is not clean")
    for arm, fork in sorted(failed_arms):
        lines.append(f"arm {arm} of fork {fork} did not complete; its rows are partial")
    lines.append("per measured row: prompt (promptSha256 + promptTokens), output (outputSha256), accepted-drafts histogram")
    lines.append(
        f"{'scenario':<24} {'fork':>4} {'iter':>4}  {'prompt':<10} {'output java':<13} {'output rust':<13} "
        f"{'output':<10} {'drafts':<7} {'status j/r':<17} verdict"
    )
    counts: dict[str, dict[str, int]] = {"prompt": {}, "output": {}, "drafts": {}}
    pairs = 0
    for fork in range(1, forks + 1):
        by_key = {arm: {(r["scenario"]["name"], r["iteration"]): r for r in rows[arm][fork]} for arm in ARMS}
        for key in sorted(set(by_key["java"]) | set(by_key["rust"])):
            java, rust = by_key["java"].get(key), by_key["rust"].get(key)
            prompt, output, drafts, verdict = compare(java, rust)
            pairs += 1
            for name, value in (("prompt", prompt), ("output", output), ("drafts", drafts)):
                counts[name][value] = counts[name].get(value, 0) + 1
            statuses = "/".join((r["status"] if r else "absent") for r in (java, rust))
            hashes = [((r["work"].get("outputSha256") or "-")[:10] if r and r["work"].get("generatedTokens") else "-") for r in (java, rust)]
            lines.append(
                f"{key[0]:<24} {fork:>4} {key[1]:>4}  {prompt:<10} {hashes[0]:<13} {hashes[1]:<13} {output:<10} "
                f"{drafts:<7} {statuses:<17} {verdict}"
            )
            if verdict != "ok" or any(r is None or r["status"] != "success" for r in (java, rust)):
                worst = 1
    lines.append(
        f"{pairs} pairs; prompts {counts['prompt']}; outputs {counts['output']} ('no output' is not an output pass); "
        f"drafts {counts['drafts']}"
    )

    lines.append("")
    lines.append("absolute figures over the successful measured rows; decode tok/s is first token to last token for both")
    lines.append(f"{'engine':<6} {'scenario':<24} {'rows':>4}  " + "  ".join(f"{name:>26}" for name, _ in METRICS) + "   (pooled medians)")
    scenarios = list(dict.fromkeys(r["scenario"]["name"] for arm in ARMS for fork in rows[arm].values() for r in fork))
    for arm in ARMS:
        every = [r for fork in rows[arm].values() for r in fork]
        for name in scenarios:
            ok = [r for r in every if r["scenario"]["name"] == name and r["status"] == "success"]
            figures = [show(median_of([v for v in map(get, ok) if v is not None])) for _, get in METRICS]
            lines.append(f"{arm:<6} {name:<24} {len(ok):>4}  " + "  ".join(f"{f:>26}" for f in figures))

    lines.append("")
    lines.append("per fork (median over the fork's successful measured rows), java | rust")
    lines.append(f"{'scenario':<24} {'fork':>4}  " + "  ".join(f"{name:>30}" for name, _ in METRICS))
    ahead = {name: [0, 0] for name in scenarios}  # per scenario: forks with a decode figure for both, forks rust >= java
    for name in scenarios:
        for fork in range(1, forks + 1):
            cells = []
            figures = {}
            for arm in ARMS:
                ok = [r for r in rows[arm][fork] if r["scenario"]["name"] == name and r["status"] == "success"]
                figures[arm] = [median_of([v for v in map(get, ok) if v is not None]) for _, get in METRICS]
            for index in range(len(METRICS)):
                cells.append(f"{show(figures['java'][index]):>13} | {show(figures['rust'][index]):<13}")
            lines.append(f"{name:<24} {fork:>4}  " + "  ".join(f"{c:>30}" for c in cells))
            java_decode, rust_decode = figures["java"][2], figures["rust"][2]
            if java_decode is not None and rust_decode is not None:
                ahead[name][0] += 1
                ahead[name][1] += rust_decode >= java_decode
    for name, (paired, wins) in ahead.items():
        if paired:
            lines.append(f"{name}: forks in which the rust decode figure is at least the java one: {wins} of {paired}")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (out_dir / "summary.txt").write_text(text)
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--java-config", type=Path, required=True)
    parser.add_argument("--rust-config", type=Path, required=True)
    parser.add_argument("--forks", type=int, default=1)
    parser.add_argument("--memory-max", default="24G", help="host memory cap of each arm (at most 32G)")
    parser.add_argument("--timeout", default="30m", help="time limit of each arm, as `timeout` reads it")
    parser.add_argument("--gpu-run", default=GPU_RUN, help="the GPU launcher: lock, memory cap, time limit")
    parser.add_argument("--out-dir", type=Path, help="default: benchmark-results/rust-gate-<time>")
    parser.add_argument("--speculation", choices=("config", "off"), default="config")
    parser.add_argument("--no-build", action="store_true", help="use the release binary and the installed Java script as they are")
    parser.add_argument("--dry-run", action="store_true", help="write the configurations, print the commands, run nothing")
    args = parser.parse_args()
    if args.forks < 1:
        parser.error("--forks must be at least 1")
    try:
        if memory_bytes(args.memory_max) > MAX_MEMORY_BYTES:
            parser.error("--memory-max must not be above 32G")
    except ValueError as error:
        parser.error(str(error))
    if not args.dry_run and shutil.which(args.gpu_run) is None:
        parser.error(f"the GPU launcher {args.gpu_run!r} is not found: give its path with --gpu-run")

    out_dir = (args.out_dir or REPO / "benchmark-results" / time.strftime("rust-gate-%Y%m%dT%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        launchers = {"java": JAVA_START_SCRIPT, "rust": Path("euhedral-bench")}
    else:
        launchers = {"java": java_start_script(build=not args.no_build), "rust": rust_binary(build=not args.no_build)}
    sources = {"java": args.java_config, "rust": args.rust_config}

    failed_arms: set[tuple[str, int]] = set()
    for fork in range(1, args.forks + 1):
        # Java first on odd forks, Rust first on even ones.
        for arm in (ARMS if fork % 2 == 1 else ARMS[::-1]):
            output = out_dir / f"{arm}-fork{fork}.jsonl"
            config = out_dir / f"{arm}-fork{fork}.json"
            config.write_text(json.dumps(derived_config(sources[arm], arm, output, args.speculation), indent=2) + "\n")
            command = scoped(arm_command(arm, config, fork, launchers), args.gpu_run, args.memory_max, args.timeout)
            if args.dry_run:
                print(shlex.join(command))
                continue
            output.unlink(missing_ok=True)  # a file that is there afterwards is this run's
            log = out_dir / f"{arm}-fork{fork}.log"
            print(f"fork {fork} {arm}: running (log {log})", flush=True)
            started = time.monotonic()
            with log.open("w") as handle:
                result = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)  # gpu-run bounds the wait and the run
            took = time.monotonic() - started
            # 1: some measured row failed or was ineligible (the rows are still written and compared below).
            print(f"fork {fork} {arm}: exit {result.returncode} after {took:.0f} s", flush=True)
            if result.returncode not in (0, 1):
                failed_arms.add((arm, fork))
                print(log.read_text()[-2000:], file=sys.stderr)
    if args.dry_run:
        return 0
    worst = report(out_dir, args.forks, failed_arms)
    return 2 if failed_arms else worst


if __name__ == "__main__":
    sys.exit(main())
