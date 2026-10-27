# Rust engine

The Rust engine runs Qwen3.8 27B from an `.edrl` artifact on the GPU, on the RustEE lattice. It lives in the
[`rust/`](../../rust/README.md) workspace and loads the same CUDA product (`libeuhedral_cuda`) as the rest of the
repository. It generates greedily, with or without MTP speculative decoding from the artifact's draft layer, one
sequence at a time. Every weight stays on the device; attention-cache pages past
`context.kv-cache-on-gpu-tokens` move to pinned host memory when the context does not fit. There is no HTTP server yet:
the benchmark binary `euhedral-bench` is the way to run it.

## Build

The CUDA product is built by Gradle, the Rust crates by cargo:

```text
mise run rust-native                                        # ./gradlew nativeBuildLinuxX64: build/native/linux-x64/lib/libeuhedral_cuda.so
cd rust
cargo build --release -p euhedral-inference-bench           # target/release/euhedral-bench
mise run rust-check                                         # layer check, fmt, clippy -D warnings, CPU tests
```

The CPU tests run on Linux and Windows in CI. The GPU tests are `#[ignore]`d and run locally, one binary at a time, in a
memory-capped scope, for example:

```text
systemd-run --user --scope -p MemoryMax=24G cargo test -p euhedral-inference-engine --test generation_gpu -- --ignored --test-threads=1
```

They read the artifact, the CUDA library and the Java references from `euhedral.*` keys in
`~/.gradle/gradle.properties` (or the matching `EUHEDRAL_TEST_*` environment variables) and skip with a message when one
is absent.

## Run

```text
euhedral-bench run CONFIG.json [--fork-id ID] [--validate-only]
euhedral-bench dump --artifact FILE --prefix N --steps N --out FILE [--cuda-library FILE]
```

`run` reads a configuration file of the Java benchmark (`benchmark/configs`, the same field names and defaults) and
writes rows in its result schema (`euhedral-inference.benchmark-result` version 1, `implementation` `rust`).
`--validate-only` builds the prompts and checks the configuration without touching the GPU. `dump` writes the
teacher-forced logits report used by the exactness checks.

The engine itself is configured by the YAML file of [`config/euhedral.example.yaml`](../../config/euhedral.example.yaml),
named by the benchmark configuration's `engineConfig` field. Every key can be overridden by `EUHEDRAL_` and the
upper-cased key path (`.` and `-` become `_`). The benchmark configuration's `artifact`, `maxContextTokens` and
`kvDeviceMinimumTokens` take precedence over the file. The prefix cache is not implemented: a configuration with
`prefixCacheBytes` above 0 is refused, as are `cpus` other than `all` and sampled generation.

## Architecture

| Crate | Contents |
| --- | --- |
| `settings` | The YAML file and its environment overrides; the only reader of the environment |
| `edrl` | Parsers of the artifact: header, metadata, tensor table, tokenizer section |
| `cuda` | `libloading` bindings to `libeuhedral_cuda`, device and pinned memory, lanes (streams with a host-mapped completion word), submissions and graphs |
| `host-gemv` | CPU GEMV kernels for host-resident layers, chosen at run time by CPU features (the planner keeps every weight on the device for now) |
| `tokenizer` | HF `tokenizers` over the artifact's packaged tokenizer |
| `engine` | Residency planner, weight loader, execution plan and stages, sequence state, quanta, CUDA graph cache, generation and MTP |
| `bench` | `euhedral-bench` and the benchmark prompt corpora |

**Execution model.** The engine owns no threads. One RustEE lattice runs everything as frames, on all CPUs: the
startup (artifact reads through `AsyncIo`, the plan, the uploads, the module warmup), the GPU submissions and the
generation. The main thread starts the runtime and waits for its latch; nothing else blocks. GPU completion is read
from each lane's host-mapped completion word, so no driver callback runs.

**A request.** The generation source owns the sequence: its cache frontiers, its recorded graphs, the quanta in flight
and the request's output. A worker that finds work for it publishes a frame that takes the owner, concludes the quanta
that retired, admits the next ones and hands the owner back. Greedy decode keeps two quanta in flight: the second runs
behind the first on the device and reads its token from the first one's device-side selection, so the host never sits
between two steps. The first token is selected by a device argmax at the end of the last prompt chunk; each quantum
returns an 8-byte key per row and no logits.

**Quanta and graphs.** A quantum runs its view's stage graph (prefill chunk, decode, verification, draft, catch-up),
each stage on the lane the plan placed it on and ordered behind its predecessors by markers. On the second sight of a
capture key the engine records the same launches into a CUDA graph on a lane of its own, without submitting anything
twice; from then on a quantum with that key is one graph launch. Decode, verification and drafts of up to 8 rows are
captured; prefill and quanta that allocate cache pages are not.

**MTP speculation.** With `speculation: artifact` each step is one device chain: the MTP catch-up over the rows the last
verification committed, the recursive drafts, and the verification of the current token and the drafts. Draft tokens
pass between the quanta on the device; the host reads the drafts and the verifier's keys once, when the verification
retires, accepts the drafts the verifier confirms, commits that many rows and admits the next step. The verification
computes each row bit for bit as one-row decode would, so the output equals greedy decode. The draft depth comes from
the artifact (4 for the RCO artifact).

## Exactness

The engine makes every launch decision the Java engine makes, and the checks run from the lowest level up:

1. **Launch shapes.** `formats_gpu` compares the linear launches of decode, verification, batch and prefill shapes
   (kernel, grid, split, row-exact route) with the Java shape goldens, and `formats_oracle_gpu` runs each format on
   the library's scalar reference.
2. **Teacher-forced logits.** `generation_gpu` forces 192 tokens after a 1100-token prefix and compares every BF16
   logits row with the Java report, byte for byte.
3. **Greedy tokens.** `generation_gpu` compares 128 greedy tokens after a chat prompt with the Java engine's.
4. **Speculation.** `mtp_gpu` checks that MTP output equals the engine's greedy output on chat prompts, that the KV and
   GDN state after a run equal greedy decode's, and that a captured run equals an uncaptured one.
5. **The benchmark.** On the scenarios below, `outputSha256` (the SHA-256 of the comma-joined token ids) and
   `promptSha256` are equal per scenario and iteration between the Rust and the Java benchmark and between the
   speculative and the greedy arm, and the accepted-drafts histogram of each speculative row equals the Java
   engine's. `tools/rust_gate.py` runs both benchmarks and checks this.

## Performance

Measured on a 16 GB RTX 5070 Ti (Linux, also driving the desktop display) with an i9-14900K, the RCO artifact
(`qwen3_8_27b_nvfp4_trellis_rco.edrl`), greedy output, the prefix cache off and 65,536 cache tokens on the device.
Each fork is a fresh process with two warmup iterations per scenario (the first quantum of a process compiles the
kernels it uses). 4K and 32K: chat corpus, 128 new tokens, 4 prompts, 3 forks. 128K: words corpus, 64 new tokens,
`maxContextTokens` 131072, 2 iterations of the one 128K prompt, 1 fork. Figures are medians over the measured rows.

| Prompt tokens | Speculation | Decode tok/s | Prefill tok/s | Time to first token |
| --- | --- | ---: | ---: | ---: |
| 4K (`prompt-to-n:4096:128`) | MTP, 4 drafts | 144.7 | 2,598 | 1.53 s |
| 32K (`prompt-to-n:32000:128`) | MTP, 4 drafts | 128.3 | 2,103 | 15.2 s |
| 128K (`prompt-to-n:128000:64`) | MTP, 4 drafts | 99.2 | 1,251 | 102.4 s |
| 4K | off | 60.1 | 2,600 | 1.53 s |
| 32K | off | 57.5 | 2,146 | 14.9 s |
| 128K | off | 51.6 | 1,299 | 98.7 s |

Decode tok/s counts tokens 2 to N over the window from the first token found to the last token found; the first token
belongs to prefill. With speculation the prefill window includes the MTP catch-up of each prompt chunk. The MTP rate
depends on the prompt: per prompt it ranged from 121 to 171 tok/s at 4K and from 113 to 141 tok/s at 32K. Fork-to-fork
spread of the medians was within 0.2 tok/s.

The run used:

```text
tools/rust_gate.py --java-config CONFIG.json --rust-config CONFIG.json --forks 3 --memory-max 24G \
    --gpu-run gpu-run --no-build
```

**Rejected: poll mode.** `host.max-batch-size: 2` makes the lattice workers poll their caches one frame at a time
instead of draining them. With MTP it measured 143.5 tok/s at 4K and 128.3 tok/s at 32K (one fork), with the same
outputs; it gave decode nothing, so the default (`auto`, drain) stays.
