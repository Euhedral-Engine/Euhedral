# Benchmarking

The `benchmark` Gradle module is an end-to-end harness. It loads `InferenceEngine`, creates a fresh
`QwenGenerationSession` for every iteration, and runs the real tokenizer -> Euhedral lattice -> CUDA
path. It is not part of `core` or `api` and is not packaged in the API JAR. The `run` command writes
end-to-end engine measurements. The separate `q3` command writes explicitly labeled operator
microbenchmarks, not engine results.

## Workflow

Change one variable at a time and keep the JSON for every run you compare.

1. **Build.**

   ```bash
   ./gradlew build
   ```

2. **Correctness and integration tests.** Unit tests need no GPU. The CUDA suites need the
   artifacts and a GPU with room for the model.

   ```bash
   ./gradlew test
   ./gradlew :core:tokenizerReferenceTest -Peuhedral.qwen.tokenizer-dir=/mnt/shared/qwen38-quant/source/qwen
   ./gradlew cudaIntegrationTest -Peuhedral.qwen.artifact=/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl
   ```

3. **Check the GPU is free.** The harness refuses to load when free device memory is below the
   artifact size plus `gpuHeadroomMiB` (default 1024). It never stops other processes. Look first:

   ```bash
   nvidia-smi --query-gpu=memory.used,memory.total --format=csv
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
   ```

4. **Validate the configuration.** This checks paths, CPUs, artifact metadata, prompt sizes, and
   model context, then prints the plan without touching the GPU. An existing output path is rejected
   unless the config explicitly opts into append or overwrite. Validation does not test native CUDA
   kernel compilation or device-driver compatibility; require the integration tests to pass before
   treating any timing as a baseline.

   ```bash
   ./gradlew :benchmark:run --args="run benchmark/configs/baseline.json --validate-only"
   ```

5. **Baseline benchmark.** The default suite with production defaults: all available processors
   and 512-token prefill chunks.

   ```bash
   ./gradlew :benchmark:run --args="run benchmark/configs/baseline.json"
   ```

6. **Retain the JSON.** Rows are appended to the configured `output` as they complete (JSONL), and a
   summary is printed at the end. `benchmark-results/` is git-ignored; archive it deliberately
   together with the commit you measured (`git rev-parse HEAD`).

7. **External NVIDIA profiling.** Profile outside Gradle so the profiler sees one JVM. Install the
   start script, then give it the same native environment `:benchmark:run` sets. The paths shown are
   the pinned CUDA inputs Gradle downloads; substitute your toolkit's `lib64` and `include` if Gradle
   uses a local toolkit.

   ```bash
   ./gradlew :benchmark:installDist nativeBuild
   export LD_LIBRARY_PATH="$PWD/build/cuda-dev/linux-x64/runtime:$LD_LIBRARY_PATH"
   export EUHEDRAL_CUDA_INCLUDE_DIR="$PWD/build/cuda-dev/linux-x64/include"
   nsys profile --trace=cuda,nvtx,osrt -o benchmark-results/smoke \
     benchmark/build/install/euhedral-inference-benchmark/bin/euhedral-inference-benchmark \
     run benchmark/configs/smoke.json
   ncu --target-processes all -o benchmark-results/smoke-kernels \
     benchmark/build/install/euhedral-inference-benchmark/bin/euhedral-inference-benchmark \
     run benchmark/configs/smoke.json
   ```

   Profiled runs are slower; give each command a fresh result path and keep their JSON separate
   from timing baselines. The example `smoke.json` output must not already exist unless a separate
   profiling config opts into overwriting it.

8. **Change one variable**, for example the prefill chunk or the CPU selection:

   ```bash
   ./gradlew :benchmark:run --args="run benchmark/configs/prefill-chunk-256.json"
   ./gradlew :benchmark:run --args="run benchmark/configs/custom-cpus.json"
   ```

9. **Rerun and compare** against the retained baseline. Use the same scenarios, prompt seed,
   warmup, and iteration counts. Repeat each configuration in several JVMs (forks, below) before
   drawing a conclusion.

10. **Retain or revert.** Keep the change only if it improves the metric you targeted without
    regressing correctness or the other scenarios. Otherwise revert it and keep both JSON files as
    evidence.

## Configuration

`run CONFIG.json [--fork-id ID] [--validate-only]`. The config is JSON. Only `artifact`,
`tokenizer`, and `cudaLibrary` are required, and unknown fields are rejected. Relative paths resolve
against the working directory, which is the repository root under `./gradlew :benchmark:run`.

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `artifact`, `tokenizer`, `cudaLibrary` | required | Model artifact, tokenizer directory, native library. |
| `cpus` | `all` | `all`, `one-per-core`, `performance`, `performance-one-per-core`, or IDs/ranges (`"2-5,8"`). |
| `excludeCpus`, `excludeCores` | `[]` | Processor IDs or Euhedral core IDs removed from the selection. |
| `prefillChunks` | `[512]` | Prefill chunk sweep. One engine load per value; each runs every scenario. |
| `scenarios` | default suite | Strings: `prefill:P`, `first-token:P`, `prompt-to-n:P:N`, `decode:P:N`. |
| `warmup`, `iterations` | `1`, `3` | Warmup and measured iterations per scenario and chunk value. |
| `generation` | `{"mode":"greedy","seed":1}` | `sample` mode also takes `temperature`, `topK`, `topP` (0.7, 20, 0.8). |
| `promptSeed` | `20260925` | Seed for prompt material. |
| `output` | `benchmark-results/euhedral-<UTC>.jsonl` | A `.json` path writes one document; any other path writes JSONL. |
| `overwrite`, `append` | `false` | Required when `output` exists. `append` is JSONL only. |
| `gpuMemory` | `false` | Record device free/total memory before and after each iteration, outside timing. |
| `gpuHeadroomMiB` | `1024` | Free memory required beyond the artifact size before loading. |
| `q3DispatchMode` | `AUTO` | `SCALAR` retains the reference implementation; `DECODE` and `PREFILL` force independently callable kernels; `AUTO` selects by token-row count. |
| `q3SmallRowThreshold` | `8` | `AUTO` uses decode at or below this row count and tiled prefill above it. Zero forces all nonempty Q3 projections through prefill. Recorded with the dispatch mode in run snapshots. |
| `shutdownTimeoutSeconds` | `10` | Engine shutdown timeout. |

CPU selection uses the core `ProcessorTopology`. Unavailable IDs are rejected, not dropped. On hosts
where Euhedral cannot classify performance and efficiency cores, `performance` selects every
available processor.

The default suite is `prefill:32`, `prefill:256`, `prefill:1024`, `prefill:4096`,
`first-token:1024`, `prompt-to-n:1024:64`, and `decode:32:256`.

### Forks

A fork is a separately launched JVM. Only an externally supplied `--fork-id` names one. Iterations
inside one JVM share `runId` and `fork.pid` and are never independent forks. To collect forks, run
the same config several times with an appending output:

```bash
for fork in 1 2 3; do
  ./gradlew :benchmark:run --args="run benchmark/configs/forks.json --fork-id $fork"
done
```

The checked-in `forks.json` sets `"append": true` and writes to `benchmark-results/forks.jsonl`.
Record the exact artifact and native library alongside these fork results; iterations inside one
JVM are not independent replicates.

## Packed Q3 operator screens

The `q3 CONFIG.json [MATRIX ROWS]` command loads real packed artifact weights and compares scalar,
decode, and tiled-prefill kernels on identical deterministic BF16 activations. Matrix names are
`mixer-output`, `mlp-gate-up`, `mlp-down`, and `vocabulary`. Without a selector it sweeps 1, 2, 4,
8, 16, 32, 256, and 512 token rows; the vocabulary sweep stops at 32 because generation now
projects at most one row. Explicit selectors can request larger vocabulary cases.

Build the distribution and set the native environment as shown above, then run:

```bash
benchmark/build/install/euhedral-inference-benchmark/bin/euhedral-inference-benchmark \
  q3 benchmark/configs/smoke.json mlp-gate-up 256
```

Use a fresh non-`.json` output path: operator screens reject existing output even when the engine configuration
allows append or overwrite. The output is always JSONL, with a distinct
`euhedral-inference.q3-microbenchmark` schema. `warmup` and `iterations` control each forced path.
The screen uses synchronous native-call timing including launch, clears 128 MiB of device memory
before each sample outside timing, and records every sample plus BF16 absolute-error distributions
against scalar. Decode must match scalar bitwise; tiled prefill must stay within one BF16 step
or 0.001 absolute error near zero. A failed gate is recorded as `failed` and aborts the screen;
such timing samples are not eligible results. That cache-clear size targets the current GPU; it is not a portable guarantee of
complete cache eviction. Engine CPU selection, generation scenarios, and dispatch selection do not
control these deliberately isolated operator calls.

These screens are for rejecting poor candidates and measuring the row crossover. They do not
replace full-model numerical qualification or independent JVM forks of `run`.

## Packed Q4/Q5 operator screens

The `q45 CONFIG.json [MATRIX ROWS]` command uses real packed weights from the first GDN and
attention layers. Matrix names are `gdn-q4`, `gdn-q5`, `attention-q4`, and `attention-q5`.
Without a selector it sweeps 1, 2, 4, 8, 9, 12, 16, 32, 256, and 512 rows. Set
`EUHEDRAL_Q45_DISPATCH` to `SCALAR`, `DECODE`, `PREFILL`, or `PREFILL64` in a separate JVM for each
forced path. Each path writes its own JSONL results and per-case raw BF16 output files;
compare the latter against the forced scalar run before accepting timings. Output files must
not exist before a run. The screen clears 128 MiB of device memory outside each timed
synchronous native call. It does not substitute for full-model or end-to-end validation.
For example, after building the benchmark distribution and writing a configuration whose
`output` points to a new `screen-scalar.jsonl` file:

```sh
EUHEDRAL_Q45_DISPATCH=SCALAR \
  benchmark/build/install/euhedral-inference-benchmark/bin/euhedral-inference-benchmark \
  q45 screen-scalar.json gdn-q4 32
```

Use separate configuration/output paths for `DECODE`, `PREFILL`, and `PREFILL64`; archive and
compare all four sets of raw BF16 files, not just their timings.

Normal inference uses `AUTO` (also the default when the variable is unset): Q4 selects decode
through 9 rows and Q5 through 4 rows, then uses the 32-row prefill tile below 64 rows and the
64-row tile from 64 rows. Decode groups 1, 2, or 4 token rows per CTA; `PREFILL` forces the
32-row tile and `PREFILL64` the 64-row tile. The optional environment
variables `EUHEDRAL_Q4_DECODE_MAX_ROWS` and `EUHEDRAL_Q5_DECODE_MAX_ROWS` override those
thresholds independently. Archive the exact environment alongside any benchmark results;
these native experiment controls are not fields in the engine snapshot. `SCALAR` retains the
original packed-weight reference kernel, without changing the persistent Q4/Q5 layout.

## Prompts

Prompts are raw text, not chat-templated. Words are drawn from a fixed list
(`PromptMaterial.WORDS`) by `SplittableRandom(promptSeed)`. The longest word prefix whose
`encodeWithModelSpecialTokens` count, the count the session itself encodes, fits the target is kept,
then punctuation suffixes are tried to reach the target exactly. Each row records the target, the
actual encoded count (`work.promptTokens`), the generator, the seed, and the prompt's SHA-256.
`--validate-only` prints the actual counts. The session's own count is checked against the
prepared count; a mismatch fails the row.

## Metrics

All timestamps are `System.nanoTime()` on the generating thread. Each duration comes from its own
boundary pair; none is derived by subtracting one phase from an aggregate.

Boundaries come from an opt-in `GenerationTimingListener` on `QwenGenerationSession.generate`. Without
a listener the session records nothing and calls nothing extra.

- **generate entry / return:** taken immediately around `session.generate`. Session creation and
  close are excluded.
- **prompt encoded:** after tokenization.
- **prefill quantum:** from before the quantum's context is built to successful runtime execution,
  before any sampling.
- **first token selected:** after the first token is sampled from the last prefill quantum's logits.
- **decode quantum:** from before context build, to execution, to next-token selection when the
  quantum samples. The last quantum of a full-length call only commits the final token.

| Metric | Boundaries | Includes | Excludes |
| ------ | ---------- | -------- | -------- |
| `tokenization` | entry -> prompt encoded | tokenization | quanta |
| `prefill` | first prefill quantum start -> last prefill execution | all prefill quanta and host work between them | tokenization, sampling |
| `firstTokenSample` | last prefill execution -> first token selected | logits sampling | output callback |
| `timeToFirstToken` | entry -> first token selected | tokenization, prefill, sampling | output callback, text decoding |
| `decode` | first decode quantum start -> selection by the last sampling decode quantum | decode quanta and host work between them (output callback, incremental text decoding) | first token, final commit, decoder flush |
| `decodeQuantaSum` | sum over sampling decode quanta of start -> selection | quanta and sampling only | host work between quanta |
| `finalCommit` | start -> execution of the commit-only quantum | committing the last token | sampling |
| `timeToLastToken` | entry -> selection of the last returned token | everything before it | final commit, flush |
| `endToEnd` | entry -> return | everything, including final commit, callbacks, decoder flush | session create/close |

The output callback is a no-op; incremental text decoding still runs.

Throughput uses actual completed work:

- `prefillTokensPerSecond` = prompt tokens executed by prefill quanta / `prefill`.
- `decodeTokensPerSecond` = tokens sampled by decode quanta / `decode`. The first token is sampled
  by prefill and belongs to `timeToFirstToken`. The unsampled final commit is excluded. A sampled
  terminator counts, because it was sampled work.
- `endToEndOutputTokensPerSecond` = returned token IDs / `endToEnd`.

A rate is null when its count or duration is zero or missing. Prefill-only scenarios
(`maxNewTokens = 0`) have no first-token time and no decode or output rate. Failed rows have no
timings or rates.

### Status

- `success`: the scenario's work completed.
- `ineligible`: `prompt-to-n` or `decode` ended early, by a terminator or otherwise. The row reports
  the tokens actually produced, never the requested count, with a reason such as
  `eos_after_17_of_256_tokens`. Exclude these rows from steady-state comparisons.
- `failed`: an exception, incomplete prefill, or prompt-count mismatch. Timings and rates are null
  and `statusReason` holds the cause.

The summary reports medians over measured `success` rows only. Warmup rows are written, marked
`"warmup": true`, and never summarized. If any measured row is failed or ineligible, the run exits
nonzero; keep its JSONL for diagnosis rather than treating it as a qualified baseline.

## Result schema

Each row has `"schema": "euhedral-inference.benchmark-result"` and `"schemaVersion": 1`. JSONL holds
one row per line. A `.json` output holds
`{"schema": "euhedral-inference.benchmark-results", "schemaVersion": 1, "results": [...]}`. The shape
below is illustrative; its values are not a measurement.

```json
{
  "schema": "euhedral-inference.benchmark-result", "schemaVersion": 1,
  "implementation": "euhedral-inference",
  "provenance": {"kind": "measured", "tool": "euhedral-inference-benchmark", "commandLine": "run benchmark/configs/baseline.json",
                 "source": null, "sourceSha256": null, "note": null},
  "runId": "<uuid per JVM>",
  "fork": {"id": null, "source": "unspecified", "pid": 1234, "jvmStartedAt": "<instant>"},
  "recordedAt": "<instant>",
  "scenario": {"name": "decode-32-256", "kind": "decode", "targetPromptTokens": 32, "requestedNewTokens": 256,
               "promptGenerator": "euhedral-words-v1", "promptSeed": 20260925, "promptSha256": "<hex>"},
  "warmup": false, "iteration": 0,
  "status": "success", "statusReason": null,
  "work": {"promptTokens": 32, "prefillQuanta": 1, "prefillTokens": 32, "generatedTokens": 256,
           "decodeSampledTokens": 255, "finalCommitQuanta": 1, "eosObserved": false},
  "timings": {"tokenization": 0, "prefill": 0, "firstTokenSample": 0, "timeToFirstToken": 0, "decode": 0,
              "decodeQuantaSum": 0, "finalCommit": 0, "timeToLastToken": 0, "endToEnd": 0},
  "throughput": {"prefillTokensPerSecond": 0.0, "decodeTokensPerSecond": 0.0, "endToEndOutputTokensPerSecond": 0.0},
  "engine": {"schemaVersion": 2, "tuning": {"workerProcessorIds": [0, 1], "prefillChunkTokens": 512,
             "q3DispatchMode": "AUTO", "q3SmallRowThreshold": 8},
             "workerCoreIds": [0], "model": {}, "generation": {}, "runtime": {}},
  "gpuMemory": {"beforeFreeBytes": 0, "afterFreeBytes": 0, "totalBytes": 0}
}
```

`engine` is the engine's `InferenceRunSnapshot`: tuning, worker cores, model identity and dimensions,
generation settings, and Java/Euhedral/native identity. Values the runtime does not expose, such as
the CUDA runtime version, are `"unavailable"`. Snapshot version 2 dropped `tuning.gpuExecutionMode`:
every run submits stream-ordered work asynchronously. Version-1 rows remain readable and ignore that
field. `gpuMemory` is null unless `gpuMemory` is enabled; it is device-wide, so it includes other
processes.

## Importing external results

External results, for example from NInfer, can be stored beside these rows if they already use this
schema and name themselves in `implementation`. This does not launch NInfer.

```bash
./gradlew :benchmark:run --args="import --input /path/to/ninfer.jsonl --implementation ninfer \
  --output benchmark-results/baseline.jsonl --append --note 'NInfer abc123, same host'"
```

Import validates every row:

- schema and version must match, and unknown fields are rejected;
- a row whose `implementation` differs from `--implementation` is rejected;
- failed rows may carry no rates;
- every reported rate must equal this document's definition over the row's own work and timings.

Imported rows get `provenance.kind = "imported"`, the absolute source path, the source file's
SHA-256, and the note. The original tool and command line are kept.
