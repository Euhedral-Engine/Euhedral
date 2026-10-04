# Benchmarking

The `benchmark` Gradle module is an end-to-end harness. It loads `InferenceEngine`, creates a fresh
`QwenGenerationSession` for every iteration, and runs the real tokenizer -> Euhedral lattice -> CUDA
path. It is not part of `core` or `api` and is not packaged in the API JAR. The `run` command writes
end-to-end engine measurements. The engine derives its execution policy (kernels, speculative
depth, host-backed weight residency) from the artifact, and prefill runs in 512-token chunks, so
the harness measures what a user of the engine gets.

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
   ./gradlew cudaIntegrationTest -Peuhedral.qwen.artifact=/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl
   ```

3. **Check the GPU is free.** The engine refuses to load, and the run exits with code 3 without
   writing results, when the model and a KV cache for `maxContextTokens` do not fit in free device
   memory. The harness never stops other processes. Look first:

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

5. **Baseline benchmark.** The default suite on all available processors with the default
   context capacity.

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

8. **Change one variable**, for example the CPU selection:

   ```bash
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
| `maxContextTokens` | `32768` | Longest sequence (prompt plus generation) the engine keeps device memory for; scenarios that need more positions are rejected. Recorded in the run snapshot. |
| `prefixCacheBytes` | `0` | Pinned host memory for the prefix cache ([PREFIX_CACHE.md](PREFIX_CACHE.md)); 0 runs without one. Iterations of a scenario repeat its prompt, so with a cache every iteration after the first restores stored state: use it to measure the restore, never in a comparison of the engine's speed. Recorded in the run snapshot. |
| `cpus` | `all` | `all`, `one-per-core`, `performance`, `performance-one-per-core`, or IDs/ranges (`"2-5,8"`). |
| `excludeCpus`, `excludeCores` | `[]` | Processor IDs or Euhedral core IDs removed from the selection. |
| `scenarios` | default suite | Strings: `prefill:P`, `first-token:P`, `prompt-to-n:P:N`, `decode:P:N`. |
| `warmup`, `iterations` | `1`, `3` | Warmup and measured iterations per scenario. |
| `generation` | `{"mode":"greedy","seed":1}` | `sample` mode also takes `temperature`, `topK`, `topP` (0.7, 20, 0.8). |
| `promptSeed` | `20260925` | Seed for prompt material. |
| `output` | `benchmark-results/euhedral-<UTC>.jsonl` | A `.json` path writes one document; any other path writes JSONL. |
| `overwrite`, `append` | `false` | Required when `output` exists. `append` is JSONL only. |
| `gpuMemory` | `false` | Record device free/total memory before and after each iteration, outside timing. |
| `promptCorpus` | `words` | `words` draws seeded random words; `chat` uses chat-templated prompts (thinking disabled) built from a fixed corpus; iteration i runs prompt i mod corpus size. |
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
| `timeToFirstToken` | entry -> first token selected | tokenization, prefix restore, prefill, sampling | output callback, text decoding |
| `decode` | first decode quantum start -> selection by the last sampling decode quantum | decode quanta and host work between them (output callback, incremental text decoding) | first token, final commit, decoder flush |
| `decodeQuantaSum` | sum over sampling decode quanta of start -> selection | quanta and sampling only | host work between quanta |
| `finalCommit` | start -> execution of the commit-only quantum | committing the last token | sampling |
| `timeToLastToken` | entry -> selection of the last returned token | everything before it | final commit, flush |
| `endToEnd` | entry -> return | everything, including final commit, callbacks, decoder flush | session create/close |

The output callback is a no-op; incremental text decoding still runs.

Throughput uses actual completed work:

- `prefillTokensPerSecond` = prompt tokens executed by prefill quanta / `prefill`. With a prefix cache, a restored
  prefix is not executed by prefill: `work.prefixRestoredTokens` counts it, `timings.prefixRestore` is the time the
  restore took, and `prefillTokens + prefixRestoredTokens` is the prompt.
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
           "decodeSampledTokens": 255, "finalCommitQuanta": 1, "eosObserved": false,
           "prefixRestoredTokens": null},
  "timings": {"tokenization": 0, "prefill": 0, "firstTokenSample": 0, "timeToFirstToken": 0, "decode": 0,
              "decodeQuantaSum": 0, "finalCommit": 0, "timeToLastToken": 0, "endToEnd": 0, "prefixRestore": null},
  "throughput": {"prefillTokensPerSecond": 0.0, "decodeTokensPerSecond": 0.0, "endToEndOutputTokensPerSecond": 0.0},
  "engine": {"schemaVersion": 4, "configuration": {"workerProcessorIds": [0, 1], "maxContextTokens": 32768,
             "artifact": "qwen3_8_27b_q3.edrl", "speculativeDepth": 2, "prefixCacheBytes": 0},
             "workerCoreIds": [0], "model": {}, "generation": {}, "runtime": {}},
  "gpuMemory": {"beforeFreeBytes": 0, "afterFreeBytes": 0, "totalBytes": 0}
}
```

`engine` is the engine's `InferenceRunSnapshot`: configuration (worker processors, context
capacity, artifact name, and the speculative depth the engine selected), worker cores, model
identity and dimensions, generation settings, and Java/Euhedral/native identity. Values the runtime
does not expose, such as the CUDA runtime version, are `"unavailable"`. `gpuMemory` is null unless
`gpuMemory` is enabled; it is device-wide, so it includes other processes.

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
