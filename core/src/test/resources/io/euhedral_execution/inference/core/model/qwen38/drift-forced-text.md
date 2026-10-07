# Euhedral-Inference

An OpenAI-compatible inference server for **Qwen3.8-27B** (text) on a single NVIDIA Blackwell GPU, written in Java 25
with a small native CUDA layer. It runs the whole model on one consumer card (developed on an RTX 5070 Ti with 16 GB),
decodes with MTP speculative decoding, and serves 32K to 64K tokens of context from that 16 GB.

You choose a weight file; the engine chooses everything else. There are no kernel, precision, or scheduling options.

## The four artifacts

An artifact is one `.edrl` file holding the quantized text model, its MTP draft layer and draft head. You make two choices:

- **Quantization**: **Q3** (3-bit weights; smallest, fastest) or **NVFP4** (4-bit floating point; higher fidelity).
- **Representation**: **uncompressed** (reads fastest) or **compressed** (smaller on disk and in memory, a little more work per token).

| Artifact | Quantization | Representation | File size | Compression |
|---|---|---|---|---|
| `q3` | Q3 | uncompressed | 11.72 GiB (12.58 GB) | |
| `q3-compressed` | Q3 | compressed | 10.16 GiB (10.91 GB) | lossless: the same values as `q3`, bit for bit |
| `nvfp4` | NVFP4 | uncompressed | 14.52 GiB (15.59 GB) | |
| `nvfp4-compressed` | NVFP4 | compressed | 13.77 GiB (14.79 GB) | 4.25 bits per weight (block scales drawn from a 16-entry table per tensor); not lossless, see [quality](#quality) |

The engine reads what an artifact is from the file and picks the fastest validated execution for it: kernels by row count,
FP8 and FP4 tensor-core prefill, the attention implementation, the speculative-decoding depth and which weights stay in
host memory. All of that is automatic.

## Performance

RTX 5070 Ti (16 GB, 70 SMs) in a desktop that also uses the GPU for its display, Linux, one run per row after one warmup. Greedy
decoding with MTP speculative decoding (two drafted tokens for Q3, three for NVFP4); decode `tok/s` is output tokens per second
after the first token. Prefill and time to first token (TTFT) include the whole prompt.

| Artifact | Context (prompt tokens) | Prefill tok/s | TTFT | Decode tok/s (MTP) |
|---|---|---|---|---|
| `q3` | 4K (3,964) | 1,974 | 2.10 s | 116.9 |
| | 16K (15,930) | 1,801 | 9.14 s | 118.7 |
| | 32K (31,906) | 1,636 | 19.9 s | 110.1 |
| | 64K (59,111) | 1,401 | 43.7 s | 98.0 |
| `q3-compressed` | 4K | 1,837 | 2.28 s | 95.0 |
| | 16K | 1,693 | 9.63 s | 97.0 |
| | 32K | 1,543 | 21.1 s | 91.1 |
| | 64K | 1,332 | 45.7 s | 82.0 |
| | 128K (128,000) | 987 | 135 s | 60.7 |
| `nvfp4` | 4K | 3,874 | 1.13 s | 70.7 |
| | 16K | 3,254 | 5.19 s | 61.2 |
| | 32K | 2,762 | 12.0 s | 63.8 |
| | 64K | 2,145 | 29.0 s | 57.7 |
| `nvfp4-compressed` | 4K | 3,815 | 1.14 s | 97.6 |
| | 16K | 3,219 | 5.23 s | 92.6 |
| | 32K | 2,728 | 12.1 s | 82.6 |
| | 64K | 2,131 | 29.2 s | 64.0 |

The 4K, 16K and 32K rows run with the default 32,768-token context; the 64K and 128K rows set `max-context-tokens` to 65,536
and 131,072. Each prompt is followed by 128 generated tokens (64 at 128K). The 128K prompt is generated text; the others are
chat prompts.

| Artifact | File | Device memory in use at 32K / 64K | Host-backed weights at 32K / 64K | Longest context run |
|---|---|---|---|---|
| `q3` | 11.72 GiB | 12.8 / 13.3 GiB | none | 64K |
| `q3-compressed` | 10.16 GiB | 11.3 / 11.8 GiB (14.0 GiB at 128K) | none | 128K |
| `nvfp4` | 14.52 GiB | 13.9 / 13.9 GiB | 1.22 / 1.77 GiB | 64K |
| `nvfp4-compressed` | 13.77 GiB | 13.9 / 13.9 GiB | 0.46 / 1.03 GiB | 64K |

Device memory is what the engine allocated at its peak during the run: the resident weights, the KV cache of the context, the
sequence state, and the workspaces. The CUDA context and kernel modules come on top. An uncompressed NVFP4 model is larger than
the card's memory budget at these contexts, so the engine holds part of it in pinned host memory and streams it in for every token.

Residency is described in [docs/NVFP4_RESIDENCY.md](docs/NVFP4_RESIDENCY.md) and [docs/COMPRESSED_Q3.md](docs/COMPRESSED_Q3.md).

## Quality

Error is measured four separate ways because they answer different questions. All use the 1279 teacher-forced tokens
that follow a 1281-token prefix of one fixed document, and the BF16 checkpoint run in llama.cpp on the CPU as the reference.

**Model and quantization error** — the artifact against the BF16 checkpoint (reference: BF16 in llama.cpp on the CPU):

| Artifact | Mean NLL | Perplexity | NLL vs BF16 (± s.e.) | KL(BF16 ‖ artifact) | Top-1 agreement |
|---|---|---|---|---|---|
| BF16 reference | 2.128 | 8.39 | | | |
| `q3`, `q3-compressed` | 2.403 | 11.05 | +0.275 ± 0.025 | 0.400 | 70.5% |
| `nvfp4` | 2.178 | 8.83 | +0.051 ± 0.011 | 0.163 | 86.4% |
| `nvfp4-compressed` | 2.183 | 8.87 | +0.056 ± 0.011 | 0.161 | 86.9% |

**Compression-induced error** — the compressed artifact against its uncompressed one:

| Pair | Result |
|---|---|
| `q3-compressed` against `q3` | lossless: the 1279 logit vectors are bitwise equal, as is every drift measurement below |
| `nvfp4-compressed` against `nvfp4` | NLL +0.005 ± 0.011, KL 0.062, top-1 agreement 87.0% |

**Relaxed-execution drift** — the production kernels against the exact scalar kernels on the same weights, 1024 teacher-forced
decode steps after a 512-token prefill:

| Artifact | Hidden-state relative error | Mean KL | Top-1 agreement |
|---|---|---|---|
| `q3`, `q3-compressed` | 7.6% | 4.7e-3 | 97.9% |
| `nvfp4` | 7.0% | 3.9e-3 | 97.3% |
| `nvfp4-compressed` | 7.4% | 3.4e-3 | 97.8% |

The error settles within about 150 positions and does not grow after that.

**Exact speculative verifier** — verification of drafts is row-exact: the logits and the KV and GDN state after a
verification are bit for bit those of one-row decoding, and speculative generation produces exactly the tokens of ordinary
greedy decoding. This holds for all four artifacts (`SpeculativeVerifyCudaIntegrationTest`,
`SpeculativeDecodeCudaIntegrationTest`).

How to reproduce each measurement is in [docs/QUALITY.md](docs/QUALITY.md).

## Choosing an artifact

- **Fastest, and the one that fits everywhere:** `q3`: 117 tok/s at 4K and 98 tok/s at 64K, with no host-backed weights. Its
  error against BF16 is the largest of the four (perplexity 11.05 against 8.39 on the quality text).
- **More room instead of speed:** `q3-compressed` returns exactly the `q3` outputs from a file 1.56 GiB smaller, which is what lets it
  hold a 128K context on this card. Its kernels decode the compressed weights as they read them, which costs about a fifth of `q3`'s
  decode speed (95 tok/s at 4K).
- **Highest fidelity:** the NVFP4 artifacts (perplexity 8.83 and 8.87). On a 16 GB card `nvfp4` keeps 1.2 to 1.8 GiB of weights in host memory at 32K to 64K of
  context and streams them in for every token; `nvfp4-compressed` needs less than half of that, which makes it faster than `nvfp4` up to 32K
  of context at a quality cost within the noise of the measurement (table above).
- A card with more memory keeps more weights on the device and runs faster; the engine measures free memory at start and decides.

## Quick start

You need a Blackwell GPU (RTX 50-series or newer, compute capability 12.x) with a current NVIDIA driver, Java 25, and a
Hugging Face checkpoint of Qwen3.8-27B. The first build downloads the pinned CUDA runtime pieces and Zig.

**1. Convert the checkpoint to an artifact** (a minute or two for Q3 and NVFP4 on a GPU, longer for the compressed forms):

```bash
python3 tools/convert_checkpoint.py --model /path/to/Qwen3.8-27B --quantization q3 \
    --ranking /path/to/token-ranking.i64 --out /models/qwen-q3.edrl
# or: --quantization q3 --compressed | --quantization nvfp4 | --quantization nvfp4 --compressed
```

The token ranking chooses the draft head's vocabulary shortlist (`--ranking`), or reuse one from an existing artifact
(`--draft-ids-from /models/qwen-q3.edrl`). See [tools/README.md](tools/README.md).

**2. Build the server:**

```bash
./gradlew :api:bootJar
```

**3. Run it:**

```bash
CUDA=build/cuda-dev/linux-x64
EUHEDRAL_CUDA_INCLUDE_DIR=$PWD/$CUDA/include LD_LIBRARY_PATH=$PWD/$CUDA/runtime \
EUHEDRAL_INFERENCE_ARTIFACT_PATH=/models/qwen-q3.edrl \
EUHEDRAL_INFERENCE_TOKENIZER_DIRECTORY=/path/to/Qwen3.8-27B \
EUHEDRAL_INFERENCE_CUDA_LIBRARY_PATH=$PWD/build/native/linux-x64/lib/libeuhedral_cuda.so \
EUHEDRAL_INFERENCE_WORKER_CPUS=2-5 \
EUHEDRAL_INFERENCE_MODEL_ID=qwen \
java --enable-native-access=ALL-UNNAMED -jar api/build/libs/euhedral-inference-api.jar
```

The server listens on port 1738.

**4. Send a request:**

```bash
curl http://localhost:1738/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen",
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
  "max_tokens": 128
}'
```

Any OpenAI client works against `http://localhost:1738/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1738/v1", api_key="unused")
reply = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Explain speculative decoding in two sentences."}],
    stream=True,
)
for chunk in reply:
    print(chunk.choices[0].delta.content or "", end="")
```

The API serves `/v1/models` and `/v1/chat/completions` (JSON or SSE streaming) with OpenAI-style function calling
([docs/TOOL_CALLING.md](docs/TOOL_CALLING.md)). `/health` reports whether the engine has loaded.

## Docker

```bash
docker build -t euhedral-inference:local .
docker run --rm --gpus all -p 1738:1738 --stop-timeout 45 \
  --mount type=bind,src=/absolute/path/model.edrl,dst=/models/model.edrl,readonly \
  --mount type=bind,src=/absolute/path/tokenizer,dst=/tokenizer,readonly \
  -e EUHEDRAL_INFERENCE_WORKER_CPUS=2-5 \
  -e EUHEDRAL_INFERENCE_MODEL_ID=qwen \
  euhedral-inference:local
```

The model and tokenizer are not in the image. It carries the CUDA runtime and NVRTC userspace libraries, the native
library and the kernel sources, but not the NVIDIA kernel driver: `--gpus all` needs the NVIDIA Container Toolkit and a
compatible host driver. Hosts without the toolkit can pass the device nodes and the driver libraries explicitly:

```text
--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-modeset \
--mount type=bind,src=/usr/lib/x86_64-linux-gnu/libcuda.so.1,dst=/opt/euhedral/lib/libcuda.so.1,readonly \
--mount type=bind,src=/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1,dst=/opt/euhedral/lib/libnvidia-ptxjitcompiler.so.1,readonly
```

`mise run deploy` pulls `origin/main`, builds an image from exactly that commit, switches the `euhedral-inference-serve`
container on localhost port 18080 after checking Docker health, `/health`, `/v1/models` and a one-token completion, and
restores the previous container if the candidate fails. It serves the `nvfp4-compressed` artifact by default; the artifact, model ID and
port are overridable through `EUHEDRAL_DEPLOY_*` variables (`scripts/deploy-main.py`).

## Configuration

The server is configured by environment variables or properties: the five above are required, and
`EUHEDRAL_INFERENCE_MAX_CONTEXT_TOKENS` (32768) and the prefix cache's size are the ones most deployments change.
[docs/OPERATIONS.md](docs/OPERATIONS.md) lists every setting and covers startup checks, health, shutdown and metrics.

One generation runs at a time; the engine sizes device memory for one sequence. The prefix cache is the only state kept between requests; it is host memory, so it takes nothing from the context length or the device. Requests are served entirely on the
engine's worker threads: the container thread hands a request over and returns, and rendering, tokenization, every
quantum and every write to the client run as lattice work (docs/FRAME_MODEL.md, "Host work on the workers").

## Building and testing

Requirements: Java 25, Gradle 9.6.1 (the wrapper is committed), Zig 0.16.0 for the native library, and Rust 1.95.0 with
cargo-zigbuild 0.23.4 for llguidance (constrained decoding, built from its pinned crates.io release); CUDA 13.1+ headers
and libraries are resolved automatically. `mise.toml` records the versions and the common tasks (`mise install`).

```bash
./gradlew build            # compiles, formats, runs the CPU tests, builds the Linux and Windows native products
./gradlew test             # CPU tests
python3 -m unittest discover -s native/tests -p 'test_*.py'   # kernel tests on the GPU (needs NumPy)
./gradlew :core:cudaIntegrationTest :api:cudaIntegrationTest  # model tests on the GPU
```

Stop any serving container before the GPU suites. The native layer is described in [native/README.md](native/README.md),
the benchmark harness in [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

## More

- [docs/FRAME_MODEL.md](docs/FRAME_MODEL.md): how a forward pass is scheduled across device lanes.
- [docs/PREFIX_CACHE.md](docs/PREFIX_CACHE.md): the state kept between requests, its checkpoints and measured restore times.
- [docs/MTP_SPECULATIVE.md](docs/MTP_SPECULATIVE.md), [docs/MTP_CONTRACT.md](docs/MTP_CONTRACT.md), [docs/MTP_VERIFIER.md](docs/MTP_VERIFIER.md): speculative decoding and its exactness contract.
- [docs/PREFILL_MX.md](docs/PREFILL_MX.md), [docs/NVFP4_NATIVE.md](docs/NVFP4_NATIVE.md), [docs/ATTENTION_DECODE.md](docs/ATTENTION_DECODE.md): the prefill, FP4 and attention kernels.
- [docs/COMPACT_Q3_REFERENCE.md](docs/COMPACT_Q3_REFERENCE.md), [docs/COMPRESSED_Q3.md](docs/COMPRESSED_Q3.md), [docs/NVFP4_COMPRESSED.md](docs/NVFP4_COMPRESSED.md), [docs/NVFP4_RESIDENCY.md](docs/NVFP4_RESIDENCY.md): the artifact formats and memory budgets.
- [docs/nvidia/](docs/nvidia/README.md): measurements of this GPU's hardware.

---

Licensed under the Apache License, Version 2.0 (see `LICENSE`).


# Qwen execution model

A Qwen execution plan defines a static DAG of execution stages. A reusable runtime instance represents
those stages as independent Euhedral frames. Admission exposes only root frames through a
Qwen-specific `LatticeSource`. Successful stages create successor readiness and publish ready
successors back to that source. CUDA stream order carries ordinary device dependencies; actual device
completion is reserved for state-publication and ownership boundaries.

Euhedral has no central scheduler or authority; scheduling is emergent. Workers take available frames
for themselves, or a frame is routed through the lattice by its hash, so every assignment is either
deterministic (hashing) or first come, first served. Euhedral-Inference defines the stages and owns
model and sequence state. Nothing in the Qwen runtime executes a stage body inline, scans for ready
work, or walks the graph after admission.

## Immutable plan and reusable graph

`QwenExecutionPlan` is the schema. It describes the stages (instructions), their immutable weight
bindings, their input and output buffers, the dependency edges, the plan views (decode and the prefill
region views), and static metadata. It is never the live scheduler. `stageTopology()` exposes the DAG
as a `StageTopology`: stages numbered in topological order, each edge tagged with the boundary it waits
for.

`StageGraph` is one reusable runtime instance of a plan view. It is built once and then rebound to one
quantum at a time:

- one `StageFrame` per stage, created with its immutable instruction and weight binding
  (`QwenStageFrame.create` chooses `EmbeddingFrame`, `WeightTransferFrame`, `RmsNormFrame`, `LinearFrame`, or
  `QwenGpuOperationFrame`);
- successor references, wired at construction;
- one frame per device-completion edge and one retirement frame;
- a persistent CUDA stream and its own `QwenExecutionSource`;
- its workspace storage (`QwenWorkspaceStorage`), which the runtime's pool keeps with the graph.

Per quantum, the graph only resets fan-in counters and per-stage flags and binds the
`QwenExecutionContext`. No frame, wrapper, successor list, graph, or device buffer is created on the hot
path. `EuhedralInferenceRuntime` keeps a pool of idle graphs per plan view and builds another graph only
when every existing one is running a quantum, for example for concurrent sequences.

## Resource lifetimes

Each resource lives as long as its natural owner, so the token boundary neither allocates nor frees:

- Workspace storage belongs to the graph. A quantum's `QwenExecutionWorkspace` acquires its buffers,
  including the token-ID buffer, from the graph's storage at admission and releases the binding at
  retirement without freeing anything; the next quantum on the graph finds the same allocations. The
  storage is a fixed table of slots, one per buffer. A slot keeps the largest allocation a binding asked
  for, so storage is bounded by the graph's largest quantum, never by token count. A larger quantum
  replaces only its undersized slots, at admission, while the graph is idle. Because the pool recycles a
  graph only after its quantum retired, storage is never shared by two live quanta. The runtime frees it
  when it closes, and only when the device proves completion; `retainedWorkspaceBytes()` reports it, so
  the engine's device bytes after a session closes are exactly the weights plus that storage.
- A sampling quantum copies its final logits row into its session's pinned `QwenHostLogits` row. The
  logits stage queues the device-to-host copy on its own lane right after the LM head, so the
  quantum's single retirement boundary proves the row complete; the CPU reads it only after a
  successful outcome, converting into a reusable FP32 scratch row. Device logits stay in the graph's
  storage. One sequence samples serially, so one row per session suffices and sessions never share one.
  Callers that read logits on the device instead receive the detached buffer, as before.
- Pinned staging for uploads (token IDs, KV page tables) comes from the binding's small cache and stays
  owned by the quantum that queued the copy until it retires.
- Persistent sequence state (KV pages, page tables, GDN state, decode scratch) belongs to the sequence
  and is released when it completes.

`QwenExecutionContext` is the quantum: its token range, sequence lease, workspace, failure and
cancellation state, and outcome. It is the graph's `StageQuantum` binding.

## Admission

Admission is small:

1. acquire an idle graph for the quantum's plan view;
2. prepare quantum-owned resources with the graph's stream selected (sequence lease, persistent state on
   first use, the binding of the graph's workspace storage), so any initialization it queues precedes
   every stage;
3. publish the root stages to the graph's source.

After that, admission is out of the execution path. A quantum whose preparation fails reaches its
terminal outcome at admission, after the stream proves that queued initialization stopped. A quantum is
admitted at most once: if admission itself fails, the failure is thrown and the quantum's outcome fails
too, so it can never be retried into a second lease. An outcome reached at admission is published only
after the graph's stream is deselected, so outcome callbacks never launch onto it.

## The Qwen `LatticeSource`

`QwenExecutionSource` is the bottom boundary between a graph and Euhedral. Its ready storage is an MPSC
queue of `AbstractFrame`; it knows nothing about what a frame computes, which frames depend on it, or
how many stages a graph has.

- `pull` hands ready frames directly to Euhedral's consumer, honours the stop condition before taking a
  frame, and never pushes. Frames that a pulled frame makes ready are appended and delivered by the same
  pull, without recursion.
- `request` accumulates demand. Demand left by an empty drain is served when a successor later becomes
  ready: a publication on a worker thread pushes against outstanding demand.
- Euhedral never calls `pull` and `request` concurrently. Publishers run on other workers and on CUDA
  driver threads, so the queue has one drain owner at a time; a publication delivers only while no
  other drain is active.
- A driver callback thread only enqueues (`publishFromCallback`); the next `pull` or `request` delivers
  the frame.
- `admit` and `terminated` count accepted quanta. `completeGracefully` closes admission, and the source
  completes only after every accepted quantum has retired.

Each reusable graph attaches its source to the lattice once, when the graph is built. A worker draining
one graph's source therefore never holds another quantum's ready frames, and independent quanta
proceed independently. A source is available to the workers registered when it is attached; graphs
are built on first use, after the lattice has started.

## Readiness, fan-out, and fan-in

A stage frame runs its operation and returns. Its finalizer, not its body, releases its successors:

```text
stage A submits its kernels to its lane
  -> A's doFinally satisfies each outgoing edge
     -> the arrival that completes B's incoming set publishes B to the source
        -> a worker with capacity takes B (first come, first served), or the lattice
           routes it to a worker by B's hash
```

Fan-in state lives in the successor: an incoming-edge count reset per quantum. Concurrent predecessors
increment it; exactly one sees the final arrival and publishes the successor, so a join is published
once. There is no central ready queue per instruction and no scan.

```text
        -> B ->
A               D      A publishes B and C; D is published by whichever of B and C arrives last.
        -> C ->
```

The graph also counts live work: published frames that have not finished and armed device-completion
edges. When that count reaches zero, no stage of the quantum can run again, whether it succeeded,
failed, or was cancelled.

## Device lanes

CPU scheduling and CUDA ordering are separate. The runtime owns a pool of CUDA streams, its lanes: one per available processor, at
most 64, plus one lane reserved for host-to-device weight transfers when the artifact stages weights from host memory. Each graph
has a home lane, which prepares its quantum and carries its retirement boundary. A stage chooses its lane each time it runs and
submits with that lane selected on the submitting thread, whichever Euhedral worker runs it:

```text
worker 3: frame A -> kernel A on lane 5, record marker A
worker 8: frame B -> kernel B on lane 5 (continues A's lane)
worker 1: frame C -> await marker A on lane 9, kernel C on lane 9 (fork)
worker 6: frame D -> await marker C on lane 5, kernel D on lane 5 (join)
device:   A -> B -> D, and A -> C -> D
```

Every stage with successors records a reusable marker after submitting. A successor that runs on
another lane awaits it on the device (`cudaStreamWaitEvent`); one on the same lane relies on stream
order; a root on another lane awaits the quantum's preparation marker. Before the retirement boundary
every other lane the quantum used joins the home lane the same way, so the single boundary covers all
of the quantum's work. A launch that awaited another lane does not use programmatic dependent launch.

Placement is the PATH policy: each stage continues the lane of the predecessor whose longest remaining path runs through it, fixed
when the graph is built, so a graph's critical chain keeps one lane in every quantum and the other branches of a fan-out take
random lanes. Every graph spreads its stages, decode and prefill alike. There are no lane options.

With stages on different lanes the stream no longer orders every write after earlier reads and
writes of the same storage, so the plan adds those edges explicitly
(`QwenExecutionPlan.withStorageHazards`, counting aliased region storage as one buffer) wherever the
data dependencies do not already imply them. Decode keeps the GDN Q4 and Q5 projections as separate
leaf frames, and every view keeps the attention producers as four leaf frames (Q projection -> QK norm
and RoPE, KV projection -> cache append), so those branches can run on different lanes.

**Rejected placements** (decode 64/1024 + 128, Nsight Systems token period, then six paired forks):
- **RANDOM and CHAIN** (RANDOM spreads every stage over random lanes; CHAIN keeps only linear chains on one lane): decode 1-4%
  slower as lanes grew (2 to 32), because every cross-lane edge costs a device-side wait and loses programmatic dependent launch,
  and most decode edges are on the critical chain.
- **FORK** (whichever successor of a stage runs first continues its lane): under FORK the side branch of a fan-out (the GDN control
  projection, the attention value projection) can run first and take the chain's lane, moving the long branch to a new lane behind
  a cross-lane wait; PATH decides statically, with no race on the claim, and was ahead by 0.57% (5 of 6 forks) at 64 + 128 and
  0.29% (6 of 6) at 1024 + 128 on 32 lanes.
- **Spreading prefill graphs alone, or leaf attention producers alone** (six paired forks each, prefill 64 / 256 / 1024): spreading
  alone -1.0% / +0.35% / +0.00%; leaf frames alone -0.09% / -0.32% / -0.06%. Together they gain +1.32% / +1.44% / +0.50%: the leaf
  frames expose the KV branch and the lanes run it beside the Q branch, and neither helps without the other.
- **Hand-placed lanes** for the side branches of each layer's projection fan-out: +1.0-1.4% on decode, so the current frame
  granularity leaves most of the overlap on the table; finer per-kernel frames are the next step for this mechanism.

Direct operator calls with no stream selected still run synchronously; only tests and diagnostics use
them.

## Submission and completion edges

`StageTopology.Boundary` makes the boundary of each edge explicit.

- `SUBMITTED`, the ordinary edge: the consumer needs only the producer's successful submission. Stream
  order runs the consumer's device work after the producer's, so the host never waits for the
  producer's completion. Every edge of the current plans is a submission edge.
- `RETIRED`, the device-completion edge: the consumer needs the producer's device work to have
  retired. The producer arms it after submitting (event plus host callback); the driver callback only
  enqueues the edge's frame, which confirms retirement on a worker before satisfying the edge. It is
  reserved for host consumption, state publication, storage release, and crossing ordering domains.

A quantum's own retirement is its single device-completion boundary. When the live count reaches zero,
the graph joins its used lanes into its home lane, records one event there and registers one host
callback. The callback
enqueues the retirement frame, and that frame:

1. confirms the boundary (and on failure proves the device idle or poisons it);
2. runs each attempted stage's retirement hook: commit on success, release temporaries always;
3. releases the quantum's workspace binding and publishes sequence state (`QwenExecutionContext.retire`);
4. returns the graph to its pool and ends the quantum's admission count;
5. publishes the outcome.

The graph is reusable before the caller observes the outcome, so the next token finds an idle graph.
There is one CUDA host callback per quantum, not one per kernel.

## Pending and committed persistent state

Persistent sequence state has distinct frontiers. For NVFP4 attention KV (`AttentionKvState`):

- reserved: `capacity`, rows backed by pages (`prepareAppend`);
- submitted: `submittedLength()`, rows whose writes are queued on the owning quantum's lanes
  (`appendSubmitted`). Later stages of the same quantum, such as causal attention, read this frontier,
  because stream order runs their reads after the writes;
- committed: `length()`, published only at the quantum's retirement (`commitSubmitted`).

Reservation runs inside the owning quantum with its stream selected. A grown page table is filled into
pinned staging and uploaded by a copy queued on that stream, ahead of the stages that read it, so
reservation never waits for the device. The staging and any outgrown table are released when the
append is committed or discarded, after retirement: nothing in the quantum references the old table,
but freeing it mid-quantum would synchronize with the queued work.

Other quanta and external readers never see beyond the committed frontier. A failed or cancelled
quantum never commits (`discardSubmitted`). The sequence position is likewise published only at
retirement. GDN recurrent and convolution state is updated in place by stream-ordered kernels; a
quantum that does not succeed leaves its sequence terminal, so no partial update is ever resumed.

## Failure and cancellation

- A failed submission records the quantum failure and publishes no successor. A join whose predecessor
  failed is never published, even if another predecessor succeeds later: that one sees the failure and
  stops.
- Cancellation stops every stage that has not yet submitted and every publication that has not
  happened. Work already submitted to the stream stays owned by the quantum until its retirement.
- The quantum still retires through exactly one boundary, so its terminal outcome is published once.
  If the boundary cannot be registered, the stream proves itself idle (stream synchronization) before
  storage is released. If that also fails, the device is poisoned and every allocation is retained
  until the process restarts.
- A CUDA driver callback never calls CUDA, runs a frame, or releases memory.
- No frame throws into Euhedral. A failed submission, an `Error` included, is recorded on the quantum:
  an escaping `Error` would complete the graph's source or end the worker.
- Euhedral finalizes a frame it rejected without running it (its worker cache retired, or nothing was
  routable) through `doFinallyWithError`. A rejected stage fails its quantum. A rejected
  device-completion or retirement frame is finished on the rejecting thread, which is never a driver
  callback, so the quantum still retires once.
- A graph is recycled, with its workspace storage, only after its quantum's boundary was confirmed: its
  device work retired, or the device was poisoned and that storage stays owned. A poisoned device also
  keeps the graphs' storage, the sessions' pinned rows, and queued staging when the runtime closes.

## Host work on the workers

Every piece of host work a request needs runs as frames on the lattice's workers; no other thread computes for it.

- **Tokenization** (`PromptTokenization`): one frame splits the prompt into pre-tokens (control tokens, NFC, the split
  expression), then publishes one frame per two pre-tokens of BPE; the last to finish joins the chunks in order. Each frame
  takes a consecutive routing seed (`FrameSeeds`), so the chunks spread across workers. Encoding equals the tokenizer's own.
- **Generation** advances in continuations of quantum retirement: the worker that publishes a quantum's outcome selects
  the token, hands its text to the caller's callback, and admits the next quantum. The callback runs before that admission,
  so a cancellation from it (a stop sequence, an invalid tool call) stops the generation before another quantum starts.
- **The server** (`ChatCompletionService`): a container thread turns a request into a `DeferredResult` and returns.
  Rendering and validation run as one frame (`EuhedralInferenceRuntime.onWorker`), encoding as tokenization frames. One
  generation runs; a bounded list waits, and the worker that finishes a generation starts the next. Stop-sequence matching
  and tool-call parsing run in the text callback. Network writes are queued per request (`SerialTasks`) and run on workers
  one at a time in output order: a write that blocks holds only its worker, and the other workers take the remaining
  work meanwhile.
- **Clients that leave** stop their generation before its next quantum, without a thread watching them. Tomcat does
  not watch an asynchronous request's connection once its body is read, so the probes ride on the generation: after
  every prefill quantum and every decoded text a probe task is queued behind the request's writes. A JSON request's
  connection is read without blocking (a servlet `ReadListener` makes `available()` read the socket; a closed
  connection reports data). A stream is watched by its writes, and while its prompt is prefilled it writes a
  keep-alive (an SSE comment, or Anthropic's `ping`) from the second prefill quantum on. The same tasks probe the queued
  requests, which give up their places. Measured on `q3`: a JSON client that left closed its session 9 ms later, and a
  stream abandoned 1 s into a 13 s prefill let the next request run 0.45 s later.
- **The prefix cache** ([PREFIX_CACHE.md](PREFIX_CACHE.md)) captures and restores sequence state between quanta, as
  frames: each frame runs at most 16 MiB of copies between the pinned arena and the sequence's buffers, one after
  another, with no stream selected, so a worker is never held for a whole checkpoint. The worker that retires a prefill
  chunk publishes its checkpoint, then admits the next chunk; a restore runs before the first quantum is admitted.
  A 16384-token restore takes 23 ms and a checkpoint's capture about 5 ms.
- **Host jobs have their own routing seeds** (`FrameSeeds.forHostWork`). A stage graph's seeds decide which workers, and
  therefore which lanes, its stages run on. When tokenization drew from the graphs' sequence, every graph built after
  it got other seeds: on nvfp4-compressed at 32K the verify step took 34.1 ms instead of 33.7 ms (CUDA graphs off), and
  placement depended on how many prompts had been encoded before a graph was built.
- **Waking a thread outside the lattice per token is expensive.** When a blocked caller drained each token's text from a
  queue, the wake-up sat between a VERIFY's retirement and the next admission: the median gap after a 4-row VERIFY was
  276 us, 229 us when nothing outside the workers was woken. The benchmark therefore drives `generateAsync`, as the server
  does; the blocking `generate` remains for tests and tools.

Measured on Qwen3.8 27B with the chat corpus, one fork of four prompts per length:

| Prompt | Tokenization |
|---|---|
| 2048 tokens | 3.5-3.7 ms |
| 4096 tokens | 6.6-10.5 ms |
| 16384 tokens | 29-37 ms |
| 32768 tokens | 50-57 ms |

## Measured behaviour

Nsight Systems, union of kernel intervals (programmatic dependent launch overlaps adjacent kernels): decode keeps the GPU busy 94%
of the window, with 0.06% idle inside a token, a 0.13 us median positive kernel gap, and one host callback per token. Prefill of
1024 tokens runs without a stream synchronization: the GPU is busy 99.93% of the window, with one gap above 100 us, the boundary
between its two quanta, and one callback per quantum.

The host boundary between decode tokens (retirement callback to the next quantum's first launch) is 0.37 ms with no device
allocation, free, or synchronous copy and no host allocation; what remained was CPU sampling (0.20 ms argmax, 0.07 ms
BF16-to-FP32 conversion), which greedy calls avoid by selecting on the device (below). Workspace storage stays with the graph, the sampled row is copied to pinned host memory before retirement,
and page-table uploads are queued on the stream, so none of them allocates or synchronizes at the token boundary.

A greedy, unconstrained call selects on the device: `euhedral_argmax_bf16` (`native/src/sampling/`), one 1024-thread CTA over the
final row, writes a 64-bit key whose order is the host argmax's (the lowest token ID among equal maxima, never NaN or negative
infinity, -0 equal to +0), and the quantum copies back those 8 bytes instead of the row. Sampling with temperature or a vocabulary
constraint still copies the row. After the LM head the 0.5 MB logits row reaches the host in 21 us and retirement is confirmed at
68 us; a 16.1 ms decode token has about 160 us of idle inside it.

## Plan views

For a complete model, `QwenExecutionPlan.forExecution(kind, rows)` selects the topology per quantum. There is no runtime or serving
switch; the plan owns its fixed views and any view requalifies through its owner.

```text
decode and VERIFY:           SMALL view (its own instance)
draft (MTP):                 the MTP draft view
prefill, M >= 64:            REGIONS view: A + B + D
prefill, M < 64:             SMALL view: A + D, ordinary FFN and attention
```

- A: rounded residual add + following RMSNorm (the final model layer keeps its plain residual).
- B: FFN gate/up with a SwiGLU epilogue, one region (`Q3_GATE_UP_SWIGLU`) for Q3 and NVFP4 weights alike: the paired MXFP8 kernel
  ([PREFILL_MX.md](PREFILL_MX.md)) or the paired native FP4 kernel ([NVFP4_NATIVE.md](NVFP4_NATIVE.md)). The FFN down is an
  ordinary linear in every view. There is no streamed FFN view.
- D: joint BF16 A/B projection + GDN control, submitted before the heavy Q4/Q5 projections.
- Every view keeps the attention producers as leaf frames: Q4 projection -> Q/K normalization and RoPE
  (into the normalized Q/K buffer), and Q5 projection -> NVFP4 K/V page append, committed at the
  quantum's retirement.
- Host-backed weights add a `WEIGHT_TRANSFER` stage per use to every view ([NVFP4_RESIDENCY.md](NVFP4_RESIDENCY.md)).

Full attention stores K and V in sequence-owned 256-token pages. Each D256 head is rotated by
normalized H256 and encoded as 128 bytes of E2M1 codes plus 16 E4M3 scales (one per contiguous group
of 16 values). Device page tables contain raw addresses; growing a sequence allocates new pages without
copying existing KV payloads, and sequence cleanup releases pages and any decode scratch only after
admitted GPU work has drained. Prefill attention is the FlashAttention-2 style leaf with producer warps for quanta of 512 or more
rows (or 128 or more once the cache holds 2048 keys), and a 32-query by 32-key tile with FP16 tensor-core operands, FP32
accumulation and online softmax otherwise ([PREFILL_MX.md](PREFILL_MX.md)). Single-token decode splits the visible prefix into
about 48-key spans (at most 64) per query head below 2048 keys, one CTA each, and merges FP32 softmax statistics in a second
kernel; from 2048 keys one three-warp tensor-core CTA per KV head and 32-key split serves the whole query-head group
([ATTENTION_DECODE.md](ATTENTION_DECODE.md)). Correctness is defined against the represented NVFP4 values within the existing
FP16/MMA and BF16-output tolerance.

Decode and VERIFY quanta that start below 1024 tokens launch their kernels with CUDA programmatic dependent launch
(`euhedral_cuda_pdl_select`): every kernel registered for it begins with `griddepcontrol.wait`
(`native/src/common/pdl.cuh`), so it cannot read a predecessor's output early. It won 12 of 12 paired forks
(about +1%) at a 64-token context and 10 of 12 at 1024, so longer contexts keep ordinary launches. Quanta replayed from
captured CUDA graphs ([CUDA_GRAPHS.md](CUDA_GRAPHS.md)) carry programmatic edges at every position instead.

`QwenExecutionPlan.reference(weights)` is the unfused oracle used by tests; staged plans (`prefix`,
`embeddingOnly`, operator slices) are also reference-only.

## Kernel leaves

The frame DAG owns irregular scheduling; a CUDA kernel should be a homogeneous, hardware-shaped
leaf. Each expensive kernel was examined for responsibilities that do not belong in one CTA or one
warp role, and split only where the hardware measured faster. Operator times use weights rotated past the 48 MB L2, as the model
sees them, on an RTX 5070 Ti. Every kernel keeps its route's numerical contract: bitwise where the route was bitwise, and the
represented-NVFP4 tolerance for decode attention.

- **Decode attention.** The per-head kernel splits the prefix into 48-key spans, so a 1024-token prefix fills the GPU (decode +
  merge 54 us at 1025 keys; attention per decode token at 1024 context 0.54 ms). In `euhedral_attention_decode_nvfp4` lane l owns
  dimensions 8l .. 8l + 7: one 32-bit code word and one scale per lane and row for K and again for V, the query re-laid out once
  through shared memory; 19.7 us per layer at 1024 context (`_exact` keeps the per-element lane + 32 d kernel, which exact numerics
  select). From 2048 keys the GQA three-warp kernel runs ([ATTENTION_DECODE.md](ATTENTION_DECODE.md)).
- **Decode RMSNorm and BF16 projections.** Batched loads and, for a single row, 40 CTAs that each recompute the same reduction and
  write one slice (no global intermediate): the RMSNorm takes 3.0 us and the BF16 projection 3.5 us.
- **Q3, Q4 and Q5 decode, contiguous ownership.** Three words of a Q3 group hold exactly 32 whole codes, so in
  `euhedral_q3_decode_contiguous` a lane owns 32 contiguous K values whose bit positions are constants, forms one FP32 dot product
  per half group and scales it once. It streams at 720-800 GB/s (gate/up 91 us, down 48 us) and Q3 time per decode token is
  11.1 ms. A Q4/Q5 half group is 16 code bytes plus one 32-bit word of fifth bits, so a lane reads its 32 codes with one 16-byte
  load (`euhedral_q4_decode_contiguous`, `euhedral_q5_decode_contiguous`): Q5 5120 -> 12288 52 us, Q4 5120 -> 4096 15 us, near 800
  GB/s. The FP32 accumulation order differs from the scalar references (`native/src/reference/kernels.cu`), which are the exact
  oracle that `CudaGpuMemory.selectExactNumerics` selects (tests only). Against them fewer than 0.1% of outputs differ, never by
  more than one BF16 ulp, and the error against an FP64 evaluation is unchanged. The 2 to 8 row twins reproduce the one-row kernel
  bit for bit ([MTP_VERIFIER.md](MTP_VERIFIER.md)).
- **GDN convolution.** 32-row blocks give 1280 CTAs at 512 rows (68 us); each output reads only the previous three inputs.
- **Decode fusion.** With every decode GEMV near the DRAM roofline (750-850 GB/s), the remaining decode time is small kernels and
  the gaps between launches. Decode runs its own instance of the short-prefill topology: rounded residual add + RMSNorm is one
  region and the GDN A/B projection + control another. For one row the residual norm keeps its columns in registers, eight per
  thread with 16-byte loads, and reduces by warp shuffles (`euhedral_residual_rms_norm_row_bf16`, within one BF16 ulp of the exact
  kernel, which exact numerics keep).
- **Prefill attention.** The 32-query tile expands whole 16-element groups per thread (one 8-byte code load, one scale, two 16-byte
  shared stores; the same exact FP16 values) and four threads share each query row's softmax statistics, so the denominator is
  summed in four partial sums (`euhedral_attention_prefill32_nvfp4_exact` keeps the key-ordered sum): 18 ms per 1024-token prompt.
  The FlashAttention-2 leaf (`euhedral_attention_prefill_fa2_nvfp4`) takes the larger grids ([PREFILL_MX.md](PREFILL_MX.md)).
- **Prefill QK norm/RoPE and GDN control.** From two rows the QK RMSNorm + RoPE kernel gives one CTA a row
  (`euhedral_attention_qk_norm_rope_rows_bf16`): it computes the row's angles once in shared memory and gives each head to a warp
  whose shuffle tree reproduces the 256-thread RMS reduction; the RoPE rotation is pinned to explicit roundings in both layouts, so
  they agree bit for bit. Decode keeps one CTA per head. The GDN A/B projection + control runs 8-row x 4-head CTAs
  (`euhedral_gdn_project_control_8x4_fp32`, 90 us), which keep every output's FMA stripes and tree (bitwise equal to the per-row
  kernel).
- **GDN recurrence, column-owned lanes.** In `euhedral_gdn_recurrence_c8_bf16` a lane owns one value column and a 32-key slice of
  its state row, so each reduction is local FMAs plus two or three shuffles and the key and query normalizations scale the reduced
  dot products. Each row's inputs load during the previous row and each local reduction keeps four partial sums (152 registers):
  260 us at 512 rows, 46 us at 64 rows, within about 3e-5 relative rms of the exact kernel. It serves prefill (two or more rows);
  single-row decode and row-exact execution keep the exact warp-tree kernel `euhedral_gdn_recurrence_bf16`.

Relaxed numerics: every kernel above whose numerics differ from its exact counterpart (contiguous Q3/Q4/Q5 decode, the MXFP8 and
native FP4 prefill routes, the one-row residual norm, the prefill attention softmax sum, the column-owned GDN recurrence, the
contiguous and GQA decode attention) is replaced by the exact kernel under exact numerics, and
`RelaxedNumericsDriftCudaIntegrationTest` compares the two end to end. The Q3 artifact's relaxed numerics measure a hidden-state
relative error of 6.0e-2, KL 5.1e-3 and top-1 agreement 0.961 over 8 windows of 48 positions, with no growth by position
([PREFILL_MX.md](PREFILL_MX.md)); a single one-ulp perturbation (the decode attention merge order) gives about 5.4% and KL 2.6e-3,
so the combined error stays at the model's sensitivity floor: one-ulp BF16 differences settle at this level in the 64-layer
recurrent model whichever operation causes them.

Rejected:

- Spreading the one-row residual add + RMSNorm over 40 CTAs that each recompute the reduction: slower than the separate kernels
  (17.39 vs 17.23 ms per decode token).
- A dedicated producer warp feeding a TMA bulk-copy ring for Q3 decode: -29% with one L2-resident weight buffer, no gain with cold
  weights or in the model.
- A second CUDA lane for the GDN decode fan-out (Q4, Q5 and the BF16 pair): 90 -> 86-88 us per layer, under 1% of a decode token.
  The branches share one DRAM bound, so the quantum's single stream is not a material limit there.
- GDN recurrence at 4, 2 or 16 value columns per warp instead of 8: all slower; narrower warps repeat the query/key loads and
  normalizations (the 4-column kernel won as an operator, 16.5 -> 14.8 us, and measured -0.1% in decode). Single-row decode keeps
  the exact kernel.
- A chunked (WY) GDN recurrence with FP32 SIMT matrix steps: 869 us at 512 rows, bound by shared-memory load issue at one load per
  FMA; it needs register blocking or tensor cores before it can compete.
- A tensor-core decode attention CTA that shared one K/V expansion across the six query heads with 256-key splits: slower at every
  measured length (64 to 32768 keys), because its MMA, softmax and rescaling phases ran on two, half of one, and all four warps in
  turn and a 1024-token prefix occupied 20 CTAs. The three-warp 32-key-split kernel from 2048 keys replaced it.


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


# Q3 artifact format

This is the contract of the `q3` artifact (`tools/convert_checkpoint.py --quantization q3`, see `tools/README.md`) and of the
Q3G64_F16S, Q4G64_F16S and Q5G64_F16S weight formats the engine executes. The `q3-compressed` artifact stores the same tensors in
the lossless P2E2 layout ([COMPRESSED_Q3.md](COMPRESSED_Q3.md)).

## Inventory

The artifact holds the text model and the MTP draft head, 785 objects and no vision tower:

- 360 BF16 objects (norms, the GDN convolution and A/B projections), `contiguous-le-v1`
- 96 FP32 objects (GDN `a_log` and `dt_bias`), `contiguous-le-v1`
- 1 I32 object (`text/draft_head_token_ids`), `contiguous-le-v1`
- 195 Q3G64_F16S objects, `row-split-k128-v1`: the token embedding, the LM head, the draft head, and per layer the attention
  and GDN output projections and the FFN `gate_up` and `down`
- 64 Q4G64_F16S objects, `row-split-k128-v1`: `attention/query_key` and `gdn/query_key`
- 64 Q5G64_F16S objects, `row-split-k128-v1`: `attention/gate_value` and `gdn/value_z`
- 5 NVFP4 objects, `row-split-k128-v1`: the MTP layer's projections (`mtp/input_projection` and the four projections of
  `mtp/layer`). NVFP4 drafts accept more tokens per verification than Q3 ones ([MTP_VERIFIER.md](MTP_VERIFIER.md)).

The container is EDRL version 2, the only version the reader accepts: each tensor descriptor carries the source dtype, the
persistent storage format and the persistent layout as separate fields.

Canonical fused text objects:

- `text/layers/<n>/attention/query_key` (Q4) and `attention/gate_value` (Q5)
- `text/layers/<n>/gdn/query_key` (Q4) and `gdn/value_z` (Q5)
- `text/layers/<n>/mlp/gate_up` and `mlp/down`
- `mtp/layer/attention/query_key_gate_value`

## Layouts

`row-split-k128-v1`: a row of K values is padded to a multiple of 128 and split into groups of 64 values. Each plane starts on a
256-byte boundary.

| Format | Code range | Base plane per group | High plane per group | Scale plane |
|---|---|---|---|---|
| Q3G64_F16S | -4..3 | 24 bytes, three bits per code | none | one little-endian binary16 scale per group |
| Q4G64_F16S | -8..7 | 32 bytes, one nibble per code | none | same |
| Q5G64_F16S | -16..15 | 32 bytes of low nibbles | 8 bytes of fifth bits | same |

The planes are stored in the order base, high, scales. A scale is `fp16(max-abs / qmax)` and codes are quantized after the scale
is rounded to binary16. Rows are padded on K to 128.

The Java loader validates descriptor metadata, reads the exact payload range, allocates the exact byte count, uploads the bytes
and keeps fused objects as fused `TensorHandle` instances. It does not quantize, dequantize, split or repack at load time.

Layout sources: `native/src/q3/layout.cuh` and `native/src/q45/layout.cuh` (device), `tools/euhedral_artifacts/grouped.py`
(converter).

## Execution

Quantized linears run by row count. One to eight rows run the decode kernels: the one-row contiguous kernel and its 2 to 8 row
twins, whose every row equals the one-row result bit for bit (`native/src/q3/contiguous.cuh`, `native/src/q45/contiguous.cuh`).
Nine or more rows run the block-scaled MXFP8 route ([PREFILL_MX.md](PREFILL_MX.md)). The scalar references in
`native/src/reference/kernels.cu` are the numerical oracle: exact numerics (`CudaGpuMemory.selectExactNumerics`, used by tests
and never selected in production) run them, and so does any shape no kernel takes.

## P2E2

The Q3 tensors of the `q3-compressed` artifact use the smaller, lossless `row-split-p2e2-v1` layout (EDRL layout ordinal 2,
`WeightLayout.ROW_SPLIT_P2E2_V1`), produced by `tools/convert_checkpoint.py --quantization q3 --compressed`. The loader checks
each P2E2 tensor's internal consistency before uploading it. See [COMPRESSED_Q3.md](COMPRESSED_Q3.md).


