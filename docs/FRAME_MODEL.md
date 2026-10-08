# Qwen execution model

A Qwen execution plan defines a static DAG of execution stages. A reusable runtime instance represents
those stages as independent Euhedral frames. Admission exposes only root frames, through the runtime's
`InferenceLake`. Successful stages create successor readiness and publish ready
successors to the lake. CUDA stream order carries ordinary device dependencies; actual device
completion is reserved for state-publication and ownership boundaries.

Euhedral has no central scheduler or authority; scheduling is emergent. Workers take available frames
for themselves, or a frame is routed through the lattice by its hash, so every assignment is either
deterministic (hashing) or first come, first served. Euhedral-Inference defines the stages and owns
model and sequence state. Nothing in the Qwen runtime executes a stage body inline, scans for ready
work, or walks the graph after admission.

## Immutable plan and reusable graph

`ExecutionPlan` owns the weights, the staging, and its `Shape`s: its own shape and the views (decode and
the prefill region views, the drafts). It is never the live scheduler. A `Shape` (`model/qwen38`, a
`GraphShape`) is the schema of one view: the stages, each an immutable stage spec (an instruction with its
weight bindings and its input and output buffers), the dependency edges, and static metadata. It is built
through `ShapeBuilder` (`runtime/graph`), the same builder Qwen4's shape uses, and `stageTopology()` exposes
its DAG as a `StageTopology`: stages numbered in topological order, each edge tagged with the boundary it
waits for.

`StageGraph` is one reusable runtime instance of a plan view. It is built once and then rebound to one
quantum at a time:

- one `StageFrame` per stage, created with its immutable stage spec: `Shape.createStage` builds the typed
  frame for the spec's kind from `Stages` (`Stages.Embed`, `WeightTransfer`, `RmsNorm`, `Linear`,
  `CausalAttention`, `KvAppend`, the GDN stages, and the DFlash2 kinds under `Stages.DFlash2`), so a stage
  body is one small class instead of a switch over operations;
- successor references, wired at construction;
- one frame per device-completion edge and one retirement frame;
- its home lane in the runtime's lane pool;
- its own storage (`WorkspaceStorage`): the input record and the logits, which the runtime's pool keeps with the
  graph. Every other buffer is a slot of the runtime's one workspace (below).

Per quantum, the graph only resets fan-in counters and per-stage flags and binds the
`Quantum`. No frame, wrapper, successor list, graph, or device buffer is created on the hot
path. `EuhedralInferenceRuntime` keeps a pool of idle graphs per plan view and builds another graph only
when every existing one is running a quantum, for example for concurrent sequences. The runtime knows no
model: the dense model's `Execution` (`model/qwen38`) owns the lake and the staging hand-over and admits
through it.

## Resource lifetimes

Each resource lives as long as its natural owner, so the token boundary neither allocates nor frees:

- The workspace belongs to the runtime. One `SharedWorkspace` per dense runtime (one `Workspace` per Qwen4 plan) is
  allocated at load, sized for the largest quantum any view runs (`maxRows`, the prefill chunk), and every graph binds
  the same buffers. Nothing grows while quanta run: a quantum larger than the workspace is refused at admission. The
  runtime frees it when it closes, and only when the device proves completion; `retainedWorkspaceBytes()` reports it,
  so the engine's device bytes after a session closes are exactly the weights plus that storage. Graphs share it
  through edges, not turns: see [The workspace and its owner](#the-workspace-and-its-owner).
- A sampling quantum copies its final logits row into its session's pinned `HostLogits` row. The
  logits stage queues the device-to-host copy on its own lane right after the LM head, so the
  quantum's single retirement boundary proves the row complete; the CPU reads it only after a
  successful outcome, converting into a reusable FP32 scratch row. Device logits stay in the graph's
  storage. One sequence samples serially, so one row per session suffices and sessions never share one.
  Callers that read logits on the device instead receive the detached buffer, as before.
- Pinned staging for uploads (token IDs, KV page tables) comes from the binding's small cache and stays
  owned by the quantum that queued the copy until it retires.
- Persistent sequence state (KV pages, page tables, GDN state, decode scratch) belongs to the sequence
  and is released when it completes.

`Quantum` is the quantum: its plan and the shape it runs, its token range, its admission to the
sequence, workspace, failure and cancellation state, and outcome. It is the graph's `StageQuantum`
binding and an `AbstractQuantum`: one template retirement (commit if nothing failed, release, settle,
seal) that runs once.

## Admission

Admission is a frame on the workspace's owner: its `idHash` is `WorkspaceOwner.HASH` and it stays ordered on it,
so admissions run one at a time (`publishOnOwner`). Frames keep their publication order when they enter the lattice
through one source and no queue of more than one partition. The lake's sinks are like GPU streams: each has one
producer partition (`EUHEDRAL_LAKE_PARTITIONS`), carries any mixture of ordered and unordered frames, and runs an
ordered frame in its own chain's order (frames of one `idHash`), not in queue order, so two owners' chains in one
sink may run at the same time. Admissions therefore bind in the order they were published. The lake has eight sinks
(`EUHEDRAL_LAKE_SINKS`): more sinks, not partitions, spread the contention of producers and consumers.

Admission is small:

1. acquire an idle graph for the quantum's shape;
2. prepare quantum-owned resources with the graph's stream selected (the sequence's admission,
   persistent state on first use, the binding of the graph's workspace storage), so any initialization it
   queues precedes every stage;
3. bind the graph behind the previous users of each workspace buffer it touches (`WorkspaceOwner.bind`), then publish
   the root stages to the lake.

After that, admission is out of the execution path. A quantum whose preparation fails reaches its
terminal outcome at admission, after the stream proves that queued initialization stopped. A quantum is
admitted at most once: if admission itself fails, the failure is thrown and the quantum's outcome fails
too, so it can never be retried into a second sequence admission. An outcome reached at admission is published only
after the graph's stream is deselected, so outcome callbacks never launch onto it.

## The workspace and its owner

A shape declares, for each stage, the workspace buffers it reads or writes (`GraphShape.workspaceBuffers`).
`WorkspaceUse` finds each buffer's entries (the stages that touch it first) and exits (the ones that touch it last).
The workspace's owner keeps, for each buffer, the graph that used it last and that graph's exits. When it binds a new
graph, each entry of a buffer waits on an external edge (`ExternalEdge`) for the previous graph's exits of that buffer
to be submitted. These edges are lock-free: each stage keeps a stack of its external waiters, and a stage that was
already submitted, or swept when its graph went idle, satisfies a late waiter at once. Two quanta therefore run at once
on two graphs wherever their buffers do not overlap, and reuse of a buffer follows stream order across graphs. A graph
never waits on its own previous binding: it was recycled only after its device work retired.

Two kinds of buffer that used to take locks are now ordinary workspace buffers:

- **The expansion scratch** (P2E2, MX and native NVFP4 activations) is one buffer. A stage that takes it declares a
  `ScratchUse`, and the shape chains the declared users with hazard edges. The stage binds the region around its
  submission (`ExecutionGpu.withScratch`). An undeclared use takes a private region in stream order (`allocateAsync`)
  and counts a fallback.
- **Staging slots** for host-backed weights are buffers. The transfer that fills a slot writes it, and the weight's
  consumer reads it. The owner's records tell admission whether a DFlash2 block's slots still hold what its preloaded
  variant needs (`Execution.preloadHolds`).

Qwen4 declares its one workspace as a single buffer on every device stage. Its plan-wide completion chain runs one
quantum at a time, so the edge only orders a graph behind the previous one.

## The lake

Graphs publish ready frames into the runtime's `InferenceLake`: several queue ingest sinks, each an upstream source of
the lattice, chosen by a frame's routing hash. Publishing only enqueues, so a stage on a worker, a request thread and a
CUDA driver callback all publish the same way. The lake counts the quanta and host tasks it was asked to carry; closing
the runtime refuses new admissions and completes once every admitted unit terminated.

## Readiness, fan-out, and fan-in

A stage frame runs its operation and returns. Its finalizer, not its body, releases its successors:

```text
stage A submits its kernels to its lane
  -> A's doFinally satisfies each outgoing edge
     -> the arrival that completes B's incoming set publishes B to the lake
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
(`ExecutionPlan.withStorageHazards`, counting aliased region storage as one buffer) wherever the
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

1. confirms the boundary (and on failure proves the device idle or poisons it), then hands the quantum its
   conclusion, steps 2 to 5, which the quantum runs in its owner's order (`StageQuantum.concludeInOrder`): a
   sequence's quanta conclude in admission order, whatever order their device work retired in, and the graph
   stays bound until then;
2. runs each attempted stage's retirement hook: commit on success, release temporaries always;
3. retires the quantum (`AbstractQuantum.retire`): commits its work, releases its workspace binding, and
   settles the sequence's frontiers;
4. returns the graph to its pool and ends the quantum's admission count;
5. publishes the outcome by throwing the quantum's continuation into the lake.

The graph is reusable before the caller observes the outcome, so the next token finds an idle graph.
There is one CUDA host callback per quantum, not one per kernel.

The generation is an ouroboros. State and frame generation sit at the top: a generation's `Admit` frame
admits a quantum into a graph. Execution sits at the bottom: the stages and the device. The completion at
the bottom throws an action back to the top: the retired quantum's continuation is the generation's
`Select`, which samples the next token and admits the next quantum (or `Finish`). Nothing waits on a
future between the two.

## Pending and committed persistent state

Persistent sequence state has distinct frontiers. For NVFP4 attention KV (`AttentionKvState`):

- reserved: `capacity`, rows backed by pages (`prepareAppend`);
- submitted: `submittedLength()`, rows whose writes are queued on the owning quantum's lanes
  (`appendSubmitted`). Later stages of the same quantum, such as causal attention, read this frontier,
  because stream order runs their reads after the writes;
- committed: `length()`, published only at the quantum's retirement (`commitSubmitted`).

Reservation runs inside the owning quantum with its stream selected. A grown page table is allocated in stream order
(`allocateAsync`), filled into pinned staging, and uploaded by a copy queued on that stream ahead of the stages that read
it. The outgrown table is freed in stream order right after the upload (`freeAsync`), so reservation never waits for the
device and never synchronizes with queued work. The other per-sequence growth (decode scratch, seed rows, DFlash2 taps,
speculative GDN state, host-logits row selections) grows the same way.

Other quanta and external readers never see beyond the committed frontier. A failed or cancelled
quantum never commits (`discardSubmitted`). GDN recurrent and convolution state is updated in place by
stream-ordered kernels; a quantum that does not succeed leaves its sequence terminal, so no partial update
is ever resumed.

The `Sequence` keeps the same two frontiers for its position, in place of an execution lease, and is a
sequencer: parallel execution, ordered completion, as Euhedral-Execution's `FrameSequencer` and its Kafka offset
collector do it. `admit(work, start, end)` takes work (a quantum, or a prefix-cache capture or restore) at the
submitted frontier and moves that frontier to `end`; several pieces may be in flight. Each piece completes in
any order, marks itself ready, and drains the sequence's `Sequencer` (`runtime/graph`): an MPSC queue in
admission order and a work-in-progress count, so whichever thread finds the oldest piece ready concludes the
ready prefix, one thread at a time, without a lock. Concluding settles a piece: it commits (the committed
frontier moves) or it is abandoned. A piece is told at its conclusion when it can no longer commit (a piece before
it failed the sequence, or committed short of where it starts, as a partly accepted verification does); a quantum
then fails, discards its stages' work and ends the sequence FAILED, because its stages may already have changed
GDN state in place. A draft quantum and a capture leave both frontiers where they are.

Admission (the generation chain) writes the submitted frontier, and the conclusion (one thread at a time) writes
the committed one, so they are volatile, with no lock and no CAS state machine. `cancel()` may come from any
thread: it publishes its flag before it reads whether work is in flight, and admission and settlement publish their
counts before they read the flag, so an admission either backs out or is in flight when the cancellation looks, and
one of the cancellation or the settlement ends the sequence CANCELLED. A quantum that decided to commit before the
cancellation still commits (SUCCESS), and the sequence then ends CANCELLED. The terminal state is
first-writer-wins.
The persistent state closes only in `complete()`, which the session's lifecycle runs after its generation ended.

The sequence allows several quanta in flight; the state they share does not yet. `AttentionKvState` takes one
append at a time, the decode scratch and the session's host logits row are one per sequence, and quanta on
different lanes have no device edge between them, so a quantum that touches that state is refused while other work
on its sequence is in flight, and today's sessions admit one quantum of a sequence at a time.

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
  an escaping `Error` would complete the lake's sink or end the worker.
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
- **Generation** is a chain of frames. `Admit` starts a step through the model's `StepPort` (a prompt chunk, a decode
  token, an MTP or DFlash2 step); when the step's quantum publishes its outcome it throws its continuation, the `Select`
  frame, into the lake (`AbstractQuantum.publishOutcome`), so nothing that reads an outcome runs inside the graph's
  retirement. `Select` samples, applies acceptance, records the token, hands its text to the caller's callback and throws
  the next `Admit`, or `Finish`, which completes the caller's result. The callback runs before that `Admit`, so a
  cancellation from it (a stop sequence, an invalid tool call) stops the generation before another quantum starts. Each
  session's `GenerationFrames` recycles the three frame types through `FrameManager`s, one frame of a chain live at a
  time. Tokenization's join throws the first `Admit`. Only tests, tools and the blocking `generate` wait, each on its own
  thread.
- **The server** (`ChatCompletionService`): a container thread turns a request into a `DeferredResult` and returns.
  Rendering and validation run as one frame (`Execution.onWorker`), encoding as tokenization frames. One
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
- **The prefix cache** ([PREFIX_CACHE.md](PREFIX_CACHE.md)) captures and restores sequence state between quanta. Its
  tree belongs to its owner: lookups, reservations, publications and releases are frames ordered on `PrefixCache.HASH`.
  Its copies are queued asynchronously on the cache's own stream in pieces of at most 16 MiB, so a worker is never held
  for a whole checkpoint, and their device completion is a frame. Each prefix step tells the session what happened and
  throws the step's `Select`. A 16384-token restore takes 23 ms and a checkpoint's capture about 5 ms (measured with
  the synchronous copies these replaced).
- **Host jobs have their own routing seeds** (`FrameSeeds.forHostWork`). A stage graph's seeds decide which workers, and
  therefore which lanes, its stages run on. When tokenization drew from the graphs' sequence, every graph built after
  it got other seeds: on nvfp4-compressed at 32K the verify step took 34.1 ms instead of 33.7 ms (CUDA graphs off), and
  placement depended on how many prompts had been encoded before a graph was built.
- **Workers idle on fixed timing** (`InferenceEngine.fragmentConfig`): a worker with nothing to run parks 15 us and looks
  again. Speculative steps that are not replayed from a captured graph submit one frame per stage, and every frame waits for
  a worker that is awake. Rejected: the lattice's adaptive idle timing, which derives each worker's park (up to 0.8 ms) and
  its choice to idle from that worker's own history of frame costs. nvfp4-compressed through the API, 32K prompt, 256
  tokens, six requests in a row: with 4 worker CPUs 83.9-86.6 tok/s against 101.1-101.8 with fixed timing; with 32 worker
  CPUs three requests at 101-103 and then 77.7-77.9, as long prefills had left fewer workers awake, against 101.5-102.9
  throughout. Idle CPU with no request is the same with either: 0.4 cores for 4 workers, 3.6-4.3 for 32.
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
BF16-to-FP32 conversion), which greedy calls avoid by selecting on the device (below). The workspace is allocated at load, the sampled row is copied to pinned host memory before retirement,
and page-table uploads are queued on the stream, so none of them allocates or synchronizes at the token boundary.

A greedy, unconstrained call selects on the device: `euhedral_argmax_bf16` (`native/src/sampling/`), one 1024-thread CTA over the
final row, writes a 64-bit key whose order is the host argmax's (the lowest token ID among equal maxima, never NaN or negative
infinity, -0 equal to +0), and the quantum copies back those 8 bytes instead of the row. Sampling with temperature or a vocabulary
constraint still copies the row. After the LM head the 0.5 MB logits row reaches the host in 21 us and retirement is confirmed at
68 us; a 16.1 ms decode token has about 160 us of idle inside it.

## Plan views

For a complete model, `ExecutionPlan.forExecution(kind, rows)` selects the topology per quantum. There is no runtime or serving
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

`ExecutionPlan.reference(weights)` is the unfused oracle used by tests; staged plans (`prefix`,
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

## No locks on the hot path

Nothing that runs per quantum or per token takes a lock, waits blocking or starts a thread, in either model. State is
confined by routing every frame that touches it to its owner: the workspace, the expert cache, the prefix cache. Device
completions arrive as frames, and order comes from edges and from a sequence's admission order (`Sequencer`).
`ArchitectureTest.theHotPathHoldsNoLockBlocksNorStartsThreads` scans `model/qwen38`, `model/qwen4`, `runtime`,
`generation`, `prefix` and `state` for locks, blocking waits, `join`s, futures' blocking `get`s, threads and executors.
It allows them only in named lifecycle and tool methods (opening lanes and pools, closing, the blocking caller-thread
`generate`) and in the startup loaders.
