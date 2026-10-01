# Qwen execution model

A Qwen execution plan defines a static DAG of execution stages. A reusable runtime instance represents
those stages as independently schedulable Euhedral frames. Admission exposes only root frames through a
Qwen-specific `LatticeSource`. Successful stages create successor readiness and publish ready
successors back to that source. Euhedral drives every stage through its normal pull/request and
scheduling machinery. CUDA stream order carries ordinary device dependencies; actual device completion
is reserved for state-publication and ownership boundaries.

Euhedral-Inference defines the stages and owns model and sequence state. Euhedral-Execution decides
when and on which worker each ready stage runs. Nothing in the Qwen runtime executes a stage body
inline, scans for ready work, or walks the graph after admission.

## Immutable plan and reusable graph

`QwenExecutionPlan` is the schema. It describes the stages (instructions), their immutable weight
bindings, their input and output buffers, the dependency edges, the plan views (decode and the prefill
region views), and static metadata. It is never the live scheduler. `stageTopology()` exposes the DAG
as a `StageTopology`: stages numbered in topological order, each edge tagged with the boundary it waits
for.

`StageGraph` is one reusable runtime instance of a plan view. It is built once and then rebound to one
quantum at a time:

- one `StageFrame` per stage, created with its immutable instruction and weight binding
  (`QwenStageFrame.create` chooses `EmbeddingFrame`, `RmsNormFrame`, `LinearFrame`, or
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
  logits stage queues the device-to-host copy on the quantum stream right after the LM head, so the
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
one graph's source therefore never holds another quantum's ready frames, and independent quanta are
scheduled independently. Euhedral offers a source to the workers registered when it is attached; graphs
are built on first use, after the lattice has started.

## Readiness, fan-out, and fan-in

A stage frame runs its operation and returns. Its finalizer, not its body, releases its successors:

```text
stage A submits its kernels to the quantum stream
  -> A's doFinally satisfies each outgoing edge
     -> the arrival that completes B's incoming set publishes B to the source
        -> Euhedral pulls or requests B and schedules it on any worker
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

## Quantum-owned CUDA ordering

CPU scheduling and CUDA ordering are separate. A quantum owns its graph's stream while it runs, and every
stage submits to that stream with it selected on the submitting thread, whichever Euhedral worker runs
the stage:

```text
worker 3: frame A -> kernel A on the quantum stream
worker 8: frame B -> kernel B on the same stream
worker 1: frame C -> kernel C on the same stream
device:   A -> B -> C
```

Stages submitted by concurrent workers (fan-out branches) are ordered by their launch calls; a join is
submitted only after all of its predecessors' launches returned. Streams are not tied to workers.
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
the graph records one event on the quantum stream and registers one host callback. The callback
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
- submitted: `submittedLength()`, rows whose writes are queued on the owning quantum's stream
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

## Measured behaviour

Compact Q3 artifact on an RTX 5070 Ti, greedy, against the previous decode chain frame (one frame
launching a whole decode token; prefill still completed each instruction through a host callback).
Six paired JVM forks, medians of fork medians:

| Scenario | Chain frame | Frame DAG | Change | DAG ahead |
|---|---|---|---|---|
| decode 64 + 128 | 27.08 tok/s | 27.46 tok/s | +1.8% | 5/6 |
| decode 1024 + 128 | 23.58 tok/s | 24.09 tok/s | +2.1% | 5/6 |
| prefill 256 | 548 tok/s | 598 tok/s | +9.5% | 6/6 |
| prefill 1024 | 591 tok/s | 626 tok/s | +5.8% | 5/6 |
| time to first token, 64-token prompt | 186 ms | 153 ms | -17.7% | 6/6 |
| time to first token, 1024-token prompt | 1766 ms | 1662 ms | -5.5% | 5/6 |

Nsight Systems, union of kernel intervals (programmatic dependent launch overlaps adjacent kernels):
decode keeps the GPU busy 94% of the window, with 0.06% idle inside a token, a 0.13 us median positive
kernel gap, and one host callback per token; prefill-1024 is busy 99% with 0.34% idle and one callback
per quantum, where the per-instruction callbacks left it 86% busy.

The remaining decode idle was the token boundary. Attributed without a profiler, the host part of
it (retirement callback to the next quantum's first launch) took 0.70 ms per token; the 2.6 ms that
Nsight reported is a profiler-inflated mean. Of the 0.70 ms, 23 `cudaMalloc` and 23 `cudaFree` of the
per-quantum workspace and token IDs cost about 0.14 ms, and the synchronous pageable logits readback
with its fresh host row and `float[]` about 0.19 ms; CPU argmax was 0.20 ms. Prefill synchronized
the quantum stream once per full-attention layer whenever a quantum grew its KV pages (32 times per
1024-token prompt), draining the device queue each time. Keeping workspace storage with the graph,
copying the sampled row to pinned host memory before retirement, and queuing page-table uploads
removed all of that. Twelve paired JVM forks against the previous `main` (two warmups, three
iterations; paired-difference medians):

| Scenario | Before | After | Change | After ahead |
|---|---|---|---|---|
| decode 64 + 128 | 28.64 tok/s | 28.92 tok/s | +1.0% | 10/12 |
| decode 1024 + 128 | 24.98 tok/s | 25.16 tok/s | +0.7% | 9/12 |
| prefill 1024 | 650.4 tok/s | 651.6 tok/s | +0.2% | 8/12 |
| prefill 256 | 617.3 tok/s | 619.7 tok/s | +0.2% | 7/12 |

The host boundary is now 0.37 ms with no device allocation, free, or synchronous copy and no host
allocation; CPU sampling (0.20 ms argmax, 0.07 ms BF16-to-FP32 conversion) is most of what remains.
Prefill-1024 runs without a stream synchronization: the GPU is busy 99.93% of the window, with one
gap above 100 us, the boundary between its two quanta. Time to first token did not change beyond
run-to-run noise.

## Plan views

For a complete model, `QwenExecutionPlan.forExecution(kind, rows)` selects the topology per quantum.
There is no runtime or serving switch; the plan owns its fixed views and any view requalifies through
its owner.

```text
prefill:
  M == 64 or 1024 and FFN 5120 x 17408 -> streamed C + A + D + F
  otherwise, M >= 64                    -> combined A + B + D + F
  M < 64                                -> A + D, ordinary FFN and attention

decode:
  reference instruction topology
```

- A: rounded residual add + following RMSNorm (the final model layer keeps its plain residual).
- B: Q3 gate/up with a SwiGLU epilogue; ordinary Q3 down.
- C: gate/up+SwiGLU into two bounded feature slots consumed by a down pass that carries FP32
  accumulators across feature regions. It forks internal CUDA streams and joins them back onto the
  quantum stream with events, so from the graph it is an ordinary stream-ordered stage.
- D: joint BF16 A/B projection + GDN control, submitted before the heavy Q4/Q5 projections.
- F: Q4/Q5 attention producers with in-place Q/K normalization and RoPE, followed by NVFP4 K/V page
  writes, committed at the quantum's retirement.

Full attention stores K and V in sequence-owned 256-token pages. Each D256 head is rotated by
normalized H256 and encoded as 128 bytes of E2M1 codes plus 16 E4M3 scales (one per contiguous group
of 16 values). Device page tables contain raw addresses; growing a sequence allocates new pages without
copying existing KV payloads, and sequence cleanup releases pages and any decode scratch only after
admitted GPU work has drained. Prefill uses 32-query by 32-key tiles with FP16 tensor-core operands,
FP32 accumulation, and online softmax. Single-token decode splits the visible prefix across CTAs and
merges FP32 softmax statistics; at GQA ratio six and a prefix of at least 1024 tokens, each tensor-core
CTA shares one K/V expansion across all six query heads. Correctness is defined against the represented
NVFP4 values within the existing FP16/MMA and BF16-output tolerance.

Decode quanta that start below 1024 tokens launch their kernels with CUDA programmatic dependent launch
(`euhedral_cuda_pdl_select`): every kernel registered for it begins with `griddepcontrol.wait`
(`native/src/pdl.cuh`), so it cannot read a predecessor's output early. It won 12 of 12 paired forks
(about +1%) at a 64-token context and 10 of 12 at 1024, so longer contexts keep ordinary launches.
`EUHEDRAL_PDL=0` disables it.

`QwenExecutionPlan.reference(weights)` is the unfused oracle used by tests; staged plans (`prefix`,
`embeddingOnly`, operator slices) are also reference-only.
