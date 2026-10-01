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

CPU scheduling and CUDA ordering are separate. The runtime owns a pool of CUDA streams, its lanes
(`EUHEDRAL_LANES`, by default one per available processor, at most 64). Each graph has a home lane,
which prepares its quantum and carries its retirement boundary. A stage chooses its lane each time it
runs and submits with that lane selected on the submitting thread, whichever Euhedral worker runs it:

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

Placement (`EUHEDRAL_LANE_PLACEMENT`): PATH (default) lets each stage continue the lane of the
predecessor whose longest remaining path runs through it, fixed when the graph is built, so a graph's
critical chain keeps one lane in every quantum and the other branches of a fan-out take random lanes;
FORK lets whichever successor of a stage runs first continue its lane; CHAIN keeps only linear chains
on one lane; RANDOM and WORKER (the submitting worker's lane) spread every stage. Every graph spreads
its stages, decode and prefill alike.

With stages on different lanes the stream no longer orders every write after earlier reads and
writes of the same storage, so the plan adds those edges explicitly
(`QwenExecutionPlan.withStorageHazards`, counting aliased region storage as one buffer) wherever the
data dependencies do not already imply them. Decode keeps the GDN Q4 and Q5 projections as separate
leaf frames, and every view keeps the attention producers as four leaf frames (Q projection -> QK norm
and RoPE, KV projection -> cache append), so those branches can run on different lanes.

Measured on decode 64/1024 + 128 (Nsight Systems token period, then six paired forks):

- RANDOM and CHAIN placement made decode 1-4% slower as lanes grew (2 to 32): every cross-lane edge
  costs a device-side wait and loses programmatic dependent launch, and most decode edges are on the
  critical chain.
- FORK placement with 2 to 32 lanes: 16.75 -> 16.65-16.71 ms per token; 32 lanes against one, decode
  +0.3% (64) and +0.6% (1024), 6 of 6 forks each, prefill unchanged. Spreading prefill graphs too
  cost 1.1% at 256 tokens. Three hand-placed lanes (the side branches of each layer's projection
  fan-out) measured +1.0-1.4%, so the current frame granularity leaves most of the overlap on the
  table; finer per-kernel frames are the next step for this mechanism.
- PATH against FORK (32 lanes, six paired forks, one build): decode 64 + 128 +0.57% (5 of 6), decode
  1024 + 128 +0.29% (6 of 6). Under FORK the side branch of a fan-out (the GDN control projection, the
  attention value projection) can run first and take the chain's lane, moving the long branch to a
  new lane behind a cross-lane wait; PATH decides statically, with no race on the claim.
- Prefill attention producers as leaf frames instead of one fused frame (six paired forks each, prefill
  64 / 256 / 1024): spreading prefill graphs alone -1.0% / +0.35% / +0.00%; the leaf frames alone
  -0.09% / -0.32% / -0.06%; both together +1.32% (5 of 6) / +1.44% (6 of 6) / +0.50% (6 of 6). The
  frames expose the KV branch and the lanes run it beside the Q branch; neither helps without the
  other. Keeping the region storage aliasing with the leaf frames measured +1.77% / -0.07% / +0.01%
  against dropping it, so the prefill workspace keeps its size.

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
gap above 100 us, the boundary between its two quanta.

That boundary was still the largest idle block of a decode token: 395 us of a 16.1 ms token, against
about 160 us of idle inside it. After the LM head the 0.5 MB logits row reached the host in 21 us and
retirement was confirmed at 68 us; the next token's ID was uploaded at 349 us, after the host had
converted and scanned 248K logits. A greedy, unconstrained call now selects on the device:
`euhedral_argmax_bf16` (`native/src/sampling/`), one 1024-thread CTA over the final row, writes a
64-bit key whose order is the host argmax's (the lowest token ID among equal maxima, never NaN or
negative infinity, -0 equal to +0), and the quantum copies back those 8 bytes instead of the row.
Sampling with temperature or a vocabulary constraint still copies the row. Six paired forks: decode
64 + 128 +0.96% (6 of 6), decode 1024 + 128 +0.84% (6 of 6). Time to first token did not change beyond
run-to-run noise.

## Plan views

For a complete model, `QwenExecutionPlan.forExecution(kind, rows)` selects the topology per quantum.
There is no runtime or serving switch; the plan owns its fixed views and any view requalifies through
its owner.

```text
prefill:
  M == 64 or 1024 and FFN 5120 x 17408 -> streamed C + A + D
  otherwise, M >= 64                    -> combined A + B + D
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
- Every view keeps the attention producers as leaf frames: Q4 projection -> Q/K normalization and RoPE
  (into the normalized Q/K buffer), and Q5 projection -> NVFP4 K/V page append, committed at the
  quantum's retirement.

Full attention stores K and V in sequence-owned 256-token pages. Each D256 head is rotated by
normalized H256 and encoded as 128 bytes of E2M1 codes plus 16 E4M3 scales (one per contiguous group
of 16 values). Device page tables contain raw addresses; growing a sequence allocates new pages without
copying existing KV payloads, and sequence cleanup releases pages and any decode scratch only after
admitted GPU work has drained. Prefill uses 32-query by 32-key tiles with FP16 tensor-core operands,
FP32 accumulation, and online softmax. Single-token decode splits the visible prefix into about
48-key spans (at most 64) per query head, one CTA each, and merges FP32 softmax statistics in a second
kernel. Every warp runs the same online-softmax loop over its own keys. A tensor-core decode CTA that
shared one K/V expansion across the six query heads of a KV head was slower at every measured length
(64 to 32768 keys): its MMA, softmax, and rescaling phases ran on two, half of one, and all four warps
in turn, and with 256-key splits a 1024-token prefix occupied 20 CTAs. Correctness is defined against
the represented NVFP4 values within the existing FP16/MMA and BF16-output tolerance.

Decode quanta that start below 1024 tokens launch their kernels with CUDA programmatic dependent launch
(`euhedral_cuda_pdl_select`): every kernel registered for it begins with `griddepcontrol.wait`
(`native/src/common/pdl.cuh`), so it cannot read a predecessor's output early. It won 12 of 12 paired forks
(about +1%) at a 64-token context and 10 of 12 at 1024, so longer contexts keep ordinary launches.
`EUHEDRAL_PDL=0` disables it.

`QwenExecutionPlan.reference(weights)` is the unfused oracle used by tests; staged plans (`prefix`,
`embeddingOnly`, operator slices) are also reference-only.

## Kernel leaves

The frame DAG owns irregular scheduling; a CUDA kernel should be a homogeneous, hardware-shaped
leaf. Each expensive kernel was examined for responsibilities that do not belong in one CTA or one
warp role, and split only where the hardware measured faster. Paired forks against the previous
state on an RTX 5070 Ti; operator times use weights rotated past the 48 MB L2, as the model sees
them. Every change keeps its route's numerical contract: bitwise where the route was bitwise, and
the represented-NVFP4 tolerance for decode attention.

- **Decode attention.** 256-key splits put a 1024-token prefix on 20 tensor-core CTAs, whose MMA,
  softmax and rescaling phases ran on two, half of one and four warps in turn. 48-key splits of the
  plain warp kernel fill the GPU instead (decode + merge 216 -> 54 us at 1025 keys); attention per
  decode token at 1024 context fell from 6.5 to 0.54 ms.
- **Decode RMSNorm and BF16 projections.** One CTA per row walked each thread's columns as a chain
  of dependent loads. Batched loads and, for a single row, 40 CTAs that each recompute the same
  reduction and write one slice (no global intermediate): 26 -> 3.0 us and 14 -> 3.5 us.
- **Q3, Q4 and Q5 decode.** A minority of lanes loaded each K128 block's words and shuffles
  redistributed them; the LSU pipe ran at 80-98% while the down projection streamed about 320 GB/s.
  The row-major layout streams at 810 GB/s when every lane loads 16 contiguous bytes, so each warp
  now streams its own two rows in 512-byte chunks through a warp-private shared double slot and
  every lane reads whole blocks with broadcast loads (`*_decode_wide`). Q3 time per decode token
  fell from 25.2 to 19.3 ms; Q4/Q5 by about 1.5 ms. Those kernels stay bitwise identical to the
  scalar reference, which fixes each lane to the K offsets lane + 32 * stripe: the remaining limit
  was the integer work of selecting each lane's bits.
- **Q3 decode, contiguous ownership.** Three words of a group hold exactly 32 whole codes, so in
  `euhedral_q3_decode_contiguous` a lane owns 32 contiguous K values whose bit positions are
  constants, forms one FP32 dot product per half group and scales it once. It streams at 720-800
  GB/s (gate/up 136 -> 91 us, down 80 -> 48 us) and Q3 time per decode token fell to 11.1 ms. Its
  FP32 accumulation order differs from the exact kernels, which remain the oracle and are selected
  with `EUHEDRAL_EXACT=1` (or `euhedral_cuda_select_exact_numerics`). Against them, fewer
  than 0.1% of outputs differ, never by more than one BF16 ulp, and the error against an FP64
  evaluation is unchanged. `RelaxedNumericsDriftCudaIntegrationTest` feeds the same forced tokens to
  an exact and a relaxed sequence: over 2048 decode positions the final hidden state differs by a
  median 5.6% and the logits by KL 2.7e-3 with no growth with position (5.5% over the first 512
  positions, 5.7% over the last). Perturbing only the decode attention merge order instead gives
  the same profile (5.4%, KL 2.6e-3), with per-position errors correlated at 0.90: one-ulp BF16
  differences settle at this level in the 64-layer recurrent model whichever operation causes them.
- **Split-K FFN down.** Prefill quanta of 64-512 rows ran the down projection on 80-320 CTAs, one or
  two partial waves. Four K splits each accumulate FP32 partials (`FFN_PARTIALS`, sharing the
  retired `QK_PROJECTED` storage) and a reduction rounds once to BF16: down at 64 rows 740 -> 432
  us, 256 rows 1480 -> 1261 us. At 64 rows the full-width gate/up plus split-K down (1253 us per
  layer) also replaces the streamed regions and their child stream (1400 us), so only 1024-row
  quanta still stream. Prefill 64 +6.4%, 256 +3.9%, 1024 +1.2%; time to first token for a 64-token
  prompt -5.3%. Splitting gate/up measured flat or slower.
- **Q4/Q5 decode, contiguous ownership.** The same shape as Q3: a half group is 16 code bytes plus
  one 32-bit word of fifth bits, so a lane reads its 32 codes with one 16-byte load
  (`euhedral_q4_decode_contiguous`, `euhedral_q5_decode_contiguous`): Q5 5120 -> 12288 64 -> 52 us,
  Q4 5120 -> 4096 21 -> 15 us, near 800 GB/s. Decode 64 + 128 +6.2%, 1024 + 128 +7.2%.
- **One MMA per weight in prefill.** Every prefill GEMM (FFN gate/up and down, the Q3 mixer, Q3
  and Q4/Q5 projections) staged each dequantized weight as two BF16 values whose sum is code * scale
  exactly and ran an MMA on each; the kernels were tensor-pipe bound. They now stage only the BF16
  rounding of code * scale (relative error at most 2^-9 per weight) and run half the MMAs: gate/up
  at 512 rows 4494 -> 3134 us, down 2492 -> 1753 us. Prefill per 1024-token prompt 1700 -> 1196 ms
  in the trace; prefill 64/256/1024 +29.5/+27.0/+27.3% and time to first token -21% in paired
  forks. The hi + lo kernels remain as `_exact` twins.
- **GDN convolution.** One thread per channel walked every row although each output reads only
  the previous three inputs; 32-row blocks give 1280 CTAs at 512 rows (238 -> 68 us).
- **Decode fusion.** With every decode GEMV near the DRAM roofline (750-850 GB/s), the remaining
  decode time is small kernels and the gaps between about 995 launches per token. Decode now runs
  its own instance of the short-prefill topology: rounded residual add + RMSNorm is one region and
  the GDN A/B projection + control another (772 launches). For one row the residual norm keeps its
  columns in registers, eight per thread with 16-byte loads, and reduces by warp shuffles
  (`euhedral_residual_rms_norm_row_bf16`, within one BF16 ulp of the exact kernel, which exact
  numerics keep). Spreading that row over 40 CTAs that each recompute the reduction was slower
  (17.39 vs 17.23 ms per token). Decode 64 + 128 +1.0%, 1024 + 128 +1.2%, 6 of 6 forks each; drift
  unchanged.
- **One prefill tile engine for every format.** The FFN's 128 x 64 tile (`ffn/down.cuh`: four warps,
  K32 generations, warp-specialized A and B producers) now takes the weight format as a policy
  (`gemm/formats.cuh`: the Q3 producer and a Q4/Q5 producer that stages a half group's four code
  words, fifth-bit word and scale from one register per lane). At 512 rows the Q5 projections ran
  at 33 TFLOPS on their 64 x 32 kernel against the FFN's 57; on the engine (`euhedral_q5_prefill_128x64`,
  Q5 quanta from 256 rows) 5120 -> 12288 takes 1899 -> 1401 us and 5120 -> 7168 1149 -> 990 us.
  It stages the same BF16 weights and accumulates K16 steps in the same order, so it matches the
  relaxed 64 x 32 kernel bit for bit. Prefill 256 +4.3%, 1024 +4.7% (6 of 6 forks each).
- **Swizzled engine tiles; Q4 and the Q3 mixer on the engine.** Nsight Compute counted 75% of the
  engine's shared-load wavefronts in gate/up as bank conflicts: A rows and B columns were stored
  densely at a 64-byte stride, so each eight-row ldmatrix phase hit two 16-byte bank groups. Every
  engine tile now stores chunk c of row r at c ^ ((r >> 1) & 3), the swizzle the K32 compact-B
  kernel already used for B. Gate/up per 1024-token prompt 405 -> 381 ms, down 206 -> 191 ms, Q5
  166 -> 152 ms. With it, Q4 also wins on the engine (512 rows: 5120 -> 7168 898 -> 787 us,
  5120 -> 4096 524 -> 506 us), and the 6144 -> 5120 mixer output from 256 rows moves off the K32
  compact-B kernel (512 rows 734 -> 600 us). Both match their previous relaxed kernels bit for bit.
  Prefill per 1024-token prompt 1129 -> 1058 ms in the trace; prefill 256 +5.8%, 1024 +7.2% and time
  to first token for a 64-token prompt -4.7% (6 of 6 forks each).
- **Prefill attention KV staging.** The 32-query attention tile expanded each cached NVFP4 element
  with its own page lookup, code-byte and scale-byte load; at about 5 TFLOPS it took 62 ms per
  1024-token prompt. Each thread now expands whole 16-element groups (one 8-byte code load, one
  scale, two 16-byte shared stores; the same exact FP16 values), and four threads share each query
  row's softmax statistics instead of one, so the denominator is summed in four partial sums
  (`euhedral_attention_prefill32_nvfp4_exact` keeps the key-ordered sum). 62 -> 18 ms per
  1024-token prompt; prefill 256 +1.5%, 1024 +4.1% (6 of 6 forks each); drift unchanged.
- **Prefill QK norm/RoPE and GDN control.** The QK RMSNorm + RoPE kernel ran one CTA per (row,
  head) and computed each element's RoPE angle with FP64 `pow`, `cos` and `sin`, which this GPU
  executes at 1/64 rate: 412 us per 512-row quantum. From two rows one CTA owns a row
  (`euhedral_attention_qk_norm_rope_rows_bf16`), computes the row's angles once in shared memory and
  gives each head to a warp whose shuffle tree reproduces the 256-thread RMS reduction; the RoPE
  rotation is pinned to explicit roundings in both layouts, so they agree bit for bit. Decode keeps
  one CTA per head. The GDN A/B projection + control ran one CTA per (row, head), each re-reading its
  weight and activation rows; 8-row x 4-head CTAs (`euhedral_gdn_project_control_8x4_fp32`) keep
  every output's FMA stripes and tree (bitwise equal): 209 -> 90 us. 13 + 20 -> under 1 + 9 ms per
  1024-token prompt; prefill 256 +2.1%, 1024 +2.5%, time to first token -1.0% (64) and -2.4%
  (1024 tokens) (6 of 6 forks each).

- **Balanced tile engine.** In the engine of `ffn/down.cuh` two warps load activations and two
  dequantize weights, and all four issue MMAs, so the dequantizing warps paced every K32 generation
  (tensor pipe active 67% of cycles). `gemm/balanced.cuh` gives every warp a quarter of the activation
  tile and one 16-column weight tile per generation, staged through registers into the other shared
  slot with one CTA barrier per generation (127 registers instead of 168). Same weights and K16 order,
  so bit for bit equal. 512 rows: gate/up 3142 -> 2914 us, Q5 5120 -> 12288 1335 -> 1146 us,
  5120 -> 7168 931 -> 716 us, the Q3 mixer 629 -> 606 us; the FFN down stayed level (1726 -> 1745 us)
  and keeps its engine, as do the exact twins. Prefill 256 +8.9%, 1024 +8.8%, time to first token for
  a 1024-token prompt -8.7% (6 of 6 forks each); decode unchanged.
  The 128-row split-K FFN down moved to it too (same K ranges, bitwise equal partials), with three
  splits below 512 rows: 256 rows 953 -> 727 us, 512 rows (four splits) 1753 -> 1455 us. Prefill 256
  +5.5%, 1024 +3.1% (6 of 6 forks each).
  The engine takes its tile height as a parameter (32F rows): the 64-row gate/up and split-K down
  leaves moved to it as well (gate/up 588 -> 504 us, down 292 -> 278 us at 64 rows): prefill 64
  +3.6%, time to first token for a 64-token prompt -3.9% (6 of 6 forks each).
  The grouped GDN input projections (Q4 and Q5 in one launch, 33-192 rows) run on it as well, 64-row
  tiles up to 64 rows and 128-row tiles above: 64 rows 383 -> 281 us, 128 rows 713 -> 439 us,
  192 rows 987 -> 730 us, bitwise equal. Prefill 64 +3.6%, 128 +5.1%, time to first token for a 64-token
  prompt -3.4% (6 of 6 forks each).
  From 65 rows the separate Q4/Q5 projections and the mixer use 64-row balanced tiles (128 rows: Q5
  5120 -> 7168 367 -> 258 us, Q4 292 -> 212 us, mixer 325 -> 238 us); at 64 rows the 64 x 32 and 32-row
  kernels fill more CTAs and stay. Prefill 128 +3.1%, 192 +5.2% (6 of 6 forks each).
- **Prefill attention, FlashAttention-2 leaf.** The 32-row WMMA tile gives each query head its own
  CTA, so the six query heads of a KV head each expand the same NVFP4 tiles, and it rescales its output
  through shared memory every tile (about 15 TFLOPS). `euhedral_attention_prefill_fa2_nvfp4` gives a
  CTA 16 query rows of one KV head's group, one warp per query head: each 32-key tile is expanded once
  into padded FP16 rows the warps share, the rotated queries and the 16 x 256 output stay in registers
  (253 registers, no spills), and the online softmax runs in registers with quad shuffles
  (mma.sync m16n8k16 FP16, FP32 accumulation). 512 rows on an empty cache 340 -> 309 us, after 1536
  keys 1690 -> 810 us, after 3584 keys 3733 -> 1481 us. With fewer CTAs it loses (64 rows 45 -> 114
  us), so it serves quanta from 512 rows, or from 128 rows once the cache holds 2048 keys. Prefill 1024
  +0.7%, 2048 +1.7% (6 of 6 forks each); the gain grows with the context. Drift with a 1024-token
  prefix matches main (median hidden 6.2%, KL 2.9e-3).
- **Decode attention, GQA tensor-core leaf.** At long contexts decode attention grows with the cache
  (196 us per layer at 16K keys, 16% of a decode token): each of the six query heads of a KV head
  re-reads and re-decodes the same NVFP4 rows one token at a time. `euhedral_attention_decode_gqa_nvfp4`
  gives one warp per (KV head, 32-key split): each 16-key tile is expanded once into the warp's shared
  rows for the whole query-head group, and the query heads are the N = 8 columns of the mma tiles
  (scores K Q^T, output V^T P^T), so the output takes 64 accumulator registers and the queries and
  probabilities enter as hi + lo FP16 parts, with the accurate expf: within about 1e-7 to 8e-5 relative
  rms of the FP32 kernel. With the merge, 2048 keys 52 -> 42 us, 4096 keys 77 -> 54 us, 16K keys 212
  -> 120 us per layer. Putting the query heads on the 16-row M side instead wasted half the output
  registers and kept FP16 queries and probabilities (7.8e-4 relative rms, top-1 agreement in the drift
  test 95.4%). It serves contexts from 2048 keys: decode 4096 + 128 +1.1% (6 of 6 forks); at 1024
  keys it measured -0.6% and stays off.
- **Decode attention, contiguous lanes.** The decode split kernel decoded each cached NVFP4 element
  with its own code-byte and scale-byte loads (lane l owned dimensions l + 32 d). In the relaxed
  `euhedral_attention_decode_nvfp4` lane l owns dimensions 8l .. 8l + 7: one 32-bit code word and one
  scale per lane and row for K and again for V, the query re-laid out once through shared memory.
  Only each dot product's FP32 order changes (`_exact` keeps the old kernel). 1024-token context
  26.0 -> 19.7 us per layer; decode 1024 + 128 +1.9%, 4096 + 128 +2.1% (5 of 6 forks each), 64 + 128
  +0.7%; drift unchanged.
- **GDN recurrence, column-owned lanes.** A warp owned eight value columns and reduced every column's
  128-key dot products across all 32 lanes: about 90 warp shuffles per token per warp, which bounded
  the prefill recurrence (583 us per 512-row quantum; loading each token's inputs a row ahead made it
  slower, 622 us). In `euhedral_gdn_recurrence_c8_bf16` a lane owns one column and a 32-key slice of its
  state row (`_c4`: four columns, 16-key slices, twice the warps), so each reduction is local FMAs
  plus two or three shuffles, and the key and query normalizations scale the reduced dot products:
  512 rows 580 -> 327 us. Outputs stay within about 3e-5 relative rms of the exact kernel. Prefill
  256 +2.2%, 1024 +2.2%, time to first token -1.2% (64) and -2.1% (1024 tokens), 6 of 6 forks each.
  Single-row decode keeps the exact kernel: `_c4` won as an operator (16.5 -> 14.8 us) but measured
  -0.1% in decode.
  The kernel runs about 11 warps per SM (its grid), latency-bound with registers to spare: each row's
  inputs now load during the previous row and each local reduction keeps four partial sums (152
  registers): 512 rows 329 -> 260 us, 64 rows 56 -> 46 us. Prefill 64 +1.1%, 256 +1.0%, 1024 +0.4%.
  A chunked (WY) form with FP32 SIMT matrix steps measured 869 us at 512 rows, bound by shared-memory
  load issue at one load per FMA; it needs register blocking or tensor cores before it can compete.
Relaxed numerics: every kernel above whose numerics differ from its exact counterpart (contiguous
Q3/Q4/Q5 decode, split-K FFN down, single-MMA prefill, the one-row residual norm, the prefill
attention softmax sum, the column-owned GDN recurrence, contiguous decode attention) is replaced by the exact kernel under
`EUHEDRAL_EXACT=1`, and `RelaxedNumericsDriftCudaIntegrationTest` compares the two settings end to
end. With all of them, a 256-token prefill and 2048 forced decode positions give a median
hidden-state difference of 5.8%, KL 3.2e-3 and the same top-1 token at 97.0% of positions, with no
growth (the later settled half 0.92x the earlier). A single one-ulp perturbation (the decode attention merge
order) gives 5.4% and KL 2.6e-3, so the combined error stays at the model's sensitivity floor.

Measured and not kept:

- Spreading the one-row residual add + RMSNorm over 40 CTAs that each recompute the reduction:
  slower than the separate kernels (17.39 vs 17.23 ms per decode token).
- 64 x 64 tiles on the prefill tile engine: slower than 128 x 64 for every Q4/Q5 shape.
- Prefetching the balanced engine's loads two K32 generations ahead (a second register set): 163
  registers, three CTAs per SM instead of four, gate/up at 512 rows 2836 -> 2997 us. On the balanced
  engine the tensor pipe is active 77% of cycles; long-scoreboard stalls (14%) are the largest
  remainder.
- Four engine CTAs per SM (128 registers): gate/up 3016 -> 3467 us, down 1650 -> 3430 us at 512
  rows. With the swizzle the engine keeps the tensor pipe active 67% of cycles; the rest is the B
  dequantization and address arithmetic of the producer warps, which also issue MMAs.

- A dedicated producer warp feeding a TMA bulk-copy ring for Q3 decode: -29% with one L2-resident
  weight buffer, no gain with cold weights or in the model.
- A second CUDA lane for the GDN decode fan-out (Q4, Q5 and the BF16 pair): 90 -> 86-88 us per
  layer, under 1% of a decode token. The branches share one DRAM bound, so the quantum's single
  stream is not a material limit there.
- GDN recurrence at 4, 2 or 16 value columns per warp instead of 8: all slower; narrower warps
  repeat the query/key loads and normalizations.
- A 64 x 32 down tile in the 64-row streamed FFN: twice the CTAs, but 4% slower in the model,
  because the down regions overlap gate/up regions and take their SMs. Two full-width leaves on
  the quantum stream tie the streamed regions (1411 vs 1400 us per layer).

Prefill is now bound by tensor-core throughput: the FFN and projection GEMMs reach 55-80% of the
FP16/FP32 tensor rate with the hi/lo BF16 weight split their bitwise contract requires, and split-K
would change the FFN's FP32 accumulation order.

Against the `main` that preceded these changes (six paired JVM forks, two warmups, three
iterations). This host has two per-JVM performance modes about 13% apart on decode; three control
forks ran in the slow one, so the table compares medians of the forks in the fast mode:

| Scenario | Before | After | Change |
|---|---|---|---|
| decode 64 + 128 | 29.1 tok/s | 42.3 tok/s | +45% |
| decode 1024 + 128 | 25.3 tok/s | 40.4 tok/s | +60% |
| prefill 64 | 445.5 tok/s | 447.5 tok/s | +0.4% |
| prefill 256 | 618 tok/s | 628 tok/s | +1.6% |
| prefill 1024 | 651 tok/s | 662 tok/s | +1.6% |
| time to first token, 64-token prompt | 146.6 ms | 146.5 ms | unchanged |
| time to first token, 1024-token prompt | 1604 ms | 1574 ms | -1.9% |
