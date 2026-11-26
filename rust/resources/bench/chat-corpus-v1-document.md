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

File docs/NVFP4_NATIVE.md:

# Native Blackwell NVFP4

The first NVFP4 execution path (`native/src/nvfp4`, docs/NVFP4_RESIDENCY.md) never used FP4 tensor
cores. It is compiled for `compute_90` like every module. Its prefill tiles expand each weight to
BF16 and run `HMMA.16816.F32.BF16`, and decode is a scalar FP32 GEMV. This campaign adds a
Blackwell-only module (`native/src/nvfp4_native`) that runs block-scaled FP4 tensor-core MMA, and
measures it against that path and against compact Q3.

Measured 2026-10-01/02 on an RTX 5070 Ti (sm_120, 70 SMs, 16 GB) with CUDA 13.1 (NVRTC 13.1.80,
driver 615.71).

## Instruction contract

Verified against the installed toolchain and on the device, not from documentation.

**What the toolchain offers.**
- NVRTC 13.1 knows `kind::mxf4nvf4`, `kind::mxf8f6f4`, `.block_scale` and `.scale_vec::{1X,2X,4X}`.
- The NVVM intrinsic table in the installed Triton lists
  `mma.block.scale.m16n8k64.row.col.mxf4nvf4.scale.4x.f32.e2m1.e2m1.f32.ue4m3`.
- No CUTLASS or CuTe sources are installed.
- CCCL's `cuda::ptx` covers `tcgen05` only, which is sm_100 and not this GPU.

**The NVFP4 instruction:**

```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
    {d0..d3}, {a0..a3}, {b0, b1}, {c0..c3}, scaleA, {0, selA}, scaleB, {0, selB};
```

It compiles to SASS **`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`**.

| Item | Contract |
|---|---|
| Target | `sm_120a` or `sm_120f`. Plain `sm_120` is rejected by ptxas. `compute_90` PTX only "compiles" because NVRTC skips ptxas for virtual targets. |
| A and B | E2M1 only. Packed eight per 32-bit register, element j in nibble j, even K in the low nibble. |
| Accumulator | FP32 only (`.f16` is rejected). |
| Scales | UE4M3, one per 16 K values (`scale_vec::4X`): four per row or column per 64-K MMA. Decoded as E4M3 with the sign bit ignored, subnormals included; `0x7f`/`0xff` are NaN. |
| Tile | m16 n8 k64; 16384 FLOP per instruction. |
| A fragment | g = lane/4, t = lane%4. a0 row g, K 8t..8t+7; a1 row g+8; a2/a3 the same rows at K+32. |
| B fragment | b0 column g, K 8t..8t+7; b1 at K+32. |
| D fragment | d0/d1 row g, columns 2t, 2t+1; d2/d3 row g+8. |
| Scale A | Row r < 8 from lane 4r+2s, row r+8 from lane 4r+2s+1. selA = s ∈ {0,1}; ptxas rejects 2 and 3. Byte b scales K 16b..16b+15. |
| Scale B | Column n from lane 4n+s, selB = s ∈ {0..3}. Byte b scales K 16b..16b+15. |
| Error | Random codes and random scales (subnormals included), all selector pairs: max relative error 1.1e-7 against float64, which is FP32 accumulation. |

**Other forms.**
- `kind::mxf4nvf4.scale_vec::2X.ue8m0` (MXFP4) gives `OMMA.SF.16864.F32.E2M1.E2M1.E8`.
- `kind::mxf8f6f4` (E4M3 activations × E2M1 weights) gives `QMMA.SF.16832.F32.E4M3.E2M1.E8`. It accepts only one UE8M0 scale per 32 values. With UE4M3 it is rejected.
- **So NVFP4 weights with their per-16 E4M3 scales cannot be consumed natively with FP8 or BF16
  activations.** Native NVFP4 requires FP4 activations.

**The existing weight layout is consumed in place.** Codes are row-major, two per byte with the even
K low, K padded to 128. A B register is one aligned `u32` of a weight row. The scale plane is
`[rows][K/16]`, so a B scale register is one aligned `u32` of four scales. The FP32 global is
applied in the epilogue. No repacking is needed.

### Tensor-core throughput

Register-only loops with 8 independent accumulator chains per warp, 560 CTAs of 256 threads. Every
instruction was confirmed in the SASS.

| SASS | Dense TFLOPS |
|---|---|
| `HMMA.16816.F32.BF16` (the BF16-expansion path) | 98 |
| `HMMA.16816.F16` | 190 |
| `QMMA.16832.F32.E4M3.E4M3` | 197 |
| `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` | **815** |

## Kernels

`native/src/nvfp4_native/native.cuh`. Loaded by `euhedral_cuda_load_native_kernel`, which compiles a
cubin for `sm_<major><minor>a` and only on compute capability 12.x.

- **`quantize_rows<kTerms>`**: one CTA per row.
  - Per-row FP32 global = amax / (6 × 448).
  - Block scale = `e4m3_rn(block amax / (6 × global))`.
  - Codes = `e2m1_rn` with saturation (`F2FP.SATFINITE.E2M1.F32`).
  - With two terms, the residual x − term0 is quantized again with the same global, and each linear
    issues one MMA per term into the same accumulator.
- **`linear<kTerms, kPaired>`**:
  - CTA tile 128 × 128 and K tile 128, with `cp.async` stages (3 for one term, 2 for two, both
    67,584 bytes).
  - Code rows are padded to 80 bytes for conflict-free `ldmatrix`.
  - 8 warps of 64 × 32, epilogue × row global × weight global.
  - The paired variant puts 16 gate rows and their 16 up rows in each warp's slice. It applies
    SwiGLU to BF16-rounded gate and up, like the BF16 regions.
- **SASS:**

| Kernel | Main instructions | Registers |
|---|---|---|
| Linear | 32 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` per K tile (64 with two terms), `LDGSTS`, `LDSM.16.M88.4`; no weight conversion | 121 |
| Quantizer | `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`, `F2FP.SATFINITE.E4M3.F32` | 42 |

**Dispatch** (`CudaGpuMemory`, `native/src/host/nvfp4_linear.c`):

| Rows | NVFP4 linear (`linearNvfp4Bf16`) | Paired gate/up + SwiGLU (`nvfp4GateUpSwiGluBf16`) |
|---|---|---|
| 1 | the BF16-activation GEMV (`euhedral_nvfp4_decode`) when K is a multiple of 1024 and N of 16, as in every model shape; else the 64 × 64 BF16-expansion tile | not formed: decode and small views run gate/up as a linear |
| 2–63 | native skinny kernel when K is a multiple of 256 (`skinny_linear`, see "Decode-like row counts"), else the 128 × 128 tile | not formed (small views run gate/up as a linear) |
| 64 | native skinny kernel (same condition) | native paired tile |
| 65 and more | native 128 × 128 tile | native paired tile |

- **Thresholds:**
  - `NVFP4_NATIVE_MIN_ROWS` = 2 for linears;
  - `NVFP4_NATIVE_REGION_MIN_ROWS` = 64 for the paired region;
  - the host's `SKINNY_MAX_ROWS` = 64.
- Native linears need K to be a multiple of 128; otherwise the BF16-expansion kernels run.
- Quantized activations, plus FP32 split-K partials for skinny shapes, go into the shared,
  event-ordered scratch (`euhedral_cuda_nvfp4_native_scratch_bytes`).
- `EUHEDRAL_NVFP4_NATIVE`:
  - `0`: the BF16-expansion kernels;
  - `1`: one term;
  - two terms by default, the only mode that passes the drift harness.
- Exact numerics (`EUHEDRAL_EXACT=1`) always use the BF16-expansion kernels, which are the
  reference twin.

## Operator results

Real layer weights from both artifacts, rotated over 4 layers. Production entry points on one
stream, CUDA events. Native times include activation quantization. Times are in ms.

| Family | Rows | Compact Q3 | NVFP4→BF16 | Native, 1 term | Native, 2 terms |
|---|---|---|---|---|---|
| gate_up + SwiGLU | 64 | 0.502 | 0.558 | 0.190 | 0.262 |
| | 256 | 1.451 | 1.544 | 0.409 | 0.632 |
| | 512 | 2.812 | 3.029 | 0.805 | 1.211 |
| | 1024 | 5.552 | 5.919 | 1.573 | 2.405 |
| | 2048 | 21.611 | 11.761 | 3.166 | 4.707 |
| FFN down | 64 | 0.275 | 0.676 | 0.132 | 0.200 |
| | 256 | 0.719 | 1.073 | 0.260 | 0.466 |
| | 512 | 1.411 | 1.861 | 0.410 | 0.724 |
| | 1024 | 3.351 | 3.184 | 0.745 | 1.275 |
| | 2048 | 11.242 | 5.983 | 1.465 | 2.502 |
| GDN output | 64 | 0.162 | 0.237 | 0.048 | 0.074 |
| | 256 | 0.348 | 0.371 | 0.088 | 0.175 |
| | 512 | 0.593 | 0.610 | 0.139 | 0.242 |
| | 1024 | 1.032 | 1.114 | 0.240 | 0.405 |
| | 2048 | 1.959 | 2.085 | 0.509 | 0.878 |

- **One term:**
  - 3.7–4.6× the BF16-expansion path and 3.5–4.5× compact Q3 at 256–2048 rows.
  - The native tile reaches 225–290 TFLOPS, about 30% of `OMMA` peak.
  - Quantization costs 1–7% of native time.
- **Two terms:** 2.1–2.75× the BF16-expansion path.
- **Compact Q3 gate_up at 2048 rows** (21.6 ms) is off its own trend. The engine never forms
  2048-row quanta at the default 512-row chunk.

Before the change, the NVFP4 linear kernels were **93% of prefill GPU time**: 53% linears, 40%
gate_up+SwiGLU, GDN recurrence 2.7%, attention 1.9% (Nsight Systems, prefill 512 + 2048).

## End to end

Four-arm paired gate at commit 28e7070, when native linears started at 64 rows and fewer rows used
the BF16-expansion tile: 6 forks, each arm in a fresh JVM, start order rotated per fork, warmup 2,
3 iterations. Default 512-row prefill chunk. NVFP4 arms load executed objects only. Medians of
per-fork medians; "ahead" counts forks against the BF16-expansion arm.

The skinny route (c632d47) later changed the 64-row and smaller rows: prefill 64 went from 1178 to
1492 tok/s and TTFT for a 64-token prompt from 55.7 to 44.8 ms with two terms. See its gate under
"Decode-like row counts". The other rows are unaffected, because they form quanta of more than 64 rows.

| Scenario | Compact Q3 | NVFP4→BF16 | Native, 2 terms (default) | Native, 1 term |
|---|---|---|---|---|
| prefill 64 (tok/s) | 728 | 510 | 1154 (+126%, 6/6) | 1529 (+200%, 6/6) |
| prefill 256 | 1145 | 1025 | 2211 (+116%, 6/6) | 3197 (+212%, 6/6) |
| prefill 512 | 1229 | 1113 | 2506 (+125%, 6/6) | 3657 (+228%, 6/6) |
| prefill 1024 | 1225 | 1112 | 2504 (+125%, 6/6) | 3645 (+228%, 6/6) |
| prefill 2048 | 1219 | 1107 | 2477 (+124%, 6/6) | 3597 (+225%, 6/6) |
| TTFT, 64-token prompt (ms) | 88.4 | 126.9 | 55.6 (6/6) | 42.4 (6/6) |
| TTFT, 1024-token prompt (ms) | 861 | 944 | 434 (6/6) | 305 (6/6) |
| decode @64 (tok/s) | 62.9 | 51.3 | 51.2 (flat) | 51.2 (flat) |
| decode @1024 (tok/s) | 61.5 | 50.6 | 50.6 (flat) | 50.6 (flat) |
| 64 + 128 end to end (ms) | 2124 | 2623 | 2553 | 2539 |
| 1024 + 128 end to end (ms) | 2939 | 3474 | 2965 | 2836 |

- **Against compact Q3:** native prefill is 2.0–2.1× faster with two terms and 3.0× with one term at
  256–2048 tokens (6/6 forks each), and TTFT halves.
- **Decode is unchanged:** one row per token still runs the GEMV, and Q3 decodes 18% faster.

## Decode-like row counts

One-row decode streams every weight once per token, so its floor is bytes over DRAM bandwidth. The
NVFP4 GEMV already reaches it (gate_up: 100 MB in 0.128 ms, 780 GB/s). A tensor-core tile needs 16
rows, so the question is what extra rows cost.

**Skinny native kernel** (`skinny_linear`):
- 4 warps over 64 output columns and all M ≤ 64 rows.
- 256-value K tiles, up to 4 `cp.async` stages.
- K is split when 64-column tiles alone cannot occupy 140 CTAs; FP32 partials are summed with
  atomics and converted once.
- It takes 2–64 rows. One row stays on the GEMV, which is as fast and keeps BF16 activations.

**Operator times** (ms; real weights rotated over 4 layers; native includes quantization and split-K
clear and finish):

| Tensor (NVFP4 MB) | M | Q3 small-row route | NVFP4 GEMV / BF16 tile | Native 128×128, 1 term | Native skinny, 1 term | Native skinny, 2 terms |
|---|---|---|---|---|---|---|
| gate_up (100.3) | 1 | 0.095 | 0.128 | 0.165 | 0.128 | 0.129 |
| | 2 | 0.356 | 0.516 | 0.171 | 0.129 | 0.130 |
| | 4 | 0.531 | 0.518 | 0.166 | 0.129 | 0.130 |
| | 8 | 1.062 | 0.517 | 0.166 | 0.129 | 0.131 |
| | 16 | 0.403 | 0.523 | 0.174 | 0.131 | 0.132 |
| | 64 | 0.630 | 0.559 | 0.188 | 0.139 | 0.178 |
| down (50.1) | 1 | 0.046 | 0.068 | 0.117 | 0.071 | 0.077 |
| | 4 | 0.379 | 0.661 | 0.118 | 0.076 | 0.076 |
| | 16 | 0.328 | 0.663 | 0.120 | 0.072 | 0.082 |
| GDN output (17.7) | 1 | 0.013 | 0.024 | 0.035 | 0.028 | 0.031 |
| | 4 | 0.122 | 0.233 | 0.035 | 0.029 | 0.029 |
| | 16 | 0.114 | 0.233 | 0.041 | 0.028 | 0.030 |
| LM head (715.2) | 1 | 0.642 | 0.897 | 1.154 | 0.888 | 0.890 |
| | 4 | 3.625 | 3.306 | 1.159 | 0.893 | 0.895 |
| | 16 | 2.609 | 3.342 | 1.171 | 0.903 | 0.904 |

**Native NVFP4 is flat in M up to about 32 rows, at the weight-streaming floor.** Q3's small-row
route and the BF16-expansion tile cost 3–8× one row as soon as M ≥ 2.

**Whole-model estimate per decode step.**
- Linears weighted by layer count: 48 GDN layers (query_key, value_z, output), 16 attention layers
  (query_key, gate_value, output), 64 × (gate_up, down), and the LM head.
- Step time = the measured NVFP4 one-row step (19.52 ms at 51.24 tok/s) + linear(M) − linear(1).
- Linears are 94% of that step (18.3 ms).
- Attention and GDN growth with M is not included.

| M | Current NVFP4 route | Native, 1 term | Native, 2 terms |
|---|---|---|---|
| 1 | 19.5 ms (GEMV) | 19.5 ms (GEMV) | 19.5 ms (GEMV) |
| 2 | 121.4 ms | 20.9 ms (1.87× rows/s) | 21.6 ms (1.81×) |
| 4 | 120.1 ms | 21.1 ms (3.70×) | 21.8 ms (3.58×) |
| 8 | 121.1 ms | 20.8 ms (7.49×) | 21.8 ms (7.15×) |
| 16 | 120.7 ms | 22.1 ms (14.2×) | 21.8 ms (14.3×) |
| 32 | 121.9 ms | 21.8 ms (28.6×) | 23.7 ms (26.4×) |
| 64 | 125.4 ms | 23.6 ms (52.8×) | 31.3 ms (40.0×) |

**End to end, against the previous commit** (128×128 tile from 64 rows, BF16-expansion tile below):
paired gate, 6 forks, default two terms.

| Scenario | Before | After | Forks ahead |
|---|---|---|---|
| prefill 8 (tok/s) | 67.7 | 278.2 (+311%) | 6/6 |
| prefill 16 | 135.4 | 557.4 (+312%) | 6/6 |
| prefill 32 | 265.3 | 1003.1 (+278%) | 6/6 |
| prefill 64 | 1177.8 | 1492.5 (+27%) | 6/6 |
| TTFT, 16-token prompt (ms) | 119.4 | 29.1 | 6/6 |
| TTFT, 64-token prompt (ms) | 55.7 | 44.8 | 6/6 |
| decode @16 / @64 (tok/s) | 51.27 / 51.28 | 51.24 / 51.28 | flat |

**At M = 1 native loses 9%** (20.0 against 18.3 ms of linears), so one-row decode keeps the GEMV.
**From M = 2 native wins outright.** An MTP verification of 1 + 3 drafts, or a batch of 4, would
cost about 21 ms per step against 19.5 ms for one token.

## Numerics

**Operator level** (Gaussian activations, real-shape converter-quantized weights):
- Native output equals float64 over the same quantized operands within BF16 output rounding (max
  2.5e-3 of the output range).
- Against the BF16-activation result:

| Activation representation | Linear output relative RMS error |
|---|---|
| One term | 9.5% (activation quantization 9.5%) |
| Two terms | 0.88% (activation 0.87%) |

- Choosing the minimum-error scale among six candidates per block instead would only reach 8.8%.

**Model level** (`RelaxedNumericsDriftCudaIntegrationTest`, NVFP4 artifact, executed objects):
- Teacher-forced: exact arm = BF16-expansion kernels with exact numerics.
- Native changes prefill only, because decode runs the same one-row GEMV in both arms.
- "Floor" = native off: only the relaxed-order kernels differ.
- Exact against exact is bitwise zero.

| Prefix + steps | Arm | Median hidden error | Worst 1/8 window | KL mean | Top-1 | Harness |
|---|---|---|---|---|---|---|
| 256 + 384 | one term | 6.39% | 10.03% | 6.00e-3 | 95.1% | fails window bound (0.10) at the first window |
| 256 + 384 | two terms | 4.64% | 5.22% | 3.78e-3 | 97.4% | passes |
| 512 + 1024 | floor | 5.45% | 9.33% | 3.10e-3 | 97.9% | passes |
| 512 + 1024 | one term | 7.37% | 11.84% | 6.59e-3 | 97.0% | fails window bound |
| 512 + 1024 | two terms | 5.50% | 10.41% | 3.39e-3 | 97.6% | fails window bound late (position 1408), where the floor reaches 9.3% |

**Two terms sit at the floor; one term doubles KL.** Neither grows with position.
- A 1024-token single-quantum prefix does not fit beside 14 GB of weights and two sequences. 512 is
  the production chunk.
- No tolerance was changed.

File docs/COMPRESSED_Q3.md:

# Compressed Q3 weights (P2E2)

P2E2 is an optional, lossless layout of the Q3G64_F16S tensors (persistent layout
`row-split-p2e2-v1`, `WeightLayout.ROW_SPLIT_P2E2_V1`). It keeps them resident on the GPU in less
memory, with the same values. Each kernel decodes only the codes it consumes, in registers.

An artifact selects it per tensor, so no engine option is needed: load a P2E2 artifact instead of
the compact one. Every Q3 route produces the same bits as on the compact artifact.

## Results

On an RTX 5070 Ti (16 GB):

| Representation | Resident weights | Decode | Prefill | Time to first token, 64-token prompt |
|---|---|---|---|---|
| compact (row-split Q3) | 11.98 GiB | baseline | baseline | 88.6 ms |
| P2E2 Q3 | 10.39 GiB weights + 0.07-0.13 GiB scratch | -5.9% to -6.3% | -3.9% (1024-2048 tokens) to -14.5% (64) | 104.1 ms (+17%) |

**Memory.** P2E2 frees 1.585 GiB of weights, 13.2% of the model and 18.5% of the Q3 tensors. The
expansion scratch takes 73 MiB of that back in generation, at most 128 MiB. The net is about
1.46-1.51 GiB more room for KV pages.

**Decode** costs 6%: the P2E2 kernel runs at 0.94x the contiguous kernel on the wide projections and
0.85x on the long-K ones.

**Prefill** costs one expansion per Q3 tensor per chunk, about 0.25 ms per layer. That is 4% at
1024-token chunks and 15% at a 64-token prompt.

Full gate: compact versus P2E2 artifact on the same build, six paired forks. P2E2 was behind in
every fork on every metric:

| Scenario | Compact | P2E2 | Change |
|---|---|---|---|
| decode 64 + 128 | 62.66 tok/s | 58.88 tok/s | -6.0% |
| decode 1024 + 128 | 61.52 tok/s | 57.64 tok/s | -6.3% |
| decode 4096 + 128 | 59.58 tok/s | 56.09 tok/s | -5.9% |
| prefill 64 | 746.8 tok/s | 638.3 tok/s | -14.5% |
| prefill 256 | 1149.8 tok/s | 1070.7 tok/s | -6.9% |
| prefill 1024 | 1228.1 tok/s | 1180.0 tok/s | -3.9% |
| prefill 2048 | 1220.1 tok/s | 1173.0 tok/s | -3.9% |
| time to first token, 1024-token prompt | 860.4 ms | 893.4 ms | +3.8% |

**Which families benefit.** Only Q3 benefits; it is 72% of the bytes and has a 0.7-bit entropy
gap. Q4 and Q5 would save at most 0.1-0.2 GiB each, BF16 and FP32 are 0.06 GiB, and none of them
were changed.

**When to use it.** It is worth using when memory is the limit: about 1.5 GiB more KV cache on a
16 GB card, for 6% of decode throughput and 4-15% of prefill. For throughput, use the compact
artifact.


## Layout

A Q3 code is a signed 3-bit integer in [-3, 3]; the converter never emits -4. Its entropy is about
2.30 bits, because -1, 0 and +1 make up 80% of the codes. P2E2 stores:

- **Primary plane**, 2 bits per code. Symbol t = 0 means -1, 1 means 0, 2 means +1 and 3 means BIG.
  Code j of a row is bits 2(j mod 16) of word j/16, sixteen codes to a little-endian u32.
- **Payload plane**, one 2-bit unit per BIG code in row-major order. Unit p = 0 means -3, 1 means -2,
  2 means +2 and 3 means +3, sixteen to a little-endian u32, least significant first.
- **Row-base plane**, one u32 per row: the payload unit of the row's first BIG code. A kernel that
  walks a row's 1024-code slices in order derives every later slice's position from its own BIG
  counts, so one word per row is the whole index.
- **Scale plane**: the row-split FP16 scales, byte for byte.

For a tensor of `rows` rows of K codes (K a multiple of 1024), each plane starts on a 256-byte
boundary:

| Plane | Offset | Size |
|---|---|---|
| primary | 0 | rows * K / 4 |
| row base | align256(rows * K / 4) | 4 * rows |
| scales | align256(row base + 4 * rows) | rows * K / 32 |
| payload | align256(scales + rows * K / 32) | 4 * (ceil(units / 16) + 80) |

The 80 zero words after the payload let kernels prefetch whole windows without bounds checks.

The payload length depends on the codes, so the descriptor's byte size is checked against bounds
rather than a formula: no shorter than an empty payload, no longer than a full one, and a whole number
of words. At load time `P2e2Layout.validate` also recounts every row's BIG codes against the row-base
plane and the payload length, so a damaged file cannot make a kernel read past its tensor.

Specification and reference encoder/decoder: `tools/convert_compact_edrl_to_p2e2.py`. Device code:
`native/src/q3/p2e2.cuh` and `euhedral_q3_p2e2_embedding` (`native/src/embedding/kernels.cu`). Host:
`P2e2Layout`.

## Producing an artifact

```bash
python3 tools/convert_compact_edrl_to_p2e2.py \
    /mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl \
    /mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3_p2e2.edrl
```

The tool needs numpy. It transcodes every Q3 tensor whose K is a multiple of 1024: all 199 Q3
tensors of the model. Every other tensor is copied unchanged.

Each transcoded tensor is decoded back and compared byte for byte with its source before the output
is renamed into place. The manifest records the sizes and the SHA-256.

## Execution routes

| Route | P2E2 handling |
|---|---|
| Decode, one row, relaxed numerics (`euhedral_q3_decode_contiguous` shapes) | `euhedral_q3_p2e2_decode` reads the compressed tensor directly |
| Token embedding | `euhedral_q3_p2e2_embedding` gathers rows directly |
| Everything else: prefill regions, split-K and streamed FFN, 2-8 rows, exact numerics, scalar | `euhedral_q3_p2e2_expand` rebuilds the row-split tensor in a shared scratch buffer, then the unchanged row-split kernel runs |
| LM head with all-token logits (tests and direct contexts only; generation asks for the last token) | expanded in output-row chunks that fit the scratch, then each chunk is copied into place |

**Fused decode kernel.** It keeps `euhedral_q3_decode_contiguous`'s lane ownership, activations and
FP32 FMA chain. Each lane owns 32 contiguous codes of a 1024-code slice.

The codes come in pairs from a 256-entry shared table of float pairs. The table is indexed by the
pair's primary nibble and by its payload bits, placed at the pair's BIG positions and zero elsewhere.
So the 64% of pairs with no BIG code hit 9 broadcast entries in distinct banks.

Payload positions come from a warp prefix sum of each lane's BIG count. Each row and slice prefetches
32 payload words, which shuffles redistribute. A warp takes a slow path when any lane holds more than
16 BIG codes, or when the slice's units outrun the prefetched words.

**Scratch.** One device region, owned by `CudaGpuMemory`, is reused by every expansion. Each use waits
for the previous one through a CUDA event recorded on the stream that used it. Growing the region
first drains the device.

It is sized to the largest expansion seen: the FFN gate/up projection (73 MiB) in generation, gate/up
plus down (109 MiB) for 1024-row streamed prefill, and at most 128 MiB.

## Correctness

Compression is lossless, and every route is bitwise identical to the same route on the compact
artifact:

- **Converter:** every tensor round-trips byte for byte during conversion
  (`tools/test_p2e2_converter.py` covers the packing, extremes, chunk boundaries and corrupted planes).
- **Native kernels** (`native/tests/test_q3_p2e2.py`, synthetic tensors that force every path: dense
  BIG lanes, overflowing slices, empty rows):
  - decode equals `euhedral_q3_decode_contiguous` bit for bit;
  - expansion, of whole tensors and of row ranges, equals the row-split bytes;
  - the embedding equals `euhedral_q3_embedding` bit for bit.
  The host entry points' geometry checks and route decisions are tested too.
- **Full model** (`QwenP2e2CudaIntegrationTest`, `-Peuhedral.qwen.p2e2-artifact`): the P2E2 artifact
  reproduces the compact artifact's logits bit for bit in every case below. Since the outputs are
  identical, compression adds no drift.
  - an 80-token prefill with all-token logits (prefill regions, plus the chunked LM head);
  - six relaxed decode steps on the P2E2 kernel;
  - one decode step under exact numerics (expanded);
  - a 5-token prompt on the small-row kernels.
- **Existing kernels:** the PTX of every existing kernel in the q3, embedding and ffn modules is
  unchanged (`ptx_compare.py` against `main`).
- **Compact artifact unaffected:** the compact artifact runs as fast on this branch as on `main`.
  Six paired forks, all seven scenarios, stayed within ±0.4% with no consistent direction: decode
  +0.07% to +0.38%, prefill -0.08% to +0.11%.

## Characterization

**Resident weights.** The compact artifact uploads all 1,118 objects: 11.98 GiB.

| Family | Bytes | Codes / scales |
|---|---|---|
| Q3 | 8.59 GiB (text 8.45, MTP 0.13) | 7.93 GiB codes, 0.66 GiB scales |
| Q5 | 2.31 GiB | |
| Q4 | 0.91 GiB | |
| W8 | 0.12 GiB | |
| BF16 and FP32 | 0.06 GiB | |

The FFN `gate_up` and `down` projections alone are 54% of the model.

Three resident tensor families are never executed by text generation: `text/draft_head` (Q3, 0.25
GiB), `mtp/*` (0.21 GiB) and the vision tower (0.27 GiB). Not uploading them would save 0.73 GiB.
That is outside this layout and was not changed.

**Code statistics.** Order-0 entropy of the codes, and the symbol probabilities, measured over every
tensor:

| Format | Stored bits/code | Entropy | Huffman | Escape code | Ideal saving | Escape-code saving |
|---|---|---|---|---|---|---|
| Q3 | 3 | 2.31 | 2.36 | 2.40 (P2E2) | 1.83 GiB (23.1%) | 1.60 GiB (20.2%) |
| Q4 | 4 | 3.46 | 3.49 | 3.57 | 0.12 GiB (13.5%) | 0.09 GiB |
| Q5 | 5 | 4.55 | 4.57 | 4.78 | 0.20 GiB (9.0%) | 0.10 GiB |

- **Q3 code probabilities:** 0: .34, ±1: .23 each, ±2: .08 each, ±3: .02 each.
- **Stable across tensors.** The distribution is nearly identical for every tensor. Conditioning on
  the row or the column gains less than 0.04 bits, so a static code is enough and no per-tensor
  tables are needed.
- **General-purpose compressors** reach only 10-11% on packed codes (zstd -19, xz -9), because the
  codes straddle byte boundaries.
- **Q4 and Q5** offer at most 0.1-0.2 GiB each and keep their layouts.

**Scales.** Every FP16 scale is fp16(bf16 max-abs / qmax), so each is exactly invertible to its
BF16 source. A tensor has 480-730 distinct scales (1,246 for the embedding), at an entropy of about
7.3 bits, so a 10-bit index per tensor would be lossless and would save about 0.25 GiB. P2E2 keeps
the FP16 plane; see *Not done* below.

## Decode kernel development

These are operator benchmarks of layer-10 weights against `euhedral_q3_decode_contiguous`. Weight
copies were rotated past L2, and every variant is bitwise identical. Times are relative to the
contiguous kernel:

| Variant | gate_up 34816x5120 | down 5120x17408 | Notes |
|---|---|---|---|
| bytes-only ceiling (primary decode only, dependent payload loads) | 1.03x | 0.78x | latency of the dependent payload load |
| pair table, raw window index | 0.57x | 0.59x | 3.6 shared wavefronts per lookup: bank conflicts |
| + masked, position-aligned window; payload words prefetched per slice | 0.59x | 0.58x | 2.1x instructions; spills |
| + slow path in a `#pragma unroll 1` loop | 0.29x | 0.28x | per-row arrays forced into local memory |
| + fully unrolled slow path | 0.87x | 0.76x | 128 registers, about 1.1 waves on down |
| 4 rows/warp, 5 CTAs/SM (96 registers) | 0.91x | 0.82x | |
| + `lop3`/`mad` for the table index (production) | **0.94x** | **0.85x** | 1.59x instructions, 59% issue |
| 2 rows/warp; 8 rows/warp; 6/9/10 CTAs/SM | 0.83-0.93x | 0.70-0.82x | rejected |
| software prefetch of the next slice | 0.92x | 0.82x | rejected: loads were not the limit |
| swizzled table (index nib + 17 * window) | 0.86x | 0.79x | rejected: BIG entries collide with the no-BIG entries' banks |
| SWAR rank prefix instead of POPC | 0.89x | 0.79x | rejected: more ALU than the POPC it removes |

**Why decode cannot be free.** On L2-resident data the contiguous kernel decodes about 2.85e12
codes/s, only about 1.4x its DRAM-fed rate. A decoder of a 20%-smaller layout therefore keeps up
only if it spends no more than about 1.3x the instructions per code.

Any variable-length code that keeps the FP32 FMA order, which bitwise identity requires, must place
each payload unit at its code's position. That costs a rank per pair: a POPC, a funnel shift, a mask
and a table lookup, about 11 instructions per pair against 7. The best kernel runs at 1.59x the
contiguous kernel's instructions.

## Prefill development

These were candidates for decoding inside the balanced tile engine's B producer at 512 rows, so that
no expanded tensor would be needed. All were bitwise identical; times are relative to the row-split
kernels:

| Variant | mixer 5120x6144 | gate_up | down (split-K) |
|---|---|---|---|
| decode in the B stage | 0.53x | 0.63x | 0.50x |
| + 6-bit field table | 0.59x | 0.59x | 0.55x |
| + 8-register state, 128-register cap | 0.60x | 0.58x | 0.57x |
| + one load per pass | about 0.62x | | |
| bound: no decode arithmetic | 0.61x | 0.60x | 0.58x |
| bound: no payload loads | 0.82x | 0.80x | 0.77x |
| bound: payload loads at fixed addresses | 0.70x | 0.62x | 0.66x |

Staging sits between the MMAs and the per-generation barrier. The extra loads and shuffles each K32
block needs cost more than the arithmetic, and even without the payload loads the engine stays
below 0.8x.

Expanding into the row-split layout instead runs at about the DRAM rate:

| Tensor | Expansion time |
|---|---|
| mixer | 18 us |
| gate_up | 160 us (790 GB/s read + write) |
| down | 73 us |
| LM head | 1.2 ms |

That is about 0.25 ms per layer per prefill chunk, on top of the unchanged kernels.

## General-purpose codecs: GDeflate and ANS

GDeflate (nvCOMP / DirectStorage) is Deflate laid out so that a warp decodes a 64 KB page in
parallel. nvCOMP decompresses whole pages into a separate buffer, so it could only replace the
expansion route; decode reads every Q3 weight once per token.

These are nvCOMP 5.3 measurements on layer-10 tensors; every decompression was exact. Ratios are
relative to the row-split tensor; P2E2 alone is 0.815:

| Input | Codec | Ratio | gate_up decompression | down decompression |
|---|---|---|---|---|
| row-split | GDeflate, entropy only (type 0) | 0.885 | 2.04 ms (35 GB/s) | 1.65 ms |
| row-split | GDeflate, maximum ratio (type 5) | 0.885 | 2.19 ms | 1.89 ms |
| row-split | nvCOMP ANS | 0.893 | 0.47 ms | 0.28 ms |
| P2E2 | GDeflate, type 0 | 0.765 | 1.97 ms | 1.69 ms |
| P2E2 | nvCOMP ANS | 0.771 | 0.45 ms | 0.24 ms |

For comparison:

- P2E2's decode GEMV of gate_up takes 94 us per token, and its expansion takes 160 us.
- zlib level 9 per 64 KB page agrees with nvCOMP on the codes: 0.91 of the packed planes. One code per
  byte reaches 0.79, and only with Huffman-only coding. The scale plane reaches 0.57, which nothing can
  read in place.

GDeflate alone compresses less than P2E2 and decompresses 12x slower than P2E2's expansion.
Stacked on P2E2 it would save another 6% of the Q3 bytes, but only through expansion, even for decode.
Rejected.

## Not done

- **Scale indices** (about 0.25 GiB; GDeflate shows the scale plane compresses 43%): a 10-bit
  dictionary index or a BF16 mantissa/exponent code would need a second table per tensor in every
  kernel. It is left for a later step.
- **Q4/Q5 escape codes:** about 0.1 GiB each.
- **Fused prefill:** see above.
- **Not uploading the unused draft head, MTP layer and vision tower:** 0.73 GiB, and independent of
  this layout.

File docs/BENCHMARKING.md:

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

File docs/NVFP4_RESIDENCY.md:

# NVFP4 residency budget

The question was whether NVFP4 weights can run near compact-Q3 speed with real VRAM headroom and
MTP retained, by keeping NVFP4 as the execution representation and streaming part of the weights
from pinned host RAM ahead of use.

**Conclusion:** Single-sequence decode reads every weight every token, and the platform's
host-to-device rate cannot stream the useful offload (at least 1.8 GiB) within Q3's 16 ms token. It
takes 44 ms even at the 44 GB/s that huge-page pinned memory reaches; 4 KiB-page pinned memory under
the IOMMU stalls at 25 GB/s.

**Staged NVFP4 can match non-MTP Q3 only with MTP accepting about 2.75 or more tokens per
verification step**, and Q3 would gain from MTP too. Prefill can hide the transfers.

Measured on 2026-10-01 on an RTX 5070 Ti (16 GB) with an i9-14900K.

## Memory budget

| Item | Size |
|---|---|
| Device total | 15.56 GiB (16,703,946,752 bytes) |
| Other processes (desktop) | about 0.46 GiB |
| Q3 run: used after loading 11.98 GiB of weights | 13.19 GiB, so about 0.75 GiB is CUDA context, NVRTC modules and engine |
| Graph and workspace pools after the first run | +0.16 GiB (they persist) |
| Q3 free for KV and per-sequence state | about 2.2 GiB, roughly 115K tokens of context |
| KV (NVFP4 pages, 16 attention layers, 4 KV heads, D256, 144-byte rows) | 18 KiB per token: 0.56 GiB at 32K, 2.25 GiB at 128K |
| GDN state (48 layers, 48 x 128 x 128 FP32, plus convolution) | about 0.15 GiB per sequence |
| NVFP4 artifact (text, LM head, MTP, draft head; Q3 embedding) | 14.52 GiB; about 0.19 GiB free after loading, so it cannot run |
| MTP layer + draft head, NVFP4 | 0.57 GiB |
| NVFP4 with block scales compressed (4-bit dictionary + escape, lossless) | about 13.77 GiB |

**Compressed NVFP4 with MTP leaves about 0.4 GiB** for KV and per-sequence state, roughly 15K
tokens. Matching Q3's headroom takes at least **1.8 GiB** of weights off the GPU (13.77 - 11.98), and
more once staging buffers are counted.

**Lossless compression of NVFP4 is limited.** The E2M1 codes carry 3.90 of their 4 bits, because the
non-uniform FP4 grid already equalizes code use. The block scales carry 3.63 of their 8. Only the
scales compress usefully, saving 0.75 GiB.

## Platform

**Link: PCIe Gen 5 x16, about 63 GB/s theoretical per direction.** Under transfer load sysfs
reports 32 GT/s x16 for both the GPU (`01:00.0`) and its root port (`00:01.0`), and `nvidia-smi`
reports gen 5 current and max, x16. The link drops to 2.5 GT/s when idle.

**The GPU hangs off the CPU's own PCIe root port.** `nvidia-smi topo -m` shows CPU affinity 0-31 on
NUMA node 0, and the PCI tree shows no switch and no chipset path. The CPU-attached NVMe has its own
root port (`00:06.0`), so it does not split the GPU's lanes.

**IOMMU: Intel VT-d in translated mode.** `dmar0` is active, all 24 groups are `DMA-FQ`, and the
kernel command line has no `iommu=pt`.

**Host DRAM: about 52-55 GB/s read.** A STREAM-style test with 16-32 threads gives copy 47 GB/s,
triad 49-53 GB/s and read 52-55 GB/s.

All copies below use genuinely pinned memory (`cudaHostAlloc`, or `cudaHostRegister`), as raw
`cudaMemcpyAsync` timed with events, 1 GiB transfers:

| Pinned host memory | H2D | D2H |
|---|---|---|
| `cudaHostAlloc` (4 KiB pages; also write-combined and portable variants) | 25.0-25.4 GB/s | 22.5 GB/s |
| `malloc` + `cudaHostRegister`, 4 KiB pages | 24.8 GB/s | 21.7 GB/s |
| `malloc` + `cudaHostRegister`, 2 MiB transparent huge pages | **43.9-44.1 GB/s** | **56.4 GB/s** |

Two copy streams over huge-page buffers give the same 43.7 GB/s.

**What limits the 4 KiB-page copies is IOMMU translation.** Translated DMA over 4 KiB pages
throttles the copy engine at about 25 GB/s. Huge pages let the IOMMU map 2 MiB at a time, and D2H
then reaches the link's practical rate.

H2D stays at 44 GB/s. That is GPU-initiated reads of host memory, which run below D2H's posted
writes, plus any translation cost left. Booting with `iommu=pt` would separate the two.

**Staging buffers must be huge-page-backed pinned memory**, registered with `cudaHostRegister`.
`cudaHostAlloc` alone caps at 25 GB/s here.

## Bandwidth budget

**Host-to-device: 44 GB/s** with huge-page pinned memory (25 GB/s with 4 KiB pages). Transfer times
below use 44 GB/s.

Streaming cost of each family per decode token, from host RAM, in compressed bytes (4.26 bits per
weight):

| Family | Bytes per layer | PCIe time per layer (44 GB/s) | Whole family |
|---|---|---|---|
| FFN gate_up (x64) | 94.9 MB | 2.16 ms | 5.66 GiB |
| FFN down (x64) | 47.4 MB | 1.08 ms | 2.83 GiB |
| GDN value_z (x48) | 33.5 MB | 0.76 ms | 1.50 GiB |
| GDN query_key (x48) | 11.2 MB | 0.25 ms | 0.50 GiB |
| GDN output (x48) | 16.7 MB | 0.38 ms | 0.75 GiB |
| attention query_key (x16) | 19.5 MB | 0.44 ms | 0.29 GiB |
| attention gate_value (x16) | 19.5 MB | 0.44 ms | 0.29 GiB |
| attention output (x16) | 16.7 MB | 0.38 ms | 0.25 GiB |
| LM head (x1) | 676.8 MB | 15.38 ms | 0.63 GiB |

Compact-Q3 decode takes 16.0 ms per token (62.7 tok/s at 64 + 128), about 0.25 ms per layer for
every family together.

## Why decode cannot be hidden

**There is no reuse window.** Every token needs every layer, so a staged weight is read once and
evicted. The time between the earliest safe prefetch and its use does not matter; the bytes per
token do. With perfect overlap a token takes max(GPU time, PCIe time).

**Even fully resident, NVFP4 is behind Q3.** It reads 13.4 GiB of weights per token against Q3's
10.7 GiB. At the same DRAM efficiency that is about 17.6 ms per token, 10% slower than Q3, before
any offload.

**Offload hides only up to the GPU time.** At 44 GB/s, 0.72 GiB per token stays under 17.6 ms
(0.4 GiB at 25 GB/s). The minimum useful offload, 1.8 GiB, is 44 ms per token: 2.7x Q3's token time.
Matching Q3 would need 120 GB/s, nearly twice PCIe 5.0 x16's theoretical rate.

**MTP does not change the ratio.** It divides streamed bytes and GPU work by the same number of
tokens accepted per verification step. Against non-MTP Q3 (16 ms per token), offloaded NVFP4 would
need about 2.75 accepted tokens per step to break even (4.8 at 25 GB/s). Euhedral has no MTP decode
yet, so acceptance has not been measured.

## Prefill

A chunk reuses its weights across all its tokens. 1.8 GiB is 44 ms per chunk, against about 417 ms
of compute for a 512-token chunk, so overlapped transfers could hide it.

Prefill-only staging is not useful alone: decode is the binding regime.

## Fully resident NVFP4

With `weightResidency: EXECUTED` the vision tower, MTP layer and draft head stay in the artifact, and
the NVFP4 base model runs from the GPU. Measured 2026-10-01, one session per artifact, warmup 1,
2 measured iterations, serving stopped:

| Artifact | Decode, 64-token prompt | Decode, 8K prompt | Prefill 512 | TTFT at 8K | Engine allocation | Device free |
|---|---|---|---|---|---|---|
| Compact Q3 | 59.3 tok/s | 54.1 tok/s | 1154 tok/s | 7.66 s | 12.09 GiB | 2.23 GiB |
| NVFP4, executed objects | 48.1 tok/s (-19%) | 45.4 tok/s (-16%) | 1052 tok/s (-9%) | 8.34 s | 14.05 GiB | 0.62 GiB |

**NVFP4 is slower than compact Q3, by about its extra bytes.** Decode reads about 25% more weight
bytes per token. 59.3 / 1.25 = 47.4 tok/s predicts the measured 48.1, so the NVFP4 GEMV streams at
Q3's efficiency, and no kernel work can close the gap at batch 1. Prefill loses less because its
tiles are compute-bound.

Greedy generation answers "The capital of France is" with " Paris."
(`QwenNvfp4GenerationCudaIntegrationTest`).

## Host-backed weights

`hostWeightBytes` (benchmark `hostWeightMiB`) keeps layer projections in pinned host memory and stages
them into a device ring of `stagingSlots` slots on every use.

- **Selection:** whole families are taken smallest tensor first (GDN query_key, GDN output, attention
  output, ...). Within the last family the layers are spread evenly. The ring then holds the
  smallest possible slot.
- **Host memory:** one arena, allocated before any payload is read, 2 MiB-aligned, with
  `MADV_HUGEPAGE` and `MADV_COLLAPSE`, and pinned with `cudaHostRegister`.
- **Execution:** each plan view gets a `WEIGHT_TRANSFER` stage per use, run on a lane reserved for
  copies. A transfer depends only on the consumer that last read its slot. The consumer depends on
  the transfer and reads the slot. Ordering uses the graph's existing cross-lane markers, with no
  host synchronization.
- **Concurrency:** a quantum that stages weights holds the ring from admission until its last stage
  submitted. The next one's preparation awaits a marker recorded after the holder's lanes joined.

Greedy tokens are identical to the resident run (`QwenNvfp4GenerationCudaIntegrationTest`).

NVFP4, executed objects, 4 slots (warmup 1, 2 iterations; resident: 48.1 / 45.4 / 1052):

| Host-backed | Staged per token | Ring | Decode @64 | Decode @8K | Prefill 512 |
|---|---|---|---|---|---|
| 512 MiB (GDN query_key) | 0.54 GB | 47 MB | 45.4 tok/s | 42.5 tok/s | 1048 tok/s |
| 1 GiB | 1.08 GB | 71 MB | 35.0 tok/s | 34.5 tok/s | 1048 tok/s |
| 2 GiB | 2.15 GB | 83 MB | 19.2 tok/s | 19.3 tok/s | 1046 tok/s |

**Up to the GPU's own token time the copies hide.** 0.54 GB is 12.4 ms of copying under a 20.8 ms
token, and costs 6%. Beyond that, decode runs at the copy rate: 2.15 GB in 52 ms is 41 GB/s, against
43.7 GB/s for bare copies. Prefill is unaffected.

**Measured and not kept:**
- **One allocation per tensor:** compaction under the loader's page-cache churn failed for a third
  of them, which fell back to 4 KiB pages. Copies then split between 26 and 43 GB/s, about 30 GB/s
  overall, and 2 GiB decoded at 15.0 tok/s.
- **Deeper rings:** 8 and 16 slots gave 14.5 and 14.2 tok/s against 15.0 for 4 slots (per-tensor
  allocation).
- **The staging DAG without its copies** costs 2% (47.2 against 48.1 tok/s).

## What was built

- `tools/convert_qwen_safetensors_to_compact_edrl.py --profile nvfp4`: the NVFP4 artifact, quantized
  on the GPU by default.
- Loading: `Nvfp4Layout`, a loader that accepts the vision-free inventory, and
  `QwenNvfp4CudaLoadIntegrationTest`.
- Execution: `native/src/nvfp4` (decode GEMV, tile-engine producer for prefill, paired gate_up and
  SwiGLU), `euhedral_cuda_linear_nvfp4_bf16` and `euhedral_cuda_nvfp4_gate_up_swiglu_bf16`, plan and
  frame dispatch for NVFP4 weights, and `WeightResidency`.
- Host-backed weights: `HostWeightSelection`, `WeightStaging`, `WeightTransferFrame`, the transfer
  lane in `LanePool`, and `euhedral_cuda_host_weights_malloc`.

File docs/MTP_CONTRACT.md:

# MTP speculative decode: contract

This is the contract written before implementation. Sources:

- **Checkpoint:** `/mnt/shared/qwen38-quant/source/qwen`: `config.json` and the safetensors index.
- **This repository's converter:** `tools/convert_qwen_safetensors_to_compact_edrl.py`.
- **llama.cpp at 67a17c17c** (`/mnt/shared/qwen38-quant/llama.cpp`): `src/models/qwen35.cpp`,
  `common/speculative.cpp`, `conversion/qwen.py` and `src/llama-memory-recurrent.cpp`. File:line
  references below are to this tree.
- **ninfer** (`~/.hermes/worktrees/ninfer-upstream`), an independent implementation:
  `tools/convert/qwen3_5.py`, `src/models/qwen3_5/execution/text.cpp`,
  `docs/maintainer/qwen3_5-model.md` and `docs/maintainer/replayssm-gdn.md`.

The two implementations agree on everything marked "established". Where they differ, the choice
below says so.

## 1. Model

**Checkpoint.** `text_config.mtp_num_hidden_layers = 1` and `mtp_use_dedicated_embeddings = false`.
There are 15 tensors:
- `mtp.fc` [5120, 10240];
- `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.norm`;
- `mtp.layers.0.*`: one full-attention decoder block with gated q, q/k norm, o_proj and a SwiGLU MLP.

There is no MTP embedding and no MTP LM head (`qwen35.cpp:113-115` falls back to the base token
embedding and base `lm_head`).

**Artifact objects** (both artifacts; NVFP4 in the NVFP4 profile):

| Object | Shape | Source |
|---|---|---|
| `mtp/input_projection` | 5120 × 10240 | `mtp.fc` |
| `mtp/embedding_norm`, `mtp/hidden_norm` | 5120 | `pre_fc_norm_embedding`, `pre_fc_norm_hidden` |
| `mtp/layer/input_norm`, `mtp/layer/post_attention_norm` | 5120 | layer norms |
| `mtp/layer/attention/query_key_gate_value` | 14336 × 5120 | rows `[q 6144; k 1024; gate 6144; v 1024]` |
| `mtp/layer/attention/query_norm`, `key_norm` | 256 | q/k norms |
| `mtp/layer/attention/output` | 5120 × 6144 | o_proj |
| `mtp/layer/mlp/gate_up`, `mlp/down` | 34816 × 5120, 5120 × 17408 | MLP |
| `mtp/final_norm` | 5120 | `mtp.norm` |

- **Attention packing.** The converter splits the per-head interleaved `q_proj`
  (`[q_h | gate_h]` per head, confirmed by `qwen35.cpp:556-573` and ninfer `qwen3_5.py:496-510`)
  into contiguous q and gate, exactly as for base layers. The packed tensor is the base layers'
  `query_key` (7168 rows) followed by `gate_value` (7168 rows).
- **Formats.**
  - NVFP4 profile: every MTP projection is NVFP4.
  - Compact Q3: `query_key_gate_value` is `W8G32_F16S`, which no Euhedral kernel executes; the
    others are Q3.
- **Norms.** Every MTP norm is the `(1 + w)` RMSNorm form (llama.cpp's converter bakes +1 into every
  `*norm.weight` except the GDN gated norm, `qwen.py:394-395`; ninfer
  `qwen3_5-model.md:116-124`). Euhedral already uses this form for base layer and q/k norms
  (`RMS_NORM_UNIT_OFFSET`, `qk_norm_rope.cuh`).

## 2. One MTP row (established)

Inputs: a token x and a hidden h. Output: an MTP hidden m, and draft logits.

1. `e = RMSNorm₁₊w(embed(x), embedding_norm)` using the base token embedding, and
   `n = RMSNorm₁₊w(h, hidden_norm)` (`qwen35.cpp:539-543`).
2. `c = input_projection · [e ; n]`: the embedding is columns 0..5119 and the hidden 5120..10239
   (`qwen35.cpp:545-548`; ninfer `mtp_pack.h`).
3. A full-attention decoder block identical to a base full-attention layer (`qwen35.cpp:553-618`):
   - `input_norm`, gated q, q/k norm;
   - partial IMROPE (64 rotary dims, θ = 1e7), the same as base layers;
   - attention over the **MTP's own KV cache** with scale `1/√256`;
   - `· sigmoid(gate)`, o_proj, then the residual with `c`;
   - `post_attention_norm`, SwiGLU FFN, residual.
4. `m = RMSNorm₁₊w(block output, final_norm)`. This is `mtp.norm`, after the block
   (`qwen35.cpp:621-628`).
5. The draft logits are the base `lm_head · m` (`qwen35.cpp:630-637`).

**Seed hidden.** h is the base model's hidden at position p **after the base final norm**: the
same vector the LM head reads (`qwen35.cpp:206-209`; ninfer `text.cpp:701-702`). The comments that
say "before final norm" (`llama-graph.h:937`) are stale.

**Alignment.** The row pairs `(h_p, x_{p+1})`. h_p is the base hidden of the row that predicted
x_{p+1}. The row's output predicts x_{p+2}.

**Recursion.** Draft k+1 uses `(m_k, x̂_{k+1})`: the previous MTP row's post-`mtp.norm` hidden and
the token it drafted (`speculative.cpp:1665-1718`; ninfer `mtp.cpp:52-67`). One MTP layer is reused
for every depth.
- llama.cpp defaults to 3 drafts (`common.h:326`); ninfer allows up to 5.
- **MTP3 = three recursive applications.**

## 3. Choices where the references differ

- **MTP row position: the hidden's position p** (ninfer, `mtp_alignment.h:21-37`), not p + 1
  (llama.cpp). RoPE is relative, so a uniform shift leaves every attention score mathematically
  unchanged, and either choice only affects drafts, never committed output.
- **No zero-hidden row for the first prompt token** (ninfer). llama.cpp pairs x₀ with h = 0
  (`speculative.cpp:1430`); training pairs only real (h_p, x_{p+1}).
- **Draft head.** The artifacts carry `text/draft_head`: the 131,072 most frequent `lm_head` rows,
  chosen by the converter's ranking, plus `text/draft_head_token_ids`, the token for each row.
  - It is **this repository's construct, not the checkpoint's**; the checkpoint's MTP uses the full
    `lm_head`.
  - **Choice: draft with the shortlist** (argmax over 131,072 rows, then map through
    `draft_head_token_ids`). It reads 360 MB instead of 682 MB per draft in NVFP4.
  - A token outside the shortlist can never be drafted. It still appears in output whenever the
    verifier produces it, so this only costs acceptance.
  - The full head stays a measurable alternative.

## 4. The speculative step (MTP3, greedy)

State before a step:
- base KV and GDN state committed through position P − 1;
- t₀ = x_P, the last committed output token, not yet run through the base model;
- three drafts d₁..d₃ for positions P+1..P+3.

1. **Verify.** Run the base model on rows `[t₀, d₁, d₂, d₃]` at positions P..P+3. Each row j
   gives `g_j = argmax lm_head(final_norm(h_{P+j}))`.
2. **Accept.** a = the largest count with `d_i = g_{i-1}` for all i ≤ a (0 ≤ a ≤ 3).
   - Committed output this step: `d₁..d_a`, then the base token `g_a`. That is a + 1 tokens
     (1..4), each exactly the token ordinary greedy decode produces, if the verifier rows are
     numerically identical to ordinary decode (see §6).
   - EOS among them ends generation at that token; tokens after it are discarded.
3. **Commit base state** through position P + a (rows 0..a). The new t₀ is g_a at position
   P + a + 1.
4. **MTP catch-up and draft 1.** Run MTP rows `(h_{P+j}, x_{P+j+1})` for j = 0..a, where
   x_{P+j+1} = d_{j+1} for j < a and g_a for j = a, at positions P+j.
   - This writes MTP KV for those positions (kept).
   - Row a's output gives d'₁ (for position P+a+2) and m₁.
5. **Drafts 2 and 3:** MTP rows `(m₁, d'₁)` at P+a+1 and `(m₂, d'₂)` at P+a+2. Their MTP KV is
   speculative: the next step's catch-up rewrites those positions from verified hiddens.

**After the prompt:**
- The base prefill must keep the post-final-norm hidden of every prompt row, not only the last.
- MTP catch-up then runs over (h_p, x_{p+1}) for p = 0..N−2, and over (h_{N−1}, g) where g is the
  first generated token, to seed d₁.

**Verifier rows, drafts and output.** 4 rows per verifier invocation: 3 proposed drafts plus the
base row. Accepted drafts number 0..3, committed tokens 1..4, and the bonus is g_a in every case.
The full-acceptance case is a = 3 with 4 tokens committed.

## 5. State mutated, and what commit must restore

| State | During drafting | During verify | Commit after a accepted |
|---|---|---|---|
| Base KV (16 layers) | none | rows P..P+3 written beyond the committed frontier | committed length += a + 1. Rejected rows stay beyond `length` and the next append overwrites them; this is the existing reserved/submitted/committed frontier with a partial publish |
| Base GDN recurrent (48 × 3 MiB FP32 = 144 MiB) and convolution (48 × 60 KiB) | none | advanced through 4 rows | must equal the state after row a. Plan: ReplaySSM (below) |
| MTP KV (1 layer) | 2 speculative rows | none | catch-up rows P..P+a committed; draft rows discarded and rewritten next step |
| Sequence position, output tokens | none | none | += a + 1 |

**GDN rollback, measured choices** (§7 of the campaign log):
- **Per-row snapshots** (llama.cpp, `llama-memory-recurrent.cpp:101`, `delta-net-base.cpp:546-602`):
  4 × 144 MiB extra and 4 full-state writes per step.
- **ReplaySSM** (ninfer, `replayssm-gdn.md`):
  - Verify runs the recurrence and convolution against a scratch copy, or from a kept checkpoint.
  - It records each row's convolved q/k/v, α and β: about 21 KB per row per layer, about 4 MiB for
    4 rows over 48 layers.
  - It then replays the accepted prefix from the committed checkpoint **with the same per-row
    transition code**, so the result is the bitwise sequential state.
  - Cost: one 144 MiB checkpoint buffer, and a replay of a + 1 rows only when a < 3.
- The convolution state after row a is the last three pre-convolution rows ending at a, rebuilt
  from the old history plus the recorded rows.

**Tests required.**
- After every step, the base KV rows, GDN recurrent state, GDN convolution state, MTP KV rows and
  sequence position must equal those of ordinary sequential decode over the same committed tokens,
  **bitwise**.
- This holds only if the verifier's rows are bitwise identical to single-row decode. See §6.

## 6. The verifier is numerically identical to single-row decode

The oracle is ordinary greedy decode, which runs every row through the decode kernels:
- the NVFP4 M = 1 GEMV with **BF16 activations**;
- split-KV decode attention;
- the decode GDN path.

Committed tokens and state equal ordinary decode exactly only if the verifier computes each of its
rows with arithmetic identical to that single-row path. Native NVFP4 OMMA cannot do that against the
GEMV: it needs FP4 activations (`kind::mxf4nvf4` only accepts E2M1 × E2M1). The multi-row prefill
attention and GDN kernels also accumulate in a different order from the decode kernels.

**Decision (2026-10-01): exact first, native as a second mode.**

- **Default mode: row-exact verification.** A VERIFY quantum selects row-exact execution
  (`euhedral_cuda_row_exact_select`). Every operator whose kernel depends on the row count then runs
  each row through its one-row path:
  - attention, q/k norm and RoPE, residual norms and the GDN recurrence, row by row;
  - NVFP4, Q3, Q4 and Q5 linears through multi-row twins of the one-row GEMV kernels
    (`nvfp4::decode_rows<M>`, `q3::contiguous_decode_rows<M>`, `q45::contiguous_decode_rows<B, M>`,
    for M = 2 to 8). Each repeats the one-row FMA sequence for every token row while streaming the
    weights once, so every row is bitwise identical to one-row decode.
- **Second mode: native numerics** (`ExecutionGpu.selectNvfp4NativeDecode`,
  `EUHEDRAL_NVFP4_NATIVE_DECODE=1`). Every NVFP4 linear, one row included, runs on the native skinny
  OMMA kernel.
  - Each row's MMA, and its activation global scale, are independent of the other rows.
  - The split-K reduction stores per-split partials and sums them in split order, so it is
    deterministic.
  - So an M-row native verification equals native single-row decode, and the oracle in this mode is
    greedy decode with the same native numerics.

Both modes are tested bitwise, on tokens and on GDN and KV state, by
`SpeculativeDecodeCudaIntegrationTest`. Results are in docs/MTP_SPECULATIVE.md.

File docs/MTP_SPECULATIVE.md:

# MTP speculative decode: results

Greedy MTP3 speculative decoding (docs/MTP_CONTRACT.md) on the RTX 5070 Ti (16 GB). It covers what
was built, how exactness is proven, and what it measured. Campaign of 2026-10-01 and 2026-10-02,
branch `perf/mtp-speculative`.

## 1. What runs

One speculative step is a chain of quanta on the existing runtime, with no central dispatcher and no
host loop over layers:

1. **VERIFY** over `[t₀, d₁, d₂, d₃]`. This is the decode topology with ALL_TOKENS logits and a device
   argmax per row; only the four token IDs cross to the host. Acceptance (`SpeculativeAcceptance`)
   resolves on retirement. The base KV commits `a + 1` rows (`AttentionKvState.commitSubmitted`).
   GDN checkpoints its state and records its per-row inputs for a replay.
2. **DRAFT catch-up**: the MTP layer over the committed outputs, seeded by the verified rows'
   post-final-norm hiddens. Its last row drafts d′₁.
3. **Two recursive DRAFT rows**, each seeded by the previous MTP hidden.

The MTP layer is a plan view (`QwenExecutionPlan.mtpDraft`):
- embedding, then `MTP_STEM` (the two norms and the `fc` projection);
- one full-attention layer with its own KV slot;
- `mtp.norm`, then the 131,072-row draft head.

The prompt prefills in chunks, each followed by its MTP catch-up. `InferenceTuning.speculativeDepth`
(the benchmark's `speculativeDepth`) enables it for greedy, unconstrained generations from a fresh
sequence.

**GDN rollback is ReplaySSM.** VERIFY checkpoints the recurrent and convolution state and records each
row's convolution inputs and α/β. On a partial accept, the next quantum restores the checkpoint and
replays the committed rows through the same row-exact kernels. Cost: about 157 MiB per sequence.
Measured per verification (1024-token prompt): 221 MiB of device-to-device copies, 0.47 ms of
copy-engine time, partly overlapping kernels.

## 2. Exactness

The oracle is ordinary one-row greedy decode, and the requirement is identical token IDs and
identical state, bitwise.

- **Verifier (row-exact).** Every row goes through the one-row arithmetic (MTP_CONTRACT §6):
  - attention, norms and the GDN recurrence run row by row;
  - linears use multi-row twins of the one-row GEMVs that repeat its FMA sequence per row.
- **Tests:**
  - `SpeculativeVerifyCudaIntegrationTest`: logits, GDN state and KV after a VERIFY quantum equal
    sequential decode, bitwise. NVFP4 at 2, 3, 4 and 8 rows over prompts of 300 to 2100 tokens
    (across the 2048-key attention kernel switch); Q3 at 4 rows; partial commits with replay.
  - `SpeculativeDecodeCudaIntegrationTest`: full MTP3 generation equals greedy decode in tokens and
    final state, for three prompts (a chat answer, 600 tokens of prose, a Java chat prompt). Passes
    on NVFP4 exact, on NVFP4 native numerics (against native greedy decode) and on compact Q3.
  - Native: `test_nvfp4.py` (decode_rows bitwise), `test_nvfp4_native.py` (skinny rows independent
    and deterministic), `test_q3_p2e2.py` and `test_q45_kernels.py` (Q3/Q4/Q5 twins bitwise).
- **Every benchmark row records `outputSha256`.** In every gate the speculative arms' hashes equal
  the oracle arm's for every prompt (§4).

## 3. Memory (NVFP4 + MTP3)

The NVFP4 artifact does not fit with MTP fully resident, so the minimum of base weights is
host-backed. MTP is never host-backed: `HostWeightSelection` only takes `text/layers/N/...`
projections.

| Item | MiB |
|---|---|
| Base layers + output head + embedding (artifact) | 14,275 |
| of which host-backed (54 GDN `query_key` tensors) | 641 |
| Resident base | 13,633 |
| MTP layer (resident) | 228 |
| Draft head (131,072 rows) + token ids (resident) | 360.5 |
| Speculative GDN (checkpoint, records, replay scratch) | about 157 |

Host-backed sizing: 640 MiB is the minimum that runs a 4096-token prompt. 512 MiB fails at 1024
tokens, even after MTP catch-up was chunked into 128-row pieces.

**Gate VRAM** (allocated peak during an iteration, and device memory free after it), at 64 / 4096
prompt tokens:

| Arm | Peak allocated GiB | Free after, MiB | H2D per output token |
|---|---|---|---|
| Q3 | 11.43 / 11.60 | 3269 / 3147 | 0 |
| Q3 + MTP3 | 12.04 / 12.25 | 2797 / 2635 | 0 |
| NVFP4 (all resident) | 14.13 / 14.31 | 711 / 582 | 0 |
| NVFP4, 640 MiB host-backed | 13.57 / 13.75 | 1355 / 1227 | 641 MiB |
| NVFP4 + MTP3, 640 MiB host-backed | 14.33 / 14.55 | 743 / 575 | 222 MiB (630 MiB per verification) |

Q3 + MTP3 needs no host-backed weights and leaves 2.6 GiB free at 4096 tokens.

## 4. Gates

Measurement setup:
- **Prompts:** `promptCorpus: "chat"`, four chat requests with thinking disabled. A 4096-token prompt
  quotes 3964-3970 tokens of a frozen document; a 64-token prompt is a standalone 29-36 token request.
- **Generation:** 256 tokens; warmup 1, then 4 iterations (one per prompt).
- **Forks:** 6 paired forks; each fork runs every arm in a fresh JVM, rotating the starting arm.
- **Rate:** per fork, committed decode tokens over decode time across the four prompts. The
  speculative decode window starts at first-token selection, so it includes the prompt's last MTP
  catch-up.
- **Exactness:** every speculative arm's `outputSha256` equals its oracle arm's for every prompt in
  every fork (Q3 = Q3 + MTP3; NVFP4 = NVFP4 host-backed = NVFP4 + MTP3; native = native + MTP3).

**Gate 1** (commit 2ca3555; Q3 verification still looped linears one row at a time):

| Arm (decode tok/s, median of 6 forks) | 64 | 4096 |
|---|---|---|
| Q3 | 59.5 | 56.4 |
| Q3 + MTP3, per-row Q3 linears | 42.5 | 41.9 |
| NVFP4 | 48.3 | 46.6 |
| NVFP4, 640 MiB host-backed | 43.7 | 42.2 |
| NVFP4 + MTP3 (exact) | **90.8** | **81.7** |
| NVFP4 native one-row decode | 43.5 | 42.1 |
| NVFP4 native + MTP3 | **92.6** | **82.5** |

Pairs, with forks ahead of 6:

| Pair | 64 | 4096 | Ahead |
|---|---|---|---|
| NVFP4 + MTP3 vs NVFP4 | 1.881× | 1.754× | 6/6 |
| NVFP4 + MTP3 vs NVFP4 640 MiB host-backed | 2.075× | 1.935× | 6/6 |
| native + MTP3 vs exact + MTP3 | 1.020× | 1.011× | 6/6 |
| native one-row vs GEMV one-row | 0.902× | 0.905× | 0/6 |
| Q3 + MTP3 (per-row) vs Q3 | 0.714× | 0.742× | 0/6 |

**Gate 2** (commit 8edc9f3: the Q3/Q4/Q5 multi-row twins):

| Arm | 64 | 4096 |
|---|---|---|
| Q3 | 59.5 | 56.4 |
| Q3 + MTP3 | **94.9** | **82.8** |
| NVFP4 + MTP3 | 90.7 | 81.7 |

| Pair | 64 | 4096 | Ahead |
|---|---|---|---|
| Q3 + MTP3 vs Q3 | 1.597× | 1.467× | 6/6 |
| NVFP4 + MTP3 vs Q3 | 1.523× | 1.448× | 6/6 |
| NVFP4 + MTP3 vs Q3 + MTP3 | 0.955× | 0.987× | 0/6 |

**Acceptance and step anatomy** (all forks; histogram = verifications that accepted 0/1/2/3 drafts):

| Arm | Context | Histogram | Mean accepted | Tokens per verification | Verifications per token | Verify ms | Draft ms | Verifications/s | TTFT ms |
|---|---|---|---|---|---|---|---|---|---|
| Q3 + MTP3 | 64 | 600/498/462/792 | 1.61 | 2.60 | 0.384 | 24.5 | 2.9 | 36.5 | 86 |
| Q3 + MTP3 | 4096 | 540/606/540/696 | 1.58 | 2.57 | 0.389 | 27.2 | 3.9 | 32.2 | 3854 |
| NVFP4 + MTP3 | 64 | 354/552/312/936 | 1.85 | 2.84 | 0.352 | 28.6 | 2.9 | 31.9 | 38 |
| NVFP4 + MTP3 | 4096 | 396/414/438/906 | 1.86 | 2.84 | 0.352 | 31.1 | 3.8 | 28.8 | 1865 |
| native + MTP3 | 64 | 360/510/372/912 | 1.85 | 2.84 | 0.352 | 27.7 | 3.0 | 32.6 | 38 |
| native + MTP3 | 4096 | 438/426/390/924 | 1.83 | 2.81 | 0.356 | 30.2 | 3.9 | 29.4 | 1869 |

Notes:
- **Draft time** is wall time between verifications: catch-up, two recursive rows and host gaps.
- **TTFT** for the ordinary arms: Q3 85 / 3743 ms, NVFP4 34 / 1813 ms. MTP adds the per-chunk
  catch-up: +1.6 to +4.5 ms at 64 tokens, +53 to +111 ms at 4096.
- **Q3 acceptance is lower** (1.6 accepted drafts against 1.85). Its MTP is Q3, with its attention
  re-quantized from W8 to NVFP4.
- **Commit and rollback:** base KV commit and MTP KV truncation only move frontiers. GDN checkpoints
  and records copy 221 MiB per verification (0.47 ms of copy engine, overlapping kernels). The
  replay of up to three committed rows is part of the next verification's time.

## 5. Profile and dependency map

Nsight Systems, 1024-token chat prompt. Times are per quantum, as exclusive kernel time: each kernel
is charged from the later of its start and the previous kernel's end.

**NVFP4: one exact M=4 verification against one decode token.** The two columns come from different
traces:
- **decode:** ordinary NVFP4 decode, all weights resident;
- **verification:** NVFP4 + MTP3 with 640 MiB of base weights host-backed.

Kernel times are comparable, since host-backed linears run the same kernels on staged copies. Span
is not: the verification span includes waiting for the host-backed weight stream.

| Category | Decode (M=1) | Verify (M=4) | Ratio |
|---|---|---|---|
| Linears (GEMV / `decode_rows4`) | 18.21 ms | 22.69 ms | 1.25× |
| Attention (row-exact, per row) | 0.58 | 2.01 | 3.5× |
| Norms (row-exact) | 0.23 | 0.87 | 3.7× |
| GDN (recurrence, convolution, control) | 1.31 | 1.03 | (lane overlap; not comparable) |
| GPU busy | 20.41 | 26.74 | 1.31× |
| Span (first kernel to last) | 21.14 | 29.62 | (mixed residency; see below) |

- The verification span exceeds its busy time by 2.9 ms. That gap is the 630 MiB host-backed weight
  stream: 15.4 ms of H2D at about 43 GB/s, overlapping compute.
- **Like-for-like cross-check.** The NVFP4 + MTP3 trace's own one-row decode quantum (the final commit
  at the budget, a single sample, same 640 MiB host-backed setup) has busy 19.53 ms and span 22.38 ms.
  The verification is 1.37× its busy time and 1.32× its span. The gated decode rates agree: an M=4
  verification costs 1.24-1.31× a host-backed decode step (§4).
- Native mode: M=4 skinny OMMA linears take 19.98 ms against 18.89 ms at M=1 (1.06×).
  Activation quantization (0.72 ms) and split-K finish (0.18 ms) come on top. Its M=1 linears are
  slower than the GEMV (19.75 against 18.21 ms), which is why native ordinary decode loses 10%.
- Q3: verification linears (Q3/Q4/Q5 twins) take 18.57 ms against 12.76 ms one-row (1.46×). GPU busy
  is 22.76 against 15.00 ms (1.52×).
- **Extra rows are not free.**
  - Linears: the first three extra rows cost 25% (NVFP4) or 46% (Q3).
  - Row-exact attention and norms scale with rows. At 4096 tokens, attention reads the KV cache once
    per row: verification grows 2.5 ms from 64 to 4096 tokens, one decode token only 0.8 ms.

**Step anatomy** (NVFP4 + MTP3, profiled step 33.0 ms):

| Part | Time |
|---|---|
| VERIFY span | 29.6 ms |
| Host gap | 0.28 ms |
| DRAFT catch-up (1 to 4 MTP rows, native skinny) | 1.4 ms |
| Host gap | 0.18 ms |
| DRAFT recursion ×2, each | 0.87 ms + 0.18 ms gap |

A one-row MTP draft costs about 0.74 ms of GEMV. Most of it is the 360 MiB draft head.

**Dependency map of one step:**

```
VERIFY(P..P+3) ─▶ accept a ─▶ DRAFT catch-up(P..P+a) ─▶ DRAFT(P+a+1) ─▶ DRAFT(P+a+2) ─▶ VERIFY(P+a+1..)
   │ base KV/GDN commit a+1 rows       │ needs the verified hiddens h_P..h_{P+a}
   └ GDN replay (a<3) runs in the next VERIFY's first frames
```

The chain is strictly serial:
- every draft needs the verified hidden states and the accepted count;
- the next verification needs all three drafts;
- inside a verification the four rows are independent only in the linears, and stay causal in
  attention and GDN.

The one independent piece of work MTP exposes is in the prompt. The catch-up for prefill chunk i
depends only on chunk i's hiddens, so it could overlap the base prefill of chunk i + 1. Today it runs
between them (+53 to +111 ms TTFT at 4096 tokens).

## 6. Rejected and open

**Rejected:**
- **`decode_rows` loop interchange.** Decode each weight value once, then FMA it into all M rows,
  with packed activations. Bitwise equal, but slower with cold weights: 1.30-1.53× one row's time,
  against 1.13-1.48× for the current kernel.
- **`decode_rows<M ≥ 4>` register prefetch of the next slice.** Bitwise equal. With cold weights,
  1.00-1.23× one row's time against 1.06-1.47×, but in-model NVFP4 + MTP3 was −1% in 5 of 6 forks
  at both contexts (verification 28.8 against 28.6 ms). Not kept.
- **Random-word benchmark prompts for MTP.** The model repeats itself and 31 of 34 steps accept all
  drafts, which overstated MTP3 at 119 tok/s. Replaced by the chat corpus.

**Open:**
- **The bottleneck is now verification itself.** Its linears' multi-row cost is 1.25× (NVFP4) and
  1.46× (Q3). Row-exact attention and norms grow linearly in rows, and attention with context.
  - Exact multi-row twins of decode attention (shared KV reads across the rows, per-row arithmetic
    unchanged) and of the residual norms are the next exact-mode candidates.
  - The Q3 twins are the larger linear lever.
- **Overlap the prompt's MTP catch-up** with the next prefill chunk (TTFT only).
- **MTP2 against MTP3:** depth is a tuning parameter (`speculativeDepth`); only depth 3 was gated.

File core/src/main/java/io/euhedral_execution/inference/core/scheduling/QwenSpeculativeDecoder.java:

package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;

/// Greedy MTP speculative decoding for one sequence (docs/MTP_CONTRACT.md §4). Every output token is
/// the token ordinary greedy decode produces, and the committed state is the state it leaves: drafts
/// only decide how many tokens one verification commits.
///
/// Each step is a chain of quanta, each on the existing runtime:
/// 1. A VERIFY quantum over `[t₀, d₁ .. d_n]`: row-exact, with greedy selections and acceptance on retirement.
/// 2. A DRAFT catch-up over the committed outputs, seeded by the verified rows' hidden states. It commits
///    those MTP cache rows, and its last row drafts d′₁.
/// 3. n − 1 recursive DRAFT rows, each seeded by the previous MTP hidden.
///
/// The prompt is prefilled in chunks that seed drafting, each followed by its catch-up. Not thread-safe;
/// one decoder per sequence.
public final class QwenSpeculativeDecoder implements AutoCloseable {

    /// Per-generation measurements. `acceptedDrafts[a]` counts verifications that accepted a drafts.
    public static final class Statistics {
        public final long[] acceptedDrafts;
        public long verifications;
        public long outputTokens;
        public long verifyNanos;
        public long catchUpNanos;
        public long recursionNanos;
        public long prefillNanos;
        /// Rejections whose base token was in the draft shortlist, and outside it.
        public long rejectionsInShortlist;
        public long rejectionsOutsideShortlist;

        Statistics(int depth) {
            this.acceptedDrafts = new long[depth + 1];
        }

        public double meanAcceptedDrafts() {
            long sum = 0;
            for (int a = 0; a < this.acceptedDrafts.length; a++) sum += a * this.acceptedDrafts[a];
            return this.verifications == 0 ? 0 : (double) sum / this.verifications;
        }

        @Override
        public String toString() {
            return "verifications " + this.verifications + ", output tokens " + this.outputTokens + ", accepted drafts "
                    + Arrays.toString(this.acceptedDrafts)
                    + ", shortlist rejections in/out " + this.rejectionsInShortlist + "/"
                    + this.rejectionsOutsideShortlist
                    + String.format(
                            java.util.Locale.ROOT,
                            ", mean %.3f, verify %.2f ms, catch-up %.2f ms, recursion %.2f ms (per verification)",
                            meanAcceptedDrafts(),
                            perStep(this.verifyNanos),
                            perStep(this.catchUpNanos),
                            perStep(this.recursionNanos));
        }

        private double perStep(long nanos) {
            return this.verifications == 0 ? 0 : nanos / 1e6 / this.verifications;
        }
    }

    /// Largest MTP catch-up quantum (prompt chunks are split).
    static final int CATCH_UP_ROWS = 128;

    private final EuhedralInferenceRuntime runtime;
    private final QwenExecutionPlan plan;
    private final QwenSequenceState sequence;
    private final IntPredicate endOfGeneration;
    private final int depth;
    private final int prefillChunk;
    private final int hidden;
    private final QwenHostLogits baseLogits;
    private final QwenHostLogits draftLogits;
    private final int[] draftTokens;
    /// Whether each vocabulary token is in the draft head's shortlist.
    private final boolean[] inShortlist;
    private Statistics statistics;

    public QwenSpeculativeDecoder(
            EuhedralInferenceRuntime runtime,
            QwenExecutionPlan plan,
            ExecutionGpu gpu,
            QwenSequenceState sequence,
            IntPredicate endOfGeneration,
            int depth,
            int prefillChunk) {
        this.runtime = Objects.requireNonNull(runtime, "runtime");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.sequence = Objects.requireNonNull(sequence, "sequence");
        this.endOfGeneration = Objects.requireNonNull(endOfGeneration, "endOfGeneration");
        if (!plan.drafts()) throw new IllegalArgumentException("the plan has no MTP draft view");
        if (depth < 1 || depth > 7) throw new IllegalArgumentException("depth must be 1 to 7");
        if (prefillChunk <= 0) throw new IllegalArgumentException("prefillChunk must be positive");
        this.depth = depth;
        this.prefillChunk = prefillChunk;
        this.hidden = plan.weights().config().hiddenSize();
        this.baseLogits = new QwenHostLogits(gpu, plan.weights().config().vocabSize());
        this.draftLogits = new QwenHostLogits(gpu, plan.draftVocabularySize());
        this.baseLogits.selectOnDevice(true);
        this.draftLogits.selectOnDevice(true);
        this.draftTokens = draftTokenIds(gpu, plan);
        this.inShortlist = new boolean[plan.weights().config().vocabSize()];
        for (int token : this.draftTokens)
            if (token >= 0 && token < this.inShortlist.length) this.inShortlist[token] = true;
    }

    /// The draft head's token for each of its rows (`text/draft_head_token_ids`).
    private static int[] draftTokenIds(ExecutionGpu gpu, QwenExecutionPlan plan) {
        TensorHandle ids = plan.weights().runtimeObjects().get("text/draft_head_token_ids");
        if (ids == null) throw new IllegalArgumentException("the model has no draft head token ids");
        int count = Math.toIntExact(ids.shape()[0]);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate((long) count * Integer.BYTES, Integer.BYTES);
            gpu.copyDeviceToHost(host, ids.deviceAddress(), host.byteSize());
            return host.toArray(ValueLayout.JAVA_INT);
        }
    }

    public Statistics statistics() {
        return this.statistics;
    }

    /// Prefills `prompt` and generates up to `maxNewTokens` tokens, exactly as greedy decode would,
    /// reporting each token as it is committed.
    public List<Integer> generate(int[] prompt, int maxNewTokens, IntConsumer onToken)
            throws InterruptedException, ExecutionException {
        return generate(prompt, maxNewTokens, onToken, null);
    }

    /// As [#generate(int[], int, IntConsumer)], reporting prefill chunks, the first token, every
    /// verification step and a final commit-only quantum to `timing` (when not null).
    public List<Integer> generate(int[] prompt, int maxNewTokens, IntConsumer onToken, GenerationTimingListener timing)
            throws InterruptedException, ExecutionException {
        if (prompt.length == 0 || maxNewTokens <= 0) throw new IllegalArgumentException("empty generation");
        this.statistics = new Statistics(this.depth);
        List<Integer> output = new ArrayList<>();
        // Prompt: prefill chunks that seed drafting, each followed by its MTP catch-up.
        int first = -1;
        int[] drafts = null;
        for (int offset = 0; offset < prompt.length; offset += this.prefillChunk) {
            int end = Math.min(prompt.length, offset + this.prefillChunk);
            boolean last = end == prompt.length;
            long started = System.nanoTime();
            execute(new QwenExecutionContext(
                            this.plan,
                            this.sequence,
                            QwenExecutionContext.ExecutionKind.PREFILL,
                            offset,
                            Arrays.copyOfRange(prompt, offset, end),
                            last ? QwenLogitsRequirement.LAST_TOKEN : QwenLogitsRequirement.NONE,
                            last ? this.baseLogits : null)
                    .seedingDraft());
            long executed = System.nanoTime();
            this.statistics.prefillNanos += executed - started;
            if (timing != null) timing.prefillQuantum(started, executed, end - offset);
            int[] next = new int[end - offset];
            System.arraycopy(prompt, offset + 1, next, 0, end - offset - 1);
            if (last) {
                first = this.baseLogits.selectedToken();
                if (timing != null) timing.firstTokenSelected(System.nanoTime(), first);
                next[next.length - 1] = first;
            } else next[next.length - 1] = prompt[end];
            int[] chunkDrafts = catchUp(offset, next, last);
            if (last) drafts = chunkDrafts;
        }
        output.add(first);
        onToken.accept(first);
        this.statistics.outputTokens++;
        if (this.endOfGeneration.test(first)) return output;
        if (maxNewTokens == 1) {
            feedFinal(first, timing);
            return output;
        }
        int current = first;
        while (true) {
            int[] rows = new int[this.depth + 1];
            rows[0] = current;
            System.arraycopy(drafts, 0, rows, 1, this.depth);
            long position = this.sequence.currentTokenPosition();
            var acceptance = new SpeculativeAcceptance(rows, this.endOfGeneration, maxNewTokens - output.size());
            long started = System.nanoTime();
            execute(new QwenExecutionContext(
                            this.plan,
                            this.sequence,
                            QwenExecutionContext.ExecutionKind.VERIFY,
                            position,
                            rows,
                            QwenLogitsRequirement.ALL_TOKENS,
                            this.baseLogits)
                    .withAcceptance(acceptance)
                    .seedingDraft());
            long executed = System.nanoTime();
            this.statistics.verifyNanos += executed - started;
            this.statistics.verifications++;
            this.statistics.acceptedDrafts[acceptance.acceptedDrafts()]++;
            int[] committed = acceptance.outputs();
            int rejected = acceptance.rejectedBaseToken();
            int rejection =
                    rejected < 0 ? -1 : rejected < this.inShortlist.length && this.inShortlist[rejected] ? 0 : 1;
            if (rejection == 0) this.statistics.rejectionsInShortlist++;
            if (rejection == 1) this.statistics.rejectionsOutsideShortlist++;
            if (timing != null)
                timing.speculativeStep(started, executed, committed.length, acceptance.acceptedDrafts(), rejection);
            for (int token : committed) {
                output.add(token);
                onToken.accept(token);
            }
            this.statistics.outputTokens += committed.length;
            current = committed[committed.length - 1];
            if (this.endOfGeneration.test(current)) return output;
            if (output.size() >= maxNewTokens) {
                feedFinal(current, timing);
                return output;
            }
            // Discard the previous step's recursive draft rows, then catch the MTP cache up.
            mtpCache().truncate(Math.toIntExact(position));
            drafts = catchUp(position, committed, true);
        }
    }

    /// MTP catch-up over the base hidden rows just seeded, paired with `tokens` (the token each row
    /// predicted), at MTP positions from `position`. When `draft` is set, its last row drafts d₁ and the
    /// recursive rows draft the rest.
    private int[] catchUp(long position, int[] tokens, boolean draft) throws InterruptedException, ExecutionException {
        AttentionSequenceStates states = states();
        long seeds = states.draftSeedRows(tokens.length, this.hidden);
        long started = System.nanoTime();
        // Pieces of at most CATCH_UP_ROWS rows: the draft view's workspace is retained at the largest
        // quantum it ran, and MTP cache appends are contiguous, so the pieces equal one catch-up.
        for (int first = 0; first < tokens.length; first += CATCH_UP_ROWS) {
            int count = Math.min(CATCH_UP_ROWS, tokens.length - first);
            boolean last = first + count == tokens.length;
            execute(new QwenExecutionContext(
                            this.plan,
                            this.sequence,
                            QwenExecutionContext.ExecutionKind.DRAFT,
                            position + first,
                            Arrays.copyOfRange(tokens, first, first + count),
                            draft && last ? QwenLogitsRequirement.LAST_TOKEN : QwenLogitsRequirement.NONE,
                            draft && last ? this.draftLogits : null)
                    .withDraftSeed(seeds + (long) first * this.hidden * Short.BYTES, count));
        }
        this.statistics.catchUpNanos += System.nanoTime() - started;
        if (!draft) return null;
        int[] drafts = new int[this.depth];
        drafts[0] = this.draftTokens[this.draftLogits.selectedToken()];
        started = System.nanoTime();
        for (int i = 1; i < this.depth; i++) {
            execute(new QwenExecutionContext(
                            this.plan,
                            this.sequence,
                            QwenExecutionContext.ExecutionKind.DRAFT,
                            position + tokens.length + i - 1,
                            new int[] {drafts[i - 1]},
                            QwenLogitsRequirement.LAST_TOKEN,
                            this.draftLogits)
                    .withDraftSeed(states.draftRecursionHidden(this.hidden), 1));
            drafts[i] = this.draftTokens[this.draftLogits.selectedToken()];
        }
        this.statistics.recursionNanos += System.nanoTime() - started;
        return drafts;
    }

    /// Ordinary decode feeds the last allowed token without sampling; so does the decoder, so both leave
    /// the same state.
    private void feedFinal(int token, GenerationTimingListener timing) throws InterruptedException, ExecutionException {
        long started = System.nanoTime();
        execute(new QwenExecutionContext(
                this.plan,
                this.sequence,
                QwenExecutionContext.ExecutionKind.DECODE,
                this.sequence.currentTokenPosition(),
                new int[] {token},
                QwenLogitsRequirement.NONE));
        long executed = System.nanoTime();
        if (timing != null) timing.decodeQuantum(started, executed, executed, false, -1);
    }

    private AttentionSequenceStates states() {
        return (AttentionSequenceStates) this.sequence.kvCacheState();
    }

    private AttentionKvState mtpCache() {
        return states().forLayer(this.plan.weights().config().numHiddenLayers());
    }

    private void execute(QwenExecutionContext context) throws InterruptedException, ExecutionException {
        var outcome = this.runtime.execute(List.of(context)).getFirst();
        if (outcome.status() != QwenExecutionContext.Status.SUCCESS)
            throw new IllegalStateException("speculative quantum " + outcome.status(), outcome.failure());
    }

    @Override
    public void close() {
        this.baseLogits.close();
        this.draftLogits.close();
    }
}

File core/src/main/java/io/euhedral_execution/inference/core/scheduling/QwenExecutionPlan.java:

package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.WeightStaging;
import io.euhedral_execution.inference.core.model_loader.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactDenseFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenMtpWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import io.euhedral_execution.inference.core.scheduling.graph.StageTopology;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/// Immutable operation instructions and dependency edges for the loaded Qwen text model.
/// Model weights are borrowed; sequence state and quantum workspace have separate owners.
public final class QwenExecutionPlan {

    public enum Kind {
        EMBEDDING,
        RMS_NORM,
        RMS_NORM_UNIT_OFFSET,
        Q3_LINEAR,
        Q4_LINEAR,
        Q5_LINEAR,
        BF16_LINEAR,
        GDN_CONTROL,
        GDN_CONVOLUTION,
        GDN_RECURRENCE,
        GDN_GATED_RMS_NORM,
        RESIDUAL_ADD,
        RESIDUAL_RMS_NORM,
        GDN_PROJECT_CONTROL,
        GDN_PROJECTIONS,
        Q3_GATE_UP_SWIGLU,
        Q3_FFN_DOWN,
        FFN_STREAMED,
        SWIGLU,
        ATTENTION_QK_NORM_ROPE,
        ATTENTION_KV_APPEND,
        ATTENTION_CAUSAL,
        /// MTP stem (docs/MTP_CONTRACT.md §2): packs [RMSNorm₁₊w(embedding); RMSNorm₁₊w(seed hidden)] per row.
        MTP_STEM,
        /// Copies a host-backed weight into its staging slot. Its weight's `hostAddress` is the source
        /// and its `deviceAddress` the slot.
        WEIGHT_TRANSFER
    }

    public enum Buffer {
        HIDDEN_STATE,
        MIXER_HIDDEN,
        FINAL_HIDDEN_STATE,
        INPUT_NORMALIZED,
        QK_PROJECTED,
        VALUE_Z_PROJECTED,
        A_PROJECTED,
        B_PROJECTED,
        GDN_ALPHA,
        GDN_BETA,
        GDN_CONVOLVED,
        GDN_RECURRENT,
        GDN_NORMALIZED,
        ATTENTION_QK_NORMALIZED,
        ATTENTION_CONTEXT,
        MIXER_DELTA,
        POST_MIXER_NORMALIZED,
        GATE_UP,
        SWIGLU,
        FFN_DELTA,
        FFN_STAGING,
        FFN_ACCUMULATORS,
        FFN_PARTIALS,
        FINAL_NORMALIZED,
        LOGITS,
        SLICE_PROJECTION,
        /// MTP stem scratch: one normalized half, then the packed [embedding; hidden] rows.
        MTP_NORMED,
        MTP_PACKED
    }

    public enum ElementType {
        BF16,
        FP32
    }

    public record BufferSpec(Buffer buffer, int width, ElementType elementType, int storageSlot) {
        public BufferSpec {
            Objects.requireNonNull(buffer, "buffer");
            Objects.requireNonNull(elementType, "elementType");
            if (width <= 0 || storageSlot < 0) {
                throw new IllegalArgumentException("buffer width must be positive");
            }
        }

        public BufferSpec(Buffer buffer, int width, ElementType elementType) {
            this(buffer, width, elementType, buffer.ordinal());
        }
    }

    public record Instruction(
            int id,
            Kind kind,
            List<Integer> dependencies,
            List<TensorHandle> weights,
            List<Buffer> inputBuffers,
            List<Buffer> outputBuffers,
            int inputWidth,
            int outputWidth,
            int outputBufferIndex,
            int layerIndex) {
        public Instruction {
            Objects.requireNonNull(kind, "kind");
            dependencies = List.copyOf(dependencies);
            weights = weights.stream().map(QwenExecutionPlan::copyHandle).toList();
            inputBuffers = List.copyOf(inputBuffers);
            outputBuffers = List.copyOf(outputBuffers);
            if (id < 0 || inputWidth < 0 || outputWidth < 0 || outputBufferIndex < -1 || layerIndex < -1) {
                throw new IllegalArgumentException("invalid instruction dimensions");
            }
        }

        public Instruction(
                int id,
                Kind kind,
                List<Integer> dependencies,
                List<TensorHandle> weights,
                List<Buffer> inputBuffers,
                List<Buffer> outputBuffers,
                int inputWidth,
                int outputWidth,
                int outputBufferIndex) {
            this(
                    id,
                    kind,
                    dependencies,
                    weights,
                    inputBuffers,
                    outputBuffers,
                    inputWidth,
                    outputWidth,
                    outputBufferIndex,
                    -1);
        }

        public Instruction(int id, Kind kind, List<Integer> dependencies, TensorHandle weight, int outputWidth) {
            this(
                    id,
                    kind,
                    dependencies,
                    weight == null ? List.of() : List.of(weight),
                    List.of(),
                    List.of(),
                    0,
                    outputWidth,
                    -1,
                    -1);
        }

        @Override
        public List<TensorHandle> weights() {
            return this.weights.stream().map(QwenExecutionPlan::copyHandle).toList();
        }

        /// Returns a defensive descriptor copy for inspection outside the instruction executor.
        public TensorHandle weight() {
            return this.weights.isEmpty() ? null : copyHandle(this.weights.getFirst());
        }

        /// Returns a borrowed device address without allocating a defensive `TensorHandle` copy.
        public long weightAddress() {
            return weightAddress(0);
        }

        public long weightAddress(int weightIndex) {
            return this.weights.get(weightIndex).deviceAddress();
        }

        /// Returns payload size without allocating a defensive `TensorHandle` copy.
        public long weightByteSize() {
            return weightByteSize(0);
        }

        public long weightByteSize(int weightIndex) {
            return this.weights.get(weightIndex).byteSize();
        }

        public WeightFormat weightFormat() {
            return weightFormat(0);
        }

        public WeightFormat weightFormat(int weightIndex) {
            return this.weights.get(weightIndex).format();
        }

        public WeightLayout weightLayout() {
            return weightLayout(0);
        }

        public WeightLayout weightLayout(int weightIndex) {
            return this.weights.get(weightIndex).layout();
        }

        public String weightName(int weightIndex) {
            return this.weights.get(weightIndex).name();
        }
    }

    private record PlanData(
            List<Instruction> instructions,
            List<Integer> projectionWidths,
            List<BufferSpec> bufferSpecs,
            boolean firstLayer,
            boolean fullModel) {
        PlanData(
                List<Instruction> instructions,
                List<Integer> projectionWidths,
                List<BufferSpec> bufferSpecs,
                boolean firstLayer) {
            this(instructions, projectionWidths, bufferSpecs, firstLayer, false);
        }
    }

    /// Prefill specializations derived from the full-model reference topology.
    /// `SMALL` covers quanta below the 64-row region tile; `STREAMED` exists only at the
    /// qualified streamed-FFN geometry.
    private enum PrefillView {
        SMALL,
        /// The small topology for decode, with the GDN Q4 and Q5 projections as separate leaf frames.
        DECODE,
        REGIONS,
        STREAMED
    }

    /// Row threshold for the gate/up and down regions (one 64-row prefill tile).
    private static final int REGION_MIN_ROWS = 64;
    /// Exact geometry at which the streamed FFN region is qualified; elsewhere gate/up + down is used.
    /// 64-row quanta run full-width gate/up and split-K down instead (1253 vs 1400 us per layer).
    private static boolean streamedFfnRows(int rows) {
        return rows == 1024;
    }

    /// K splits of the FFN down's FP32 partials (euhedral_ffn_down_splits in qwen_ffn_policy.h).
    private static final int FFN_DOWN_SPLITS = 4;

    private static final int STREAMED_FFN_HIDDEN = 5120;
    private static final int STREAMED_FFN_INTERMEDIATE = 17408;
    /// Two bounded BF16 feature slots of 4096 features each.
    private static final int STREAMED_FFN_STAGING_WIDTH = 2 * 4096;

    private final QwenWeights weights;
    private final List<Instruction> instructions;
    private final List<List<Integer>> successors;
    private final StageTopology stageTopology;
    private final List<Integer> projectionWidths;
    private final List<BufferSpec> bufferSpecs;
    private final boolean firstLayer;
    private final boolean reuseStorage;
    private final QwenExecutionPlan owner;
    private final QwenExecutionPlan smallPrefill;
    private final QwenExecutionPlan decode;
    private final QwenExecutionPlan regionPrefill;
    private final WeightStaging staging;
    /// The MTP draft view; null without loaded MTP weights and draft head.
    private final QwenExecutionPlan mtpDraft;
    private final QwenExecutionPlan streamedPrefill;

    /// Fixed storage lifetime pairs of the region prefill views: each value lives in its owner's storage.
    static final List<Map.Entry<Buffer, Buffer>> REGION_STORAGE = List.of(
            Map.entry(Buffer.FINAL_HIDDEN_STATE, Buffer.HIDDEN_STATE),
            Map.entry(Buffer.POST_MIXER_NORMALIZED, Buffer.INPUT_NORMALIZED),
            Map.entry(Buffer.FFN_DELTA, Buffer.MIXER_DELTA),
            Map.entry(Buffer.SWIGLU, Buffer.VALUE_Z_PROJECTED),
            Map.entry(Buffer.FFN_STAGING, Buffer.VALUE_Z_PROJECTED),
            Map.entry(Buffer.FFN_ACCUMULATORS, Buffer.QK_PROJECTED),
            Map.entry(Buffer.FFN_PARTIALS, Buffer.QK_PROJECTED));

    /// Adds the ordering edges that a single device stream used to provide implicitly. For every
    /// storage (aliased buffers count as one when `reuseStorage`), each reader is ordered after the
    /// last writer, and each writer after the last writer and every reader since it. Edges the
    /// dependencies already imply transitively are not added.
    static int[][] withStorageHazards(List<Instruction> instructions, int[][] dependencies, boolean reuseStorage) {
        int count = instructions.size();
        Map<Buffer, Buffer> owners = new EnumMap<>(Buffer.class);
        if (reuseStorage)
            for (Map.Entry<Buffer, Buffer> pair : REGION_STORAGE) owners.put(pair.getKey(), pair.getValue());
        int storages = Buffer.values().length;
        int[] lastWriter = new int[storages];
        java.util.Arrays.fill(lastWriter, -1);
        List<List<Integer>> readers = new ArrayList<>(storages);
        for (int storage = 0; storage < storages; storage++) readers.add(new ArrayList<>());
        java.util.BitSet[] ancestors = new java.util.BitSet[count];
        int[][] ordered = new int[count][];
        for (int stage = 0; stage < count; stage++) {
            Instruction instruction = instructions.get(stage);
            java.util.LinkedHashSet<Integer> edges = new java.util.LinkedHashSet<>();
            java.util.BitSet reach = new java.util.BitSet(count);
            for (int dependency : dependencies[stage]) {
                edges.add(dependency);
                reach.or(ancestors[dependency]);
                reach.set(dependency);
            }
            java.util.TreeSet<Integer> required = new java.util.TreeSet<>();
            for (Buffer buffer : instruction.inputBuffers()) {
                int storage = owners.getOrDefault(buffer, buffer).ordinal();
                if (lastWriter[storage] >= 0) required.add(lastWriter[storage]);
            }
            for (Buffer buffer : instruction.outputBuffers()) {
                int storage = owners.getOrDefault(buffer, buffer).ordinal();
                if (lastWriter[storage] >= 0) required.add(lastWriter[storage]);
                required.addAll(readers.get(storage));
            }
            // Latest first: an earlier requirement is often already an ancestor of a later one.
            for (int producer : required.descendingSet()) {
                if (producer == stage || reach.get(producer)) continue;
                edges.add(producer);
                reach.or(ancestors[producer]);
                reach.set(producer);
            }
            ancestors[stage] = reach;
            ordered[stage] = edges.stream().mapToInt(Integer::intValue).toArray();
            for (Buffer buffer : instruction.inputBuffers())
                readers.get(owners.getOrDefault(buffer, buffer).ordinal()).add(stage);
            for (Buffer buffer : instruction.outputBuffers()) {
                int storage = owners.getOrDefault(buffer, buffer).ordinal();
                lastWriter[storage] = stage;
                readers.get(storage).clear();
            }
        }
        return ordered;
    }

    /// Builds the production plan. For a complete model, prefill quanta automatically select the
    /// retained region architecture by row count and geometry; decode runs the small topology.
    /// Staged plans for partial weights (embedding-only, layer zero) remain reference-only.
    public QwenExecutionPlan(QwenWeights weights) {
        this(weights, (WeightStaging) null);
    }

    /// Builds the production plan for weights that may be host-backed: every executed view stages
    /// them through `staging`, which must be present when any weight is host-backed.
    public QwenExecutionPlan(QwenWeights weights, WeightStaging staging) {
        this(Objects.requireNonNull(weights, "weights"), planFromLoadedWeights(weights), null, false, staging);
    }

    /// Unfused reference topology for every execution kind. This is a correctness oracle for tests
    /// and is never selected by the runtime.
    public static QwenExecutionPlan reference(QwenWeights weights) {
        Objects.requireNonNull(weights, "weights");
        PlanData data = planFromLoadedWeights(weights);
        return new QwenExecutionPlan(
                weights,
                new PlanData(data.instructions(), data.projectionWidths(), data.bufferSpecs(), data.firstLayer()),
                null,
                false);
    }

    /// The staging ring of this plan family, or null when no weight is host-backed.
    public WeightStaging staging() {
        return this.staging;
    }

    /// Owning plan whose views belong to one family; runners admit any view of their owner.
    QwenExecutionPlan executionOwner() {
        return this.owner;
    }

    /// Selects the qualified topology for a quantum. Any view requalifies through its owner.
    public QwenExecutionPlan forExecution(QwenExecutionContext.ExecutionKind kind, int rows) {
        Objects.requireNonNull(kind, "kind");
        if (rows <= 0) throw new IllegalArgumentException("rows must be positive");
        QwenExecutionPlan family = this.owner;
        if (family.regionPrefill == null) return family;
        // Decode runs its own instance of the small topology: rounded residual add + RMSNorm and the
        // joint GDN A/B projection + control are single region launches, as in short prefill quanta.
        if (kind == QwenExecutionContext.ExecutionKind.DRAFT) {
            if (family.mtpDraft == null)
                throw new IllegalStateException("the model has no loaded MTP layer and draft head");
            return family.mtpDraft;
        }
        if (kind == QwenExecutionContext.ExecutionKind.DECODE || kind == QwenExecutionContext.ExecutionKind.VERIFY)
            return family.decode;
        if (rows < REGION_MIN_ROWS) return family.smallPrefill;
        if (streamedFfnRows(rows) && family.streamedPrefill != null) return family.streamedPrefill;
        return family.regionPrefill;
    }

    static final String DRAFT_HEAD = "text/draft_head";

    /// Whether this plan's family can draft with MTP.
    public boolean drafts() {
        return this.owner.mtpDraft != null;
    }

    /// Rows of the draft head: the draft view's logits width.
    public int draftVocabularySize() {
        TensorHandle head = this.weights.runtimeObjects().get(DRAFT_HEAD);
        if (head == null) throw new IllegalStateException("the model has no draft head");
        return Math.toIntExact(head.shape()[0]);
    }

    /// The MTP draft view (docs/MTP_CONTRACT.md §2): embedding of each row's token, the MTP stem with the
    /// context's seed hidden rows, the input projection, then the MTP layer, built by [#fullModel] as a
    /// one-layer model with `mtp/final_norm` and the draft head, at the layer index after the base
    /// layers (its own attention cache).
    private static PlanData mtpDraft(QwenWeights weights) {
        QwenConfig c = weights.config();
        QwenMtpWeights mtp = weights.mtp();
        TensorHandle head = weights.runtimeObjects().get(DRAFT_HEAD);
        int hidden = c.hiddenSize();
        int mtpLayer = c.numHiddenLayers();
        QwenConfig one = new QwenConfig(
                Math.toIntExact(head.shape()[0]),
                hidden,
                1,
                c.numAttentionHeads(),
                c.numKeyValueHeads(),
                c.attentionHeadDim(),
                c.intermediateSize(),
                c.linearNumKeyHeads(),
                c.linearNumValueHeads(),
                c.linearKeyHeadDim(),
                c.linearValueHeadDim(),
                c.linearConvKernelDim(),
                c.rmsNormEpsilon(),
                c.ropeTheta(),
                c.partialRotaryFactor(),
                c.maxPositionEmbeddings(),
                c.hiddenActivation(),
                new QwenLayerType[] {QwenLayerType.FULL_ATTENTION},
                c.numExperts(),
                c.numExpertsPerToken(),
                c.moeIntermediateSize(),
                c.sharedExpertIntermediateSize(),
                c.tieWordEmbeddings(),
                c.attentionOutputGate(),
                0);
        QwenWeights single = new QwenWeights(
                one, weights.tokenEmbedding(), new QwenLayerWeights[] {mtp.layer()}, mtp.finalNorm(), head, null);
        PlanData layer = fullModel(single, validateEmbedding(weights.tokenEmbedding(), c.vocabSize(), hidden));
        TensorHandle projection = validateQuantized(mtp.projection(), hidden, 2 * hidden, WeightFormat.Q3_G64_FP16);
        TensorHandle embeddingNorm = validateNorm(mtp.embeddingNorm(), hidden);
        TensorHandle hiddenNorm = validateNorm(mtp.hiddenNorm(), hidden);
        List<Instruction> source = layer.instructions();
        Instruction embedding = source.getFirst();
        if (embedding.kind() != Kind.EMBEDDING)
            throw new IllegalStateException("one-layer plan must start with the embedding");
        List<Instruction> nodes = new ArrayList<>();
        nodes.add(new Instruction(
                0,
                Kind.EMBEDDING,
                List.of(),
                embedding.weights,
                List.of(),
                List.of(Buffer.HIDDEN_STATE),
                0,
                hidden,
                -1,
                -1));
        nodes.add(new Instruction(
                1,
                Kind.MTP_STEM,
                List.of(0),
                List.of(embeddingNorm, hiddenNorm),
                List.of(Buffer.HIDDEN_STATE),
                List.of(Buffer.MTP_PACKED),
                hidden,
                2 * hidden,
                -1,
                mtpLayer));
        nodes.add(new Instruction(
                2,
                Kind.Q3_LINEAR,
                List.of(1),
                List.of(projection),
                List.of(Buffer.MTP_PACKED),
                List.of(Buffer.HIDDEN_STATE),
                2 * hidden,
                hidden,
                -1,
                mtpLayer));
        for (int index = 1; index < source.size(); index++) {
            Instruction next = source.get(index);
            nodes.add(new Instruction(
                    index + 2,
                    next.kind(),
                    next.dependencies().stream().map(id -> id + 2).toList(),
                    next.weights,
                    next.inputBuffers(),
                    next.outputBuffers(),
                    next.inputWidth(),
                    next.outputWidth(),
                    next.outputBufferIndex(),
                    next.layerIndex() == 0 ? mtpLayer : next.layerIndex()));
        }
        List<BufferSpec> buffers = new ArrayList<>(layer.bufferSpecs());
        buffers.add(new BufferSpec(Buffer.MTP_NORMED, hidden, ElementType.BF16));
        buffers.add(new BufferSpec(Buffer.MTP_PACKED, 2 * hidden, ElementType.BF16));
        return new PlanData(List.copyOf(nodes), layer.projectionWidths(), List.copyOf(buffers), true, false);
    }

    /// Builds an embedding-only plan for callers that intentionally validate only token lookup.
    public static QwenExecutionPlan embeddingOnly(QwenWeights weights) {
        Objects.requireNonNull(weights, "weights");
        int hiddenSize = weights.config().hiddenSize();
        int vocabularySize = weights.config().vocabSize();
        TensorHandle embedding = validateEmbedding(weights.tokenEmbedding(), vocabularySize, hiddenSize);
        return new QwenExecutionPlan(weights, embeddingOnly(embedding, hiddenSize));
    }

    /// Builds a dependency-graph prefix ending after the requested real layer, for staged validation.
    public static QwenExecutionPlan prefix(QwenWeights weights, int layerCount) {
        Objects.requireNonNull(weights, "weights");
        if (layerCount <= 0 || layerCount > weights.config().numHiddenLayers()) {
            throw new IllegalArgumentException("layerCount must select a non-empty model prefix");
        }
        QwenExecutionPlan fullPlan = new QwenExecutionPlan(weights);
        int terminalId = -1;
        for (Instruction instruction : fullPlan.instructions) {
            if (instruction.layerIndex() == layerCount - 1
                    && instruction.kind() == Kind.RESIDUAL_ADD
                    && instruction.outputBuffers().contains(Buffer.FINAL_HIDDEN_STATE)) {
                terminalId = instruction.id();
            }
        }
        if (terminalId < 0) throw new IllegalArgumentException("requested layer prefix has no final residual");
        List<Instruction> instructions = List.copyOf(fullPlan.instructions.subList(0, terminalId + 1));
        boolean[] usedBuffers = new boolean[Buffer.values().length];
        for (Instruction instruction : instructions) {
            for (Buffer buffer : instruction.inputBuffers()) usedBuffers[buffer.ordinal()] = true;
            for (Buffer buffer : instruction.outputBuffers()) usedBuffers[buffer.ordinal()] = true;
        }
        List<BufferSpec> buffers = fullPlan.bufferSpecs.stream()
                .filter(spec -> usedBuffers[spec.buffer().ordinal()])
                .toList();
        return new QwenExecutionPlan(weights, new PlanData(instructions, List.of(), buffers, true));
    }

    /// A standalone operator slice retained for low-level operation validation.
    public QwenExecutionPlan(QwenWeights weights, TensorHandle normWeight, List<TensorHandle> projections) {
        this(Objects.requireNonNull(weights, "weights"), operatorSlice(weights, normWeight, projections));
    }

    private QwenExecutionPlan(QwenWeights weights, PlanData data) {
        this(weights, data, null, false);
    }

    private QwenExecutionPlan(QwenWeights weights, PlanData data, QwenExecutionPlan owner, boolean reuseStorage) {
        this(weights, data, owner, reuseStorage, null);
    }

    private QwenExecutionPlan(
            QwenWeights weights, PlanData data, QwenExecutionPlan owner, boolean reuseStorage, WeightStaging staging) {
        this.owner = owner == null ? this : owner;
        this.staging = owner == null ? staging : owner.staging;
        this.weights = weights;
        this.instructions = List.copyOf(data.instructions());
        this.projectionWidths = List.copyOf(data.projectionWidths());
        this.bufferSpecs = List.copyOf(data.bufferSpecs());
        this.firstLayer = data.firstLayer();
        this.reuseStorage = reuseStorage;
        List<List<Integer>> edges = new ArrayList<>();
        for (int index = 0; index < this.instructions.size(); index++) {
            edges.add(new ArrayList<>());
        }
        for (Instruction instruction : this.instructions) {
            for (int dependency : instruction.dependencies()) {
                if (dependency < 0 || dependency >= instruction.id()) {
                    throw new IllegalArgumentException("instruction dependencies must point to earlier work");
                }
                edges.get(dependency).add(instruction.id());
            }
        }
        this.successors = edges.stream().map(List::copyOf).toList();
        int[][] dependencies = new int[this.instructions.size()][];
        for (Instruction instruction : this.instructions) {
            dependencies[instruction.id()] = instruction.dependencies().stream()
                    .mapToInt(Integer::intValue)
                    .toArray();
        }
        // Stages may run on different device lanes, so the topology also orders every pair of stages
        // that touch the same storage (one of them writing) which the data dependencies leave unordered.
        this.stageTopology = StageTopology.submitted(withStorageHazards(this.instructions, dependencies, reuseStorage));
        if (owner != null || !data.fullModel()) {
            this.smallPrefill = null;
            this.decode = null;
            this.regionPrefill = null;
            this.streamedPrefill = null;
            this.mtpDraft = null;
            return;
        }
        // Drafting needs the MTP attention split like a base layer's, which only a speculative load does.
        this.mtpDraft = weights.mtp() != null
                        && weights.mtp().layer().mixer()
                                instanceof
                                io.euhedral_execution.inference.core.model_loader.layer_weights
                                        .QwenCompactAttentionWeights
                        && weights.runtimeObjects().containsKey(DRAFT_HEAD)
                ? new QwenExecutionPlan(weights, staged(mtpDraft(weights), this.staging), this, false)
                : null;
        QwenConfig config = weights.config();
        this.smallPrefill = prefillPlan(weights, data, PrefillView.SMALL, this);
        this.decode = prefillPlan(weights, data, PrefillView.DECODE, this);
        this.regionPrefill = prefillPlan(weights, data, PrefillView.REGIONS, this);
        this.streamedPrefill =
                config.hiddenSize() == STREAMED_FFN_HIDDEN && config.intermediateSize() == STREAMED_FFN_INTERMEDIATE
                        ? prefillPlan(weights, data, PrefillView.STREAMED, this)
                        : null;
    }

    private static QwenExecutionPlan prefillPlan(
            QwenWeights weights, PlanData data, PrefillView view, QwenExecutionPlan owner) {
        PlanData selected = prefillView(data, view);
        // The named lifetime pairs are qualified only for the region views with a fused FFN, not for
        // a shape that falls back to ordinary FFN.
        boolean regions = view == PrefillView.REGIONS || view == PrefillView.STREAMED;
        boolean hasFusedFfn = selected.instructions().stream()
                .anyMatch(i -> i.kind() == Kind.Q3_GATE_UP_SWIGLU || i.kind() == Kind.FFN_STREAMED);
        return new QwenExecutionPlan(weights, staged(selected, owner.staging), owner, regions && hasFusedFfn);
    }

    /// Whether this view copies host-backed weights into the staging ring.
    public boolean stagesWeights() {
        return this.instructions.stream().anyMatch(i -> i.kind() == Kind.WEIGHT_TRANSFER);
    }

    /// Rewrites a view so that each use of a host-backed weight reads a staging slot. Use `u` takes slot
    /// `u mod slots`; its transfer follows the use that last read the slot (use `u - slots`), or starts
    /// the quantum, and precedes the consumer. Transfers are leaves apart from those two edges, so each
    /// copy starts as soon as its slot is free.
    static PlanData staged(PlanData data, WeightStaging staging) {
        List<Instruction> source = data.instructions();
        boolean hostBacked = source.stream().flatMap(i -> i.weights.stream()).anyMatch(TensorHandle::hostBacked);
        if (!hostBacked) return data;
        if (staging == null) throw new IllegalArgumentException("host-backed weights need a staging ring");
        int slots = staging.slots();
        for (Instruction instruction : source) {
            long uses = instruction.weights.stream()
                    .filter(TensorHandle::hostBacked)
                    .count();
            if (uses > slots) throw new IllegalArgumentException("an instruction stages more weights than slots");
            for (TensorHandle weight : instruction.weights) {
                if (weight.hostBacked() && weight.byteSize() > staging.slotBytes())
                    throw new IllegalArgumentException("host-backed weight exceeds a staging slot: " + weight.name());
            }
        }
        // Each use's weight, in order, and the transfers still to emit once a slot frees.
        List<TensorHandle> useWeights = new ArrayList<>();
        List<Integer> useLayers = new ArrayList<>();
        for (Instruction instruction : source) {
            for (TensorHandle weight : instruction.weights) {
                if (weight.hostBacked()) {
                    useWeights.add(weight);
                    useLayers.add(instruction.layerIndex());
                }
            }
        }
        List<Instruction> result = new ArrayList<>();
        int[] remapped = new int[source.size()];
        int[] transferIds = new int[useWeights.size()];
        java.util.function.IntConsumer emitTransfer = use -> {
            TensorHandle weight = useWeights.get(use);
            TensorHandle transfer = new TensorHandle(
                    weight.name(),
                    weight.shape(),
                    weight.dataType(),
                    weight.format(),
                    weight.layout(),
                    staging.slotAddress(use % slots),
                    weight.byteSize(),
                    weight.hostAddress());
            List<Integer> dependencies = use < slots ? List.of() : List.of(consumerOf(use - slots, source, remapped));
            transferIds[use] = result.size();
            result.add(new Instruction(
                    result.size(),
                    Kind.WEIGHT_TRANSFER,
                    dependencies,
                    List.of(transfer),
                    List.of(),
                    List.of(),
                    0,
                    0,
                    -1,
                    useLayers.get(use)));
        };
        for (int use = 0; use < Math.min(slots, useWeights.size()); use++) emitTransfer.accept(use);
        int use = 0;
        for (Instruction instruction : source) {
            List<Integer> dependencies = new ArrayList<>(remapDependencies(instruction.dependencies(), remapped));
            List<TensorHandle> weights = new ArrayList<>();
            int first = use;
            for (TensorHandle weight : instruction.weights) {
                if (!weight.hostBacked()) {
                    weights.add(weight);
                    continue;
                }
                dependencies.add(transferIds[use]);
                weights.add(new TensorHandle(
                        weight.name(),
                        weight.shape(),
                        weight.dataType(),
                        weight.format(),
                        weight.layout(),
                        staging.slotAddress(use % slots),
                        weight.byteSize()));
                use++;
            }
            remapped[instruction.id()] = result.size();
            result.add(new Instruction(
                    result.size(),
                    instruction.kind(),
                    dependencies.stream().distinct().sorted().toList(),
                    weights,
                    instruction.inputBuffers(),
                    instruction.outputBuffers(),
                    instruction.inputWidth(),
                    instruction.outputWidth(),
                    instruction.outputBufferIndex(),
                    instruction.layerIndex()));
            for (int staged = first; staged < use; staged++) {
                if (staged + slots < useWeights.size()) emitTransfer.accept(staged + slots);
            }
        }
        return new PlanData(
                List.copyOf(result), data.projectionWidths(), data.bufferSpecs(), data.firstLayer(), data.fullModel());
    }

    /// The remapped id of the instruction that reads use `use`; it is always already emitted.
    private static int consumerOf(int use, List<Instruction> source, int[] remapped) {
        int seen = 0;
        for (Instruction instruction : source) {
            for (TensorHandle weight : instruction.weights) {
                if (weight.hostBacked() && seen++ == use) return remapped[instruction.id()];
            }
        }
        throw new IllegalStateException("no consumer for staged use " + use);
    }

    /// Rewrites the reference layer DAG into the retained prefill regions:
    /// A residual+RMSNorm, D early joint GDN projection/control, and, from 64 rows, B gate/up+SwiGLU
    /// (or C streamed FFN). The attention producers stay four leaf frames (Q projection, QK norm and
    /// RoPE, KV projection, cache append), so the KV branch runs on its own lane beside the Q branch.
    private static PlanData prefillView(PlanData data, PrefillView view) {
        boolean regions = view != PrefillView.SMALL && view != PrefillView.DECODE;
        boolean streamed = view == PrefillView.STREAMED;
        List<Instruction> source = earlyControlOrder(data.instructions());
        List<Instruction> result = new ArrayList<>();
        int[] remapped = new int[source.size()];
        java.util.Arrays.fill(remapped, -1);
        for (int index = 0; index < source.size(); index++) {
            Instruction first = source.get(index);
            Instruction next = index + 1 < source.size() ? source.get(index + 1) : null;
            if (first.kind() == Kind.Q4_LINEAR
                    && next != null
                    && next.kind() == Kind.Q5_LINEAR
                    && index + 2 < source.size()
                    && source.get(index + 2).kind() == Kind.GDN_CONVOLUTION
                    && view != PrefillView.DECODE) {
                if (!next.dependencies().equals(first.dependencies())
                        || !first.inputBuffers().equals(next.inputBuffers())
                        || !first.outputBuffers().equals(List.of(Buffer.QK_PROJECTED))
                        || !next.outputBuffers().equals(List.of(Buffer.VALUE_Z_PROJECTED)))
                    throw new IllegalStateException("unexpected GDN projection topology");
                remapped[first.id()] = result.size();
                remapped[next.id()] = result.size();
                result.add(new Instruction(
                        result.size(),
                        Kind.GDN_PROJECTIONS,
                        remapDependencies(first.dependencies(), remapped),
                        List.of(first.weight(), next.weight()),
                        first.inputBuffers(),
                        List.of(Buffer.QK_PROJECTED, Buffer.VALUE_Z_PROJECTED),
                        first.inputWidth(),
                        first.outputWidth(),
                        -1,
                        first.layerIndex()));
                index++;
                continue;
            }
            if (first.kind() == Kind.RESIDUAL_ADD
                    && next != null
                    && next.kind() == Kind.RMS_NORM_UNIT_OFFSET
                    && !next.outputBuffers().contains(Buffer.FINAL_NORMALIZED)
                    && next.dependencies().equals(List.of(first.id()))
                    && next.inputBuffers().equals(first.outputBuffers())) {
                remapped[first.id()] = result.size();
                remapped[next.id()] = result.size();
                result.add(new Instruction(
                        result.size(),
                        Kind.RESIDUAL_RMS_NORM,
                        remapDependencies(first.dependencies(), remapped),
                        next.weights(),
                        first.inputBuffers(),
                        List.of(
                                first.outputBuffers().getFirst(),
                                next.outputBuffers().getFirst()),
                        first.inputWidth(),
                        first.outputWidth(),
                        -1,
                        first.layerIndex()));
                index++;
                continue;
            }
            if (streamed
                    && first.kind() == Kind.Q3_LINEAR
                    && first.weight().format() == WeightFormat.Q3_G64_FP16
                    && next != null
                    && first.outputBuffers().equals(List.of(Buffer.GATE_UP))
                    && next.kind() == Kind.SWIGLU
                    && first.inputWidth() == STREAMED_FFN_HIDDEN
                    && first.outputWidth() == 2 * STREAMED_FFN_INTERMEDIATE
                    && index + 2 < source.size()) {
                Instruction down = source.get(index + 2);
                if (down.kind() != Kind.Q3_LINEAR
                        || !down.dependencies().equals(List.of(next.id()))
                        || !next.dependencies().equals(List.of(first.id()))
                        || !down.outputBuffers().equals(List.of(Buffer.FFN_DELTA)))
                    throw new IllegalStateException("unexpected streamed FFN topology");
                remapped[first.id()] = result.size();
                remapped[next.id()] = result.size();
                remapped[down.id()] = result.size();
                result.add(new Instruction(
                        result.size(),
                        Kind.FFN_STREAMED,
                        remapDependencies(first.dependencies(), remapped),
                        List.of(first.weight(), down.weight()),
                        first.inputBuffers(),
                        List.of(Buffer.FFN_DELTA, Buffer.FFN_STAGING, Buffer.FFN_ACCUMULATORS),
                        first.inputWidth(),
                        down.outputWidth(),
                        -1,
                        first.layerIndex()));
                index += 2;
                continue;
            }
            if (regions
                    && first.kind() == Kind.Q3_LINEAR
                    && next != null
                    && first.outputBuffers().equals(List.of(Buffer.GATE_UP))
                    && next.kind() == Kind.SWIGLU
                    && next.dependencies().equals(List.of(first.id()))
                    && first.inputWidth() % 128 == 0
                    && first.outputWidth() % 32 == 0) {
                remapped[first.id()] = result.size();
                remapped[next.id()] = result.size();
                result.add(new Instruction(
                        result.size(),
                        Kind.Q3_GATE_UP_SWIGLU,
                        remapDependencies(first.dependencies(), remapped),
                        first.weights(),
                        first.inputBuffers(),
                        next.outputBuffers(),
                        first.inputWidth(),
                        first.outputWidth(),
                        -1,
                        first.layerIndex()));
                index++;
                continue;
            }
            if (first.kind() == Kind.BF16_LINEAR
                    && first.outputBuffers().equals(List.of(Buffer.A_PROJECTED))
                    && index + 2 < source.size()) {
                Instruction b = source.get(index + 1);
                Instruction last = source.get(index + 2);
                if (b.kind() != Kind.BF16_LINEAR
                        || !b.outputBuffers().equals(List.of(Buffer.B_PROJECTED))
                        || last.kind() != Kind.GDN_CONTROL
                        || !first.dependencies().equals(b.dependencies())
                        || !first.inputBuffers().equals(b.inputBuffers())
                        || !last.dependencies().equals(List.of(first.id(), b.id()))) {
                    throw new IllegalStateException("GDN projection/control region has unexpected topology");
                }
                remapped[first.id()] = result.size();
                remapped[b.id()] = result.size();
                remapped[last.id()] = result.size();
                List<TensorHandle> regionWeights = new ArrayList<>(first.weights());
                regionWeights.addAll(b.weights());
                regionWeights.addAll(last.weights());
                result.add(new Instruction(
                        result.size(),
                        Kind.GDN_PROJECT_CONTROL,
                        remapDependencies(first.dependencies(), remapped),
                        regionWeights,
                        first.inputBuffers(),
                        last.outputBuffers(),
                        first.inputWidth(),
                        first.outputWidth(),
                        -1,
                        first.layerIndex()));
                index += 2;
                continue;
            }
            List<Integer> dependencies = remapDependencies(first.dependencies(), remapped);
            remapped[first.id()] = result.size();
            boolean ffnDown = regions
                    && first.kind() == Kind.Q3_LINEAR
                    && first.inputBuffers().equals(List.of(Buffer.SWIGLU))
                    && first.outputBuffers().equals(List.of(Buffer.FFN_DELTA));
            boolean splitDown = ffnDown
                    && first.inputWidth() == STREAMED_FFN_INTERMEDIATE
                    && first.outputWidth() == STREAMED_FFN_HIDDEN;
            result.add(new Instruction(
                    result.size(),
                    ffnDown ? Kind.Q3_FFN_DOWN : first.kind(),
                    dependencies,
                    first.weights(),
                    first.inputBuffers(),
                    splitDown ? List.of(Buffer.FFN_DELTA, Buffer.FFN_PARTIALS) : first.outputBuffers(),
                    first.inputWidth(),
                    first.outputWidth(),
                    first.outputBufferIndex(),
                    first.layerIndex()));
        }
        boolean needsGateUp = result.stream().anyMatch(i -> i.outputBuffers().contains(Buffer.GATE_UP));
        boolean needsSwiGlu = result.stream().anyMatch(i -> i.outputBuffers().contains(Buffer.SWIGLU));
        List<BufferSpec> buffers = new ArrayList<>(data.bufferSpecs().stream()
                .filter(spec -> needsSwiGlu || spec.buffer() != Buffer.SWIGLU)
                .filter(spec -> needsGateUp || spec.buffer() != Buffer.GATE_UP)
                .filter(spec -> spec.buffer() != Buffer.A_PROJECTED && spec.buffer() != Buffer.B_PROJECTED)
                .toList());
        if (result.stream().anyMatch(i -> i.outputBuffers().contains(Buffer.FFN_PARTIALS))) {
            buffers.add(spec(Buffer.FFN_PARTIALS, FFN_DOWN_SPLITS * STREAMED_FFN_HIDDEN, ElementType.FP32));
        }
        if (result.stream().anyMatch(i -> i.kind() == Kind.FFN_STREAMED)) {
            buffers.add(spec(Buffer.FFN_STAGING, STREAMED_FFN_STAGING_WIDTH, ElementType.BF16));
            buffers.add(spec(Buffer.FFN_ACCUMULATORS, STREAMED_FFN_HIDDEN, ElementType.FP32));
        }
        return new PlanData(result, data.projectionWidths(), buffers, data.firstLayer());
    }

    private static List<Instruction> earlyControlOrder(List<Instruction> instructions) {
        List<Instruction> result = new ArrayList<>(instructions.size());
        for (int index = 0; index < instructions.size(); index++) {
            if (index + 4 < instructions.size()
                    && instructions.get(index).kind() == Kind.Q4_LINEAR
                    && instructions.get(index + 1).kind() == Kind.Q5_LINEAR
                    && instructions.get(index + 2).kind() == Kind.BF16_LINEAR
                    && instructions.get(index + 3).kind() == Kind.BF16_LINEAR
                    && instructions.get(index + 4).kind() == Kind.GDN_CONTROL) {
                result.addAll(instructions.subList(index + 2, index + 5));
                result.add(instructions.get(index));
                result.add(instructions.get(index + 1));
                index += 4;
            } else result.add(instructions.get(index));
        }
        return result;
    }

    private static List<Integer> remapDependencies(List<Integer> dependencies, int[] remapped) {
        return dependencies.stream()
                .map(id -> {
                    int mapped = remapped[id];
                    if (mapped < 0) throw new IllegalStateException("region depends on unpublished work");
                    return mapped;
                })
                .distinct()
                .toList();
    }

    public List<Instruction> instructions() {
        return this.instructions;
    }

    /// The stage DAG a reusable frame graph instantiates. Every instruction dependency is a submission
    /// edge: the consumer's operation is ordered after the producer's by the quantum's stream, so it
    /// may launch as soon as the producer's launch succeeded. A quantum's only device-completion
    /// boundary is its retirement, where sequence state is published and quantum storage released.
    public StageTopology stageTopology() {
        return this.stageTopology;
    }

    public List<Integer> successors(int instructionId) {
        return this.successors.get(instructionId);
    }

    List<Integer> projectionWidths() {
        return this.projectionWidths;
    }

    public List<BufferSpec> bufferSpecs() {
        return this.bufferSpecs;
    }

    /// The views this plan owns besides its own topology; empty for a view or a reference plan.
    List<QwenExecutionPlan> executionVariants() {
        if (this.owner != this || this.regionPrefill == null) return List.of();
        return this.streamedPrefill == null
                ? List.of(this.decode, this.smallPrefill, this.regionPrefill)
                : List.of(this.decode, this.smallPrefill, this.regionPrefill, this.streamedPrefill);
    }

    boolean reusePrefillStorage() {
        return this.reuseStorage;
    }

    public boolean hasFirstLayer() {
        return this.firstLayer;
    }

    public int bufferWidth(Buffer buffer) {
        return this.bufferSpecs.stream()
                .filter(spec -> spec.buffer() == buffer)
                .mapToInt(BufferSpec::width)
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("buffer is not in this plan: " + buffer));
    }

    public ElementType bufferElementType(Buffer buffer) {
        return this.bufferSpecs.stream()
                .filter(spec -> spec.buffer() == buffer)
                .map(BufferSpec::elementType)
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("buffer is not in this plan: " + buffer));
    }

    public QwenWeights weights() {
        return this.weights;
    }

    private static PlanData planFromLoadedWeights(QwenWeights weights) {
        Objects.requireNonNull(weights.config(), "config");
        int hiddenSize = weights.config().hiddenSize();
        int vocabularySize = weights.config().vocabSize();
        if (hiddenSize <= 0 || vocabularySize <= 0) {
            throw new IllegalArgumentException("invalid model dimensions");
        }
        TensorHandle embedding = validateEmbedding(weights.tokenEmbedding(), vocabularySize, hiddenSize);
        QwenLayerWeights[] layers = weights.layers();
        if (layers == null || layers.length == 0) {
            return embeddingOnly(embedding, hiddenSize);
        }
        if (layers.length == 1 && layers[0] != null && layers[0].index() == 0) {
            return firstLayer(weights, embedding);
        }
        if (layers.length != weights.config().numHiddenLayers()) {
            throw new IllegalArgumentException("loaded weights do not contain every declared text layer");
        }
        return fullModel(weights, embedding);
    }

    private static PlanData embeddingOnly(TensorHandle embedding, int hiddenSize) {
        return new PlanData(
                List.of(node(
                        0,
                        Kind.EMBEDDING,
                        List.of(),
                        List.of(embedding),
                        List.of(),
                        List.of(Buffer.HIDDEN_STATE),
                        0,
                        hiddenSize)),
                List.of(),
                List.of(),
                false);
    }

    private static PlanData firstLayer(QwenWeights weights, TensorHandle embedding) {
        QwenConfig config = weights.config();
        QwenLayerWeights[] layers = weights.layers();
        if (layers[0] == null || layers[0].index() != 0) {
            throw new IllegalArgumentException("loaded weights do not contain actual text layer zero");
        }
        QwenLayerType[] layerTypes = config.layerTypes();
        if (layerTypes == null || layerTypes.length == 0 || layerTypes[0] != QwenLayerType.GATED_DELTA_NET) {
            throw new IllegalArgumentException("layer zero is not the loaded GDN topology");
        }
        if (!(layers[0].mixer() instanceof QwenCompactGatedDeltaNetWeights gdn)
                || !(layers[0].ffn() instanceof QwenCompactDenseFfnWeights ffn)) {
            throw new IllegalArgumentException("layer zero must use compact GDN and dense FFN weights");
        }
        int hidden = config.hiddenSize();
        int intermediate = config.intermediateSize();
        int keyHeads = config.linearNumKeyHeads();
        int valueHeads = config.linearNumValueHeads();
        int keyWidth = config.linearKeyHeadDim();
        int valueWidthPerHead = config.linearValueHeadDim();
        int kernelWidth = config.linearConvKernelDim();
        if (hidden <= 0
                || intermediate <= 0
                || keyHeads <= 0
                || valueHeads <= 0
                || valueHeads % keyHeads != 0
                || keyWidth != 128
                || valueWidthPerHead != 128
                || kernelWidth < 2
                || !Double.isFinite(config.rmsNormEpsilon())
                || config.rmsNormEpsilon() <= 0) {
            throw new IllegalArgumentException("unsupported layer-zero GDN geometry");
        }
        int keyProjected = Math.multiplyExact(Math.multiplyExact(2, keyHeads), keyWidth);
        int valueProjected = Math.multiplyExact(valueHeads, valueWidthPerHead);
        int valueZProjected = Math.multiplyExact(2, valueProjected);
        int convolutionChannels = Math.addExact(keyProjected, valueProjected);

        TensorHandle inputNorm = validateNorm(layers[0].inputNorm(), hidden);
        TensorHandle postNorm = validateNorm(layers[0].postAttentionNorm(), hidden);
        TensorHandle aLog = validateFp32Vector(gdn.aLog(), valueHeads);
        TensorHandle dtBias = validateFp32Vector(gdn.dtBias(), valueHeads);
        TensorHandle convolution = validateBf16Matrix(gdn.convolution(), kernelWidth, convolutionChannels);
        TensorHandle aProjection = validateBf16Matrix(gdn.aProjection(), valueHeads, hidden);
        TensorHandle bProjection = validateBf16Matrix(gdn.bProjection(), valueHeads, hidden);
        TensorHandle queryKey = validateQuantized(gdn.queryKey(), keyProjected, hidden, WeightFormat.Q4_G64_FP16);
        TensorHandle valueZ = validateQuantized(gdn.valueZ(), valueZProjected, hidden, WeightFormat.Q5_G64_FP16);
        TensorHandle gdnNorm = validateNorm(gdn.norm(), valueWidthPerHead);
        TensorHandle gdnOutput = validateQuantized(gdn.output(), hidden, valueProjected, WeightFormat.Q3_G64_FP16);
        TensorHandle gateUp =
                validateQuantized(ffn.gateUp(), Math.multiplyExact(2, intermediate), hidden, WeightFormat.Q3_G64_FP16);
        TensorHandle down = validateQuantized(ffn.down(), hidden, intermediate, WeightFormat.Q3_G64_FP16);

        List<Instruction> nodes = new ArrayList<>();
        nodes.add(node(
                0, Kind.EMBEDDING, List.of(), List.of(embedding), List.of(), List.of(Buffer.HIDDEN_STATE), 0, hidden));
        nodes.add(node(
                1,
                Kind.RMS_NORM_UNIT_OFFSET,
                List.of(0),
                List.of(inputNorm),
                List.of(Buffer.HIDDEN_STATE),
                List.of(Buffer.INPUT_NORMALIZED),
                hidden,
                hidden));
        nodes.add(node(
                2,
                Kind.Q4_LINEAR,
                List.of(1),
                List.of(queryKey),
                List.of(Buffer.INPUT_NORMALIZED),
                List.of(Buffer.QK_PROJECTED),
                hidden,
                keyProjected));
        nodes.add(node(
                3,
                Kind.Q5_LINEAR,
                List.of(1),
                List.of(valueZ),
                List.of(Buffer.INPUT_NORMALIZED),
                List.of(Buffer.VALUE_Z_PROJECTED),
                hidden,
                valueZProjected));
        nodes.add(node(
                4,
                Kind.BF16_LINEAR,
                List.of(1),
                List.of(aProjection),
                List.of(Buffer.INPUT_NORMALIZED),
                List.of(Buffer.A_PROJECTED),
                hidden,
                valueHeads));
        nodes.add(node(
                5,
                Kind.BF16_LINEAR,
                List.of(1),
                List.of(bProjection),
                List.of(Buffer.INPUT_NORMALIZED),
                List.of(Buffer.B_PROJECTED),
                hidden,
                valueHeads));
        nodes.add(node(
                6,
                Kind.GDN_CONTROL,
                List.of(4, 5),
                List.of(aLog, dtBias),
                List.of(Buffer.A_PROJECTED, Buffer.B_PROJECTED),
                List.of(Buffer.GDN_ALPHA, Buffer.GDN_BETA),
                valueHeads,
                valueHeads));
        nodes.add(node(
                7,
                Kind.GDN_CONVOLUTION,
                List.of(2, 3),
                List.of(convolution),
                List.of(Buffer.QK_PROJECTED, Buffer.VALUE_Z_PROJECTED),
                List.of(Buffer.GDN_CONVOLVED),
                convolutionChannels,
                convolutionChannels));
        nodes.add(node(
                8,
                Kind.GDN_RECURRENCE,
                List.of(6, 7),
                List.of(),
                List.of(Buffer.GDN_CONVOLVED, Buffer.GDN_ALPHA, Buffer.GDN_BETA),
                List.of(Buffer.GDN_RECURRENT),
                convolutionChannels,
                valueProjected));
        nodes.add(node(
                9,
                Kind.GDN_GATED_RMS_NORM,
                List.of(8, 3),
                List.of(gdnNorm),
                List.of(Buffer.GDN_RECURRENT, Buffer.VALUE_Z_PROJECTED),
                List.of(Buffer.GDN_NORMALIZED),
                valueProjected,
                valueProjected));
        nodes.add(node(
                10,
                Kind.Q3_LINEAR,
                List.of(9),
                List.of(gdnOutput),
                List.of(Buffer.GDN_NORMALIZED),
                List.of(Buffer.MIXER_DELTA),
                valueProjected,
                hidden));
        nodes.add(node(
                11,
                Kind.RESIDUAL_ADD,
                List.of(0, 10),
                List.of(),
                List.of(Buffer.HIDDEN_STATE, Buffer.MIXER_DELTA),
                List.of(Buffer.MIXER_HIDDEN),
                hidden,
                hidden));
        nodes.add(node(
                12,
                Kind.RMS_NORM_UNIT_OFFSET,
                List.of(11),
                List.of(postNorm),
                List.of(Buffer.MIXER_HIDDEN),
                List.of(Buffer.POST_MIXER_NORMALIZED),
                hidden,
                hidden));
        nodes.add(node(
                13,
                Kind.Q3_LINEAR,
                List.of(12),
                List.of(gateUp),
                List.of(Buffer.POST_MIXER_NORMALIZED),
                List.of(Buffer.GATE_UP),
                hidden,
                2 * intermediate));
        nodes.add(node(
                14,
                Kind.SWIGLU,
                List.of(13),
                List.of(),
                List.of(Buffer.GATE_UP),
                List.of(Buffer.SWIGLU),
                2 * intermediate,
                intermediate));
        nodes.add(node(
                15,
                Kind.Q3_LINEAR,
                List.of(14),
                List.of(down),
                List.of(Buffer.SWIGLU),
                List.of(Buffer.FFN_DELTA),
                intermediate,
                hidden));
        nodes.add(node(
                16,
                Kind.RESIDUAL_ADD,
                List.of(11, 15),
                List.of(),
                List.of(Buffer.MIXER_HIDDEN, Buffer.FFN_DELTA),
                List.of(Buffer.FINAL_HIDDEN_STATE),
                hidden,
                hidden));

        List<BufferSpec> buffers = List.of(
                spec(Buffer.HIDDEN_STATE, hidden, ElementType.BF16),
                spec(Buffer.MIXER_HIDDEN, hidden, ElementType.BF16),
                spec(Buffer.FINAL_HIDDEN_STATE, hidden, ElementType.BF16),
                spec(Buffer.INPUT_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.QK_PROJECTED, keyProjected, ElementType.BF16),
                spec(Buffer.VALUE_Z_PROJECTED, valueZProjected, ElementType.BF16),
                spec(Buffer.A_PROJECTED, valueHeads, ElementType.FP32),
                spec(Buffer.B_PROJECTED, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_ALPHA, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_BETA, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_CONVOLVED, convolutionChannels, ElementType.BF16),
                spec(Buffer.GDN_RECURRENT, valueProjected, ElementType.BF16),
                spec(Buffer.GDN_NORMALIZED, valueProjected, ElementType.BF16),
                spec(Buffer.MIXER_DELTA, hidden, ElementType.BF16),
                spec(Buffer.POST_MIXER_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.GATE_UP, 2 * intermediate, ElementType.BF16),
                spec(Buffer.SWIGLU, intermediate, ElementType.BF16),
                spec(Buffer.FFN_DELTA, hidden, ElementType.BF16));
        return new PlanData(List.copyOf(nodes), List.of(), buffers, true);
    }

    private static PlanData fullModel(QwenWeights weights, TensorHandle embedding) {
        QwenConfig config = weights.config();
        QwenLayerType[] layerTypes = config.layerTypes();
        QwenLayerWeights[] layers = weights.layers();
        int hidden = config.hiddenSize();
        int intermediate = config.intermediateSize();
        int queryHeads = config.numAttentionHeads();
        int keyValueHeads = config.numKeyValueHeads();
        int attentionHeadDim = config.attentionHeadDim();
        int rotaryDim = (int) Math.round(attentionHeadDim * config.partialRotaryFactor());
        int keyHeads = config.linearNumKeyHeads();
        int valueHeads = config.linearNumValueHeads();
        int keyHeadDim = config.linearKeyHeadDim();
        int valueHeadDim = config.linearValueHeadDim();
        int kernelWidth = config.linearConvKernelDim();
        if (layerTypes == null
                || layerTypes.length != config.numHiddenLayers()
                || layers.length != config.numHiddenLayers()
                || hidden <= 0
                || hidden % 128 != 0
                || intermediate <= 0
                || queryHeads <= 0
                || keyValueHeads <= 0
                || queryHeads % keyValueHeads != 0
                || attentionHeadDim != 256
                || !config.attentionOutputGate()
                || !Double.isFinite(config.ropeTheta())
                || config.ropeTheta() <= 0
                || !Double.isFinite(config.partialRotaryFactor())
                || config.partialRotaryFactor() <= 0
                || config.partialRotaryFactor() > 1
                || rotaryDim <= 0
                || rotaryDim % 2 != 0
                || keyHeads <= 0
                || valueHeads <= 0
                || valueHeads % keyHeads != 0
                || keyHeadDim != 128
                || valueHeadDim != 128
                || kernelWidth < 2
                || kernelWidth > 32
                || !Double.isFinite(config.rmsNormEpsilon())
                || !Float.isFinite((float) config.rmsNormEpsilon())
                || (float) config.rmsNormEpsilon() <= 0) {
            throw new IllegalArgumentException("unsupported Qwen text model geometry");
        }

        int attentionQueryWidth = Math.multiplyExact(queryHeads, attentionHeadDim);
        int attentionKeyWidth = Math.multiplyExact(keyValueHeads, attentionHeadDim);
        int attentionQkWidth = Math.addExact(attentionQueryWidth, attentionKeyWidth);
        int attentionGateValueWidth = Math.addExact(attentionQueryWidth, attentionKeyWidth);
        int gdnQueryKeyWidth = Math.multiplyExact(Math.multiplyExact(2, keyHeads), keyHeadDim);
        int gdnValueWidth = Math.multiplyExact(valueHeads, valueHeadDim);
        int gdnValueZWidth = Math.multiplyExact(2, gdnValueWidth);
        int gdnConvolutionWidth = Math.addExact(gdnQueryKeyWidth, gdnValueWidth);

        TensorHandle finalNorm = validateNorm(weights.finalNorm(), hidden);
        TensorHandle outputHead =
                validateQuantized(weights.lmHead(), config.vocabSize(), hidden, WeightFormat.Q3_G64_FP16);
        List<Instruction> nodes = new ArrayList<>();
        nodes.add(node(
                0, Kind.EMBEDDING, List.of(), List.of(embedding), List.of(), List.of(Buffer.HIDDEN_STATE), 0, hidden));
        int precedingLayer = 0;

        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            QwenLayerWeights layer = layers[layerIndex];
            if (layer == null || layer.index() != layerIndex) {
                throw new IllegalArgumentException("loaded text layers must be indexed contiguously from zero");
            }
            if (!(layer.ffn() instanceof QwenCompactDenseFfnWeights ffn)) {
                throw new IllegalArgumentException("all loaded text layers must use compact dense FFN weights");
            }
            TensorHandle inputNorm = validateNorm(layer.inputNorm(), hidden);
            TensorHandle postNorm = validateNorm(layer.postAttentionNorm(), hidden);
            Buffer layerInput = layerIndex == 0 ? Buffer.HIDDEN_STATE : Buffer.FINAL_HIDDEN_STATE;
            int normId = addNode(
                    nodes,
                    Kind.RMS_NORM_UNIT_OFFSET,
                    layerIndex,
                    List.of(precedingLayer),
                    List.of(inputNorm),
                    List.of(layerInput),
                    List.of(Buffer.INPUT_NORMALIZED),
                    hidden,
                    hidden);

            int mixerProjectionId;
            if (layerTypes[layerIndex] == QwenLayerType.GATED_DELTA_NET) {
                if (!(layer.mixer() instanceof QwenCompactGatedDeltaNetWeights gdn)) {
                    throw new IllegalArgumentException("declared GDN layer has incompatible compact weights");
                }
                TensorHandle aLog = validateFp32Vector(gdn.aLog(), valueHeads);
                TensorHandle dtBias = validateFp32Vector(gdn.dtBias(), valueHeads);
                TensorHandle convolution = validateBf16Matrix(gdn.convolution(), kernelWidth, gdnConvolutionWidth);
                TensorHandle aProjection = validateBf16Matrix(gdn.aProjection(), valueHeads, hidden);
                TensorHandle bProjection = validateBf16Matrix(gdn.bProjection(), valueHeads, hidden);
                TensorHandle queryKey =
                        validateQuantized(gdn.queryKey(), gdnQueryKeyWidth, hidden, WeightFormat.Q4_G64_FP16);
                TensorHandle valueZ = validateQuantized(gdn.valueZ(), gdnValueZWidth, hidden, WeightFormat.Q5_G64_FP16);
                TensorHandle gdnNorm = validateNorm(gdn.norm(), valueHeadDim);
                TensorHandle gdnOutput =
                        validateQuantized(gdn.output(), hidden, gdnValueWidth, WeightFormat.Q3_G64_FP16);

                int queryKeyId = addNode(
                        nodes,
                        Kind.Q4_LINEAR,
                        layerIndex,
                        List.of(normId),
                        List.of(queryKey),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.QK_PROJECTED),
                        hidden,
                        gdnQueryKeyWidth);
                int valueZId = addNode(
                        nodes,
                        Kind.Q5_LINEAR,
                        layerIndex,
                        List.of(normId),
                        List.of(valueZ),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.VALUE_Z_PROJECTED),
                        hidden,
                        gdnValueZWidth);
                int aProjectionId = addNode(
                        nodes,
                        Kind.BF16_LINEAR,
                        layerIndex,
                        List.of(normId),
                        List.of(aProjection),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.A_PROJECTED),
                        hidden,
                        valueHeads);
                int bProjectionId = addNode(
                        nodes,
                        Kind.BF16_LINEAR,
                        layerIndex,
                        List.of(normId),
                        List.of(bProjection),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.B_PROJECTED),
                        hidden,
                        valueHeads);
                int controlId = addNode(
                        nodes,
                        Kind.GDN_CONTROL,
                        layerIndex,
                        List.of(aProjectionId, bProjectionId),
                        List.of(aLog, dtBias),
                        List.of(Buffer.A_PROJECTED, Buffer.B_PROJECTED),
                        List.of(Buffer.GDN_ALPHA, Buffer.GDN_BETA),
                        valueHeads,
                        valueHeads);
                int convolutionId = addNode(
                        nodes,
                        Kind.GDN_CONVOLUTION,
                        layerIndex,
                        List.of(queryKeyId, valueZId),
                        List.of(convolution),
                        List.of(Buffer.QK_PROJECTED, Buffer.VALUE_Z_PROJECTED),
                        List.of(Buffer.GDN_CONVOLVED),
                        gdnConvolutionWidth,
                        gdnConvolutionWidth);
                int recurrenceId = addNode(
                        nodes,
                        Kind.GDN_RECURRENCE,
                        layerIndex,
                        List.of(controlId, convolutionId),
                        List.of(),
                        List.of(Buffer.GDN_CONVOLVED, Buffer.GDN_ALPHA, Buffer.GDN_BETA),
                        List.of(Buffer.GDN_RECURRENT),
                        gdnConvolutionWidth,
                        gdnValueWidth);
                int gatedNormId = addNode(
                        nodes,
                        Kind.GDN_GATED_RMS_NORM,
                        layerIndex,
                        List.of(recurrenceId, valueZId),
                        List.of(gdnNorm),
                        List.of(Buffer.GDN_RECURRENT, Buffer.VALUE_Z_PROJECTED),
                        List.of(Buffer.GDN_NORMALIZED),
                        gdnValueWidth,
                        gdnValueWidth);
                mixerProjectionId = addNode(
                        nodes,
                        Kind.Q3_LINEAR,
                        layerIndex,
                        List.of(gatedNormId),
                        List.of(gdnOutput),
                        List.of(Buffer.GDN_NORMALIZED),
                        List.of(Buffer.MIXER_DELTA),
                        gdnValueWidth,
                        hidden);
            } else if (layerTypes[layerIndex] == QwenLayerType.FULL_ATTENTION) {
                if (!(layer.mixer() instanceof QwenCompactAttentionWeights attention)) {
                    throw new IllegalArgumentException(
                            "declared full-attention layer has incompatible compact weights");
                }
                TensorHandle queryKey =
                        validateQuantized(attention.queryKey(), attentionQkWidth, hidden, WeightFormat.Q4_G64_FP16);
                TensorHandle gateValue = validateQuantized(
                        attention.gateValue(), attentionGateValueWidth, hidden, WeightFormat.Q5_G64_FP16);
                TensorHandle queryNorm = validateNorm(attention.queryNorm(), attentionHeadDim);
                TensorHandle keyNorm = validateNorm(attention.keyNorm(), attentionHeadDim);
                TensorHandle attentionOutput =
                        validateQuantized(attention.output(), hidden, attentionQueryWidth, WeightFormat.Q3_G64_FP16);

                int queryKeyId = addNode(
                        nodes,
                        Kind.Q4_LINEAR,
                        layerIndex,
                        List.of(normId),
                        List.of(queryKey),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.QK_PROJECTED),
                        hidden,
                        attentionQkWidth);
                int gateValueId = addNode(
                        nodes,
                        Kind.Q5_LINEAR,
                        layerIndex,
                        List.of(normId),
                        List.of(gateValue),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.VALUE_Z_PROJECTED),
                        hidden,
                        attentionGateValueWidth);
                int qkNormRopeId = addNode(
                        nodes,
                        Kind.ATTENTION_QK_NORM_ROPE,
                        layerIndex,
                        List.of(queryKeyId),
                        List.of(queryNorm, keyNorm),
                        List.of(Buffer.QK_PROJECTED),
                        List.of(Buffer.ATTENTION_QK_NORMALIZED),
                        attentionQkWidth,
                        attentionQkWidth);
                int kvAppendId = addNode(
                        nodes,
                        Kind.ATTENTION_KV_APPEND,
                        layerIndex,
                        List.of(qkNormRopeId, gateValueId),
                        List.of(),
                        List.of(Buffer.ATTENTION_QK_NORMALIZED, Buffer.VALUE_Z_PROJECTED),
                        List.of(),
                        attentionQkWidth + attentionGateValueWidth,
                        0);
                int attentionId = addNode(
                        nodes,
                        Kind.ATTENTION_CAUSAL,
                        layerIndex,
                        List.of(qkNormRopeId, gateValueId, kvAppendId),
                        List.of(),
                        List.of(Buffer.ATTENTION_QK_NORMALIZED, Buffer.VALUE_Z_PROJECTED),
                        List.of(Buffer.ATTENTION_CONTEXT),
                        attentionQkWidth + attentionGateValueWidth,
                        attentionQueryWidth);
                mixerProjectionId = addNode(
                        nodes,
                        Kind.Q3_LINEAR,
                        layerIndex,
                        List.of(attentionId),
                        List.of(attentionOutput),
                        List.of(Buffer.ATTENTION_CONTEXT),
                        List.of(Buffer.MIXER_DELTA),
                        attentionQueryWidth,
                        hidden);
            } else {
                throw new IllegalArgumentException("unsupported declared Qwen layer type at " + layerIndex);
            }

            int mixerResidualId = addNode(
                    nodes,
                    Kind.RESIDUAL_ADD,
                    layerIndex,
                    List.of(precedingLayer, mixerProjectionId),
                    List.of(),
                    List.of(layerInput, Buffer.MIXER_DELTA),
                    List.of(Buffer.MIXER_HIDDEN),
                    hidden,
                    hidden);
            int postNormId = addNode(
                    nodes,
                    Kind.RMS_NORM_UNIT_OFFSET,
                    layerIndex,
                    List.of(mixerResidualId),
                    List.of(postNorm),
                    List.of(Buffer.MIXER_HIDDEN),
                    List.of(Buffer.POST_MIXER_NORMALIZED),
                    hidden,
                    hidden);
            TensorHandle gateUp = validateQuantized(
                    ffn.gateUp(), Math.multiplyExact(2, intermediate), hidden, WeightFormat.Q3_G64_FP16);
            TensorHandle down = validateQuantized(ffn.down(), hidden, intermediate, WeightFormat.Q3_G64_FP16);
            int gateUpId = addNode(
                    nodes,
                    Kind.Q3_LINEAR,
                    layerIndex,
                    List.of(postNormId),
                    List.of(gateUp),
                    List.of(Buffer.POST_MIXER_NORMALIZED),
                    List.of(Buffer.GATE_UP),
                    hidden,
                    2 * intermediate);
            int swigluId = addNode(
                    nodes,
                    Kind.SWIGLU,
                    layerIndex,
                    List.of(gateUpId),
                    List.of(),
                    List.of(Buffer.GATE_UP),
                    List.of(Buffer.SWIGLU),
                    2 * intermediate,
                    intermediate);
            int downId = addNode(
                    nodes,
                    Kind.Q3_LINEAR,
                    layerIndex,
                    List.of(swigluId),
                    List.of(down),
                    List.of(Buffer.SWIGLU),
                    List.of(Buffer.FFN_DELTA),
                    intermediate,
                    hidden);
            precedingLayer = addNode(
                    nodes,
                    Kind.RESIDUAL_ADD,
                    layerIndex,
                    List.of(mixerResidualId, downId),
                    List.of(),
                    List.of(Buffer.MIXER_HIDDEN, Buffer.FFN_DELTA),
                    List.of(Buffer.FINAL_HIDDEN_STATE),
                    hidden,
                    hidden);
        }

        int finalNormId = addNode(
                nodes,
                Kind.RMS_NORM_UNIT_OFFSET,
                -1,
                List.of(precedingLayer),
                List.of(finalNorm),
                List.of(Buffer.FINAL_HIDDEN_STATE),
                List.of(Buffer.FINAL_NORMALIZED),
                hidden,
                hidden);
        addNode(
                nodes,
                Kind.Q3_LINEAR,
                -1,
                List.of(finalNormId),
                List.of(outputHead),
                List.of(Buffer.FINAL_NORMALIZED),
                List.of(Buffer.LOGITS),
                hidden,
                config.vocabSize());

        return new PlanData(
                List.copyOf(nodes),
                List.of(),
                fullBufferSpecs(
                        hidden,
                        intermediate,
                        config.vocabSize(),
                        Math.max(gdnQueryKeyWidth, attentionQkWidth),
                        Math.max(gdnValueZWidth, attentionGateValueWidth),
                        valueHeads,
                        gdnConvolutionWidth,
                        gdnValueWidth,
                        attentionQkWidth,
                        attentionQueryWidth),
                true,
                true);
    }

    private static List<BufferSpec> fullBufferSpecs(
            int hidden,
            int intermediate,
            int vocabulary,
            int queryKeyWidth,
            int valueZWidth,
            int valueHeads,
            int convolutionWidth,
            int valueWidth,
            int attentionQkWidth,
            int attentionQueryWidth) {
        return List.of(
                spec(Buffer.HIDDEN_STATE, hidden, ElementType.BF16),
                spec(Buffer.MIXER_HIDDEN, hidden, ElementType.BF16),
                spec(Buffer.FINAL_HIDDEN_STATE, hidden, ElementType.BF16),
                spec(Buffer.INPUT_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.QK_PROJECTED, queryKeyWidth, ElementType.BF16),
                spec(Buffer.VALUE_Z_PROJECTED, valueZWidth, ElementType.BF16),
                spec(Buffer.A_PROJECTED, valueHeads, ElementType.FP32),
                spec(Buffer.B_PROJECTED, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_ALPHA, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_BETA, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_CONVOLVED, convolutionWidth, ElementType.BF16),
                spec(Buffer.GDN_RECURRENT, valueWidth, ElementType.BF16),
                spec(Buffer.GDN_NORMALIZED, valueWidth, ElementType.BF16),
                spec(Buffer.ATTENTION_QK_NORMALIZED, attentionQkWidth, ElementType.BF16),
                spec(Buffer.ATTENTION_CONTEXT, attentionQueryWidth, ElementType.BF16),
                spec(Buffer.MIXER_DELTA, hidden, ElementType.BF16),
                spec(Buffer.POST_MIXER_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.GATE_UP, 2 * intermediate, ElementType.BF16),
                spec(Buffer.SWIGLU, intermediate, ElementType.BF16),
                spec(Buffer.FFN_DELTA, hidden, ElementType.BF16),
                spec(Buffer.FINAL_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.LOGITS, vocabulary, ElementType.BF16));
    }

    private static PlanData operatorSlice(
            QwenWeights weights, TensorHandle normWeight, List<TensorHandle> projections) {
        Objects.requireNonNull(weights.config(), "config");
        Objects.requireNonNull(projections, "projections");
        int hiddenSize = weights.config().hiddenSize();
        int vocabularySize = weights.config().vocabSize();
        if (hiddenSize <= 0
                || vocabularySize <= 0
                || (normWeight == null && !projections.isEmpty())
                || (normWeight != null && projections.isEmpty())) {
            throw new IllegalArgumentException("invalid operator slice");
        }
        TensorHandle embedding = validateEmbedding(weights.tokenEmbedding(), vocabularySize, hiddenSize);
        List<Instruction> nodes = new ArrayList<>();
        nodes.add(node(
                0,
                Kind.EMBEDDING,
                List.of(),
                List.of(embedding),
                List.of(),
                List.of(Buffer.HIDDEN_STATE),
                0,
                hiddenSize));
        if (normWeight != null) {
            TensorHandle norm = validateNorm(normWeight, hiddenSize);
            nodes.add(node(
                    1,
                    Kind.RMS_NORM,
                    List.of(0),
                    List.of(norm),
                    List.of(Buffer.HIDDEN_STATE),
                    List.of(Buffer.INPUT_NORMALIZED),
                    hiddenSize,
                    hiddenSize));
            for (int index = 0; index < projections.size(); index++) {
                TensorHandle projection = Objects.requireNonNull(projections.get(index), "projection");
                long[] shape = projection.shape();
                if (shape == null || shape.length != 2 || shape[0] <= 0 || shape[0] > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("invalid Q3 projection shape");
                }
                nodes.add(new Instruction(
                        nodes.size(),
                        Kind.Q3_LINEAR,
                        List.of(1),
                        List.of(validateQuantized(projection, shape[0], hiddenSize, WeightFormat.Q3_G64_FP16)),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.SLICE_PROJECTION),
                        hiddenSize,
                        Math.toIntExact(shape[0]),
                        index));
            }
        }
        List<Integer> widths = nodes.stream()
                .filter(instruction -> instruction.kind() == Kind.Q3_LINEAR)
                .map(Instruction::outputWidth)
                .toList();
        return new PlanData(List.copyOf(nodes), widths, List.of(), false);
    }

    private static Instruction node(
            int id,
            Kind kind,
            List<Integer> dependencies,
            List<TensorHandle> weights,
            List<Buffer> inputs,
            List<Buffer> outputs,
            int inputWidth,
            int outputWidth) {
        return new Instruction(id, kind, dependencies, weights, inputs, outputs, inputWidth, outputWidth, -1);
    }

    private static int addNode(
            List<Instruction> nodes,
            Kind kind,
            int layerIndex,
            List<Integer> dependencies,
            List<TensorHandle> weights,
            List<Buffer> inputs,
            List<Buffer> outputs,
            int inputWidth,
            int outputWidth) {
        int id = nodes.size();
        nodes.add(new Instruction(
                id, kind, dependencies, weights, inputs, outputs, inputWidth, outputWidth, -1, layerIndex));
        return id;
    }

    private static BufferSpec spec(Buffer buffer, int width, ElementType type) {
        return new BufferSpec(buffer, width, type);
    }

    private static TensorHandle validateNorm(TensorHandle norm, int width) {
        return validateDirect(norm, new long[] {width}, WeightFormat.BF16, (long) width * Short.BYTES);
    }

    private static TensorHandle validateFp32Vector(TensorHandle vector, int width) {
        return validateDirect(vector, new long[] {width}, WeightFormat.FP32, (long) width * Float.BYTES);
    }

    private static TensorHandle validateBf16Matrix(TensorHandle matrix, int rows, int columns) {
        return validateDirect(
                matrix, new long[] {rows, columns}, WeightFormat.BF16, (long) rows * columns * Short.BYTES);
    }

    private static TensorHandle validateDirect(
            TensorHandle handle, long[] expectedShape, WeightFormat format, long expectedByteSize) {
        Objects.requireNonNull(handle, "weight");
        long[] shape = handle.shape();
        if (shape == null
                || !java.util.Arrays.equals(shape, expectedShape)
                || handle.deviceAddress() == 0
                || handle.dataType() != TensorDataType.BF16
                || handle.format() != format
                || handle.layout() != WeightLayout.CONTIGUOUS_LE_V1
                || handle.byteSize() != expectedByteSize) {
            throw new IllegalArgumentException("unsupported direct weight layout or dimensions: " + handle.name());
        }
        return copyHandle(handle);
    }

    /// Projections may be NVFP4 wherever a row-split format is registered; the token embedding, a
    /// gather, has Q3 kernels only.
    private static final WeightFormat EMBEDDING_FORMAT = WeightFormat.Q3_G64_FP16;

    private static TensorHandle validateEmbedding(TensorHandle handle, long rows, int width) {
        if (handle.format() != EMBEDDING_FORMAT) {
            throw new IllegalArgumentException("unsupported token embedding format: " + handle.format());
        }
        return validateQuantized(handle, rows, width, EMBEDDING_FORMAT);
    }

    private static TensorHandle validateQuantized(TensorHandle handle, long rows, int width, WeightFormat format) {
        Objects.requireNonNull(handle, "weight");
        long[] shape = handle.shape();
        if (shape == null
                || shape.length != 2
                || shape[0] != rows
                || shape[1] != width
                || width % 64 != 0
                || (handle.deviceAddress() == 0 && !handle.hostBacked())
                || handle.dataType() != TensorDataType.BF16
                || !(handle.format() == format || handle.format() == WeightFormat.NVFP4)
                || !(handle.layout() == WeightLayout.ROW_SPLIT_K128_V1
                        || (handle.layout() == WeightLayout.ROW_SPLIT_P2E2_V1 && format == WeightFormat.Q3_G64_FP16))
                || !CompactTensorLayout.acceptsByteSize(
                        shape, handle.dataType(), handle.format(), handle.layout(), handle.byteSize())) {
            throw new IllegalArgumentException("unsupported quantized weight layout or dimensions: " + handle.name());
        }
        return copyHandle(handle);
    }

    private static TensorHandle copyHandle(TensorHandle handle) {
        return new TensorHandle(
                handle.name(),
                handle.shape().clone(),
                handle.dataType(),
                handle.format(),
                handle.layout(),
                handle.deviceAddress(),
                handle.byteSize(),
                handle.hostAddress());
    }
}
