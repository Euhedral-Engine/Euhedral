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
(about +1%) at a 64-token context and 10 of 12 at 1024, so longer contexts keep ordinary launches.

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
