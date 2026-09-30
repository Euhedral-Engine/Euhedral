# Qwen instruction and frame model

Euhedral-Inference defines operations and owns model/runtime state. Euhedral-Execution schedules
individual ready operations. A frame is an instruction, not a prebuilt request pipeline or a
transformer-layer control loop.

## State and ownership

- `QwenWeights` supplies model-lifetime, GPU-resident weights. Frames borrow handles; a quantum never
  releases model weights. The runtime owner must not unload weights while any frame can use them.
- `QwenExecutionPlan` is an immutable description of instruction kinds, weight handles, dependency
  edges, and output dimensions. Its published instructions and edges cannot be mutated by callers.
  Constructing it does not allocate a quantum or GPU workspace.
- `QwenSequenceState` is shared request-lifetime state. An atomic lease protects sequence mutation,
  and cancellation may race with frame completion. Independent frames reading the same normalized
  buffer do not gain independent sequence mutation rights.
- `QwenExecutionContext` owns one inference quantum: copied token IDs, the sequence lease, per-node
  dependency counters, outstanding-frame count, failure/cancellation outcome, and its GPU workspace.
  Intermediate buffers have distinct addresses for the embedding, normalized activation, and each
  projection. They are retained until all admitted frames finish; no per-access buffer lock is used.
- `QwenExecutionRunner` implements `LatticeSource` directly. It admits quanta and queues each ready
  context in the MPSC partition belonging to its immutable instruction. Worker completion and
  admission may enqueue concurrently; neither path creates a frame. The scheduler controls `pull`
  and `request`, which have exclusive entry. On that serialized source path, the runner checks out a
  pooled frame and supplies the queued quantum context. `request` synchronously drains ready work and
  pushes frames directly to Euhedral. `pull` delivers frames to its consumer without accumulating
  demand or pushing downstream. Empty requests do not leave outstanding credits; Euhedral must request
  again to service later arrivals. Both calls bulk-drain ready work without a per-frame demand counter.
  The Euhedral-owned `pull` consumer and downstream `push` do not throw.

## Instruction availability

The concrete `EmbeddingFrame`, `RmsNormFrame`, and `LinearFrame` each live in their own source file
under `scheduling/frames`. `QwenGpuOperationFrame` handles reusable control, recurrent, elementwise,
residual, and activation instructions. A quantum initially queues its embedding operation context.
Each frame invokes a standalone CUDA operation and synchronizes before reporting completion.
`QwenWorkGenerator` follows precomputed successor edges only; it decrements dependency counters and
queues each newly ready operation context. It does not walk the full model or call a layer method that
drives the next operation. Independent projections from one normalized activation receive separate
`LinearFrame` instances and output buffers.

```text
embedding output -> RmsNormFrame -> LinearFrame (projection A)
                                  -> LinearFrame (projection B)
```

All three frame types borrow their weight handles and buffer addresses. Their frame-local
completion state has one worker owner; shared dependency counts and terminal ownership use atomics.
Each successor's work reservation is recorded before enqueue, while the completing frame retains its
own reservation through fan-out; inline execution therefore cannot finalize a workspace while a
sibling instruction remains pending. CUDA operations do not import Euhedral types.

`QwenWorkGenerator` owns a Euhedral `FrameManager` for each fixed instruction: this keeps a linear
or GPU-operation frame's weight binding immutable even when several instructions exist. Checkout passes
only the new quantum context; no per-checkout wrapper or instruction copy is created. `getOrCreate`
runs only on the serialized `pull`/`request` path, so there is no manager lock on frame checkout.
Successful and failed finalization enqueue successor contexts, clear the completed quantum reference,
and then return the frame to its manager. Undelivered frames are cleared and recycled on the source
path if their quantum is cancelled. The manager's MPSC return queue accepts concurrent worker returns.
Reuse is best-effort when its bounded pool is full; execution does not depend on a frame being retained.
Frame references expire at finalization: callers must not retain or invoke a recycled frame, since the
manager can issue the same object for another quantum.

For the compact Qwen3.5 artifact, `new QwenExecutionPlan(weights)` inspects the actual loaded layer-zero
`QwenLayerWeights`. It rejects a topology that is not the loaded GDN form and publishes the complete
17-instruction graph: embedding, unit-offset input RMSNorm, independent Q4 query/key, Q5 value/z, and
BF16 control projections, GDN control/convolution/recurrent/gated-normalization operations, mixer
projection, residual, post-mixer RMSNorm, Q3 gate/up, SwiGLU, Q3 down, and the final residual. The
projection branches share only their immutable normalized-input dependency and write distinct buffers.
`QwenGdnSequenceState` owns the persistent convolution tail and FP32 recurrent matrix; the submission
workspace owns only transient activation buffers. Sequence state is closed on completion, cancellation,
or failure, after no frame can still access it. The explicit norm/projection constructor remains a
low-level operator slice for existing tests and does not construct a synthetic transformer layer.

## Prefill and decode routes

For a complete model, `QwenExecutionPlan.forExecution(kind, rows)` selects the topology per quantum.
There is no runtime or serving switch; the plan owns its fixed views and any view requalifies through
its owner.

```text
prefill:
  M == 256 and FFN 5120 x 17408 -> streamed C + A + D + F
  otherwise, M >= 64            -> combined A + B + D + F
  M < 64                        -> A + D, ordinary FFN and attention

decode:
  reference instruction topology
```

- A: rounded residual add + following RMSNorm (the final model layer keeps its plain residual).
- B: Q3 gate/up with a SwiGLU epilogue; ordinary Q3 down.
- C: gate/up+SwiGLU into two bounded feature slots consumed by a down pass that carries FP32
  accumulators across feature regions. It owns internal CUDA streams even in SYNC outer mode.
- D: joint BF16 A/B projection + GDN control, submitted before the heavy Q4/Q5 projections.
- F: Q4/Q5 attention producers with in-place Q/K normalization and RoPE, followed by
  NVFP4 K/V page writes, committed once at successful completion.

Full attention stores K and V in sequence-owned 256-token pages. Each D256 head is
rotated by normalized H256 and encoded as 128 bytes of E2M1 codes plus 16 E4M3
scales (one per contiguous group of 16 values). Device page tables contain raw
addresses; growing a sequence allocates new pages without copying existing KV
payloads. The completion callback publishes appended length, and sequence cleanup
releases pages and any decode scratch only after admitted GPU work has drained.

Queries use the same H256 transform; the weighted value sum is transformed back
before applying the query gate. Prefill uses 32-query by 32-key tiles with FP16
tensor-core operands, FP32 accumulation, and online softmax, without a quadratic
score allocation. Single-token decode splits the visible prefix across CTAs and
merges FP32 softmax statistics. At GQA ratio six and a prefix of at least 1024
tokens, each tensor-core CTA shares one K/V expansion across all six query heads;
shorter prefixes and other head ratios retain warp-based FP32 decode.
KV expansion to FP16 is exact; cache quantization and FP16 query/probability operands
are numerical approximation boundaries, so this path does not promise bitwise
agreement with the former BF16 cache.

Correctness is defined against the represented NVFP4 values, not accumulated
BF16-model hidden states or logits. Codec rounding, saturation, signed values,
and normalized Hadamard ordering are checked independently. Both prefill and
decode must match an FP64 represented-value attention oracle within the existing
FP16/MMA and BF16-output tolerance. Page payload preservation, append positions,
sequence state, ownership, and lifecycle invariants remain exact. Memcheck,
racecheck, initcheck, synccheck, and operation beyond 64K on the target device
remain required. Matching upstream V inputs is diagnostic evidence only.

Decode scratch is reserved lazily on the first decode, then reused until sequence
release. This one-time allocation is not a steady-state memory leak; release and
scratch-address stability remain independently enforced.

The complete model admits only native-supported projection/attention geometry (128-aligned hidden
width, 256-wide attention heads, a GDN convolution kernel of 2..32, and RMS epsilon representable
as a positive finite float). B's 128-aligned input and 32-aligned output follow from those admitted
weights. Workspace lifetime sharing applies only when both F and B/C occur in the selected view.
Region views share storage through fixed, named lifetime pairs
(`QwenExecutionWorkspace.configureRegionStorage`); a shared view cannot be detached. Lower-level
Q3/Q4/Q5 kernel dispatch is chosen natively by shape. `QwenExecutionPlan.reference(weights)` is
the unfused oracle used by tests; staged plans (`prefix`, `embeddingOnly`, operator slices) are
also reference-only.

## Completion boundary

All GPU operations remain synchronous. An operation's output becomes ready only after CUDA
synchronization. On success, a terminal consumer may read the still-live GPU buffers; then the
workspace and temporary token-ID buffer are released and the sequence lease is committed. The
caller receives a copy of the completion future, so it cannot publish a false terminal outcome.
Failure or cancellation prevents new successors, but already admitted frames still finalize before buffer
release. Ready contexts that have not been materialized are reaped only when the source next services
`pull` or `request`; without another source call, a cancelled quantum can remain outstanding. A failed
token-ID free is retried at terminal cleanup without discarding its address.
External cancellation is cooperative and does not interrupt an executing GPU call.
Cancellation that wins before the sequence lease is claimed produces a cancelled quantum, even if
it arrives between initial validation and the claim. Workspace allocations begin only after the
quantum owns the workspace object, retaining partial allocations for terminal cleanup or later retry.

`completeGracefully` closes new admission and signals source completion after every accepted quantum
has finalized. Queue emptiness alone is not a completion signal. `LatticeSource.complete()` currently
uses the same resource-safe graceful behavior. An executor abandoned by an uncaught JVM `Error` or a
permanently unavailable scheduler can leave a quantum pending; neither case can safely free buffers
can still be in use. Fatal shutdown and forced abandonment require a separate ownership policy.

The native RMSNorm, mixed Q4/Q5/Q3 linear, BF16 control, GDN, residual, and SwiGLU wrappers bind the
calling worker to its CUDA device before their first kernel load and before each launch. This permits CPU workers to move a ready frame between
threads while the current implementation remains single-device; multi-device context selection and
cross-device model residency are not implemented.

The architecture allows a future scheduler to defer GPU capacity, choose a CPU-capable operation,
coalesce ready frames, or run another sequence without changing instruction dependencies. This slice
does not yet generalize the graph across later layers or implement KV cache, full attention, MTP,
sampling, batching, CUDA Graphs, serving APIs, or asynchronous GPU completion.
