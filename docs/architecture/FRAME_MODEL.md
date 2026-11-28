# Qwen execution model

Measurements and rejected variants record the hardware and implementation at the time of each experiment; they do not establish current defaults.

A Qwen execution plan (`Plan`, `rust/crates/engine/src/plan/`) defines a static DAG of execution stages. A reusable
RustEE `StageGraph` represents those stages as independent frames. A quantum is started on a graph, which publishes only
its root stages to a lake. Successful stages create successor readiness and publish the ready successors to the lake.
CUDA stream order carries ordinary device dependencies; actual device completion is reserved for state-publication and
ownership boundaries.

Euhedral has no central scheduler or authority; scheduling is emergent. Workers take available frames for themselves, or a
frame is routed through the lattice by its hash, so every assignment is either deterministic (hashing) or first come, first
served. The engine crate defines the stages and owns model and sequence state. Nothing in the Qwen runtime executes a stage
body inline, scans for ready work, or walks the graph after the quantum started.

## The plan and its stage graphs

`Plan` is built once from the loaded model's weights (`Plan::build`). It owns one list of stages per view
family (the base model's, the MTP layer's, and with a DFlash2 drafter its Context and Block lists), each in dependency order with its dependencies and its ordering edges. It is
never the live scheduler. From a list it makes the `StageGraphSpec` of each view: the stages, each an immutable stage (a
kernel launch recipe with its weight bindings and its input and output buffer slots), the edges, and the lanes each stage may
use. A stage implements `Stage<Quantum>` and its launch (`Launch::launch`) makes the kernel launches and copies of one
quantum and nothing else (no waits, no markers, no signals), so a graph capture can record it.

`Executor` (`quantum/`) builds the `StageGraph`s of every view once, for each home lane, and starts each quantum on the graph of its
view and home lane. It stores:

- the stage-by-stage graph of each view variant (`View`, `Variant`) for each home lane;
- for the variants that are captured, the one-stage graph that runs the whole quantum as one `GraphBody` of the sequence's
  `SharedGraphCache` ([CUDA_GRAPHS.md](../kernels/CUDA_GRAPHS.md));
- the copier of the sequence's prefix-cache jobs.

Per quantum, the graph resets its fan-in counts and per-stage flags and binds the `Quantum` (`stages::Quantum`): its plan
view and kind, its token range, its start position, its rows' input record, its `KvView` of the attention cache, its failure
and cancellation state, and its outcome. A started quantum is a pooled run of the graph: after warm-up it allocates no frame,
successor list, graph or device buffer. The engine knows no model beyond the plan it was given.

## Resource lifetimes

Each resource lives as long as its natural owner, so the token boundary neither allocates nor frees:

- The `Workspace` belongs to the engine runtime (`EngineRuntime`). It holds one device buffer per slot of the plan, sized at
  load for the largest quantum any view runs (the prefill chunk's rows), and every graph binds the same buffers. Nothing
  grows while quanta run: a quantum larger than the workspace is refused at admission. The runtime frees it when it closes,
  and only when the device proves completion. Graphs share it through edges, not turns: see [The workspace and its
  owner](#the-workspace-and-its-owner).
- A sampling quantum's final logits row is copied to pinned host memory by a copy the logits stage queues on its own lane
  right after the LM head, so the quantum's single retirement boundary proves the row complete; the CPU reads it only after a
  successful outcome. A greedy request selects on the device instead (see [Measured behaviour](#measured-behaviour)).
- Pinned staging for uploads (token ids, KV page tables) comes from the engine's pinned pool and stays owned by the quantum
  that queued the copy until it retires.
- Persistent sequence state (`SequenceState`: attention KV pages, page tables, GDN state, decode scratch) belongs to the
  sequence and is released when the sequence's owner drops it, after its last quantum has retired.

A `Quantum` retires through RustEE's retirement: its outcome (`QuantumOutcome`) is the argument of the engine's `retire`
callback, which runs once, on a worker, after the quantum's device work completed.

## Admission

A request's sequence has an owner (`generation/source.rs`): one `Owner` holds everything the sequence changes between quanta,
its caches' frontiers, the graphs it recorded, the quanta in flight and what the request produced. The owner sits in a
`Mailbox` (`generation/mailbox.rs`) whose claim one thread at a time holds. A frame that has something to tell the owner (a
quantum concluded, a logits row arrived, a parked quantum's lane work ran) posts a message and then drives the mailbox: it runs
a turn of the owner on its own thread if the claim is free, and the holder of the claim sees the message if it is not.
Nothing polls the owner and nothing waits for it. A turn takes in what arrived, concludes the quanta in admission order, admits the
next ones and ends the request; the outcome is told to the caller after the claim is released.

Admission of a quantum is small:

1. the sequence's owner reserves the quantum's cache rows and uploads its input record on the owner's lane, so any
   initialization it queues precedes every stage;
2. the workspace's owner (`generation/order.rs`) decides what the quantum starts behind, and the executor starts it on a
   graph;
3. the graph publishes the quantum's root stages to the `stages` lake.

After that, admission is out of the execution path. A quantum whose start fails reaches its terminal outcome at admission,
and its slot in the conclusion order is passed so the quanta behind it still conclude. A quantum is started at most once.

## The workspace and its owner

The engine has one workspace, and its buffers are reused layer after layer inside a quantum, so the quanta of every sequence
use it one at a time. The workspace's owner (`generation/order.rs`) is the one place that decides what a quantum starts
behind. A quantum starts behind two quanta, as two `ExternalEdge`s of its graph:

- the last on the workspace, whatever sequence it belongs to, so that no two quanta use the buffers at once;
- its own sequence's last, the quantum before it in its chain, whose selection it takes as its token.

With one sequence they are the same quantum and the edge set is the chain's: one edge. Work that touches only a sequence's
state takes only the second edge. An external edge is lock-free: each stage keeps a stack of its external waiters, and a stage
that was already submitted, or swept when its quantum quiesced, satisfies a late waiter at once. A graph never waits on its
own previous binding: it is reused only after its quantum retired.

The owner holds the workspace and a sequence's next admission until `Order::retired` tells it that the quantum it waits on
has retired.

The owner sits in a `Mailbox`, like a sequence's owner: with the claim free (always, with one active sequence) the start runs
inline in the same turn; otherwise it is a deferred operation which whoever next takes the claim runs, in the order posted.
Nothing waits. No frame of this path is ordered in the lake: the frames that drive a mailbox are unordered (a random routing
hash), and the order that matters is carried by the mailbox's operations queue and the owners' inboxes, which keep each
poster's order. A hash shared by two sequences' frames would let one sequence's stalled head hold the other back.

Expansion scratch (the expansion of P2E2, MX and native NVFP4 activations) is one buffer; the plan chains the stages that
take it with ordering edges (`plan/hazards.rs`). The staging slots of host-backed weights are buffers too: the transfer
that fills a slot writes it, and the weight's consumer reads it.

## The lakes

The runtime has three lakes (`EngineRuntime`): `stages`, which carries the GPU submission frames, `host`, which carries CPU
tasks (a read's checks, parsing, planning), and `control`, which carries the steps of a startup and of a shutdown. A `Lake` is several queue ingest sinks, each an upstream source of the lattice, chosen by a frame's
routing hash. Publishing only enqueues, so a stage on a worker, a request thread and a CUDA driver callback all publish the
same way. The lake counts the units it was asked to carry; closing the runtime refuses new work and completes once every
admitted unit terminated.

## Readiness, fan-out, and fan-in

A stage frame runs its operation and returns. Its finalizer (`do_finally`), not its body, releases its successors:

```text
stage A submits its kernels to its lane
  -> A's finalizer satisfies each outgoing edge
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

The graph also counts live work: published frames that have not finished and armed completions. When that count reaches zero,
no stage of the quantum can run again, whether it succeeded, failed, or was cancelled.

## Device lanes

CPU scheduling and CUDA ordering are separate. The runtime owns a pool of CUDA streams, its lanes (`LaneSet`). The number of
stage lanes is the width of the plan's widest graph, the most stages that no edge orders (`Plan::fan_out_width`, and `Plan::width(view)` for one view; `Plan::lane_width` asks it
before a plan exists); a graph that is a chain needs one. The lanes are the stage lanes, then the owner's lane, then the transfer
lanes, with the staging lane last; the owner's and the transfer lanes are separate from the stage lanes. Each quantum has a home
lane, chosen from a hash of its id and the time it starts so that the roots of different quanta do not land on one lane
(`quantum/home.rs`); the home lane prepares the quantum and carries its retirement boundary. A stage chooses its lane each
time it runs and submits with that lane selected on the submitting thread, whichever Euhedral worker runs it:

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
every other lane the quantum used joins the retire lane the same way, so the single boundary covers all
of the quantum's work. A launch that awaited another lane does not use programmatic dependent launch.

Placement is the path policy (`plan/placement.rs`): each stage continues the lane of the predecessor whose longest remaining
path runs through it, fixed when the graph is built, so a graph's critical chain keeps one lane in every quantum and the other
branches of a fan-out take other lanes. Every graph spreads its stages, decode and prefill alike. There are no lane options.

With stages on different lanes the stream no longer orders every write after earlier reads and
writes of the same storage, so the plan adds those edges explicitly (`plan/hazards.rs`): for every storage, each reader is
ordered after the last writer and each writer after the last writer and every reader since it, unless the data dependencies
already imply the edge. Decode keeps the GDN projections as separate leaf frames, and every view keeps the
attention producers as four leaf frames (Q/K projection -> QK norm and RoPE, value/Z projection -> cache append), so those branches
can run on different lanes.

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


## Submission and completion edges

`EdgeKind` makes the boundary of each edge explicit.

- `Submission`, the ordinary edge: the consumer needs only the producer's successful submission. Stream order runs the
  consumer's device work after the producer's, so the host never waits for the producer's completion. The plan's stage
  dependencies are submission edges.
- `Completion`, the device-completion edge: the consumer needs the producer's device work to have completed, found by a poll
  of the lattice's `Completions` source. It is reserved for host consumption, state publication, storage release, and crossing
  ordering domains (the Flash-Next plan uses it between a router and the expert fetches that read its selection,
  `flash/plan.rs`).

A quantum's own retirement is its single device-completion boundary. The graph's retirement boundary is the latest marker of
each lane the quantum used: the other lanes join the retire lane on the device, and one retirement armed there ends the
quantum. The CUDA driver callback only fills a cell and wakes the lattice's `Completions` source; the retirement frame runs
on a worker, and:

1. confirms the boundary (a failed boundary fails the quantum with `QuantumOutcome::DeviceFailed`);
2. runs each attempted stage's retirement hook;
3. calls the engine's `retire` with the quantum and its `QuantumOutcome`, which publishes its message to the sequence's owner;
4. returns the quantum's run to the graph's pool.

The graph is reusable before the caller observes the outcome, so the next token finds an idle graph. There is one CUDA host
callback per quantum, not one per kernel.

The generation is an ouroboros. State and frame generation sit at the top: the sequence's owner admits a quantum into a graph.
Execution sits at the bottom: the stages and the device. The completion at the bottom throws a message back to the top: the
retired quantum's conclusion, posted to the owner's mailbox, starts a turn that samples the next token and admits the next
quantum (or ends the request). Nothing waits on a future between the two. A greedy request keeps two quanta in flight, so the
device runs the next step behind the one that is retiring.

## Pending and committed persistent state

Persistent sequence state has distinct frontiers. For the attention KV cache (`KvCache`, `state/kv.rs`):

- reserved: rows backed by pages;
- submitted: rows whose writes are queued on the owning quantum's lanes. Later stages of the same quantum, such as causal
  attention, read this frontier, because stream order runs their reads after the writes;
- committed: published once the quantum's device work retired; nothing beyond it is visible to other quanta.

Reservation runs inside the owning quantum with its lane selected. A grown page table is allocated in stream order, filled
into pinned staging, and uploaded by a copy queued on that lane ahead of the stages that read it. The outgrown table is freed
in stream order after the upload by the owner's next turn (`ReleasedBuffers`), so reservation never waits for the device and
never synchronizes with queued work. A quantum takes a `KvView` at admission, holding the pages and page tables of that
moment, so a table grown after it does not move what it reads.

A quantum that stopped part-way retires behind the work it queued, and its rows never become part of the sequence
(`discard`). GDN recurrent and convolution state is updated in place by stream-ordered kernels.

A sequence is a sequencer: parallel execution, ordered completion. The sequence's quanta (and its prefix-cache captures and
restores) are admitted in order and each completes in any order, but they conclude in admission order, through a RustEE
`Sequencer` (`generation/conclusion.rs`): the owner takes a place in the order when it publishes the work, and the place
travels with the work. The frame that retires the work completes its place; work dropped before it ran passes it. The thread
that completes the place at the head of the order concludes the ready prefix, in admission order, and hands each conclusion
to the owner's mailbox without waiting. The conclusion (commits, the listener, the failure of a request) runs in the owner's
turn, on state the owner alone touches. One order serves a sequence's quanta and copies: they conclude in the order they were
submitted, and none concludes twice. Cancellation (`CancelHandle`) asks the generation to end at its next quantum boundary: the request ends once
through its listener as `FinishReason::Cancelled`, and the sequence is free for the next request.

Several quanta of one sequence can be in flight. Edges order their device accesses, including KV appends; conclusions publish
only the rows each retired quantum committed. See [speculative decoding](05-speculative-decoding.md) for how MTP and DFlash2
steps use this.

## Failure and cancellation

- A failed stage submission records the quantum's first cause (by one compare-and-swap) and publishes no successor; the
  quantum ends as `QuantumOutcome::Failed`. A join whose predecessor failed is never published.
- Cancellation (a quantum's kill switch) stops every stage that has not yet submitted and every publication that has not
  happened. Work already submitted to the stream stays owned by the quantum until its retirement; the quantum ends as
  `QuantumOutcome::Killed`.
- The quantum still retires through exactly one boundary, so its terminal outcome is published once. A killed quantum holds
  for the latest marker of each lane it used before it concludes, so nothing it holds is let go under work that may still use
  it.
- A CUDA driver callback never calls CUDA, runs a frame, or releases memory.
- A panic in a frame is contained where the frame ends in place (`StageGraph::contained_panics` counts it); a `retire` that
  panics has already concluded its quantum, so its run is pooled again.
- A frame the lattice rejects without running it is finalized through `do_finally_with_error`; a rejected stage fails its
  quantum, and a rejected completion or retirement frame is finished on the rejecting thread, which is never a driver
  callback, so the quantum still retires once.
- A graph is recycled, with its storage, only after its quantum's boundary was confirmed.

## Host work on the workers

Every piece of host work a request needs runs as frames on the lattice's workers; no other thread computes for it.

- **The request** (`rust/crates/api-server`): a connection's body becomes a planned generation as a chain of small frames
  (parse, map, grammar, template, encode and plan, `planning.rs`), each with its own routing hash and published by the step
  before it, so a large body or a long prompt never holds a worker for the whole chain. The prompt is encoded in one frame
  (`Backend::encode_prompt`).
- **Admission control** (`Admission`): how many generations run and how many wait is one owner in a `ClaimCell`; every
  operation from outside is a command posted to its inbox, and a frame drives the owner. A turn only decides; what it decides
  is carried out as frames of their own after the claim is given back.
- **Generation** is the sequence's owner (see [Admission](#admission)): a turn admits the next quantum, and a quantum's
  retirement posts the message that starts the next turn. The caller's listener runs on a worker, and a cancellation from it
  stops the generation before another quantum starts. Only tests, tools and the blocking caller of `generate` wait, each on its
  own thread, on a `Handoff`.
- **The answer** (`api-server/src/answer/`): the answer to one request is an `Entry` of the admission with an owner touched by
  one turn at a time. A streamed request writes its events through the transport one piece at a time, the continuation of a
  write posts the command that starts the next one, and the events that come meanwhile are held and sent together. What is
  held is bounded: a client that does not read is cut off and its generation cancelled. A client that leaves is a command to
  the owner, so its generation is cancelled without a thread watching it.
- **The prefix cache** ([PREFIX_CACHE.md](../PREFIX_CACHE.md)) captures and restores sequence state between quanta. Its tree
  belongs to its owner: a lookup, a restore or a capture is a request posted to the cache's mailbox, answered by a closure once
  the owner's claim is free. Its copies are queued asynchronously in pieces of at most the piece size, so a worker is never held
  for a whole checkpoint, and the copy job ends in one report to the cache.
- **Workers idle on fixed timing** (`fragment_config`): a worker with nothing to run parks 15 us and looks again. A decode
  quantum is a graph of some 290 stages, each a frame handed on once the one before it completes, and nothing wakes a worker
  for a frame offered to its cache, so each hand-off waits for a worker to come round. Rejected: the lattice's adaptive idle
  timing, which derives each worker's park (up to 0.8 ms) from that worker's own history of frame costs, so every hand-off
  waits out such a park. nvfp4-compressed through the API, 32K prompt, 256 tokens, six requests in a row (measured on the
  earlier host implementation): with 4 worker CPUs 83.9-86.6 tok/s against 101.1-101.8 with fixed timing; with 32 worker CPUs
  three requests at 101-103 and then 77.7-77.9, as long prefills had left fewer workers awake, against 101.5-102.9
  throughout. A lake attaches its sinks to the lattice for a busy period: beside them it attaches an idle watch, and when no
  unit has been admitted for `host.idle-release` (0 keeps them attached) the watch detaches both; the next unit attaches a
  fresh set. A frame is published only while a unit it belongs to is admitted, so none is in flight when the sinks go.
- **Waking a thread outside the lattice per token is expensive.** When a blocked caller drained each token's text from a
  queue, the wake-up sat between a verify quantum's retirement and the next admission: the median gap after a 4-row verify quantum was 276 us,
  229 us when nothing outside the workers was woken (measured on the earlier host implementation). The benchmark therefore
  drives the asynchronous path, as the server does; the blocking call remains for tests and tools.

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

`View::of(kind, rows)` selects the stage graph of a quantum (`plan/views.rs`). There is no runtime or serving switch; the plan
owns its fixed views.

```text
prefill chunk:                 PrefillChunk (the fused topology, whatever its rows)
decode:                        Decode (one row)
VERIFY, up to 8 rows:          Verify (the decode kernels)
VERIFY, more rows:             VerifyWide (the quantized linears take the expansion scratch)
MTP draft, up to 8 rows:       Draft
MTP catch-up, more rows:       CatchUp
DFlash2 drafter:               Context (committed rows) and Block (the anchor and the mask rows)
```

- The rounded residual add and the following RMSNorm are one launch.
- The joint GDN A/B projection and its control are one launch, ahead of the heavy projections.
- Every view keeps the attention producers as leaf frames: the Q/K projection -> Q/K normalization and RoPE (into the
  normalized Q/K buffer), and the value/Z projection -> the K/V page append, committed at the quantum's retirement.
- The FFN down is an ordinary linear in every view. A format that offers a paired gate/up kernel gets it
  ([PREFILL_MX.md](../kernels/PREFILL_MX.md), [NVFP4_NATIVE.md](../kernels/NVFP4_NATIVE.md)); there is no streamed FFN view.
- Host-backed weights add a weight-transfer stage per use to every view ([NVFP4_RESIDENCY.md](../quantization/NVFP4_RESIDENCY.md)).

Full attention stores K and V in sequence-owned 256-token pages. With the NVFP4 cache format, each D256 head is rotated by
normalized H256 and encoded as 128 bytes of INT4 midrise codes (magnitudes 0.5 through 7.5) plus 16 E4M3 scales (one per
contiguous group of 16 values). The BF16 cache keeps unrotated 512-byte head rows; `context.kv-cache-format` chooses the
format. Device page tables contain raw addresses; growing a sequence allocates new pages without copying existing KV
payloads, and sequence cleanup releases pages and any decode scratch only after admitted GPU work has drained. Prefill
attention is the FlashAttention-2 style leaf with producer warps for quanta of 512 or more rows (or 128 or more once the
cache holds 2048 keys), and a 32-query by 32-key tile with FP16 tensor-core operands, FP32 accumulation and online softmax
otherwise ([PREFILL_MX.md](../kernels/PREFILL_MX.md)). Single-token decode splits the visible prefix into about 48-key
spans (at most 64) per query head below 2048 keys, one CTA each, and merges FP32 softmax statistics in a second kernel;
from 2048 keys one three-warp tensor-core CTA per KV head and 32-key split serves the whole query-head group
([ATTENTION_DECODE.md](../kernels/ATTENTION_DECODE.md)). Correctness is defined against the represented NVFP4 values
within the existing FP16/MMA and BF16-output tolerance.

When the full cache and weights do not fit, the planner can move pages beyond `context.kv-cache-on-gpu-tokens` to pinned
host memory before offloading weights. Each sequence slot owns host pages per attention layer and one device region shared
by those layers. A layer copies its host pages into the region before attention and copies appended rows back afterward;
carried-state edges protect region reuse. Prefix checkpoints then stop at the device-page frontier. See
[memory architecture](04-memory-architecture.md).

Decode and verify quanta that start below 1024 tokens launch their kernels with CUDA programmatic dependent launch
(selected per quantum): every kernel registered for it begins with `griddepcontrol.wait` (`native/src/common/pdl.cuh`), so
it cannot read a predecessor's output early. It won 12 of 12 paired forks (about +1%) at a 64-token context and 10 of 12 at
1024, so longer contexts keep ordinary launches. Quanta replayed from captured CUDA graphs
([CUDA_GRAPHS.md](../kernels/CUDA_GRAPHS.md)) carry programmatic edges at every position instead.

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
  select). From 2048 keys the GQA three-warp kernel runs ([ATTENTION_DECODE.md](../kernels/ATTENTION_DECODE.md)).
- **Decode RMSNorm and BF16 projections.** Batched loads and, for a single row, 40 CTAs that each recompute the same reduction and
  write one slice (no global intermediate): the RMSNorm takes 3.0 us and the BF16 projection 3.5 us.
- **Q3, Q4 and Q5 decode, contiguous ownership.** Three words of a Q3 group hold exactly 32 whole codes, so in
  `euhedral_q3_decode_contiguous` a lane owns 32 contiguous K values whose bit positions are constants, forms one FP32 dot product
  per half group and scales it once. It streams at 720-800 GB/s (gate/up 91 us, down 48 us) and Q3 time per decode token is
  11.1 ms. A Q4/Q5 half group is 16 code bytes plus one 32-bit word of fifth bits, so a lane reads its 32 codes with one 16-byte
  load (`euhedral_q4_decode_contiguous`, `euhedral_q5_decode_contiguous`): Q5 5120 -> 12288 52 us, Q4 5120 -> 4096 15 us, near 800
  GB/s. The FP32 accumulation order differs from the scalar references (`native/src/reference/kernels.cu`), which are the exact
  oracle that the launch table selects (tests only). Against them fewer than 0.1% of outputs differ, never by
  more than one BF16 ulp, and the error against an FP64 evaluation is unchanged. The 2 to 8 row twins reproduce the one-row kernel
  bit for bit ([MTP_VERIFIER.md](../speculation/MTP_VERIFIER.md)).
- **GDN convolution.** 32-row blocks give 1280 CTAs at 512 rows (68 us); each output reads only the previous three inputs.
- **Decode fusion.** With every decode GEMV near the DRAM roofline (750-850 GB/s), the remaining decode time is small kernels and
  the gaps between launches. Decode runs its own instance of the short-prefill topology: rounded residual add + RMSNorm is one
  region and the GDN A/B projection + control another. For one row the residual norm keeps its columns in registers, eight per
  thread with 16-byte loads, and reduces by warp shuffles (`euhedral_residual_rms_norm_row_bf16`, within one BF16 ulp of the exact
  kernel, which exact numerics keep).
- **Prefill attention.** The 32-query tile expands whole 16-element groups per thread (one 8-byte code load, one scale, two 16-byte
  shared stores; the same exact FP16 values) and four threads share each query row's softmax statistics, so the denominator is
  summed in four partial sums (`euhedral_attention_prefill32_nvfp4_exact` keeps the key-ordered sum): 18 ms per 1024-token prompt.
  The FlashAttention-2 leaf (`euhedral_attention_prefill_fa2_nvfp4`) takes the larger grids ([PREFILL_MX.md](../kernels/PREFILL_MX.md)).
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
`euhedral-bench drift --candidate exact-numerics` ([QUALITY.md](../evaluation/QUALITY.md)) compares the two end to end: the same forced tokens on the exact kernels, then on the production kernels, bounded by the drift thresholds. The Q3 artifact's relaxed numerics measure a hidden-state
relative error of 6.0e-2, KL 5.1e-3 and top-1 agreement 0.961 over 8 windows of 48 positions, with no growth by position
([PREFILL_MX.md](../kernels/PREFILL_MX.md)); a single one-ulp perturbation (the decode attention merge order) gives about 5.4% and KL 2.6e-3,
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

## What host progression costs

Measured on the i9-14900K with an RTX 5070 Ti, 32 workers, with a probe that stamps every frame at publication and at the
start and end of its execution (a throwaway patch, not in the tree), on the host implementation before the Rust engine. The
Rust engine's hops have not been re-measured; the figures show the order of magnitude and the structure. The numbers are
medians unless said otherwise.

- **A hop is cheap.** A frame published from a worker starts 4-8 us later (p99 16-32 us): the publishing worker, or an
  awake neighbour, finds it in a sink on its next cycle. An ordered (owner) frame costs 15-20 us, because it takes the
  request, route and cache path instead of the direct pull.
- **Dense decode (Q3, speculation off, graphs replayed).** From the start of a token's retirement frame to the start of
  the next token's replay frame: 170-180 us, of which the retirement 36, the selection 20, the admission 85, and the hops
  about 30. The token is 17 ms of device time, so the whole boundary is 1%. Speculative steps have three such boundaries,
  about 0.5 ms of 20 ([CUDA_GRAPHS.md](../kernels/CUDA_GRAPHS.md)). Where a boundary frame runs matters little: the
  retirement is 27% slower on an E-core (42 against 33 us), the admission is the same on either, and the replay frame's
  0.5-0.9 ms check of every stage (E-core: twice that) runs behind the launched graph.
- **Frame-by-frame quanta.** A 64-token prefill is 756 stage frames: the host submits all of them in 18-30 ms, and the
  device finishes 31-45 ms later. Stage hops are hidden by the device.
- **Flash-Next decode.** Per layer, from the end of the router stage to the first fetch starting: 200 us, of which the
  device and the driver callback 155 and the host chain (completion-edge frame, fetch publication, owner hop) 41. The loads
  themselves, 0.4-0.6 ms of read per miss, set the token's pace.

Found by that probe and fixed: **the expert cache's fetch retries.** A fetch that found the cache full published itself
again: 20-25 million frames in a 66 s Flash-Next screen, all routed to the cache owner's one worker, which they kept
busy; the frames that free the resource (a release, a retirement) waited behind them, 0.4-1 ms in prefill and cold decode.
The cache's bookkeeping is now a source of the lattice (`residency/expert_cache/source.rs`), attached like the device
completions: requests, releases, read completions and copy retirements are records posted to its owner, the one worker
that polls applies them in order, and a request the cache cannot serve waits with its cause and is asked again once per
change of that cause, with nothing running again meanwhile. Paired against the retrying fetch (two forks each): fetches that found the cache full 34,000
against 120 a token in a 512-token prefill and 7,400 against 26 in a 4096-token one; the hop from a copy's retirement to its
application 18-56 us against 8-9; prefill unchanged (113-115 / 494-545 tokens/s); decode at 4096 tokens 61.4 and 60.3 ms
against 54.3 and 52.0 cold, 44.5 and 47.7 against 39.1 and 38.6 warm; at 64 tokens within noise.

Tried and not kept, each against a paired control (differences inside the run-to-run spread of 2-4%):

- **A frame published into the publishing worker's own cache** (an executor with a handle to its worker; the fetch
  retries): the owner worker is saturated by the retries either way, so a retry costs about the same.
- **Placing ordered frames on performance cores** (`RoutingPolicy::CacheLocal` on every ordered frame, so the owner chains
  left their hash-chosen E-cores): neutral. Flash-Next's cache owner ran on an E-core, the admission on another.
- **A custom executor.** With the redundant owner hop gone, no chain of frames with one ordered hash is left for a worker to
  continue inline, and the unordered chains (retirement, selection, replay) are hops of 5-8 us that the scheduler already
  serves from the same worker's next cycle.

Known costs this leaves:

- **One owner for the workspace.** The admission (85 us) serializes all sessions' starts on one owner: capacity of about
  11,000 admissions a second, far beyond what one GPU's decode needs.
- **Per-quantum fingerprints.** A third of the admission is the capture key (`CaptureKey`), which holds the allocation identity of
  every state address the quantum's launches bake in; on the earlier host implementation each was a concurrent-map lookup
  (about 30 us a quantum). The Rust engine has not been re-measured.

## No locks on the hot path

Nothing that runs per quantum or per token takes a lock, waits blocking or starts a thread. State is
confined by routing every frame that touches it to its owner (the workspace's, the prefix cache's) or by making it a source
that one poller at a time owns (the expert cache), or behind a mailbox claim that one thread at a time holds. Device
completions arrive as frames, and order comes from edges and from a sequence's admission order (`Sequencer`). The hot path
holds no lock block, blocking wait or thread start: the engine's blocking calls live in named lifecycle and tool functions
(opening lanes and pools, closing, the blocking caller of `generate`) and in the startup loaders.
