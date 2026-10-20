# Flash-Next on the lattice

Flash-Next has no scheduler, executor or threads of its own. A step (a prefill chunk or a decode token) is the **shape of a
graph of stage frames** that the lattice's runtime instantiates, runs and recycles, exactly as it does for the dense model. The
shape defines the work: its stages, the edges between them, and what each stage does. The lifecycle belongs to the lattice: a
frame's completion is what makes its successors available, and a worker with capacity takes them first come, first served.

```
producers                          the lake                            consumers
request threads ─┐             ┌─ ingest sink 0 ─┐
stage frames    ─┼─ publish ──>├─ ingest sink 1 ─┼─ upstream sources of the lattice ──> workers pull ──> frames run
driver callbacks ─┘  (by hash)  └─ ingest sink n ─┘                                      (doFinally publishes what is next)
```

`Qwen4Runtime` owns the root source of the lattice, an `InferenceLake` (`EuhedralInferenceRuntime.newLake`): several queue ingest sinks (EE's
`QueueIngestSink` over partitioned MPSC queues), each attached to the lattice as an upstream source of its own. Every producer
throws frames into the lake: a stage that made its successors ready, a driver callback (which may only enqueue), a request
thread. A frame goes into the sink its routing hash selects, so the frames of one lane keep to one sink and the frames that may run
in parallel spread over all of them. Workers pull from many sources at once instead of queuing at one: producers and consumers share
no hot cache line, and the cores work side by side. Nothing pushes: a frame is made runnable by being published, and a worker takes
it when it has capacity (a pull-driven system). The owner of the lattice builds the lake and hands it to the runtime, so the API
decides the lattice and the model plugs into it.

## The shape of a step

```
n-gram ids -> gather in parts ──┐
embed ──────────────────────────> PLE -> attention block -> route ─┬─> shared expert ─────────────────────────┐
  ^                                          └─(route copy retires)─> plan ─┬─> fetch e0 ─> expert e0 ──┤
  |                                                                         ├─> fetch e1 ─> expert e1 ──┤
  |                                                                         └─> fetch en ─> expert en ──┴─> finish
  └──────────────────── next layer <─────────────────────────────────────────────────────────────────────────┘ ... -> head
```

| Piece | Role |
| --- | --- |
| `ExecutionPlan` | The model-level resources the stages read (weights, the layers' operators, the expert cache) and the entry point (`start`). It builds `Shape`s, one per row capacity (a decode token, a short chunk, a full chunk) and diagnostic variant, and serves one quantum at a time through a completion chain. |
| `Shape` | A `GraphShape`: the static topology, the frame of each stage, and the storage a graph of the shape owns. |
| `Stages` | The stage frames. Each is a `StageFrame` (the dense model's execution stage), submitting its device work to the lane it was given or doing host work. |
| `Workspace` | The device workspace of one row capacity and the MoE block resources (`MoeLayer`); the graphs of a capacity take turns on it through leases. |
| `Quantum` | A `StageQuantum`: what the stages read, whether the step stopped, and the terminal work after retirement (commit the sequence, close leases a stopped block still holds, report to the listener). |
| `EuhedralInferenceRuntime` | Unchanged lifecycle: admits a quantum of any model to an idle graph of its shape (building one when none is idle), publishes its roots, recycles the graph at retirement. `Qwen4Runtime` builds it with `Lanes.of(LANES)`. |
| `ExpertCache` / `ExpertCacheShard` | The slab, the markers, the host store and the transfer; the directory, the slots' states and the recency. No lock, no wait. |
| `ExpertCacheOwner` | The owner of the cache's bookkeeping (and the host tier's), a source of the lattice: what changes it is a record posted to a lock-free queue, and a worker that polls the source applies the records in order, one poller at a time. |
| `ExpertLoad` | One miss as frames: the read in parts and their join, or the copy out of RAM; the owner's submit and retire. |
| `Join` | The fan-in of frames spawned at run time: the last arrival publishes the continuation. |
| `AsyncReads` | File reads that hold no worker: a frame submits the read to io_uring, and the read's completion is a frame that the reads' sink emits when the workers poll it. |
| `InferenceLake` / `FrameLake` | The pool of ready work: queue ingest sinks, each an upstream source of the lattice, that every producer publishes into. It counts the units it carries and completes once they all ended. |
| `HostFrames` / `HostTasks` | Where the work around the graph runs: tokenization, the expert hierarchy's other asynchronous work. The generation's own frames (`Admit`, `Select`, `Finish`) are thrown into the lake by the session's `GenerationFrames`. `run` from a worker or ordinary thread, `runFromCallback` from a CUDA driver thread (enqueue only). |

### The step is a stream

- **No frame is ordered unless its state needs it.** The stages and the device-completion frames are ordered by the graph's
  edges, not by a routing hash: each has a hash of its own, so whichever worker is free runs it. The expert cache's bookkeeping
  needs no ordered frame either: it belongs to a source (below).
- **Every expert is its own branch.** The plan stage groups the block's pairs by expert and copies the block's description to
  the device. For each expert the block names there is a fetch stage and an expert stage. The fetch takes the expert from the
  cache; the expert stage runs its kernels over its own work items as soon as its expert is held, whatever the other experts are
  doing, and closes its lease behind a marker of its lane. The finish adds the experts' outputs in ascending expert order (the
  order the sum needs, on the device, after all of them) and combines the sum with the shared expert's gated output. A layer has
  as many fetch and expert stages as its largest block can name (ten for a decode token); the plan stage learns how many a block
  uses, and the rest complete in place.
- **The cache's state is changed only by the source that owns it.** A fetch stage posts a request to the cache's owner and ends
  when the owner answers: a resident expert is a lease at once; a missing one reserves a slot, makes the lease and starts its load,
  and the stage ends when the load has submitted the copy and hands it the lease. When every slot that could hold the expert is
  pinned, or the staging buffers or the disk's reads are all in use, the request waits in the owner, which asks the cache again
  only when something was given back (a lease released, a read ended, a copy retired or submitted, a tier slot settled); the
  requests that wait for the same staging buffer, read or shard are asked once per change, not once each. No other expert waits
  for it, and no frame runs again meanwhile: a poll of a source with nothing to do returns at once. A lease closing on an expert
  stage's thread posts a release, a copy's retirement (a driver callback, which may only enqueue) posts a retire record, a read's
  completion posts the tier's; the owner applies them in order when a worker polls it.
- **A miss is frames, and no frame waits for the disk.** The fetch reserves the slot and makes the lease (the slot's address and
  the marker its copy will record). A record read from the artifact is read in parts, page-aligned ranges read straight into their
  destination: each part's frame submits its read to the kernel (`AsyncReads`, io_uring) and ends, and the read's completion is a
  frame of its own that the workers find when they poll the reads' sink, as a device completion is a frame its driver callback
  publishes (a fill's part then copies its range from the tier slot into the staging buffer). The parts join into the frame that
  submits the device copy. A hit in the host tier is one frame that copies the record out of RAM and submits. Whichever frame
  submits the copy hands the lease to the fetch; the expert's kernels wait for the marker on the device, and no host thread waits
  for the bytes. The copy's retirement publishes the owner's retire frame, which makes the slot resident, settles the host tier
  and gives the staging buffer back.
- **Staging is the store's detail.** A load takes a pinned staging buffer when it starts writing its record and gives it back when
  its copy retired; the store keeps the free ones in a lock-free list and pins one more when the list is empty, so nothing waits for
  one. The copy stream is the transfer's choice. Neither is visible to the graph.

### Edges are the synchronization

- A **submission edge** is satisfied when the producer's device work was queued on its lane; the device's own ordering (and a
  marker when the consumer is on another lane) sequences the work. Most edges are this kind.
- A **device-completion edge** (route → plan) is satisfied when the producer's work retired: its driver callback only enqueues the
  frame that follows. It exists because the host reads the router's choice.
- A **host stage** (fetch) submits nothing to a lane and orders nothing on the device; a stage that follows it is placed on any
  lane, since there is no lane to continue. A **deferred** stage (a fetch that started a
  load) ends when its asynchronous work does: when the load hands it the lease.
- The **per-layer embedding's host work** is stages too. The n-gram row ids depend on the tokens alone, so computing them and
  gathering the rows from the mapped table are roots of the graph: the ids first, then the gather in parts that each take a
  range of the rows (a gather too short to be worth a frame per part stays in one), while the embedding and the layers before
  the per-layer embedding run. The stage that copies the records to the device and expands them waits for the parts.
- A stage that **has nothing to do** (an expert beyond what the block names) completes in place on the thread that satisfied its last
  edge, still joining its predecessors' device order so that its successors wait for what it stood after.
The only blocking left is where an external API is itself synchronous and is isolated in one frame: the host gather of a part of
the per-layer embedding's n-gram rows (page faults of a mapped file), a copy of a record out of the host tier (a memory copy, which
is work), `stream.recover` on a failure path, and the positional read a part falls back to on a machine without io_uring.

## How the expert path maps onto Euhedral

The expert hierarchy is expressed the way `euhedral-reactor-core` expresses a reactive pipeline on the lattice: describe the
independent work, describe its dependencies, route state to its owner, and let the lattice run it.

| Application meaning | Euhedral form | Reactor integration | Expert path |
| --- | --- | --- | --- |
| A unit of work | A frame per item, published and never run by its producer | `EuhedralOperator` makes each element a `CallbackFrame` | A fetch, a read part, a copy, a submit, a retirement, a release, an expert's kernels |
| Parallel work | A routing hash of its own | `flatMap` randomizes the hash | Every expert is a branch; reads and copies spread over the workers |
| Owner-confined state | A source: what changes it is posted to it, and the one worker that polls it applies the records | `concatMap` keeps the route | The cache's bookkeeping: requests, retirements, releases |
| Fan-in | Count arrivals; the last one continues | `FrameSequencer` marks a frame ready and drains | `Join` of a read's parts; the finish after every expert |
| Continuation | A frame's own end, or a callback that only publishes | `giveToReceiver`, `doFinally` | A part's read completes as a frame; a part arrives at its join; a copy's retirement publishes its frame |
| A full resource | Nothing to pull until it frees | The sink's emit answers `RETRY` | A fetch that finds every slot pinned waits in the owner's source, asked again when something was given back |

Each load counts as a unit of the lake until its copy retired, so closing the runtime waits for it.

## Failure, cancellation, shutdown

A failure or cancellation stops new stages (a stage that has not started does not run) and lets what is in flight end: outstanding
expert loads still complete (their leases are closed on arrival), submitted kernels retire at the quantum's single boundary, the
chunk's KV state is discarded by the attention stage's retirement hook, uploads are released, and the session's `Select` frame sees the failure.
Nothing is interrupted. Closing the runtime stops admission, waits for every accepted step to retire, then releases the graphs and
their workspaces and the storage (the lattice still runs the copies' completions) before the host work drains and detaches.

## Diagnostics

While timings or observers are set the plan uses a variant of the same shape that splits each layer's attention block and puts a
device-completion edge between the pieces, so the wall time of each component and the state after each layer can be read. It is for
tests and measurement; production uses the plain shape.

## Tests

`LatticeHostTest` (real lattice): host work is frames on an attached source; independent work uses several workers; parked
continuations occupy no worker; a driver-callback publication runs on a worker; close drains chains and detaches.
`StageGraphHostStageTest`: host stages, stages completed in place, and deferred stages (what each does to the edges, the lanes and the
quantum's retirement). `InferenceLakeTest`: sinks attach once and lazily, frames route by hash, completion waits for every admitted
unit. `JoinTest`: the last arrival publishes the continuation once. `ExpertCacheShardTest`: a lease before its bytes arrive carries the copy's marker, the slot stays
unevictable until the copy retired, least-recently-used eviction, fences order a refill, shards partition slots and experts.
`ExpertCacheOwnerTest`: a hit publishes nothing, a miss is a copy and the owner's submit and retire, a request for a full cache waits in the source and is asked again only after a release, requests waiting for the same buffer are asked once per change, a stopped quantum's request is abandoned, a read in parts fans out from
its first part and joins into the submit, loads in flight together each have a staging buffer and the store pins more when it runs
out, a fetch that finds every slot pinned changes nothing and succeeds once a release ran, a closed lease reaches the cache only
through the owner's release frame, failures return the slot and the buffer. `ExpertRoutingTest`: each expert's items are
contiguous and every row adds its pairs in ascending expert order. `ExpertCudaIntegrationTest`: the result is the same bits
whatever order the experts run in and however few slots hold them. `FileExpertStoreTest`, `GpuExpertTransferTest`: staging buffers, and one copy stream taking the copies of
many buffers from many threads. The architecture test also guards that the expert path holds no lock, starts no thread, and has no
source or sink of its own, queue of work, admission, waves, lane, or lock around the owner's state.
`ArchitectureTest`: a source guard against private executors, pools, virtual threads, async `CompletableFuture` forms and
blocking waits in the execution path, and that a step is the shape of a graph (no executor or step machine).
`Qwen4EngineCudaIntegrationTest`: one lattice worker generates the reference continuation; host work progresses during a generation;
cancelling in the middle of a prefill leaves no lease, pin or step; closing leaves nothing attached.
`ModelFixtureCudaIntegrationTest` and `GreedyAgreementCudaIntegrationTest` run the shape against the reference layer by layer
and token by token.
