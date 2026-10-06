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

`EuhedralInferenceRuntime` is the root source of the lattice. It owns an `InferenceLake`: several queue ingest sinks (EE's
`QueueIngestSink` over partitioned MPSC queues), each attached to the lattice as an upstream source of its own. Every producer
throws frames into the lake: a stage that made its successors ready, a driver callback (which may only enqueue), a request
thread. A frame goes into the sink its routing hash selects, so the frames of one lane keep to one sink and the frames that may run
in parallel spread over all of them. Workers pull from many sources at once instead of queuing at one: producers and consumers share
no hot cache line, and the cores work side by side. Nothing pushes: a frame is made runnable by being published, and a worker takes
it when it has capacity (a pull-driven system). The owner of the lattice builds the lake and hands it to the runtime, so the API
decides the lattice and the model plugs into it.

## The shape of a step

```
embed -> [PLE] -> attention block -> route ─┬─> shared expert ────────────────────────────┐
  ^                                          └─(route copy retires)─> plan                 │
  |                                                                    │                  v
  |        wave 0:  load ──────────────────────────────────────────> wave 0 ─> wave 1 ... ─> finish
  |        wave w:  load (after wave w-window was submitted) ───────> wave w                  │
  └─────────────────────────────────────── next layer <──────────────────────────────────────┘  ...  -> head
```

| Piece | Role |
| --- | --- |
| `Qwen4ExecutionPlan` | The model-level resources the stages read (weights, the layers' operators, the expert cache) and the entry point (`start`). It builds `Qwen4Shape`s, one per row capacity (a decode token, a short chunk, a full chunk) and diagnostic variant, and serves one quantum at a time through a completion chain. |
| `Qwen4Shape` | A `GraphShape`: the static topology, the frame of each stage, and the storage a graph of the shape owns. |
| `Qwen4Stages` | The stage frames. Each is a `StageFrame` (the dense model's execution stage), submitting its device work to the lane it was given or doing host work. |
| `Qwen4GraphStorage` | The device workspace of one row capacity and the MoE block resources (`Qwen4MoeLayer`); the graphs of a capacity take turns on it through leases. |
| `Qwen4Quantum` | A `StageQuantum`: what the stages read, whether the step stopped, and the terminal work after retirement (commit the sequence, close leases a stopped block still holds, report to the listener). |
| `EuhedralInferenceRuntime` | Unchanged lifecycle: admits a quantum to an idle graph of its shape (building one when none is idle), publishes its roots, recycles the graph at retirement. `Qwen4Runtime` uses it with no dense plan. |
| `ExpertItem` | One expert of a block as a frame: claim the expert in the cache, read, submit the copy on the lane's copy stream, hold the lease that carries the copy's marker; a second frame of the lane runs when the copy retired. |
| `InferenceLake` / `FrameLake` | The pool of ready work: queue ingest sinks, each an upstream source of the lattice, that every producer publishes into. It counts the units it carries and completes once they all ended. |
| `HostFrames` / `HostTasks` | Where the work around the graph runs: the generation chain, tokenization, the expert hierarchy's other asynchronous work. `run` from a worker or ordinary thread, `runFromCallback` from a CUDA driver thread (enqueue only). |

### The step is a stream

- **The chain keeps its lane.** Every stage of a step that must follow the one before it (the attention block, routing, the plan,
  the waves, the finish, the head) and the device-completion frames between them share one routing hash per graph, so the lattice
  places them on the same lane: a frame that ends publishes its successor into the lane's sink and the worker that ended it takes
  it next. The side branch (the shared expert) and everything else that may run in parallel is spread over the lanes.
- **Experts are items.** The plan stage spawns the block's experts as frames, one per expert (`ExpertItem`, recycled through a
  `FrameManager`), in lanes: expert `p` runs on lane `p % lanes`, one at a time per lane, and the lanes (one pinned staging slot and one
  copy stream each) spread over the workers. An item finds its expert in the cache (a resident one is held at once) or reads the
  record into its staging slot and submits the host-to-device copy on its lane's own copy stream. The lease it holds carries the
  marker the copy records behind itself: the wave's kernels wait for that marker on the device, and no host thread waits for the
  bytes. When the copy retires (a driver callback that publishes a second frame of the lane) the staging slot is free and the lane's
  next expert becomes runnable.
- **Waves overlap.** A wave's stage runs as soon as every one of its experts was taken or its copy submitted. The experts of a
  later wave start as the cache's slots free up: an item is runnable when its lane's previous expert ended and the wave `window`
  before its own was submitted and retired. The bounds hold by construction; nothing queues, parks or holds a permit.

### Edges are the synchronization

- A **submission edge** is satisfied when the producer's device work was queued on its lane; the device's own ordering (and a
  marker when the consumer is on another lane) sequences the work. Most edges are this kind.
- A **device-completion edge** (route → plan) is satisfied when the producer's work retired: its driver callback only enqueues the
  frame that follows. It exists because the host reads the router's choice.
- A **host stage** (plan, load) submits nothing to a lane and orders nothing on the device. A **deferred** stage (load) ends when its
  asynchronous work does: it is the arrival point of its wave's items and completes on whichever item reports the last one.
- A **spawned frame** (an expert item) is a part of a stage's work that runs on its own. It counts as live work of the quantum, which
  cannot retire before it ended.
- A stage that **has nothing to do** (a wave beyond what the block uses) completes in place on the thread that satisfied its last
  edge, still joining its predecessors' device order so that its successors wait for what it stood after.
- The **gates of the items** (the lane's previous expert ended; the wave `window` before was submitted and retired) bound the loads
  outstanding by the pinned staging slots and the cache's slots *by construction*. Nothing queues and nothing parks: a request past a
  bound fails loudly, and the sizes (a wave holds at most `slots / (window + 1)` experts, at most 32) keep it from happening.

The only blocking left is where an external API is itself synchronous and is isolated in one frame: the positional read of one
expert record, the host gather of the per-layer embedding's n-gram rows, and `stream.recover` on a failure path.

## Failure, cancellation, shutdown

A failure or cancellation stops new stages (a stage that has not started does not run) and lets what is in flight end: outstanding
expert loads still complete (their leases are closed on arrival), submitted kernels retire at the quantum's single boundary, the
chunk's KV state is discarded by the attention stage's retirement hook, uploads are released, and the listener sees the failure.
Nothing is interrupted. Closing the runtime stops admission, waits for every accepted step to retire, then releases the graphs and
their workspaces and the storage (the lattice still runs the copies' completions) before the host work drains and detaches.

## Diagnostics

While timings or observers are set the plan uses a variant of the same shape that splits each layer's attention block and puts a
device-completion edge between the pieces, so the wall time of each component and the state after each layer can be read. It is for
tests and measurement; production uses the plain shape.

## Tests

`Qwen4LatticeHostTest` (real lattice): host work is frames on an attached source; independent work uses several workers; parked
continuations occupy no worker; a driver-callback publication runs on a worker; close drains chains and detaches.
`StageGraphHostStageTest`: host stages, stages completed in place, and deferred stages (what each does to the edges, the lanes and the
quantum's retirement). `InferenceLakeTest`: sinks attach once and lazily, frames route by hash, completion waits for every admitted
unit. `ExpertCacheStreamTest`: a lease before its bytes arrive carries the copy's marker, the slot stays unevictable until the copy
retired, a device failure unmaps the expert. `ExpertCacheAsyncTest`: coalescing, bounds that fail loudly, failures, prefetch, close.
`Qwen4ArchitectureTest`: a source guard against private executors, pools, virtual threads, async `CompletableFuture` forms and
blocking waits in the execution path, and that a step is the shape of a graph (no executor or step machine).
`Qwen4EngineCudaIntegrationTest`: one lattice worker generates the reference continuation; host work progresses during a generation;
cancelling in the middle of a prefill leaves no lease, pin or step; closing leaves nothing attached.
`Qwen4ModelFixtureCudaIntegrationTest` and `Qwen4GreedyAgreementCudaIntegrationTest` run the shape against the reference layer by layer
and token by token.
