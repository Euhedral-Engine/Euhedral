# Flash-Next on the lattice

Flash-Next has no scheduler, executor or threads of its own. A step (a prefill chunk or a decode token) is the **shape of a
graph of stage frames** that the lattice's runtime instantiates, runs and recycles, exactly as it does for the dense model. The
shape defines the work: its stages, the edges between them, and what each stage does. The lifecycle belongs to the lattice: a
frame's completion is what makes its successors available, and a worker with capacity takes them first come, first served.

```
request ──> generation chain ──> quantum of a Qwen4Shape ──> stage frames (embed, layers, head)
             sampling, text       admitted to the runtime     each layer: attention, route, plan,
             (continuations)      one graph per quantum       loads and waves, finish
                   │                       │                          │
                   └──────── frames on lattice workers; CUDA driver callbacks only enqueue ────────┘
```

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
| `HostFrames` / `HostTasks` | Where the work around the graph runs: expert reads, copy completions, the generation chain, tokenization. `run` from a worker or ordinary thread, `runFromCallback` from a CUDA driver thread (enqueue only). |

### Edges are the synchronization

- A **submission edge** is satisfied when the producer's device work was queued on its lane; the device's own ordering (and a
  marker when the consumer is on another lane) sequences the work. Most edges are this kind.
- A **device-completion edge** (route → plan) is satisfied when the producer's work retired: its driver callback only enqueues the
  frame that follows. It exists because the host reads the router's choice.
- A **host stage** (plan, load) submits nothing to a lane and orders nothing on the device. A **deferred** stage (load) ends when its
  asynchronous work does: it asks the cache for every expert of its wave and completes on whichever frame delivers the last lease
  (a resident expert's lease arrives before the stage returns).
- A stage that **has nothing to do** (a wave beyond what the block uses) completes in place on the thread that satisfied its last
  edge, still joining its predecessors' device order so that its successors wait for what it stood after.
- The **window edges** (a wave's load starts after the wave `window` before it was submitted) bound the loads outstanding by the
  cache's slots and the pinned staging slots *by construction*. Nothing queues and nothing parks: a request past a bound fails
  loudly, and the sizes (a wave holds at most `min(cache slots, staging slots) / window` experts) keep it from happening.

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
quantum's retirement). `ExpertCacheAsyncTest`: coalescing, bounds that fail loudly, failures, prefetch, close.
`Qwen4ArchitectureTest`: a source guard against private executors, pools, virtual threads, async `CompletableFuture` forms and
blocking waits in the execution path, and that a step is the shape of a graph (no executor or step machine).
`Qwen4EngineCudaIntegrationTest`: one lattice worker generates the reference continuation; host work progresses during a generation;
cancelling in the middle of a prefill leaves no lease, pin or step; closing leaves nothing attached.
`Qwen4ModelFixtureCudaIntegrationTest` and `Qwen4GreedyAgreementCudaIntegrationTest` run the shape against the reference layer by layer
and token by token.
