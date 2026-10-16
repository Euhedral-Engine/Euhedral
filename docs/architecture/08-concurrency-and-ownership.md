# Concurrency and ownership

This diagram distinguishes host frame routing, mutable-state ownership and device completion. It describes the shared runtime machinery and the dense execution that uses it. An owner is an ordering domain enforced by frames, not a dedicated operating-system thread.

```mermaid
flowchart TB
    Request[Request planning and host tasks] --> Lake[InferenceLake: hash-selected ingest sinks]
    Admit[Admit frames ordered by WorkspaceOwner.HASH] --> Lake
    Callback[Driver callbacks and asynchronous completion frames] --> Lake
    Lake --> Fabric[ControlPlaneLattice workers]
    Fabric --> Owner[WorkspaceOwner binds graphs one at a time]
    Owner --> Shared[Order shared-buffer first users after prior last users]
    Owner --> Sequence[SequenceOwner orders carried state within each sequence]
    Fabric --> Stage[StageFrame: one executing thread per published frame]
    Stage --> Edge[Atomic predecessor arrivals]
    Edge --> Ready[Last arrival publishes successor]
    Ready --> Lake
    Stage --> GPU[LanePool streams: event waits and kernel submission]
    GPU --> Callback
    Callback --> Retire[Quantum retirement and ordered sequence commit]
    Retire --> Select[Generation Select then Admit or Finish]
    Select --> Lake
```

## Ownership contracts

| Resource or state | Owner and ordering rule |
|---|---|
| Process lattice lifecycle | `InferenceEngine` claims exclusive process ownership, starts it and stops it during close. |
| Sessions and persistent sequence state | Engine tracks sessions; a generation's frame chain serializes its decisions. Sequence commit follows admitted work order. |
| Workspace access records and pass assembly | Frames ordered on `WorkspaceOwner.HASH`; graph binding adds dependencies for buffer reuse. |
| Carried sequence state | `SequenceOwner` records prior users per key, accessed during workspace-owner admission; separate sequences have separate records. |
| Reusable graph and storage | Runtime graph pool; returned after quantum retirement before the outcome is published. |
| Stage mutable fields | One executing thread after publication; concurrent predecessors update arrival state before publication. |
| CUDA streams and transfer lane | Runtime `LanePool`, shared across graphs; CUDA event waits enforce ordering across streams. |
| Host buffers used by DMA | Retained until GPU completion is proven; Java submission completion alone is insufficient. |

A sink routes frames by hash. The default lake uses eight sinks with one producer partition each. Equal hashes keep an ordered chain on one sink; different chains can progress concurrently. Increasing producer partitions removes ordering across producers, so the default ordering assumption matters to owner admissions. Worker placement and CUDA stream choice are separate decisions.

Stages satisfy submission edges after successful device submission. Deferred stages can instead satisfy their edges when asynchronous work completes. Failures are recorded on the quantum and conclude through its continuation, including rejection handling; they must not strand a generation or leak its admitted lake unit.

The lake counts admitted units such as generations, quanta and host tasks. Graceful completion closes admission and completes its sources after all units terminate. During normal idle periods it can detach its sinks and watch, then attach a fresh set for later work. Engine shutdown closes tracked sessions, stops the model's execution, closes the lattice, and releases model/GPU resources in the engine's shutdown order.

## Implementation anchors

- [InferenceEngine](../../core/src/main/java/io/euhedral_execution/inference/core/InferenceEngine.java): exclusive lifecycle claim and `close`.
- [InferenceLake](../../core/src/main/java/io/euhedral_execution/inference/core/runtime/graph/InferenceLake.java): sinks, admitted-unit counts and idle detach.
- [WorkspaceOwner](../../core/src/main/java/io/euhedral_execution/inference/core/runtime/graph/WorkspaceOwner.java), [SequenceOwner](../../core/src/main/java/io/euhedral_execution/inference/core/runtime/graph/SequenceOwner.java) and [StageFrame](../../core/src/main/java/io/euhedral_execution/inference/core/runtime/graph/StageFrame.java).
- [Generation](../../core/src/main/java/io/euhedral_execution/inference/core/generation/Generation.java), [Admit](../../core/src/main/java/io/euhedral_execution/inference/core/generation/Admit.java) and [Quantum](../../core/src/main/java/io/euhedral_execution/inference/core/model/qwen38/Quantum.java).
- [GPU buffer contracts](../../core/src/main/java/io/euhedral_execution/inference/core/gpu/ExecutionGpu.java) and [frame model](../FRAME_MODEL.md).

This diagram omits the external Euhedral-Execution library's internal routing implementation and Flash-Next expert-cache ownership details. It documents the contracts used by this repository.
