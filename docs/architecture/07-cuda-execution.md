# CUDA execution

This diagram shows dense linear dispatch and device ordering. Artifact format, layout, row count and the quantum's numerical contract select the route. Attention, GDN and elementwise stages use the same stream and dependency machinery but have their own kernels.

```mermaid
flowchart TB
    Instruction[ExecutionPlan instruction and bound tensor handles] --> Format[LinearFormats selects format and layout implementation]
    Format --> NV[Nvfp4Format: plain or SD4]
    Format --> TR[TrellisFormat: EXL3 payload and companions]
    Format --> Other[Grouped Q3/Q4/Q5, P2E2 or BF16]
    NV --> Binding[CudaGpuMemory dispatch with row-exact selection]
    TR --> TRoute{Row count and numerical mode}
    TRoute -->|Decode / VERIFY| TMMA[Rotations, trellis tensor-core route, epilogue]
    TRoute -->|Large prefill| TBF[Decode weights to BF16, GEMM, epilogue]
    TMMA --> Table[KernelTable / table kernel launch]
    TBF --> Table
    Other --> Binding
    Table --> Binding
    Binding --> ABI[Java FFM to euhedral_cuda C ABI]
    ABI --> Native[Native route selection and CUDA launches]
    Lane[StageFrame selects a LanePool stream] --> Binding
    Pred[Predecessor markers] --> Wait[Cross-stream event waits]
    Wait --> Lane
    Native --> Marker[Record submission marker]
    Marker --> Successor[Publish ready successor frames]
    Native --> Completion[Asynchronous completion and retirement]
```

`CudaGpuMemory` binds the stable C ABI through Java's Foreign Function and Memory API. Device addresses are opaque longs. Kernel launches use the stream selected for the calling stage; direct operations without a selected stream are synchronous diagnostic paths.

NVFP4 dispatch distinguishes its plain and compressed scale layouts and selects compatible kernels. Production multi-row prefill can use accelerated numerical routes; verification requires row-exact execution. Optional native routes can report unavailable so a supported fallback can run. Row-exact verification and diagnostic scalar exact numerics are distinct contracts.

Trellis weights carry the packed object plus rotation and segment companions. Small-row execution uses rotations and the trellis tensor-core path. Large prefill can decode the object into BF16 scratch and run GEMM. `TrellisFormat` keeps verification on the row-exact tensor-core route; thresholds and scratch requirements come from its settings and object geometry.

A stage first awaits predecessor markers from other streams, then submits work and satisfies submission edges. This permits successors to enqueue GPU work before earlier device work finishes, while preserving CUDA dependencies. Retirement waits for completion before publishing the quantum's outcome and recycling storage. Eligible shapes can replay a captured CUDA graph instead of repeating stage submissions.

## Implementation anchors

- [LinearFormats](../../core/src/main/java/io/euhedral_execution/inference/core/artifact/format/LinearFormats.java), [Nvfp4Format](../../core/src/main/java/io/euhedral_execution/inference/core/artifact/format/Nvfp4Format.java), [TrellisFormat](../../core/src/main/java/io/euhedral_execution/inference/core/artifact/format/TrellisFormat.java) and [TrellisSettings](../../core/src/main/java/io/euhedral_execution/inference/core/artifact/format/TrellisSettings.java).
- [CudaGpuMemory](../../core/src/main/java/io/euhedral_execution/inference/core/gpu/CudaGpuMemory.java), [KernelTable](../../core/src/main/java/io/euhedral_execution/inference/core/gpu/KernelTable.java) and [C ABI](../../native/include/euhedral_cuda.h).
- [Native implementations](../../native/src), especially `trellis`, `nvfp4`, `nvfp4_native`, `q3` and `reference`.
- [StageFrame](../../core/src/main/java/io/euhedral_execution/inference/core/runtime/graph/StageFrame.java), [LanePool](../../core/src/main/java/io/euhedral_execution/inference/core/runtime/graph/LanePool.java) and [CUDA graphs](../CUDA_GRAPHS.md).

This is a dispatch map rather than an exhaustive kernel catalog. It omits block geometry, tuning measurements and native build steps.
