# CUDA execution

This diagram shows dense linear dispatch and device ordering. Artifact format, layout, row count and the quantum's numerical contract select the route. Attention, GDN and elementwise stages use the same stream and dependency machinery but have their own kernels.

```mermaid
flowchart TB
    Instruction[ExecutionPlan instruction and bound tensor handles] --> Format[LinearFormats selects format and layout implementation]
    Format --> NV[Nvfp4Format: plain or SD4]
    Format --> TR[TrellisFormat: EXL3 payload and companions]
    Format --> Other[Grouped Q3/Q4/Q5, P2E2 or BF16]
    NV --> Binding[CUDA launch table with row-exact selection]
    TR --> TRoute{Row count and numerical mode}
    TRoute -->|Row-exact or small rows| TMMA[Rotations, FP16 operands with FP32 sums, epilogue]
    TRoute -->|Whole-K prefill within tensor-core threshold| TFP8[Block-scaled FP8 when available, otherwise FP16 accumulation]
    TRoute -->|Above tensor-core threshold| TBF[Decode weights to BF16, GEMM, epilogue]
    TMMA --> Table[KernelTable / table kernel launch]
    TFP8 --> Table
    TBF --> Table
    Other --> Binding
    Table --> Binding
    Binding --> ABI[Rust cuda crate to euhedral_cuda C ABI]
    ABI --> Native[Native route selection and CUDA launches]
    Lane[stage frame selects a lane pool stream] --> Binding
    Pred[Predecessor markers] --> Wait[Cross-stream event waits]
    Wait --> Lane
    Native --> Marker[Record submission marker]
    Marker --> Successor[Publish ready successor frames]
    Native --> Completion[Asynchronous completion and retirement]
```

The `cuda` crate binds the stable C ABI. Device addresses are opaque longs. Kernel launches use the stream selected for the calling stage; direct operations without a selected stream are synchronous diagnostic paths.

NVFP4 dispatch distinguishes its plain and compressed scale layouts and selects compatible kernels. Production multi-row prefill can use accelerated numerical routes; verification requires row-exact execution. Optional native routes can report unavailable so a supported fallback can run. Row-exact verification and diagnostic scalar exact numerics are distinct contracts.

Trellis weights carry the packed object plus rotation and segment companions. Small-row execution uses rotations, FP16 operands and FP32 sums on the trellis tensor-core path. At the whole-K threshold, production prefill uses block-scaled FP8 kernels when available, otherwise the FP16 whole-K kernels; verification retains FP16 operands and FP32 sums to match one-row execution. Above `kernels.trellis.tensor-core-max-tokens`, non-row-exact launches decode weights into BF16 scratch and run GEMM. The defaults keep the full 512-row prefill chunk on the tensor-core route, with whole-K blocks from 64 rows and wider blocks from 128. `TrellisSettings` derives thresholds from the settings tree and device geometry.

A stage first awaits predecessor markers from other streams, then submits work and satisfies submission edges. This permits successors to enqueue GPU work before earlier device work finishes, while preserving CUDA dependencies. Retirement waits for completion before publishing the quantum's outcome and recycling storage. Eligible shapes can replay a captured CUDA graph instead of repeating stage submissions.

## Implementation anchors

- [formats](../../rust/crates/engine/src/formats/mod.rs): the [NVFP4](../../rust/crates/engine/src/formats/nvfp4.rs), [trellis](../../rust/crates/engine/src/formats/trellis.rs) and [Q3](../../rust/crates/engine/src/formats/q3.rs) linear formats.
- [cuda crate](../../rust/crates/cuda/src/lib.rs), [launch table](../../rust/crates/cuda/src/launch/table.rs) and the [C ABI](../../native/include/euhedral_cuda.h).
- [Native implementations](../../native/src), especially `trellis`, `nvfp4`, `nvfp4_native`, `q3` and `reference`.
- [frames](../../rust/crates/engine/src/frames.rs), [runtime lanes](../../rust/crates/engine/src/runtime.rs) and [CUDA graphs](../kernels/CUDA_GRAPHS.md).

This is a dispatch map rather than an exhaustive kernel catalog. It omits block geometry, tuning measurements and native build steps.
