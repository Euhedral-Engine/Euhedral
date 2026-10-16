# Architecture diagrams

Start with the system overview, then follow the concern you are changing. These hand-maintained Mermaid diagrams describe contracts and responsibilities grounded in the code at `22aa9e12`; they are not generated call graphs. The overview includes both model runtimes. Detailed dense execution diagrams explicitly scope themselves to Qwen3.8.

| Priority | Diagram | Main question |
|---|---|---|
| 1 | [System architecture](01-system-architecture.md) | How do API, core, native CUDA, lattice and artifact tooling fit together? |
| 2 | [Request lifecycle](02-request-lifecycle.md) | How does a request move through planning, admission, prefill, generation and response completion? |
| 3 | [Execution engine](03-execution-engine.md) | How do plans, quantum parts, graph dependencies and shared buffers become GPU work? |
| 4 | [Memory architecture](04-memory-architecture.md) | Which resources live on device, in pinned host memory or on disk, and how are they staged? |
| 5 | [Speculative decoding](05-speculative-decoding.md) | How do proposals, row-exact target verification and committed state stay consistent? |
| 6 | [Quantization](06-quantization.md) | How do calibration, format allocation and conversion produce a loadable artifact? |
| 7 | [CUDA execution](07-cuda-execution.md) | How do formats select routes and how are launches synchronized? |
| 8 | [Concurrency and ownership](08-concurrency-and-ownership.md) | Which ordering domain owns mutable state and when is asynchronous work complete? |

Every page has a stated boundary and links to implementation anchors. Mermaid source stays in Markdown so GitHub and compatible editors can render it directly.

## Maintaining the diagrams

When a change alters module boundaries, request flow, graph dependencies, residency, speculative commit semantics or resource ownership, update the corresponding page alongside the implementation. Trace changed arrows through their implementation anchors. State whether an arrow means a call, ownership, data movement or an ordering dependency. Keep conditional routes visibly conditional and preserve the distinction between submission and device completion.

Existing focused design documents under `docs/` supply additional detail and historical measurements. Use current code for defaults and execution branches; dated benchmark results do not establish current policy. The diagram pages do not repeat repository size counts or throughput figures because those drift independently of architecture.

Structural dependency graphs can supplement these behavioral views. If added, generate them for a named module or package with the command and revision recorded, and keep them focused enough to answer a specific dependency question. This documentation adds no graph generator.

For Flash-Next, continue with the existing [execution](../FLASH_NEXT_EXECUTION.md), [residency](../FLASH_NEXT_RESIDENCY.md), [lattice](../FLASH_NEXT_LATTICE.md) and [disk](../FLASH_NEXT_DISK.md) documents. For dense execution, the [frame model](../FRAME_MODEL.md), [MTP contract](../MTP_CONTRACT.md) and [CUDA graphs](../CUDA_GRAPHS.md) expand the key invariants.
