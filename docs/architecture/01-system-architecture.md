# System architecture

This map shows deployment boundaries and the main responsibilities checked at commit `498f7d2c`. Arrows describe runtime use or artifact flow. The detailed execution diagrams focus on dense Qwen3.8; Qwen4/Flash-Next has its own runtime and expert storage hierarchy.

```mermaid
flowchart TB
    Clients[OpenAI and Anthropic clients] --> API[api: HTTP controllers and generation service]
    Config[YAML settings and environment overrides] --> Settings[settings loader: validate before the server starts]
    Settings --> Unpack[Native files: locate the CUDA library and kernel sources in the bundle]
    Unpack --> API
    Settings --> Engine
    API --> Engine[core: InferenceEngine]
    Bench[benchmark: standalone measurement entry point] --> Engine
    Engine --> Runtime[Model runtime: dense or Flash-Next]
    Engine --> Lattice[Process-wide ControlPlaneLattice]
    Runtime --> Frames[Generation and execution frames]
    Frames --> Lattice
    Runtime --> Plan[Dense execution plan or Flash-Next execution]
    Plan --> GPU[CUDA bindings and GPU resources]
    GPU --> Native[native: CUDA kernels and driver operations]
    Artifacts[EDRL artifact: weights, configuration, tokenizer and chat template] --> Engine
    Quant[quantization: calibration and rounding] --> Codes[Codes or EXL3 checkpoint]
    Codes --> Converter[tools: checkpoint converter]
    Checkpoint[Source checkpoint] --> Quant
    Checkpoint --> Converter
    Converter --> Artifacts
```

The engine owns the lifecycle of the process fabric and sessions. Dense loading opens the GPU and model before starting the lattice; Flash-Next starts the lattice and lets the Flash-Next runtime own its GPU and storage resources. The lattice schedules host work; CUDA streams and events govern device work.

The server reads one settings tree before starting the HTTP listener. YAML and per-key environment overrides supply model, sampling, batching, memory, host, kernel and HTTP settings. Invalid settings stop startup. The server bundle carries the CUDA binding, CUDA runtime, NVRTC, kernel sources, headers and llguidance next to the binary, and the server finds them there. The host supplies the NVIDIA driver.

Both model runtimes load the tokenizer and chat template from the artifact's tokenizer section; a missing or invalid section fails startup with a diagnostic. Serving needs no separate checkpoint directory. Sampling defaults come from settings, with request values overriding them. Execution strategy and weight formats come from the artifact, while batching, memory limits and kernel thresholds come from settings.

Quantization is an offline application. It reuses inference kernels and lattice machinery, then supplies rounded codes to the Python converter. It is not a request stage in the serving process. `native` and `tools` are repository directories rather than crates in the Rust workspace.

## Implementation anchors

- [settings](../../rust/crates/settings/src/lib.rs) and [loader](../../rust/crates/settings/src/loader.rs): the settings tree, validated before the server starts.
- [server](../../rust/crates/server/src/lib.rs) and [engine start](../../rust/crates/server/src/engine.rs): process startup, the engine and shutdown.
- [startup](../../rust/crates/engine/src/startup.rs) and [runtime](../../rust/crates/engine/src/runtime.rs): model load, the lattice, the lakes and the lanes.
- [tokenizer section](../../rust/crates/edrl/src/tokenizer_section.rs): embedded tokenizer validation and upgrade diagnostics.
- [engine model](../../rust/crates/engine/src/model/mod.rs) and [flash](../../rust/crates/engine/src/flash): dense and Flash-Next model ownership.
- [Quantization workflow](../../quantization/README.md) and [converter](../../tools/convert_checkpoint.py).

This diagram omits individual layers, kernel variants, request state transitions and physical memory layout. Those are covered by the focused diagrams in this directory.
