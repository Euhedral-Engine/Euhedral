# Documentation

Start with the repository [quick start](../README.md#quick-start) to convert a checkpoint, build the server and send a request.

## Using the server

| Guide | Contents |
|---|---|
| [API compatibility](API_COMPATIBILITY.md) | Supported request fields across all three APIs |
| [Anthropic Messages](ANTHROPIC_API.md) | Requests, responses and streaming for Anthropic clients |
| [OpenAI Responses](RESPONSES_API.md) | Stateless requests, responses and streaming |
| [Operations](OPERATIONS.md) | Configuration, startup, health, shutdown and metrics |
| [Reasoning](REASONING.md) | Thinking controls and reasoning output |
| [Tool calling](TOOL_CALLING.md) | Function tools, tool choice and conversation replay |
| [Structured output](STRUCTURED_OUTPUT.md) | JSON output and schema enforcement |
| [Prefix cache](PREFIX_CACHE.md) | Reusing prompt prefixes and configuring the cache |

## Implementation and measurement

| Folder | Contents |
|---|---|
| [Architecture](architecture/README.md) | System diagrams, execution model and multi-sequence execution |
| [Evaluation](evaluation/README.md) | Testing, benchmarking, quality measurement and release checks |
| [Quantization](quantization/README.md) | Artifact formats, compression, mixed quantization and residency |
| [Speculation](speculation/README.md) | MTP and DFlash2 contracts, kernels, measurements and drafter training |
| [Kernels](kernels/README.md) | Attention, tensor-core prefill, native FP4 and CUDA graphs |
| [Flash-Next](flash-next/README.md) | Model execution, artifacts, storage tiers, expert caching and performance |
| [NVIDIA hardware](nvidia/README.md) | Hardware measurements and the microbenchmark harness |
| [Design specs](superpowers/specs/) | Dated design specifications |
| [Implementation plans](superpowers/plans/) | Dated implementation plans |

Artifact conversion is covered in [tools](../tools/README.md); calibration and drafter training are covered in the
[quantization module](../quantization/README.md). The [native module](../native/README.md) describes the CUDA source layout.
