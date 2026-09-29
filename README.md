# Euhedral-Inference

A Java-based ML/model inference engine.

## Project Structure

This is a multi-module Gradle project:

| Module | Purpose |
| ------ | ------- |
| `core` | Core inference engine: model loading, execution, and tensor handling. |
| `api`  | Spring Boot OpenAI-compatible HTTP API. Depends on `core`. |
| `benchmark` | End-to-end benchmark harness (not packaged with the API). Depends on `core`. |

## Requirements

- Java 25 (JDK)
- Gradle 9.6.1 (the committed wrapper handles the rest)
- Zig 0.16.0 for native builds; CUDA 13.1+ headers and libraries are resolved automatically

The `mise.toml` records compatible tool versions and common tasks. Install them or activate your
preferred toolchain manager before using the Gradle wrapper.

## Building

```bash
./gradlew build
```

The API boot JAR builds and packages Linux x86_64 and Windows x86_64 CUDA native products,
including NVRTC `.cu` sources. See [native/README.md](native/README.md) for the target
manifest, pinned NVIDIA development inputs, and target-specific override properties.

## Running

```bash
./gradlew :api:run
```

The API serves `/v1/models` and `/v1/chat/completions` (JSON or SSE streaming). Chat
Completions accepts OpenAI-style function `tools`, `tool_choice` (`auto`, `none`, `required`,
or a named function), `parallel_tool_calls`, and assistant/tool-message replay. When functions
are offered, token sampling is constrained to a JSON `tool_calls` envelope or, for `auto`, a
JSON `content` envelope for an ordinary answer. The latter is returned as normal assistant text,
not as JSON on the wire. The server validates a complete call before returning OpenAI-style
`tool_calls` and `finish_reason: "tool_calls"`. With callable tools, SSE emits tool calls and
ordinary answers only after their JSON envelopes validate, so plain-answer text may be buffered
until generation ends. Without callable tools, plain text streams incrementally.
Pass each returned call ID back in a `role: "tool"` message with its result, then send the
next request. Tool results are JSON-quoted in the model prompt, not executed by the server.
The model may still choose a non-tool answer with `tool_choice: "auto"`; malformed calls
fail rather than being returned as successful tool calls. Sampling enforces JSON syntax and
offered function names, not the full argument schema. Argument checking covers top-level
types, required names, and `additionalProperties: false`; it does not enforce every JSON Schema
constraint (such as nested schemas, enum values, or numeric bounds). Requests for function
`strict: true` are rejected rather than promising full JSON Schema constrained decoding.

## Container

Build the Linux x86_64 image (JDK 25 and Zig 0.16.0 build stage; minimal Java 25 runtime):

```text
docker build -t euhedral-inference:local .
```

Mount your EDRL model and tokenizer checkpoint. They are not included in the image:

```text
docker run --rm --gpus all -p 1738:1738 \
  --mount type=bind,src=/absolute/path/model.edrl,dst=/models/model.edrl,readonly \
  --mount type=bind,src=/absolute/path/tokenizer,dst=/tokenizer,readonly \
  -e EUHEDRAL_INFERENCE_WORKER_CPUS=2-5 \
  -e EUHEDRAL_INFERENCE_MODEL_ID=qwen \
  euhedral-inference:local
```

The defaults are port 1738, `/models/model.edrl`, `/tokenizer`, and
`/opt/euhedral/lib/libeuhedral_cuda.so`. Override those paths with
`EUHEDRAL_INFERENCE_ARTIFACT_PATH`, `EUHEDRAL_INFERENCE_TOKENIZER_DIRECTORY`, and
`EUHEDRAL_INFERENCE_CUDA_LIBRARY_PATH`; set `PORT` at runtime for the server port and publish
that same container port with `-p` (for example, `-e PORT=18900 -p 18900:18900`). Docker
does not embed a static exposed-port declaration, since it would become misleading when
`PORT` changes. The health probe reads the same setting. The image carries
CUDA runtime/NVRTC userspace libraries, matching headers for NVRTC, the native library, and
installed kernel sources; it does not contain the NVIDIA kernel driver. With `--gpus all`, NVIDIA
Container Toolkit and a compatible host driver are required. `/health` is checked by the image health
probe after the engine has loaded. Set worker CPUs to processor IDs available in your container.

On hosts without NVIDIA Container Toolkit, `--gpus all` cannot initialize the container. On a
Linux host exposing `/dev/nvidia*` and the driver libraries at the paths below, replace
`--gpus all` with these explicit device and read-only driver mounts (adjust host paths for your
distribution):

```text
--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
--device /dev/nvidia-modeset \
--mount type=bind,src=/usr/lib/x86_64-linux-gnu/libcuda.so.1,dst=/opt/euhedral/lib/libcuda.so.1,readonly \
--mount type=bind,src=/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1,dst=/opt/euhedral/lib/libnvidia-ptxjitcompiler.so.1,readonly
```

This is a host-specific alternative to NVIDIA Container Toolkit, not an image-bundled driver.

On the configured deployment host, run `mise run deploy` from a clean, checked-out `main`.
The task pulls `origin/main` with fast-forward only, builds an image from exactly that commit,
switches `euhedral-inference-serve` on localhost port 18080, checks Docker health, `/health`,
`/v1/models`, the expected public model ID, and a one-token chat completion. If the candidate
fails before verification, it restores and verifies the previous container. **After** verification,
it removes old Euhedral deployment and experimental probe containers and all previous
`euhedral-inference:*` image tags; rollback images are not retained. Other projects' containers
and images are not touched. Defaults for the artifact, tokenizer, driver libraries, port, and
public model ID are defined in `scripts/deploy-main.py`; override them with the
`EUHEDRAL_DEPLOY_MODEL`, `EUHEDRAL_DEPLOY_TOKENIZER`, `EUHEDRAL_DEPLOY_DRIVER`,
`EUHEDRAL_DEPLOY_PTX`, `EUHEDRAL_DEPLOY_PORT`, and `EUHEDRAL_DEPLOY_MODEL_ID` environment
variables when moving hosts. This script expects the explicit NVIDIA device and driver mounts
shown above, not NVIDIA Container Toolkit.

Common `mise run` tasks: `format-check`, `format`, `test`, `native-build`, `native-test`,
`native-verify`, `cuda-test`, `benchmark-validate`, `benchmark-smoke`,
`benchmark-baseline`, `full-build`, `test-deploy`, and `deploy`. `full-build` performs
formatting checks, CPU tests, both native product builds/verification/packaging, and the API
boot JAR. Reserve the GPU and temporarily stop the serving container before running
`native-test`, `cuda-test`, `benchmark-smoke`, or `benchmark-baseline`; those tasks do not
stop or restart a production service on their own.

## Testing

```bash
./gradlew test
```

## Benchmarking

The `benchmark` module measures the engine end to end: tokenizer, Euhedral lattice, and CUDA, with a
fresh session per iteration. It writes versioned JSON results. See [docs/BENCHMARKING.md](docs/BENCHMARKING.md)
for the build → test → baseline → profile → change → rerun workflow, the configuration file, and
metric definitions.

```bash
./gradlew :benchmark:run --args="run benchmark/configs/baseline.json --validate-only"
./gradlew :benchmark:run --args="run benchmark/configs/baseline.json"
```

## Formatting

Formatting is enforced via [Spotless](https://github.com/diffplug/spotless) with Palantir Java Format.

```bash
# Apply formatting to all sources
./gradlew spotlessApply

# Check formatting (used in CI)
./gradlew spotlessCheck
```

---

Licensed under the Apache License, Version 2.0 (see `LICENSE`).