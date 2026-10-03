# Euhedral-Inference

An OpenAI-compatible inference server for **Qwen3.8-27B** (text) on a single NVIDIA Blackwell GPU, written in Java 25
with a small native CUDA layer. It runs the whole model on one consumer card (developed on an RTX 5070 Ti with 16 GB),
decodes with MTP speculative decoding, and serves 32K to 64K tokens of context from that 16 GB.

You choose a weight file; the engine chooses everything else. There are no kernel, precision, or scheduling options.

## The four artifacts

An artifact is one `.edrl` file holding the quantized text model, its MTP draft layer and draft head. You make two choices:

- **Quantization**: **Q3** (3-bit weights; smallest, fastest) or **NVFP4** (4-bit floating point; higher fidelity).
- **Representation**: **uncompressed** (reads fastest) or **compressed** (smaller on disk and in memory, a little more work per token).

| Artifact | Quantization | Representation | File size | Compression |
|---|---|---|---|---|
| `q3` | Q3 | uncompressed | @Q3_SIZE@ | |
| `q3-compressed` | Q3 | compressed | @Q3C_SIZE@ | lossless: the same values as `q3`, bit for bit |
| `nvfp4` | NVFP4 | uncompressed | @NV_SIZE@ | |
| `nvfp4-compressed` | NVFP4 | compressed | @NVC_SIZE@ | 4.25 bits per weight (block scales drawn from a 16-entry table per tensor); not lossless, see [quality](#quality) |

The engine reads what an artifact is from the file and picks the fastest validated execution for it: kernels by row count,
FP8 and FP4 tensor-core prefill, the attention implementation, the speculative-decoding depth and which weights stay in
host memory. All of that is automatic.

## Performance

RTX 5070 Ti (16 GB, 70 SMs), Ryzen/Intel desktop with the GPU also driving the display, Linux, driver 615.71. Greedy decoding
of chat prompts with MTP speculative decoding; `tok/s` is output tokens per second after the first token. One run per cell.

@PERFORMANCE_TABLE@

Memory is the artifact's resident device bytes (weights, KV cache and sequence state for that context), and the pinned host
memory the engine uses for weights that do not fit beside the KV cache. The longest context shown for each artifact is the
longest this card holds with desktop use; see [docs/NVFP4_RESIDENCY.md](docs/NVFP4_RESIDENCY.md).

## Quality

Error is measured four separate ways because they answer different questions. All use the 1279 teacher-forced tokens
that follow a 1281-token prefix of one fixed document, and the BF16 checkpoint run in llama.cpp on the CPU as the reference.

@QUALITY_TABLES@

How to reproduce each measurement is in [docs/QUALITY.md](docs/QUALITY.md).

## Choosing an artifact

- **Fastest decode, fits everywhere:** `q3`. Choose `q3-compressed` when you need the extra ~1.5 GB of VRAM; the outputs are identical.
- **Highest fidelity:** `nvfp4`. It needs roughly 2 GB of weights in host memory at 32K context on a 16 GB card, which costs speed.
- **NVFP4 on a small card:** `nvfp4-compressed` brings that host memory down; its quality cost is in the table above.
- Larger cards keep more weights on the device and run faster; the engine measures free memory at start and decides.

## Quick start

You need a Blackwell GPU (RTX 50-series or newer, compute capability 12.x) with a current NVIDIA driver, Java 25, and a
Hugging Face checkpoint of Qwen3.8-27B. The first build downloads the pinned CUDA runtime pieces and Zig.

**1. Convert the checkpoint to an artifact** (a minute or two for Q3 and NVFP4 on a GPU, longer for the compressed forms):

```bash
python3 tools/convert_checkpoint.py --model /path/to/Qwen3.8-27B --quantization q3 \
    --ranking /path/to/token-ranking.i64 --out /models/qwen-q3.edrl
# or: --quantization q3 --compressed | --quantization nvfp4 | --quantization nvfp4 --compressed
```

The token ranking chooses the draft head's vocabulary shortlist (`--ranking`), or reuse one from an existing artifact
(`--draft-ids-from /models/qwen-q3.edrl`). See [tools/README.md](tools/README.md).

**2. Build the server:**

```bash
./gradlew :api:bootJar
```

**3. Run it:**

```bash
CUDA=build/cuda-dev/linux-x64
EUHEDRAL_CUDA_INCLUDE_DIR=$PWD/$CUDA/include LD_LIBRARY_PATH=$PWD/$CUDA/runtime \
EUHEDRAL_INFERENCE_ARTIFACT_PATH=/models/qwen-q3.edrl \
EUHEDRAL_INFERENCE_TOKENIZER_DIRECTORY=/path/to/Qwen3.8-27B \
EUHEDRAL_INFERENCE_CUDA_LIBRARY_PATH=$PWD/build/native/linux-x64/lib/libeuhedral_cuda.so \
EUHEDRAL_INFERENCE_WORKER_CPUS=2-5 \
EUHEDRAL_INFERENCE_MODEL_ID=qwen \
java --enable-native-access=ALL-UNNAMED -jar api/build/libs/euhedral-inference-api.jar
```

The server listens on port 1738.

**4. Send a request:**

```bash
curl http://localhost:1738/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen",
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
  "max_tokens": 128
}'
```

Any OpenAI client works against `http://localhost:1738/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1738/v1", api_key="unused")
reply = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Explain speculative decoding in two sentences."}],
    stream=True,
)
for chunk in reply:
    print(chunk.choices[0].delta.content or "", end="")
```

The API serves `/v1/models` and `/v1/chat/completions` (JSON or SSE streaming) with OpenAI-style function calling
([docs/TOOL_CALLING.md](docs/TOOL_CALLING.md)). `/health` reports whether the engine has loaded.

## Docker

```bash
docker build -t euhedral-inference:local .
docker run --rm --gpus all -p 1738:1738 \
  --mount type=bind,src=/absolute/path/model.edrl,dst=/models/model.edrl,readonly \
  --mount type=bind,src=/absolute/path/tokenizer,dst=/tokenizer,readonly \
  -e EUHEDRAL_INFERENCE_WORKER_CPUS=2-5 \
  -e EUHEDRAL_INFERENCE_MODEL_ID=qwen \
  euhedral-inference:local
```

The model and tokenizer are not in the image. It carries the CUDA runtime and NVRTC userspace libraries, the native
library and the kernel sources, but not the NVIDIA kernel driver: `--gpus all` needs the NVIDIA Container Toolkit and a
compatible host driver. Hosts without the toolkit can pass the device nodes and the driver libraries explicitly:

```text
--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-modeset \
--mount type=bind,src=/usr/lib/x86_64-linux-gnu/libcuda.so.1,dst=/opt/euhedral/lib/libcuda.so.1,readonly \
--mount type=bind,src=/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1,dst=/opt/euhedral/lib/libnvidia-ptxjitcompiler.so.1,readonly
```

`mise run deploy` pulls `origin/main`, builds an image from exactly that commit, switches the `euhedral-inference-serve`
container on localhost port 18080 after checking Docker health, `/health`, `/v1/models` and a one-token completion, and
restores the previous container if the candidate fails. Its defaults are in `scripts/deploy-main.py`.

## Configuration

Everything the server reads is below. Each setting is a property (`euhedral.inference.*`, `euhedral.api.*`) or its
environment variable (`EUHEDRAL_INFERENCE_*`, `EUHEDRAL_API_*`); `PORT` sets the port.

| Setting | Default | Meaning |
|---|---|---|
| `artifact-path` | required | The `.edrl` artifact to serve. |
| `tokenizer-directory` | required | The checkpoint directory (`tokenizer.json`, `chat_template.jinja`, `generation_config.json`). |
| `cuda-library-path` | required | `libeuhedral_cuda.so` (`euhedral_cuda.dll` on Windows). |
| `worker-cpus` | required | Processor IDs or ranges for the engine's worker threads, for example `2-5,8`. |
| `model-id` | required | The name clients send as `model`. |
| `max-context-tokens` | 32768 | The longest prompt plus completion a request may use. The engine keeps device memory for that much KV cache and holds weights in pinned host memory when both do not fit; asking for more than the card can hold fails at start with a clear message. |
| `shutdown-timeout` | 10s | How long shutdown waits for the engine. |
| `euhedral.api.default-max-tokens` | 4096 | Completion length when a request sets none. |
| `euhedral.api.max-queued-generations` | 16 | Requests waiting behind the running generation; more are refused with 503. |
| `euhedral.api.max-request-bytes` | 1048576 | Request body limit. |
| `euhedral.api.request-timeout` | 30m | Per-request limit. |

One generation runs at a time; the engine sizes device memory for one sequence.

## Building and testing

Requirements: Java 25, Gradle 9.6.1 (the wrapper is committed), Zig 0.16.0 for the native library; CUDA 13.1+ headers and
libraries are resolved automatically. `mise.toml` records the versions and the common tasks.

```bash
./gradlew build            # compiles, formats, runs the CPU tests, builds the Linux and Windows native products
./gradlew test             # CPU tests
python3 -m unittest discover -s native/tests -p 'test_*.py'   # kernel tests on the GPU (needs NumPy)
./gradlew :core:cudaIntegrationTest :api:cudaIntegrationTest  # model tests on the GPU
```

Stop any serving container before the GPU suites. The native layer is described in [native/README.md](native/README.md),
the benchmark harness in [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

## More

- [docs/FRAME_MODEL.md](docs/FRAME_MODEL.md): how a forward pass is scheduled across device lanes.
- [docs/MTP_SPECULATIVE.md](docs/MTP_SPECULATIVE.md), [docs/MTP_CONTRACT.md](docs/MTP_CONTRACT.md), [docs/MTP_VERIFIER.md](docs/MTP_VERIFIER.md): speculative decoding and its exactness contract.
- [docs/PREFILL_MX.md](docs/PREFILL_MX.md), [docs/NVFP4_NATIVE.md](docs/NVFP4_NATIVE.md), [docs/ATTENTION_DECODE.md](docs/ATTENTION_DECODE.md): the prefill, FP4 and attention kernels.
- [docs/COMPACT_Q3_REFERENCE.md](docs/COMPACT_Q3_REFERENCE.md), [docs/COMPRESSED_Q3.md](docs/COMPRESSED_Q3.md), [docs/NVFP4_COMPRESSED.md](docs/NVFP4_COMPRESSED.md), [docs/NVFP4_RESIDENCY.md](docs/NVFP4_RESIDENCY.md): the artifact formats and memory budgets.
- [docs/nvidia/](docs/nvidia/README.md): measurements of this GPU's hardware.

---

Licensed under the Apache License, Version 2.0 (see `LICENSE`).
