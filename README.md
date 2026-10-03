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
| `q3` | Q3 | uncompressed | 11.72 GiB (12.58 GB) | |
| `q3-compressed` | Q3 | compressed | 10.16 GiB (10.91 GB) | lossless: the same values as `q3`, bit for bit |
| `nvfp4` | NVFP4 | uncompressed | 14.52 GiB (15.59 GB) | |
| `nvfp4-compressed` | NVFP4 | compressed | 13.77 GiB (14.79 GB) | 4.25 bits per weight (block scales drawn from a 16-entry table per tensor); not lossless, see [quality](#quality) |

The engine reads what an artifact is from the file and picks the fastest validated execution for it: kernels by row count,
FP8 and FP4 tensor-core prefill, the attention implementation, the speculative-decoding depth and which weights stay in
host memory. All of that is automatic.

## Performance

RTX 5070 Ti (16 GB, 70 SMs) in a desktop that also uses the GPU for its display, Linux, one run per row after one warmup. Greedy
decoding with MTP speculative decoding (two drafted tokens for Q3, three for NVFP4); decode `tok/s` is output tokens per second
after the first token. Prefill and time to first token (TTFT) include the whole prompt.

| Artifact | Context (prompt tokens) | Prefill tok/s | TTFT | Decode tok/s (MTP) |
|---|---|---|---|---|
| `q3` | 4K (3,964) | 1,974 | 2.10 s | 116.9 |
| | 16K (15,930) | 1,801 | 9.14 s | 118.7 |
| | 32K (31,906) | 1,636 | 19.9 s | 110.1 |
| | 64K (59,111) | 1,401 | 43.7 s | 98.0 |
| `q3-compressed` | 4K | 1,837 | 2.28 s | 95.0 |
| | 16K | 1,693 | 9.63 s | 97.0 |
| | 32K | 1,543 | 21.1 s | 91.1 |
| | 64K | 1,332 | 45.7 s | 82.0 |
| | 128K (128,000) | 987 | 135 s | 60.7 |
| `nvfp4` | 4K | 3,874 | 1.13 s | 70.7 |
| | 16K | 3,254 | 5.19 s | 61.2 |
| | 32K | 2,762 | 12.0 s | 63.8 |
| | 64K | 2,145 | 29.0 s | 57.7 |
| `nvfp4-compressed` | 4K | 3,815 | 1.14 s | 97.6 |
| | 16K | 3,219 | 5.23 s | 92.6 |
| | 32K | 2,728 | 12.1 s | 82.6 |
| | 64K | 2,131 | 29.2 s | 64.0 |

The 4K, 16K and 32K rows run with the default 32,768-token context; the 64K and 128K rows set `max-context-tokens` to 65,536
and 131,072. Each prompt is followed by 128 generated tokens (64 at 128K). The 128K prompt is generated text; the others are
chat prompts.

| Artifact | File | Device memory in use at 32K / 64K | Host-backed weights at 32K / 64K | Longest context run |
|---|---|---|---|---|
| `q3` | 11.72 GiB | 12.8 / 13.3 GiB | none | 64K |
| `q3-compressed` | 10.16 GiB | 11.3 / 11.8 GiB (14.0 GiB at 128K) | none | 128K |
| `nvfp4` | 14.52 GiB | 13.9 / 13.9 GiB | 1.22 / 1.77 GiB | 64K |
| `nvfp4-compressed` | 13.77 GiB | 13.9 / 13.9 GiB | 0.46 / 1.03 GiB | 64K |

Device memory is what the engine allocated at its peak during the run: the resident weights, the KV cache of the context, the
sequence state, and the workspaces. The CUDA context and kernel modules come on top. An uncompressed NVFP4 model is larger than
the card's memory budget at these contexts, so the engine holds part of it in pinned host memory and streams it in for every token.

Residency is described in [docs/NVFP4_RESIDENCY.md](docs/NVFP4_RESIDENCY.md) and [docs/COMPRESSED_Q3.md](docs/COMPRESSED_Q3.md).

## Quality

Error is measured four separate ways because they answer different questions. All use the 1279 teacher-forced tokens
that follow a 1281-token prefix of one fixed document, and the BF16 checkpoint run in llama.cpp on the CPU as the reference.

**Model and quantization error** — the artifact against the BF16 checkpoint (reference: BF16 in llama.cpp on the CPU):

| Artifact | Mean NLL | Perplexity | NLL vs BF16 (± s.e.) | KL(BF16 ‖ artifact) | Top-1 agreement |
|---|---|---|---|---|---|
| BF16 reference | 2.128 | 8.39 | | | |
| `q3`, `q3-compressed` | 2.403 | 11.05 | +0.275 ± 0.025 | 0.400 | 70.5% |
| `nvfp4` | 2.178 | 8.83 | +0.051 ± 0.011 | 0.163 | 86.4% |
| `nvfp4-compressed` | 2.183 | 8.87 | +0.056 ± 0.011 | 0.161 | 86.9% |

**Compression-induced error** — the compressed artifact against its uncompressed one:

| Pair | Result |
|---|---|
| `q3-compressed` against `q3` | lossless: the 1279 logit vectors are bitwise equal, as is every drift measurement below |
| `nvfp4-compressed` against `nvfp4` | NLL +0.005 ± 0.011, KL 0.062, top-1 agreement 87.0% |

**Relaxed-execution drift** — the production kernels against the exact scalar kernels on the same weights, 1024 teacher-forced
decode steps after a 512-token prefill:

| Artifact | Hidden-state relative error | Mean KL | Top-1 agreement |
|---|---|---|---|
| `q3`, `q3-compressed` | 7.6% | 4.7e-3 | 97.9% |
| `nvfp4` | 7.0% | 3.9e-3 | 97.3% |
| `nvfp4-compressed` | 7.4% | 3.4e-3 | 97.8% |

The error settles within about 150 positions and does not grow after that.

**Exact speculative verifier** — verification of drafts is row-exact: the logits and the KV and GDN state after a
verification are bit for bit those of one-row decoding, and speculative generation produces exactly the tokens of ordinary
greedy decoding. This holds for all four artifacts (`SpeculativeVerifyCudaIntegrationTest`,
`SpeculativeDecodeCudaIntegrationTest`).

How to reproduce each measurement is in [docs/QUALITY.md](docs/QUALITY.md).

## Choosing an artifact

- **Fastest, and the one that fits everywhere:** `q3`: 117 tok/s at 4K and 98 tok/s at 64K, with no host-backed weights. Its
  error against BF16 is the largest of the four (perplexity 11.05 against 8.39 on the quality text).
- **More room instead of speed:** `q3-compressed` returns exactly the `q3` outputs from a file 1.56 GiB smaller, which is what lets it
  hold a 128K context on this card. Its kernels decode the compressed weights as they read them, which costs about a fifth of `q3`'s
  decode speed (95 tok/s at 4K).
- **Highest fidelity:** the NVFP4 artifacts (perplexity 8.83 and 8.87). On a 16 GB card `nvfp4` keeps 1.2 to 1.8 GiB of weights in host memory at 32K to 64K of
  context and streams them in for every token; `nvfp4-compressed` needs less than half of that, which makes it faster than `nvfp4` up to 32K
  of context at a quality cost within the noise of the measurement (table above).
- A card with more memory keeps more weights on the device and runs faster; the engine measures free memory at start and decides.

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
restores the previous container if the candidate fails. It serves the `nvfp4-compressed` artifact by default; the artifact, model ID and
port are overridable through `EUHEDRAL_DEPLOY_*` variables (`scripts/deploy-main.py`).

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
