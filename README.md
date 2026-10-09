# Euhedral-Inference

An inference server for **Qwen3.8-27B** (text) on a single NVIDIA Blackwell GPU, written in Java 25 with a small native
CUDA layer. It runs the whole model on one consumer card (developed on an RTX 5070 Ti with 16 GB), decodes with MTP
speculative decoding, serves 32K to 64K tokens of context from that 16 GB, and speaks the OpenAI Chat Completions, OpenAI
Responses and Anthropic Messages APIs, with reasoning, tool calling, schema-constrained output and a prefix cache.

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

Any artifact can also carry the DFlash2 drafter (`z-lab/Qwen3.8-27B-DFlash2`): the converter appends it with
`--extend ARTIFACT --dflash2 DIR`, and an artifact that holds it speculates with DFlash2 instead of MTP, with the same
exact verification ([docs/DFLASH2.md](docs/DFLASH2.md)). On the compressed NVFP4 artifact it decodes 122.0 tok/s at 4K, 95.4 at 32K,
77.1 at 64K and 38.5 at 128K tokens of context (chat corpus, 256 generated tokens, 64 at 128K; MTP on the same target: 130.8, 101.0,
91.3 and 52.7); its larger drafter leaves less of the 16 GB for the target, and the four artifacts above draft with MTP.

## Performance

RTX 5070 Ti (16 GB, 70 SMs) in a desktop that also uses the GPU for its display, Linux, one run per row after one warmup. Greedy
decoding with MTP speculative decoding (two drafted tokens for Q3, four for NVFP4); decode `tok/s` is output tokens per second
after the first token. Prefill and time to first token (TTFT) include the whole prompt.

Context is the prompt length: 4K = 3,964 tokens, 32K = 32,612, 64K = 63,362, 128K = 128,000.

**Prefill (tokens/s)**

| Artifact | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| `q3` | 1,867 | 1,543 | 1,294 | 967 |
| `q3-compressed` | 1,860 | 1,547 | 1,310 | 995 |
| `nvfp4` | 3,929 | 2,754 | 2,088 | 1,347 |
| `nvfp4-compressed` | 3,867 | 2,727 | 2,064 | 1,356 |

**Time to first token**

| Artifact | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| `q3` | 2.13 s | 21.2 s | 49.1 s | 133 s |
| `q3-compressed` | 2.14 s | 21.1 s | 48.5 s | 129 s |
| `nvfp4` | 1.02 s | 11.9 s | 30.5 s | 95.3 s |
| `nvfp4-compressed` | 1.05 s | 12.0 s | 30.8 s | 94.7 s |

**Decode (tokens/s, MTP)**

| Artifact | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| `q3` | 109.4 | 104.5 | 92.5 | 67.2 |
| `q3-compressed` | 94.6 | 91.7 | 82.2 | 63.6 |
| `nvfp4` | 116.2 | 101.1 | 72.1 | 29.8 |
| `nvfp4-compressed` | 146.4 | 102.2 | 87.7 | 52.2 |

The 4K and 32K rows run with the default 32,768-token context; the 64K and 128K rows set `max-context-tokens` to 65,536
and 131,072. Each prompt is followed by 128 generated tokens (64 at 128K). The 128K prompt is generated text; the others are
chat prompts.

| Artifact | File | Device memory in use at 32K / 64K / 128K | Host-backed weights at 32K / 64K / 128K | Longest context run |
|---|---|---|---|---|
| `q3` | 11.72 GiB | 12.8 / 13.4 / 14.1 GiB | none | 128K |
| `q3-compressed` | 10.16 GiB | 11.4 / 11.9 / 13.1 GiB | none | 128K |
| `nvfp4` | 14.52 GiB | 14.2 / 14.2 / 14.2 GiB | 1.02 / 1.58 / 2.79 GiB | 128K |
| `nvfp4-compressed` | 13.77 GiB | 14.2 / 14.2 / 14.3 GiB | 0.27 / 0.84 / 1.97 GiB | 128K |

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

- **Fastest at short contexts, and the highest fidelity:** `nvfp4-compressed` (perplexity 8.87): 146 tok/s at 4K. On a 16 GB
  card it keeps 0.27 to 0.84 GiB of weights in host memory at 32K to 64K (1.97 GiB at 128K) and streams them in for every token,
  which slows it to 102 tok/s at 32K, 88 at 64K and 52 at 128K.
- **Fastest at long contexts, and the one that fits everywhere:** `q3`: 109 tok/s at 4K, 105 at 32K, 93 at 64K and 67 at 128K,
  with no host-backed weights. Its error against BF16 is the largest of the four (perplexity 11.05 against 8.39 on the quality text).
- **More room instead of speed:** `q3-compressed` returns exactly the `q3` outputs from a file 1.56 GiB smaller, which leaves more
  of the card to the KV cache of a long context (64 tok/s at 128K, against `q3`'s 67). Its kernels decode the compressed weights
  as they read them, which costs about a seventh of `q3`'s decode speed (95 tok/s at 4K).
- `nvfp4` (perplexity 8.83) keeps 1.0 to 1.6 GiB of weights in host memory at 32K to 64K (2.8 GiB at 128K) and decodes slower than
  `nvfp4-compressed` at every context in the table, at a quality difference within the noise of the measurement; its prefill is within 1%.
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

Anthropic's client works against the same server:

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:1738", api_key="unused")
message = client.messages.create(
    model="qwen", max_tokens=512, messages=[{"role": "user", "content": "Name three prime numbers."}]
)
print(message.content[-1].text)
```

## The API

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions, JSON or server-sent events |
| `POST /v1/responses` | OpenAI Responses, stateless ([docs/RESPONSES_API.md](docs/RESPONSES_API.md)) |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` | Anthropic Messages ([docs/ANTHROPIC_API.md](docs/ANTHROPIC_API.md)) |
| `GET /v1/models`, `GET /v1/models/{id}` | The served model, in OpenAI's or (with `anthropic-version`) Anthropic's format |
| `GET /health` | Readiness ([docs/OPERATIONS.md](docs/OPERATIONS.md)) |
| `GET /metrics` | Prometheus metrics ([docs/OPERATIONS.md](docs/OPERATIONS.md#metrics)) |

All three request formats run on one pipeline, so they behave alike:

- **Reasoning**: the model thinks before it answers (Chat Completions and Responses by default, Messages when asked,
  as each API defines it); the reasoning comes back separately (`reasoning_content`, a `reasoning` item, a `thinking`
  block), and its effort is selectable ([docs/REASONING.md](docs/REASONING.md)).
- **Tool calling**: OpenAI and Anthropic tool definitions, single and parallel calls, tool results; `strict` tools get
  arguments constrained to their schema. The server returns calls; the client runs them (an MCP client sits in the
  client, not in the server) ([docs/TOOL_CALLING.md](docs/TOOL_CALLING.md)).
- **Structured output**: `json_object` and `json_schema`, enforced token by token with llguidance; a schema that cannot
  be enforced exactly is refused instead of approximated ([docs/STRUCTURED_OUTPUT.md](docs/STRUCTURED_OUTPUT.md)).
- **Prefix cache**: a request that repeats an earlier prompt's start prefills only what follows it, whichever API sent
  either; responses report the cached tokens ([docs/PREFIX_CACHE.md](docs/PREFIX_CACHE.md)).
- **Cancellation**: a client that disconnects stops its generation before the next quantum, whether it was waiting,
  prefilling or decoding.

A field that would change the output and is not implemented is refused with 400, never ignored.
[docs/API_COMPATIBILITY.md](docs/API_COMPATIBILITY.md) classifies every field of every request.

## Docker

The container takes its settings from a `.env` file: copy [.env.example](.env.example) to `.env` (git ignores it) and
set the artifact and checkpoint paths, the model ID and the worker CPUs.

```bash
docker build -t euhedral-inference:local .
set -a; . ./.env; set +a
docker run --rm --gpus all --env-file .env -p "$PORT:$PORT" --stop-timeout 45 \
  --mount type=bind,src="$EUHEDRAL_ARTIFACT_FILE",dst=/models/model.edrl,readonly \
  --mount type=bind,src="$EUHEDRAL_CHECKPOINT_DIR",dst=/tokenizer,readonly \
  euhedral-inference:local
```

The model and tokenizer are not in the image. It carries the CUDA runtime and NVRTC userspace libraries, the native
library and the kernel sources, but not the NVIDIA kernel driver: `--gpus all` needs the NVIDIA Container Toolkit and a
compatible host driver. Hosts without the toolkit can pass the device nodes and the driver libraries explicitly:

```text
--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-modeset \
--mount type=bind,src=/usr/lib/x86_64-linux-gnu/libcuda.so.1,dst=/opt/euhedral/lib/libcuda.so.1,readonly \
--mount type=bind,src=/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1,dst=/opt/euhedral/lib/libnvidia-ptxjitcompiler.so.1,readonly
```

`mise run deploy` reads the same `.env`. It pulls `origin/main`, builds an image from exactly that commit, switches the
`euhedral-inference-serve` container (published on `127.0.0.1:$PORT`) after checking Docker health, `/health`,
`/v1/models` and a one-token completion, and restores the previous container if the candidate fails. With
`EUHEDRAL_CUDA_DRIVER_FILE` and `EUHEDRAL_PTX_JIT_FILE` set it passes the device nodes and those libraries instead of
`--gpus all`.

## Configuration

The server is configured by environment variables or properties: the five above are required, and
`EUHEDRAL_INFERENCE_MAX_CONTEXT_TOKENS` (32768) and the prefix cache's size are the ones most deployments change.
[docs/OPERATIONS.md](docs/OPERATIONS.md) lists every setting and covers startup checks, health, shutdown and metrics.

One generation runs at a time; the engine sizes device memory for one sequence. The prefix cache is the only state kept between requests; it is host memory, so it takes nothing from the context length or the device. Requests are served entirely on the
engine's worker threads: the container thread hands a request over and returns, and rendering, tokenization, every
quantum and every write to the client run as lattice work (docs/FRAME_MODEL.md, "Host work on the workers").

## Building and testing

Requirements: Java 25, Gradle 9.6.1 (the wrapper is committed), Zig 0.16.0 for the native library, and Rust 1.95.0 with
cargo-zigbuild 0.23.4 for llguidance (constrained decoding, built from its pinned crates.io release); CUDA 13.1+ headers
and libraries are resolved automatically. `mise.toml` records the versions and the common tasks (`mise install`).

```bash
./gradlew build            # compiles, formats, runs the CPU tests, builds the Linux and Windows native products
./gradlew test             # CPU tests
python3 -m unittest discover -s native/tests -p 'test_*.py'   # kernel tests on the GPU (needs NumPy)
./gradlew :core:cudaTest :api:cudaTest  # every CUDA test on the GPU; --tests needs :core:cudaIntegrationTest
```

Stop any serving container before the GPU suites. The native layer is described in [native/README.md](native/README.md),
the benchmark harness in [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

## More

- [docs/API_COMPATIBILITY.md](docs/API_COMPATIBILITY.md), [docs/ANTHROPIC_API.md](docs/ANTHROPIC_API.md), [docs/RESPONSES_API.md](docs/RESPONSES_API.md): the request formats, field by field.
- [docs/REASONING.md](docs/REASONING.md), [docs/TOOL_CALLING.md](docs/TOOL_CALLING.md), [docs/STRUCTURED_OUTPUT.md](docs/STRUCTURED_OUTPUT.md): reasoning, tools and constrained output.
- [docs/OPERATIONS.md](docs/OPERATIONS.md): settings, startup, health, shutdown and metrics.

- [docs/FRAME_MODEL.md](docs/FRAME_MODEL.md): how a forward pass is scheduled across device lanes.
- [docs/PREFIX_CACHE.md](docs/PREFIX_CACHE.md): the state kept between requests, its checkpoints and measured restore times.
- [docs/MTP_SPECULATIVE.md](docs/MTP_SPECULATIVE.md), [docs/MTP_CONTRACT.md](docs/MTP_CONTRACT.md), [docs/MTP_VERIFIER.md](docs/MTP_VERIFIER.md): speculative decoding and its exactness contract.
- [docs/DFLASH2.md](docs/DFLASH2.md): the DFlash2 drafter, its artifact, checkpoints, validation and measurements.
- [docs/PREFILL_MX.md](docs/PREFILL_MX.md), [docs/NVFP4_NATIVE.md](docs/NVFP4_NATIVE.md), [docs/ATTENTION_DECODE.md](docs/ATTENTION_DECODE.md): the prefill, FP4 and attention kernels.
- [docs/COMPACT_Q3_REFERENCE.md](docs/COMPACT_Q3_REFERENCE.md), [docs/COMPRESSED_Q3.md](docs/COMPRESSED_Q3.md), [docs/NVFP4_COMPRESSED.md](docs/NVFP4_COMPRESSED.md), [docs/NVFP4_RESIDENCY.md](docs/NVFP4_RESIDENCY.md): the artifact formats and memory budgets.
- [docs/FLASH_NEXT_ARTIFACT.md](docs/FLASH_NEXT_ARTIFACT.md), [docs/FLASH_NEXT_RESIDENCY.md](docs/FLASH_NEXT_RESIDENCY.md): the Qwen3.8-Flash-Next (`qwen4_exp`) artifact, and how its storage is placed and its experts cached.
- [docs/FLASH_NEXT_EXECUTION.md](docs/FLASH_NEXT_EXECUTION.md), [docs/FLASH_NEXT_QSA.md](docs/FLASH_NEXT_QSA.md), [docs/FLASH_NEXT_EXPERTS.md](docs/FLASH_NEXT_EXPERTS.md), [docs/FLASH_NEXT_REFERENCE.md](docs/FLASH_NEXT_REFERENCE.md): running Flash-Next as a text model (ordinary generation, no speculation, no vision): the four-stream residual, GDN, sparse attention, the expert waves, the sequence state, its validation against the upstream implementation and its first measurements.
- [docs/nvidia/](docs/nvidia/README.md): measurements of this GPU's hardware.

---

Licensed under the Apache License, Version 2.0 (see `LICENSE`).
