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
| `q3` | Q3 | uncompressed | 11.68 GiB (12.54 GB) | LM head in NVFP4 |
| `q3-compressed` | Q3 | compressed | 11.00 GiB (11.81 GB) | lossless layout of a Q3 limited to the codes -3..3, MLP down projections in row-split Q3 (see [quality](#quality)); LM head in NVFP4 |
| `nvfp4` | NVFP4 | uncompressed | 14.52 GiB (15.59 GB) | |
| `nvfp4-compressed` | NVFP4 | compressed | 13.77 GiB (14.79 GB) | 4.25 bits per weight (block scales drawn from a 16-entry table per tensor); not lossless, see [quality](#quality) |

One mixed artifact, `NVFP4-Trellis-RCO`, takes each projection of the text layers from QTIP-style trellis codes at 2.5 to
4.0 bits per weight (exllamav3's EXL3 format, calibrated by the quantization module) as a search allocated them for the least
error with a 128K context on the device: 114 of its 320 projections at 4.0 bits, 88 at 3.5, 45 at 3.0 and 73 at 2.5, with a Q3
token embedding and NVFP4 heads, 11.17 GiB (11.99 GB) ([docs/quantization/MIXED_QUANTIZATION.md](docs/quantization/MIXED_QUANTIZATION.md),
[docs/quantization/TRELLIS.md](docs/quantization/TRELLIS.md)).

All four artifacts are converted from Swift-1.5 (`ukisai/Swift-1.5-Qwen3.8-27b`, UkisAI's fine-tune of Qwen3.8-27B); the
converter takes either checkpoint.

The engine reads what an artifact is from the file and picks the fastest validated execution for it: kernels by row count,
FP8 and FP4 tensor-core prefill, the attention implementation, the speculative-decoding depth and which weights stay in
host memory. All of that is automatic.

Any artifact can also carry the DFlash2 drafter (`z-lab/Qwen3.8-27B-DFlash2`): the converter appends it with
`--extend ARTIFACT --dflash2 DIR`, and an artifact that holds it speculates with DFlash2 instead of MTP, with the same
exact verification ([docs/speculation/DFLASH2.md](docs/speculation/DFLASH2.md)). The drafter was trained on Qwen3.8-27B, which Swift-1.5
fine-tunes; added to `nvfp4-compressed` (15.04 GiB with the drafter), it decodes 158.6 tok/s at 4K, 128.4 at 32K, 91.4 at 64K
and 49.4 at 128K tokens of context (chat corpus and 256 generated tokens; at 128K generated text and 64 tokens; MTP on the same
target: 146.6, 129.9, 103.5 and 49.7); its larger drafter leaves less of the 16 GB for the target, and the artifacts above draft
with MTP.

## Performance

RTX 5070 Ti (16 GB, 70 SMs) in a desktop that also uses the GPU for its display, Linux, one run per row after one warmup. Greedy
decoding with MTP speculative decoding (two drafted tokens for Q3, four for NVFP4 and `NVFP4-Trellis-RCO`); decode `tok/s` is
output tokens per second after the first token. Prefill and time to first token (TTFT) include the whole prompt. Measured on
2026-10-16 (commit 1eea731; `NVFP4-Trellis-RCO` on 2026-10-19 at commit fea6b96f) with the GPU overclocked: +150 MHz core and +1500 MHz
memory clock offsets and a 350 W power limit (stock is 300 W). Decode depends on how many drafted tokens the verifier accepts on
each prompt, which moves single rows by several percent. Rows that stream weights from host memory vary more, by up to 20%
between runs: the `nvfp4` 4K and 32K and the `nvfp4-compressed` 128K decode rows are the mean of two runs.

Context is the prompt length: 4K = 3,964 tokens, 32K = 32,612, 64K = 63,362, 128K = 128,000.

**Prefill (tokens/s)**

| Artifact | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| `q3` | 2,099 | 1,726 | 1,447 | 1,085 |
| `q3-compressed` | 1,986 | 1,655 | 1,399 | 1,058 |
| `nvfp4` | 4,111 | 2,883 | 2,182 | 1,399 |
| `nvfp4-compressed` | 4,033 | 2,844 | 2,160 | 1,413 |
| `NVFP4-Trellis-RCO` | 2,586 | 2,039 | 1,663 | 1,203 |

**Decode (tokens/s, MTP)**

| Artifact | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| `q3` | 130.7 | 126.5 | 109.4 | 69.6 |
| `q3-compressed` | 114.4 | 103.1 | 93.3 | 65.0 |
| `nvfp4` | 105.3 | 102.7 | 64.1 | 32.4 |
| `nvfp4-compressed` | 135.0 | 124.7 | 89.6 | 49.7 |
| `NVFP4-Trellis-RCO` | 147.2 | 114.6 | 93.1 | 78.3 |

**Time to first token**

| Artifact | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| `q3` | 1.90 s | 19.0 s | 43.9 s | 119 s |
| `q3-compressed` | 2.01 s | 19.7 s | 45.5 s | 121 s |
| `nvfp4` | 0.97 s | 11.4 s | 29.1 s | 91.7 s |
| `nvfp4-compressed` | 0.99 s | 11.5 s | 29.4 s | 90.9 s |
| `NVFP4-Trellis-RCO` | 1.54 s | 16.1 s | 38.2 s | 107 s |

The 4K and 32K rows run with the default 32,768-token context; the 64K and 128K rows set `max-context-tokens` to 65,536
and 131,072. Each prompt is followed by 128 generated tokens (64 at 128K). The 128K prompt is generated text; the others are
chat prompts.

| Artifact | File | Device memory in use at 32K / 64K / 128K | Host-backed weights at 32K / 64K / 128K | Longest context run |
|---|---|---|---|---|
| `q3` | 11.68 GiB | 12.8 / 13.4 / 14.1 GiB | none | 128K |
| `q3-compressed` | 11.00 GiB | 12.2 / 12.8 / 14.0 GiB | none | 128K |
| `nvfp4` | 14.52 GiB | 14.2 / 14.2 / 14.2 GiB | 1.02 / 1.58 / 2.79 GiB | 128K |
| `nvfp4-compressed` | 13.77 GiB | 14.2 / 14.2 / 14.2 GiB | 0.27 / 0.84 / 2.02 GiB | 128K |
| `NVFP4-Trellis-RCO` | 11.17 GiB | 12.3 / 12.9 / 14.0 GiB | none | 128K |

Device memory is what the engine allocated at its peak during the run: the resident weights, the KV cache of the context, the
sequence state, and the workspaces. The CUDA context and kernel modules come on top. An uncompressed NVFP4 model is larger than
the card's memory budget at these contexts, so the engine holds part of it in pinned host memory and streams it in for every token.

Residency is described in [docs/quantization/NVFP4_RESIDENCY.md](docs/quantization/NVFP4_RESIDENCY.md) and [docs/quantization/COMPRESSED_Q3.md](docs/quantization/COMPRESSED_Q3.md).

## Quality

Error is measured four separate ways because they answer different questions. The model and quantization error is
measured on a held-out quality set ([docs/evaluation/RELEASE_TEST.md](docs/evaluation/RELEASE_TEST.md)): 112 chunks of 2,560 tokens, each one
independent conversation or document (BF16 Swift-1.5's own answers in conversation, coding, tool use and five other
languages, with their chat-template, thinking and tool-call tokens, and human-written prose and code published after the
model's training), the second half of every chunk scored (143,360 tokens), with BF16 Swift-1.5 in llama.cpp as the
reference.

All four artifacts' decoder layers and heads are rounded by GPTQ: each linear's rounding minimizes the error of
its outputs on calibration activations of the quantized model, each column's error carried into the columns not yet
rounded (the `quantization` module, [docs/evaluation/QUALITY.md](docs/evaluation/QUALITY.md#gptq)); `q3` also rounds its MTP layer by GPTQ.
The token embedding (and the other artifacts' MTP layer) are rounded by calibrated scale search: every group's scale minimizes its
error weighted by the importance of each input column (`tools/convert_checkpoint.py --imatrix`, see
[tools/README.md](tools/README.md)) with UkisAI's Swift-1.5 importance matrix (`imatrix-swift15-v1mix.gguf`, from
`ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF`), and is measured against the Swift-1.5 BF16 checkpoint. `q3-compressed` limits its
Q3 tensors other than the MLP down projections to the codes -3..3, which its lossless layout stores, while `q3` also uses -4 ([docs/quantization/COMPRESSED_Q3.md](docs/quantization/COMPRESSED_Q3.md)), so the two hold different values.

**Model and quantization error** — the artifact against the BF16 checkpoint on the held-out set:

| Artifact | Mean NLL | Perplexity | NLL vs BF16 (± s.e.) | KL(BF16 ‖ artifact) | Top-1 agreement |
|---|---|---|---|---|---|
| Swift-1.5 BF16 reference | 0.609 | 1.84 | | | |
| `q3` | 0.699 | 2.01 | +0.090 ± 0.007 | 0.106 | 91.0% |
| `q3-compressed` | 0.720 | 2.05 | +0.111 ± 0.008 | 0.131 | 89.8% |
| `nvfp4` | 0.654 | 1.92 | +0.045 ± 0.005 | 0.047 | 94.3% |
| `nvfp4-compressed` | 0.648 | 1.91 | +0.040 ± 0.005 | 0.046 | 94.3% |
| `NVFP4-Trellis-RCO` | 0.660 | 1.93 | +0.051 ± 0.006 | 0.072 | 92.9% |

The perplexity is low because most of the set is the model's own text. In every domain the two NVFP4 forms have the least
error and `q3-compressed` the most; tool use has the largest error for every artifact (KL 0.133 for `nvfp4` to 0.282 for `q3-compressed`) and prose the smallest
(0.02 to 0.07). Per domain: [docs/evaluation/RELEASE_TEST.md](docs/evaluation/RELEASE_TEST.md#quality-by-domain).

**Compression-induced error** — the compressed artifact against its uncompressed one:

| Pair | Result |
|---|---|
| `q3-compressed` against the row-split Q3 with the same values (`q3_p2e2 expand`) | lossless: every route's logits are bitwise equal (`P2e2CudaIntegrationTest`) |
| NVFP4-SD4 against `nvfp4`, both converted from Qwen3.8-27B with the same importance matrix (on one fixed document, 1279 scored tokens) | NLL +0.005 ± 0.009, KL 0.037, top-1 agreement 89.8% |

**Relaxed-execution drift** — the production kernels against the exact scalar kernels on the same weights, 1024 teacher-forced
decode steps after a 512-token prefill (measured on the round-to-nearest artifacts):

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

How to reproduce each measurement is in [docs/evaluation/QUALITY.md](docs/evaluation/QUALITY.md).

## Choosing an artifact

- **Fastest prefill, and decode at short contexts:** `nvfp4-compressed` (KL 0.046 against BF16 on the held-out
  set): 4,033 tok/s of prefill and 135 tok/s of decode at 4K. On a 16 GB card it keeps 0.27 to 0.84 GiB of weights in host memory at 32K to 64K
  (2.02 GiB at 128K) and streams them in for every token: 125 tok/s at 32K, 90 at 64K and 50 at 128K.
- **Least error with every weight on the device:** `NVFP4-Trellis-RCO` (KL 0.072): prefill 2,586 tok/s at 4K and 1,203 at
  128K, decode 147 tok/s at 4K, 115 at 32K, 93 at 64K and 78 at 128K.
- **Faster prefill with every weight on the device:** `q3` (KL 0.106): 131 tok/s at 4K, 127 at 32K, 109 at 64K and 70 at
  128K, prefill 2,099 tok/s at 4K and 1,085 at 128K.
- **More room instead of speed:** `q3-compressed` (KL 0.131) is a file about 0.7 GiB smaller than `q3`, which leaves more
  of the card to the KV cache of a long context. Its kernels decode the compressed weights as they read them: 114 tok/s at 4K,
  103 at 32K, 93 at 64K and 65 at 128K.
- `nvfp4` (KL 0.047) keeps 1.02 to 1.58 GiB of weights in host memory at 32K to 64K (2.79 GiB at 128K) and decodes slower
  than `nvfp4-compressed` at every context; the two NVFP4 forms differ by less than the noise of the quality measurement, and
  their prefill is within 2%.
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

This builds the jar for the machine's platform, `api/build/libs/euhedral-inference-linux-x64.jar`
(`:api:bootJarWindowsX64` builds `euhedral-inference-windows-x64.jar`). The jar carries its native files: the CUDA
library, the CUDA runtime and NVRTC, the kernel sources and the CUDA headers. It unpacks them into the user's cache
directory at startup, so the machine needs only the NVIDIA driver.

**3. Run it:**

```bash
cat > euhedral.yaml <<'YAML'
model:
  artifact: /models/qwen-q3.edrl
  name: qwen
YAML
java --enable-native-access=ALL-UNNAMED -jar api/build/libs/euhedral-inference-linux-x64.jar --config euhedral.yaml
```

The file needs only those two keys: the artifact carries its tokenizer and chat template.
[config/euhedral.example.yaml](config/euhedral.example.yaml) sets every key and says what it does and its default,
among them the `sampling` defaults a request that names none gets (temperature 1.0, top-k 20, top-p 0.95). Any key can also be given as a variable (`model.name` is `EUHEDRAL_MODEL_NAME`), and
`EUHEDRAL_CONFIG` names the file in place of `--config`. The server listens on port 1738.

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
- **Tool calling**: OpenAI and Anthropic tool definitions, single and parallel calls, tool results, in the checkpoint's
  native `<tool_call>` format; `required` and named `strict` tools get arguments constrained to their schema. The server returns calls; the client runs them (an MCP client sits in the
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
set the artifact path, the model name and the worker CPUs (`host.cpus`).

```bash
docker build -t euhedral-inference:local .
set -a; . ./.env; set +a
docker run --rm --gpus all --env-file .env -p "$PORT:$PORT" --stop-timeout 45 \
  --mount type=bind,src="$EUHEDRAL_ARTIFACT_FILE",dst=/models/model.edrl,readonly \
  euhedral-inference:local
```

The model (with the tokenizer its artifact carries) is not in the image. It carries the CUDA runtime and NVRTC userspace libraries, the native
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

The server is configured by one YAML file, with any key overridable by an environment variable: the three `model`
keys are required, and `context.max-tokens` (32768) and the prefix cache's size are the ones most deployments change.
[config/euhedral.example.yaml](config/euhedral.example.yaml) lists every key and
[docs/OPERATIONS.md](docs/OPERATIONS.md) covers startup checks, health, shutdown and metrics.

One generation runs at a time; the engine sizes device memory for one sequence. The prefix cache is the only state kept between requests; it is host memory, so it takes nothing from the context length or the device. Requests are served entirely on the
engine's worker threads: the container thread hands a request over and returns, and rendering, tokenization, every
quantum and every write to the client run as lattice work (docs/architecture/FRAME_MODEL.md, "Host work on the workers").

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
the benchmark harness in [docs/evaluation/BENCHMARKING.md](docs/evaluation/BENCHMARKING.md).

## More

The [documentation index](docs/README.md) groups user guides and implementation references by topic.

- [docs/API_COMPATIBILITY.md](docs/API_COMPATIBILITY.md), [docs/ANTHROPIC_API.md](docs/ANTHROPIC_API.md), [docs/RESPONSES_API.md](docs/RESPONSES_API.md): the request formats, field by field.
- [docs/REASONING.md](docs/REASONING.md), [docs/TOOL_CALLING.md](docs/TOOL_CALLING.md), [docs/STRUCTURED_OUTPUT.md](docs/STRUCTURED_OUTPUT.md): reasoning, tools and constrained output.
- [docs/OPERATIONS.md](docs/OPERATIONS.md): settings, startup, health, shutdown and metrics.

- [docs/architecture/FRAME_MODEL.md](docs/architecture/FRAME_MODEL.md): how a forward pass is scheduled across device lanes.
- [docs/PREFIX_CACHE.md](docs/PREFIX_CACHE.md): the state kept between requests, its checkpoints and measured restore times.
- [docs/speculation/MTP_SPECULATIVE.md](docs/speculation/MTP_SPECULATIVE.md), [docs/speculation/MTP_CONTRACT.md](docs/speculation/MTP_CONTRACT.md), [docs/speculation/MTP_VERIFIER.md](docs/speculation/MTP_VERIFIER.md): speculative decoding and its exactness contract.
- [docs/speculation/DFLASH2.md](docs/speculation/DFLASH2.md): the DFlash2 drafter, its artifact, checkpoints, validation and measurements.
- [docs/speculation/MTP_DRAFT_CONFIDENCE.md](docs/speculation/MTP_DRAFT_CONFIDENCE.md): MTP draft confidence per output position, and draft-length policies screened on it.
- [docs/speculation/MTP_DRAFTER.md](docs/speculation/MTP_DRAFTER.md): the MTP drafter's corpus, training and export, and the drafter NVFP4-Trellis-RCO ships.
- [docs/kernels/PREFILL_MX.md](docs/kernels/PREFILL_MX.md), [docs/kernels/NVFP4_NATIVE.md](docs/kernels/NVFP4_NATIVE.md), [docs/kernels/ATTENTION_DECODE.md](docs/kernels/ATTENTION_DECODE.md): the prefill, FP4 and attention kernels.
- [docs/quantization/COMPACT_Q3_REFERENCE.md](docs/quantization/COMPACT_Q3_REFERENCE.md), [docs/quantization/COMPRESSED_Q3.md](docs/quantization/COMPRESSED_Q3.md), [docs/quantization/NVFP4_COMPRESSED.md](docs/quantization/NVFP4_COMPRESSED.md), [docs/quantization/NVFP4_RESIDENCY.md](docs/quantization/NVFP4_RESIDENCY.md): the artifact formats and memory budgets.
- [docs/flash-next/FLASH_NEXT_ARTIFACT.md](docs/flash-next/FLASH_NEXT_ARTIFACT.md), [docs/flash-next/FLASH_NEXT_RESIDENCY.md](docs/flash-next/FLASH_NEXT_RESIDENCY.md): the Qwen3.8-Flash-Next (`qwen4_exp`) artifact, and how its storage is placed and its experts cached.
- [docs/flash-next/FLASH_NEXT_EXECUTION.md](docs/flash-next/FLASH_NEXT_EXECUTION.md), [docs/flash-next/FLASH_NEXT_QSA.md](docs/flash-next/FLASH_NEXT_QSA.md), [docs/flash-next/FLASH_NEXT_EXPERTS.md](docs/flash-next/FLASH_NEXT_EXPERTS.md), [docs/flash-next/FLASH_NEXT_REFERENCE.md](docs/flash-next/FLASH_NEXT_REFERENCE.md): running Flash-Next as a text model (ordinary generation, no speculation, no vision): the four-stream residual, GDN, sparse attention, the expert waves, the sequence state, its validation against the upstream implementation and its first measurements.
- [docs/nvidia/](docs/nvidia/README.md): measurements of this GPU's hardware.

---

Licensed under the Apache License, Version 2.0 (see `LICENSE`).
