# Euhedral-Inference

Run **Qwen3.8-27B** and **Swift-1.5** locally on a single NVIDIA Blackwell GPU. Euhedral-Inference
supports OpenAI and Anthropic clients, streaming, reasoning, tool calling, and JSON output.

On a 16 GB RTX 5070 Ti, the main benchmark reaches **147 tokens/s at 4K context**.
All five formats were tested with contexts up to **128K tokens**. See the [benchmarks](#performance) for results by format and context length.

[Quick start](#quick-start) · [Choose a model](#choosing-an-artifact) · [Performance](#performance) · [Quality](#quality) · [Documentation](docs/README.md)

## Quick start

You need a Blackwell GPU with a current NVIDIA driver, Java 25, and a model in Euhedral's `.edrl` format.
An artifact is a model file ready to serve. To make one from a Hugging Face checkpoint, follow the
[conversion guide](docs/GETTING_STARTED.md#convert-a-checkpoint).

**1. Build the server.** [Build requirements](docs/development/BUILDING.md) are listed separately.

```bash
./gradlew :api:bootJar
```

**2. Save this as `euhedral.yaml`**, using your artifact's path:

```yaml
model:
  artifact: /models/qwen-q3.edrl
  name: qwen
```

**3. Start the server** (Linux):

```bash
java --enable-native-access=ALL-UNNAMED -jar api/build/libs/euhedral-inference-linux-x64.jar --config euhedral.yaml
```

**4. Send a request:**

```bash
curl http://localhost:1738/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen",
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
  "max_tokens": 128
}'
```

See [client examples](docs/GETTING_STARTED.md#connect-a-client), [Docker deployment](docs/DEPLOYMENT.md),
or the [configuration guide](docs/OPERATIONS.md) for the next steps.

## Choosing an artifact

These formats trade model size, speed, and similarity to the original model. Results below use Swift-1.5,
a fine-tune of Qwen3.8-27B. The converter accepts either checkpoint.

| Artifact | File size | When to choose it |
|---|---|---|
| `q3` | 11.68 GiB | Fast generation with room for long contexts on a 16 GB card |
| `q3-compressed` | 11.00 GiB | The smallest file and more spare GPU memory |
| `nvfp4` | 14.52 GiB | Higher fidelity in the uncompressed format |
| `nvfp4-compressed` | 13.77 GiB | Higher fidelity and fast prefill |
| `NVFP4-Trellis-RCO` | 11.17 GiB | The fastest generation in these tests, with a compact mixed format |

Q3 uses fewer bits per weight; NVFP4 preserves more of the original model's predictions.
The [quality comparison](#quality) shows the measured differences. Conversion options are in the
[artifact converter guide](tools/README.md), and the mixed format is described in the
[mixed quantization guide](docs/quantization/MIXED_QUANTIZATION.md).

## Performance

Measured on a **16 GB RTX 5070 Ti**, running Linux and also driving the desktop display.
**All benchmark measurements used an overclock: +150 MHz core, +1500 MHz memory, and a 350 W power limit (stock: 300 W).**
All five artifacts were tested at 4K, 32K, 64K, and 128K context lengths.

### Prefill (t/s)

| Artifact | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| `q3` | 2,099 | 1,726 | 1,447 | 1,085 |
| `q3-compressed` | 1,986 | 1,655 | 1,399 | 1,058 |
| `nvfp4` | 4,111 | 2,883 | 2,182 | 1,399 |
| `nvfp4-compressed` | 4,033 | 2,844 | 2,160 | 1,413 |
| `NVFP4-Trellis-RCO` | 2,586 | 2,039 | 1,663 | 1,203 |

### Decode (t/s)

| Artifact | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| `q3` | 130.7 | 126.5 | 109.4 | 69.6 |
| `q3-compressed` | 114.4 | 103.1 | 93.3 | 65.0 |
| `nvfp4` | 105.3 | 102.7 | 64.1 | 32.4 |
| `nvfp4-compressed` | 135.0 | 124.7 | 89.6 | 49.7 |
| `NVFP4-Trellis-RCO` | 147.2 | 114.6 | 93.1 | 78.3 |

### Time to first token

Time from starting the request to the first output token; lower is better.

| Artifact | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| `q3` | 1.90s | 19.0s | 43.9s | 119s |
| `q3-compressed` | 2.01s | 19.7s | 45.5s | 121s |
| `nvfp4` | 0.97s | 11.4s | 29.1s | 91.7s |
| `nvfp4-compressed` | 0.99s | 11.5s | 29.4s | 90.9s |
| `NVFP4-Trellis-RCO` | 1.54s | 16.1s | 38.2s | 107s |

<details>
<summary>Memory use</summary>

| Artifact | File | Device memory in use at 32K / 64K / 128K | Host-backed weights at 32K / 64K / 128K | Longest context run |
|---|---|---|---|---|
| `q3` | 11.68 GiB | 12.8 / 13.4 / 14.1 GiB | none | 128K |
| `q3-compressed` | 11.00 GiB | 12.2 / 12.8 / 14.0 GiB | none | 128K |
| `nvfp4` | 14.52 GiB | 14.2 / 14.2 / 14.2 GiB | 1.02 / 1.58 / 2.79 GiB | 128K |
| `nvfp4-compressed` | 13.77 GiB | 14.2 / 14.2 / 14.2 GiB | 0.27 / 0.84 / 2.02 GiB | 128K |
| `NVFP4-Trellis-RCO` | 11.17 GiB | 12.3 / 12.9 / 14.0 GiB | none | 128K |

Device memory is the engine's peak allocation. CUDA context and kernel modules use additional memory.
Some NVFP4 weights stay in system RAM when the model and context exceed the GPU's memory budget.

</details>

<details>
<summary>Benchmark conditions and variability</summary>

Measured on 2026-10-16 at `1eea731`; `NVFP4-Trellis-RCO` on 2026-10-19 at `fea6b96f`.
Each result is one measured run after a warmup, except the `nvfp4` 4K/32K and `nvfp4-compressed` 128K
generation speeds, which are means of two runs.

Generation uses greedy decoding with MTP: two drafted tokens for Q3 and four for NVFP4 and
`NVFP4-Trellis-RCO`. Prompt lengths are 3,964, 32,612, 63,362, and 128,000 tokens.
The 4K and 32K tests allow 32,768 tokens of context; the 64K and 128K tests allow 65,536 and 131,072.
Each prompt is followed by 128 generated tokens, or 64 at 128K. The 128K prompt is generated text;
the others are chat prompts. Prefill and time to first token include the whole prompt.

Decode speeds vary by several percent with draft acceptance. Runs that stream weights from RAM
can vary by up to 20%. These are measurements of this machine and configuration.
Use the [benchmarking guide](docs/evaluation/BENCHMARKING.md) to run your own comparisons.

</details>

<details>
<summary>DFlash2 comparison</summary>

The optional DFlash2 drafter was also tested with `nvfp4-compressed` (15.04 GiB including the drafter).

| Drafter | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| MTP | 146.6 | 129.9 | 103.5 | 49.7 |
| DFlash2 | 158.6 | 128.4 | 91.4 | 49.4 |

Decode (t/s). This separate comparison uses 256 generated tokens, or 64 at 128K;
the 128K prompt is generated text and the others use the chat corpus. DFlash2 uses more memory,
leaving less room for the target model. See the [DFlash2 guide](docs/speculation/DFLASH2.md) for setup and details.

</details>

## Quality

These results compare each artifact with the original BF16 Swift-1.5 model on a held-out set of
112 conversations and documents. Lower KL means closer predictions; higher top-1 agreement means
more often choosing the same next token.

| Artifact | Mean NLL | Perplexity | NLL vs BF16 (± s.e.) | KL(BF16 ‖ artifact) | Top-1 agreement |
|---|---|---|---|---|---|
| Swift-1.5 BF16 reference | 0.609 | 1.84 | | | |
| `q3` | 0.699 | 2.01 | +0.090 ± 0.007 | 0.106 | 91.0% |
| `q3-compressed` | 0.720 | 2.05 | +0.111 ± 0.008 | 0.131 | 89.8% |
| `nvfp4` | 0.654 | 1.92 | +0.045 ± 0.005 | 0.047 | 94.3% |
| `nvfp4-compressed` | 0.648 | 1.91 | +0.040 ± 0.005 | 0.046 | 94.3% |
| `NVFP4-Trellis-RCO` | 0.660 | 1.93 | +0.051 ± 0.006 | 0.072 | 92.9% |

The set covers conversation, reasoning, coding, tool use, multiple languages, prose, and code.
It includes the original model's own answers, which lowers perplexity. The two NVFP4 formats have
the smallest measured differences from the original model; tool use has the largest differences
for every format. See [results by domain](docs/evaluation/RELEASE_TEST.md#quality-by-domain).

<details>
<summary>Compression and numerical checks</summary>

Compression is also checked separately from quantization:

| Pair | Result |
|---|---|
| `q3-compressed` against the row-split Q3 with the same values (`q3_p2e2 expand`) | lossless: every route's logits are bitwise equal (`P2e2CudaIntegrationTest`) |
| NVFP4-SD4 against `nvfp4`, both converted from Qwen3.8-27B with the same importance matrix (on one fixed document, 1279 scored tokens) | NLL +0.005 ± 0.009, KL 0.037, top-1 agreement 89.8% |

Production kernels are compared with exact scalar kernels on the same weights, over 1,024 forced decode
steps after a 512-token prefill. These results use round-to-nearest artifacts:

| Artifact | Hidden-state relative error | Mean KL | Top-1 agreement |
|---|---|---|---|
| `q3`, `q3-compressed` | 7.6% | 4.7e-3 | 97.9% |
| `nvfp4` | 7.0% | 3.9e-3 | 97.3% |
| `nvfp4-compressed` | 7.4% | 3.4e-3 | 97.8% |

The error settles within about 150 positions and does not grow afterward. Speculative verification is
also checked for identical logits, state, and greedy output against ordinary decoding on all four
Q3/NVFP4 artifacts.

</details>

The main quality comparison scores 143,360 token positions against BF16 Swift-1.5 in llama.cpp.
The Q3/NVFP4 results use GPTQ-calibrated artifacts and UkisAI's Swift-1.5 importance matrix;
the compressed Q3 artifact uses a reduced code range, so it holds different values from `q3`.
See [quality methodology](docs/evaluation/QUALITY.md) and the [release test](docs/evaluation/RELEASE_TEST.md)
for preparation, measurement, and reproduction details.

## API and configuration

Connect an OpenAI client to `http://localhost:1738/v1`, or an Anthropic client to `http://localhost:1738`.
The server supports Chat Completions, Responses, and Messages, including streaming.

- [Reasoning](docs/REASONING.md): control thinking effort and read reasoning output.
- [Tool calling](docs/TOOL_CALLING.md): define tools and return their results.
- [Structured output](docs/STRUCTURED_OUTPUT.md): request JSON or apply a JSON schema.
- [API compatibility](docs/API_COMPATIBILITY.md): supported fields and endpoints.

Use [the example configuration](config/euhedral.example.yaml) to adjust context length, sampling, and concurrency.
The [operations guide](docs/OPERATIONS.md) covers environment variables, health checks, and metrics.

## Development

See [building and testing](docs/development/BUILDING.md) to work on the project, or start with the
[architecture overview](docs/architecture/README.md). The [documentation index](docs/README.md) lists all guides.

Licensed under the [Apache License, Version 2.0](LICENSE).
