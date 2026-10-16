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

Two trellis artifacts store the decoder linears as QTIP-style trellis codes (exllamav3's EXL3 format, calibrated by the
quantization module) with a Q3 token embedding and NVFP4 heads: `trellis` at 3.0 bits per weight, 10.26 GiB (11.01 GB),
and at 3.5 bits per weight, 11.67 GiB (12.54 GB) ([docs/TRELLIS.md](docs/TRELLIS.md)).

All four artifacts are converted from Swift-1.5 (`ukisai/Swift-1.5-Qwen3.8-27b`, UkisAI's fine-tune of Qwen3.8-27B); the
converter takes either checkpoint.

The engine reads what an artifact is from the file and picks the fastest validated execution for it: kernels by row count,
FP8 and FP4 tensor-core prefill, the attention implementation, the speculative-decoding depth and which weights stay in
host memory. All of that is automatic.

Any artifact can also carry the DFlash2 drafter (`z-lab/Qwen3.8-27B-DFlash2`): the converter appends it with
`--extend ARTIFACT --dflash2 DIR`, and an artifact that holds it speculates with DFlash2 instead of MTP, with the same
exact verification ([docs/DFLASH2.md](docs/DFLASH2.md)). On the compressed NVFP4 form of Qwen3.8-27B, the model the drafter was trained
for, it decodes 131.9 tok/s at 4K, 112.3 at 32K, 87.3 at 64K and 57.3 at 128K tokens of context (chat corpus and 256 generated
tokens; at 128K generated text and 64 tokens; MTP on the same target: 138.1, 128.6, 100.7 and 82.5); its larger drafter leaves
less of the 16 GB for the target, and the four artifacts above draft with MTP.

## Performance

RTX 5070 Ti (16 GB, 70 SMs) in a desktop that also uses the GPU for its display, Linux, one run per row after one warmup. Greedy
decoding with MTP speculative decoding (two drafted tokens for Q3, four for NVFP4 and trellis); decode `tok/s` is output tokens
per second after the first token. Prefill and time to first token (TTFT) include the whole prompt. Measured on 2026-10-11
(commit 66f972c; `q3` and `q3-compressed` on 2026-10-12, commit 01b4b44) with the GPU at its stock clocks. Decode depends on how
many drafted tokens the verifier accepts on each prompt, which moves single rows by several percent. The trellis rows were
measured on 2026-10-16 (commit 5fae37a) with the GPU overclocked: +150 MHz core and +1500 MHz memory clock offsets and a 350 W
power limit (stock is 300 W).

Context is the prompt length: 4K = 3,964 tokens, 32K = 32,612, 64K = 63,362, 128K = 128,000.

**Prefill (tokens/s)**

| Artifact | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| `q3` | 2,101 | 1,728 | 1,449 | 1,087 |
| `q3-compressed` | 1,997 | 1,659 | 1,400 | 1,060 |
| `nvfp4` | 4,118 | 2,885 | 2,185 | 1,418 |
| `nvfp4-compressed` | 4,040 | 2,849 | 2,165 | 1,437 |
| `trellis` 3.0 bpw | 1,230 | 1,122 | 1,039 | 821 |
| `trellis` 3.5 bpw | 1,333 | 1,151 | 1,028 | 839 |

**Decode (tokens/s, MTP)**

| Artifact | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| `q3` | 136.5 | 113.9 | 107.8 | 83.2 |
| `q3-compressed` | 105.8 | 105.0 | 102.6 | 63.3 |
| `nvfp4` | 138.8 | 114.1 | 85.7 | 38.7 |
| `nvfp4-compressed` | 142.0 | 109.3 | 93.7 | 82.1 |
| `trellis` 3.0 bpw | 137.7 | 145.3 | 109.1 | 82.8 |
| `trellis` 3.5 bpw | 143.2 | 106.3 | 107.5 | 54.3 |

**Time to first token**

| Artifact | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| `q3` | 1.90 s | 18.9 s | 43.9 s | 118 s |
| `q3-compressed` | 2.00 s | 19.7 s | 45.4 s | 121 s |
| `nvfp4` | 0.97 s | 11.4 s | 29.1 s | 90.5 s |
| `nvfp4-compressed` | 0.99 s | 11.5 s | 29.4 s | 89.3 s |
| `trellis` 3.0 bpw | 3.23 s | 29.1 s | 61.1 s | 156 s |
| `trellis` 3.5 bpw | 2.98 s | 28.4 s | 61.8 s | 153 s |

The 4K and 32K rows run with the default 32,768-token context; the 64K and 128K rows set `max-context-tokens` to 65,536
and 131,072. Each prompt is followed by 128 generated tokens (64 at 128K). The 128K prompt is generated text; the others are
chat prompts.

| Artifact | File | Device memory in use at 32K / 64K / 128K | Host-backed weights at 32K / 64K / 128K | Longest context run |
|---|---|---|---|---|
| `q3` | 11.68 GiB | 12.8 / 13.4 / 14.1 GiB | none | 128K |
| `q3-compressed` | 11.00 GiB | 12.2 / 12.8 / 14.0 GiB | none | 128K |
| `nvfp4` | 14.52 GiB | 14.4 / 14.4 / 14.4 GiB | 0.84 / 1.40 / 2.59 GiB | 128K |
| `nvfp4-compressed` | 13.77 GiB | 14.4 / 14.4 / 14.4 GiB | 0.08 / 0.65 / 1.78 GiB | 128K |
| `trellis` 3.0 bpw | 10.26 GiB | 11.4 / 12.0 / 13.1 GiB | none | 128K |
| `trellis` 3.5 bpw | 11.67 GiB | 12.8 / 13.4 / 14.0 GiB | none / none / 0.15 GiB | 128K |

Device memory is what the engine allocated at its peak during the run: the resident weights, the KV cache of the context, the
sequence state, and the workspaces. The CUDA context and kernel modules come on top. An uncompressed NVFP4 model is larger than
the card's memory budget at these contexts, so the engine holds part of it in pinned host memory and streams it in for every token.

