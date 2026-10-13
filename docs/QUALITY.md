# Measuring quality

Four different quantities are measured, because they answer different questions and add up to no single score.

| Quantity | Question | Tool |
|---|---|---|
| Model/quantization error | How far is the artifact from the BF16 checkpoint? | teacher-forced NLL and KL against a BF16 reference |
| Compression-induced error | What does the compressed form cost over the uncompressed one? | teacher-forced reports of the two artifacts, compared |
| Relaxed-execution drift | How far do the production kernels' relaxed accumulation orders move a given artifact from exact arithmetic on the same weights? | `RelaxedNumericsDriftCudaIntegrationTest` |
| Exact speculative verifier | Does MTP decoding produce exactly the tokens and state that ordinary greedy decoding produces? | `SpeculativeVerifyCudaIntegrationTest`, `SpeculativeDecodeCudaIntegrationTest` |

All teacher-forced measurements use the same 2560 tokens: `benchmark/src/main/resources/.../chat-corpus-v1-document.md`
tokenized with the Qwen tokenizer. A report is the model's BF16 logits at 1279 forced positions (a 1281-token prefill, then
one decode step per token).

## Model and quantization error

The reference is the BF16 checkpoint run in llama.cpp on the CPU (`llama-perplexity`, built without CUDA), which writes the
reference log-probabilities of the second half of a 2560-token chunk:

```bash
llama-perplexity -m Qwen3.8-27B-BF16.gguf -f chat-corpus-v1-document.md -c 2560 -b 512 --chunks 1 \
    --kl-divergence-base bf16.kld
```

An artifact converted from another checkpoint is measured against that checkpoint's BF16 reference: for Swift-1.5, convert it
with llama.cpp's `convert_hf_to_gguf.py --outtype bf16` and run the same command on the result.

Each artifact is run over the same positions (prefix 1281, 1279 steps; the NVFP4 artifacts need
`-Peuhedral.quality.host-mib=1280` on a 16 GB card):

```bash
./gradlew :core:cudaIntegrationTest --tests '*TeacherForcedQualityCudaIntegrationTest' \
    -Peuhedral.quality.artifact=/models/qwen3_8_27b_q3.edrl -Peuhedral.quality.report=q3.bin \
    -Peuhedral.quality.prefix=1281 -Peuhedral.quality.steps=1279
python3 tools/compare_reference.py bf16.kld q3.bin nvfp4.bin
```

`compare_reference.py` prints the mean NLL of the forced tokens, the paired NLL difference to the reference with its standard
error, the mean KL(reference || artifact) and the top-1 agreement. The reference's log-probabilities are stored in 16 bits over
the 16 nats below each position's maximum, which bounds the KL resolution at about 1e-4. Entries further down are clamped to
one floor value, about 2% of the probability mass over a 248K vocabulary, so the KL sums only the entries above
log-probability -16, as llama.cpp's `--kl-divergence` does: a model's tail lies below that floor, and summing the clamped
entries adds about 0.13 nats to every artifact's KL. The engine's reports hold BF16 logits, whose rounding adds about 0.02.

## Holdout texts

One 1279-token document resolves about 0.02 nats on NVFP4 artifacts and 0.05 on Q3 ones, so changes of that size need the
holdout: four texts of 25 chunks of 2560 tokens each (the second half of every chunk scored, about 32K scored tokens per
text), every artifact scored against the BF16 checkpoint it was converted from.

| Text | Content | Built by |
|---|---|---|
| T1 | `chat-corpus-v1-document.md`, one chunk | (the repository) |
| T2 | wikitext-2 `wiki.test.raw`, the text public KL figures for this model use | `tools/holdout/texts.py` |
| T3 | source code from CPython, the Go standard library, serde_json and cJSON at pinned tags, about 16K tokens each | `tools/holdout/texts.py` |
| T4 | Swift-1.5's own answers, thinking and chat-template tokens included, to 150 public training-split prompts (GSM8K, MBPP, Hermes function calling) | `tools/holdout/t4_prompts.py`, `tools/holdout/t4_generate.py` |

T4 is generated greedily by Swift-1.5 at Q8_0 (54 GB of BF16 does not fit the desktop's memory), through `llama-server`
with `--jinja`; its documents are scored with `--parse-special`, so template tokens stay single tokens.

**References** run in llama.cpp's CUDA build with the BF16 weights in host memory, streamed through the GPU (`-ngl 0`; a
2560-token chunk takes about 10 seconds, against 196 on the CPU). On T1 the GPU and CPU references differ by a mean KL of
3e-5 and agree on the top-1 token at 99.8% of positions.

```bash
llama-perplexity -m Swift-1.5-BF16.gguf -f t2.txt -c 2560 -b 2560 -ub 2560 --chunks 25 -ngl 0 --kl-divergence-base swift_t2.kld
tools/holdout/score.sh nvfp4c-t2 qwen3_8_27b_nvfp4_compressed.edrl swift_t2.kld results.txt
```

`score.sh` runs `TeacherForcedQualityCudaIntegrationTest` with `-Peuhedral.quality.tokens=REFERENCE.kld`, which forces
every chunk of the reference on a fresh sequence, then `compare_reference.py --detail`, which reports chunk-level standard
errors (tokens within a document are correlated), KL percentiles and delta-p, on the GPU when PyTorch with CUDA is
installed (about 2 seconds per chunk). The test forces 8 tokens per speculative
verification (`-Peuhedral.quality.rows`, 1 for one-row decode): verification is row-exact, so its reports are byte for byte
those of one-row decode (checked on T1), and a 2560-token chunk takes about 21 seconds instead of about 170.

The BF16 references stream the weights from the page cache, so they need the 54 GB GGUF to stay cached: with nothing else
holding memory a chunk takes about 10 seconds; beside a 29 GB process it took about 40.

Swift-1.5 artifacts, each scored against Swift-1.5 BF16 (`dNLL ± chunk-level SE / mean KL / top-1`):

| Artifact | T1 (1 chunk) | T2 wikitext (25) | T3 code (24) | T4 own generations (25) |
|---|---|---|---|---|
| nvfp4-compressed | +0.033 ± 0.007 / 0.028 / 90.3% | +0.002 ± 0.005 / 0.032 / 93.2% | +0.038 ± 0.005 / 0.059 / 96.8% | +0.093 ± 0.021 / 0.149 / 93.9% |
| nvfp4 | +0.024 ± 0.008 / 0.028 / 90.9% | +0.012 ± 0.005 / 0.032 / 93.1% | +0.035 ± 0.005 / 0.059 / 96.9% | +0.079 ± 0.027 / 0.148 / 93.4% |
| q3 | +0.106 ± 0.016 / 0.090 / 85.2% | +0.030 ± 0.006 / 0.080 / 89.2% | +0.133 ± 0.017 / 0.190 / 93.6% | +0.006 ± 0.034 / 0.294 / 89.8% |
| q3-compressed | +0.101 ± 0.017 / 0.106 / 84.0% | +0.022 ± 0.013 / 0.102 / 87.8% | +0.165 ± 0.021 / 0.225 / 92.7% | +0.070 ± 0.031 / 0.292 / 90.0% |

On T4 the chunk-level NLL errors are large (0.03 to 0.11 nats): its chunks mix near-deterministic answer text with
open-ended thinking, so mean KL and top-1 agreement resolve it better than dNLL.

**Large files on a shared exFAT drive are read, never memory-mapped.** Page faults on big mappings of such a drive hung the
kernel of the development machine (Linux 7.0) three times. The converter, `compare_reference.py` and
`dequantize_gguf.py` use positioned reads; llama.cpp, which maps its models, reads them from a local ext4 disk (the BF16
GGUFs and `weight_only.sh`'s work directory).

## GPTQ

Every artifact's decoder layers, LM head and draft head are rounded by GPTQ (`tools/gptq/`, Frantar et al., 2022):
each linear's columns are rounded in order, and each column's rounding error is carried into the columns not yet
rounded through the Cholesky factor of the inverse Hessian H = XᵀX of the linear's inputs, so the rounding minimizes
the output error on the calibration activations rather than the weight error. Scales are chosen as the converter
chooses them (group scales by weighted search within the codes the layout stores, -3..3 for `q3-compressed`'s
P2E2 tensors; block scales by weighted search; the SD4 table by its error DP), weighted by diag(H), on the
error-updated weights; the codes go into the converter with `--codes-from`, and the token embedding and the MTP layer
are rounded by the converter (`--imatrix`).

**Calibration set** (`tools/gptq/calibration.py`): 218 sequences of 2048 tokens, none from the holdout: 90 of
Swift-1.5's own answers (thinking, tool calls and chat-template tokens included) to 187 training-split GSM8K, MBPP and
Hermes function-calling prompts that T4 does not use, generated greedily by the engine serving `nvfp4-compressed`;
64 of wikitext-2 train; 64 of CPython, Go, serde_json and cJSON source files other than T3's.

**Calibration** (`tools/gptq/calibrate.py`) runs the sequences through a PyTorch port of the model (`tools/gptq/qwen38.py`;
its BF16 forward pass on T1 is within KL 0.0005 of llama.cpp's BF16 reference) one layer at a time: the Hessians
accumulate on the GPU while the sequences pass through the layer, the layer's objects are rounded, and the sequences
pass through the rounded layer to give the next layer its inputs. The hidden states stay in pinned host memory. It takes
46 to 53 minutes per artifact on the 16 GB card (about 45 s per layer), with damping 0.01.

**Engine checks** of the GPTQ artifacts, `nvfp4-compressed` (`nvfp4`): 2.94 (2.99) tokens per verification at 4K and
3.01 (3.17) at 32K over the benchmark's chat prompts, 118 (102) tok/s at 4K and 103 (105) at 32K. Acceptance varies by
up to 0.75 tokens per verification between the four prompts of each length. Rounding the draft head by the converter
instead gave 2.95 at 32K, so the draft head stays rounded by GPTQ. `q3` (`q3-compressed`): 2.45 (2.47) tokens per
verification at 4K and 2.47 (2.50) at 32K, 122 (107) tok/s at 4K and 111 (97) at 32K.

## Weight-only error

`tools/holdout/weight_only.sh` writes an artifact's dequantized weights into a copy of the BF16 GGUF
(`tools/holdout/dequantize_gguf.py`; tensors matching `--keep` stay BF16) and scores it with
`llama-perplexity --kl-divergence` against the same reference. That isolates what the weights cost from what the engine's
execution adds, and keeping one tensor family at BF16 measures how much of the error that family carries. On T2 against
Swift-1.5 BF16 (25 chunks; token-level standard errors), measured on artifacts rounded by calibrated scale search
throughout (before GPTQ):

| Weights | dNLL | Mean KL | KL p99 | Top-1 |
|---|---|---|---|---|
| `nvfp4` | +0.010 ± 0.002 | 0.042 | 0.36 | 91.6% |
| `nvfp4-compressed` | +0.018 ± 0.002 | 0.043 | 0.36 | 91.6% |
| `q3` | +0.052 ± 0.003 | 0.116 | 1.12 | 86.3% |
| `q3-compressed` (MLP down row-split, docs/COMPRESSED_Q3.md) | +0.066 ± 0.004 | 0.142 | 1.46 | 84.9% |

llama.cpp's own types, quantized from the same BF16 GGUF with the same importance matrix (`llama-quantize --imatrix`;
IQ3_XXS needs an importance entry for every tensor, so its MTP layer is pruned) and requantized to BF16 for scoring:

| Weights | File | dNLL | Mean KL | KL p99 | Top-1 |
|---|---|---|---|---|---|
| IQ3_XXS (LM head Q5_K, embedding IQ3_S, attention Q and K IQ2_S, attention V and GDN QKV Q4_K) | 11.19 GB | +0.066 ± 0.003 | 0.109 | 1.11 | 86.6% |
| IQ3_S (LM head Q6_K, attention V and GDN QKV Q4_K) | 12.60 GB | +0.036 ± 0.003 | 0.072 | 0.72 | 90.0% |
| Q3_K_M (LM head Q6_K, MLP down Q4_K/Q5_K, attention output, attention V and GDN QKV Q4_K) | 13.50 GB | +0.033 ± 0.003 | 0.062 | 0.63 | 90.3% |

`q3` is 12.54 GB. IQ3_S, at about its size, stores the LM head in Q6_K and every other matrix but the Q4_K ones in a
lattice codebook of 3.44 bits per weight.

The LM head of the Q3 artifacts is NVFP4. Measured before the group scales were signed (tools/README.md), with every matrix in
Q3 as the rest of `q3` stores it, the LM head included, the weights' KL is 0.193; the head's format alone moves it:

| LM head | Head size | dNLL | Mean KL | KL p99 | Top-1 |
|---|---|---|---|---|---|
| Q3G64 | 0.50 GB | +0.112 ± 0.004 | 0.193 | 1.45 | 79.2% |
| Q4G64 (rejected: the engine runs it only through new head dispatch, for more KL than NVFP4) | 0.67 GB | +0.071 ± 0.004 | 0.144 | 1.36 | 83.5% |
| NVFP4 (shipped) | 0.71 GB | +0.066 ± 0.004 | 0.140 | 1.35 | 84.2% |
| Q5G64 (rejected: new head dispatch and 0.12 GB more for 0.004 KL) | 0.83 GB | +0.067 ± 0.004 | 0.135 | 1.35 | 84.7% |
| BF16 | 2.54 GB | +0.063 ± 0.004 | 0.132 | 1.34 | 85.0% |

With the head in Q3 and unsigned scales, one family kept at BF16 (in parentheses, the family's share of the main model's 26.9B matrix weights):

| Kept at BF16 | dNLL | Mean KL | KL p99 | Top-1 |
|---|---|---|---|---|
| LM head (`output`, 4.7%) | +0.063 ± 0.004 | 0.132 | 1.34 | 85.0% |
| MLP gate and up (42.5%) | +0.122 ± 0.004 | 0.151 | 1.09 | 81.6% |
| MLP down (21.2%) | +0.099 ± 0.004 | 0.157 | 1.18 | 81.1% |
| full-attention projections (6.2%) | +0.098 ± 0.004 | 0.184 | 1.35 | 79.5% |
| GDN output (`ssm_out`, 5.6%) | +0.082 ± 0.004 | 0.186 | 1.37 | 79.5% |
| GDN input projections (`attn_qkv`, `attn_gate`, 15.0%) | +0.114 ± 0.004 | 0.191 | 1.42 | 79.4% |
| token embedding (4.7%) | +0.115 ± 0.004 | 0.193 | 1.46 | 79.4% |
| full-attention Q, K and V only (Q4/Q5, 4.4%) | +0.105 ± 0.004 | 0.192 | 1.45 | 79.3% |

**Bits per family.** The Q3 artifacts store the GDN value_z projections in Q4 and the MLP down projections in Q4 in the
first and last six layers (Q3 elsewhere). Probes on `q3` with signed scales (weight-only KL 0.127 on T2), each changing
one family's format:

| Change | Mean KL | Bytes | |
|---|---|---|---|
| value_z Q5 to Q4 | 0.127 | -0.38 GB | kept |
| MLP down Q4 in layers 0-7 and 56-63 | 0.114 | +0.18 GB | kept, as layers 0-5 and 58-63 |
| MLP down Q4 in every layer | 0.102 | +0.71 GB | rejected: the bytes |
| GDN and attention outputs Q3 to Q4 | 0.118 | +0.25 GB | rejected: on T4 (the model's own generations) dNLL +0.18 in the engine |
| query_key Q4 to Q3 | 0.135 | -0.20 GB | rejected |

The engine adds little on top of its weights: with those scale-search weights its mean KL on T2 was 0.046 for `nvfp4`
and 0.119 for `q3`, against 0.042 and 0.116 for the weights alone in llama.cpp.

The LM head carries a third of the KL in 4.7% of the weights. The Q4 and Q5 families (GDN input projections and
full-attention Q, K and V) carry about 0.004 together, and the token embedding almost none. Per-token KL and top-1 order the families consistently; dNLL does not (quantization noise moves the forced token's
probability either way), so allocation decisions use KL.

## Compression-induced error

`compare_teacher_forced.py` compares reports with one another. P2E2 is lossless, so a `q3-compressed` report is byte for byte the
report of the row-split artifact with the same values (`q3_p2e2 expand`, docs/COMPRESSED_Q3.md); `nvfp4-compressed` (SD4) is
compared with `nvfp4`:

```bash
python3 tools/compare_teacher_forced.py q3_compressed_expanded.bin q3_compressed.bin
python3 tools/compare_teacher_forced.py nvfp4.bin nvfp4_compressed.bin
```

## Relaxed-execution drift

The test runs one sequence on the exact kernels (the scalar references) and one on the production kernels over the same forced
tokens, and reports per-step hidden-state error, KL and top-1 agreement against the exact arm:

```bash
./gradlew :core:cudaIntegrationTest --tests '*RelaxedNumericsDriftCudaIntegrationTest' \
    -Peuhedral.qwen.artifact=/models/qwen3_8_27b_q3.edrl \
    -Peuhedral.numerics.drift.steps=512 -Peuhedral.numerics.drift.prefix=512
```

It fails when the mean KL reaches 1e-2 at a starting position, when more than 2% of the positions flip the top-1 token
where the exact arm's top-1 logit led the production arm's choice by more than 0.25 (flips within that lead, two BF16
steps, are near-ties: 3 to 17 per 256 positions on the NVFP4 artifacts, depending on the weights), or when the median
over the starting positions of the peak 32-step hidden-state error reaches 12% or of its late/early ratio exceeds 1.5.

## Exact speculative verifier

Speculative decoding verifies drafts with the base model in row-exact mode: every row of a verification is bit for bit what a
one-row decode at that position computes. `SpeculativeVerifyCudaIntegrationTest` checks the logits and the KV and GDN state
against one-row decode; `SpeculativeDecodeCudaIntegrationTest` checks that generated tokens and final state equal those of
ordinary greedy decoding, across prompts that exercise accepted, partly accepted and rejected drafts. Pass
`-Peuhedral.speculative.artifact=ARTIFACT` and, for NVFP4 on a 16 GB card, `-Peuhedral.speculative.host-mib=1280`.
