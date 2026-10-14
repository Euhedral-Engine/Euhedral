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

Each artifact is teacher-forced over the same positions (prefix 1281, 1279 steps) by the quantization module's `score`
command (quantization/README.md):

```bash
./gradlew :quantization:run --args="score --artifact /models/qwen3_8_27b_q3.edrl --results results.txt \
    --reference bf16.kld --name q3-t1"
```

`score` prints the mean NLL of the forced tokens, the paired NLL difference to the reference with its standard
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
./gradlew :quantization:run --args="score --artifact qwen3_8_27b_nvfp4_compressed.edrl --results results.txt \
    --reference swift_t2.kld --name nvfp4c-t2"
```

`score` forces every chunk of the reference on a fresh sequence, three chunks at a time, scores each logits row on the
GPU where the engine left it, and reports chunk-level standard errors (tokens within a document are correlated), KL
percentiles and delta-p. It forces 8 tokens per speculative verification (`--rows`, 1 for one-row decode):
verification is row-exact, so its rows are bit for bit those of one-row decode. A 25-chunk text takes about 200
seconds on the 16 GB card; T1 about 9.

The BF16 references stream the weights from the page cache, so they need the 54 GB GGUF to stay cached: with nothing else
holding memory a chunk takes about 10 seconds; beside a 29 GB process it took about 40.

Swift-1.5 artifacts, each scored against Swift-1.5 BF16 (`dNLL ± chunk-level SE / mean KL / top-1`):

| Artifact | T1 (1 chunk) | T2 wikitext (25) | T3 code (24) | T4 own generations (25) |
|---|---|---|---|---|
| nvfp4-compressed | +0.033 ± 0.007 / 0.028 / 90.3% | +0.002 ± 0.005 / 0.032 / 93.2% | +0.038 ± 0.005 / 0.059 / 96.8% | +0.093 ± 0.021 / 0.149 / 93.9% |
| nvfp4 | +0.024 ± 0.008 / 0.028 / 90.9% | +0.012 ± 0.005 / 0.032 / 93.1% | +0.035 ± 0.005 / 0.059 / 96.9% | +0.079 ± 0.027 / 0.148 / 93.4% |
| q3 | +0.088 ± 0.014 / 0.086 / 84.6% | +0.040 ± 0.005 / 0.080 / 89.2% | +0.130 ± 0.015 / 0.183 / 93.6% | +0.081 ± 0.035 / 0.263 / 90.6% |
| q3-compressed | +0.101 ± 0.017 / 0.106 / 84.0% | +0.022 ± 0.013 / 0.102 / 87.8% | +0.165 ± 0.021 / 0.225 / 92.7% | +0.070 ± 0.031 / 0.292 / 90.0% |

On T4 the chunk-level NLL errors are large (0.03 to 0.11 nats): its chunks mix near-deterministic answer text with
open-ended thinking, so mean KL and top-1 agreement resolve it better than dNLL.

**Large files on a shared exFAT drive are read, never memory-mapped.** Page faults on big mappings of such a drive hung the
kernel of the development machine (Linux 7.0) three times. The converter, `score` and
`dequantize_gguf.py` use positioned reads; llama.cpp, which maps its models, reads them from a local ext4 disk (the BF16
GGUFs and `weight_only.sh`'s work directory).

## GPTQ

Every artifact's decoder layers, LM head and draft head are rounded by GPTQ (the `quantization` module, Frantar et al., 2022):
each linear's columns are rounded in order, and each column's rounding error is carried into the columns not yet
rounded through the Cholesky factor of the inverse Hessian H = XᵀX of the linear's inputs, so the rounding minimizes
the output error on the calibration activations rather than the weight error. Scales are chosen as the converter
chooses them (group scales by weighted search within the codes the layout stores, -3..3 for `q3-compressed`'s
P2E2 tensors; block scales by weighted search; the SD4 table by its error DP), weighted by diag(H), on the
error-updated weights; the codes go into the converter with `--codes-from`, and the token embedding is rounded by the
converter (`--imatrix`). `q3` is rounded in act-order (columns by decreasing diag(H), each group's scale chosen when
its first column comes up) with its MTP layer rounded by GPTQ too, on the rows the engine feeds it; the other three
artifacts round their columns in order and leave the MTP layer to the converter. Act-order with the MTP layer was
measured on all four recipes (KL T1 / T2 / T3 / T4): `q3` 0.086 / 0.080 / 0.183 / 0.263, adopted; `q3-compressed`
0.105 / 0.096 / 0.220 / 0.307, `nvfp4-compressed` 0.028 / 0.029 / 0.058 / 0.151 and `nvfp4` 0.029 / 0.032 / 0.061 /
0.158, not adopted (no clear gain over the table above, T4 worse). A calibration set of 384 sequences (737 answers)
gave `nvfp4-compressed` KL 0.035 on T2 and was rejected. A wider NVFP4 block-scale search (`calibrate --nvfp4-above N --nvfp4-scale-max M`: up to N E4M3 codes above
the nearest scale, global scale `amax / (6 M)`) was measured with act-order and the MTP layer: `nvfp4-compressed` at
5 / 256 gave KL 0.028 / 0.030 / 0.058 / 0.156 and at 6 / 448 0.028 / 0.031 / 0.058 / 0.153 (the SD4 table already takes
the wider reach); `nvfp4` at 5 / 256 gave 0.026 / 0.031 / 0.057 / 0.157. Neither beats the shipped artifacts on T4, so
the search stays at 2 codes above the nearest scale with E4M3's 448. Block-wise training of `q3` (`quantization train-blocks`: every layer in turn trains
its group scales and norms, codes fixed, so that it maps the quantized model's hidden states onto the BF16 layer's
outputs; 200 calibration sequences, 2 epochs) cut each block's held-out output error by about 36% on its own (layers 10
and 11) and gave KL 0.088 / 0.080 / 0.202 / 0.225 on T1-T4: the model's own generations improve, code gets worse, so
`q3` keeps its GPTQ scales. End-to-end distillation from there (`quantization train-model`: the same scales and norms
trained through all 64 layers at once against the BF16 model's top-64 log-probabilities plus one bucket for the rest of
the vocabulary; 200 sequences, Adam over 8 sequences per update, 1 epoch) lowered the held-out calibration KL from 0.173
to 0.165 and gave KL 0.087 / 0.081 / 0.194 / 0.310 on T1-T4: the model's own generations get much worse, so it is not
adopted either. The LM head stays NVFP4: rounded by GPTQ in Q5G64 instead (`calibrate --output-head Q5G64_F16S`,
the converter's `--output-head`; 0.12 GiB more), it gave KL 0.084 / 0.079 / 0.183 / 0.264 for `q3` and 0.027 / 0.031 /
0.061 / 0.159 for `nvfp4` on T1-T4. The pre-calibration recipes, converted from Swift-1.5 with the converter of
October 10, give KL 0.193 / 0.199 / 0.254 / 0.466 (`q3`) and 0.037 / 0.047 / 0.073 / 0.228 (`nvfp4-compressed`). On
the 118 T4 prompts, Swift-1.5 thinks 248.6 tokens on average before answering; greedy, `q3` thinks 269.9 (the
pre-calibration `q3` 366.6) and `nvfp4-compressed` 261.5 (282.3). MTP tokens per verification at 4K / 32K: `q3` 2.431
/ 2.466 (2.454 / 2.454), `nvfp4-compressed` 2.936 / 3.006 (2.953 / 3.117). A residual-stream rotation (`quantization rotate`: Q = diag(s) (H_20 ⊗ H_256) /
sqrt(5120), norm weights folded into the linears they feed, the MTP layer left unrotated so the artifact does not
draft), then GPTQ in the rotated basis, gave KL 0.080 / 0.077 / 0.178 / 0.255 for `q3` (dNLL T2 +0.022) and 0.031 /
0.030 / 0.058 / 0.154 for `nvfp4` (dNLL T4 +0.180). Rotation makes every weight family's groups of 64 Gaussian (excess
kurtosis -0.09, the value for 64 Gaussian samples); before it, only the attention key/value and GDN a/b projections
had heavier tails (0.5 to 1.2).

**Calibration set** (`tools/gptq/calibration.py`): 218 sequences of 2048 tokens, none from the holdout: 90 of
Swift-1.5's own answers (thinking, tool calls and chat-template tokens included) to 187 training-split GSM8K, MBPP and
Hermes function-calling prompts that T4 does not use, generated greedily by the engine serving `nvfp4-compressed`;
64 of wikitext-2 train; 64 of CPython, Go, serde_json and cJSON source files other than T3's.

**Calibration** (`quantization/README.md`) runs the sequences through the BF16 checkpoint on the engine's own kernels
one layer at a time: the Hessians accumulate on the GPU while the sequences pass through the layer, the layer's
objects are rounded (cuSOLVER for the Cholesky factor, the module's kernels for the column runs), and the sequences
pass through the rounded layer to give the next layer its inputs. The hidden states stay in pinned host memory. It
takes about 13 minutes per artifact on the 16 GB card, with damping 0.01.

**Engine checks** of the GPTQ artifacts, `nvfp4-compressed` (`nvfp4`): 2.94 (2.99) tokens per verification at 4K and
3.01 (3.17) at 32K over the benchmark's chat prompts, 118 (102) tok/s at 4K and 103 (105) at 32K. Acceptance varies by
up to 0.75 tokens per verification between the four prompts of each length. Rounding the draft head by the converter
instead gave 2.95 at 32K, so the draft head stays rounded by GPTQ. `q3` (`q3-compressed`): 2.43 (2.47) tokens per
verification at 4K and 2.47 (2.50) at 32K, 127 (107) tok/s at 4K and 112 (97) at 32K.

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

## KV cache

The cache stores each key and value row after the H256 rotation as 4-bit codes with an E4M3 scale per 16 elements,
144 bytes per 256-element row (`native/src/attention/nvfp4_kv.cuh`). Measurement builds swap the code grid inside the
same rows: `EUHEDRAL_CUDA_KV_CODEC=1`, `2` or `3` in the environment of the process that compiles the kernels stores
midrise integer grids, magnitudes m + 0.5 for m below 8, 4 and 2 (INT4, INT3 and INT2), with the scale chosen from the
group's largest value as for E2M1; every kernel, the exact twins included, decodes the chosen grid. A fresh Gradle
daemon carries the variable into the scoring JVM:

```bash
./gradlew --stop
EUHEDRAL_CUDA_KV_CODEC=2 ./gradlew :quantization:run --args="score --artifact qwen3_8_27b_nvfp4_compressed.edrl \
    --results results.txt --reference swift_t2.kld --name int3-t2"
```

T2 is also scored at a 32K context: four 32768-token chunks of the wikitext test text, the second half of each scored
(65,532 positions 16K to 32K deep), against Swift-1.5 BF16 in llama.cpp with an F16 cache (`llama-perplexity -c 32768
-b 4096 -ub 4096 --chunks 4 -ngl 0`, 14 minutes; the logits file is 32.5 GB). Mean KL / top-1 agreement:

| Artifact, cache | T1 | T2 | T2 at 32K |
|---|---|---|---|
| `nvfp4-compressed`, E2M1 | 0.0279 / 90.3% | 0.0315 / 93.2% | 0.0288 / 93.4% |
| `nvfp4-compressed`, INT4 midrise | 0.0254 / 90.7% | 0.0300 / 93.3% | 0.0269 / 93.6% |
| `nvfp4-compressed`, INT3 midrise | 0.0369 / 89.7% | 0.0366 / 92.5% | 0.0357 / 92.6% |
| `nvfp4-compressed`, INT2 midrise | 0.1253 / 82.7% | 0.0839 / 88.3% | 0.0765 / 88.7% |
| `q3`, E2M1 | 0.0897 / 85.2% | 0.0798 / 89.2% | 0.0808 / 89.2% |
| `q3`, INT4 midrise | 0.0891 / 84.9% | 0.0793 / 89.3% | 0.0804 / 89.2% |
| `q3`, INT3 midrise | 0.1021 / 84.4% | 0.0845 / 88.9% | 0.0875 / 88.6% |
| `q3`, INT2 midrise | 0.1912 / 79.6% | 0.1284 / 88.6% | 0.1188 / 86.1% |

The `nvfp4-compressed` weights alone (dequantized into a BF16 GGUF, F16 cache in llama.cpp) score 0.0273 / 93.7% on T2
and 0.0240 / 94.0% on T2 at 32K, so the cache, the FP4 prefill and the relaxed kernels together add 0.004 to 0.005 KL at
both depths. KL at 32K is no higher than at 2,560 tokens with any grid: neither the cache nor the GDN state accumulates
error with depth on this text. INT2 is rejected (2.5 to 4.5 times the KL). INT3 stays within 0.01 KL and 1 top-1 point
of E2M1 at 32K on both artifacts; a 3-bit cache would still need long-context retrieval and a reasoning task.

## Exact speculative verifier

Speculative decoding verifies drafts with the base model in row-exact mode: every row of a verification is bit for bit what a
one-row decode at that position computes. `SpeculativeVerifyCudaIntegrationTest` checks the logits and the KV and GDN state
against one-row decode; `SpeculativeDecodeCudaIntegrationTest` checks that generated tokens and final state equal those of
ordinary greedy decoding, across prompts that exercise accepted, partly accepted and rejected drafts. Pass
`-Peuhedral.speculative.artifact=ARTIFACT` and, for NVFP4 on a 16 GB card, `-Peuhedral.speculative.host-mib=1280`.
