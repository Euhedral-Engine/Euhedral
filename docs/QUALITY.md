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
the 16 nats below each position's maximum, which bounds the KL resolution at about 1e-4.

## Compression-induced error

`compare_teacher_forced.py` compares reports with one another. `q3-compressed` is lossless, so its report is byte for byte the
`q3` report; `nvfp4-compressed` (SD4) is compared with `nvfp4`:

```bash
python3 tools/compare_teacher_forced.py q3.bin q3_compressed.bin
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

## Exact speculative verifier

Speculative decoding verifies drafts with the base model in row-exact mode: every row of a verification is bit for bit what a
one-row decode at that position computes. `SpeculativeVerifyCudaIntegrationTest` checks the logits and the KV and GDN state
against one-row decode; `SpeculativeDecodeCudaIntegrationTest` checks that generated tokens and final state equal those of
ordinary greedy decoding, across prompts that exercise accepted, partly accepted and rejected drafts. Pass
`-Peuhedral.speculative.artifact=ARTIFACT` and, for NVFP4 on a 16 GB card, `-Peuhedral.speculative.host-mib=1280`.
