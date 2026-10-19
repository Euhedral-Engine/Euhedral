# Release test

A held-out quality set: the README's quality figures and release checks are measured on it. The research texts T1–T4
([QUALITY.md](QUALITY.md)) have informed many decisions and stay the benchmarks for screening variants; this set
does not choose among them. It answers whether the changes made Swift-1.5 better overall, or fitted the texts measured
over and over.

| Set | May influence |
|---|---|
| Calibration corpus | weights |
| T1–T4 | research decisions (screening, choosing among variants) |
| Release test | nothing: it measures the artifacts' published quality and confirms decisions already made |

## Version 2

Frozen 2026-10-17. 112 chunks of 2,560 tokens (286,720 tokens); the second half of each chunk is scored (positions
1281..2559, 143,360 scored tokens). Each chunk is one independent conversation or document.

| Domain | Chunks | Source |
|---|---|---|
| General conversation and reasoning | 24 (12 greedy, 12 sampled) | BF16 Swift-1.5 generations |
| Coding and agent workflows | 24 (12 greedy, 12 sampled) | BF16 Swift-1.5 generations |
| Tool use and structured output | 16 (greedy) | BF16 Swift-1.5 calls; tool results from authored fixtures |
| Multilingual (es, zh, ja, ar, hi, code-switching) | 16 (8 greedy, 8 sampled) | BF16 Swift-1.5 generations |
| Natural and technical prose | 16 | human-written, first published after 2026-09-21 |
| Source code and structured documents (Java, Rust, Python, SQL, config, data) | 16 | human-written, first published after 2026-09-21 |

- **Prompts and documents** (`tools/t5/records.py`, `sources.py`, `overlap.py`): newly written prompts and human
  material published after 2026-09-21 (the newest date on Swift-1.5's model card), from sites and repositories other
  than T2's and T3's; checked for n-gram overlap with T1–T4, their prompts and the calibration corpus, and for dates
  by commit or publication date.
- **Generation** (`tools/t5/generate.py`): the engine serving the BF16 Swift-1.5 artifact (BF16 attention caches,
  every weight on an RTX PRO 6000) through its OpenAI-compatible API, 57 conversations at once; sampled conversations at
  Swift-1.5's published settings (temperature 1.0, top-p 0.95, top-k 20, min-p 0) with random draws keyed by seed and
  position; tool calls answered from the fixtures. Chat template, thinking and tool-call tokens are kept. The 32
  conversations that offer tools (all of tool use, 8 of coding) were generated with the checkpoint's native
  `<tool_call>` format ([TOOL_CALLING.md](TOOL_CALLING.md)), so the history the server replays matches what the model
  wrote: 0.5% of follow-up requests in tool use, 0% in coding, do not continue the previous prompt and completion token
  for token. Version 1 had generated them through a JSON envelope the checkpoint was not trained on and is retired.
- **Chunks** (`tools/t5/chunks.py`): the first 2,560 tokens of each conversation or document. llama.cpp scores text, so
  a chunk is the tokenization of its own text: two conversations whose first 2,560 ids encode differently once decoded
  are cut at the nearest prefix that encodes to exactly 2,560 tokens (one cut one token later, one with two tokens
  re-encoded). Chunks are ordered so that no token spans a join: every document chunk follows a generated chunk (which
  starts with the template's special token) whose end does not merge with the document's start. The tokens llama.cpp
  read back equal the frozen token file.
- **References**: BF16 Swift-1.5 in llama.cpp (commit 67a17c1, CUDA build, weights in host memory streamed through
  the GPU), independent of the engine that wrote the text:
  `LLAMA_PERPLEXITY_PARSE_SPECIAL=1 llama-perplexity -m Swift-1.5-Qwen3.8-27B-BF16.gguf -f text.txt -c 2560 -b 2560
  -ub 2560 --chunks 112 -ngl 0 --kl-divergence-base reference.kld` (BF16 perplexity 1.8713 on the scored halves).

**Cross-check** (on version 1, whose 80 other chunks version 2 shares): the engine's BF16 artifact (the one that wrote the text; 13.9 GB on the device, 21.8 GB in host memory,
18.9 GB read from disk) scored against the references once: mean KL 0.0029 ± 0.0011, top-1 agreement 98.6%, KL median
0.00004, p95 0.0020, p99 0.0081, dNLL +0.017 ± 0.004 (on T1: KL 0.00057, top-1 97.3%). The median and tail percentiles
are at the kernel noise floor; the mean and dNLL are raised by a few positions (largest KL 24.7). 14,331 s.

The set lives outside the repository. Its manifest (`frozen/manifest.json`, written by `tools/t5/freeze.py`) records
the sha256 and size of every file the test depends on: the corpus, the references, the BF16 model file, the tokenizer
and chat template, the generation record and settings, the prompt, document, fixture and check files, and the
scorer's commit (8494683).

| File | sha256 |
|---|---|
| `frozen/manifest.json` | `eccd6a99f59f821fe6e6f81c6e1d4bb4839f3e809f067b1c314de2d871cff474` |
| `frozen/tokens.bin` (286,720 int32) | `6b4e8f10af13ad9f3c0793d4eec6c68585e51dcfc09d37357c8d4c43ca27a3bd` |
| `frozen/chunks.jsonl` | `5319e9130b2a61922f11290664c33d67e9f7106cbed1f6333d719f501544900f` |
| `frozen/text.txt` | `75026930a31f04ecec23ec49d6db651a3e05a0caa32c1ecf76eb01faa2747cf5` |
| `frozen/reference.kld` (71,144,979,604 bytes) | `a5eb3505e53bee093010652ec90efceab8d80e38e1c7b61368d350e0e3a51a66` |
| BF16 model (`Swift-1.5-Qwen3.8-27B-BF16.gguf`) | `f1be0345521b1de10fc65aaf6bbbf2e49f7dff9b9c55b3b47070b71a201ebeee` |
| `tokenizer.json` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `generation/progress.json` | `0c36140c85dd86211c74d775e7f96fe9cbe0d26d25156ef4ff8c4550ba96248e` |

Any change to a recorded file makes a new version.

## Quality by domain

KL(BF16 ‖ artifact) and top-1 agreement per domain, measured 2026-10-17 (`NVFP4-Trellis-RCO` on 2026-10-19; the README's
quality table is the overall column):

| Artifact | Overall | Conversation | Coding | Tool use | Multilingual | Prose | Code |
|---|---|---|---|---|---|---|---|
| `q3` | 0.106 / 91.0% | 0.111 / 89.1% | 0.090 / 91.3% | 0.238 / 90.9% | 0.077 / 91.0% | 0.054 / 92.0% | 0.072 / 92.2% |
| `q3-compressed` | 0.131 / 89.8% | 0.141 / 88.1% | 0.109 / 90.4% | 0.282 / 89.7% | 0.098 / 89.9% | 0.069 / 90.8% | 0.091 / 90.5% |
| `nvfp4` | 0.047 / 94.3% | 0.038 / 93.5% | 0.036 / 94.9% | 0.133 / 93.7% | 0.024 / 94.8% | 0.019 / 95.0% | 0.045 / 94.0% |
| `nvfp4-compressed` | 0.046 / 94.3% | 0.037 / 93.6% | 0.038 / 95.0% | 0.135 / 93.6% | 0.024 / 94.8% | 0.018 / 94.8% | 0.033 / 93.8% |
| `NVFP4-Trellis-RCO` | 0.072 / 92.9% | 0.064 / 92.0% | 0.060 / 93.3% | 0.192 / 92.2% | 0.044 / 93.0% | 0.033 / 93.3% | 0.050 / 93.7% |

## Scoring

`tools/t5/run.py` scores the candidates with `quantization score --values` and prints, per domain and overall:

| Metric | Purpose |
|---|---|
| Mean KL(reference ‖ candidate) | primary quantization quality measure |
| Top-1 agreement | whether quantization changes the preferred token |
| KL p95 / p99 | severe errors hidden by averages |
| dNLL | change in likelihood of the observed tokens |
| Paired KL difference against the baseline, 95% interval | specialization or regression by domain |

The interval is a bootstrap over chunks (each one conversation or document), resampled within each domain. Expected
resolution: overall differences of a few percent; per domain, at 16 chunks, about 10%.

## Rules

- **Aggregates only.** The runner checks every file against the manifest, then prints only aggregate and per-domain
  numbers; the per-token values are deleted after the run. Nothing per token, per chunk or per prompt is looked at.
- **No screening.** Variants are screened and chosen on T1–T4; this set measures the artifacts that are published and
  confirms a decision already made.

A behavioural check (thinking length, checkable task success, MTP tokens per verification) on a subset of the prompts
is reported beside the test under its own thresholds; it is not part of the KL gate.
