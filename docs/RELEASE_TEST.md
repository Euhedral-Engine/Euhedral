# Release test

A held-out quality test that confirms quantization decisions at release checkpoints. The research texts T1–T4
([QUALITY.md](QUALITY.md)) have informed many decisions and stay the research benchmarks; this test informs none. It
answers one question per release: did the changes make Swift-1.5 better overall, or did they fit the texts measured
over and over?

| Set | May influence |
|---|---|
| Calibration corpus | weights |
| T1–T4 | research decisions (screening, choosing among variants) |
| Release test | nothing: it only confirms a decision already made |

## Version 1

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
  position; tool calls answered from the fixtures. Chat template, thinking and tool-call tokens are kept.
- **Chunks** (`tools/t5/chunks.py`): the first 2,560 tokens of each conversation or document. llama.cpp scores text, so
  a chunk is the tokenization of its own text: two conversations whose first 2,560 ids encode differently once decoded
  are cut at the nearest prefix that encodes to exactly 2,560 tokens (one cut one token later, one with two tokens
  re-encoded). Chunks are ordered so that no token spans a join: every document chunk follows a generated chunk (which
  starts with the template's special token) whose end does not merge with the document's start. The tokens llama.cpp
  read back equal the frozen token file.
- **References**: BF16 Swift-1.5 in llama.cpp (commit 67a17c1, CUDA build, weights in host memory streamed through
  the GPU), independent of the engine that wrote the text:
  `LLAMA_PERPLEXITY_PARSE_SPECIAL=1 llama-perplexity -m Swift-1.5-Qwen3.8-27B-BF16.gguf -f text.txt -c 2560 -b 2560
  -ub 2560 --chunks 112 -ngl 0 --kl-divergence-base reference.kld` (BF16 perplexity 1.8696 on the scored halves).

The set lives outside the repository. Its manifest (`frozen/manifest.json`, written by `tools/t5/freeze.py`) records
the sha256 and size of every file the test depends on: the corpus, the references, the BF16 model file, the tokenizer
and chat template, the generation record and settings, the prompt, document, fixture and check files, and the
scorer's commit (2c6118e).

| File | sha256 |
|---|---|
| `frozen/manifest.json` | `9128d54c41af9a828bbdce6011e45444889a13412e6c3fa2299aa77100001843` |
| `frozen/tokens.bin` (286,720 int32) | `3f86634c8ac7723b7fe09849a4d5aaf3a720e6d78e592b4d2e3bd0da2272f553` |
| `frozen/chunks.jsonl` | `d0b5bbcad35f428382c8d33dbcdd806b84173f66193cb92da319ac229d05b658` |
| `frozen/text.txt` | `1b372495bec31ad505dbf6e21358816238aea1398ddbf1abb3fe3ae9f6f2d88f` |
| `frozen/reference.kld` (71,144,979,604 bytes) | `e4aa5af61832367dc7532476fee0fcf05851a8f6acd9664ad0904d083de55ef4` |
| BF16 model (`Swift-1.5-Qwen3.8-27B-BF16.gguf`) | `f1be0345521b1de10fc65aaf6bbbf2e49f7dff9b9c55b3b47070b71a201ebeee` |
| `tokenizer.json` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `generation/progress.json` | `90e5825fffccb68a9e059e60acdab20af92e8cc3430150fdca3e22e63697d8a2` |

Any change to a recorded file makes a new version.

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
- **Release checkpoints only.** The test runs at designated release checkpoints, never after individual experiments.
- **Retirement.** Once a result informs a quantization change, this version is retired for that change; the next
  checkpoint uses a newly built version.

A behavioural check (thinking length, checkable task success, MTP tokens per verification) on a subset of the prompts
is reported beside the test under its own thresholds; it is not part of the KL gate.
