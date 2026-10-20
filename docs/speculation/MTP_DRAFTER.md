# Training the MTP drafter

The MTP layer drafts the tokens that verification checks ([MTP_CONTRACT.md](MTP_CONTRACT.md)). This page covers
fine-tuning that layer on the target's own answers, the corpus and trainer that do it (quantization/README.md, *MTP
drafter corpus* and *MTP drafter training*), and the drafter NVFP4-Trellis-RCO ships. Verification stays exact:
outputs are the target's whatever the drafter drafts.

## Corpus

`tools/mtp_corpus_prompts.py` writes 3995 prompts from training splits:

| Domain | Prompts | Source |
|---|---|---|
| chat | 700 | UltraChat 200k first turns |
| chat-turn | 300 | UltraChat second turns after the dataset's first exchange |
| code | 800 | Magicoder-OSS-Instruct-75K |
| tool-call | 450 | Hermes function calling, the call itself |
| tool-result | 450 | Hermes function calling, the answer after the dataset's tool calls and responses |
| agent | 500 | coding-agent turns over pallets projects after a directory listing and a file read |
| multilingual | 395 | Aya in Spanish, Chinese, Japanese, Arabic and Hindi |
| math | 400 | GSM8K |

None shares a user message with T4, the held-out set or the calibration set.

`quantization generate` answered them on NVFP4-Trellis-RCO with thinking on:

- 75% sampled as the server samples by default: temperature 1.0, top-k 20 and top-p 0.95 through the engine's
  `TokenSampler`, seeded by the prompt id;
- 25% greedy.

That is 5.37 M answer tokens and 80 GiB of post-final-norm hiddens, at 236 tok/s (batch 8). In the sampled answers,
82.8% of tokens are the target's greedy choice, which bounds a sampled verification's acceptance of any drafter.

Every position's label is the target's greedy choice: under sampling, a draft is accepted with the probability the
target gives it, which the greedy choice maximizes. The drafter reads the text's own tokens as its inputs, as the
decoder feeds them.

## Training

`train-drafter` on that corpus:

- 3915 training and 80 held-out sequences;
- every parameter of the layer, from Swift-1.5's MTP weights, projections NVFP4-rounded as exported;
- depth 4 unrolled, windows of 1024 lineages over up to 4096 key positions;
- an update per 16384 lineages, 313 updates in one epoch;
- AdamW at 2e-5 with 10 warmup updates and cosine decay to a tenth.

The epoch takes 79 minutes on the RTX 5070 Ti.

Held-out accepted drafts per lineage (all held-out lineages whose four drafts have targets, checked against the text's
own tokens):

| Update | All | Greedy answers | Sampled answers |
|---|---|---|---|
| 0 (Swift-1.5's MTP weights, NVFP4-rounded) | 2.2146 | 2.3474 | 2.1589 |
| 100 | 2.2560 | 2.3869 | 2.2012 |
| 200 | 2.2718 | 2.4091 | 2.2142 |
| 313 | 2.2806 | 2.4191 | 2.2225 |

Conditional acceptance at the end: 0.79, 0.77, 0.79 and 0.80 at drafts 1 to 4.

`export-drafter` writes the layer into the artifact as NVFP4 projections (round to nearest, byte for byte the
converter's) and BF16 norms; the base artifact's tokenizer is carried.

## In the model

NVFP4-Trellis-RCO with the trained drafter, MTP depth 4, greedy, 256 tokens per prompt
(`MtpDraftLengthScreenCudaIntegrationTest`).

24 drafting prompts (`tools/mtp_draft_prompts.py`, about 1K to 5K tokens):

| Category | Tokens per verification | tok/s |
|---|---|---|
| chat | 3.17 | 111.5 |
| code | 3.75 | 128.6 |
| agentic | 3.82 | 129.9 |
| all | 3.51 | 121.5 |

Longer prompts:

| Prompts | Tokens per verification | tok/s |
|---|---|---|
| 8 at 32K (two rounds) | 3.53 | 130.0 |
| 128K chat document (119.5K tokens) | 3.71 | 112.5 |
| 128K agent edit (119.6K tokens) | 4.00 | 121.4 |
| 128K agent explanation (123.4K tokens, 98 output tokens) | 3.77 | 109.4 |

## Rejected

- **Draft depth 5 with the trained drafter.** On the 24 prompts it ran at 123.7 tok/s overall (chat 113.0, code 130.3,
  agentic 135.5; 3.83 tokens per verification). At 32K it ran at 124.9 tok/s (chat 105.7, agentic 159.2; 3.78 tokens per
  verification): each verified row costs more at long context. Depth 6 ran at 122.2 tok/s on the 24 prompts.
- **Greedy answers only for the corpus.** Greedy decoding of a model tuned for sampling at temperature 1 loops: 31% of
  420 greedy answers ran to the 4096-token cap, against 20% of the first sampled ones. Loops are easy drafts that do not
  occur in sampled serving.
- **The trellis 4.0 bpw MTP layer that NVFP4-Trellis-RCO carried before.** It drafted 3.40 tokens per verification on
  the 24 prompts (118.2 tok/s) and 3.66 and 3.88 on the 128K chat document and agent edit.
