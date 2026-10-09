# MTP draft confidence

How far the MTP head's drafts hold up from each output position, and how well the head's own probabilities predict
that. This is the measurement for draft-length policies that decide during decoding how many drafts a verification
checks, such as a confidence-gated drafting loop (DLoop, arXiv 2610.07659): keep drafting while the head stays
confident, then verify everything at once.

## Method

`MtpDraftConfidenceCudaIntegrationTest` runs `MtpDecoder` in observation mode (`observe(listener, true)`):
- each step drafts 16 tokens (`MtpDecoder.MAX_DRAFTS`) and verifies one (2 rows), so each step commits exactly one
  token, and every output position starts a step;
- the draft rows come back to the host, which selects each draft as the device argmax does (the lowest shortlist row
  among equal maxima) and records the natural log of its probability under a softmax over the head's 131,072-row
  shortlist;
- the output is greedy decode's (each step commits a verified token), so a draft prefix's acceptance at a position is
  its agreement with the generated tokens.

Because drafts from a position depend only on that position, `tools/mtp_loop_screen.py` replays any draft-length
policy on the report exactly. From position i a policy drafts n tokens, the verification commits min(agreement, n) + 1
tokens, and the next step starts at i plus that count. Predicted tok/s comes from a cost model fitted to measured MTP4
steps ([DFLASH2.md](DFLASH2.md) section 6, `nvfp4-compressed`):
- each drafted token costs a quarter of MTP4's draft time;
- a verification costs MTP4's verify time, plus 0.5, 0.6, 0.8 or 1.5 ms per row beyond 5 (4K, 16K, 32K or 64K);
- a 9-row or wider verification adds an estimated 6 ms, for the exact 9-16-row linears ([NVFP4_NATIVE.md](NVFP4_NATIVE.md)).

The model predicts MTP4 on chat prompts at 120.5 tok/s at 4K, against 114.5 measured on the benchmark's chat corpus.
Predictions rank policies; they are not measurements.

The prompts (`tools/mtp_draft_prompts.py`, rendered with the checkpoint's chat template, repository files at `main`)
come in three sets of 8:
- **chat:** the benchmark chat corpus's tasks, four over a 3.5K-token document and four standalone, thinking off;
- **code:** tests, review, a whole-file edit and a long refactor over real repository files (up to 7.3K tokens), plus
  Java, Rust, TypeScript and a thinking-on Python task;
- **agentic:** coding-agent transcripts with five tools, tool calls and real tool results (file reads, a grep, test
  output, an edit), 1K-9K tokens, seven of eight with thinking on. These turns end at the agent's next tool call,
  so five generate 67-190 tokens.

Each prompt generates up to 512 tokens, for 9,956 positions in total. Artifact: `qwen3_8_27b_nvfp4_compressed`, greedy.
The run takes 9 minutes.

## Results

**Depth profile.** P(draft j accepted | drafts 1..j-1 accepted) / the head's mean probability of draft j:

| Set | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 |
|---|---|---|---|---|---|---|---|---|
| chat | 0.85 / 0.85 | 0.78 / 0.85 | 0.74 / 0.85 | 0.73 / 0.85 | 0.71 / 0.85 | 0.70 / 0.85 | 0.71 / 0.84 | 0.70 / 0.83 |
| code | 0.91 / 0.92 | 0.88 / 0.92 | 0.86 / 0.92 | 0.84 / 0.91 | 0.83 / 0.91 | 0.81 / 0.90 | 0.81 / 0.90 | 0.80 / 0.90 |
| agentic | 0.90 / 0.89 | 0.87 / 0.90 | 0.84 / 0.90 | 0.84 / 0.90 | 0.85 / 0.90 | 0.83 / 0.90 | 0.85 / 0.92 | 0.84 / 0.92 |

Past the first draft, the conditional acceptance settles at 0.70-0.85 and stays there out to d8. The head's
probability is calibrated at d1 and too high by 0.05-0.10 from d2 on: the head trained on one step reads its own
hidden states as if they were the target's.

**Fixed depths and the gate signal:**

| Set | MTP3 tokens / verification | MTP4 | MTP5 | MTP4 blocks fully accepted | AUROC, first 4 drafts | AUROC, drafts 5-8 given 1-4 |
|---|---|---|---|---|---|---|
| chat | 2.86 | 3.11 | 3.28 | 35.8% | 0.860 | 0.865 |
| code | 3.30 | 3.81 | 4.13 | 58.0% | 0.916 | 0.904 |
| agentic | 3.28 | 3.82 | 4.27 | 56.1% | 0.907 | 0.938 |
| all | 3.10 | 3.49 | 3.75 | 48.6% | 0.899 | 0.907 |

AUROC is the summed log-probability of a block of four drafts as a predictor of "every draft in the block is
accepted", the DLoop gate's signal.

**Draft-length policies** (predicted decode tok/s at 4K / 16K / 32K / 64K):
- *fixed depth* drafts a fixed number of tokens;
- *cumulative t, min m* drafts one token at a time until the running sum of log-probabilities falls below t, with at
  least m drafts;
- *loop k g* drafts blocks of k and adds another while the last block's sum is at least g (DLoop's rule);
- all are capped at 7 drafts (8 rows).

| Set | Policy | Tokens / verification | Drafts / verification | Predicted tok/s |
|---|---|---|---|---|
| chat | MTP4 | 3.11 | 4.00 | 120.5 / 111.8 / 104.3 / 70.6 |
| chat | cumulative -0.5, min 3 | 3.29 | 4.12 | 126.7 / 117.4 / 109.5 / 74.0 |
| chat | loop k3 g-0.5 | 3.30 | 4.27 | 125.8 / 116.6 / 108.7 / 73.4 |
| code | MTP4 | 3.81 | 4.00 | 147.4 / 136.6 / 127.5 / 86.3 |
| code | cumulative -0.5, min 3 | 4.45 | 5.01 | 163.2 / 151.2 / 140.4 / 94.3 |
| code | loop k3 g-0.25 | 4.27 | 4.61 | 159.9 / 148.2 / 137.9 / 92.8 |
| agentic | MTP4 | 3.82 | 4.00 | 147.9 / 137.1 / 128.0 / 86.6 |
| agentic | cumulative -0.5, min 3 | 4.29 | 4.67 | 160.1 / 148.3 / 138.0 / 92.9 |
| agentic | cumulative -1.0, min 1 | 4.53 | 5.34 | 163.1 / 151.1 / 140.1 / 93.9 |
| all | MTP4 | 3.49 | 4.00 | 135.1 / 125.3 / 116.9 / 79.1 |
| all | cumulative -0.5, min 3 | 3.87 | 4.53 | 145.5 / 134.8 / 125.5 / 84.6 |

A gate that knew the outcome would draft exactly the drafts the verification accepts. With this head it would reach
4.09 tokens per verification over all prompts at 7 drafts (chat 3.47, code 4.64, agentic 4.71) and 4.43 at 16
(3.60, 5.23, 5.44): that is the most a draft-length policy can extract from the head as it is.

## Rejected

- **Caps of 12 and 16 drafts** with the current head. Wider than 8 rows, verification pays for the exact 9-16-row
  linears, and the head's 0.70-0.85 per-draft acceptance rarely carries a chain that far. The best 16-draft policy
  leads the best 7-draft one only on agentic prompts at 64K (94.1 against 93.9 tok/s).
- **DLoop's block rule for this drafter.** At 0.95-1 ms per MTP draft against a 22-26 ms verification, a per-token
  cumulative threshold stops at the right draft instead of at a block boundary. It matched or beat the best block
  rule on every set.

## Open

- **Loop-aware training of the MTP layer** (DLoop section 3.3: unroll the head over its own hidden states for one to
  three extra blocks, loss on the first and last block). It targets the overconfidence from d2 on and the
  0.70-0.85 per-draft acceptance, which bound every policy above. The training mix needs code and agentic data (tool
  calls, tool results, thinking) as well as chat: those sets fully accept 56-58% of MTP4 blocks, against 36% for chat,
  and gain the most from deeper drafts.
- **A gated depth in the decoder**: the draft row's log-probability on the device (a log-sum-exp beside the shortlist
  argmax), the stopping rule between recursion quanta, and VERIFY quanta captured at every row count from 2 to 8.
