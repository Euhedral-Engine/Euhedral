# Multi-sequence execution

## Baseline (one sequence)

Measured 2026-10-14 by `Bf16BaselineCudaIntegrationTest` on the BF16 Swift-1.5 artifact
(`qwen3_8_27b_bf16.edrl`, Qwen3.8, 54 GB): one 16 GB RTX 5070 Ti, a 4096-token context, BF16 KV cache, the pinned
host budget set to 24 GiB (`euhedral.residency.host-budget-bytes`, fixed in the test: with the host's own budget
the whole 54 GB model fits in pinned memory, nothing is read from disk, and decode runs at 0.63 tok/s; 24 GiB
reproduces a disk-tier machine). After load 14.4 GB of the card's 16.3 GB is in
use (including about 0.5 GB of desktop use), 19550 MiB of the weights are read from the artifact file on every pass
and 38920 MiB are staged to the device per pass (the 19550 MiB read plus the pinned host share). Every pass of the
model reads the same bytes whatever its row count, so a pass costs about 3.1 s.

Greedy generation of 64 tokens from the chat-rendered prompt "Explain how a heat pump works." (20 prompt tokens);
both modes produce identical tokens.

| Mode | Time (prefill included) | Tokens per second | Quanta | Disk per quantum | Transfer per quantum |
| --- | --- | --- | --- | --- | --- |
| Speculation off | 199.6 s | 0.321 | 1 PREFILL, 64 DECODE | 19550 MiB | 38920 MiB |
| Speculation on (MTP) | 70.5 s | 0.907 | 1 PREFILL, 1 DECODE, 21 VERIFY, 84 DRAFT | 19550 MiB (DRAFT: 0) | 38920 MiB (DRAFT: 0) |

With speculation on, the 21 verifications emit 3.00 tokens each (105 rows), and the drafting quanta read nothing.

### Prefill

A prefill quantum sweeps the weights once per 512 rows (one chunk); a longer prompt is one quantum that sweeps
them several times. Bytes below are the totals the quantum issued.

| Case | Tokens prefilled | Time | Disk | Transfer |
| --- | --- | --- | --- | --- |
| 2048-token prompt, prefix cache off | 2048 | 12.1 s | 78200 MiB | 155680 MiB |
| 512-token turn appended to the conversation | 512 | 3.1 s | 19550 MiB | 38920 MiB |
| 2048-token prompt, prefix cache on, first time | 2048 | 12.6 s | 78200 MiB | 155680 MiB |
| 2048-token prompt, prefix cache on, repeated (1536 tokens restored) | 512 | 3.1 s | 19550 MiB | 38920 MiB |

Prefill costs 5.9 ms per token for a full chunk and 3.1 s for any prompt of up to 512 tokens.

## Prefill share at N concurrent sequences

Assumptions: a pass costs 3.1 s and serves every sequence in it, so a batched verify of N sequences takes the same
time as one (the sweep of the weights is the cost); a prefill quantum serves one sequence, so prefills serialize
with the batched passes. Sequences per card: the card has 1.9 GB free after the engine loads (which includes one
sequence's state and KV cache); each further sequence needs 336 MiB of state (`Residency.SEQUENCE_BYTES`: generate
peaks 1986 MiB higher with 8 sequences than with 2) plus BF16 KV of 64 KiB per token over 4096 tokens (256 MiB), about 0.6 GiB, so N = 4.

Per conversation (a 1.5K-token prompt; 200 generated tokens per assistant turn; speculation at 3.0 tokens per pass):

| Conversation | Prefill sweeps | Prefill | Decode passes | Decode wall at N = 4 | Prefill share |
| --- | --- | --- | --- | --- | --- |
| Tool use: 5 assistant turns, 4 tool results of 300 tokens | 3 + 4 = 7 | 21.7 s | 333 | 258 s | 7.8% |
| Plain: 4 user messages (3 follow-ups of about 100 tokens), 4 assistant turns | 3 + 3 = 6 | 18.6 s | 267 | 207 s | 8.2% |

Prefill would take about 8% of wall time at N = 4: prefill not batched: 8%. The share reaches 20% when a
conversation generates fewer than about 340 tokens in total (about 70 per turn), or at N of about 8, which the
card's memory does not allow.
