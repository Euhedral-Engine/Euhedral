# Multi-sequence execution

## Baseline (one sequence)

Measured 2026-10-14 by `Bf16BaselineCudaIntegrationTest` on the BF16 Swift-1.5 artifact
(`qwen3_8_27b_bf16.edrl`, Qwen3.8, 54 GB): one 16 GB RTX 5070 Ti, a 4096-token context, BF16 KV cache, the pinned
host budget set to 24 GiB (`memory.host-weights-limit`, fixed in the test: with the host's own budget
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

Prefill would take about 8% of wall time at N = 4 when it is not batched. The share reaches 20% when a
conversation generates fewer than about 340 tokens in total (about 70 per turn), or at N of about 12, which the
card's memory does not allow.

## BF16 passes

Measured 2026-10-15 on the BF16 Swift-1.5 artifact (54 GB) with the server (context 6144 tokens, 28 GiB of pinned host
memory for weights, `memory.host-weights-limit`, four worker processors), a 16 GB RTX 5070 Ti and an NVMe
artifact file. Scripted conversations (chat prompts of 100 to 700 tokens, at most 256 new tokens per request, greedy
with MTP speculation) were run through the Python client with `--parallel` equal to the server's sequence count. Rates
come from the server's pass log (`Execution.passes` at DEBUG) and the speculative counters: tokens per sequence per
verification is 1 plus the mean accepted drafts (2.6 here).

`max-sequences=0` resolves to 1 on this machine (more sequences move weights to the disk tier), so the count is set
explicitly.

| Sequences | LM head | Prefix cache | Disk per pass | Verify pass period | Segments per verify pass | Decode tok/s | Whole-run tok/s | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | device | off | 17850 MiB | 2.87 s | 3.93 | 3.7 | 3.4 | ran 9 min |
| 8 | device | off | 20570 MiB | 3.11 s | 7.84 | 6.7 | 5.2 | ran 10 min |
| 8 | host-mapped | off | 20570 MiB | 3.40 s | 7.65 | 5.9 | 4.7 | ran 10 min |
| 8 | device | 2 GiB | 20570 MiB | 3.12 s | 7.92 | 6.5 | 5.4 | ran 10 min |
| 12 | host-mapped | off | 23630 MiB | 3.88 s | 9.25 | 6.3 (not sustained) | | stalled after 1.5 min |

Decode tok/s is segments per verify pass times tokens per verification over the pass period; whole-run tok/s divides the
tokens of all verification passes by the run's wall time, so it includes the prefill passes at the start and the
ramp-down. Every pass reads the same bytes whatever its row count, so the pass period grows only with the disk bytes
(20570 MiB at 6.6 GB/s is 3.1 s). The period between two quanta that read the disk is the same for a prefill chunk
(3.19 s), a lone decode (3.09 s) and a verification (3.11 s): drafting and sampling between passes add at most 0.04 s
(about 1%) to a pass. The prefill share of wall time was 17% to 22% with these short conversations (one prefill chunk
costs a whole pass); mapping the LM head into host memory adds about 0.3 s to a pass and frees device memory only for
more than eight sequences.

Serving settings: 8 sequences, device LM head, 2 GiB prefix cache (`generation/server.env` in the store holds the
command line). Rejected: 12 sequences (host-mapped head), which stalled; no run longer than 10 minutes was made.

### Stalls found and fixed

A server with several sequences stalled (all GPU work idle, workers spinning in a CUDA call) within minutes. The device
waits on a window flag that a read raises, and only a worker that polls the read source reaps the read. The lattice
keeps one worker polling when it is quiet, so when that worker blocks inside a CUDA call (a launch into a full stream,
or a free of pinned memory, which synchronizes the device) nothing reaps the read the device waits for. Two mechanisms
were removed:

- Prompts admitted together each registered their own reads and their graphs submitted at once, filling the streams
  and blocking every worker. Quanta that read the artifact are now admitted one at a time, in admission order.
- Pinned upload and readback buffers (a closing session's logits buffer, KV page-table staging) were freed on a
  worker. Released buffers are kept by capacity for reuse until the memory closes.

With both in place eight sequences ran for 10 minutes three times; twelve sequences stalled with a worker blocked in a
launch. The cause that remains is the same: a worker blocked in a launch while the only polling worker is the one that
blocked.
