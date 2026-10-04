# Prefix cache

A request that starts with tokens an earlier request already processed prefills only what follows them. The engine
keeps the sequence state that prefill produced in a pinned host arena, finds it again by the token prefix it covers,
and copies it back into a fresh sequence before the first prefill quantum. Nothing in the HTTP API changes, and the
cache takes no device memory: the context length and the weight residency are those of an engine without it.

## Settings

| Property | Default | Meaning |
|---|---|---|
| `euhedral.inference.prefix-cache-bytes` | 4294967296 (4 GiB) | Pinned host memory for the cache. 0 turns it off. Memory that cannot be pinned turns it off with a warning, and the engine serves as before. |
| `euhedral.inference.prefix-cache-checkpoint-tokens` | 2048 | Prompt tokens between stored checkpoints. A multiple of 512, the prefill chunk. |

The arena is one allocation, 2 MiB aligned with `MADV_HUGEPAGE` and `MADV_COLLAPSE`, registered with `cudaHostRegister`:
the routine that allocates the host-backed weight arena ([NVFP4_RESIDENCY.md](NVFP4_RESIDENCY.md)), because per-tensor
pinning falls to 4 KiB pages.

## What is stored

A checkpoint is the complete state of a sequence at a token position `p`, and `p` is always a multiple of 512:

- the KV pages of the 16 full-attention layers (NVFP4, 18 KiB per token, 256-token pages);
- the GDN convolution and recurrent buffers of the 48 linear layers (146.8 MiB, whatever the position);
- for a prompt that speculates, the MTP layer's cache for rows `[0, p - 1)` and the base hidden row of position
  `p - 1` (below).

Checkpoints form a radix tree over token ids (`PrefixTree`). A node covers the tokens between its parent's position and
its own and owns only the KV pages of that span, so prompts that share a prefix store it once. A lookup walks down the
nodes whose tokens equal the prompt's, comparing the tokens themselves, and takes the deepest match that leaves at least
one prompt token to prefill: the logits of the last token have to be computed. Nodes are evicted least recently used
first, only when they have no children and no restore is reading them.

A prefill takes a checkpoint after every `prefix-cache-checkpoint-tokens`, and at the last chunk boundary before the
prompt ends (where the next chunk is the final one). The second is what lets a follow-up that repeats or extends the
prompt skip all but the last chunk, whatever the prompt's length.

## Why checkpoints sit on the chunk grid

Prefill state depends on how the prompt was split into chunks. The same 1536 tokens prefilled in chunks of other sizes
leave different bits in almost every buffer (`PrefillPartitionCudaIntegrationTest`, 192 buffers: every GDN buffer and
every full KV page; qwen3_8_27b_q3):

| Chunk size | Buffers that differ from the 512-token partition |
|---|---|
| 512 (repeated) | 0 |
| 1024 | 189 |
| 256 | 189 |
| 700 | 189 |
| 1000 | 191 |
| 1536 | 190 |

A sequence restored at a multiple of 512 prefills the rest in the chunks a cold run uses, so it ends with the state a
cold run has, buffer for buffer (`PrefixCacheCudaIntegrationTest`), and the tokens it samples are the cold run's. A
checkpoint at any other position would continue in other chunks. That rules out two checkpoints that look natural, the
state at the end of a prompt and the state at the end of a generation: decode state is not prefill state either. They are
not stored.

## Capture and restore

Both move state in frames of at most 16 MiB of copies each (`EuhedralInferenceRuntime.onWorker`), so no worker holds a
long copy, and both run between quanta: GDN state changes with every quantum, so a checkpoint is copied before the next
chunk is admitted. A capture reserves its bytes first (evicting for room), copies, then publishes; a copy that fails gives
the bytes back and the generation goes on. A restore allocates the sequence's state as a first quantum does, copies the
chain's KV pages and the last node's GDN state in, and publishes the position; the pages of a chain are copied
ancestor first.

A cancelled generation takes no checkpoint at its end, and one cancelled while restoring ends with the sequence released.

## Speculative prompts

A greedy, unconstrained prompt speculates with MTP, and the MTP layer has a KV cache of its own. After a prefill chunk
that ended at `p`, its catch-up has appended row `p - 1`, which pairs the base hidden of position `p - 1` with the token at
`p`, a token that belongs to whichever prompt resumes from there. A checkpoint therefore stores MTP rows below `p - 1` and
the hidden row, and a restore recomputes row `p - 1` with the resuming prompt's next token before it prefills. Rows
before `p - 1` do not depend on what follows.

A prompt that does not speculate (sampling, a constrained output) stores checkpoints without MTP state; a speculating
prompt restores through nodes that have it and stops at the first that does not. A span stored both ways keeps both
nodes.

## Measured

RTX 5070 Ti, i9-14900K, `qwen3_8_27b_nvfp4_compressed`, one worker, greedy with MTP (depth 3), 4 GiB cache, 2026-10-04.
One fork; its iterations agree within about 2%. Each row is the same prompt sent three times, cold (the cache holds
nothing of it) and then warm:

| Prompt tokens | Time to first token, cold | Time to first token, warm | Restored | Prefilled | Restore |
|---|---|---|---|---|---|
| 4096 | 1403 ms | 245-246 ms | 3584 | 512 | 10-12 ms |
| 16384 | 5430 ms | 597-599 ms | 15872 | 512 | 23 ms |
| 32000 | 12517 ms | 930-950 ms | 31744 | 256 | 40 ms |

The cold times are those of the cache-off run (4096: 1371 ms, 16384: 5430 ms, 32000: 12523 ms); the cache-on run's
first iteration of 4096 took 1404 ms. The scenarios of one run share their prompt text, so the cold iteration of the
16384 prompt restored the 4096 prompt's checkpoint at 4096 (4410 ms, 12288 tokens prefilled), and the 32000 prompt's
restored 16384 of its tokens (7564 ms): a prompt that shares only a prefix reuses it.

Capturing: a cold 16384-token prefill took 4970 ms without a cache and 5019 ms taking nine checkpoints (eight on the
interval and one at the last chunk boundary): about 5 ms each, GDN state included. From the layout, those nine
checkpoints are 9 x 147 MiB of GDN state and 18 KiB of KV per token, about 1.6 GiB of the arena (not read back from it).

At the default size the arena holds the GDN state of at most 27 checkpoints, so a deployment with many distinct long
prefixes evicts; the interval and the size are settings for that reason.

## Limits

- A prompt shorter than 512 tokens has no checkpoint, and one of 513 to 1024 tokens reuses at most its first 512.
- The GDN state of a checkpoint is 147 MiB regardless of the position, so a short span costs far more in GDN state than
  in KV. A smaller interval stores more checkpoints in the same arena at a lower reuse per checkpoint.
- Only the first prompt of a session uses the cache. The server opens a session per request.

## Rejected

- Checkpoints at the end of a prompt and of a generation, at any position: not exact (above).
- A hash of the token prefix as the key: the tree compares tokens, which is exact and needs no hash to collide.
- Keeping the last sequence on the device: the device holds the weights, one sequence's KV and its working set; a second
  sequence would come out of the context length or the weights' residency.
