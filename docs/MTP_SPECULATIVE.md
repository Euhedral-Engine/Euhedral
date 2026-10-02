# MTP speculative decode: results

Greedy MTP3 speculative decoding (docs/MTP_CONTRACT.md) on the RTX 5070 Ti (16 GB). It covers what
was built, how exactness is proven, and what it measured. Campaign of 2026-10-01 and 2026-10-02,
branch `perf/mtp-speculative`.

## 1. What runs

One speculative step is a chain of quanta on the existing runtime, with no central dispatcher and no
host loop over layers:

1. **VERIFY** over `[t₀, d₁, d₂, d₃]`. This is the decode topology with ALL_TOKENS logits and a device
   argmax per row; only the four token IDs cross to the host. Acceptance (`SpeculativeAcceptance`)
   resolves on retirement. The base KV commits `a + 1` rows (`AttentionKvState.commitSubmitted`).
   GDN checkpoints its state and records its per-row inputs for a replay.
2. **DRAFT catch-up**: the MTP layer over the committed outputs, seeded by the verified rows'
   post-final-norm hiddens. Its last row drafts d′₁.
3. **Two recursive DRAFT rows**, each seeded by the previous MTP hidden.

The MTP layer is a plan view (`QwenExecutionPlan.mtpDraft`):
- embedding, then `MTP_STEM` (the two norms and the `fc` projection);
- one full-attention layer with its own KV slot;
- `mtp.norm`, then the 131,072-row draft head.

The prompt prefills in chunks, each followed by its MTP catch-up. `InferenceTuning.speculativeDepth`
(the benchmark's `speculativeDepth`) enables it for greedy, unconstrained generations from a fresh
sequence.

**GDN rollback is ReplaySSM.** VERIFY checkpoints the recurrent and convolution state and records each
row's convolution inputs and α/β. On a partial accept, the next quantum restores the checkpoint and
replays the committed rows through the same row-exact kernels. Cost: about 157 MiB per sequence.
Measured per verification (1024-token prompt): 221 MiB of device-to-device copies, 0.47 ms of
copy-engine time, partly overlapping kernels.

## 2. Exactness

The oracle is ordinary one-row greedy decode, and the requirement is identical token IDs and
identical state, bitwise.

- **Verifier (row-exact).** Every row goes through the one-row arithmetic (MTP_CONTRACT §6):
  - attention, norms and the GDN recurrence run row by row;
  - linears use multi-row twins of the one-row GEMVs that repeat its FMA sequence per row.
- **Tests:**
  - `SpeculativeVerifyCudaIntegrationTest`: logits, GDN state and KV after a VERIFY quantum equal
    sequential decode, bitwise. NVFP4 at 2, 3, 4 and 8 rows over prompts of 300 to 2100 tokens
    (across the 2048-key attention kernel switch); Q3 at 4 rows; partial commits with replay.
  - `SpeculativeDecodeCudaIntegrationTest`: full MTP3 generation equals greedy decode in tokens and
    final state, for three prompts (a chat answer, 600 tokens of prose, a Java chat prompt). Passes
    on NVFP4 exact, on NVFP4 native numerics (against native greedy decode) and on compact Q3.
  - Native: `test_nvfp4.py` (decode_rows bitwise), `test_nvfp4_native.py` (skinny rows independent
    and deterministic), `test_q3_p2e2.py` and `test_q45_kernels.py` (Q3/Q4/Q5 twins bitwise).
- **Every benchmark row records `outputSha256`.** In every gate the speculative arms' hashes equal
  the oracle arm's for every prompt (§4).

## 3. Memory (NVFP4 + MTP3)

The NVFP4 artifact does not fit with MTP fully resident, so the minimum of base weights is
host-backed. MTP is never host-backed: `HostWeightSelection` only takes `text/layers/N/...`
projections.

| Item | MiB |
|---|---|
| Base layers + output head + embedding (artifact) | 14,275 |
| of which host-backed (54 GDN `query_key` tensors) | 641 |
| Resident base | 13,633 |
| MTP layer (resident) | 228 |
| Draft head (131,072 rows) + token ids (resident) | 360.5 |
| Speculative GDN (checkpoint, records, replay scratch) | about 157 |

Host-backed sizing: 640 MiB is the minimum that runs a 4096-token prompt. 512 MiB fails at 1024
tokens, even after MTP catch-up was chunked into 128-row pieces.

**Gate VRAM** (allocated peak during an iteration, and device memory free after it), at 64 / 4096
prompt tokens:

| Arm | Peak allocated GiB | Free after, MiB | H2D per output token |
|---|---|---|---|
| Q3 | 11.43 / 11.60 | 3269 / 3147 | 0 |
| Q3 + MTP3 | 12.04 / 12.25 | 2797 / 2635 | 0 |
| NVFP4 (all resident) | 14.13 / 14.31 | 711 / 582 | 0 |
| NVFP4, 640 MiB host-backed | 13.57 / 13.75 | 1355 / 1227 | 641 MiB |
| NVFP4 + MTP3, 640 MiB host-backed | 14.33 / 14.55 | 743 / 575 | 222 MiB (630 MiB per verification) |

Q3 + MTP3 needs no host-backed weights and leaves 2.6 GiB free at 4096 tokens.

## 4. Gates

Measurement setup:
- **Prompts:** `promptCorpus: "chat"`, four chat requests with thinking disabled. A 4096-token prompt
  quotes 3964-3970 tokens of a frozen document; a 64-token prompt is a standalone 29-36 token request.
- **Generation:** 256 tokens; warmup 1, then 4 iterations (one per prompt).
- **Forks:** 6 paired forks; each fork runs every arm in a fresh JVM, rotating the starting arm.
- **Rate:** per fork, committed decode tokens over decode time across the four prompts. The
  speculative decode window starts at first-token selection, so it includes the prompt's last MTP
  catch-up.
- **Exactness:** every speculative arm's `outputSha256` equals its oracle arm's for every prompt in
  every fork (Q3 = Q3 + MTP3; NVFP4 = NVFP4 host-backed = NVFP4 + MTP3; native = native + MTP3).

**Gate 1** (commit 2ca3555; Q3 verification still looped linears one row at a time):

| Arm (decode tok/s, median of 6 forks) | 64 | 4096 |
|---|---|---|
| Q3 | 59.5 | 56.4 |
| Q3 + MTP3, per-row Q3 linears | 42.5 | 41.9 |
| NVFP4 | 48.3 | 46.6 |
| NVFP4, 640 MiB host-backed | 43.7 | 42.2 |
| NVFP4 + MTP3 (exact) | **90.8** | **81.7** |
| NVFP4 native one-row decode | 43.5 | 42.1 |
| NVFP4 native + MTP3 | **92.6** | **82.5** |

Pairs, with forks ahead of 6:

| Pair | 64 | 4096 | Ahead |
|---|---|---|---|
| NVFP4 + MTP3 vs NVFP4 | 1.881× | 1.754× | 6/6 |
| NVFP4 + MTP3 vs NVFP4 640 MiB host-backed | 2.075× | 1.935× | 6/6 |
| native + MTP3 vs exact + MTP3 | 1.020× | 1.011× | 6/6 |
| native one-row vs GEMV one-row | 0.902× | 0.905× | 0/6 |
| Q3 + MTP3 (per-row) vs Q3 | 0.714× | 0.742× | 0/6 |

**Gate 2** (commit 8edc9f3: the Q3/Q4/Q5 multi-row twins):

| Arm | 64 | 4096 |
|---|---|---|
| Q3 | 59.5 | 56.4 |
| Q3 + MTP3 | **94.9** | **82.8** |
| NVFP4 + MTP3 | 90.7 | 81.7 |

| Pair | 64 | 4096 | Ahead |
|---|---|---|---|
| Q3 + MTP3 vs Q3 | 1.597× | 1.467× | 6/6 |
| NVFP4 + MTP3 vs Q3 | 1.523× | 1.448× | 6/6 |
| NVFP4 + MTP3 vs Q3 + MTP3 | 0.955× | 0.987× | 0/6 |

**Acceptance and step anatomy** (all forks; histogram = verifications that accepted 0/1/2/3 drafts):

| Arm | Context | Histogram | Mean accepted | Tokens per verification | Verifications per token | Verify ms | Draft ms | Verifications/s | TTFT ms |
|---|---|---|---|---|---|---|---|---|---|
| Q3 + MTP3 | 64 | 600/498/462/792 | 1.61 | 2.60 | 0.384 | 24.5 | 2.9 | 36.5 | 86 |
| Q3 + MTP3 | 4096 | 540/606/540/696 | 1.58 | 2.57 | 0.389 | 27.2 | 3.9 | 32.2 | 3854 |
| NVFP4 + MTP3 | 64 | 354/552/312/936 | 1.85 | 2.84 | 0.352 | 28.6 | 2.9 | 31.9 | 38 |
| NVFP4 + MTP3 | 4096 | 396/414/438/906 | 1.86 | 2.84 | 0.352 | 31.1 | 3.8 | 28.8 | 1865 |
| native + MTP3 | 64 | 360/510/372/912 | 1.85 | 2.84 | 0.352 | 27.7 | 3.0 | 32.6 | 38 |
| native + MTP3 | 4096 | 438/426/390/924 | 1.83 | 2.81 | 0.356 | 30.2 | 3.9 | 29.4 | 1869 |

Notes:
- **Draft time** is wall time between verifications: catch-up, two recursive rows and host gaps.
- **TTFT** for the ordinary arms: Q3 85 / 3743 ms, NVFP4 34 / 1813 ms. MTP adds the per-chunk
  catch-up: +1.6 to +4.5 ms at 64 tokens, +53 to +111 ms at 4096.
- **Q3 acceptance is lower** (1.6 accepted drafts against 1.85). Its MTP is Q3, with its attention
  re-quantized from W8 to NVFP4.
- **Commit and rollback:** base KV commit and MTP KV truncation only move frontiers. GDN checkpoints
  and records copy 221 MiB per verification (0.47 ms of copy engine, overlapping kernels). The
  replay of up to three committed rows is part of the next verification's time.

## 5. Profile and dependency map

Nsight Systems, 1024-token chat prompt. Times are per quantum, as exclusive kernel time: each kernel
is charged from the later of its start and the previous kernel's end.

**NVFP4: one exact M=4 verification against one decode token.** The two columns come from different
traces:
- **decode:** ordinary NVFP4 decode, all weights resident;
- **verification:** NVFP4 + MTP3 with 640 MiB of base weights host-backed.

Kernel times are comparable, since host-backed linears run the same kernels on staged copies. Span
is not: the verification span includes waiting for the host-backed weight stream.

| Category | Decode (M=1) | Verify (M=4) | Ratio |
|---|---|---|---|
| Linears (GEMV / `decode_rows4`) | 18.21 ms | 22.69 ms | 1.25× |
| Attention (row-exact, per row) | 0.58 | 2.01 | 3.5× |
| Norms (row-exact) | 0.23 | 0.87 | 3.7× |
| GDN (recurrence, convolution, control) | 1.31 | 1.03 | (lane overlap; not comparable) |
| GPU busy | 20.41 | 26.74 | 1.31× |
| Span (first kernel to last) | 21.14 | 29.62 | (mixed residency; see below) |

- The verification span exceeds its busy time by 2.9 ms. That gap is the 630 MiB host-backed weight
  stream: 15.4 ms of H2D at about 43 GB/s, overlapping compute.
- **Like-for-like cross-check.** The NVFP4 + MTP3 trace's own one-row decode quantum (the final commit
  at the budget, a single sample, same 640 MiB host-backed setup) has busy 19.53 ms and span 22.38 ms.
  The verification is 1.37× its busy time and 1.32× its span. The gated decode rates agree: an M=4
  verification costs 1.24-1.31× a host-backed decode step (§4).
- Native mode: M=4 skinny OMMA linears take 19.98 ms against 18.89 ms at M=1 (1.06×).
  Activation quantization (0.72 ms) and split-K finish (0.18 ms) come on top. Its M=1 linears are
  slower than the GEMV (19.75 against 18.21 ms), which is why native ordinary decode loses 10%.
- Q3: verification linears (Q3/Q4/Q5 twins) take 18.57 ms against 12.76 ms one-row (1.46×). GPU busy
  is 22.76 against 15.00 ms (1.52×).
- **Extra rows are not free.**
  - Linears: the first three extra rows cost 25% (NVFP4) or 46% (Q3).
  - Row-exact attention and norms scale with rows. At 4096 tokens, attention reads the KV cache once
    per row: verification grows 2.5 ms from 64 to 4096 tokens, one decode token only 0.8 ms.

**Step anatomy** (NVFP4 + MTP3, profiled step 33.0 ms):

| Part | Time |
|---|---|
| VERIFY span | 29.6 ms |
| Host gap | 0.28 ms |
| DRAFT catch-up (1 to 4 MTP rows, native skinny) | 1.4 ms |
| Host gap | 0.18 ms |
| DRAFT recursion ×2, each | 0.87 ms + 0.18 ms gap |

A one-row MTP draft costs about 0.74 ms of GEMV. Most of it is the 360 MiB draft head.

**Dependency map of one step:**

```
VERIFY(P..P+3) ─▶ accept a ─▶ DRAFT catch-up(P..P+a) ─▶ DRAFT(P+a+1) ─▶ DRAFT(P+a+2) ─▶ VERIFY(P+a+1..)
   │ base KV/GDN commit a+1 rows       │ needs the verified hiddens h_P..h_{P+a}
   └ GDN replay (a<3) runs in the next VERIFY's first frames
```

The chain is strictly serial:
- every draft needs the verified hidden states and the accepted count;
- the next verification needs all three drafts;
- inside a verification the four rows are independent only in the linears, and stay causal in
  attention and GDN.

The one independent piece of work MTP exposes is in the prompt. The catch-up for prefill chunk i
depends only on chunk i's hiddens, so it could overlap the base prefill of chunk i + 1. Today it runs
between them (+53 to +111 ms TTFT at 4096 tokens).

## 6. Rejected and open

**Rejected:**
- **`decode_rows` loop interchange.** Decode each weight value once, then FMA it into all M rows,
  with packed activations. Bitwise equal, but slower with cold weights: 1.30-1.53× one row's time,
  against 1.13-1.48× for the current kernel.
- **`decode_rows<M ≥ 4>` register prefetch of the next slice.** Bitwise equal. With cold weights,
  1.00-1.23× one row's time against 1.06-1.47×, but in-model NVFP4 + MTP3 was −1% in 5 of 6 forks
  at both contexts (verification 28.8 against 28.6 ms). Not kept.
- **Random-word benchmark prompts for MTP.** The model repeats itself and 31 of 34 steps accept all
  drafts, which overstated MTP3 at 119 tok/s. Replaced by the chat corpus.

**Open:**
- **The bottleneck is now verification itself.** Its linears' multi-row cost is 1.25× (NVFP4) and
  1.46× (Q3). Row-exact attention and norms grow linearly in rows, and attention with context.
  - Exact multi-row twins of decode attention (shared KV reads across the rows, per-row arithmetic
    unchanged) and of the residual norms are the next exact-mode candidates.
  - The Q3 twins are the larger linear lever.
- **Overlap the prompt's MTP catch-up** with the next prefill chunk (TTFT only).
- **MTP2 against MTP3:** depth is a tuning parameter (`speculativeDepth`); only depth 3 was gated.
