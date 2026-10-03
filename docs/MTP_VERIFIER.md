# MTP verifier: depth, exact multi-row kernels, draft costs

This follows [MTP_SPECULATIVE.md](MTP_SPECULATIVE.md). It covers:
- the speculative depth each artifact runs (2 for Q3, 3 for NVFP4);
- the exact multi-row kernels that make verification cheap per row;
- why Q3 artifacts carry an NVFP4 MTP layer, and the draft head's economics;
- the verifier's cost profile at long context.

The exact verifier is the authority: every configuration below has output hashes equal to ordinary greedy decode. Verification
never takes a relaxed-numerics route; the engine has no option for depth or numerics.

Setup:
- RTX 5070 Ti, 350 W, driver 615.71.09.
- Chat prompt corpus, 256 generated tokens; contexts are 64 / 1024 / 4096 prompt tokens, and 16K / 32K for the production-context
  figures.
- Gates use paired forks. A rate is per fork: committed decode tokens over decode time across the four prompts.
- The desktop's own GPU use (Xorg, Chrome's GPU process, GNOME) moves between about 470 and 610 MiB, which is why NVFP4 arms that
  host-back weights are sensitive to it.

## Results at production contexts

Production harnesses send about 10K tokens before any conversation history, so 16K and 32K are the primary figures. 1K and 4K are
short API calls and light chat. 64-token figures appear below only as synthetic or diagnostic data, never as a headline.

Decode tok/s, chat corpus v2, 256 generated tokens; one paired fork at 16K/32K, three at 1K/4K:

| Configuration | 1K | 4K | 16K | 32K |
|---|---|---|---|---|
| `q3`: NVFP4 MTP layer, MTP2, exact twins, draft twins | 100.6 * | 91.1 * | 81.6 | 64.1 |
| `nvfp4`: MTP3, exact twins, draft twins | 82.5 † * | 81.0 † * | 57.8 | 50.4 |

Notes:
- **Host-backed weights:** † 1 GiB at 1K/4K. 1.5 GiB everywhere else for NVFP4, because NVFP4 + MTP runs out of memory at 32K with
  1 GiB. The engine now plans this ([NVFP4_RESIDENCY.md](NVFP4_RESIDENCY.md)).
- **\* No draft twins:** measured before they existed. The draft twins matter at long context; at 1K/4K the draft step is 2-4 ms.
- **Exactness:** output hashes of every arm equal the non-speculative arm's for every prompt.

**Memory at long context:**
- The `q3` artifact at 48K tokens fits with 2.4 GiB free (43 tok/s, one prompt).
- NVFP4 + MTP needs 1.5 GiB host-backed at 32K (14.2 GiB peak, 1.4 GiB free). Every 128 MiB on the host costs about 2-2.5%.
- NVFP4 ordinary decode at 32K runs with 1 GiB host-backed (30.6 tok/s).

MTP adds the prompt's MTP catch-up: about 1.5 s at 32K on Q3.

## Speculative depth

Each extra verifier row costs about 3-4 ms, and acceptance keeps rising with depth but not enough to pay for it past the chosen
depth. NVFP4 acceptance by depth (1024-token prompt; verification time includes the host-backed weight stream):

| Depth | Tokens per verification | Verify ms |
|---|---|---|
| MTP1 | 1.85 | 26.1 |
| MTP2 | 2.48 | 27.1 |
| **MTP3** | 2.94 | 29.5 |
| MTP4 | 3.21 | 33.5 |
| MTP5 | 3.34 | 36.3 |

In-process screen after the twins (predicted tok/s = tokens per step / (verify + catch-up + recursion)); the Q3 rows use the NVFP4
MTP layer:

| Arm | Context | MTP2 | MTP3 | MTP4 | MTP5 |
|---|---|---|---|---|---|
| `q3` | 1024 | **100.2** | 98.4 | 96.8 | |
| `q3` | 4096 | **89.5** | 88.2 | 83.6 | |
| `nvfp4` (1 GiB host) | 1024 | 75.3 | **81.7** | 80.7 | 77.0 |
| `nvfp4` (1 GiB host) | 4096 | 69.2 | **74.3** | 73.5 | 71.8 |

**Depth policy.** Q3 runs MTP2 (it ties MTP3 at 1024 tokens; the smaller depth wins the tie) and NVFP4 runs MTP3
(`ArtifactProfile.speculativeDepth`).

**Rejected depths:**
- Q3 MTP3: 98.4 and 88.2 tok/s in the screen above, behind MTP2 at both contexts.
- NVFP4 MTP4 and MTP5: 80.7 / 77.0 at 1024 tokens and 73.5 / 71.8 at 4096, behind MTP3.
- MTP1 on Q3: a 2-row verification took longer than a 3-row one: the Q3 2-row twins are slower than the 3-row ones (gate/up
  8.45 ms at M=2 against 7.44 ms at M=3; codegen), and MTP1 is never the best depth.

## Screening method

Candidates are chosen with cheap targeted measurements; end-to-end gates run once, on the winners.
- **Acceptance** is deterministic for a fixed corpus: one pass of the four chat prompts settles it.
- **Timing** compares within one process: the model loads once and every requested depth runs, interleaved in 2 rounds. Comparing
  across processes varied by up to 16% (the same NVFP4 configuration measured 81.9 and 95.0 tok/s in two processes), mostly from
  host-backed transfer bandwidth.

## Exact multi-row attention and row norms

Row-exact verification runs decode attention (split-KV decode plus merge), q/k norm + RoPE and the residual RMS norm once over all
rows, every row computed exactly as the one-row launch at its position:
- **Attention:** the verified row is on `gridDim.y`.
  - Each row derives its own length, split count and block partition, and so its kernel choice
    (per-head below 2048 keys, GQA from 2048), exactly as the one-row dispatch.
  - The one-row kernels are wrappers of the same block functions: decode and merge compile to
    identical SASS, and GQA differs only in shared-memory base offsets.
- **Q/K norm + RoPE:** the one-row kernel already maps block -> (row, head) at position start + row.
- **Residual RMS norm:** the row-owned kernel runs with one CTA per row.
- **Decode scratch:** one area per sequence (instead of one per attention layer), with one one-row area per verified row. That
  costs about 25 MB less per sequence for ordinary decode.

**Exactness.**
- **Native test** (`test_attention_nvfp4.py`): every row is bitwise one-row decode for M = 2, 3, 4, 5 and 8, at starts 40, 253,
  1500, 2042 and 3000 (pages, partial pages, the 2048-key switch). A wrong split count fails it.
- **In-model:** `SpeculativeVerifyCudaIntegrationTest` and `SpeculativeDecodeCudaIntegrationTest` (MTP_SPECULATIVE §2).
- **Gate:** output hashes are identical to the non-speculative arm's (Q3 216 rows, NVFP4 197).

## Exact linear twins

Cold-weight operator timing: weight copies rotated past L2, bitwise equal to the one-row GEMV; times are multiples of the one-row
time on each model shape.

| Kernel | Registers | Blocks/SM | Time, multiple of one row (by shape) |
|---|---|---|---|
| one-row decode | 64 | 8 | 1 |
| `decode_rows3` | 159 | 3 | 0.96-1.15× |
| `decode_rows4` | 223 | 2 | 1.14-1.41× |
| `decode_rows5` | 220 | 2 | 1.16-1.45× |

**The M=4 cliff is occupancy.** The fully unrolled token loop reaches 223 registers (no spills), which drops residency from 3 to 2
blocks per SM.

**Synthetic whole-verifier linear model.** Cold weights, NVFP4 artifact shapes at M = 1-6, weighted by launches per verification:
- attention q/k and g/v ×32, attention out ×16;
- GDN qkv ×48, GDN z ×48, GDN out ×48;
- MLP gate/up ×64, MLP down ×64;
- LM head ×1.

The shipped exact linears cost 20.7 ms at M=1, 1.02-1.04× that at M=2, 1.06-1.07× at M=3, 1.28-1.31× at M=4 and 1.33× at M=5.

**Where the registers go.** The 4 × 32 decoded weight values per lane are kept live and reused across tokens: one table lookup per
weight serves every token row. That is what makes the twins cheap per row, and it is the M=4 cliff. Every exact variant that
frees those registers costs more than the lost occupancy.

**Rejected variants** (all bitwise, all cold; multiples of one row's time):
- **1 weight row per warp:** about 2× one row at M=4.
- **2 weight rows per warp:** 1.25-1.38×, no better than 4.
- **Register bounds:** `__launch_bounds__(128, 3)` 1.56-1.83× and `(128, 4)` 3.2-4.8×, from spills.
- **Packed BF16 activations, unpacked at each FMA:** the compiler generates the same 223-register kernel.
- **Token groups of 2 per weight pass:** 1.55× at M=3, 1.59× at M=4, 2.34× at M=5; each extra pass re-streams the weights (L1
  cannot hold them).
- **Token groups of 3 per weight pass:** 0.98× at M=3, 1.59× at M=4, 1.77× at M=5; the same, from M=4.
- **Token loop not unrolled, rotating accumulators:** 1.13× at M=3, 1.47× at M=4, 1.68× at M=5; still 188 registers at M=4.
- **Loop interchange and next-slice prefetch:** [MTP_SPECULATIVE.md](MTP_SPECULATIVE.md) section 6.

Exact M=3 linears already cost 1.14× one row in the Q3 profile, which is part of why Q3 prefers MTP2. At M=4 the NVFP4 twins
remain about 1.25× one row in the model.

## Why Q3 artifacts carry an NVFP4 MTP layer

A Q3 MTP layer accepted fewer drafts than the NVFP4 one. Controlled swap on the Q3 base, single pass, MTP2 at 1024 tokens, with the
MTP objects taken from the NVFP4 artifact:

| MTP layer | Mean accepted drafts | Tokens per verification |
|---|---|---|
| Q3 | 1.41 | 2.41 |
| NVFP4 attention pack only | 1.38 | 2.38 |
| NVFP4 whole MTP layer | **1.49** | **2.48** |
| NVFP4 layer + NVFP4 draft head | 1.44 | 2.44 |

**Answer.** The 3-bit MTP projections cause the gap, not the base model's quantization: with an NVFP4 MTP layer, Q3 accepts exactly
what NVFP4 accepts at the same depth (1.49). The NVFP4 layer also drafts faster: multi-row catch-up takes 1.4 against 2.0 ms at
1024 tokens and 2.4 against 3.75 ms at 4096, because the Q3 multi-row kernels are slow for those catch-up rows.

The `q3` artifact therefore stores the MTP layer in NVFP4 (about +50 MB of VRAM) and keeps its Q3 draft head (the NVFP4 draft head
row accepted slightly less). Speculative decode on it equals greedy bitwise.

## Draft-head economics

Every rejected draft is classified by whether the base token it should have been is in the shortlist
(`rejectionsInShortlist` / `rejectionsOutsideShortlist`). Outside the shortlist:
- Q3: 2.9% of rejections;
- NVFP4: 3.4-8.3% by context.

The MTP choosing a different in-shortlist token accounts for the rest.

Smaller shortlists, from per-verification shortlist ranks of accepted drafts (a draft beyond the shortlist could not have been
proposed):

| Shortlist rows | Head (NVFP4) | Q3 MTP2 accepted drafts | NVFP4 MTP3 accepted drafts |
|---|---|---|---|
| 131,072 (shipped) | 360 MiB | 1.410 | 1.963 |
| 98,304 | 270 MiB | -1.7% | -2.8% |
| 65,536 | 180 MiB | -4.2% | -6.9% |
| 32,768 | 90 MiB | -14% | -18% |

**Rejected:** a smaller shortlist loses more acceptance than the transfer it saves (90-180 MiB), and a larger head (the full
248,320-row vocabulary, 682 MiB) could only recover the outside-shortlist cases, at +0.4 ms per draft row and +322 MiB of VRAM.

## Residency sensitivity

Host-backed sensitivity in the in-process screen (NVFP4, exact MTP3, 1024 tokens): 87.4 tok/s with 640 MiB host-backed, 85.3 with
768 MiB and 81.7 with 1024 MiB. Each 128 MiB moved to the host costs about 2-2.5%. 640 MiB was the minimum on a quiet desktop;
with the desktop's GPU use (up to about 640 MiB) it failed in some runs. A shared decode scratch per sequence saves about 25 MB.

## Draft catch-up attention on the row twins

The long-context gate showed drafting growing from about 3 ms per step at 4K to 10-13 ms at 32K, because MTP catch-up quanta (2-4
rows, not row-exact) ran attention through the 32-row prefill tile, one CTA per head over the whole cache.

Draft quanta of at most 8 rows bring decode scratch, and the attention dispatch uses the row twins for any quantum of at most 8 rows
that provides scratch. Prefill brings no scratch and uses the prefill kernels. Drafts only propose tokens, so verifier exactness is
untouched. Attention time for r rows on the twins (real head geometry):

| Keys | 2 rows | 3 rows | 4 rows |
|---|---|---|---|
| 1K | 35 µs | 45 µs | 56 µs |
| 4K | 65 µs | 90 µs | 101 µs |
| 16K | 218 µs | 324 µs | 378 µs |
| 32K | 452 µs | 651 µs | 753 µs |

Draft time per step on the final configuration is 2.3 ms (Q3, 16K) and 3.0 ms (Q3, 32K). **Rejected:** the 32-row prefill tile
for draft quanta took 255-261 µs for 2-4 rows at 1K, 1.01-1.04 ms at 4K, 4.6 ms at 16K and 9.5 ms at 32K.

## Profile at 16K (Nsight Systems)

Exclusive kernel time per verification quantum (each kernel is charged from the later of its start and the previous kernel's end):

| Category | `q3` verify M=3 | `nvfp4` verify M=4 |
|---|---|---|
| Linears (all) | 17.53 ms | 25.91 ms |
| Attention proper | 6.09 | 8.22 |
| Norms + q/k norm/RoPE | 0.29 | 0.54 |
| GDN (recurrence, convolution, control) | 1.05 | 1.05 |
| GPU busy | 25.25 | 36.02 |
| Span | 27.69 | 49.13 |
| Idle inside span | 2.44 | 13.11 |
| H2D, overlapped | none | 1540 MiB in 45.9 ms |

Step anatomy:
- **Q3 MTP2:** median step 30.3 ms. Verification span 27.7 ms; catch-up about 2.5 ms; one recursive
  draft 0.87 ms; host gaps 0.31 + 0.24 + 0.19 ms.
- **NVFP4 MTP3:** median step 53.5 ms. Verification span 49.1 ms; catch-up 2.5 ms; two recursive drafts
  of 1.0 ms each; gaps about 0.3 ms each.

**Repeated stable regions.** The step is the same chain every time: VERIFY (one row-exact quantum), then
catch-up (1-4 rows), then MTP-1 recursive drafts. Each quantum's kernel sequence is fixed for a given
row count and context bucket.

**CPU/GPU gaps.**
- Between quanta, about 0.2-0.34 ms of host time sits on the critical path: 3-4 boundaries per step,
  about 0.8-1.2 ms, 3-4% of a Q3 step.
- Inside the Q3 verification span there is 2.4 ms of idle.
- Inside NVFP4 quanta, idle is dominated by waiting for the host-backed weight stream: 13-15 ms.

## Dependency map (one speculative step)

```
                       ┌───────────── host: acceptance (token IDs only), 0.3 ms ─────────────┐
VERIFY(rows P..P+d) ──▶ commit a+1 rows (KV frontier; GDN checkpoint/replay) ──▶ DRAFT catch-up(P..P+a)
   │  linears: M-row exact twins (weights once)              │                     │ needs verified hiddens
   │  attention: per-row split-KV twins (KV read per row)    │                     ▼
   │  GDN: one launch over all rows (sequential inside)      │               DRAFT(P+a+1) ─▶ … ─▶ DRAFT(P+a+d-1)
   │  NVFP4: 1.5 GiB host-weight stream overlaps the quantum │                     │
   └──────────────────────────────────────────────────────────┴─────────────────────▼
                                                                    VERIFY(next step)
```

**Independent work.**
- Verification needs every draft, and every draft needs the verified hiddens and the accepted count, so
  quanta stay a strict chain.
- Inside verification, rows are independent in linears, norms and attention, but causal through GDN.
- The prompt's MTP catch-up (independent of the next prefill chunk) is the only overlap MTP exposes; it
  costs TTFT, not decode.

## Findings

1. **Depth:** Q3 runs MTP2 (it wins or ties every context in the in-process screen, and is the smaller choice); NVFP4 runs MTP3.
2. **Verifier cost at 16K:** the `q3` verification keeps the GPU busy 25.25 ms (linears 17.53, attention 6.09) and the `nvfp4`
   verification 36.02 ms (linears 25.91, attention 8.22). Norms are flat in the row count.
3. **Exact verification against the M=1 bandwidth floor:** linears 1.06-1.07× at M=3 and 1.28-1.31× at M=4 (synthetic, cold
   weights). The M=4 step is register occupancy (223 registers hold the decoded weights that make the twins cheap), and every exact
   variant that frees them was slower.
4. **Why Q3 accepted less:** its 3-bit MTP projections, not the attention-pack re-quantization and not the base model. With the NVFP4
   MTP layer, Q3 accepts exactly as NVFP4 does at the same depth.
5. **The shortlist does not limit acceptance.** Only 1.5-8.3% of rejections have their base token outside it, and smaller heads
   lose more acceptance than they save.
6. **The dominant bottleneck:**
   - Q3: verification linears (58% of busy time), then verification attention, which grows with context (24% at 16K, about 30%
     at 32K).
   - NVFP4: the host-backed weight stream (1.5 GiB per quantum at about 35 GB/s) bounds every quantum at long context.
7. **Stable regions and gaps:** a fixed 3-4-quantum chain per step, 0.8-1.2 ms of host gaps per step, and 2.4 ms of idle inside the
   Q3 verification span.

## Rejected: native-numerics verification and drafting

- **Native FP4 verification** (every NVFP4 linear of a verifier quantum on the native skinny kernel, one row included). Its
  verification time is nearly flat in M: 31.3 ms at M=3 and 32.5 ms at M=6 (1024 tokens, 1 GiB host-backed), linears scaling 1.06×
  from M=1 to M=4, and it preferred depth 4. It ran 4.2% (1024) and 3.8% (4096) faster than exact MTP3 in the screen, but:
  - teacher-forced against ordinary NVFP4 decode over 384 positions: settled KL 3-6e-3 per 48-position window, top-1 agreement
    96.4%, hidden relative error about 5%, with no drift. This passes the relaxed policy bounds (KL < 1e-2, top-1 >= 95%) at
    roughly twice the relaxed-kernel calibration's KL (2.6e-3, top-1 97.6%);
  - all four corpus prompts' 256-token outputs differed from exact greedy, and native one-row decode ran 10% slower than the GEMV.
  Verification therefore stays row-exact.
- **Native FP4 drafting.** Every NVFP4 linear in draft quanta ran native: per-thread activation terms (one or two) and draft-only
  routing. With the exact verifier the outputs stayed bitwise equal to greedy. NVFP4 MTP3, one fork, chat corpus:

  | Draft numerics | Draft ms per step (64 / 1024 / 4096) | Mean accepted drafts (64 / 1024 / 4096) |
  |---|---|---|
  | GEMV for one row, the shipped route | 3.1 / 3.4 / 4.1 | 1.85 / 1.96 / 1.86 |
  | native two-term | 3.2 / 3.5 / 4.2 | 1.86 / 1.96 / 1.86 |
  | native one-term | 3.1 / 3.4 / 4.2 | 1.86 / 1.93 / (partial run) |

  Native one-row OMMA is not faster than the GEMV, and one-row drafting is dominated by the 360 MiB draft head read. The mechanism
  was removed: no gain, and a third numerics path to maintain. What paid was the draft attention path (row twins), +10-14% at
  16K/32K.

## Next

- **Long-context verifier attention** is the clearest kernel target. The verifier rows share split boundaries at long context, so
  an exact fused twin could stage each KV tile once for every row. At 32K the 3-row twins take 0.65 ms per layer, about 10 ms per
  verification.
- **CUDA Graphs** would only remove the 0.8-1.2 ms of per-step host gaps, about 3-4%.
- **Heterogeneous compute:** no target exposed.
