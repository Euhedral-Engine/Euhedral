# MTP verifier campaign: depth, exact multi-row kernels, native modes

This follows the MTP3 campaign (docs/MTP_SPECULATIVE.md). It covers:
- choosing the speculative depth;
- making exact verification cheaper per row;
- deciding the roles of the native-numerics verifier and of native drafting;
- explaining Q3's lower acceptance and the draft head's economics.

The exact verifier stays the authority: every exact-mode result below has output hashes equal to
ordinary greedy decode.

Setup:
- RTX 5070 Ti, 350 W, driver 615.71.09.
- Control: `043ab8c` (merged main after PR #46).
- Chat prompt corpus, 256 generated tokens; contexts are 64 / 1024 / 4096 prompt tokens.
- Gates use paired forks. A rate is per fork: committed decode tokens over decode time across the four
  prompts.

**Host notes.**
- The desktop's own GPU use (Xorg, Chrome's GPU process, GNOME) moves between about 470 and 610 MiB.
- NVFP4 + MTP with 640 MiB host-backed leaves only about 600 MiB of headroom: it ran out of memory in
  some runs at 1024 and 4096 tokens. NVFP4 speculative arms here use 768 MiB, then 1024 MiB, of
  host-backed base weights.
- A kind cluster on the same host added a load average of about 8 during one gate. It was stopped and
  that gate was rerun.

## Results at production contexts

Production harnesses send about 10K tokens before any conversation history, so 16K and 32K are the
primary comparison. 1K and 4K are short API calls and light chat. 64-token figures appear below only as
synthetic or diagnostic data, never as a headline.

**Final configuration against the pre-campaign baseline** (decode tok/s, chat corpus v2, 256 generated
tokens; one paired fork at 16K/32K, three at 1K/4K):

| Configuration | 1K | 4K | 16K | 32K |
|---|---|---|---|---|
| Q3 + MTP3 (`main`, baseline) | 90.5 | 76.1 | 62.8 | 46.5 |
| **Q3 final**: NVFP4 MTP layer artifact, MTP2, exact twins, draft twins | 100.6 * | 91.1 * | **81.6 (+30%)** | **64.1 (+38%)** |
| NVFP4 + MTP3 exact (`main`, baseline) | 77.4 † | 77.5 † | 44.3 | 38.3 |
| **NVFP4 final**: MTP3, exact twins, draft twins | 82.5 † * | 81.0 † * | **57.8 (+30%)** | **50.4 (+32%)** |
| NVFP4 native MTP4 (relaxed numerics) ‡ | 86.3 † * | 78.9 † * | 60.5 | 46.6 |

Notes:
- **Host-backed weights:** † 1 GiB at 1K/4K. 1.5 GiB everywhere else for NVFP4, because NVFP4 + MTP
  runs out of memory at 32K with 1 GiB.
- **\* No draft twins:** measured before they existed (`fe4877a`). The draft twins matter at long context;
  at 1K/4K the draft step is 2-4 ms.
- **‡ Native:** measured on the build before the draft twins.
- **Gains at 1K/4K before the draft twins:** Q3 +11.3% / +19.3% (3/3 forks), NVFP4 exact +5.6% (3/3)
  / +5.2% (5/6).
- **Exactness:** output hashes of every final exact arm equal the baseline's for every prompt.

**Memory at long context:**
- Q3 final at 48K tokens fits with 2.4 GiB free (43 tok/s, one prompt).
- NVFP4 + MTP needs 1.5 GiB host-backed at 32K (14.2 GiB peak, 1.4 GiB free). Every 128 MiB on the host
  costs about 2-2.5%.
- NVFP4 ordinary decode at 32K runs with 1 GiB host-backed (30.6 tok/s).

**Prefill (TTFT):**

| Context | Q3 | NVFP4 |
|---|---|---|
| 16K | about 17.5 s | about 9.7 s |
| 32K | about 40 s | about 24 s |

MTP adds the prompt's MTP catch-up: about 1.5 s at 32K on Q3.

## Phase 0: control reproduced

Correctness suites on `043ab8c`: core CUDA 60/60, api CUDA 4/4, core 283, lattice 39, benchmark 42,
api 81, native all OK.

The sweep below reproduces the previous campaign's rows:

| Arm | 64 | 4096 |
|---|---|---|
| Q3 | 59.6 | 56.5 |
| Q3 + MTP3 | 94.9 | 82.7 |
| NVFP4 | 48.3 | 46.6 |
| NVFP4 + MTP3 exact | 90.8 | 81.7 |
| NVFP4 + MTP3 native | 92.7 | 82.4 |

## Phase 1: speculative depth

Measured before the realistic-context policy: the 64-token column is diagnostic only. Control build, 3 forks (NVFP4 at 1024/4096: 2 valid forks, fork 3 ran out of memory at 640 MiB),
decode tok/s:

| Arm | 64 | 1024 | 4096 | Tokens per verification (1024) | Verify ms (1024) |
|---|---|---|---|---|---|
| Q3 | 59.6 | 58.2 | 56.5 | | |
| Q3 MTP1 | 75.2 | 74.2 | 68.6 | 1.82 | 22.6 |
| **Q3 MTP2** | **99.2** | **99.4** | **88.2** | 2.41 | 21.7 |
| Q3 MTP3 | 94.9 | 97.7 | 82.7 | 2.83 | 25.7 |
| Q3 MTP4 | 88.0 | 91.4 | 77.1 | 3.02 | 28.7 |
| Q3 MTP5 | 79.6 | 83.4 | 69.8 | 3.11 | 32.1 |
| NVFP4 | 48.3 | 47.6 | 46.6 | | |
| NVFP4 MTP1 | 69.0 | 67.7 | 62.8 | 1.85 | 26.1 |
| NVFP4 MTP2 | 85.9 | 84.8 | 78.7 | 2.48 | 27.1 |
| **NVFP4 MTP3** | **90.8** | **90.2** | **81.7** | 2.94 | 29.5 |
| NVFP4 MTP4 | 85.5 | 85.7 | 78.3 | 3.21 | 33.5 |
| NVFP4 MTP5 | 84.2 | 81.2 | 75.4 | 3.34 | 36.3 |
| native MTP3 | 92.7 | 90.0 | 82.4 | 2.87 | 28.6 |
| **native MTP4** | **94.6** | **92.7** | **83.5** | 3.14 | 29.7 |

- **Pairs, forks ahead:**
  - Q3 MTP3 against MTP2 is 0.958 / 0.983 / 0.938×, 0 of 3 forks.
  - NVFP4 MTP3 against MTP2 is 1.056 / 1.063 / 1.038×, every fork.
  - NVFP4 MTP4 against MTP3 is 0.94-0.96×, 0 forks.
- **Each extra verifier row costs about 3-4 ms.** Acceptance keeps rising with depth (NVFP4 MTP5
  commits 3.3 tokens per verification), but not enough to pay for it.
- **Native prefers depth 4:** its per-row cost is lower (MTP4 verification 29.7 ms, against 33.5 ms exact).
- **Q3 anomaly:** an MTP1 verification (2 rows) took longer than an MTP2 one (3 rows). The 2-row Q3 twins
  are slower than the 3-row ones (Phase 2).

## Phase 2: exact verifier cost by operator

Method:
- Nsight Systems, 1024-token chat prompt, control build.
- Times are per quantum: exclusive kernel time per category (each kernel is charged from the later of
  its start and the previous kernel's end), GPU busy, and span (first kernel to last).
- Copies overlap kernels and are listed separately.
- NVFP4 M=1 is ordinary decode with the same 768 MiB host-backed weights as the verifier.
- `spec_categorize.py` in the skill's project folder.

| Category | NVFP4 M=1 | NVFP4 M=4 (MTP3) | Ratio | Q3 M=1 | Q3 M=3 (MTP2) | Ratio |
|---|---|---|---|---|---|---|
| Linear MLP gate/up | 8.37 ms | 9.74 ms | 1.16× | 6.59 ms | 7.44 ms | 1.13× |
| Linear o-proj, GDN out, MLP down | 5.94 | 7.42 | 1.25× | 4.90 | 5.79 | 1.18× |
| Linear GDN input projections | 1.99 | 3.29 | 1.65× | 2.06 | 2.26 | 1.10× |
| Linear attention q/k/v | 0.77 | 1.02 | 1.33× | 0.67 | 0.70 | 1.05× |
| Linear LM head | 0.86 | 0.98 | 1.14× | 0.61 | 0.73 | 1.19× |
| **All linears** | **17.92** | **22.45** | **1.25×** | **14.83** | **16.92** | **1.14×** |
| Attention proper (decode + merge) | 0.39 | 1.59 | 4.05× | 0.43 | 1.32 | 3.1× |
| Q/K norm + RoPE | 0.13 | 0.27 | 2.1× | 0.10 | 0.26 | 2.5× |
| RMS / residual norm | 0.24 | 0.85 | 3.6× | 0.26 | 0.70 | 2.7× |
| GDN recurrence + convolution | 0.41 | 0.70 | 1.7× | 0.41 | 0.69 | 1.7× |
| Argmax | 0.015 | 0.064 | 4.3× | 0.016 | 0.049 | 3.1× |
| GPU busy | 20.00 | 26.40 | 1.32× | 16.85 | 20.56 | 1.22× |
| Wall span | 23.23 | 29.16 | 1.26× | 18.76 | 24.00 | 1.28× |
| Idle inside span | 3.23 | 2.76 | | 1.91 | 3.45 | |
| H2D (overlapped) | 776 MiB, 21.1 ms | 765 MiB, 19.3 ms | | none | none | |
| D2D GDN checkpoints (overlapped) | | 221 MiB, 0.47 ms | | | 208 MiB, 0.46 ms | |

Q3's linear families are Q3/Q4/Q5 kernels, but their shapes are the same families. GDN control is
not shown: lane overlap makes its exclusive attribution unreliable.

**Bottleneck model.**
- **NVFP4.** M=4 linears add 4.5 ms; non-linear work (attention, norms, argmax, GDN) adds about 2.0 ms.
  - With 768 MiB host-backed, ordinary decode is nearly transfer-bound: 21.1 ms of H2D against 20.0 ms
    busy.
- **Q3.** M=3 linears add 2.1 ms and non-linear work 1.5 ms.
  - Q3 verification also has 1.5 ms more idle inside its span than decode, with no host stream: host
    time for the per-row launches.

**The MTP1 anomaly.** The Q3 2-row twins are slower than the 3-row ones: gate/up 8.45 ms at M=2 against
7.44 ms at M=3, and down/out 7.13 against 5.79 ms. Codegen; only MTP1 uses M=2 and it is never the best
depth.

Step anatomy:
- Q3 MTP2 step: median 26.7 ms. Verification span 24.0 ms; draft catch-up span 3.9 ms (busy 2.0), one
  recursive draft 0.77 ms; host gaps 0.34 + 0.22 + 0.24 ms.
- NVFP4 MTP3 step: median 32.0 ms. Verification span 29.2 ms; drafts 1.32 + 2 × 0.89 ms; gaps about
  0.3 ms after verification and 0.18 ms after each draft.

## Phases 3-4: exact multi-row attention and row norms

**Change** (`427e6b6`). Row-exact verification ran decode attention (split-KV decode plus merge), q/k
norm + RoPE and the residual RMS norm once per verified row. Each now runs once over all rows, every row
computed exactly as the one-row launch at its position:
- **Attention:** the verified row is on `gridDim.y`.
  - Each row derives its own length, split count and block partition, and so its kernel choice
    (per-head below 2048 keys, GQA from 2048), exactly as the one-row dispatch.
  - The one-row kernels became wrappers of the same block functions: decode and merge compile to
    identical SASS, and GQA differs only in shared-memory base offsets.
- **Q/K norm + RoPE:** the one-row kernel already maps block -> (row, head) at position start + row.
- **Residual RMS norm:** the row-owned kernel runs with one CTA per row.
- **Decode scratch:** now one area per sequence (instead of one per attention layer), with one one-row
  area per verified row. That saves about 25 MB per sequence for ordinary decode.

**Exactness.**
- **Native test:** every row is bitwise one-row decode for M = 2, 3, 4, 5 and 8, at starts 40, 253,
  1500, 2042 and 3000 (pages, partial pages, the 2048-key switch). A wrong split count fails it.
- **In-model:** `SpeculativeVerifyCudaIntegrationTest` passes on Q3 for M = 2, 3, 4, 5 and 8 with 300-
  and 2045-token prompts, and on NVFP4 at M=4 with a 2045-token prompt. The decode test passes on both
  artifacts.
- **Gate:** output hashes are identical across control and candidates (Q3 216 rows, NVFP4 197).

**Gate** (6 paired forks, after the cluster was stopped):

| Q3 MTP2 (M=3) | 64 | 1024 | 4096 | Forks ahead |
|---|---|---|---|---|
| Attention twin alone | +2.0% | +2.2% | +3.8% | 6/6 |
| All twins | +4.5% | +4.7% | +5.3% | 6/6 |
| Q/K + norm twins on top of attention | +2.8% | +2.3% | +2.4% | 6, 6, 5/6 |

NVFP4 MTP3 at 768 MiB host-backed: all twins +4.8 / +5.0 / +6.3% (5/6, 4/6, 3/5 forks ahead). Runs that
ran out of memory as desktop GPU use varied left this gate inconclusive; it is re-gated at 1 GiB in the
final gates.

## Phase 5: exact linear twins

Cold-weight operator timing: weight copies rotated past L2, all candidates bitwise equal to
`decode_rows<M>`; ratios are to the one-row GEMV on each model shape.

| Kernel | Registers | Blocks/SM | Time, multiple of one row (by shape) |
|---|---|---|---|
| one-row decode | 64 | 8 | 1 |
| `decode_rows3` | 159 | 3 | 0.96-1.15× |
| `decode_rows4` | 223 | 2 | 1.14-1.41× |
| `decode_rows5` | 220 | 2 | 1.16-1.45× |

**The M=4 cliff is occupancy.** The fully unrolled token loop reaches 223 registers (no spills), which
drops residency from 3 to 2 blocks per SM.

Rejected variants (all bitwise, all cold):
- **1 weight row per warp:** about 2× one row at M=4.
- **2 weight rows per warp:** 1.25-1.38×, no better than 4.
- **Register bounds:** `__launch_bounds__(128, 3)` 1.56-1.83× and `(128, 4)` 3.2-4.8×, from spills.
- **Packed BF16 activations, unpacked at each FMA:** the compiler generates the same 223-register kernel.
- **Previous campaign, not repeated:** loop interchange; next-slice prefetch (cold win, in-model loss).

**Synthetic whole-verifier linear model.** Cold weights, NVFP4 artifact shapes at M = 1-6, weighted by
launches per verification:
- attention q/k and g/v ×32, attention out ×16;
- GDN qkv ×48, GDN z ×48, GDN out ×48;
- MLP gate/up ×64, MLP down ×64;
- LM head ×1.

Production exact linears cost 20.7 ms at M=1, 1.02-1.04× at M=2, 1.06-1.07× at M=3, 1.28-1.31× at
M=4 and 1.33× at M=5. Two more exact variants, rejected on the model alone:

| Variant | M=3 | M=4 | M=5 | Why |
|---|---|---|---|---|
| Token groups of 2 per weight pass | 1.55× | 1.59× | 2.34× | each extra pass re-streams the weights (L1 cannot hold them) |
| Token groups of 3 per weight pass | 0.98× | 1.59× | 1.77× | the same, from M=4 |
| Token loop not unrolled, rotating accumulators | 1.13× | 1.47× | 1.68× | still 188 registers at M=4 |

**Where the registers go.** The 4 × 32 decoded weight values per lane are kept live and reused across
tokens: one table lookup per weight serves every token row. That is what makes the twins cheap per
row, and it is the M=4 cliff. Every exact variant that frees those registers costs more than the lost
occupancy.

Exact M=3 linears already cost 1.14× one row in the Q3 profile, which is part of why Q3 prefers MTP2.
At M=4 the NVFP4 twins remain about 1.25× one row in the model.

## Screening method (from Phase 6 on)

Candidates are chosen with cheap targeted measurements; end-to-end gates run once, on the winners.
- **Acceptance** is deterministic for a fixed corpus: one pass of the four chat prompts settles it.
- **Timing** compares within one process. `SpeculativeDepthScreenCudaIntegrationTest`
  (`-Peuhedral.speculative.screen=true`) loads the model once and runs every requested depth, exact and
  native, interleaved in 2 rounds. Comparing across processes varied by up to 16% (the same NVFP4
  configuration measured 81.9 and 95.0 tok/s in two processes), mostly from host-backed transfer
  bandwidth.

**In-process screen after the twins** (predicted tok/s = tokens per step / (verify + catch-up +
recursion)):

| Arm | Context | MTP2 | MTP3 | MTP4 | MTP5 |
|---|---|---|---|---|---|
| Q3 | 1024 | **99.3** | 98.5 | 92.1 | |
| Q3 | 4096 | **81.6** | 78.3 | 73.9 | |
| Q3 + NVFP4 MTP layer | 1024 | **100.2** | 98.4 | 96.8 | |
| Q3 + NVFP4 MTP layer | 4096 | **89.5** | 88.2 | 83.6 | |
| NVFP4 exact (1 GiB host) | 1024 | 75.3 | **81.7** | 80.7 | 77.0 |
| NVFP4 exact (1 GiB host) | 4096 | 69.2 | **74.3** | 73.5 | 71.8 |
| NVFP4 native (1 GiB host) | 1024 | 72.3 | 80.9 | **85.1** | 85.8 |
| NVFP4 native (1 GiB host) | 4096 | 66.7 | 74.8 | **77.1** | 77.6 |

**Depth answers.** Q3 runs MTP2 (it ties MTP3 at 1024 tokens; the smaller depth wins the tie), NVFP4
exact runs MTP3, and NVFP4 native runs MTP4 (it ties MTP5).

## Phase 6: native verifier

- **Verification time is nearly flat in M.** Native verification costs 31.3 ms at M=3 and 32.5 ms at
  M=6 (1024 tokens, 1 GiB host-backed), against exact 30.4 to 37.6 ms.
  - Its linears scale at 1.06× from M=1 to M=4.
  - With 1 GiB host-backed, every quantum also streams about 1 GiB over PCIe, which sets a floor near
    the native verification time.
- **Speed:** native MTP4 is +4.2% (1024) and +3.8% (4096) over exact MTP3 in the screen, and +4.3% in
  a single benchmark pass.
- **Numerics:** teacher-forced against ordinary NVFP4 decode over 384 positions
  (`RelaxedNumericsDriftCudaIntegrationTest`, `candidate=native`):
  - settled KL 3-6e-3 per 48-position window, top-1 agreement 96.4%, hidden relative error about 5%;
  - no drift: the late half is not above the early half;
  - this passes the relaxed policy bounds (KL < 1e-2, top-1 >= 95%), but at roughly twice the
    relaxed-kernel calibration's KL (2.6e-3, top-1 97.6%).
- **Generated output:** all four corpus prompts' 256-token outputs differ from exact greedy. Native
  MTP3 and MTP4 outputs are identical to each other: the mode is deterministic against its own oracle.
- **Decision: optional relaxed fast mode, not the default.** The 4% gain changes every generated text,
  and native one-row decode is 10% slower than the GEMV.

## Phase 7: native drafting (rejected)

Every NVFP4 linear in draft quanta ran native: per-thread activation terms (one or two) and draft-only
routing. With the exact verifier the outputs stayed bitwise equal to greedy (decode test).

Measured on the chat corpus (NVFP4 MTP3, one fork):

| Draft numerics | Draft ms per step (64 / 1024 / 4096) | Mean accepted drafts (64 / 1024 / 4096) |
|---|---|---|
| default (GEMV for one row) | 3.1 / 3.4 / 4.1 | 1.85 / 1.96 / 1.86 |
| native two-term | 3.2 / 3.5 / 4.2 | 1.86 / 1.96 / 1.86 |
| native one-term | 3.1 / 3.4 / 4.2 | 1.86 / 1.93 / (partial run) |

Native one-row OMMA is not faster than the GEMV, and one-row drafting is dominated by the 360 MiB draft
head read. The mechanism was removed: no gain, and a third numerics path to maintain.

## Phase 8: the Q3 acceptance gap

The compact Q3 artifact quantizes the MTP layer to Q3, with W8 for the attention pack (re-quantized to
NVFP4 at load).

Controlled swap on the Q3 base, single pass, MTP2 at 1024 tokens, with the MTP objects taken from the
NVFP4 artifact:

| MTP layer | Mean accepted drafts | Tokens per verification |
|---|---|---|
| Q3 (current) | 1.41 | 2.41 |
| NVFP4 attention pack only | 1.38 | 2.38 |
| NVFP4 whole MTP layer | **1.49** | **2.48** |
| NVFP4 layer + NVFP4 draft head | 1.44 | 2.44 |
| (NVFP4 artifact's own MTP2) | 1.49 | 2.48 |

**Answer.** The W8 re-quantization is not the cause. The 3-bit MTP projections are: with an NVFP4 MTP
layer, Q3 accepts exactly what NVFP4 accepts at the same depth. The base model's quantization does not
matter for acceptance here.

The NVFP4 layer also drafts faster: multi-row catch-up takes 1.4 against 2.0 ms at 1024 tokens and 2.4
against 3.75 ms at 4096. The Q3 multi-row kernels are slow for those catch-up rows.

**Implementation.**
- The converter's `--mtp-format nvfp4` (`9a75096`) produced `qwen3_5_27b_compact_q3_nvmtp.edrl`:
  - its 1106 non-MTP objects are byte-identical to the compact Q3 artifact's;
  - its 12 MTP objects are byte-identical to the NVFP4 artifact's.
- Cost: about +50 MB of VRAM.
- Speculative decode on it equals greedy bitwise.

## Phase 9: draft-head economics

Every rejected draft is classified by whether the base token it should have been is in the shortlist
(`rejectionsInShortlist` / `rejectionsOutsideShortlist`). Outside the shortlist:
- Q3: 2.9% of rejections;
- NVFP4: 3.4-8.3% by context.

The MTP choosing a different in-shortlist token accounts for the rest.

Smaller shortlists, from per-verification shortlist ranks of accepted drafts (a draft beyond the
shortlist could not have been proposed):

| Shortlist rows | Head (NVFP4) | Q3 MTP2 accepted drafts | NVFP4 MTP3 accepted drafts |
|---|---|---|---|
| 131,072 (current) | 360 MiB | 1.410 | 1.963 |
| 98,304 | 270 MiB | -1.7% | -2.8% |
| 65,536 | 180 MiB | -4.2% | -6.9% |
| 32,768 | 90 MiB | -14% | -18% |

A larger head (the full 248,320-row vocabulary, 682 MiB) could only recover the outside cases, at
+0.4 ms per draft row and +322 MiB of VRAM. Neither direction pays; the 131,072-row head stays.

## Phase 10: residency thresholds

Host-backed sensitivity in the in-process screen (1024 tokens):

| Host-backed | Exact MTP3 | Native MTP4 |
|---|---|---|
| 640 MiB | 87.4 | 90.4 |
| 768 MiB | 85.3 | 88.9 |
| 1024 MiB | 81.7 | 85.1 |

Each 128 MiB moved to the host costs about 2-2.5%.

Small memory changes available in this campaign:
- one shared decode scratch per sequence: about 25 MB;
- a smaller shortlist: 90-180 MiB, which loses more acceptance than the transfer it saves.

None removes the host stream, so no threshold is crossed. 640 MiB was the minimum on a quiet desktop.
With the desktop's current GPU use (up to about 640 MiB) it failed in some runs, so the final NVFP4
gates use 1 GiB.

## Draft catch-up attention on the row twins (`5d2da8d`)

The long-context gate showed drafting growing from about 3 ms per step at 4K to 10-13 ms at 32K. MTP
catch-up quanta (2-4 rows, not row-exact) ran attention through the 32-row prefill tile, which is one
CTA per head over the whole cache.

**Synthetic** (real head geometry, attention time for r rows):

| Keys | prefill32, 2-4 rows | Twins, 2 rows | Twins, 3 rows | Twins, 4 rows |
|---|---|---|---|---|
| 1K | 255-261 µs | 35 | 45 | 56 µs |
| 4K | 1.01-1.04 ms | 65 | 90 | 101 µs |
| 16K | 4.6 ms | 218 | 324 | 378 µs |
| 32K | 9.5 ms | 452 | 651 | 753 µs |

**Change.** Draft quanta of at most 8 rows bring decode scratch, and the attention dispatch uses the
twins for any quantum of at most 8 rows that provides scratch. Prefill brings no scratch and is
unchanged. Drafts only propose tokens, so verifier exactness is untouched (decode test on both
artifacts; acceptance unchanged within noise).

**In-model check** (one process, interleaved, Q3 + NVFP4 MTP layer, MTP2):

| Context | Catch-up per step | Predicted |
|---|---|---|
| 16K | 8.66 -> 5.07 ms | +10% |
| 32K | 20.8 -> 13.7 ms | +14% |

Both include the prompt's own catch-up, amortized over the steps. In the final gate (above), draft time
per step fell from 7.0 to 2.3 ms (Q3, 16K) and from 11.4 to 3.0 ms (Q3, 32K).

## Final profile at 16K (Nsight Systems, final build)

Exclusive kernel time per quantum, against each configuration's own one-row decode:

| Category | Q3 M=1 | Q3 verify M=3 | Ratio | NVFP4 M=1 | NVFP4 verify M=4 | Ratio |
|---|---|---|---|---|---|---|
| Linears (all) | 14.69 ms | 17.53 ms | 1.19× | 21.94 ms | 25.91 ms | 1.18× |
| Attention proper | 2.83 | 6.09 | 2.15× | 3.35 | 8.22 | 2.45× |
| Norms + q/k norm/RoPE | 0.34 | 0.29 | | 0.42 | 0.54 | |
| GDN (recurrence, convolution, control) | 1.05 | 1.05 | | 1.14 | 1.05 | |
| GPU busy | 19.14 | 25.25 | 1.32× | 26.19 | 36.02 | 1.38× |
| Span | 20.90 | 27.69 | 1.32× | 40.70 | 49.13 | 1.21× |
| Idle inside span | 1.76 | 2.44 | | 14.51 | 13.11 | |
| H2D, overlapped | none | none | | about 1.5 GiB | 1540 MiB in 45.9 ms | |

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
- Inside the verification span, Q3 has 2.4 ms of idle against 1.8 ms for decode.
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

## Answers

1. **Q3 depth:** MTP2. It wins or ties every context in the in-process screen and is the smaller choice.
2. **NVFP4 depth:** MTP3 exact. Native (relaxed) prefers MTP4.
3. **Remaining M-row verifier penalty at 16K:**
   - Q3: busy 1.32× one row (linears 1.19×, attention 2.15×).
   - NVFP4: 1.38× one row (linears 1.18×, attention 2.45×).
   - Norms are now flat, against 3.6× before the twins.
4. **Exact verification against the M=1 bandwidth floor:** linears 1.06-1.07× at M=3 and 1.28-1.31× at
   M=4 (synthetic, cold weights). The M=4 step is register occupancy (223 registers hold the decoded
   weights that make the twins cheap), and every exact variant that frees them was slower.
5. **Native verification:**
   - +4% at 1K-4K over exact (MTP4 against MTP3).
   - At 16K/32K +7% / +2.5%, against the exact build without draft twins.
   - With 1.5 GiB host-backed, NVFP4 quanta are transfer-bound, which caps native's linear advantage.
6. **Native as a production fast mode:** optional only. It changes every generated text: top-1 96.4%,
   KL 3-6e-3, no drift, within the relaxed policy but near its top-1 bound. Not the default.
7. **One-term or aggressive native drafting:** no. One-row native OMMA is not faster than the GEMV, and
   acceptance was unchanged. What paid was the draft attention path (row twins), +10-14% at 16K/32K.
8. **Why Q3 accepted less:** its 3-bit MTP projections, not the W8 attention re-quantization and not the
   base model. With the NVFP4 MTP layer, Q3 accepts exactly as NVFP4 does at the same depth, and its
   catch-up is faster (`--mtp-format nvfp4`).
9. **Is the shortlist limiting acceptance?** No. Only 1.5-8.3% of rejections have their base token
   outside it, and smaller heads lose more acceptance than they save.
10. **Small memory reductions:** none removes the NVFP4 host stream (a shared decode scratch saves about
    25 MB per sequence). NVFP4 + MTP needs 1-1.5 GiB host-backed on this desktop: 1.5 GiB at 32K.
11. **New dominant bottleneck:**
    - Q3: verification linears (58% of busy time), then verification attention, which grows with
      context (24% at 16K, about 30% at 32K).
    - NVFP4: the host-backed weight stream (1.5 GiB per quantum at about 35 GB/s) bounds every quantum
      at long context.
12. **Stable regions and gaps:** see the step anatomy above. A fixed 3-4-quantum chain per step, 0.8-1.2
    ms of host gaps per step, and 2.4 ms of idle inside the Q3 verification span.

**Next campaign.**
- **Long-context verifier attention** is the clearest kernel target. The verifier rows share split
  boundaries at long context, so an exact fused twin could stage each KV tile once for every row. At 32K
  the 3-row twins take 0.65 ms per layer, about 10 ms per verification.
- **CUDA Graphs** would only remove the 0.8-1.2 ms of per-step host gaps, about 3-4%.
- **NVFP4 residency or compression** is the lever for NVFP4: the host stream is the long-context bound.
- **Heterogeneous compute:** no target exposed.
