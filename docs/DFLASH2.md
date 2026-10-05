# DFlash2 speculative decoding

DFlash2 ([z-lab/dflash](https://github.com/z-lab/dflash), checkpoint `z-lab/Qwen3.8-27B-DFlash2`) is a block-diffusion drafter: it
proposes the 7 tokens after an anchor in one pass of 5 small transformer layers, conditioned on the target's own hidden states,
and a candidate selector traces one path through each position's top 16 tokens. The target verifies the 8 rows exactly, as for
MTP ([MTP_CONTRACT.md](MTP_CONTRACT.md)), so greedy output is ordinary greedy output whatever the drafter proposes.

An artifact that holds the drafter (`dflash2/` objects) speculates with it; one that holds the MTP layer and draft head speculates
with MTP. There is no option for it: `ArtifactProfile.speculation()` reads it from the artifact.

## 1. The drafter (upstream semantics)

Configuration of the published checkpoint (recorded in `dflash2/config`): 5 layers, hidden 5120, intermediate 17408, 32 query
and 8 KV heads of 128, block 8, mask token 248070, sliding window 2048, RMSNorm epsilon 1e-6, RoPE theta 1e7 (full-width,
`rotate_half`), dynamic convolution kernel 2 in groups of 16 channels, selector top-16 and rank 256, target layers 5, 19, 33,
47 and 61.

- **Taps.** The residual stream after target layers 5, 19, 33, 47 and 61 (Hugging Face `hidden_states[l + 1]`), concatenated per
  row (25,600 values).
- **Context.** `hidden_norm(fc(taps))`; each draft layer projects it to keys (with `k_norm` and RoPE at the row's position) and
  values. Upstream keeps these in the draft model's KV cache; positions are absolute.
- **Block.** The target's embedding of `[anchor, mask × 7]` at positions `P .. P + 7`. Each layer: `input_norm`, the
  attention convolution's prepare kernel, attention (queries from the block; keys and values from the context and the block;
  not causal; a query at `q` sees keys `k` with `|q - k| < 2048`), its finish kernel, residual; `post_attention_norm`, the MLP
  convolution's prepare kernel, SwiGLU MLP, finish kernel, residual. Then the final norm.
- **Grouped dynamic causal convolution.** A kernel projection of the normalized rows gives, per row, 2 parts × 2 taps × 320
  groups. Prepare and finish each compute `out = Σ_o (base[o][c] + dynamic[t][o][c / 16]) · x[t - o][c]` over the block's rows
  (`x[-1] = 0`), as two rounded steps per tap.
- **Proposal.** The target's output head over rows 1..7, the top 16 logits per row, the selector's rank-256 projection of each
  row, and a walk from the anchor: `score[k] = logit[k] + ((predecessor[prev] ⊙ h) · successor[candidate k])`, argmax, next.
- **After a verification** that commits `a + 1` tokens (the anchor and `a` accepted drafts), their target rows become context,
  and the bonus token is the next anchor.

`tools/dflash2_reference.py` runs the upstream model unmodified over fixed inputs and writes every intermediate as fixtures.

## 2. Execution

One step is three quanta on the existing runtime, chained as continuations (`DFlash2Decoder`):

1. **DRAFT** (the block view, `QwenExecutionPlan.dflash2Block`): embedding, the 5 layers over all 8 rows at once, final norm,
   output head over rows 1..7, top-16 (`euhedral_dflash_topk_bf16`), selector projection and walk
   (`euhedral_dflash_select_bf16`); the 7 tokens, their candidates and scores are copied to pinned host memory before the
   quantum retires. The base position does not move.
2. **VERIFY** of the anchor and the first 5 drafts: the existing row-exact verification (6 rows on the one-row kernels'
   twins), with acceptance on retirement. It taps every row. The depth is `ArtifactProfile.speculativeDepth()` (section 6).
3. **DRAFT_CONTEXT** (the context view, `dflash2Context`) over the committed rows' taps: `fc`, `hidden_norm`, then per layer the
   key/value projection and `euhedral_dflash_context_kv_bf16` into the ring.

The prompt prefills in chunks that tap their rows, each followed by its DRAFT_CONTEXT quantum; the first token's block then
starts the loop. Speculation applies to greedy, unconstrained generations from a fresh sequence, as for MTP.

**Taps.** A `DFLASH_TAP` stage after each tapped layer copies that layer's output rows (the residual the next layer's fused
residual-norm region writes) into the sequence's tap rows, `[row][tap][5120]`, only in quanta that seed drafting. Its position in
the plan after the next layer's input norm keeps the prefill and decode views' residual/norm fusion intact; it reads only the
residual, so it does not change the target's numerics. Verification rows are row-exact, so their taps are one-row decode's.

**Lifetime and ownership** (`DFlash2SequenceState`, owned by the sequence's `AttentionSequenceStates`):

- Tap rows: written by the latest target quantum that seeds drafting (a prefill chunk or a verification), consumed by the
  DRAFT_CONTEXT quantum that follows it, overwritten by the next. A sequence's quanta are serial, so one buffer serves; it is
  sized by the largest such quantum (a 512-row chunk: 26 MB). No history of taps is kept.
- Context ring: per layer, keys and values of the last 2048 committed positions, `[slot][1024]` BF16 with position `p` at slot
  `p % 2048` (5 layers × 2 × 4 MiB = 40 MiB). It is all the drafter keeps of a sequence, whatever its length, because the
  drafter attends only within the window. `contextLength` (positions committed into the ring) is published at the retirement of
  the DRAFT_CONTEXT quantum that wrote them; a block starts only when it equals the sequence position.
- Released with the sequence.

**Kernels** (`native/src/dflash/kernels.cu`, `native/src/host/dflash.c`) round where the published PyTorch model rounds:
- a BF16 linear on m16n8k16 tensor cores with FP32 accumulation and one rounding; a row's bits do not depend on how many rows the
  call has (both its 16-row and 64-row tilings run the same K order), so context rows computed in a prefill chunk equal those of
  a verification;
- Qwen3 RMSNorm (`bf16(w · bf16(x / rms))`);
- the dynamic convolution, bit for bit the reference's two rounded steps per tap;
- Q/K head norm and full-width RoPE with BF16 cosines and sines, the ring append, the sliding-window block attention (FP32 scores
  and softmax);
- SwiGLU with the SiLU rounded before the up product;
- top-16 ordered by logit then token, and the selector walk.

NVFP4 projections run on the existing NVFP4 dispatch (`linearNvfp4Bf16`): from 2 rows that is the native Blackwell FP4
tensor-core route (two-term NVFP4 activations, block-scaled FP4 MMA), for the block's 8 rows and a prefill chunk's context rows
alike. The output head is the target's.

## 3. Artifact

`tools/convert_checkpoint.py --dflash2 DIR` adds the drafter to a conversion; `--extend ARTIFACT --dflash2 DIR` appends it to an
existing artifact, copying every object byte for byte (the target is then identical to the source artifact's).
`--dflash2-projections nvfp4-fc` (the shipped artifact) stores the 25 projection objects and the feature fusion `fc` as
plain NVFP4; `nvfp4` the projection objects only, `nvfp4-all` the convolution kernel projections too (section 6).
`--dflash2-projections nvfp4` stores the 25 projection objects (per layer `attention/query`, `attention/key_value` (k rows then
v), `attention/output`, `mlp/gate_up` (gate rows then up), `mlp/down`: the checkpoint's 35 q/k/v/o/gate/up/down matrices) as
plain NVFP4; norms, convolution bases and projections, `fc`, the selector projection and codebooks stay BF16. `dflash2/config`
(I32) records the configuration, so the runtime reads no Hugging Face file.

A load uploads only the drafter the artifact selects (the MTP layer and draft head of an artifact that also holds them are not
loaded) and keeps the selector codebooks (2 × 127 MB) in mapped host memory: a step reads 120 of their rows.

## 4. Prefix checkpoints

A DFlash2 checkpoint at position `p` holds the base state and the drafter's ring (`DFlash2Checkpoint`, 40 MiB per node). The
ring at `p` is the last 2048 positions' keys and values, and every DRAFT_CONTEXT row depends only on its own taps, so a sequence
restored at `p` and a cold one have the same ring bit for bit, and prefill continues from `p` in the chunks a cold run uses:
nothing is recomputed (unlike MTP, whose row `p - 1` pairs with the next prompt's token). Storing the taps instead (105 MB for 2048
rows) or the normalized context rows (21 MB, with 5 layers of key/value projections to recompute on restore) were not needed: the
ring is the smaller of the two that restore without arithmetic.

## 5. Validation

- `test_dflash.py` (native): each kernel against a NumPy oracle with the reference's rounding points; the convolution, top-16
  and selector bit for bit; the linear's rows identical at 1, 3, 8, 16, 17, 64 and 65 rows.
- `DFlash2FixtureCudaIntegrationTest`: the engine's intermediates against `tools/dflash2_reference.py` run on the engine's own
  taps and embedded block rows, with the artifact's NVFP4 output head (FP32 products, as the engine's kernels compute):
  - relative RMS against the BF16 reference: layer 0 below 0.6% (its input norm bit for bit), the last layer below 2%, logits
    1.2%;
  - against the FP32 reference the engine (final hidden 1.7%, logits 1.25%) is as close as the BF16 reference (1.7%, 1.2%);
  - its top-16 sets equal the FP32 reference's, and its proposal equals both references'.
- `DFlash2SpeculativeDecodeCudaIntegrationTest`: DFlash2 generations equal ordinary greedy decode in tokens and leave the same
  KV and GDN state bit for bit, for a chat answer, 600 tokens of prose and a code prompt, 160 tokens each.
- `DFlash2PrefixCacheCudaIntegrationTest`: restored generations equal cold and uncached ones (a restore at a checkpoint, one
  before a prompt's tail, repeats); the restored ring and target state equal a cold run's bit for bit; a small cache evicts and
  stays exact; a cancelled generation releases its device state.

## 6. Measured

RTX 5070 Ti (16 GB), i9-14900K, `qwen3_8_27b_nvfp4_compressed` as the target in every arm (the DFlash2 artifacts extend it, so
the target bytes are identical). Chat corpus v2 (`promptCorpus: "chat"`), greedy, 256 generated tokens, warmup 1; 4 prompts
at 4K and 16K, 2 at 32K, 1 at 64K; the engine's context is sized to the scenario. Every arm's output hashes equal the ordinary
arm's for every prompt. The DFlash2 NVFP4 artifact is `nvfp4-fc`, verifying 5 drafts.

Decode tok/s, time to first token, and the speculative step (per verification):

| Context | Arm | Decode tok/s | TTFT s | Prefill tok/s | Tokens / verification | Verify ms | Draft ms | Host-backed MiB |
|---|---|---|---|---|---|---|---|---|
| 4K | ordinary | 51.6 | 1.02 | 3909 | | | | 0 |
| 4K | MTP3 | 81.6 | 1.05 | 3791 | 2.74 | 29.98 | 2.96 | 0 |
| 4K | DFlash2 BF16 | 29.3 | 1.09 | 3670 | 3.01 | 93.52 | 8.27 | 2922 |
| 4K | DFlash2 NVFP4 | 83.2 | 1.04 | 3847 | 2.97 | 30.82 | 4.04 | 468 |
| 16K | ordinary | 48.5 | 4.79 | 3345 | | | | 213 |
| 16K | MTP3 | 77.2 | 5.00 | 3204 | 2.79 | 31.73 | 2.97 | 213 |
| 16K | DFlash2 BF16 | 24.2 | 5.11 | 3137 | 2.96 | 112.21 | 8.34 | 3177 |
| 16K | DFlash2 NVFP4 | 76.9 | 4.84 | 3312 | 2.89 | 32.71 | 4.02 | 733 |
| 32K | ordinary | 46.3 | 11.44 | 2849 | | | | 468 |
| 32K | MTP3 | 81.1 | 12.09 | 2695 | 2.97 | 33.32 | 3.23 | 468 |
| 32K | DFlash2 BF16 | 25.7 | 12.80 | 2549 | 3.07 | 111.15 | 8.61 | 3368 |
| 32K | DFlash2 NVFP4 | 79.9 | 11.49 | 2834 | 3.09 | 34.78 | 4.01 | 924 |
| 64K | ordinary | 32.0 | 28.92 | 2199 | | | | 1052 |
| 64K | MTP3 | 67.3 | 31.12 | 2045 | 3.15 | 42.27 | 4.51 | 1052 |
| 64K | DFlash2 BF16 | 23.4 | 33.31 | 1911 | 3.31 | 131.96 | 9.32 | 4016 |
| 64K | DFlash2 NVFP4 | 62.9 | 29.12 | 2184 | 3.23 | 45.57 | 5.75 | 1498 |

- **Draft time** is wall time between verifications: the context quantum of the committed rows (0.33-0.35 ms; 0.70 at 64K)
  and the block (3.67 ms; 5.01 at 64K). The prompt's context quanta are inside the TTFT; they cost less than MTP's prompt
  catch-up, which is why DFlash2's prefill rate is the higher of the two speculative arms.
- **Verification** of 6 rows costs what MTP3's 4 rows cost, plus the PCIe cost of the weights the drafter's device memory
  pushes to the host: at a given context the DFlash2 NVFP4 artifact needs about 0.45-0.55 GiB more device memory than MTP.
- **Accepted drafts per verification** (histogram, 0..5 accepted) at 4K: `[84, 83, 66, 35, 24, 52]` over 344 verifications.
  On short chat requests (`DFlash2QualityCudaIntegrationTest`, 8 prompts of under 40 tokens) the drafter verifies 3.0 drafts
  on average (4.0 tokens per verification) with 7 verified; the corpus above quotes long documents, where it drafts less well.
- The BF16 drafter (3.6 GiB) forces 2.9-4.0 GiB of the target to the host, which the verifier reads over PCIe every step.

**Kernel time of one draft block** (Nsight Systems, 4K, NVFP4 projections; per block): the drafter's NVFP4 projections about
1.1 ms (native FP4 skinny kernel), the target's output head over 7 rows 0.65 ms, BF16 linears (convolution kernel
projections, selector projection) 0.63 ms, block attention 0.41 ms (5 × 82 us), top-16 0.07 ms, selector walk 14 us, norms,
convolutions and the rest 0.2 ms.

**Prefix cache** (4 GiB, the same prompt three times; the warm ones restore all but the last chunk):

| Context | Arm | Cold TTFT | Warm TTFT | Restore |
|---|---|---|---|---|
| 16K | MTP3 | 5.30 s | 0.164-0.176 s | 32-33 ms |
| 16K | DFlash2 NVFP4 | 5.21 s | 0.171-0.181 s | 36-38 ms |
| 32K | MTP3 | 12.50 s | 0.213-0.228 s | 50-56 ms |
| 32K | DFlash2 NVFP4 | 11.95 s | 0.227-0.231 s | 53-62 ms |

A DFlash2 node adds its 40 MiB ring to the base state; a restore recomputes nothing.

## 7. Memory

| Item | DFlash2 NVFP4 (`nvfp4-fc`) | DFlash2 BF16 |
|---|---|---|
| Drafter projections (5 layers × q, kv, o, gate/up, down) | 858 MiB | 3,050 MiB |
| `fc` | 70 MiB | 250 MiB |
| Convolution kernel projections (BF16) | 125 MiB | 125 MiB |
| Norms, convolution bases, selector projection | 3 MiB | 3 MiB |
| Selector codebooks (mapped host memory) | 243 MiB host | 243 MiB host |
| Context ring (per sequence) | 40 MiB | 40 MiB |
| Tap rows (per sequence, a 512-row chunk) | 25 MiB | 25 MiB |
| Artifact | 16.15 GB | 18.63 GB |

The device-resident drafter is 1,056 MiB (NVFP4) or 3,428 MiB (BF16). The MTP layer and draft head (588 MiB) are not loaded with a DFlash2 artifact. Peak allocated device memory in the runs above is
13.9-14.0 GiB for every arm; retained workspace 112-134 MiB.

## 8. Rejected

- **Verifying all 7 drafts.** The exact verifier's NVFP4-SD4 row twins cost 120.7 us at 7 rows and 125.5 at 8 against 92.8 at 6
  (MLP down shape, cold weights; registers 223-234, no spills), and a 7-row verification 35.5 ms against 30.4 for 6 rows (4K
  chat prompts). With 7 verified the arm decoded 69.3 tok/s at 4K (3.07 tokens per verification).
- **The drafter's linears on the BF16-activation row twins** instead of the native FP4 route: block 5.26 ms, acceptance unchanged
  (3.04 tokens per verification), 66.3 tok/s at 4K.
- **NVFP4 convolution kernel projections** (`nvfp4-all`): 2.9% fewer accepted drafts than the BF16 drafter on the quality
  prompts, beyond the 2% threshold; 82.4 tok/s at 4K.
- **The first block attention** (each thread summing two output values over every key): 915 us per layer over a full window.
- **One-pass top-16** (one CTA per row over 248,320 logits): 400 us per block.
- **Eight chunks in flight in the BF16 linear:** 255 GB/s against 373 at four on the convolution projections (registers).
- **The tiled GDN control from 8 rows:** 19.6 us against 9.8 for the per-row kernel at 8 rows (bitwise equal kernels; it now
  starts at 48 rows).

## 9. Open

- **Verification is 85-88% of a step.** A 7- and 8-row twin of the NVFP4-SD4 decode GEMV without the 6-to-7-row step would let
  DFlash2 verify all its drafts (4.0 against 3.6 tokens per verification on short requests).
- **Device memory.** The drafter needs about 0.5 GiB more than MTP's layer and draft head; every 128 MiB on the host costs about
  2-2.5% of verification. Candidates: SD4 scale tables for the drafter's projections, a deterministic split-K BF16 linear for the
  narrow convolution and selector projections, or the MTP draft head's 131,072-row shortlist for the drafter's output head.
- **The block** (3.7 ms): its BF16 linears stream at 373 GB/s, and the context quantum could join the block quantum.
- **Q3 artifacts** have no DFlash2 drafter yet.
