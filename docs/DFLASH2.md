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

1. **DRAFT** (the block view, `ExecutionPlan.dflash2Block`): embedding, the 5 layers over all 8 rows at once, final norm,
   output head over rows 1..7, top-16 (`euhedral_dflash_topk_bf16`), selector projection and walk
   (`euhedral_dflash_select_bf16`); the 7 tokens, their candidates and scores are copied to pinned host memory before the
   quantum retires. The base position does not move.
2. **VERIFY** of the anchor and the first 6 drafts: the existing row-exact verification (7 rows, every row bit for bit a
   one-row decode), with acceptance on retirement. It taps every row. The depth is `ArtifactProfile.speculativeDepth()` (section 6).
3. **DRAFT_CONTEXT** (the context view, `dflash2Context`) over the committed rows' taps: `fc`, `hidden_norm`, then per layer the
   key/value projection and `euhedral_dflash_context_kv_bf16` into the ring.

The prompt prefills in chunks that tap their rows, each followed by its DRAFT_CONTEXT quantum; the first token's block then
starts the loop. Speculation applies to greedy, unconstrained generations from a fresh sequence, as for MTP.

**Taps.** A `DFLASH_TAP` stage after each tapped layer copies that layer's output rows (the residual the next layer's fused
residual-norm region writes) into the sequence's tap rows, `[row][tap][5120]`, only in quanta that seed drafting. Its position in
the plan after the next layer's input norm keeps the prefill and decode views' residual/norm fusion intact; it reads only the
residual, so it does not change the target's numerics. Verification rows are row-exact, so their taps are one-row decode's.

**Lifetime and ownership** (`DFlash2State`, owned by the sequence's `AttentionStates`):

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
  a verification. Outputs up to 1536 wide (the convolution kernel and selector projections) split K over a CTA's four warps,
  summed in warp order, so the block's narrow projections keep enough loads in flight;
- Qwen3 RMSNorm (`bf16(w · bf16(x / rms))`);
- the dynamic convolution, bit for bit the reference's two rounded steps per tap;
- Q/K head norm and full-width RoPE with BF16 cosines and sines, the ring append;
- the sliding-window block attention on tensor cores: one CTA per KV head and key split (8 splits) serves every row of the block,
  the rows times the group's 4 query heads (32 queries) forming the M rows of the MMAs, so each key and value is read once per
  block. Scores (BF16 products, FP32 sums) and the softmax's probabilities (hi + lo BF16 parts) stay in shared memory; a merge
  kernel combines the splits in order. 35 us per layer over a full window with 8 rows;
- SwiGLU with the SiLU rounded before the up product;
- top-16 ordered by logit then token, and the selector walk.

NVFP4 projections run on the existing NVFP4 dispatch (`linearNvfp4Bf16`): up to 8 rows (the block, a verification's context
rows) on the tensor-core decode kernels ([NVFP4_NATIVE.md](NVFP4_NATIVE.md), "Decode kernels"), more rows (a prefill chunk's
context rows) on the native Blackwell FP4 route. The output head is the target's.

**Prefetch of host-backed weights.** When some target weights are host-backed, a verification streams them through the staging
ring ([NVFP4_RESIDENCY.md](NVFP4_RESIDENCY.md)), and its first layers wait for the first copies. The block quantum, whose transfer
lane is otherwise idle for its few milliseconds, copies the decode view's first ring slots; a decode or verification quantum that
finds them loaded runs a view without those transfers (`ExecutionPlan.preloadedVariant`). The runtime marks the ring loaded
only when every stage of the prefetching quantum ran, under the ring's hold, so any other staging quantum in between (another
sequence's prefill) clears it.

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

RTX 5070 Ti (16 GB), i9-14900K, `qwen3_8_27b_nvfp4_compressed` as the target in every arm (the DFlash2 artifact extends it, so
the target bytes are identical). Chat corpus v2 (`promptCorpus: "chat"`), greedy, 256 generated tokens, warmup 1; 4 prompts
at 4K and 16K, 2 at 32K, 1 at 64K; the engine's context is sized to the scenario. Every arm's output hashes equal the ordinary
arm's for every prompt. The DFlash2 artifact is `nvfp4-fc`, verifying 6 drafts; MTP drafts 4. The DFlash2 64K row is the
median of three runs (63.8, 64.8, 67.5 tok/s).

Decode tok/s, time to first token, and the speculative step (per verification):

| Context | Arm | Decode tok/s | TTFT s | Prefill tok/s | Tokens / verification | Verify ms | Draft ms | Host-backed MiB |
|---|---|---|---|---|---|---|---|---|
| 4K | ordinary | 53.4 | 1.02 | 3906 | | | | 0 |
| 4K | MTP4 | 114.5 | 1.06 | 3782 | 3.08 | 22.05 | 3.78 | 0 |
| 4K | DFlash2 | 113.9 | 1.04 | 3845 | 3.19 | 23.05 | 3.42 | 468 |
| 16K | ordinary | 51.1 | 4.78 | 3347 | | | | 213 |
| 16K | MTP4 | 106.3 | 5.01 | 3202 | 3.02 | 23.97 | 3.89 | 213 |
| 16K | DFlash2 | 105.8 | 4.83 | 3318 | 3.01 | 24.72 | 3.38 | 733 |
| 32K | ordinary | 48.4 | 11.45 | 2849 | | | | 468 |
| 32K | MTP4 | 109.5 | 12.06 | 2702 | 3.27 | 25.68 | 4.17 | 468 |
| 32K | DFlash2 | 102.4 | 11.53 | 2830 | 3.11 | 27.15 | 3.38 | 924 |
| 64K | ordinary | 36.9 | 28.90 | 2205 | | | | 1052 |
| 64K | MTP4 | 79.2 | 31.07 | 2048 | 3.49 | 37.86 | 6.27 | 1052 |
| 64K | DFlash2 | 64.8 | 29.12 | 2187 | 3.54 | 49.12 | 5.51 | 1498 |

- **Draft time** is wall time between verifications: the context quantum of the committed rows (0.29-0.31 ms; 0.65 at 64K)
  and the block (3.07 ms at 4K to 32K, 4.8 ms at 64K, where its kernels span 2.9 ms). The prompt's context quanta are inside the TTFT; they cost less than MTP's prompt
  catch-up, which is why DFlash2's prefill rate is the higher of the two speculative arms.
- **Verification** of 7 rows costs about 1 ms more than MTP4's 5 rows at 4K and 16K. The rest of the difference is the PCIe cost
  of the weights the drafter's device memory pushes to the host: at a given context the DFlash2 artifact needs about 0.45-0.55
  GiB more device memory than MTP, so it streams 468 MiB of target projections per verification already at 4K, where MTP
  streams none. At 64K the verification is bound by those copies (1.5 GiB per step, the transfer lane busy 86% of the time).
- **Accepted drafts** on short chat requests (`DFlash2QualityCudaIntegrationTest`, 8 prompts of under 40 tokens, 7 verified): 3.1
  per verification (4.1 tokens per verification); the corpus above quotes long documents, where it drafts less well.

**Kernel time of one draft block** (Nsight Systems, 64K context, per block; span 2.91 ms): the drafter's NVFP4 projections
1.31 ms (tensor-core decode kernels, 8 rows), the target's output head over 7 rows 0.81 ms, block attention 0.25 ms (5 × 50
us, with the merge), BF16 linears (convolution kernel projections, selector projection) 0.22 ms, norms 0.11 ms, top-16 0.11
ms, the rest 0.1 ms.

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

The device-resident drafter is 1,056 MiB (NVFP4) or 3,428 MiB (BF16). The MTP layer and draft head (588 MiB) are not loaded
with a DFlash2 artifact. Peak allocated device memory in the runs of section 6: 13.6-13.7 GiB ordinary, 13.8-13.9 GiB MTP4,
13.9-14.0 GiB DFlash2; retained workspace 90-150 MiB.

## 8. Rejected

- **Verifying 5 or all 7 drafts.** One run per cell, decode tok/s with the tensor-core decode kernels (before the block's
  split-K linears, tensor-core attention and ring prefetch): 5 drafts 107.8 / 99.6 / 95.7 / 65.0 and 63.0 at 4K / 16K / 32K /
  64K, 6 drafts 109.2 / 99.7 / 96.7 / 66.2 and 65.6, 7 drafts 107.8 at 4K. Six wins or ties at every context.
- **A deeper staging ring** for the host-backed weights: 8 and 12 slots decoded 92.9 and 90.0 tok/s at 32K against 95.7 with 4,
  and 62.9 against 66.0 at 64K with 8. The streaming is bound by PCIe bandwidth, not by how far copies run ahead, and every slot
  is device memory that pushes another weight to the host.
- **Per-row block attention** (one CTA per row and KV head, FP32 dot products over every key):
  126 us per layer in the model, 82 us alone; each row re-read the window.
- **NVFP4 convolution kernel projections** (`nvfp4-all`): 2.9% fewer accepted drafts than the BF16 drafter on the quality
  prompts, beyond the 2% threshold; 82.4 tok/s at 4K.
- **The first block attention** (each thread summing two output values over every key): 915 us per layer over a full window.
- **One-pass top-16** (one CTA per row over 248,320 logits): 400 us per block.
- **Eight chunks in flight in the BF16 linear:** 255 GB/s against 373 at four on the convolution projections (registers).
- **The tiled GDN control from 8 rows:** 19.6 us against 9.8 for the per-row kernel at 8 rows (bitwise equal kernels; it now
  starts at 48 rows).

## 9. Open

- **Device memory.** The drafter needs about 0.5 GiB more than MTP's layer and draft head, so the target streams projections over
  PCIe from 4K on. Candidates: SD4 scale tables for the drafter's projections (about 50 MiB); host-backing the drafter's small
  projections (query and output, 118 MiB) so their copies run while the block leaves the transfer lane idle, in exchange for as
  many target bytes (an estimated 3-4% at 64K, where verification is bound by those copies); the MTP draft head's 131,072-row
  shortlist for the drafter's output head (0.4 ms less per block, at more device memory).
- **Verification attention** reads the KV cache once per verified row (about 0.6 ms per layer for 7 rows at 64K). A kernel that
  shares each key and value tile across rows must also leave room for more rows in flight per SM than the row twins' 4
  ([ATTENTION_DECODE.md](ATTENTION_DECODE.md)).
- **The block** (2.9 ms): 2.1 ms of it streams the drafter's projections and the output head at the DRAM rate; the context
  quantum could join the block quantum.
- **Q3 artifacts** have no DFlash2 drafter yet.
