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
2. **VERIFY** of `[anchor, d₁ .. d₇]`: the existing row-exact verification (8 rows on the one-row kernels' twins), with
   acceptance on retirement. It taps all 8 rows.
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
