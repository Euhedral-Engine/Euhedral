# MTP speculative decode: contract

This is the contract written before implementation. Sources:

- **Checkpoint:** `/mnt/shared/qwen38-quant/source/qwen`: `config.json` and the safetensors index.
- **This repository's converter:** `tools/convert_checkpoint.py` (`tools/euhedral_artifacts/inventory.py` lists the objects).
- **llama.cpp at 67a17c17c** (`/mnt/shared/qwen38-quant/llama.cpp`): `src/models/qwen35.cpp`,
  `common/speculative.cpp`, `conversion/qwen.py` and `src/llama-memory-recurrent.cpp`. File:line
  references below are to this tree.
- **ninfer** (`~/.hermes/worktrees/ninfer-upstream`), an independent implementation:
  `tools/convert/qwen3_5.py`, `src/models/qwen3_5/execution/text.cpp`,
  `docs/maintainer/qwen3_5-model.md` and `docs/maintainer/replayssm-gdn.md`.

The two implementations agree on everything marked "established". Where they differ, the choice
below says so.

## 1. Model

**Checkpoint.** `text_config.mtp_num_hidden_layers = 1` and `mtp_use_dedicated_embeddings = false`.
There are 15 tensors:
- `mtp.fc` [5120, 10240];
- `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.norm`;
- `mtp.layers.0.*`: one full-attention decoder block with gated q, q/k norm, o_proj and a SwiGLU MLP.

There is no MTP embedding and no MTP LM head (`qwen35.cpp:113-115` falls back to the base token
embedding and base `lm_head`).

**Artifact objects** (all four artifacts; the table lists each object's shape):

| Object | Shape | Source |
|---|---|---|
| `mtp/input_projection` | 5120 × 10240 | `mtp.fc` |
| `mtp/embedding_norm`, `mtp/hidden_norm` | 5120 | `pre_fc_norm_embedding`, `pre_fc_norm_hidden` |
| `mtp/layer/input_norm`, `mtp/layer/post_attention_norm` | 5120 | layer norms |
| `mtp/layer/attention/query_key_gate_value` | 14336 × 5120 | rows `[q 6144; k 1024; gate 6144; v 1024]` |
| `mtp/layer/attention/query_norm`, `key_norm` | 256 | q/k norms |
| `mtp/layer/attention/output` | 5120 × 6144 | o_proj |
| `mtp/layer/mlp/gate_up`, `mlp/down` | 34816 × 5120, 5120 × 17408 | MLP |
| `mtp/final_norm` | 5120 | `mtp.norm` |

- **Attention packing.** The converter splits the per-head interleaved `q_proj`
  (`[q_h | gate_h]` per head, confirmed by `qwen35.cpp:556-573` and ninfer `qwen3_5.py:496-510`)
  into contiguous q and gate, exactly as for base layers. The packed tensor is the base layers'
  `query_key` (7168 rows) followed by `gate_value` (7168 rows).
- **Formats.** Every MTP projection is plain NVFP4 (`row-split-k128-v1`) in all four artifacts, including the compressed ones.
  Q3 drafts accept fewer tokens per verification than NVFP4 ones, so the `q3` artifact stores its MTP layer in NVFP4
  ([MTP_VERIFIER.md](MTP_VERIFIER.md)). The draft head follows the artifact: Q3 in `q3` and `q3-compressed`, NVFP4 in `nvfp4` and
  `nvfp4-compressed`.
- **Norms.** Every MTP norm is the `(1 + w)` RMSNorm form (llama.cpp's converter bakes +1 into every
  `*norm.weight` except the GDN gated norm, `qwen.py:394-395`; ninfer
  `qwen3_5-model.md:116-124`). Euhedral already uses this form for base layer and q/k norms
  (`RMS_NORM_UNIT_OFFSET`, `qk_norm_rope.cuh`).

## 2. One MTP row (established)

Inputs: a token x and a hidden h. Output: an MTP hidden m, and draft logits.

1. `e = RMSNorm₁₊w(embed(x), embedding_norm)` using the base token embedding, and
   `n = RMSNorm₁₊w(h, hidden_norm)` (`qwen35.cpp:539-543`).
2. `c = input_projection · [e ; n]`: the embedding is columns 0..5119 and the hidden 5120..10239
   (`qwen35.cpp:545-548`; ninfer `mtp_pack.h`).
3. A full-attention decoder block identical to a base full-attention layer (`qwen35.cpp:553-618`):
   - `input_norm`, gated q, q/k norm;
   - partial IMROPE (64 rotary dims, θ = 1e7), the same as base layers;
   - attention over the **MTP's own KV cache** with scale `1/√256`;
   - `· sigmoid(gate)`, o_proj, then the residual with `c`;
   - `post_attention_norm`, SwiGLU FFN, residual.
4. `m = RMSNorm₁₊w(block output, final_norm)`. This is `mtp.norm`, after the block
   (`qwen35.cpp:621-628`).
5. The draft logits are the base `lm_head · m` (`qwen35.cpp:630-637`).

**Seed hidden.** h is the base model's hidden at position p **after the base final norm**: the
same vector the LM head reads (`qwen35.cpp:206-209`; ninfer `text.cpp:701-702`). The comments that
say "before final norm" (`llama-graph.h:937`) are stale.

**Alignment.** The row pairs `(h_p, x_{p+1})`. h_p is the base hidden of the row that predicted
x_{p+1}. The row's output predicts x_{p+2}.

**Recursion.** Draft k+1 uses `(m_k, x̂_{k+1})`: the previous MTP row's post-`mtp.norm` hidden and
the token it drafted (`speculative.cpp:1665-1718`; ninfer `mtp.cpp:52-67`). One MTP layer is reused
for every depth.
- llama.cpp defaults to 3 drafts (`common.h:326`); ninfer allows up to 5.
- **MTP3 = three recursive applications.**

## 3. Choices where the references differ

- **MTP row position: the hidden's position p** (ninfer, `mtp_alignment.h:21-37`), not p + 1
  (llama.cpp). RoPE is relative, so a uniform shift leaves every attention score mathematically
  unchanged, and either choice only affects drafts, never committed output.
- **No zero-hidden row for the first prompt token** (ninfer). llama.cpp pairs x₀ with h = 0
  (`speculative.cpp:1430`); training pairs only real (h_p, x_{p+1}).
- **Draft head.** The artifacts carry `text/draft_head`: the 131,072 most frequent `lm_head` rows,
  chosen by the converter's token ranking (`--ranking`, or copied from an existing artifact with `--draft-ids-from`), plus
  `text/draft_head_token_ids`, the token for each row.
  - It is **this repository's construct, not the checkpoint's**; the checkpoint's MTP uses the full
    `lm_head`.
  - **Choice: draft with the shortlist** (argmax over 131,072 rows, then map through
    `draft_head_token_ids`). It reads 360 MiB instead of 682 MiB per draft in NVFP4.
  - A token outside the shortlist can never be drafted. It still appears in output whenever the
    verifier produces it, so this only costs acceptance.
  - The full head stays a measurable alternative.

## 4. The speculative step (greedy)

The text below is written for depth 3; the artifact fixes the depth (2 for Q3 artifacts, 3 for NVFP4 artifacts, `ArtifactProfile`),
and every count scales with it (a verifier of depth + 1 rows).

State before a step:
- base KV and GDN state committed through position P − 1;
- t₀ = x_P, the last committed output token, not yet run through the base model;
- three drafts d₁..d₃ for positions P+1..P+3.

1. **Verify.** Run the base model on rows `[t₀, d₁, d₂, d₃]` at positions P..P+3. Each row j
   gives `g_j = argmax lm_head(final_norm(h_{P+j}))`.
2. **Accept.** a = the largest count with `d_i = g_{i-1}` for all i ≤ a (0 ≤ a ≤ 3).
   - Committed output this step: `d₁..d_a`, then the base token `g_a`. That is a + 1 tokens
     (1..4), each exactly the token ordinary greedy decode produces, if the verifier rows are
     numerically identical to ordinary decode (see §6).
   - EOS among them ends generation at that token; tokens after it are discarded.
3. **Commit base state** through position P + a (rows 0..a). The new t₀ is g_a at position
   P + a + 1.
4. **MTP catch-up and draft 1.** Run MTP rows `(h_{P+j}, x_{P+j+1})` for j = 0..a, where
   x_{P+j+1} = d_{j+1} for j < a and g_a for j = a, at positions P+j.
   - This writes MTP KV for those positions (kept).
   - Row a's output gives d'₁ (for position P+a+2) and m₁.
5. **Drafts 2 and 3:** MTP rows `(m₁, d'₁)` at P+a+1 and `(m₂, d'₂)` at P+a+2. Their MTP KV is
   speculative: the next step's catch-up rewrites those positions from verified hiddens.

**After the prompt:**
- The base prefill must keep the post-final-norm hidden of every prompt row, not only the last.
- MTP catch-up then runs over (h_p, x_{p+1}) for p = 0..N−2, and over (h_{N−1}, g) where g is the
  first generated token, to seed d₁.

**Verifier rows, drafts and output.** 4 rows per verifier invocation: 3 proposed drafts plus the
base row. Accepted drafts number 0..3, committed tokens 1..4, and the bonus is g_a in every case.
The full-acceptance case is a = 3 with 4 tokens committed.

## 5. State mutated, and what commit must restore

| State | During drafting | During verify | Commit after a accepted |
|---|---|---|---|
| Base KV (16 layers) | none | rows P..P+3 written beyond the committed frontier | committed length += a + 1. Rejected rows stay beyond `length` and the next append overwrites them; this is the existing reserved/submitted/committed frontier with a partial publish |
| Base GDN recurrent (48 × 3 MiB FP32 = 144 MiB) and convolution (48 × 60 KiB) | none | advanced through 4 rows | must equal the state after row a. Plan: ReplaySSM (below) |
| MTP KV (1 layer) | 2 speculative rows | none | catch-up rows P..P+a committed; draft rows discarded and rewritten next step |
| Sequence position, output tokens | none | none | += a + 1 |

**GDN rollback, measured choices** (§7 of the campaign log):
- **Per-row snapshots** (llama.cpp, `llama-memory-recurrent.cpp:101`, `delta-net-base.cpp:546-602`):
  4 × 144 MiB extra and 4 full-state writes per step.
- **ReplaySSM** (ninfer, `replayssm-gdn.md`):
  - Verify runs the recurrence and convolution against a scratch copy, or from a kept checkpoint.
  - It records each row's convolved q/k/v, α and β: about 21 KB per row per layer, about 4 MiB for
    4 rows over 48 layers.
  - It then replays the accepted prefix from the committed checkpoint **with the same per-row
    transition code**, so the result is the bitwise sequential state.
  - Cost: one 144 MiB checkpoint buffer, and a replay of a + 1 rows only when a < 3.
- The convolution state after row a is the last three pre-convolution rows ending at a, rebuilt
  from the old history plus the recorded rows.

**Tests required.**
- After every step, the base KV rows, GDN recurrent state, GDN convolution state, MTP KV rows and
  sequence position must equal those of ordinary sequential decode over the same committed tokens,
  **bitwise**.
- This holds only if the verifier's rows are bitwise identical to single-row decode. See §6.

## 6. The verifier is numerically identical to single-row decode

The oracle is ordinary greedy decode, which runs every row through the decode kernels:
- the NVFP4 M = 1 GEMV with **BF16 activations**;
- split-KV decode attention;
- the decode GDN path.

Committed tokens and state equal ordinary decode exactly only if the verifier computes each of its
rows with arithmetic identical to that single-row path. Native NVFP4 OMMA cannot do that against the
GEMV: it needs FP4 activations (`kind::mxf4nvf4` only accepts E2M1 × E2M1), and the MXFP8 route quantizes activations too. The multi-row prefill
attention and GDN kernels also accumulate in a different order from the decode kernels.

**Decision: row-exact verification.** A VERIFY quantum selects row-exact execution (`euhedral_cuda_row_exact_select`). Every
operator whose kernel depends on the row count then runs each row through its one-row path:
- attention, q/k norm and RoPE, residual norms and the GDN recurrence, row by row;
- Q3, Q4 and Q5 linears through multi-row twins of the one-row GEMV kernels
  (`q3::contiguous_decode_rows<M>`, `q45::contiguous_decode_rows<B, M>`, for M = 2 to 8). Each
  repeats the one-row FMA sequence for every token row while streaming the weights once, so every
  row is bitwise identical to one-row decode.
- NVFP4 linears through the tensor-core decode kernels (`nvfp4::decode_rows<M>`, M = 1 to 8), whose
  token rows are independent MMA columns with exact products, so every row is bitwise identical to
  one-row decode ([NVFP4_NATIVE.md](NVFP4_NATIVE.md)).

Row-exact execution declines the block-scaled MXFP8 route and the native FP4 route, whose activation quantization and tile
accumulation differ from the GEMV. There is no second numerics mode for verification: a verifier on native FP4 kernels would
change every generated text ([MTP_VERIFIER.md](MTP_VERIFIER.md)).

This is tested bitwise, on tokens and on GDN and KV state, by `SpeculativeVerifyCudaIntegrationTest` and
`SpeculativeDecodeCudaIntegrationTest`. Results are in [MTP_SPECULATIVE.md](MTP_SPECULATIVE.md).
