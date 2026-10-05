# Flash-Next reference harness

Deterministic fixtures for the Flash-Next (`Qwen/Qwen3.8-Flash-Next`, HF `model_type` `qwen4_exp`) text model, produced by
the **unmodified** upstream transformers implementation over the weights of the NVFP4 EDRL v3 artifact
(`docs/FLASH_NEXT_ARTIFACT.md`). The Java engine and the CUDA operators compare against them; the tensor names, dtypes and
shapes below are the contract.

* Upstream: transformers `qwen4_exp` at revision `f5ab85619d989359ef47b5efed8a91a15045627b` (source copy in
  `~/src/third_party/transformers-qwen4/`, installed from that commit in the venv).
* Code: `tools/flash_next_reference.py` (CLI), `tools/flash_next_reference/` (package), `tools/test_flash_next_reference.py`.
* Fixtures are not in the repository: `/home/brandon/fixtures/flash-next/<case>/` (about 1 GB for all cases).

## Regenerating

```bash
cd ~/src/Euhedral-Inference
PY=~/.cache/flash-next-ref-venv/bin/python        # torch 2.14 cu130, transformers (pinned commit), safetensors, numpy, tokenizers
export PYTHONPATH=tools
$PY tools/flash_next_reference.py list-cases
$PY tools/flash_next_reference.py check-artifact [--full]       # config vs metadata, CRC-32 of a sample of objects
$PY tools/flash_next_reference.py fixtures --out /home/brandon/fixtures/flash-next --case short   # or --case all
$PY tools/flash_next_reference.py fixtures --out DIR --case layer_qsa_short --kv-format nvfp4
$PY tools/flash_next_reference.py generate --prompt "The capital of France is" --steps 6        # or --prompt-ids 760 6511 ...
$PY tools/flash_next_reference.py fp32-chunk-check               # chunk invariance of real layers in fp32
$PY -m unittest tools/test_flash_next_reference.py               # unit tests (real-artifact tests skip without it)
```

`--out` is the fixture root; a case is written to `OUT/<case>/` (existing contents are replaced). `--device cuda` is the
default; fast kernels (`fla`, `causal_conv1d`) are absent, so upstream falls back to its pure-torch reference code, which is
what the fixtures use. The GPU must be free for a model case (6.5 GiB peak); each case releases it before the next.

## Numerics contract

* **NVFP4 linears** (`ExactNvfp4Linear`): every NVFP4 tensor is kept packed on the GPU (codes + E4M3 scales + global scale)
  and expanded per call to FP32 weights `e2m1(code) * (e4m3(scale) * global)` (the association of
  `euhedral_artifacts.nvfp4.dequantize_nvfp4_rows`; asserted bit-equal to it, for fixed tensors and expert records, in the
  tests). The product is `F.linear(x.float(), W_fp32).to(x.dtype)`: FP32 products and accumulation, **one** BF16 rounding of
  the output. TF32 is off (`allow_tf32=False`, precision `highest`) and so is the reduced-precision BF16 split-K reduction.
  Codes are row-major, two per byte, the even k in the LOW nibble; scales are `[rows, K/16]` E4M3 at the 256-aligned
  `row-split-k128-v1` offset (`nvfp4_offsets`).
* **BF16-stored tensors** (hyper-connection projections and `block_inject`, routers, the indexer projection, `in_proj_a/b`,
  conv weights, norms, `A_log`, `dt_bias`, `shared_expert_gate`, token embedding, `lm_head`) are used as stored with the
  ordinary BF16 torch ops of upstream.
* **Experts**: `LazyExperts` subclasses upstream's `Qwen4ExpTextExperts`. Its forward is upstream's loop verbatim; only
  `gate_up_proj[e]` / `down_proj[e]` come from that expert's artifact record (expanded exactly, `exact_linear`). The
  accumulation is upstream's: a BF16 `final_hidden_states`, `index_add_` of each expert's term (the BF16 product of the
  down output and the BF16 routing weight) in **ascending expert-index order**, every term rounded to BF16 before it is
  added. A test shows the loop equals upstream's own `Qwen4ExpTextExperts` on the same expanded weights bit for bit in FP32.
  The rows an expert sees are ordered the way upstream's `torch.where(expert_mask[e])` yields them: by top-k position first,
  then by token (`expert/{e}/tokens` is in that order, and `out`/`weighted` rows follow it).
* **Router**: BF16 logits; softmax in FP32; `torch.topk`; the top values are divided by their FP32 sum (`norm_topk_prob`) and
  rounded to BF16. The deterministic comparison rule is a stable descending sort (ties by ascending expert index).
* **Attention**: `eager` (`attn_implementation="eager"`): BF16 `q.k^T * scale + mask`, FP32 softmax rounded to BF16, BF16
  `p.v`. (An upstream model loaded the usual way would use `sdpa`; the fixtures pin eager so `core` is a definition, not a
  kernel's result.) The QSA mask is upstream's: the causal mask plus the selected-token mask of the indexer.
* **Experts implementation**: the eager per-expert loop above (the default of a stock load would be `grouped_mm`).
* **Indexer, per query**: for query row `i` the visible tokens are `0..p_i`; the complete blocks are the first
  `(p_i+1)//4` groups of 4 consecutive visible tokens; their RAW index keys (`index_qk_proj` output before any norm) are
  pooled by an FP32 mean, rounded to BF16, passed through `k_layernorm`, then RoPE at the **first position of the block**;
  scores `relu(q . k)` summed over the 4 heads in FP32 and divided by `sqrt(128)` (the query is `q_layernorm` + RoPE at its
  own position); `topk(min(512, nb))` over blocks; the selected tokens are those blocks' tokens plus the tail (the visible
  tokens of the incomplete last block). Everything is recomputed per query row (a Python loop upstream); there is no cache
  of pooled keys, only of the raw keys (`indexer_keys`).
* **Cache states** are what `DynamicCache(config=cfg)` stores after the chunk: GDN `conv_states[0]` is `[1, 10240, 4]` (the
  last 4 raw QKV inputs, kernel size not kernel-1) and `recurrent_states[0]` FP32 `[1, 48, 128, 128]`; PLE `conv_states[1]`
  `[1, 10240, 9]` (`(4-1)*3` gated-normed values, dilation 3) and `conv_states[2]` the last 2 token ids as int64 `[1, 2]`,
  initialised with EOS; QSA `keys`/`values` `[1, 2, len, 256]` and `indexer_keys` `[1, len, 128]`.
* Dropout, gradients, `use_deterministic_algorithms` are off; two runs of a case produce byte-identical files (tested for
  `short` and `layer_moe`).
* **Replay assertions**: the PLE layer and the QSA indexer are recorded through subclasses that carry a copy of upstream's
  forward body. On every call the unmodified upstream forward is also run on the same inputs (cache states put back in
  between) and the harness fails unless the outputs and the resulting cache states are bit-identical (`replays_checked` in
  the manifest metadata counts them).

### KV format `--kv-format nvfp4`

Emulates the engine's NVFP4 KV cache by applying the codec to `key_states`/`value_states` (after `k_norm` and RoPE,
`[1, 2, rows, 256]`) before they enter the cache, so attention sees `H^T dequant(quant(H k))`. Ported from
`native/src/attention/nvfp4_kv.cuh` (the device codec) with the semantics of `Nvfp4KvReference.java`:

* a row is one head: 256 BF16 values; rotation by the normalized Sylvester H256 (entries +-1/16) in FP32 with the device's
  butterfly (five stages over index bits 0..4 with `lane&stride ? peer-v : v+peer`, three stages over bits 5..7, `* 0.0625`);
* per group of 16 rotated values: scale = E4M3 (round to nearest even, saturating) of `min(448, max(2^-9, fp32(max|x|)/6))`
  (IEEE division); all-zero group: scale code 0, codes 0; codes = `E2M1(x / fp32(scale))` (IEEE division, ties to the even
  code: `<=0.25`->0, `<0.75`->1, `<=1.25`->2, `<1.75`->3, `<=2.5`->4, `<3.5`->5, `<=5`->6, else 7, sign bit kept);
* the stored row is 144 bytes (`pack_rows`: 128 code bytes, even element in the low nibble, then 16 scale bytes).

Deviations from the engine: the engine keeps the rotated representation and rotates queries/outputs instead; here the
inverse rotation (same butterfly, FP32) is applied to the represented row and the result is rounded to BF16 (the cache
dtype). Indexer keys are not quantized. Verified: the torch codec equals the CUDA codec **bit for bit** (rotation forward and
inverse, codes, scales) on random and degenerate rows, via NVRTC and `native/tests/gpu_harness.py` in a subprocess
(`test_bit_exact_against_the_cuda_codec`), and agrees with the float64 `Nvfp4KvReference` semantics except on rare exact
ties (a float32 vs float64 rotation). On the real layer 3, the cached K/V differ from the BF16 cache by about 9.5% relative
L2 and the layer output by about 15% (4-bit codes).

## Fixture format

`manifest.json`: `{"metadata": {...}, "tensors": {name: {"dtype", "shape", "file"}}}`; dtypes `bf16`, `f32`, `i32`, `i64`,
`u8`; each file is the raw little-endian tensor (bf16 as 16-bit words, row-major). The batch dimension is dropped
everywhere (`rows` first). A tensor with the same bytes as an earlier one shares its file (`L{n}/in` is the file of
`L{n-1}/out`; `moe/in` that of `mlp_hc/mixed`): follow the manifest's `file`. Metadata carries the case, the upstream
revision, versions, chunk table and timings, the prompt/decoded tokens, `router_tie_stats`, `replays_checked`, and the
sizes. Names are relative to the case directory; every chunk or decode step is a prefix `c{k}/` (files in `c{k}/`).
"absent" means not captured.

Symbols: `r` = rows of the chunk, `nb` = complete blocks, `R` = logit rows. All layer names are `L{n}/...`.

### Model level (every model case, every chunk)

| name | dtype | shape |
|---|---|---|
| `tokens` | i64 | `[r]` |
| `embedding` | bf16 | `[r, 2560]` |
| `streams0` | bf16 | `[r, 10240]` (embedding repeated 4x; = `L0/in`) |
| `L{n}/in`, `L{n}/out` (n = 0..47) | bf16 | `[r, 10240]` |
| `final_hc/normed`, `final_hc/down`, `final_hc/up` | bf16 | `[r, 10240]`, `[r, 320]`, `[r, 10240]` |
| `final_mix` | bf16 | `[r, 2560]` |
| `logits` | bf16 | `[R, 248320]`, `R = min(r, --logit-rows)`, default 1 |
| `logits_rows` | i32 | `[R]` |

There is no final norm: the mixer output goes straight to the BF16 `lm_head` (an ordinary BF16 linear).

### Selected layers (`--capture-layers`; the defaults are in the case list)

| name (under `L{n}/`) | dtype | shape |
|---|---|---|
| `after_ple` (the attn_hc input; differs from `in` only on layer 1), `mid` (the mlp_hc input) | bf16 | `[r, 10240]` |
| `attn_hc/normed`, `mlp_hc/normed` | bf16 | `[r, 10240]` |
| `attn_hc/down`, `mlp_hc/down` (raw linear output, before `/4` and silu) | bf16 | `[r, 320]` |
| `attn_hc/up`, `mlp_hc/up` (raw, before sigmoid) | bf16 | `[r, 10240]` |
| `attn_hc/mixed`, `mlp_hc/mixed` | bf16 | `[r, 2560]` |
| `attn_hc/injection`, `mlp_hc/injection` (`2*sigmoid(block_inject(normed)/4)`) | bf16 | `[r, 4]` |

PLE layer (layer 1), `ple/...`:

| name | dtype | shape |
|---|---|---|
| `ids` (columns 0..7 the 2-gram heads, 8..15 the 3-gram heads) | i64 | `[r, 16]` |
| `rows` (dequantized table rows) | bf16 | `[r, 16, 160]` |
| `emb` | bf16 | `[r, 2560]` |
| `key_proj` | bf16 | `[r, 10240]` |
| `key_normed`, `query_normed` | bf16 | `[r, 4, 2560]` |
| `value` | bf16 | `[r, 2560]` |
| `gate_raw` (the `/sqrt(2560)` sum), `gate` (after signed sqrt, before sigmoid) | bf16 | `[r, 4, 1]` |
| `gated_value`, `gated_normed`, `conv` (silu of the dilated conv), `out` | bf16 | `[r, 10240]` |
| `conv_state` | bf16 | `[1, 10240, 9]` |
| `ngram_context` (`conv_states[2]`) | i64 | `[1, 2]` |

GDN layers, `gdn/...`:

| name | dtype | shape |
|---|---|---|
| `in` (= `attn_hc/mixed`) | bf16 | `[r, 2560]` |
| `qkv_proj`, `conv_out` (silu of the causal conv of qkv; the last `r` rows of the function's output) | bf16 | `[r, 10240]` |
| `z_proj`, `norm_out` (gated RMSNorm output) | bf16 | `[r, 6144]` |
| `b_proj`, `a_proj`, `beta` | bf16 | `[r, 48]` |
| `g` | f32 | `[r, 48]` |
| `core_out` (delta-rule output; chunked for `r > 1`, recurrent for one-row steps) | bf16 | `[r, 48, 128]` |
| `out` | bf16 | `[r, 2560]` |
| `conv_state` | bf16 | `[1, 10240, 4]` |
| `recurrent_state` | f32 | `[1, 48, 128, 128]` |

QSA layers (3, 7, ..., 47), `attn/...`:

| name | dtype | shape |
|---|---|---|
| `in` | bf16 | `[r, 2560]` |
| `index_q` (after `q_layernorm`), `index_q_rope` | bf16 | `[r, 4, 128]` |
| `index_raw_k` (this chunk's raw keys) | bf16 | `[r, 128]` |
| `index_block_keys` (all complete blocks of the visible history at the END of the chunk: pooled, normed, RoPE at the block start; absent while `nb = 0`) | bf16 | `[nb, 128]` |
| `index_scores` (per row over that row's own complete blocks, `-inf` beyond) | f32 | `[r, nb_max]` |
| `block_ids` (selected blocks in `torch.topk` order, -1 padded), `block_ids_sorted` | i32 | `[r, 512]` |
| `token_ids` (upstream's `selected_token_indices`, -1 padded) | i32 | `[r, 2051]` |
| `q_proj` | bf16 | `[r, 12288]` |
| `gate` | bf16 | `[r, 6144]` |
| `q` (after `q_norm` + RoPE) | bf16 | `[r, 24, 256]` |
| `k`, `v` (this chunk, after `k_norm` + RoPE, before the KV codec) | bf16 | `[r, 2, 256]` |
| `core` (before the sigmoid gate), `gated` | bf16 | `[r, 6144]` |
| `out` | bf16 | `[r, 2560]` |
| `kv_k`, `kv_v` (cache after the chunk; after the codec for `nvfp4`) | bf16 | `[1, 2, len, 256]` |
| `indexer_keys` | bf16 | `[1, len, 128]` |

MoE, `moe/...`:

| name | dtype | shape |
|---|---|---|
| `in` (= `mlp_hc/mixed`) | bf16 | `[r, 2560]` |
| `router_logits` | bf16 | `[r, 512]` |
| `topk_ids` (`torch.topk` order) | i32 | `[r, 10]` |
| `topk_probs_raw` (softmax of the FP32 logits, top values, before normalisation) | f32 | `[r, 10]` |
| `topk_weights` (normalised, as upstream returns them) | bf16 | `[r, 10]` |
| `boundary_tie` (10th and 11th probabilities equal: ambiguous selection) | u8 | `[r]` |
| `topk_stable_order` (extra: the stable rule's ids, descending, ties by ascending index) | i32 | `[r, 10]` |
| `shared_out` (before the sigmoid gate), `routed_sum`, `out` | bf16 | `[r, 2560]` |
| `shared_gate` (`sigmoid(shared_expert_gate(x))`) | bf16 | `[r, 1]` |
| `expert_ids` (ascending unique experts hit) | i32 | `[U]` |
| `expert/{e}/tokens` (row indices, upstream's order) | i32 | `[n]` |
| `expert/{e}/out` (down output, unweighted), `expert/{e}/weighted` (times the BF16 routing weight) | bf16 | `[n, 2560]` |

### Pure functions: `ngram_ids`

`multipliers` i64 `[3]`, `head_offsets` i64 `[16]`, `head_vocab_sizes` i64 `[16]` (artifact metadata, asserted equal to
`_build_layer_multipliers(vocab, 3, 0, 1234)` and `_find_nth_prime_after`); `splitmix_in`/`splitmix_out` i64 `[8]` (the
splitmix64 finalizer of upstream, uint64 as two's-complement words). Five token streams `s{i}` with EOS (248044) at assorted
places (at the start, adjacent, last): `s{i}/tokens` i64 `[T]`; `s{i}/oneshot/ids` i64 `[T, 16]` (no cache);
`s{i}/c{k}/ids` i64 `[rows, 16]` and `s{i}/c{k}/ngram_context` i64 `[1, 2]` (cache protocol, several splits including all
one-token chunks). The harness asserts the chunked ids concatenate to the one-shot ids.

## Cases

Model cases run all 48 layers with real weights; layer cases run one decoder layer with real weights on deterministic
synthetic 4-stream rows `N(0, sigma^2)` (sigma = std of 2048 sampled real embedding rows, 0.0091; chunk `k` uses seed
`1000 + k`; the PLE layer gets real token ids). In layer cases the model-level names apply to that layer only (`L{n}/in`,
`L{n}/out`; no `embedding`, `streams0`, `final_*`, `logits`).

| case | content | capture |
|---|---|---|
| `short` | 20-token prompt (one chunk), then 3 greedy decode steps | 0, 1, 2, 3, 47 |
| `eos` | 40 tokens, EOS at 5, 13, 20, 21, 39; chunks [13, 1, 26] | 0, 1, 3 |
| `chunks_a` / `chunks_b` / `chunks_c` | the same 70-token text as [70] / [32, 32, 6] / [1]*8 + [62] (model level only) | none |
| `decode` | 40-token prefill, 12 greedy decode steps, logits every step | none |
| `layer_gdn` | layer 0, chunks [2, 1, 3, 1, 5, 16, 1, 1] (a 2-row first chunk: zero left-padded conv state) | 0 |
| `layer_ple` | layer 1, token ids with EOS at 3, 9, 10, 20, 31, chunks [9, 1, 1, 5, 16] (crosses the 9-row conv history) | 1 |
| `layer_qsa_short` | layer 3, chunks [1, 2, 1, 1, 3, 4] (cumulative 1, 3, 4, 5, 8, 12 tokens; the first complete block at 4) then 6 one-row steps | 3 |
| `layer_qsa_long` | layer 3, 5 x 512 + 140 = 2700 tokens (675 blocks > 512: the budget is exceeded from token 2048 on), then 8 one-row steps | 3 |
| `layer_moe` | layer 0 MoE only: 1, 8 and 64 input rows (the mlp_hc mixed output of synthetic streams) with repeated and scaled rows that route alike | 0 (moe) |
| `ngram_ids` | the n-gram id function (above) | - |

`layer_qsa_long` stores full detail only for chunks 4, 5 (the last two prefill chunks, the first with 2048+ visible tokens),
6 (the first decode row) and the last decode row; the other chunks hold `tokens`, `in`, `out` and
`attn/{index_raw_k, block_ids, token_ids, core, out}`, which keeps the case at 410 MB. `index_scores` exist for the
detailed chunks only.

### Sizes, times, memory

Measured on this machine (RTX 5070 Ti 16 GB, artifact on NVMe, page cache about 36 GB). "first" is the first run (page
cache mostly cold for the experts it touches; `short` reads about 4,800 expert records, 13 GB), "again" a repeat. Seconds are
those the CLI prints (they include building the model, about 2 s); the interpreter and CUDA start add about 3 s.

| case | MB | tensors | files | first s | again s | GPU peak GiB |
|---|---|---|---|---|---|---|
| `short` | 107 | 3043 | 2364 | 15.7 | 7.4 | 6.45 |
| `eos` | 109 | 2394 | 1887 | 13.1 | 7.7 | 6.46 |
| `chunks_a` | 74 | 105 | 57 | 12.7 | 6.6 | 6.46 |
| `chunks_b` | 75 | 315 | 170 | 12.0 | 9.7 | 6.46 |
| `chunks_c` | 78 | 945 | 506 | 12.4 | 11.6 | 6.46 |
| `decode` | 61 | 1365 | 730 | 13.9 | 13.8 | 6.46 |
| `layer_gdn` | 37 | 800 | 661 | 13.4 | 0.4 | 0.53 |
| `layer_ple` | 34 | 866 | 721 | 13.3 | 0.9 | 0.55 |
| `layer_qsa_short` | 6.9 | 1081 | 846 | 4.2 | 0.5 | 0.64 |
| `layer_qsa_long` | 410 | 2953 | 2885 | 5.8 | 4.8 | 1.25 |
| `layer_moe` | 9.1 | 909 | 764 | 2.8 | 0.6 | 0.79 |
| `ngram_ids` | 0.5 | 85 | 85 | 0.1 | 0.1 | 0 |

All cases together are 983 MB. The 6.45 GiB of a model case is the fixed weights (5.5 GB packed/BF16, including the 1.27 GB
embedding and head) plus a 768 MiB LRU of packed expert records and the per-call expansions.

## What the checks showed

* **Finite and shaped**: every tensor of every case is finite (`index_scores` carry the intended `-inf` padding); shapes are
  the catalogue's.
* **The model is the model**: `short` continues "The quick brown fox ... casting" with " long shadows across"; `generate`
  continues "The capital of France is" with " Paris. The capital of Germany".
* **Chunk invariance**: in FP32 on real weights the layers are chunk invariant: relative L2 6.7e-7 (layer 0), 3.3e-6 and
  9.4e-6 (layer 1, PLE), 8.3e-7 (layer 3, QSA) between [70], [32, 32, 6] and [1]*8+[62] (`fp32-chunk-check`; a tiny
  synthetic model is tested the same way). In BF16 the model-level outputs drift by BF16 noise that grows with depth:
  relative L2 of `L{n}/out` between `chunks_a` and `chunks_b` (`chunks_c` is similar: 6.0e-3 at layer 0, 1.0e-1 at layer 47): layer 0 5.9e-3, 1 6.7e-3,
  2 7.1e-3, 3 1.3e-2, 7 2.1e-2, 15 3.2e-2, 23 4.2e-2, 31 8.3e-2, 39 9.8e-2, 47 8.8e-2, `final_mix` 9.3e-2; the last-row logits
  differ by at most 0.75 (logit magnitude about 20), with identical top-5. Expect the engine's deep-layer comparisons to
  need tolerances of this order unless the inputs of the layer are fed from the fixtures.

## Upstream behaviour worth knowing

* **Router ties are common.** The logits are BF16, so softmax probabilities tie often. Across `short`, `eos`, `chunks_*`,
  `decode` (15,600 row-layer routings) the 10th and 11th probabilities were equal in 22% of the rows and the order of
  `torch.topk`'s ids differed from the stable (ascending-index) order in 33%; the selected **set** differed in none of
  them (`router_tie_stats.set_differs_rows` = 0, `topk_nondeterministic_rows` = 0: CUDA `torch.topk` kept the lower index at
  the boundary and repeated runs agreed). Every layer saw boundary ties. The order among ties affects only which top-k
  position a tied expert gets (the weights themselves are equal) and the FP32 summation order of the normalisation.
  `boundary_tie` marks the rows; the deterministic rule is `topk_stable_order`.
* **A one-ulp router-logit difference flips routing.** An engine whose BF16 router logits differ by one unit from the
  reference's can legitimately choose another expert; compare router outputs score-aware (see below).
* **QSA selection only matters past 2048 visible tokens**: while `nb <= 512` every block is selected and attention is dense
  over the visible tokens (`token_ids` = all visible tokens). Beyond that, the selection is purely score-ranked: a query at
  position `p` with `(p+1) % 4 == 0` has no tail, so its **own block (and its own token) is selected only if it ranks in
  the top 512 blocks**; with a tail the last 1 to 3 tokens are always kept. In `layer_qsa_long` (synthetic rows, where a
  query resembles its own key) the own block was selected in all 163 rows of chunks 4 and 5 that have a complete own block
  and every query token was in its own selection (652 of 652), but nothing in the code guarantees it.
* **The indexer pools RAW keys, per visible prefix**: pooling happens before `k_layernorm`, over the first
  `(p+1)//4` complete blocks of the row's own visible tokens; the RoPE of a block uses the position of its first token.
  Block keys of earlier blocks are recomputed for every query row (and never cached), so they never change after the block is
  complete.
* **GDN conv state is 4 inputs**, not 3: the chunk path concatenates the 4 stored inputs with the new ones, convolves, and drops
  the first 4 outputs; a first chunk shorter than 4 is zero padded on the left (same as the causal pad). One-row steps use
  `causal_conv1d_update` and the recurrent delta rule; longer chunks the chunked delta rule (chunk size 64).
* **N-gram ids reset at EOS**: a token `shift` positions back is used only inside the same EOS-delimited segment, otherwise
  EOS (248044) stands in; the cache keeps the last 2 ids, initialised to EOS, so a fresh sequence behaves like one that
  follows an EOS. Ids are `(x0*m0 XOR x1*m1 [XOR x2*m2]) % head_vocab_size + head_offset` in int64.
* **PLE layer index**: the checkpoint's `ple_layer_ids` is `[2]` (1-based); the artifact's `ple.layers` is `[1]`
  (0-based) and the n-gram module's own `ple_layer_index` is 0 (it selects the multipliers and head primes).
* **No final norm** and no residual stream in the usual sense: the four hyper-connection streams are mixed once at the end
  (`hyper_connection_mixer`, no block-inject), RMSNorm weights are `(1 + w)`, norms of 10240 values are grouped per 2560.
* The GDN `A_log`/`dt_bias` and conv weights are BF16 in the artifact; `g = -exp(A_log) * softplus(a + dt_bias)` is FP32.
* Upstream's default (`sdpa`, `grouped_mm`) is not what the fixtures use; a stock `from_pretrained` run would differ from
  the fixtures in the last BF16 bits of the attention and expert outputs.
* `DynamicCache.get_seq_length()` reads the first attention layer, so a layer-isolated run of a GDN layer sees position 0 for
  every chunk (GDN does not use positions).

## Tolerances the engine tests should treat score-aware

| item | why | suggested treatment |
|---|---|---|
| `moe/topk_ids`, `expert_ids`, `expert/*` | BF16 router logits tie and flip by one unit | compare sets; accept a differing expert when the reference probabilities of the swapped experts are within the logit's one-ulp band (the 10th/11th margin from `topk_probs_raw`) or `boundary_tie`; compare weights per expert |
| `moe/topk_weights`, `topk_probs_raw` | the FP32 normalisation sum order depends on the order of tied entries | BF16 rounding tolerance |
| `attn/block_ids`, `token_ids` | a near-tied pair of block scores can swap | compare through `index_scores`: every id the engine selects must score within an ulp-scale band of the reference's budget-th score; ids with scores strictly inside the top set must match |
| `attn/index_scores` | FP32 reductions over 128 BF16 products per head | relative 1e-3 against the BF16 inputs |
| `moe/routed_sum`, `moe/out` | BF16 accumulation order is part of the reference (ascending expert) | exact if the routing matches; otherwise BF16 noise |
| `gdn/core_out`, `gdn/recurrent_state` | chunked (fla-style) vs sequential delta rule | FP32 state tolerance about 1e-5 relative, BF16 output tolerance |
| deep `L{n}/out`, logits | BF16 noise grows with depth (above) | feed each layer the fixture's `in`; for the whole model compare top-k logits, not raw values |
| anything with `--kv-format nvfp4` | codec error is part of the reference but the engine keeps the rotated domain | compare against the BF16 cache's results with the 4-bit tolerance, or the codec bit-exactly |

## Tests

`tools/test_flash_next_reference.py` (27 tests, about 3 s): NVFP4 expansion vs `dequantize_nvfp4_rows` (bit-exact, CPU
and GPU, per-row global scales), `ExactNvfp4Linear` semantics, the KV codec (butterfly vs the H256 definition, tie
rounding, scale clamps, agreement with the float64 reference semantics, **bit-exact against the CUDA codec**), the recorder
format, the stable selection rule, the n-gram ids against a scalar implementation and the cache protocol for several
splits, the table assertions, `LazyExperts` against upstream's experts (fp32 bit-equal; BF16 accumulation order), the
recording subclasses (replay passes on a tiny model; a deliberately broken subclass fails loudly), fp32 chunk invariance and
cache-state agreement of the harness on a tiny model, the `nvfp4` KV path, and, when the artifact and a GPU exist, the
config cross-check, CRC-32s, expert and n-gram row expansion against the repo's dequantizer, and byte-determinism of a layer
case.
