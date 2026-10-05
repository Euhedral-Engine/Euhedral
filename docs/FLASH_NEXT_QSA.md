# Flash-Next sparse attention (QSA)

Execution of the sparse-attention layers of Qwen3.8-Flash-Next (HF `qwen4_exp`): 12 of its 48 layers (3, 7, ..., 47) run
`Qwen4ExpTextAttention` with its `Qwen4ExpTextQSAIndexer` (upstream revision `f5ab85619d989359ef47b5efed8a91a15045627b`,
`modular_qwen4_exp.py`). This note records what the code computes, the selection rule, the kernels, the numerics and what was
measured. Sources: `native/src/qwen4/qsa_{norm,index,attention}.cuh`, `core/.../qwen4/Qwen4Qsa{Ops,State,Layer}.java`.

## What upstream computes

Model facts: hidden 2,560, 24 query heads, 2 KV heads, head dimension 256. `q_proj` produces `[rows, 12288]`, per head
`[q (256) | gate (256)]`; `q_norm` and `k_norm` are RMSNorms over each head with `1 + weight`, shared by all heads. RoPE rotates the
first 64 of 256 values (`rotate_half` pairing `i` with `i + 32`), theta 1e7, and the three M-RoPE streams are identical for text.
The output is `o_proj(attention * sigmoid(gate))`. `q/k/v/o` are NVFP4; the indexer projection and the indexer norms are BF16.

The indexer (per layer): 4 query heads and one key head of 128 values from `index_qk_proj` (`[640, 2560]`), blocks of
`compress_ratio = 4` tokens, `budget = 2048` tokens = `block_topk = 512` blocks.

- The indexer queries (4 x 128 per row) get `q_layernorm` and RoPE (the same 64-value cos/sin table: the first 64 of 128 values).
- The raw token keys (128 per row) are neither normalized nor rotated.
- A query at position `p` sees the tokens `0 .. p`. It has `nb = (p + 1) / 4` complete blocks. The key of block `j` is the mean (FP32,
  rounded to BF16) of its four raw keys, then `k_layernorm`, then RoPE **at the position of the block's first token** `4 j`.
  The query is rotated at its own position, so a score depends on the offset `p - 4 j`.
- `score(j) = sum over the 4 heads of relu(q_h . key_j) / sqrt(128)` in FP32.
- The row keeps `min(512, nb)` blocks of highest score. Its attention runs over the tokens of those blocks plus the tail,
  the visible tokens of the incomplete block: `4 nb .. p`.
- Attention: scale `1/sqrt(256)`, softmax in FP32. The causal mask and the selection mask are both applied.

Consequences worth knowing, all reproduced and none "fixed":

- Until the history exceeds 2,048 tokens every block is selected and QSA is dense causal attention. From position 2,051 on a row
  chooses (`nb = 513`).
- When `p + 1` is a multiple of 4 the tail is empty and the query's own token is present only if its own block was selected, so a
  row can fail to attend to itself. In the 165 such rows of the long fixture (synthetic streams) it never happened; the newest
  complete block was left out in 36 of 509 and 24 of 140 selecting rows of the last two prefill chunks.
- The scores are not softmaxed, and `relu` makes exact zeros common: 2.5% of the selecting rows of the long fixture (16 of
  649) have a 512th score of exactly 0 with 3 to 16 blocks tied at it, so the tie rule decides which of them the row attends.
- `torch.topk` leaves the order and the tie choice unspecified. In all 16 rows above the fixture's choice is the **lowest ids**
  among the tied blocks, which is the rule below.
- Block keys do not depend on the query. Upstream recomputes every pooled key for every query row; this implementation computes
  a key once, when its block completes.

## Selection rule

Per row, `k = min(512, nb)`. The selected blocks are the `k` highest scores; blocks scoring exactly the `k`-th score are taken
**lowest id first**. The ids are written **ascending** with their count (`counts[row]`), so attention reads the cache in order and
the result does not depend on `topk`'s order. The scores are FP32 sums of exact BF16 products on tensor cores; the operator test
holds them to 2e-5 of the row's largest score against a double-precision evaluation, and equal inputs give equal bits, so exact
ties are exact ties. Against upstream's own FP32 scores a block may swap with a neighbour of nearly equal score: see the measurements.

## Data flow

```
x [rows, 2560]
  q_proj, k_proj, v_proj (NVFP4)  index_qk_proj (BF16) -> [rows, 640] = 4 q heads | raw key
  q: norm + RoPE in place         indexer q: norm + RoPE in place
  k: norm + RoPE -> NVFP4 pages   raw keys -> pooled blocks -> k_layernorm + RoPE -> block keys (kept)
  v -> NVFP4 pages                incomplete block -> raw tail (kept, double buffered)
                                  scores (row tile x blocks) -> top-k -> ids, counts
  attention over the NVFP4 pages (selected blocks + tail), sigmoid gate -> gated
  o_proj (NVFP4) -> out [rows, 2560]
```

`Qwen4QsaLayer.run` executes this for a chunk of rows starting at the state's committed length and returns the attention block's
result before the hyper-connection injection. It reserves the cache pages and marks the chunk submitted; the caller commits
(`Qwen4QsaState.commit`) after the quantum retired or discards. The weights are device addresses (NVFP4 projections with their byte
sizes; the caller resolves staging); the scratch is the caller's (`scratchBytes(rows)`, `scratch(base, rows)`).

**Chunk independence.** A block key is final when its block completes, from raw keys that are either in the chunk or in the
carried tail (the up to 3 raw keys of the incomplete block). Row `p` of a chunk uses exactly the blocks `j < (p + 1) / 4`, including
blocks that complete inside the chunk, so any chunking gives the same selection and the same attention up to floating-point
summation order of the attention (the number of key splits depends on the row count). With row-exact projections (below) chunks of
1 to 5 rows, 31, 512 and one shot of 2,300 tokens agree to a relative RMS of 1.6e-3 or better (the BF16 rounding of the output).

**State** (`Qwen4QsaState`, one per layer and sequence): the KV pages (an `AttentionKvState`, pages of 256 tokens allocated as the
sequence grows), the pooled block keys (`[maxBlocks][128]` BF16, allocated for the maximum context), and two raw tails
(`[3][128]` BF16): a chunk reads the committed tail and writes the other, `commit` swaps them, so a discarded chunk leaves the
committed state intact. Per token: 576 bytes of KV (2 heads x K and V x 144) and 64 bytes of block keys (256 per block), 640 in
all; at 262,144 tokens 160 MiB per layer (144 + 16), 1.875 GiB for the 12 layers.

## Kernels

All in the Flash-Next module (`native/src/qwen4/kernels.cu`, table `native/src/host/qwen4_ops.c`, enum `Qwen4Kernel`), launched
through `Qwen4QsaOps`; arguments are verified against the compiled signatures at launch.

| kernel | geometry | what |
|---|---|---|
| `head_norm_rope_bf16` | a warp per (row, head), strided in and out, in place allowed | per-head RMSNorm with one shared weight, then RoPE of the first 64 values; BF16 after every operation |
| `qsa_pool_keys_bf16` | a warp per completed block | mean of the four raw keys (chunk rows or the tail) in FP32, to BF16 |
| `qsa_tail_bf16` | one CTA | raw keys of the incomplete block after the chunk into the other tail |
| `qsa_scores` | CTA = 32 rows x 64 blocks | BF16 tensor cores (m16n8k16, FP32 accumulation): tile row = 4 rows x 4 heads, keys staged once per CTA and read by ldmatrix; relu, head sum by shuffles, / sqrt(128) |
| `qsa_select` | CTA of 1024 per row | radix selection of the k-th score (four 8-bit passes, warp-level digit search), then an in-index-order compaction with one packed two-flag scan per 1024 blocks: lowest ids among ties, ids ascending; rows with `nb <= 512` list their blocks and read no score |
| `qsa_attention` | CTA of 3 warps per (row, KV head, key split) | flash attention over 16-key tiles of the virtual key list (selected blocks, then tail); the 12 query heads of a KV head are the m16 dimension (4 padding rows). Warp 0 scores (K fragments built straight from the raw NVFP4 codes), warp 1 expands V rows to FP16 tiles, warp 2 does the softmax and accumulates the output; handoffs by mbarriers, raw tiles by cp.async (two in flight). Splits write partials, one split writes `core` and `gated` |
| `qsa_merge` | a warp per (row, head) | combines the splits, rotates back, writes `core` and `gated` |
| `qsa_kv_append` | a warp per (row, KV head) | the dense engine's NVFP4 quantizer (`nvfp4kv::quantize_row`) into the pages at `start + row` |

The key list of a row is virtual: block id `b_i` gives the keys `4 b_i .. 4 b_i + 3`, the tail follows. A tile of 16 keys needs 4
ids; there is no 2,051-wide token list and no dense matrix. Rows with every block selected pass no ids (the identity).

Scores are tiled over rows so that the scratch stays bounded: the FP32 scores of a tile of rows times the blocks they see fit
`scoreBytes` (default 16 MiB; one row of 65,536 blocks is 256 KiB), so a 512-row chunk at 262K tokens scores 64 rows at a time.

## Numerics

- Norms, RoPE and the pooled keys round to BF16 after every operation as PyTorch does. cos and sin are computed from
  `inv_freq = 1 / theta^(2 i / 64)` in FP32, the phase `position * inv_freq` rounded to FP32, `cosf`/`sinf`, rounded to BF16.
  Checked against torch for 765,216 table entries (23,913 positions up to 262,143, all 32 frequencies): identical to torch's CPU
  computation; 3 cosines and 4 sines differ from torch's CUDA results at BF16 rounding boundaries.
- The cache is the dense engine's NVFP4 codec (Hadamard-rotated rows, E2M1 codes, E4M3 scale per 16): queries are rotated by the
  same transform, the output is rotated back. The decoded pages equal the fixtures' codec emulation to 1.7e-3 relative RMS, the
  BF16 rounding the fixtures apply to the decoded cache.
- Scores `S = Q K^T` and `O += P V` use FP16 operands with FP32 accumulation: the rotated queries and the probabilities round to
  FP16 (K and V values are exact in FP16), the softmax is FP32 (`__expf`). Against a double-precision evaluation over the decoded
  cache the output differs by 1.65e-3 to 1.70e-3 relative RMS, the BF16 rounding of the result itself.
- Upstream's eager attention rounds its scores and probabilities to BF16 and keeps BF16 cache values: on its own cache its output
  differs from a double-precision evaluation by 1.7e-3 to 2.4e-3 relative RMS, and an exact evaluation over the unrounded NVFP4
  cache differs from it by 3.0e-3 (short fixture). That is the floor against this reference.

## Validation

Tests are `Qwen4Qsa*CudaIntegrationTest` (`./gradlew :core:cudaIntegrationTest --tests '*Qwen4Qsa*'`; a Blackwell GPU, and for the
layer tests the artifact and the fixtures). The references are in `Qwen4QsaReference`: a port of the upstream per-query indexer
loop and attention, written with PyTorch's rounding.

- operators on synthetic data: norm and RoPE (strided, in place, positions to 16,000,000), pooling across chunkings of 97 tokens (1,
  2, 3, 4, 5 rows and mixtures; blocks completing across chunk boundaries; the tail), scores and selection (first block, under, at
  and over the budget, 65,500 blocks, exact ties, tiling), attention against the decoded cache (dense and selected, key splits of
  1, 8 and 64 including empty splits, rows that drop their own block);
- the whole layer on the real layer-3 weights against `layer_qsa_short` and `layer_qsa_long` fixtures (below), against
  chunkings of a 2,300-token random sequence, and at the end of the maximum context (261,632 tokens of random rows through the
  layer, then a chunk of 8 rows: selection and attention against references computed from the device's own block keys and cache).

The fixtures in `~/fixtures/flash-next/layer_qsa_*` were recorded with the BF16 cache; the engine's cache is NVFP4, which moves the
layer output by about 15%, so the tests prefer fixtures recorded with `--kv-format nvfp4` (`~/fixtures/flash-next-qsa-nvfp4`, or
the root named by `-Deuhedral.qwen4.fixtures`). The harness's light chunks of the long case omit the attention input, which the
test needs to replay the layer from the first chunk: the fixtures used here were made by adding `L{n}/attn/in`,
`L{n}/attn/index_scores` and `L{n}/attn/block_ids_sorted` to `flash_next_reference.cases.LIGHT_QSA` from a three-line script that
then calls `run_case("layer_qsa_long", kv_format="nvfp4")`; the harness itself is unchanged. The linears run row-exact
(`selectRowExact`) so a row's projections do not depend on its chunk.

### Tolerances achieved

Largest over the chunks of each fixture of the relative RMS error against the NVFP4-cache fixtures (all rows of a chunk):

| tensor | short | long |
|---|---|---|
| q (after norm + RoPE) | 5.8e-5 | 1.0e-4 |
| gate | 3.8e-5 | 6.3e-5 |
| k (after norm + RoPE) | 0 | 4.5e-5 |
| v | 5.8e-6 | 3.9e-5 |
| indexer q after RoPE | 3.8e-4 | 1.7e-4 |
| indexer raw k | 1.8e-5 | 1.0e-4 |
| pooled block keys | 0 | 2.3e-4 |
| cache pages vs fixture cache | 1.7e-3 | 2.1e-3 |
| core (attention before the gate) | 4.1e-3 | 4.3e-3 |
| gated | 4.2e-3 | 3.8e-3 |
| out (after `o_proj`) | 5.5e-3 | 6.6e-3 |

The core error is the reference's own rounding (above), not the kernel's: against double precision on the device's own cache the
core differs by 1.7e-3. Selection: in the two selecting prefill chunks (512 and 140 rows) 7 and 8 blocks in total differ from
the reference's top-512; a swapped block's score is within 6e-4 of the row's highest score from the 512th score (checked score
aware against the fixture's `index_scores`, tolerance 2e-3); in the 8 decode rows none differ. At the end of the maximum context the selection
equals the reference exactly (score gap 0).

## Measurements

RTX 5070 Ti, CUDA 13.1, layer 3 of the real artifact. The history is built by running the layer over random rows; times are the
average of repeated runs on the stream, in milliseconds (`Qwen4QsaTimingCudaIntegrationTest`, `EUHEDRAL_QWEN4_TIMING=1`).
"block" is one `run` including four projections, the output projection and the retirement wait; the columns are its parts.

| history | rows | block | projections | select (scores, top-k) | attention | o_proj |
|---|---|---|---|---|---|---|
| 1K | 1 | 0.090 | 0.034 | - | 0.015 | 0.012 |
| 1K | 64 | 0.257 | 0.077 | - | 0.080 | 0.027 |
| 1K | 512 | 1.227 | 0.347 | - | 0.626 | 0.101 |
| 8K | 1 | 0.109 | 0.023 | 0.008 (0.004, 0.004) | 0.023 | 0.012 |
| 8K | 64 | 0.300 | 0.077 | 0.012 (0.007, 0.005) | 0.138 | 0.041 |
| 8K | 512 | 1.671 | 0.336 | 0.058 (0.023, 0.035) | 0.998 | 0.086 |
| 32K | 1 | 0.101 | 0.023 | 0.011 (0.004, 0.008) | 0.023 | 0.012 |
| 32K | 64 | 0.373 | 0.076 | 0.023 (0.015, 0.009) | 0.153 | 0.027 |
| 32K | 512 | 1.810 | 0.349 | 0.150 (0.156, 0.119) | 1.013 | 0.085 |
| 128K | 1 | 0.123 | 0.022 | 0.033 (0.008, 0.025) | 0.023 | 0.012 |
| 128K | 64 | 0.424 | 0.076 | 0.073 (0.047, 0.026) | 0.158 | 0.026 |
| 128K | 512 | 2.415 | 0.313 | 0.681 (0.449, 0.260) | 1.150 | 0.085 |
| 262K | 1 | 0.164 | 0.022 | 0.066 (0.015, 0.051) | 0.023 | 0.012 |
| 262K | 64 | 0.504 | 0.076 | 0.140 (0.089, 0.051) | 0.165 | 0.026 |
| 262K | 512 | 3.089 | 0.343 | 1.204 (0.714, 0.412) | 1.234 | 0.100 |

The "select" times are the whole selection of the chunk; scores and top-k are measured on one tile and scaled by the number of
tiles. A 512-row chunk at 262K tokens scores 64 rows at a time (8 tiles). At 1K history all rows still see at most 512 blocks and
selection is skipped (the ids are the identity).

Memory: scratch `scratchBytes(rows)` (default 16 MiB of scores included): 17.6 MiB for 1 row, 22.4 for 8, 24.7 for 64, 27.3 for
128, 43.2 for 512 (the rest at 512 rows: q/gate 12 MiB, gated 6 MiB, ids 1 MiB, k/v/indexer about 2 MiB). A layer with its state
at 262,144 tokens, scratch for 512 rows and its weights peaked at 242 MiB of device allocations.

## Findings that shaped the implementation

- Attention: 12 query heads per KV head fit one m16 tile (4 padding rows), so each key is read once for all of them. A single warp
  per unit held the queries (64 registers) and the 16 x 256 output (128 registers) while it also expanded K and V: 255 registers
  with spills, and about half of the kernel time was the expansion (removing it halved the time). Three
  warps per unit with small live sets (165 registers, no spills) halved it, and a smaller shared-memory footprint (three CTAs
  per SM instead of two) took another quarter. Building K's B fragments straight from the codes removed the K tile from
  shared memory; V still needs one for the transposed fragments.
- The radix digit search by one thread and two block scans per 1024 blocks cost 126 us for one row at 65,536 blocks; the
  warp-level search and the packed scan took it to 51 us.
- The block scores with global B loads (4 bytes per lane) cost 1.16 ms for 512 rows at 262K; staging the keys once per CTA took
  it to 0.71 ms.

## Known gaps

- Positions and lengths are kernel arguments (`start`, block counts, grids), so a decode step is not replayable from a CUDA graph
  as it is; the dense engine keeps its position in device memory for that.
- The rotary dimension is fixed at 64 values (the lane owns both halves of a pair), the indexer geometry at 4 heads of 128 over
  blocks of 4, the heads at 256 with at most 16 query heads per KV head; the layer constructor rejects anything else.
- At 128K and beyond a 512-row chunk spends 0.7 to 1.2 ms in selection (scores then top-k, one CTA per row for the top-k); a larger
  scores scratch or a multi-CTA top-k would shorten it.
- Selection ties follow the lowest-id rule, which matched upstream's `torch.topk` in every observed tie (16 rows); `topk` does not
  promise it.
- Rolling a committed sequence back to an earlier length is not supported (the raw tail only holds the incomplete block);
  `Qwen4QsaState.reset` and discarding the chunk in flight are.
