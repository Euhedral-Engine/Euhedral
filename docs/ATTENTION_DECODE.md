# Long-context decode and verify attention

Item 1 of docs/nvidia/opportunities.md: at 16K, one-row decode attention took 2.83 ms per token (Q3,
16 full-attention layers) while the NVFP4 KV it reads (288 MiB) needs 0.35 ms at the measured 855 GB/s.
This log covers the investigation, what was kept and what was not, and the measurements.

Setup: RTX 5070 Ti (sm_120, 70 SMs), driver 615.71, desktop session running. Head geometry of the
model: 24 query heads, 4 KV heads (group 6), D256, 144-byte rows (128 code bytes + 16 E4M3 scales), 256-token
pages. Kernels: `native/src/attention/nvfp4_decode_gqa.cuh` (decode, row twin) and `nvfp4_attention.cuh`
(merge).

## Why the kernel was slow

The GQA decode kernel is one single-warp CTA per (KV head, 32-key split), at most 4 x 64 = 256 CTAs: 3.7 warps
per SM, one per scheduler at most. The doc item blamed bytes in flight. A skip-toggle ladder on a copy of the
kernel (16K keys, cold KV, decode kernel only; each phase removed in turn) says otherwise:

| Variant | Time |
|---|---|
| production | 140 µs |
| + cp.async ring (K and V rows prefetched two tiles ahead), same arithmetic | 131 µs |
| + lookup-table dequantization (below) | 96 µs |
| + full-tile path with all loads hoisted | 83 µs |
| no loads at all (compute only) | 66 µs |
| loads only (no expand, QK, softmax or PV) | 28 µs |

- The cp.async ring alone barely helped: memory was not the limiter. The phases added up almost exactly
  (loads 28 + expand 24 + QK 11 + softmax 9 + PV 12 = 83): a lone warp per scheduler runs them one after another
  with about 0.1 instructions per cycle (1435 instructions per 16-key tile in 5-6 µs).
- The loads-only floor of 28 µs is 650 GB/s on 18 MiB, so DRAM was never the 8x gap; issue latency was.
- Smaller key splits (128, 256) gave only 78 and 76 µs: each CTA re-rotates the query heads (Hadamard) and the
  merge grows. Rejected.
- More resident CTAs mattered more than depth: a 4-stage ring (31 KiB of shared memory) fell to 3 CTAs per
  SM and was slower than 2 stages. Named barriers (9 ids used) cap an SM at 1 resident CTA; a first
  warp-specialized version ran 2.4x slower than its single-warp twin for that reason alone.

## What was kept: three warps per CTA, bit-identical

Each (KV head, split) CTA now has three warps with the same arithmetic order as before:
- **scores:** K tile by cp.async (two stages), A fragments built directly from the raw 144-byte rows (no FP16
  staging), the QK mma chain, scores to shared memory;
- **values:** V tile by register prefetch, expanded to padded FP16 rows in shared memory;
- **consumer:** online softmax, P hi/lo, PV mma chain, partial write.

Tiles hand off in order through shared-memory mbarriers. 24.3 KiB of shared memory per CTA keeps 4 CTAs per
SM resident, so all 256 CTAs of a long context are in one wave.

**Exact-product lookup.** Dequantization was software: E2M1 decode, FP32 multiply by the scale, convert to
FP16. A 256-entry table maps a byte (two E2M1 codes) to an FP16 pair and one `hmul2` applies the group scale,
which is a hardware `cvt.rn.f16x2.e4m3x2` of the scale byte. Every product (at most five significant bits,
2^-10 to 2688) is exact in FP16, so this equals the old result bit for bit for every legal cache scale.

**Merge.** One warp per query head summed the 64 splits with a dependent load chain: 17 µs per layer at 16K.
Lanes now gather the split statistics (max is order-free), and the rows are loaded eight at a time; every sum
still adds in split order. 7 µs, bit-identical.

**Proof.** `test_gqa_decode_warp_specialized_equals_single_warp_reference_bitwise`
(native/tests/test_attention_nvfp4.py) compares partials and merged outputs with the previous kernels, kept
as test-only controls in `native/src/attention/reference_decode_gqa.cuh`. It covers lengths 1 to 16385 (empty
splits, partial last tiles, page edges), one row and the row twin, realistic scales and every finite E4M3 scale
byte. Two deliberate mutations (a softmax scale, a LUT sign bit) both fail it.

## Operator bench

`synthetic/attention_decode.py` in the performance skill: cold KV (several layer-sized caches rotated past
L2), production against candidates, partials compared bitwise first. Decode + merge, per layer:

| Keys | Rows | Before | After | Speedup |
|---|---|---|---|---|
| 4K | 1 | 54 µs | 21 µs | 2.6x |
| 4K | 3 | 100 µs | 37 µs | 2.7x |
| 16K | 1 | 156 µs | 38 µs | 4.1x |
| 16K | 3 | 332 µs | 84 µs | 3.9x |
| 32K | 1 | 290 µs | 67 µs | 4.3x |
| 32K | 3 | 639 µs | 151 µs | 4.2x |
| 64K | 1 | 568 µs | 116 µs | 4.9x |
| 64K | 3 | 1606 µs | 308 µs | 5.2x |

At 64K one row reaches about 82% of the 855 GB/s DRAM rate (decode kernel 107 µs for 72 MiB).

## In-model gate

Control: `main` at `ef33b25`; plain decode, Q3 compact artifact, 128 generated tokens, warmup 2 and 3
iterations (4K-16K: 5 paired forks, stopped before the sixth because the result was clear; 32K: one fork,
warmup 1, 2 iterations).

| Scenario | Control tok/s | Candidate tok/s | Change | Forks ahead |
|---|---|---|---|---|
| decode 64 | 62.99 | 63.01 | +0.0% | 3/5 |
| decode 1K | 61.63 | 61.86 | +0.4% | 5/5 |
| decode 4K | 59.68 | 61.78 | +3.5% | 5/5 |
| decode 16K | 54.33 | 60.52 | **+11.4%** | 5/5 |
| decode 32K | 48.55 | 58.90 | **+21.3%** | 1/1 |

TTFT is unchanged (-0.0% at every length). At 64 and 1K the GQA kernel is not used (it starts at 2048 keys); the
0.4% at 1K is the faster merge shared with the per-head kernel.

Validation: the bitwise test above, the whole native suite (116 tests), `spotlessCheck test nativeVerify
nativePackage :api:bootJar`, and the in-model `SpeculativeVerifyCudaIntegrationTest` and
`SpeculativeDecodeCudaIntegrationTest` (row-exact verification equals one-row decode, outputs equal greedy).

## Not done

- **Fused verifier rows.** The row twin is about 4x faster per row, but each of the M rows still expands the
  same KV tiles. A kernel that expands each K and V tile once for all rows (columns 6 M of the mma N dimension,
  one consumer warp per 8 columns, falling back to the row twin when the rows' split spans differ, about 3% of
  launches) would take 3 rows at 16K from 84 toward about 45 µs per layer. Not built; the MTP verify gate was not
  run either (the verify rows share the kernel and are covered by the exactness tests above).
- **Merge into the decode kernel** (last CTA merges): about 5 µs of the 38 µs at 16K.
- **Query rotation** is repeated by all 64 splits of a KV head (about 13 µs of fixed cost per CTA at 16K).
