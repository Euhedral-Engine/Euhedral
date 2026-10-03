# Long-context decode and verify attention

The GQA decode attention kernel serves one query-head group (6 heads) per KV head over an NVFP4 cache (D256, 144-byte rows:
128 code bytes and 16 E4M3 scales; 256-token pages). It runs from 2048 keys for single-row decode and for the verifier's row twin (below 2048 keys the per-head
decode kernel runs; a group of more than 8 query heads also keeps the per-head kernel). Sources: `native/src/attention/nvfp4_decode_gqa.cuh` (decode, row twin), `nvfp4_attention.cuh` (merge), `nvfp4_pipe.cuh` (shared
pipeline helpers). Hardware for every measurement: RTX 5070 Ti (sm_120, 70 SMs), driver 615.71, desktop session running.

## Design

One CTA per (KV head, key split): 4 KV heads times at most 64 splits of 32 or more keys, so 256 CTAs. The CTA is three warps with
the arithmetic order of a single-warp loop (the single-warp kernels are kept as test controls in `reference_decode_gqa.cuh`,
and the exact-numerics oracle runs the `_exact` per-head twins instead of this kernel):

- **scores warp:** the K tile arrives by a two-stage cp.async ring; A fragments of the QK product are built straight from the raw
  144-byte rows through a lookup (no FP16 staging); the QK mma chain; scores to shared memory;
- **values warp:** the V tile is prefetched into registers and expanded to padded FP16 rows in shared memory;
- **consumer warp:** online softmax, P as hi and lo FP16 parts, the PV mma chain, the partial write.

Tiles of 16 keys hand off in order through shared-memory mbarriers. The CTA uses 24.3 KiB of shared memory and 158 registers, so
4 CTAs per SM are resident and the 256 CTAs of a long context run in one wave. Named barriers would cap the SM at one resident CTA
(nine ids against sixteen per SM), which is why mbarriers carry the hand-offs.

**Exact-product lookup.** A 256-entry table maps a byte of two E2M1 codes to an FP16 pair; one FP16 multiply by the group scale
(a hardware `cvt.rn.f16x2.e4m3x2` of the scale byte) follows. Every product (at most five significant bits, 2^-10 to 2688) is exact in
FP16, so the result equals the FP32 product rounded to FP16 for every legal cache scale.

**Merge.** One warp per query head. Lanes gather the split statistics (the maximum is order-free) and the per-split rows are loaded
eight at a time; every sum still adds the splits in order.

**Equality.** Partials and merged outputs are bit for bit those of the single-warp kernels and the one-lane merge, for one row and for
the row twin (`test_gqa_decode_warp_specialized_equals_single_warp_reference_bitwise`: lengths 1 to 16385, empty splits, partial last
tiles, page edges, every finite E4M3 scale byte).

## Performance

Decode plus merge, per layer, cold KV (several layer-sized caches rotated past L2); the 3-row column is the verifier's row twin.

| Keys | 1 row | 3 rows |
|---|---|---|
| 4K | 21 us | 37 us |
| 16K | 38 us | 84 us |
| 32K | 67 us | 151 us |
| 64K | 116 us | 308 us |

At 64K the one-row decode kernel streams the 72 MiB cache at 107 us, 82% of the 855 GB/s DRAM rate. Of the 38 us at 16K, the decode
kernel takes 31 us and the merge 7 us.

In the model (the `q3` artifact), decode runs at 61.7 tok/s at a 4K context, 60.5 at 16K and 59.0 at 32K. MTP2 decode (3-row
verify, chat corpus) runs at 111 tok/s at 16K and 101 at 32K.

## Rejected

- **A deeper cp.async ring, or finer key splits.** The loads alone take 28 us of the 16K one-row kernel; the time is the phases of one
  warp running one after another at about 0.1 instructions per cycle. A cp.async ring on the single-warp kernel left it at 131 us;
  128 and 256 splits ran at 78 and 76 us because every CTA repeats the query rotation and the merge grows. A 4-stage ring takes 31 KiB
  of shared memory, drops the SM to 3 resident CTAs and runs slower than 2 stages.
- **Named barriers for the hand-offs.** Nine ids cap the SM at one resident CTA.
- **Fused verifier rows** (expanding each K and V tile once for all rows). Removing all V expansion and all K and V traffic for rows 1 and
  2 of the 3-row twin saves 3% at 16K and 19% at 64K, an upper bound; a fused kernel also needs three accumulator sets of 64 registers,
  one consumer warp per 8 query columns and the producers, about 7 warps at 160 registers, which leaves fewer than two CTAs per SM.
- **Shared query rotation** (rotate once per KV head, not per split). Skipping the Hadamard entirely saves 2.5% at 16K and 0.7% at 64K.
- **Merge inside the decode kernel** (the last CTA merges). The merge is 7 us of 38 us at 16K and fusing removes the launch gap, not the work.
