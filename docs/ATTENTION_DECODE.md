# Long-context decode and verify attention

The GQA decode attention kernel serves one query-head group (6 heads) per KV head over an NVFP4 cache (D256, 144-byte rows:
128 code bytes and 16 E4M3 scales; 256-token pages). It runs from 2048 keys for single-row decode and for the verifier's row twin (below 2048 keys the per-head
decode kernel runs; a group of more than 8 query heads also keeps the per-head kernel). Sources: `native/src/attention/nvfp4_decode_gqa.cuh` (decode, row twin), `nvfp4_decode_gqa_fused.cuh` (fused verifier rows),
`nvfp4_attention.cuh` (merge), `nvfp4_pipe.cuh` (shared pipeline helpers). Hardware for every measurement: RTX 5070 Ti (sm_120, 70 SMs), driver 615.71, desktop session running.

## Cache formats

A sequence keeps its attention caches in one of two formats, chosen per engine (`InferenceConfig.kvCache`,
`euhedral.inference.kv-cache`; unset, the artifact's, NVFP4):

- **NVFP4** (`nvfp4_kv.cuh`): 144-byte head rows of the H256-rotated keys and values, 4-bit codes on the INT4 midrise grid
  (magnitudes 0.5 to 7.5) with an E4M3 scale per 16 values (docs/QUALITY.md, KV cache). Queries are rotated into the same basis
  and outputs back out of it.
- **BF16** (`bf16_kv.cuh`): 512-byte head rows, the keys and values as computed, unrotated.

Both use the same 256-token pages and page tables. Every attention kernel (append, 32-row and FlashAttention-2 prefill, per-head and
GQA decode, merge, the verifier's row twins, the `_exact` twins) is one body templated on a cache-format policy (`Nvfp4Kv`,
`Bf16Kv` in `nvfp4_attention.cuh`), instantiated as `euhedral_attention_*_nvfp4` and `euhedral_attention_*_bf16`; the host picks
the set by the cache's format (`euhedral_cuda_attention_causal_bf16`, `euhedral_cuda_attention_kv_append_bf16`) and the routes and
split counts are the same for both, so every verification row is one-row decode bit for bit in either format. The per-head
kernels read BF16 rows as FP32. The tensor-core kernels widen them to FP16 tiles: exact from 2^-14 to 65504, saturated to
+-65504 beyond, so the tiles stay finite. A batched quantum's sequences share one format, and a sequence's capture key carries it.

Against a float64 oracle of gated causal GQA attention over the cached BF16 values, every route adds at most 1.3e-5 relative RMS
error beyond the BF16 rounding of its output (`native/tests/test_attention_bf16.py`: prefill up to 512 rows and 4096 keys, decode up
to 32768 keys).

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

**Exact-product lookup.** A 256-entry table maps a byte of two cache codes (INT4 midrise magnitudes m + 0.5) to an FP16 pair; one
FP16 multiply by the group scale (a hardware `cvt.rn.f16x2.e4m3x2` of the scale byte) follows. Every product (at most eight significant
bits, 2^-10 to 3360) is exact in FP16, so the result equals the FP32 product rounded to FP16 for every legal cache scale.

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

## Fused verifier rows

A verification of R rows over an NVFP4 cache runs one CTA per (KV head, key split) for all rows at once
(`euhedral_attention_decode_gqa_nvfp4_fused<T>`): the KV head's R x 6 query heads are the columns of T mma tiles of 8, and each
16-key tile of K and V is loaded and expanded once for all of them. Warps: T score warps (one query tile each, the QK chain over
the shared raw K tile), one loader warp (raw K and V tiles by a four-stage cp.async ring, V expanded once to FP16 rows) and T
consumer warps (online softmax, P hi/lo and the PV chain of one query tile). Up to T = 4 (5 rows) two CTAs fit per SM (96
registers, 43 KiB of shared memory); T = 5 and 6 run one.

Every query column is computed by the operations of the one-row kernel at its row's length, in the same order, so the partials are
bit for bit the row twin's and verification stays exact (`test_fused_verifier_rows_equal_row_twins_bitwise`, and
`SpeculativeVerifyCudaIntegrationTest` at 4100 keys with 5 rows and 9000 keys with 4). That needs the rows to share one split
partition: a split spans whole 16-key tiles (`split_span`), so the partition of 64 splits changes only where the length crosses a
multiple of 1024 keys. The fused kernel serves the rows that share the last row's partition; the row twin, launched after it,
serves the rest (in the rare step that crosses a multiple of 1024) and returns at once for the others.

Decode partials per layer, cold KV, 5 rows (MTP depth 4), real head geometry:

| Keys | 2K | 4K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| Fused, 5 rows | 31.9 us | 41.5 us | 85.8 us | 147.2 us | 272.5 us | 505.0 us |

The fused kernel is held back by the tensor pipe (71% active at 64K under Nsight Compute): every product is two mma.sync
(hi and lo FP16 parts at FP32 accumulation, half rate on this card). Its advantage depends on how fully the rows fill the 8-column
tiles, so the host launches it from a measured length per row count: 2 rows from 49152 keys, 3 from 16384, 4 from 4096, 5 from
2048, 6 and 7 from 16384, 8 from 12288. BF16 caches keep the row twin.

In the model (NVFP4-Trellis-RCO, MTP depth 4, the README scenarios), a verification takes 19.64 ms at 4K, 21.36 ms at 32K,
22.82 ms at 64K and 26.80 ms at 128K, and decode runs at 135.0, 119.8, 107.5 and 88.1 tok/s.

## Cache pages in host memory

When the weights and every sequence's whole attention cache do not fit on the GPU, the pages past each sequence's first
`kv-device-minimum-tokens` tokens (default 65536) live in pinned host memory, and only then do weights leave the device
(`ResidencyPlanner`). The planner sizes one pinned arena at load: per sequence slot, each attention layer's host pages, and
one device region that holds a single layer's host pages, which the slot's layers share (`KvHostPages`). A sequence takes a
slot when its state is attached and returns it when the state closes; a sequence that finds every slot taken is refused,
never kept waiting. The MTP layer's cache stays on the device.

Pages never move. A cache's page table addresses its host pages in the region (`AttentionKvState.HostRun`), so every
attention kernel reads them unchanged. Each layer's append copies the host pages the quantum reads into the region before it
writes, and the pages it wrote back afterwards, on its own stream; the layer's attention runs after the append and the next
layer's append after that attention, so one region serves all layers. The KV stages of a host-page engine also carry the
region from one quantum of a sequence to the next. Decode, verification and draft quanta (at most 8 rows, the captured
ones) copy every reserved host page in and the last two back, so what they queue changes only when the reservation grows,
which is in the capture fingerprint; prompt chunks copy only the pages they reach and write. Prefix-cache checkpoints stop
at the device share.

Greedy output is bit for bit that of the same engine with every page on the device (`HostKvPagesCudaIntegrationTest`:
a 3000-token prompt over 1024 device tokens, decode, MTP verification and drafting).

NVFP4-Trellis-RCO, 32K chat prompt, MTP depth 4, every weight on the GPU, by the cache share kept on it:

| Device share | Host tokens | ms per verification | Decode tok/s | Prefill tok/s |
|---|---|---|---|---|
| 32768 (all) | 0 | 20.8 | 147.6 / 123.1 | 2055 |
| 24576 | 8192 | 24.5 | 128.9 / 107.3 | 2045 |
| 16384 | 16384 | 27.9 | 115.8 / 95.9 | 2032 |
| 8192 | 24576 | 31.3 | 104.4 / 86.9 | 2014 |

(Two chat prompts per row; outputs identical across rows.) Each 8192 host tokens add about 3.5 ms to a verification: 145 MiB
of pages copied in per step, about 41 GB/s.

## Rejected

- **Reading host pages in place** (page table entries pointing at mapped pinned host memory, no copies): the kernels' 16-byte
  loads reach 2 to 3 GB/s over the bus. Per layer, 1-row decode takes 22 ms at 64K keys and the 5-row fused verifier 74 ms
  at 128K.
- **A deeper cp.async ring, or finer key splits.** The loads alone take 28 us of the 16K one-row kernel; the time is the phases of one
  warp running one after another at about 0.1 instructions per cycle. A cp.async ring on the single-warp kernel left it at 131 us;
  128 and 256 splits ran at 78 and 76 us because every CTA repeats the query rotation and the merge grows. A 4-stage ring takes 31 KiB
  of shared memory, drops the SM to 3 resident CTAs and runs slower than 2 stages.
- **Named barriers for the hand-offs.** Nine ids cap the SM at one resident CTA.
- **Deeper hand-off rings in the fused kernel** (three V and score slots in place of two): no change at 16K and 64K. Six raw
  stages (53 KiB of shared memory) leave one CTA per SM: 16% slower at 64K than four.
- **Register prefetch of V in the fused kernel's loader** (one tile ahead, as in the one-row kernel): with one tile of K and V in
  flight per CTA and one CTA per SM the kernel waited on DRAM latency and ran no faster than the row twins.
- **Arithmetic E2M1 expansion** (measured on the earlier E2M1 code grid: code bits placed as an FP16 subnormal and scaled by 2^14, in place of the pair table; exact): 1 to 2%
  on the fused kernel and the twin, within noise. Shared-memory wavefronts are not what holds the fused kernel back.
- **Dropping the lo FP16 parts** (queries and probabilities as single FP16 values; changes the numerics of one-row decode too):
  the fused kernel takes 191.6 us at 64K, 29% less (with the arithmetic expansion). Not adopted: the error budget of the relaxed
  kernels is set against the FP32 kernel.
- **Row twins in split-major launch order** (the rows of one KV head and key split adjacent, so they share the split's keys and
  values through L2; same per-CTA arithmetic). Real head geometry, random cache, per layer:

  | Keys | 4 rows | 6 rows | 8 rows |
  |---|---|---|---|
  | 16K | 101.7 us (row-major 102.3) | 149.7 (143.2) | 198.0 (189.7) |
  | 32K | 194.6 (193.2) | 285.6 (267.4) | 377.7 (349.9) |
  | 64K | 371.8 (427.6) | 543.7 (635.7) | 717.2 (839.2) |

  The twin is bound by its CTAs' dependency chains at four resident CTAs per SM (registers and shared memory), not by DRAM, so
  sharing the reads helps only at 64K and costs up to 8% at 32K. The same residency bounds any kernel that serves several rows
  from one CTA: one score and one consumer warp per row keep the rows in flight per SM at four unless the queries leave registers.
- **Shared query rotation** (rotate once per KV head, not per split). Skipping the Hadamard entirely saves 2.5% at 16K and 0.7% at 64K.
- **Merge inside the decode kernel** (the last CTA merges). The merge is 7 us of 38 us at 16K and fusing removes the launch gap, not the work.
