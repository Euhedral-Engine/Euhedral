# Memory system of the RTX 5070 Ti

GB203 in the 5070 Ti has:
- 16 GB of GDDR7 on a 256-bit bus at 28 Gbps (896 GB/s), with on-die ECC that is always on.
- 48 MiB of L2: 16 of the die's 64 MiB are fused off, but all 8 memory controllers stay.
- 128 KB of L1/shared memory per SM.

Every number below was measured on the card (logs: [microbench/RESULTS.md](microbench/RESULTS.md),
`nvbench mem`, `nvbench l2`, `nvbench misc`). The SM clock was about 2.86 GHz during the runs.

## DRAM

### Bandwidth needs bytes in flight

**Streaming reads, 2 GiB buffer:**

| Bytes in flight per SM | GB/s |
|---|---|
| 4 KB (8 warps × one 16-byte load) | 725 |
| 8 KB | 844–847 |
| ≥ 16 KB | 850–855 |

- **Peak is 855 GB/s, 95.4% of the 896 GB/s spec.** That is 4.24 bytes per SM clock.
- **Write:** 828 GB/s. **Copy:** 762 GB/s (read and write counted).
- **Load width doesn't matter once enough bytes are in flight.** All of these reach 850–855 GB/s:
  - `LDG.32` ×8 per thread;
  - `LDG.128`;
  - `ld.global.nc.v8.u32`, which compiles to a real 256-bit load, `LDG.E.ENL2.256.CONSTANT`, on
    sm_120.
- **Wider loads cut instruction count.** A 256-bit load issues half the instructions per byte of an
  `LDG.128` and a quarter of `LDG.32` ×8. That matters to issue-bound kernels (see
  [sm-pipes.md](sm-pipes.md)).
- **Availability:** the 256-bit form needs PTX 8.8+ and an sm_100+/sm_120 target, so it is not
  reachable from `compute_90` PTX.

**Rule of thumb: keep at least 8 KB per SM in flight to saturate DRAM.** Little's law with the
loaded latency gives the same answer. A decode kernel with only about 4 warps per SM and a few
outstanding 16-byte loads per thread runs far below DRAM speed whatever its arithmetic does.

### TMA bulk copies

A ring of `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` copies (SASS
`UBLKCP.S.G` plus `SYNCS` mbarrier ops) reaches 853–856 GB/s:
- One elected thread per CTA issues them.
- Every configuration saturates DRAM: 4–32 KiB chunks, 2–8 stages, 1–4 CTAs per SM.
- **Even one CTA per SM with 64 KiB in flight saturates DRAM.** This is the cheapest way, in issue
  slots, to keep a lot of memory in flight.
- It leaves every other warp free for math.
- Euhedral tried a TMA ring for Q3 decode and found no gain on cold weights (docs/FRAME_MODEL.md).
  That kernel was already at DRAM speed, so the result is consistent with this measurement.

### Latency

Dependent pointer chase, random 128-byte lines, SM cycles at about 2.87 GHz:

| Footprint | Path | Cycles | About |
|---|---|---|---|
| ≤ 64 KiB | L1 hit (`ld.global.ca`) | 45 | 16 ns |
| ≤ 2 MiB | L2 hit (`.cg`) | 356 | 124 ns |
| 3 MiB | L2 | 397 | |
| 4–8 MiB | L2 | 501–537 | |
| 16–40 MiB | L2 | 670–702 | 240 ns |
| 64 MiB | mostly DRAM | 728 | |
| 256 MiB – 1 GiB | DRAM | 766–774 | 270 ns |
| shared memory | dependent `LDS` | 34 | |

- **L2 hit latency doubles as the footprint grows past about 2 MiB,** even though the data still
  fits in L2.
- **It is not the TLB.** A walk touching one line per 2 MiB page over 1024 pages (2 GiB) stays at
  356–364 cycles. The L2 structure itself must make large footprints slower to hit.
- Bandwidth-bound code doesn't care; L2 bandwidth is flat from 8 MiB to 48 MiB. Latency-bound code
  (pointer chasing, small dependent reads, a single-warp GDN control step) sees near-DRAM latency
  from a "hit".

## L2

### Capacity and bandwidth

Re-reading a working set with `ld.global.cg` (4 loads in flight per thread):

| Working set | GB/s |
|---|---|
| 1 MiB | 3,071 |
| 4 MiB | 3,464 |
| 8–48 MiB | 3,545–3,583 |
| 56 MiB | 2,010 |
| 64 MiB | 993 |
| ≥ 96 MiB | 850–864 |

- **All 48 MiB is usable** at about 3.55 TB/s, which is 17.8 bytes per SM clock and 4.2× DRAM.
- **It matters for benchmarks.** Single-layer weights (13–36 MB) fit entirely, which is why operator
  benches must rotate cold weight copies; an L2-hot bench measures instruction issue, not the
  DRAM-bound reality.

### Residency control works on GeForce

**Device limits:**
- Persisting L2 maximum (`cudaDevAttrMaxPersistingL2CacheSize`): 30 MiB.
- Maximum access-policy window: 128 MiB.

**Test:** read a hot buffer, run a 1 GiB streaming read that evicts it, read the hot buffer again.
GB/s of the second hot read:

| Hot set | Back-to-back (L2) | Plain | `createpolicy` hints | Persisting window | Window + hints |
|---|---|---|---|---|---|
| 8 MiB | 1,900 | 819 | 2,048 | 2,048 | 2,048 |
| 16 MiB | 2,621 | 910 | 2,881 | 2,731 | 2,731 |
| 24 MiB | 2,613 | 819 | 2,562 | 3,072 | 3,072 |
| 32 MiB | 2,865 | 819 | 2,731 | 2,731 | 2,731 |

**Two independent mechanisms work:**
1. **PTX hints, no host setup.** Use `createpolicy.fractional.L2::evict_last.b64 p, 1.0` on the
   reused data and `L2::evict_first` on the stream, through `ld.global.L2::cache_hint`. The hot set
   survives a 1 GiB stream and reads back at about 3× DRAM speed.
2. **Stream access-policy window.** Set `cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, ≤ 30 MiB)`,
   then `CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW` with `hitProp = PERSISTING`. Reset it with
   `cuCtxResetPersistingL2Cache`.

**For Euhedral:**
- **Weights:** decode reads every weight once per token, so they should be `evict_first`. Today no
  load carries a cache hint.
- **What's worth protecting** is data reused across kernels within a token:
  - activations and residuals;
  - norm vectors;
  - the GDN state between its fan-out kernels;
  - KV pages read by several verify rows.
- **What doesn't fit:** the per-sequence GDN state (144 MiB in FP32), the KV cache at 16K+ (about
  300 MB) and the MTP drafting stack (more than 200 MB) are all too large to persist.
- **Expected gain** is bounded by today's non-weight L2 miss traffic. Measure it first with Nsight
  Compute `lts__t_sectors_srcunit_tex_lookup_miss`.

### Hardware data compression is enabled

`CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 1` on this GeForce card. A `cuMemCreate`
allocation with `allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC` is granted.

**Cold 1 GiB read, by data pattern:**

| Data in a compressible allocation | GB/s |
|---|---|
| all zeros | 4,999 |
| constant 1.0f | 3,567 |
| 90% zero 32-bit words | 2,994 |
| u32 values 0..255 (three zero bytes per word) | 1,691 |
| 50% zero 128-byte lines | 1,547 |
| 50% zero words | 999 |
| bytes 0..15 (4-bit values in 8-bit containers) | 851 |
| BF16 with a fixed exponent | 850 |
| FP32 with a fixed exponent | 850 |
| random | 850 |

- **What compresses:** only zero and constant structure, at word and byte granularity. Bit-packed
  entropy and narrow exponent ranges get nothing.
- **Cost:** a plain allocation reads at 854 GB/s, so random data costs nothing. Writes into a
  compressible allocation are 4.5% slower (791 vs 828 GB/s).
- **Footprint:** it does not reduce capacity
  ([Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)).
- **Verdict for Euhedral:**
  - Useless for quantized weights, BF16 activations, the FP32 GDN state and NVFP4 KV.
  - Useful only for buffers that really are zero-heavy: masks, zero-initialized scratch, sparse
    intermediates.

## L1, shared memory and distributed shared memory

**Limits:**
- 128 KB of unified L1/shared per SM.
- Shared memory: up to 100 KiB per SM. Per block, 48 KiB by default and 99 KiB with the opt-in
  attribute; each block reserves 1 KiB.
- 1536 threads, 24 blocks and 64K registers per SM.

**Local shared memory:**
- Reads ran at about 52 bytes per clock per CTA, 2 CTAs per SM, 18.9 TB/s aggregate.
- Dependent `LDS` latency was 34 cycles.

**Thread-block clusters:**
- **Size:** 8 portable. The non-portable attribute raises the maximum potential cluster size to 12
  (GPC-limited); 16 does not schedule.
- **Fragmentation:** with clusters of 8, only 17 clusters (136 of 140 CTA slots) fit at once.
- **DSMEM is slow on this part.** Reading a peer CTA's shared memory with `ld.shared::cluster.v4` ran
  at 2.3 bytes per clock per CTA, **0.71 TB/s aggregate: 27× below local shared memory and below
  DRAM.**
- **No multicast:** GeForce has no TMA multicast, so CUTLASS's SM120 GEMMs use 1×1×1 clusters.
- **Verdict:** clusters are for small exchanges here (flags, reductions), not data sharing. This
  matches Euhedral keeping its cluster/DSM Q3 kernels test-only.

## GDDR7 notes

**Reliability features:**
- Per JEDEC and NVIDIA, GDDR7 has on-die ECC (always on, single-bit correction), CRC and Error
  Detection and Replay on the link
  ([JEDEC](https://www.jedec.org/news/pressreleases/jedec-publishes-gddr7-graphics-memory-standard),
  whitepaper p.14).
- With replay, an unstable memory overclock shows up as lower bandwidth before it shows up as
  errors. Tune memory offsets with a bandwidth sweep, not a stability test alone (see
  [system-tuning.md](system-tuning.md)).

**Clocks:**
- The memory clock ran at 13.8 GHz (27.6 Gbps) under load, against 14.0 GHz maximum.
- Idle drops it to 405 MHz. First-launch timing includes the climb out of P8.

## Rules for Euhedral kernels

1. **Size grids for bytes in flight.**
   - At least 8 KB per SM: for example 16 warps × one `LDG.128`, or 8 warps × two.
   - Long-context decode attention launched as 4 × 64 single-warp CTAs is about 3.7 warps per SM.
     That cannot approach DRAM speed; doc numbers imply about 160 GB/s for the GQA kernel at 16K.
   - Split-K (flash-decoding) across many more CTAs is the fix.
2. **Prefer one TMA-issuing thread with deep stages** where a kernel is issue-bound. It frees the
   other warps' issue slots.
3. **Mark weight streams `evict_first`,** and test `evict_last` on whatever a quantum re-reads.
4. **Time operators with cold weights,** because 48 MiB holds a whole layer.
5. **Don't use clusters or DSMEM to share bulk data** on this part.
