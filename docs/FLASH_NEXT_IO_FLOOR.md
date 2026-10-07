# Flash-Next I/O floor

Where Flash-Next's storage traffic comes from, how much of the drive it uses, and what the memory hierarchy allows.
Measurements only: no change to scheduling, cache policy or GPU execution.

Machine: RTX 5070 Ti, i9-14900K, 61 GiB RAM, the artifact on a Samsung 990 Pro (nvme1n1, PCIe Gen4 x4, also the
root file system and swap), 32 workers. Unless stated otherwise, the host budget is the benchmark's 22,000 MiB (a
20.2 GiB RAM tier of 7,852 records, 3,102 device slots, 5,120 context). Decode is in tokens/s.

## Instruments

`Qwen4PerformanceCudaIntegrationTest` now prints three `io:` lines per phase (`Qwen4IoProbe`):

- **SSD bytes by source.** The artifact's whole drive in `/proc/diskstats`, split into:
  - expert reads, from the engine's counters, as demand and prefetch;
  - other reads of this process (`/proc/self/io` less the expert reads): n-gram page faults;
  - swap-in (`pswpin`);
  - the rest of the machine.
- **The drive's use.** Busy time (`io_ticks`). Requests in flight (time in queue over wall time); a 2.77 MB record is
  about 22 requests of 128 KiB. A 10 ms sampler classes each window as saturated (at least 85% of 7.0 GB/s), partial or
  idle.
- **Per phase.** Expert reads per token and per layer, the time the phase's bytes take at 7.0 GB/s, major faults per
  token, and the n-gram tables' resident bytes before and after (`NgramStore.residentBytes`, `mincore`).

Two temporary toggles, not committed:

- skipping the disk: each read reads 4 KiB and completes as whole, so a missed record costs only its copy to the
  device;
- `madvise` on the n-gram mapping.

The drive (`O_DIRECT` reads of whole 2.77 MB records at random offsets):

| records in flight | GB/s | latency median / p90 |
|---|---|---|
| 1 | 5.09 | 526 / 593 us |
| 2 | 6.93 | 777 / 903 us |
| 4 | 6.90 | 1556 / 1731 us |
| 8 | 6.90 | 3122 / 3338 us |
| 4 KiB, 1 in flight | | 62 / 69 us |

A record takes about 130 us of fixed latency plus 396 us of transfer at 7 GB/s. The drive reaches its ceiling only with
two or more records in flight.

## Bytes per token by source

| phase | SSD MB/token | expert demand | expert prefetch | n-gram faults | swap, rest of machine |
|---|---|---|---|---|---|
| prefill 512 | 45.5 | 45.5 | 0 | 0 (1.56 cold) | 0 |
| prefill 4096 | 9.1 | 9.1 | 0 | 0 (1.08 cold) | 0 |
| decode cold, 64 context | 164.6 | 79.9 | 84.7 | 0 (1.44 cold) | 0 |
| decode warm, 64 context | 112.2 | 48.0 | 64.2 | 0 | 0 |
| decode cold, 4096 context | 169.2 | 89.6 | 79.6 | 0 (1.54 cold) | 0 |
| decode warm, 4096 context | 116.0 | 46.1 | 69.9 | 0 | 0 |

Expert reads per token:

- prefill 512: 16.4 (0.34 per layer);
- prefill 4096: 3.3 (0.07 per layer);
- decode cold: 59-61 (1.24-1.27 per layer), of which 29-32 are demand;
- decode warm: 41-42 (0.84-0.87 per layer), of which 17 are demand.

The prefetch is about half of decode's reads; 40-60% of the records it reads are used.

The device cache hits:

- 0% in prefill: a chunk asks for about 442 distinct experts per layer, which no 3,102-slot cache keeps;
- 73-76% in decode cold, 87-88% in decode warm.

The RAM tier catches:

- 37-44% of the device misses in prefill;
- 72-78% in decode cold and 70-74% in decode warm.

The disk serves the rest:

- 56-63% of all requests in prefill (every one of them misses the device);
- 6-7% in decode cold and 3-4% in decode warm (1 − device hit × RAM hit; the prefetch reads come on top).

## The drive's use

| phase | GB/s over the phase | drive busy | saturated / partial / idle 10 ms windows | records in flight while busy | the bytes at 7 GB/s, share of the wall time |
|---|---|---|---|---|---|
| prefill 512 | 6.69 | 88% | 81 / 19 / 1% | 7.8 | 96% |
| prefill 4096 | 5.99 | 79% | 74 / 22 / 4% | 7.8 | 86% |
| decode cold 64 | 3.81 | 63% | 0 / 99 / 1% | 1.4 | 54% |
| decode warm 64 | 3.37 | 57% | 0 / 100 / 0% | 1.2 | 48% |
| decode cold 4096 | 3.93 | 64% | 0 / 100 / 0% | 1.6 | 56% |
| decode warm 4096 | 3.37 | 57% | 0 / 99 / 1% | 1.1 | 48% |

**Prefill is at the drive's floor.** Bytes per token over 7 GB/s predicts 154 tokens/s for 512 tokens (measured
147) and 769 for 4096 (measured 662). The 4096-token gap is the time between layers when no record is asked for yet.

**Decode is not.** Bytes per token over 7 GB/s predicts:

| scenario | predicted from all bytes | from the demand bytes alone | measured |
|---|---|---|---|
| decode cold 64 | 42.5 | 87.8 | 23.1 |
| decode warm 64 | 62.4 | 144 | 30.0 |
| decode cold 4096 | 41.4 | 78.4 | 23.3 |
| decode warm 4096 | 60.3 | 151 | 29.1 |

The drive is never saturated in decode: a step asks for about one record at a time. A layer's misses are known only
once its router has run, and there are 0.6-0.7 demand reads per layer cold. So decode pays a record's latency for each
demand read rather than its share of the bandwidth.

Across the host-memory sweep below, a decode step's time fits

    ms/token = t0 + 0.42-0.44 ms x demand disk reads/token

| scenario | t0 | t0 as tokens/s | largest error of the fit |
|---|---|---|---|
| decode cold 64 | 30.2 ms | 33.1 | 3.6 ms |
| decode warm 64 | 25.1 ms | 39.8 | 2.7 ms |
| decode cold 4096 | 28.6 ms | 35.0 | 5.0 ms |
| decode warm 4096 | 26.1 ms | 38.3 | 4.2 ms |

The fit covers all 17 runs, from no tier to a 37.8 GiB tier. Each demand read costs about one record's service time
on the critical path. The prefetch's reads queue on the same drive but cost little.

The SSD's share of the step at the benchmark's tier, measured minus t0:

| scenario | SSD time | share of the step |
|---|---|---|
| decode cold 64 | 13.0 ms | 30% |
| decode warm 64 | 8.2 ms | 25% |
| decode cold 4096 | 14.4 ms | 34% |
| decode warm 4096 | 8.3 ms | 24% |

With the disk skipped, the steps were 26.5 / 22.7 / 24.9 / 22.4 ms in the same order. Those runs compute with stale
weights, which changes routing (device hits 82-91%) and makes them an optimistic bound.

## Host-RAM scaling

The host budget swept twice, with the automatic planner run inside a 42 GiB cgroup. Each run is a fresh process; the
two passes agree within 1-3% (prefill 4096 at 34,000 MiB 5.7%). Values are their means.

| host budget (MiB) | tier (GiB, records) | expert SSD MB/token: prefill 4096 / decode cold 4096 / decode warm 4096 | expert reads per token, decode cold 4096 (demand) | prefill 512 / 4096 (tokens/s) | decode cold 64 / 4096 | decode warm 64 / 4096 |
|---|---|---|---|---|---|---|
| 0 or 4,000 | none | 14.4 / 319 / 176 | 115.3 (115.3) | 86 / 455 | 12.1 / 13.3 | 20.1 / 18.7 |
| 10,000 | 8.5, 3,307 | 12.1 / 261 / 152 | 94.2 (58.6) | 108 / 528 | 16.1 / 17.1 | 25.4 / 24.6 |
| 16,000 | 14.4, 5,580 | 10.6 / 212 / 137 | 76.4 (44.0) | 132 / 594 | 19.3 / 20.2 | 26.0 / 27.2 |
| 22,000 | 20.2, 7,852 | 9.1 / 168 / 115 | 60.7 (32.2) | 148 / 665 | 23.1 / 23.2 | 30.0 / 29.6 |
| 28,000 | 26.1, 10,124 | 7.5 / 126 / 88 | 45.3 (22.2) | 158 / 758 | 26.7 / 26.6 | 33.1 / 33.2 |
| 34,000 | 32.0, 12,396 | 6.1 / 89 / 67 | 32.2 (14.0) | 176 / 842 | 29.9 / 30.4 | 36.2 / 35.4 |
| automatic (42 GiB cgroup) | 34.3, 13,305 | 5.8 / 78 / 56 | 28.0 (11.8) | 186 / 859 | 31.0 / 31.4 | 37.5 / 35.2 |
| 40,000 (one pass) | 37.8, 14,668 | 5.4 / 60 / 45 | 21.7 (8.5) | 201 / 889 | 31.5 / 32.4 | 36.4 / 35.8 |

Notes on the sweep:

- **A budget of 4,000 MiB leaves no tier.** The pinned embedding and staging take it first.
- **The automatic planner** takes `MemAvailable` less a 10% margin (at least 4 GiB), the JVM heap's growth and 1 GiB.
  Run uncapped on this desktop, it pinned enough to bring `MemAvailable` down to 6 GB, and the watchdog stopped it.
  Inside a 42 GiB cgroup it chose a 34.3 GiB tier.
- **40,000 MiB is the practical ceiling of this machine.** It completed, but it pushed other processes into swap,
  which sits on the artifact's drive.

Beyond physical memory, `tools/expert_cache_sim.py` replays the recorded demand (`/tmp/demand3.bin`, 8 requests): S3-FIFO
device, the tier's sampled LFU with scan admission, startup fill. Its demand reads relative to the 7,852-record tier,
applied to the fit, predict decode:

| tier (records, GiB) | simulated decode demand MB/token | decode cold 4096 | warm 4096 | cold 64 | warm 64 |
|---|---|---|---|---|---|
| 7,852, 20.2 | 94.8 | 23.6 | 29.8 | 23.6 | 30.4 |
| 10,000, 25.8 | 61.3 | 26.7 | 32.3 | 26.2 | 33.2 |
| 12,000, 30.9 | 39.4 | 29.1 | 34.2 | 28.3 | 35.3 |
| 14,000, 36.1 | 25.0 | 31.0 | 35.6 | 29.9 | 36.8 |
| 16,000, 41.3 | 15.9 | 32.4 | 36.5 | 31.0 | 37.9 |
| 18,000, 46.4 | 9.8 | 33.3 | 37.2 | 31.8 | 38.6 |
| 22,000, 56.7 | 2.9 | 34.5 | 38.0 | 32.7 | 39.5 |
| 24,576, 63.4 (every record) | 0 | 35.0 | 38.3 | 33.1 | 39.8 |

The prediction matches the measured points it overlaps. At 10,124 records it gives 26.6 against 26.7 for decode cold
at 4096; at 12,396 and 14,668 records it is within 1 tokens/s.

- **Most of the expert traffic is gone at about 16,000-18,000 records (41-46 GiB).** Decode's demand reads fall by
  83-90% from today's tier, and the prefill reads by 64-75%.
- **All of it needs every record: 63.4 GiB (64,896 MiB).** That means about 96 GiB of RAM on the machine, with the
  n-gram rows and the system beside it.
- **Decode gains little past about 40 GiB.** It is within 7% of its zero-SSD rate there.

## N-gram traffic

The tables (26.8 GiB) are mapped from the artifact; a step gathers about 16 rows of 96 bytes per token.

With their pages resident from earlier runs, they cost no SSD traffic and no faults. With the artifact's page cache
dropped first (`posix_fadvise DONTNEED`):

| phase | resident rows | cold rows | cold, `MADV_RANDOM` |
|---|---|---|---|
| prefill 512 (tokens/s) | 147 | 127 | 139 |
| prefill 4096 (tokens/s) | 662 | 445 | 533 |
| decode cold 64 / 4096 (tokens/s) | 23.2 / 23.2 | 22.6 / 22.3 | 23.0 / 22.7 |
| n-gram SSD MB/token, prefill 4096 / decode cold 4096 | 0 / 0 | 1.08 / 1.54 | 0.04 / 0.06 |
| major faults per token, prefill 4096 / decode cold 4096 | 0 / 0 | 8.5 / 12.2 | 9.8 / 15.7 |
| n-gram resident after the run | 6.06 GiB | 5.11 GiB | 0.18 GiB |

- **Bandwidth: they don't matter.** Cold rows are at most 1.6 MB/token: 1% of decode's bytes and 11% of prefill
  4096's.
- **Latency: they do matter when cold.** Each fault is synchronous in a step's gather and waits behind the expert reads
  queued on the same drive: prefill 4096 loses 33%, decode 2-4%.
- **The kernel reads 128 KiB around each faulting 4 KiB page.** `MADV_RANDOM` reads only the page: 30 times fewer
  bytes and memory for the same rows. It recovers half of prefill's loss, but the faults remain.
- **The rows a workload touches have to stay resident.** A pinned cgroup cap shows the risk: with 8 GiB at budget 0,
  prefill 4096 fell from 455 to 376 tokens/s with 5.5 faults per token.

## PCIe topology and a second drive

- **GPU:** CPU root port, PCIe Gen5 x16. Pinned copies measured 25 GB/s from one copy test; under the engine's
  concurrent expert copies the link carried up to 40.8 GB/s (`nvidia-smi dmon`).
- **990 Pro:** behind the Z790 chipset (port `00:1a.0`, DMI 4.0 x8, about 16 GB/s for everything on the chipset).
- **The CPU's own Gen4 x4 M.2 slot (`00:06.0`):** holds the 970 EVO Plus, a Gen3 drive.
- **Two more Gen3 drives** (960 EVO, 980) are on the chipset.

Read together with the 990 Pro (4 records in flight each), the 970 EVO Plus gives 6.90 + 3.62 = 10.5 GB/s: the paths
do not limit each other.

A Gen4 drive in the CPU slot beside the 990 Pro would give about 14 GB/s, the chipset carrying only one drive's 7.
Two Gen4 drives both on the chipset would approach the DMI limit.

## Storage scaling model

Prefill is modelled as its non-disk remainder plus bytes over bandwidth, floored at the disk-skipped time. Decode is
the fit's t0 plus its demand reads, each costing the record's service time (130 us + 2.77 MB / bandwidth, scaled from
0.43 ms at 7 GB/s). Both are at the benchmark's 20.2 GiB tier, against what was measured there:

| scenario | measured | 7 GB/s | 10 GB/s | 14 GB/s | 20 GB/s | no expert SSD reads |
|---|---|---|---|---|---|---|
| prefill 512 | 147 | 147 | 206 | 282 | 389 | 423 |
| prefill 4096 | 662 | 662 | 892 | 1,161 | 1,501 | 1,728 |
| decode cold 64 | 23.1 | 23.6 | 25.2 | 26.4 | 27.4 | 33.1 |
| decode warm 64 | 30.0 | 30.4 | 32.1 | 33.4 | 34.4 | 39.8 |
| decode cold 4096 | 23.3 | 23.6 | 25.5 | 26.9 | 28.1 | 35.0 |
| decode warm 4096 | 29.1 | 29.8 | 31.4 | 32.5 | 33.4 | 38.3 |

"No expert SSD reads" is the disk-skipped run for prefill and the fit's t0 for decode.

- **Prefill scales almost linearly with bandwidth.** It doubles at about 14 GB/s.
- **Decode gains 7-20%** from 10 to 20 GB/s: a faster drive shortens each record's transfer but not its latency, and
  decode reads about one record at a time.
- **RAM does more for decode.** 41 GiB of tier on today's drive (32.4 tokens/s cold at 4096) beats a 20 GB/s drive
  with today's tier (28.1).

## Compression

A record is the gate, up and down projections of one expert (3 × 2560 × 640 weights) at 4.51 bits per weight:
2.77 MB.

The lossless coder's measurements on these weights (`docs/NVFP4_LOSSLESS.md`) give 4.09 bits per weight for the routed
experts: E2M1 codes at 3.86-3.87 bits and scales at 0.22, with rANS. That is 9.2% less, 2.51 MB per record.

- **Bytes:** prefill 4096 would read 8.3 MB/token instead of 9.1, decode cold 4096 81 MB of demand instead of 89.
- **Ceiling at 7 GB/s:**
  - prefill 512, 147 to 161 tokens/s; prefill 4096, 662 to 719;
  - decode 1-2% (23.3 to 24.1 cold at 4096): decode pays latency per read, and the transfer shrinks by only 9%.
- **Decompression throughput:** it would have to run at the disk's 7 GB/s of compressed input, 7.7 GB/s of records,
  on every disk-served record.
  - The GPU decoder measured 14.2 GB/s using the whole GPU. That is half the GPU's time during a prefill, which is
    already 1.21 s of kernels per 4096 tokens, so the gain disappears.
  - On the CPU it would be a new rANS decoder at about 8 GB/s across the workers, on the path between the read and the
    copy of every record.

Not worth it: at most 9% on prefill, 1-2% on decode, with decompression on the critical path of every disk-served
record. Not implemented.

## What limits it now

At the benchmark's tier:

- **Prefill = storage bandwidth.** The drive is saturated in 74-81% of the time and its bytes take 86-96% of the wall
  time.
  - With the disk skipped, prefill 4096 runs at 1,728 tokens/s: 52 GB of expert copies to the device (22 GB/s
    average, peaks of 40 GB/s on the link) overlapped with 1.21 s of kernels.
  - The next limit is PCIe transfer and GPU execution together.
- **Decode = demand-read latency + GPU execution + copies and host bubbles.**
  - Each demand disk read costs about 0.43 ms on the critical path: 8-14 ms of a 33-43 ms step.
  - The rest (t0, 25-30 ms):
    - GPU kernels, 16.2-16.7 ms per step (`docs/FLASH_NEXT_GPU.md`);
    - copies of tier hits to the device, 164-359 MB/token, exposed for about 3 ms;
    - host bubbles of about 5.5 ms: the router's round trip, the token boundary and launch gaps.
  - With storage removed, decode runs at 33-40 tokens/s and the next bottleneck is GPU execution (about 60% of the
    step), then the tier-to-device copies and the host bubbles.
