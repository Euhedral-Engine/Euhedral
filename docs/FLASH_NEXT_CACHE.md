# Flash-Next expert cache policy

Which records the device cache and the RAM tier keep. The engine's demand is recorded once
(`Qwen4DemandRecordingCudaIntegrationTest`, `EUHEDRAL_QWEN4_TRACE=FILE`) and replayed offline through policies
(`tools/expert_cache_sim.py`), so a policy is judged in seconds on the same demand.

## The recording

Eight requests of 512, 2048, 300, 4096, 1200, 700, 3000 and 150 tokens, slices of the repository's documents, one after
another; each is prefilled in the plan's chunks (up to 4096 rows) and continued greedily for 256 tokens. 1.10 million
expert requests: 119,564 in prefill (12,006 tokens) and 983,040 in decode (2,048 tokens). The engine's cache at the
recording: 3,386 device slots, 7,852 tier slots of 24,576 records (`RAM_CACHED` at 22,000 MiB).

The replay matches the engine: decode misses the device 24% of the time (the engine's telemetry: 15-30%), and the tier
serves 63-66% of device misses (the engine: 50-65%).

## Decode's reuse

- 35% of decode requests reuse an expert the same layer used one token earlier, 54% within 4 tokens, 90% within 64.
- Over 2,048 decode tokens a layer uses 344-474 of its 512 experts.
- Per-layer LRU at 70 slots (the device's share) hits 76%; 120 slots 87%; 200 slots 93%.
- Static popularity (the most requested records of the whole run) is poor for decode (42% device miss): which experts
  are hot depends on the request.

## Policies

Each level was replayed with the other at today's policy (device LRU, tier TinyLFU), without the startup fill:

| device policy | prefill device miss | decode device miss |
|---|---|---|
| LRU (today) | 98.4% | 24.0% |
| scan-resistant LRU (a prefill's records enter cold) | 86.3% | 24.0% |
| per-layer LRU | 86.2% | 24.5% |
| layer-aware recency (age in the layer's visits) | 96.5% | 23.6% |
| ARC | 89.8% | 23.3% |
| S3-FIFO | 88.3% | 22.3% |
| LFU / TinyLFU / W-TinyLFU | 87.6 / 84.6 / 84.8% | 26.5 / 37.3 / 36.3% |
| SIEVE | 92.5% | 45.3% |
| Belady (offline) | 82.4% | 10.6% |

| tier policy | prefill disk (MB/token) | decode disk (MB/token) |
|---|---|---|
| LRU | 24.6 | 110.7 |
| per-layer LRU | 18.1 | 118.7 |
| TinyLFU (today) | 16.6 | 110.1 |
| W-TinyLFU | 17.1 | 107.8 |
| LFU | 18.5 | 102.2 |
| ARC / S3-FIFO | 21.8 / 21.2 | 104.3 / 109.7 |
| static popularity (offline) | 14.7 | 106.2 |
| Belady (offline) | 17.3 | 52.4 |

The tier sees only the device's misses, from which recency is already filtered: frequency serves decode best there, and
admission keeps a prefill's sweeps from replacing what decode uses. Combining them, LFU eviction (a sample of 32 held
records, the least requested, the least recent among equals) with admission only for the records a prefill chunk asks
for (`slfu-scan`), and S3-FIFO on the device:

| recording | tier slots | today (LRU / TinyLFU) | S3-FIFO / slfu-scan | decode disk MB/token | decode device misses/token |
|---|---|---|---|---|---|
| 1 (8 requests, 256 decode) | 7,852 | 412.7 GB | 382.9 GB (-7.2%) | 110.4 to 99.1 | 115.2 to 107.2 |
| 2 (10 requests, 400 decode) | 7,852 | 713.5 GB | 640.2 GB (-10.3%) | 123.5 to 107.0 | 142.5 to 130.2 |
| 1 | 12,000 | 231.4 GB | 213.1 GB (-7.9%) | 57.8 to 46.5 | 115.2 to 107.2 |
| 2 | 12,000 | 349.5 GB | 306.2 GB (-12.4%) | | 142.5 to 130.2 |

(with the startup fill). The online policies differ by a few percent; the tier's capacity and knowledge of the future
differ by far more: 12,000 tier slots halve the disk bytes, and Belady halves decode's again.

## In the engine

The tier's `FREQUENCY` policy (`RamTierShard`: the least requested of 32 sampled ready slots, the least recent among
equals; admission only for a prefill chunk's records) and the device's `S3_FIFO` (`ExpertCacheShard`: a small and a
main queue, 2-bit counts, a ghost stamp per key; a leased slot leaves its queue and returns to its tail) are the
defaults; `EUHEDRAL_QWEN4_TIER_POLICY=partitioned` and `EUHEDRAL_QWEN4_GPU_POLICY=global` restore the old ones. The
device's policy switch had never reached the cache before; it does now.

Quick screens (one process each, about 50 s; the decode phases feed the prompt's tokens, 32 steps twice):

| scenario | before (partitioned tier, LRU device) | frequency tier | frequency tier, S3-FIFO device |
|---|---|---|---|
| prefill 512 (tokens/s, 3 runs) | 148 | 153 | 138-140 (1 run each) |
| prefill 4096 (tokens/s, 3 runs) | 472 | 482 | 483-486 |
| decode cold 64 (ms/token) | 48.9 | 48.0-48.3 | 47.5-48.0 |
| decode cold 4096 (ms/token) | 53.2 | 52.6-53.1 | 46.6-47.1 |
| decode warm 64 (ms/token) | 38.9 | 36.6-37.1 | 36.2-36.9 |
| decode warm 4096 (ms/token) | 43.9 | 37.2-37.3 | 37.5-38.1 |

The warm decode repeats the same 32 tokens, which suits recency; S3-FIFO costs it 0.5-0.8 ms there and gains 6 ms in
cold decode at 4096 context, where its device hit rate rises from 71.8% to 77.4%.
