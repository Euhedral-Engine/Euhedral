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
