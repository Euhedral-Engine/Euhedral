# Flash-Next expert prefetch

A decode step's experts come from the disk when neither the device cache nor the host tier holds them, and each such
read is on the step's critical path. A prefetch reads, ahead of the request, the records a later layer is predicted to
ask for into the host tier, so the request copies them from memory.

## The bound

With the artifact reads skipped (every record costs only its copy to the device; a temporary experiment, not
committed), the screens give the most any prefetch could buy:

| scenario | reading the disk | disk skipped |
|---|---|---|
| prefill 512 / 4096 (tokens/s) | 136 / 487 | 463 / 983 |
| decode cold at 4096 (ms/token) | 46.3 | 24.6 |
| decode warm at 4096 (ms/token) | 37.6 | 24.5 |

## Predictors

Measured on recordings of real requests (`Qwen4DemandRecordingCudaIntegrationTest`, `tools/expert_prefetch_eval.py`):
the share of the next layer's experts a prediction names (recall).

| predictor | top 10 | top 20 | top 32 |
|---|---|---|---|
| the experts the layer used for the previous token | 31-36% | | |
| the layer's most requested experts | 16% | 26% | |
| experts that followed this layer's choice before (co-occurrence) | 40-43% | 56-61% | |
| the next layer's router applied to this layer's input | 63.8% | 81.7% | 88.5% |
| the router of the layer 2 ahead | 53.8% | 71.3% | 79.8% |
| the router of the layer 3 ahead | 48.9% | 65.7% | 74.7% |

Replayed with prefetch into the tier (`tools/expert_prefetch_sim.py`, every prefetch assumed in time), decode's demand
disk reads per token against the prefetch reads spent:

| predictor, reads per layer | demand disk (MB/token) | prefetch (MB/token) |
|---|---|---|
| none | 93.2 | 0 |
| co-occurrence, 1 (recording 1) | 86.6 (of 99.1) | 130.0 |
| router top 10, 1 | 64.4 | 74.5 |
| router top 10, 2 | 54.7 | 107.7 |
| router top 20, 4 | 33.3 | 326.2 |

Prefetching a prefill's records into the tier did not help (14.8 against 15.0 MB/token): the tier is full, and the
reads either face admission or replace records that are needed.

## In the engine

A decode step's graph gets, per layer, a side branch: `PREDICT` runs the next layer's router on the layer's input behind
the route on its lane, and the host stage `PREFETCH`, after it retires, ranks the best `K` and publishes the owner's
prefetch frame. The owner reads up to `B` of them that the device and the tier lack straight into tier slots
(`TierFill`, no device copy), only while the disk has a read to spare; speculative work never waits. A request that
finds its record being prefetched publishes itself again until it is in, then copies it from the tier.

Running the prediction inside the route stage put it on the plan's critical path: the prediction alone cost 1-2
ms/token and the prefetch saved only 0.4 ms. As a side branch the prediction costs nothing measurable.

Quick screens (ms/token):

| setting (K, B, distance) | cold 64 | cold 4096 | warm 64 | warm 4096 |
|---|---|---|---|---|
| off (two runs) | 48.0-48.5 | 47.5-48.0 | 36.4-36.7 | 37.3-38.6 |
| 10, 1, 1 (two runs) | 44.6-46.6 | 44.6-45.0 | 35.7-36.2 | 35.5-35.7 |
| 6, 1, 1 | 45.4 | 45.0 | 35.3 | 35.9 |
| 16, 1, 1 | 44.9 | 45.7 | 35.1 | 35.4 |
| 10, 2, 1 | 43.6 | 46.3 | 35.3 | 36.0 |
| 10, 1, 2 | 45.3 | 45.6 | 36.2 | 36.1 |
| 10, 2, 2 | 45.6 | 48.4 | 37.0 | 39.1 |
| 16, 2, 1 | 44.3 | 45.8 | 38.1 | 39.5 |

Prefetches read per token (10, 1, 1): 22-31, of which 9-18 are used by a request. The default is 10, 1, 1.
