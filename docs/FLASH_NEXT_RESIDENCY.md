# Flash-Next storage, residency and the expert cache

The user chooses the artifact and the maximum context. Everything else is derived at load time from the device's free memory,
the host's pinnable memory, the context, and the sizes of the artifact's actual objects: where every object lives, how large the
routed-expert cache is, how experts reach it. No setting exposes the cache size, the offload, the staging depth or the host
stores.

```
artifact -> Qwen4ArtifactReader / Qwen4Validator -> Qwen4ResidencyPlanner -> Qwen4Model
                                                      |                         |
                                                      |                         +- fixed objects (device / staged / mapped)
                                                      |                         +- NgramStore (host)
                                                      +- expert cache budget -> +- ExpertCache <- HostExpertStore <- artifact file
```

`Qwen4Storage.load(InferenceConfig)` runs this from the engine's own configuration (artifact path, maximum context). It runs no
model: `InferenceEngine.load` refuses a `qwen4_exp` artifact up front with the same message that names `Qwen4Storage`.

## Storage classes

| class | meaning | used for |
|---|---|---|
| `DEVICE_RESIDENT` | permanently in device memory | routers, norms and control weights, hyper-connections, GDN, QSA, shared experts, output head |
| `HOST_MAPPED` | pinned host memory that kernels read in place | token embedding (a row gather); the output head when the context needs the room |
| `HOST_STAGED` | held on the host, staged to a device ring (a fixed projection) or gathered by rows (n-gram) on use | projections moved off the device; the n-gram tables |
| `DEVICE_CACHED` | brought into a bounded device cache on demand | routed experts |
| `DEFERRED` | in the artifact and validated, not loaded | MTP and vision unless the mode selects them |

Fixed objects keep the dense engine's `TensorHandle` (device address, host address, mapped flag) and its staging ring
(`WeightStaging`, 4 slots). A cached expert is not a handle: it is a *lease* on a slot (`ExpertLease`), valid until closed.

## Fixed-memory planner

`Qwen4ResidencyPlanner` is the fixed-memory half. It keeps no replacement state.

```
free device memory
  - runtime reserve (CUDA context, kernel modules, graph pools)           1,024 MiB
  - workspace of one execution step                                         130 MiB
  - sequence state of the maximum context (KV, indexer keys, GDN state)
  - fixed objects placed on the device
  - the staging ring of fixed objects moved to the host
  = expert cache, in whole slots (never more than one slot per expert)
```

Sequence state comes from the configuration (`Qwen4SequenceState`): KV is NVFP4 pages of 256 tokens, 576 bytes per token in each
of the 12 sparse-attention layers (1.69 GiB at 262,144 tokens); the indexer keeps one BF16 key of 128 values per 4 tokens per
layer (192 MiB at 262,144); GDN keeps 110 MiB regardless of length. The workspace is sized from the topology (hyper-connection
streams, the widest mixer's intermediates, router and selected-expert activations for a 512-token chunk, decode logits); it is a
bound to be tightened with measured high-water marks when execution exists.

Context has priority. A longer context takes memory from the expert cache first. Only when the cache would fall below its
minimum (`2 x experts-per-token` = 20 slots: the selected experts of a token and a second set in flight) does a fixed object
move to the host, lowest priority first and the fewest bytes that suffice. The planner never moves a fixed object to enlarge the
cache: every fixed byte is read on every token, whereas a cached expert is hit only on some tokens. The crossover points are
therefore computed from byte budgets, not thresholds. A context is refused only when the minimum cache does not fit even with
every movable object on the host; the plan then says which sizes disagree.

Device-residency priority, highest first, and the order objects leave the device (the reverse):

| priority | objects | can leave the device |
|---|---|---|
| 1 | routers, norms, control vectors, every object up to 4 MiB (GDN convolutions and gates, hyper-connection norms and inject weights) | never |
| 2 | hyper-connection mixer projections (BF16, 6.6 MB each) | last, staged |
| 3 | GDN projections | staged |
| 4 | QSA projections and the indexer matrix | staged |
| 5 | shared experts | staged |
| 6 | output head (1.27 GB) | mapped, read in place: a staging slot would have to hold all of it |
| 7 | the n-gram projections of layer 1 | first, staged |

Within a rank the planner takes whole families (the same object in every layer) smallest tensor first and spreads the last family's
layers evenly, so the staging ring holds the smallest possible tensor and transfers interleave with resident work. The token
embedding is always `HOST_MAPPED` when the host can pin it. MTP and vision count only when the mode selects them.

### Host

Pinned memory is the host's `MemAvailable` less 10% (at least 4 GiB), taken in this order: the mapped embedding, staged fixed
objects, the experts. The expert store is a single pinned huge-page arena when every expert record fits in what is left, as for
`ArenaExpertStore`; otherwise records stay in the artifact file (and the operating system's page cache) and pass through a small pinned
staging pool (`FileExpertStore`, `max(16, 2 x experts-per-token)` slots of one record). The n-gram tables are mapped read-only from the
file unless the experts are pinned too and there is room: they are gathered a few rows per token, and unpinned memory is worth more as
the experts' page cache.

## Expert cache

`ExpertCache` holds `slotCount x slotBytes` device bytes in one allocation, sliced into equal slots that each hold the largest record
(2,768,896 bytes). Keys are `(bank ordinal, expert)`; the directory is an `int[]` over all experts and slot metadata is in primitive
arrays, so 25,088 experts cost no objects. States: empty, loading, resident, in use. Replacement is least recently used over
resident, unpinned slots only.

- A hit pins its slot and returns a lease; a miss reserves a free slot, else evicts the least recently used one, opens the record in
  the host store and starts the transfer outside the cache's lock.
- Concurrent misses for one expert coalesce into one transfer; each waiter holds its own claim.
- A loading or leased slot is never evicted or refilled. An interrupted or timed-out acquirer drops only its own claim; a started
  transfer completes and the expert stays resident. A failed transfer fails its waiters, empties the slot and leaves nothing behind.
- `lease.close(fence)` records that device work read the slot; the next refill's copy stream waits for the fence on the device (the host
  never waits), so a refill cannot overtake the kernels reading the slot.
- `GpuExpertTransfer` copies on its own stream and completes on a thread that is not the CUDA driver's.
- `close()` rejects new requests, waits for transfers in flight, invalidates open leases and frees the slab, the stores and the stream
  exactly once.

Telemetry (`Qwen4Model.telemetry()`): fixed device bytes, fixed host-backed bytes, host-mapped bytes, context reserve, expert-cache
bytes and slots, hits, misses, evictions, transfer bytes, transfer time, wait time, n-gram bytes staged and rows gathered.

## N-gram tables

`NgramStore` holds the 128 shards on the host and addresses rows by hash head and entry: a head covers `heads_vocab_sizes[head]` rows from
`heads_offsets[head]`, a global row is `shard x 2,500,012 + local row`, and a row is 90 contiguous bytes. `gather` copies chosen rows into
host memory; `stage` copies only those rows through a pinned upload buffer to the device. The tables never enter the expert cache and are
never copied to the device whole. Execution gathers the rows as 96-byte records ([FLASH_NEXT_EXECUTION.md](FLASH_NEXT_EXECUTION.md)).

## Measurements

### Placement matrix

Expert slots by free device memory and maximum context, planned at the real object sizes (`Qwen4ResidencyPlannerTest`, text mode,
host with 48 GiB available); `(+N)` is the MiB of fixed objects moved to the host to make room for the minimum cache of 20 slots.
Fixed objects that stay resident take 4,190 MiB; the cache holds at most one slot (2.64 MiB) per expert, 24,576.

| free device memory | 4,096 | 16,384 | 32,768 | 65,536 | 131,072 | 262,144 |
|---|---|---|---|---|---|---|
| 3 GiB | 35 (+2573) | 22 (+2630) | 25 (+2756) | 31 (+3012) | 43 (+3525) | does not fit |
| 4 GiB | 31 (+1538) | 22 (+1606) | 25 (+1732) | 30 (+1985) | 44 (+2503) | 41 (+3456) |
| 5 GiB | 305 (+1226) | 270 (+1226) | 225 (+1226) | 134 (+1226) | 44 (+1479) | 20 (+2376) |
| 6 GiB | 249 | 215 | 170 | 79 | 340 (+1226) | 43 (+1412) |
| 7 GiB | 637 | 603 | 557 | 467 | 285 | 364 (+1226) |
| 8 GiB | 1,025 | 991 | 945 | 854 | 673 | 309 |
| 15 GiB | 3,739 | 3,705 | 3,660 | 3,569 | 3,387 | 3,024 |
| 24 GiB | 7,229 | 7,195 | 7,150 | 7,059 | 6,877 | 6,514 |

The cache shrinks as the context grows until it would fall under 20 slots; then the first fixed objects leave the device (the n-gram
projections, then the output head, 1,213 MiB, read in place from mapped host memory), and the room they free is given to the
cache, so the slot count jumps up at that point and falls again as the context grows. Objects up to 4 MiB never move, so the
smallest device that can serve a context is the sum of its sequence state, the 1,154 MiB of runtime reserve and workspace, the 333 MiB
of objects that never move, the staging ring and 20 slots (3 GiB serves 131,072 tokens, 4 GiB 262,144).

### The GPU of this machine

RTX 5070 Ti, 15,030 MiB free after the CUDA context, host with 48 GiB available (`Qwen4StorageCudaIntegrationTest`, the real
artifact):

| maximum context | KV | indexer | context and reserve | fixed resident | expert cache | slots |
|---|---|---|---|---|---|---|
| 4,096 | 27 MiB | 3 MiB | 1,294 MiB | 4,190 MiB | 9,543 MiB | 3,614 |
| 32,768 | 216 MiB | 24 MiB | 1,504 MiB | 4,190 MiB | 9,334 MiB | 3,535 |
| 131,072 | 864 MiB | 96 MiB | 2,224 MiB | 4,190 MiB | 8,619 MiB | 3,264 |
| 262,144 | 1,728 MiB | 192 MiB | 3,184 MiB | 4,190 MiB | 7,657 MiB | 2,900 |

Host: the token embedding is mapped (1,212 MiB pinned); the 63 GiB of expert records exceed what the host can pin, so they stay in the artifact
file behind 20 pinned staging slots (52 MiB) and the page cache; the n-gram tables (27,465 MiB) are mapped from the file; MTP (1,431 MiB) and
vision (856 MiB) stay in the artifact. Pinned memory in all: 1,265 MiB. With the device limited to 5 GiB at 262,144 tokens the same
load keeps 1,813 MiB resident, stages 1,164 MiB of projections through a 67 MiB ring, maps the output head (2,425 MiB mapped in all) and
runs a cache of the minimum 20 slots.

### Expert transfers

On a real stream (`Qwen4ExpertCacheCudaIntegrationTest`: 6 banks of 64 experts, 8 slots, records of 2.7 MB): every record read back from the
device equals the file's bytes and CRC-32; 369 file-backed misses moved 1.00 GB at 8.7 GB/s and 373 arena-backed misses 1.02 GB at 9.3 GB/s,
measured as bytes over the summed start-to-completion times of single 2.7 MB copies, so each includes the retirement latency of its own
event. Against the real artifact on the load test above, 1,820 misses (5.04 GB) took 0.27 to 0.33 s of summed transfer time, 15 to 18 GB/s.
These are not peak pipelined rates, which are not measured here; the huge-page arena's 44 GB/s ([NVFP4_RESIDENCY.md](NVFP4_RESIDENCY.md))
applies to large pipelined copies of the arena store. Loading the 1 GB arena from a cached file took 0.42 s with 16 readers.
