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
never copied to the device whole. N-gram execution is not implemented.
