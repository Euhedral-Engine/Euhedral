# Flash-Next host expert tier

The routed experts are about 63 GiB. They sit behind a device cache of a few thousand slots, and a device miss reads its record
from the artifact (the OS page cache, when the machine can keep it) unless the host holds it. The host side is a hierarchy with a
real ordinary-RAM tier:

```
GPU expert cache <-- async H2D --- pinned staging --- RAM copy --- RAM tier <-- cold read --- artifact (EDRL / NVMe)
 (device slots)      (one copy      (one slot per     (one memcpy   (pageable,    (positional reads,
                      stream per     copy lane)        of a record)  large)        one per missing record)
                      lane)
```

A GPU eviction does not evict from the RAM tier: once warm, a GPU miss is a RAM hit and no artifact read. Pinned memory stays a
small transfer tier (the staging slots, the host-mapped token embedding, any staged fixed objects); the experts are never pinned.

## Memory planning

`HostBudget` carries two budgets drawn from the same physical memory:

| Budget | Meaning |
| --- | --- |
| `pinnableBytes` | memory the runtime may page-lock |
| `residentBytes` | ordinary memory the runtime may hold; pinned memory is carved out of it, not added to it |

The pinned budget comes from the machine (`MemAvailable`, less room for the operating system: 10%, never closer than 4 GiB to
exhaustion). The ordinary-memory budget is never assumed: the engine takes none that the user did not offer. The user states it with
`EUHEDRAL_HOST_MEMORY_MIB` (or `-Deuhedral.host.memory-mib`), and the operating system's margin is still subtracted from it. Unset,
there is no RAM tier and every device miss reads the artifact (`FILE_BACKED`). The staging depth, transfer concurrency, wave geometry
and chunk size remain derived.

The planner spends the pinned part first, then gives the experts what the resident budget has left:

| Mode | When | Behaviour |
| --- | --- | --- |
| `RAM_RESIDENT` | every record fits | all records are read once at startup by several readers in 32 MiB chunks of consecutive records; no routed expert is read from the artifact during inference |
| `RAM_CACHED` | more records than the device cache but not all | bounded, lazily filled, layer-aware replacement |
| `FILE_BACKED` | the tier would not exceed the device cache | every device miss reads the artifact |

The tier's memory is one anonymous mapping, committed as records arrive (an arena allocation where the platform has no `mmap`), cut
into equal page-aligned slots.

## The tier is sharded with the device cache, and owned the same way

An expert is a `(bank, expert)` key, and the key's hash chooses its shard. The device cache and the tier use the same hash, so the
tier shard with index *i* holds the records of the experts that device shard *i* serves, and it is owned by the same serial lattice
source (`ExpertSource`). Its directory, slot states, pins and recency lists are plain arrays touched only from that source's
`request` and `pull`, one thread at a time: no lock, no atomic on the path. Shards share nothing but the memory block and the key
space, so the path through the tier runs side by side on as many shards as there are.

Before a load frame is published the source asks its tier shard what the load does, and gives the answer to the frame in the lane's
`TierDirective`:

| Directive | The load frame |
| --- | --- |
| `HIT` | copies the record out of the tier slot into the lane's staging slot; the slot is pinned until the load reports |
| `FILL` | reads the artifact into the slot reserved for the record, then copies it into the staging slot; the slot is pinned until the load reports |
| `BYPASS` | reads the artifact into the staging slot alone (the record is being filled, or every slot is pinned) |

The frame touches only memory the directive names. The slot's state changes in the source, when the load reports that its record
was staged (`filled` or `used`) or could not be read (`abandoned`, which returns the slot). A slot that is filling or pinned is
never a victim.

## Replacement: layer-aware

The model visits layer 0 to 47 and starts again, so the reuse distance of an expert is a whole pass and a global LRU evicts exactly
what the next pass needs. `BANK_PARTITIONED` gives each layer a quota of a shard's slots in proportion to its experts, lets a layer
borrow the slots of layers that are not using theirs, takes a victim from the layer furthest over its quota when a layer under its
quota needs one, and otherwise evicts the layer's own least recently used record. `GLOBAL_LRU` is the baseline; `RamTierTest` runs
a cyclic trace through both.

## Startup fill

A resident tier is filled before the model serves anything, by `RamTierPreload`: 16 reader threads take chunks of consecutive
records of one bank (about 32 MiB) from a shared counter and read each record straight into its slot. The threads end with the load;
nothing else in the tier starts a thread.

## Telemetry

`Qwen4Model.hierarchyStats()` keeps the tiers apart: the GPU cache, the RAM tier by layer (hits, misses, bypasses, evictions,
resident experts and bytes, preload throughput), artifact reads (records, bytes, read time, concurrent-read high-water mark), the
staging copies (records staged, RAM-to-pinned bytes and time) and H2D (bytes, time, copies). The tier's counters are written by the
shard's owner and read from anywhere, so a reading under load is approximate.
