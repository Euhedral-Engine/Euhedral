# Flash-Next host expert tier

The routed experts are about 63 GiB. They sit behind a device cache of a few thousand slots, and a device miss reads its record
from the artifact (the OS page cache, when the machine can keep it) unless the host holds it. The host side is a hierarchy with a
real ordinary-RAM tier:

```
GPU expert cache <-- async H2D --- RAM tier (pinned) <------------- direct reads --- artifact (EDRL / NVMe)
 (device slots)   \                                                                    (io_uring, O_DIRECT, in parts)
                   `-- async H2D --- pinned staging <-------------- direct reads ---'
                                     (one buffer per read in flight: records the tier does not take)
```

A GPU eviction does not evict from the RAM tier: once warm, a GPU miss is a RAM hit and no artifact read. When the machine can
page-lock the tier beside the plan's other pinned memory, the tier is pinned (`EUHEDRAL_QWEN4_PIN_TIER=0` keeps it pageable): the
device's copy reads a record straight from its slot, and a record the tier takes is read straight into its slot, so neither needs a
staging buffer. A pageable tier copies a hit into a staging buffer first.

## Memory planning

`HostBudget` carries two budgets drawn from the same physical memory:

| Budget | Meaning |
| --- | --- |
| `pinnableBytes` | memory the runtime may page-lock: what the machine can spare |
| `residentBytes` | ordinary memory the runtime may devote to residency and caches; pinned memory is carved out of it, not added to it |

By default the budget is automatic: what the machine can spare now (`MemAvailable`, or the headroom of the process's cgroup and
its ancestors when that is less), less room for the operating system (10%, never closer than 4 GiB to exhaustion) and for the
runtime itself (the JVM heap's room to grow, at most 2 GiB; 1 GiB for native allocations; and 6 GiB of page cache for the
n-gram rows, which settle there in a run). `EUHEDRAL_HOST_MEMORY_MIB` (or
`-Deuhedral.host.memory-mib`) states the budget instead and is respected as stated: smaller than the automatic one (0 means no RAM
tier, so every device miss reads the artifact, `FILE_BACKED`) or larger, with a warning when the machine cannot spare that much
now. The budget controls memory only: the tier's records, the staging buffers, the copy streams and the chunk size are derived within it. The startup report names the budget and where it came from.

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
tier shard with index *i* holds the records of the experts that device shard *i* serves. Both belong to the cache's owner
(`ExpertCacheOwner`): their directories, slot states, pins and recency lists are plain arrays changed only while a worker polls the
owner, a source of the lattice that applies the records posted to it (a request, a release, a retirement); the lattice never polls a
source twice at a time, so there is no lock and no atomic on the path.

When a miss becomes a load the owner's fetch asks the tier what the load does, and gives the answer to the load's frames in its
`TierDirective`:

| Directive | The load's frames |
| --- | --- |
| `HIT` | copy the record to the device from its slot (pinned tier), or out of it into the load's staging buffer; the slot is pinned until the copy retires |
| `FILL` | read the artifact into the slot reserved for the record and copy it from there (pinned tier), or on into the staging buffer; the slot is pinned until the copy retires |
| `BYPASS` | read the artifact into the staging buffer alone (the record is being filled, every slot is pinned, or admission refused it) |

The read and copy frames touch only memory the directive names. The slot's state changes in the owner's frames, once the copy
retired (`filled` or `used`) or the record could not be read (`abandoned`, which returns the slot). A slot that is filling or
pinned is never a victim.

The fetch reserves everything a load needs before it asks the tier: the device slot, a staging buffer (given back at once when a
pinned tier takes the record) and, for a record the tier does not hold, one of the disk's reads in flight. A fetch that finds any
of them taken waits in the owner and leaves the tier as it was: planning a fill takes a slot and evicts its record, and a fetch
that waits for the disk would otherwise evict a record on every try.

## Replacement: layer-aware

The model visits layer 0 to 47 and starts again, so the reuse distance of an expert is a whole pass and a global LRU evicts exactly
what the next pass needs. `BANK_PARTITIONED` gives each layer a quota of a shard's slots in proportion to its experts, lets a layer
borrow the slots of layers that are not using theirs, takes a victim from the layer furthest over its quota when a layer under its
quota needs one, and otherwise evicts the layer's own least recently used record. `GLOBAL_LRU` is the baseline; `RamTierTest` runs
a cyclic trace through both.

## Admission

A prefill visits nearly every expert of a layer once per chunk, so recency alone replaces each record just before the next chunk
needs it. With admission (the default; `EUHEDRAL_QWEN4_TIER_ADMISSION=0` turns it off), a record that finds no free slot replaces
the victim only if it was asked for more often than the victim was; otherwise it is read around the tier (`BYPASS`). The request
counts are halved every ten requests per slot, so a record that was hot long ago does not keep its slot. `RamTierTest` runs
repeated sweeps through both: recency serves none of them, admission half.

## Startup fill

A resident tier is filled before the model serves anything, by `RamTierPreload`: 16 reader threads take chunks of consecutive
records of one bank (about 32 MiB) from a shared counter and read each record straight into its slot. The threads end with the load;
nothing else in the tier starts a thread.

## Reads in parts

Every record of the artifact is 4096-aligned in offset and length, so the artifact is opened with `O_DIRECT`
(`EUHEDRAL_QWEN4_DIRECT_READS=0` reads through the page cache): the reads go from the device to the slot or buffer, and the page
cache keeps the memory-mapped n-gram rows instead of records the tier already holds. At most `EUHEDRAL_QWEN4_READS` records (8)
are read at once: the owner's fetch takes a read, and the frame that has the record gives it back on whichever worker runs it,
before the device copy, so the disk is offered the next record as soon as one is in. The drive reads fastest with a few records
in flight and slows past about 30 MB (docs/FLASH_NEXT_DISK.md).

A record the tier does not hold is read in page-aligned parts, each straight into its destination (the tier slot of a fill, or the
staging buffer), and no worker waits for the disk: each part's frame submits its read to the kernel (io_uring, through
`AsyncReads`) and ends, and the read's completion is a frame that the workers find when they poll the reads' sink. A fill's
completion then copies its range into the staging buffer. Whichever part runs first takes the staging buffer, and the others use
it; the parts join (`Join`, a fan-in edge whose arrivals are decided at run time) into the frame that submits the device copy. The
artifact sees up to the read bound of records in flight. A record is one part by default (`EUHEDRAL_QWEN4_READ_PARTS`): the drive
reads a whole record at 6.9 GB/s and a quarter record at 5.2 GB/s at most. A source that cannot read ranges is read whole, and a
machine without io_uring reads each part on the worker that runs it.

## Device cache replacement

The device cache can replace by the same layer quotas as the tier (`ReplacementPolicy.BANK_PARTITIONED`). The global recency
order stays the default: on the model's decode the two keep the same experts (a few popular experts per layer dominate, and no
cyclic pass evicts them), and `ExpertCacheShardTest` shows where they differ, on a trace that does cycle.

## Telemetry

`Qwen4Model.hierarchyStats()` keeps the tiers apart: the GPU cache, the RAM tier by layer (hits, misses, bypasses, evictions,
resident experts and bytes, preload throughput), artifact reads (records, bytes, read time, concurrent-read high-water mark), the
staging copies (records staged, RAM-to-pinned bytes and time, staging buffers pinned) and H2D (bytes, time, copies). The cache's
owner also times its loads (`ExpertCacheOwner.Timings`): from the fetch to the first frame, the read or copy, the hop to the
owner's submit, the copy itself, and the hop from its retirement callback to its frame; and it counts the fetches that found the
cache full. The performance record prints them per load and per token. The tier's counters are written by the shard's owner and read from anywhere, so a
reading under load is approximate.
