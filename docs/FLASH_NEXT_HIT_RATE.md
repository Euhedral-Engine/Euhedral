# Flash-Next hit rate and allocation

Where the memory goes and what it buys. Screens are `PerformanceCudaIntegrationTest` (prefill 512 and 4096, decode at 64 and
4096 context, cold and warm; 32 workers, the automatic host budget). Measurements on the desktop with a 16 GiB GPU
(PCIe Gen5 x16), 64 GiB of RAM and the artifact on a Gen4 x4 drive.

## Where a decode step's time goes

Warm decode at 4096 context, 27 ms/token: the route wait 12 ms (the device work up to each layer's router and its
readback), the acquire 9.5 ms (a layer with a device miss waits for the copy of a 2.64 MiB record, 160-190 us), the rest in
the experts. With the token repeated, so that the device hit rate rises from 87% to 93%, the step is 25.7 ms and the acquire
8.5 ms: the acquire is paid per layer that misses, not per record.

## Sensitivity (`tools/expert_device_prefetch_sim.py`, a recording of 8 requests, 2,048 decode tokens)

| device slots | device miss | misses/token |
|---|---|---|
| 2,700 | 32.2% | 154.6 |
| 3,092 (today) | 28.9% | 138.7 |
| 4,000 | 22.6% | 108.4 |
| 6,000 | 13.3% | 64.0 |

| tier slots | demand disk reads/token |
|---|---|
| 13,955 (today) | 10.6 |
| 16,000 | 6.6 |
| 18,000 | 4.1 |
| 20,000 | 2.3 |

The device is full: the engine's peak use leaves 756 MiB free of the 16 GiB, which is the 700 MiB the system keeps and the
kernel reserve in use. A GiB of device slots is worth about 15 misses/token; a GiB of tier about 1 demand read/token.

## Rejected: prefetch into the device cache

Replayed with an upper bound (every prefetch lands in time), the router's next-layer prediction brought into the device
cache:

| prediction, copies per layer | device miss | prefetch copies/token (used) |
|---|---|---|
| none | 28.9% | 0 |
| top 10, 2 | 28.3% | 50 (12) |
| top 10, 10 | 28.1% | 73 (17) |
| top 20, 10 | 28.5% | 245 (41) |

The records it brings in evict as many as it saves: not built.

## The host budget under the server's heap

The planner reserved the JVM heap's whole room to grow. A server started without `-Xmx` (the container's command, the
benchmark) may grow its heap to a quarter of the machine, 14.7 GiB here, though the engine's heap holds under 1 GiB; the
expert tier lost a third. The test JVM's heap (512 MiB) hid it. `-Peuhedral.test.heap=15752m` runs the screens with the
server's heap.

The reserve is now at most 2 GiB for the heap, 1 GiB native, and 6 GiB for the page cache the memory-mapped n-gram rows
settle at (5.1-6.1 GiB after a run); the machine keeps room for that cache, which the budget did not count before.

Paired screens, 3 forks alternating, no cgroup (the lowest `MemAvailable` seen was 12 GiB):

| scenario | before | after | change | forks ahead |
|---|---|---|---|---|
| tier (records) | 11,088 | 13,536 | +22.1% | 3/3 |
| prefill 512 (tokens/s) | 167.0 | 188.0 | +12.6% | 3/3 |
| prefill 4096 (tokens/s) | 788 | 827 | +4.9% | 3/3 |
| decode cold at 64 (tokens/s) | 26.7 | 27.3 | +2.2% | 2/3 |
| decode warm at 64 (tokens/s) | 31.8 | 34.3 | +7.9% | 3/3 |
| decode cold at 4096 (tokens/s) | 28.1 | 29.0 | +3.2% | 3/3 |
| decode warm at 4096 (tokens/s) | 34.5 | 35.9 | +4.1% | 3/3 |

Without the n-gram reserve the same heap reserve gave a 36 GiB budget here and a lowest `MemAvailable` near 6 GiB on
earlier runs: not kept.

## Disk time in prefill and the lookahead

A prefill chunk's block asks for nearly every expert of its layer (442 of 512 on average), but the owner learned which
ones only when the layer's router had run: the disk idled from the end of one layer's reads to the next layer's plan.
Measured before the lookahead (`disk:` lines of the screen, no cgroup): the reads were in flight 92% of the time in a
prefill of 512 tokens, 68% at 4096 and 54-68% at 16,384, at 6.8-6.9 GB/s while they were.

Order of the work in a prompt (`LayerMajorSpike`, a throwaway test: layers 1-47 over four chunks of 4,096 tokens, chunk
by chunk or layer by layer, the same single-layer primitive and host round trips for both):

| order | time | device misses | H2D | artifact reads |
|---|---|---|---|---|
| chunk by chunk | 38.6-47.1 s | 69,800-80,000 | 193-221 GB | 48-69 GB |
| layer by layer | 27.1-33.3 s | 28,900-29,900 | 80-83 GB | 25-26 GB |

A layer-by-layer prompt moves a third of the bytes, but a chunk of 4,096 rows already computes at about the pace of a
whole prefill layer today (108 ms per layer with every expert resident, against 104 ms), so the bytes are not what
bounds a chunk; the idle disk is.

The lookahead predicts the layer `D` ahead (1) from this layer's input: its router applied to the chunk's rows with the
router's own top-k, the experts any row names, most named first (`PREDICT`, `PREFETCH` stages of the prefill shapes, as
decode has them). The owner keeps these as wants and, whenever the disk has a read to spare and no fetch waits, loads
them into device slots through the ordinary fetch (tier directive, staging, reads), holding each loaded record's lease
until the layer is done (`aheadDone` from its finish), at most a third of the slots. The layer's own fetches then find
the records resident. `EUHEDRAL_QWEN4_AHEAD=D[,N]` (`0`: off).

Paired screens, 4 forks alternating, 3 GiB test heap, no cgroup:

| scenario | before | after | change | forks ahead |
|---|---|---|---|---|
| prefill 512 (tokens/s) | 192 | 207 | +7.8% | 4/4 |
| prefill 512 disk busy | 90.5% | 97.0% | | 4/4 |
| prefill 4096 (tokens/s) | 813 | 1,015 | +24.8% | 4/4 |
| prefill 4096 disk busy | 67% | 85% | | 4/4 |
| prefill 16384 (tokens/s) | 851 | 1,036 | +21.6% | 4/4 |
| prefill 16384 disk busy | 68.5% | 80% | | 4/4 |

Distance 2 against 1 (3 forks): prefill 4096 -7.6%, 16384 -17.3% (the prediction is worse and the holds longer): not
kept.

## Not kept

- **Zero-copy decode** (`ZeroCopySpike`, throwaway): the ten experts of a decode token read in place from the pinned
  tier over the bus take 1.07-1.17 ms; copied to the device first and then computed 0.94-0.99 ms; with the records on
  the device 0.27-0.48 ms. The bus is the bound either way (about 25 GB/s read in place, 29 GB/s copied), and the copy
  engine is faster than the kernels' loads: no gain, and a record read in place would never become a device hit.
- **Decode prefetch settings** under the larger tier (3 forks, 16 candidates and 2 reads per layer against 10 and 1):
  cold 64 -0.7%, cold 4096 -1.5%, warm 64 +3.9% (3/3), warm 4096 +1.9% (2/3), prefill unchanged: flat, as before.
- **Dropping the n-gram rows' page cache** before a run (the rows a fresh prompt asks for are not resident): prefill 4096
  -9.0% (0/3), 16384 +12.4% (2/3) against a warm cache; the runs were noisy (another tenant on the machine), so the
  cost of cold rows is not established beyond "up to 9%".
- **Lossless NVFP4 coding of the records** (`tools/nvfp4z.py`: 4.09 bits per weight against 4.5): 9% fewer bytes on the
  disk and the bus and 10% more records in the tier, for a decoder on the load path: not built.

## Copy streams

One copy stream queued a layer's copies one behind the other (a record alone copies in 71-77 us, 37 GB/s; the engine's
loads measured 188 us submit to retire). A record split over two to four streams copies in the same 71-77 us
(`CopySplitSpike`, throwaway: the link is the bound), so the gain is in the copies of different loads overlapping.
Paired screens (`EUHEDRAL_QWEN4_COPY_STREAMS`):

| streams, before to after | forks | prefill 4096 | decode cold 64 / 4096 | decode warm 64 / 4096 |
|---|---|---|---|---|
| 1 to 2 | 3 | +27.9% (2/3, noisy) | +3.8% / +1.7% | +4.3% / +4.3% |
| 2 to 4 | 3 | +6.3% (3/3) | -0.7% / +0.7% | -0.8% / -0.3% |
| 4 to 8 | 3 | +4.1% (2/3) | -2.9% / -5.1% | +2.5% / +2.8% |
| 1 to 4 | 4 | +4.8% (4/4) | +4.5% / +2.4% | +3.9% / +4.5% |

Four is the default. Prefill 4096 reads at 6.4 GB/s with the disk busy 96% of the time.

## Decode: where a token goes, and launching the experts together

A decode token at 4096 context is about 1,800 kernels after this change (2,700 before), and a layer takes about 450 us.
Bounds with the work removed (a temporary probe, results wrong, interleaved in one process, 24 pairs of 16 tokens):

| arm | ms/token | tokens/s |
|---|---|---|
| everything | 24.6-28.0 | 36-41 |
| fetches made, expert kernels not run | 19.2 | 52 |
| nothing fetched, no expert kernels | 16.4 | 61 |

So the attention chain and the router's round trip are two thirds of a token even with free experts. The chain is
memory-bound: about 5 GB of weights a token (the hyper-connection mixers 2.2 GB in BF16, the output head 1.3 GB, the
attention projections) at 650 GB/s of the card's 896. The misses cost the PCIe time of their records, about 185 MB a token
at 37 GB/s, 5 ms, none of it hidden: a layer's compute is 100 us and its missing expert arrives 135 us after the router.

Each expert was launched by a frame of its own: an 8-byte copy of its slot address, two kernel launches and a marker, 39 us
of host time for 10 us of kernels, so the ten experts of a layer took 160 us (kernels started 16-20 us apart). A decode
block now asks the cache for all its experts in one record (`FETCHALL`, `ExpertCacheOwner#requestAll`); the experts the
device holds are leased by the time the owner answers, and one stage (`GROUP`) launches them as runs of consecutive
experts, one launch of each kernel and one copy of the slot addresses per run (their work items are consecutive and each
names its own expert's slot: no kernel changed, the sums are the same). An expert that must be loaded keeps its own fetch
and expert stages and launches when it arrives. `EUHEDRAL_QWEN4_GROUP=0` launches each on its own.

`DecodeAbCudaIntegrationTest` takes two arms of the plan in turn, 16-32 tokens each, on one sequence in one process, so
that what moves on the machine cancels (`EUHEDRAL_QWEN4_AB=1,0`; variant bit 0 is per-expert launches):

| run | per-expert launches | grouped | change | pairs ahead |
|---|---|---|---|---|
| 1 | 25.45 ms (39.3 tok/s) | 22.54 ms (44.4) | +12.9% (+-1.4) | 31/32 |
| 2 | 23.99 ms (41.7) | 21.74 ms (46.0) | +9.4% (+-1.4) | 29/32 |
| 3 | 25.41 ms (39.3) | 22.17 ms (45.1) | +14.7% (+-1.4) | 32/32 |

### Not kept (same harness, 24 pairs)

- The plan asks for the experts itself, one stage fewer: +0.4% (+-1.8%).
- The owner applies the batch in the frame that posted it instead of waiting for a worker to poll it: +0.6% (+-2.2%).
- The route's device-completion edge polled through a lattice source instead of the driver's callback (paired screens, 3
  forks): decode cold 64 +0.7%, warm 64 0.0%, cold 4096 +2.1%, warm 4096 +0.3%.
- Lanes 1 or 4 against 2: within the 5% noise between processes.
- Protecting the experts the router predicts for the next layer from eviction (replay of the recording): the device miss
  rate is unchanged at 28.9%.
