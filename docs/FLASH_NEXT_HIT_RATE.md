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
