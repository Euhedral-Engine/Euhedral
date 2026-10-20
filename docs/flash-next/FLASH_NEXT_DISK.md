# Flash-Next expert disk loading

The goal is to move an expert from request to delivery as fast as the hardware allows: until the disk (a Samsung 990
Pro, PCIe Gen4 x4) or the PCIe link is the limit. Every number here comes from the RTX 5070 Ti machine with the
artifact on the 990 Pro, 32 lattice workers (every logical processor), `RAM_CACHED` at 22,000 MiB, unless it says
otherwise. Rejected variants are results and are kept.

## The drive

O_DIRECT `preadv` from Python threads, one read in flight per thread, over the 68 GB artifact (`ddbench2.py MODE THREADS
KIB`):

| read size | 2 in flight | 4 | 8 | 16 |
|---|---|---|---|---|
| 128 KiB, sequential | 1.79 | 3.00 | 4.64 | 4.93 |
| 672 KiB (a quarter record), random | 4.22 | 5.04 | 5.12 | 4.92 |
| 2.7 MiB (a record), random | 6.89 | 6.88 | 6.84 | 5.46 |
| 8 MiB, sequential | 7.12 | 7.07 | 5.90 | 5.37 |
| 32 MiB, sequential | 7.09 | 6.37 | 5.82 | 4.99 |

GB/s. Sequential and random reads are the same at a record and above. A record read whole reaches 6.9 GB/s; a quarter
record caps at 5.2 GB/s whatever the depth. More than about 20-30 MB in flight lowers the rate to about 5 GB/s. The
ceiling is 7.1 GB/s, in reads of 8 MiB or more with two to four in flight.

## The expert path at the start

Prefill of 4096 tokens, 64 records in flight, each read in four parts:

- 424 tokens/s; the disk ran at a median of 4.0 GB/s while reading (max 4.9).
- Per load: read 19 ms, submit hop 0.8 ms, copy 7.8 ms, retire hop 4.5 ms.
- 5,300 fetches per token found the read bound or the staging pool full and published themselves again: about 2.4
  million frames a second, all routed to the cache's owner, whose frames run in order. The retirements that free a read
  wait behind them. (Since replaced: a fetch that finds the read bound or the staging pool full waits in the cache owner's
  source, which asks it again only when a read, a buffer or a slot was given back.)

## Startup fill

The tier starts with each layer's share of slots filled by its lowest experts, read in runs of up to 8 MiB straight into
consecutive slots (`RamTierPreload`, four loading threads, blocking O_DIRECT reads): 7,776 records, 20.1 GiB at
7.09 GB/s, the drive's ceiling. A preloaded record gives way to the first request of a record the tier lacks, as a free
slot would, so the tier adapts as an empty one does.

## Screens

Single prefill-only runs (`EUHEDRAL_QWEN4_PERF_PREFILL_ONLY=1`), tokens/s for 512 and 4096 tokens:

| arm | 512 | 4096 | disk median / max GB/s |
|---|---|---|---|
| 4 parts, 64 reads (start) | 114 | 424 | 4.03 / 4.94 |
| whole records, 64 reads | 116 | 419 | 3.97 / 5.15 |
| whole records, 16 reads | 116 | 401 | 3.86 / 5.48 |
| whole records, 8 reads | 126 | 439 | 4.23 / 6.74 |

With the co-tenant `ollama` server stopped (it swung single runs by about 20%):

| arm | 512 | 4096 | disk median / max GB/s |
|---|---|---|---|
| no startup fill | 113 | 424 | 4.06 / 4.96 |
| startup fill | 110 | 422 | 3.91 / 7.18 (the fill) |
| startup fill, 32 lake sinks | 107 | 419 | 3.87 / 7.16 |
| startup fill, 64 lake sinks | 107 | 422 | 3.94 / 7.13 |
| 32 lake sinks, whole records, 8 reads | 131 | 449 | 4.75 / 7.20 |

The startup fill is neutral once a warm run has filled the tier (the benchmark warms it before measuring); it serves the
first requests after a start. More lake sinks shorten the submit hop (1.1 ms to 0.3 ms) but do not move prefill.

## A read is held only while it reads

The owner counted a load as reading until its copy retired: with 8 records in flight a read took 1.9 ms, and the copy
and the retire hop held it another 8.9 ms. The count became a physical resource, given back by the frame that has the
record. With it (32 lake sinks):

| reads in flight | 512 | 4096 | disk median / p90 GB/s |
|---|---|---|---|
| 4 | 143 | 467 | 4.36 / 7.10 |
| 8 | 145 | 465 | 4.34 / 7.07 |
| 12 | 144 | 465 | 4.35 / 7.08 |
| 16 | 119 | 406 | 3.79 / 7.07 |
| 8, completions from 8 ingest sources | 87 | 311 | 3.00 / 6.53 |

While reads are in flight the disk runs at 6.95-7.0 GB/s: in prefill 512 it has reads 91% of the time, in prefill 4096
61%. The rest of the time no expert is being read.

## Gates

Paired, 6 forks, a fresh process per arm, alternating order; control `5e2e9c7`, candidate `7a86589`:

| gate | scenario | control | candidate | change | ahead |
|---|---|---|---|---|---|
| bundle | prefill 4096 (tokens/s) | 433.5 | 477.5 | +10.2% | 6/6 |
| | prefill 512 (tokens/s) | 111 | 147 | +32.4% | 6/6 |
| | disk median (GB/s) | 4.15 | 4.59 | +10.7% | 6/6 |
| whole records, 8 reads, alone | prefill 4096 | 440 | 472.5 | +7.4% | 6/6 |
| | prefill 512 | 110.5 | 134 | +21.3% | 6/6 |
| early read release, alone | prefill 4096 | 473 | 489 | +3.4% | 6/6 |
| | prefill 512 | 134 | 148 | +10.5% | 6/6 |

The disk-busy measure (`AsyncReads.busyNanos`): while reads are in flight the drive reads at 6.92-6.99 GB/s, its ceiling.
Fetches that find the cache full in prefill 4096 wait for one of the 8 reads (5,300 per token), a few for a staging buffer
(60), none for a device slot; they wait only while the disk is busy. When the disk idles (37% of prefill 4096, 9% of
prefill 512) nothing is waiting to be read: the next layer's experts are not known until its router runs.

## Decode

Quick screens (about 50 s each, one process): whole records against quarter-record parts, 8 reads in flight:

| scenario | whole records | 4 parts |
|---|---|---|
| decode cold 64 / 4096 (ms/token) | 47.9 / 52.6 | 50.9 / 55.1 |
| decode warm 64 / 4096 (ms/token) | 38.3 / 43.6 | 42.6 / 46.0 |
| prefill 512 / 4096 (tokens/s) | 148 / 486 | 120 / 419 |

Splitting a record does not deliver it sooner in decode either. A decode miss spends about 0.4 ms reading (one record at a
time from the disk takes about 1 ms), 0.1-0.15 ms in its copy, and microseconds in the hops between its frames. One fork of
a full gate against `5e2e9c7`: decode cold 64 / 4096 51.4 / 55.3 to 48.3 / 51.1 ms/token, warm 42.2 / 46.2 to 38.4 / 42.3.

