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
  wait behind them.

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
