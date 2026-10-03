# Compressed Q3 weights (P2E2)

P2E2 is the lossless layout of the `q3-compressed` artifact: the Q3G64_F16S tensors of the `q3` artifact stored in 81.5% of their
row-split bytes (persistent layout `row-split-p2e2-v1`, `WeightLayout.ROW_SPLIT_P2E2_V1`). It keeps them resident on the GPU in
less memory, with the same values. Each kernel decodes only the codes it consumes, in registers.

The artifact selects the layout per tensor, so no engine option exists: the engine reads the layout from the tensors
(`ArtifactProfile`) and every Q3 route produces the same bits as on the `q3` artifact. See
[COMPACT_Q3_REFERENCE.md](COMPACT_Q3_REFERENCE.md) for the artifact.

## Performance

On an RTX 5070 Ti (16 GB):

- **Memory.** The 195 Q3 tensors of the `q3` artifact are 8.45 GiB of its 11.72 GiB payload; at 81.5% they free about 1.56 GiB
  (derived from the measured ratio). The expansion scratch takes back at most 128 MiB of weights plus the activation region (see
  Execution routes), so the net is about 1.4-1.5 GiB more room for KV pages.
- **Decode** (the fused P2E2 kernels): one row decodes at 58.9 tok/s at a 64-token context, 57.6 at 1024 and 56.1 at 4096, 128
  generated tokens, greedy. With MTP2 (three-row verification) the model decodes 95.0 tok/s at 4K, 97.0 at 16K, 91.1 at 32K,
  82.0 at 59K and 60.7 at 128K (chat corpus, 128 generated tokens; 64 at 128K). On cold weights at the model's shapes the kernels
  take, for gate_up (34816 x 5120) and down (5120 x 17408): one row 91 and 52 us, two rows 111 and 67 us, three rows 134 and 74 us,
  four rows 174 and 102 us.
- **Prefill** (more than eight rows) expands each Q3 tensor into the row-split layout once per launch, at about the DRAM rate:

  | Tensor | Expansion time |
  |---|---|
  | mixer | 18 us |
  | gate_up | 160 us (790 GB/s read + write) |
  | down | 73 us |
  | LM head | 1.2 ms |

  That is about 0.25 ms per layer per prefill chunk, on top of the unchanged row-split kernels.

**Why decode cannot be free.** On L2-resident data the contiguous kernel decodes about 2.85e12 codes/s, only about 1.4x its
DRAM-fed rate. A decoder of a 20%-smaller layout therefore keeps up only if it spends no more than about 1.3x the instructions per
code. Any variable-length code that keeps the FP32 FMA order, which bitwise identity requires, must place each payload unit at its
code's position. That costs a rank per pair: a POPC, a funnel shift, a mask and a table lookup, about 11 instructions per pair
against 7.

**Decoding inside the prefill GEMM's weight producer** so that no expanded tensor is needed. On the former BF16 tile engine's B
stage at 512 rows (bitwise identical, relative to the row-split kernels, mixer 5120x6144 / gate_up / down split-K): 0.53x / 0.63x
/ 0.50x, improved at best to about 0.60x / 0.58x / 0.57x. The bounds show why: with no decode arithmetic 0.61x / 0.60x / 0.58x,
with no payload loads 0.82x / 0.80x / 0.77x, with payload loads at fixed addresses 0.70x / 0.62x / 0.66x. The extra loads and
shuffles each K32 block needs cost more than the arithmetic. Expansion into the row-split layout runs at about the DRAM rate
(table above), so prefill expands first.

**General-purpose codecs (GDeflate and ANS).** GDeflate (nvCOMP / DirectStorage) is Deflate laid out so that a warp decodes a 64 KB
page in parallel. nvCOMP decompresses whole pages into a separate buffer, so it could only replace the expansion route; decode
reads every Q3 weight once per token. nvCOMP 5.3 measurements on layer-10 tensors, every decompression exact; ratios are relative
to the row-split tensor (P2E2 alone is 0.815):

| Input | Codec | Ratio | gate_up decompression | down decompression |
|---|---|---|---|---|
| row-split | GDeflate, entropy only (type 0) | 0.885 | 2.04 ms (35 GB/s) | 1.65 ms |
| row-split | GDeflate, maximum ratio (type 5) | 0.885 | 2.19 ms | 1.89 ms |
| row-split | nvCOMP ANS | 0.893 | 0.47 ms | 0.28 ms |
| P2E2 | GDeflate, type 0 | 0.765 | 1.97 ms | 1.69 ms |
| P2E2 | nvCOMP ANS | 0.771 | 0.45 ms | 0.24 ms |

zlib level 9 per 64 KB page agrees with nvCOMP on the codes: 0.91 of the packed planes. One code per byte reaches 0.79, and only
with Huffman-only coding. The scale plane reaches 0.57, which nothing can read in place. GDeflate alone compresses less than P2E2
and decompresses 12x slower than P2E2's expansion (160 us for gate_up); stacked on P2E2 it would save another 6% of the Q3 bytes,
but only through expansion, even for decode.

## Rejected: faster multi-row decode

Variants of the three-row kernel, timed on cold gate_up (three rows: 134 us as shipped):

- **Two weight rows per warp instead of four**, for fewer live registers and more occupancy: 191 us at three rows (288 us at two,
  225 us at four).
- **Activations in two halves of 16 values**, so the registers hold half as many x values: 131 us at three rows, 172 us at four
  (149 us at four rows with three CTAs per SM). A 2% gain at three rows for a restructured loop.
- **Prefetching the next slice's primary words, scales and payload window** ahead of the current slice's FMAs: 134.5 us (one row
  95.8 us). The loop is limited by instructions and register-bound occupancy, not by load latency.
- **Replicating the pair table 4, 8 and 16 times** so that lanes use distinct banks: 153 to 162 us at three rows. Shared-memory
  conflicts are not the cost.
- **Bound: looking up pairs without their payload bits** (wrong results; the lookup index only): 104 us at three rows and 68 us at one
  row. The rank, funnel shift and mask that place each payload unit are about a quarter of the kernel, and the format requires them.

## Not done

- **Scale indices** (about 0.25 GiB; GDeflate shows the scale plane compresses 43%): a 10-bit
  dictionary index or a BF16 mantissa/exponent code would need a second table per tensor in every
  kernel.
- **Q4/Q5 escape codes:** about 0.1 GiB each.
