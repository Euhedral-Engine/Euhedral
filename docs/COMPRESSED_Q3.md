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
- **Decode** (one row, the fused P2E2 kernel): 58.9 tok/s at a 64-token context, 57.6 at 1024 and 56.1 at 4096, 128 generated
  tokens, greedy. The kernel runs at 0.94x the contiguous kernel's rate on the wide projections and 0.85x on the long-K ones; a
  gate/up GEMV takes 94 us.
- **Prefill and verification** expand each Q3 tensor into the row-split layout once per launch, at about the DRAM rate:

  | Tensor | Expansion time |
  |---|---|
  | mixer | 18 us |
  | gate_up | 160 us (790 GB/s read + write) |
  | down | 73 us |
  | LM head | 1.2 ms |

  That is about 0.25 ms per layer per prefill chunk, on top of the unchanged row-split kernels.

**Which families benefit.** Only Q3 does: it is 72% of the bytes and has a 0.7-bit entropy gap. Q4 and Q5 would save at most
0.1-0.2 GiB each, BF16 and FP32 are 0.05 GiB, and none of them were changed.

**When to use it.** When memory is the limit: about 1.5 GiB more KV cache on a 16 GB card. It costs decode throughput and prefill
time; for throughput use the `q3` artifact.

## Layout

A Q3 code is a signed 3-bit integer in [-3, 3]; the converter never emits -4. Its entropy is about 2.30 bits, because -1, 0 and
+1 make up 80% of the codes. P2E2 stores:

- **Primary plane**, 2 bits per code. Symbol t = 0 means -1, 1 means 0, 2 means +1 and 3 means BIG.
  Code j of a row is bits 2(j mod 16) of word j/16, sixteen codes to a little-endian u32.
- **Payload plane**, one 2-bit unit per BIG code in row-major order. Unit p = 0 means -3, 1 means -2,
  2 means +2 and 3 means +3, sixteen to a little-endian u32, least significant first.
- **Row-base plane**, one u32 per row: the payload unit of the row's first BIG code. A kernel that
  walks a row's 1024-code slices in order derives every later slice's position from its own BIG
  counts, so one word per row is the whole index.
- **Scale plane**: the row-split FP16 scales, byte for byte.

For a tensor of `rows` rows of K codes (K a multiple of 1024), each plane starts on a 256-byte
boundary:

| Plane | Offset | Size |
|---|---|---|
| primary | 0 | rows * K / 4 |
| row base | align256(rows * K / 4) | 4 * rows |
| scales | align256(row base + 4 * rows) | rows * K / 32 |
| payload | align256(scales + rows * K / 32) | 4 * (ceil(units / 16) + 80) |

The 80 zero words after the payload let kernels prefetch whole windows without bounds checks.

The payload length depends on the codes, so the descriptor's byte size is checked against bounds
rather than a formula: no shorter than an empty payload, no longer than a full one, and a whole number
of words. At load time `P2e2Layout.validate` also recounts every row's BIG codes against the row-base
plane and the payload length, so a damaged file cannot make a kernel read past its tensor.

Reference encoder and decoder: `tools/euhedral_artifacts/q3_p2e2.py`. Device code: `native/src/q3/p2e2.cuh` and
`euhedral_q3_p2e2_embedding` (`native/src/embedding/kernels.cu`). Host: `P2e2Layout`.

## Producing an artifact

```bash
python3 tools/convert_checkpoint.py --model /mnt/shared/qwen38-quant/source/qwen --quantization q3 --compressed \
    --draft-ids-from /mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl \
    --out /mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3_compressed.edrl
```

The converter builds the `q3` artifact beside the output, transcodes every Q3 tensor whose K is a multiple of 1024 (all 195 Q3
tensors of the model) and copies every other tensor unchanged. Each transcoded tensor is decoded back and compared byte for byte
with its source before the output is renamed into place. The manifest records the sizes and the SHA-256 (`tools/README.md`).

## Execution routes

| Route | P2E2 handling |
|---|---|
| Decode, one row, relaxed numerics (`euhedral_q3_decode_contiguous` shapes) | `euhedral_q3_p2e2_decode` reads the compressed tensor directly |
| Token embedding | `euhedral_q3_p2e2_embedding` gathers rows directly |
| Everything else: 2-8 rows (speculative verification and drafting), prefill linears and the paired gate/up region, exact numerics, scalar reference | `euhedral_q3_p2e2_expand` rebuilds the row-split tensor in a shared scratch buffer, then the unchanged row-split route runs; for the MXFP8 route the expansion reserves the activation region behind the expanded weights |
| LM head with all-token logits (tests and direct contexts only; generation asks for the last token) | expanded in output-row chunks that fit the scratch, then each chunk is copied into place; the split-K choice follows the whole tensor's rows, so the chunked result equals the whole one |

**Fused decode kernel.** It keeps `euhedral_q3_decode_contiguous`'s lane ownership, activations and
FP32 FMA chain. Each lane owns 32 contiguous codes of a 1024-code slice.

The codes come in pairs from a 256-entry shared table of float pairs. The table is indexed by the
pair's primary nibble and by its payload bits, placed at the pair's BIG positions and zero elsewhere.
So the 64% of pairs with no BIG code hit 9 broadcast entries in distinct banks.

Payload positions come from a warp prefix sum of each lane's BIG count. Each row and slice prefetches
32 payload words, which shuffles redistribute. A warp takes a slow path when any lane holds more than
16 BIG codes, or when the slice's units outrun the prefetched words.

**Scratch.** One device region, owned by `CudaGpuMemory`, is reused by every expansion and by the quantized activations of the
MXFP8 and native FP4 routes. Each use waits for the previous one through a CUDA event recorded on the stream that used it. Growing
the region first drains the device. The expanded weights of one linear take at most 128 MiB (the gate/up expansion is 72 MB);
larger tensors expand in output-row chunks.

## Correctness

Compression is lossless, and every route is bitwise identical to the same route on the `q3` artifact:

- **Converter:** every tensor round-trips byte for byte during conversion
  (`tools/test_p2e2_converter.py` covers the packing, extremes, chunk boundaries and corrupted planes).
- **Native kernels** (`native/tests/test_q3_p2e2.py`, synthetic tensors that force every path: dense
  BIG lanes, overflowing slices, empty rows):
  - decode equals `euhedral_q3_decode_contiguous` bit for bit;
  - expansion, of whole tensors and of row ranges, equals the row-split bytes;
  - the embedding equals `euhedral_q3_embedding` bit for bit.
  The host entry points' geometry checks and route decisions are tested too.
- **Full model** (`QwenP2e2CudaIntegrationTest`, `-Peuhedral.qwen.artifact` and `-Peuhedral.qwen.q3-compressed-artifact`): the P2E2
  artifact reproduces the row-split artifact's logits bit for bit, and its weights are at least 1.5 GiB smaller. Cases:
  - an 80-token prefill with all-token logits (prefill regions, plus the chunked LM head);
  - six relaxed decode steps on the P2E2 kernel;
  - one decode step under exact numerics (expanded);
  - a 5-token prompt on the small-row kernels.

## Characterization

**Code statistics.** Order-0 entropy of the codes, and the symbol probabilities, measured over every Q3, Q4 and Q5 tensor:

| Format | Stored bits/code | Entropy | Huffman | Escape code | Ideal saving | Escape-code saving |
|---|---|---|---|---|---|---|
| Q3 | 3 | 2.31 | 2.36 | 2.40 (P2E2) | 23.1% | 20.2% |
| Q4 | 4 | 3.46 | 3.49 | 3.57 | 13.5% | |
| Q5 | 5 | 4.55 | 4.57 | 4.78 | 9.0% | |

- **Q3 code probabilities:** 0: .34, ±1: .23 each, ±2: .08 each, ±3: .02 each.
- **Stable across tensors.** The distribution is nearly identical for every tensor. Conditioning on
  the row or the column gains less than 0.04 bits, so a static code is enough and no per-tensor
  tables are needed.
- **General-purpose compressors** reach only 10-11% on packed codes (zstd -19, xz -9), because the
  codes straddle byte boundaries.
- **Q4 and Q5** offer at most 0.1-0.2 GiB each and keep their layouts.

**Scales.** Every FP16 scale is fp16(bf16 max-abs / qmax), so each is exactly invertible to its
BF16 source. A tensor has 480-730 distinct scales (1,246 for the embedding), at an entropy of about
7.3 bits, so a 10-bit index per tensor would be lossless and would save about 0.25 GiB. P2E2 keeps
the FP16 plane; see *Not done* below.

## Rejected

**Decode kernel variants.** Operator benchmarks of layer-10 weights against `euhedral_q3_decode_contiguous`, weight copies rotated
past L2, every variant bitwise identical. Speeds are relative to the contiguous kernel (gate_up 34816x5120 / down 5120x17408):

- bytes-only ceiling (primary decode only, dependent payload loads): 1.03x / 0.78x. The latency of the dependent payload load
  bounds the design.
- pair table with a raw window index: 0.57x / 0.59x; 3.6 shared wavefronts per lookup, from bank conflicts.
- masked, position-aligned window with payload words prefetched per slice: 0.59x / 0.58x; 2.1x the instructions, and spills.
- slow path in a `#pragma unroll 1` loop: 0.29x / 0.28x; the per-row arrays are forced into local memory.
- fully unrolled slow path: 0.87x / 0.76x; 128 registers, about 1.1 waves on down.
- 4 rows per warp at 5 CTAs per SM (96 registers): 0.91x / 0.82x.
- 2 rows per warp, 8 rows per warp, 6, 9 or 10 CTAs per SM: 0.83-0.93x / 0.70-0.82x.
- software prefetch of the next slice: 0.92x / 0.82x; the loads were not the limit.
- swizzled table (index nibble + 17 * window): 0.86x / 0.79x; BIG entries collide with the no-BIG entries' banks.
- SWAR rank prefix instead of POPC: 0.89x / 0.79x; more ALU than the POPC it removes.

The shipped kernel adds `lop3` and `mad` for the table index to the 4-rows-per-warp, 5-CTAs-per-SM form: 0.94x / 0.85x at 1.59x
the contiguous kernel's instructions and 59% issue.

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

## Not done

- **Scale indices** (about 0.25 GiB; GDeflate shows the scale plane compresses 43%): a 10-bit
  dictionary index or a BF16 mantissa/exponent code would need a second table per tensor in every
  kernel.
- **Q4/Q5 escape codes:** about 0.1 GiB each.
