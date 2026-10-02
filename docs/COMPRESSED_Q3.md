# Compressed Q3 weights (P2E2)

P2E2 is an optional, lossless layout of the Q3G64_F16S tensors (persistent layout
`row-split-p2e2-v1`, `WeightLayout.ROW_SPLIT_P2E2_V1`). It keeps them resident on the GPU in less
memory, with the same values. Each kernel decodes only the codes it consumes, in registers.

An artifact selects it per tensor, so no engine option is needed: load a P2E2 artifact instead of
the compact one. Every Q3 route produces the same bits as on the compact artifact.

## Results

On an RTX 5070 Ti (16 GB):

| Representation | Resident weights | Decode | Prefill | Time to first token, 64-token prompt |
|---|---|---|---|---|
| compact (row-split Q3) | 11.98 GiB | baseline | baseline | 88.6 ms |
| P2E2 Q3 | 10.39 GiB weights + 0.07-0.13 GiB scratch | -5.9% to -6.3% | -3.9% (1024-2048 tokens) to -14.5% (64) | 104.1 ms (+17%) |

**Memory.** P2E2 frees 1.585 GiB of weights, 13.2% of the model and 18.5% of the Q3 tensors. The
expansion scratch takes 73 MiB of that back in generation, at most 128 MiB. The net is about
1.46-1.51 GiB more room for KV pages.

**Decode** costs 6%: the P2E2 kernel runs at 0.94x the contiguous kernel on the wide projections and
0.85x on the long-K ones.

**Prefill** costs one expansion per Q3 tensor per chunk, about 0.25 ms per layer. That is 4% at
1024-token chunks and 15% at a 64-token prompt.

Full gate: compact versus P2E2 artifact on the same build, six paired forks. P2E2 was behind in
every fork on every metric:

| Scenario | Compact | P2E2 | Change |
|---|---|---|---|
| decode 64 + 128 | 62.66 tok/s | 58.88 tok/s | -6.0% |
| decode 1024 + 128 | 61.52 tok/s | 57.64 tok/s | -6.3% |
| decode 4096 + 128 | 59.58 tok/s | 56.09 tok/s | -5.9% |
| prefill 64 | 746.8 tok/s | 638.3 tok/s | -14.5% |
| prefill 256 | 1149.8 tok/s | 1070.7 tok/s | -6.9% |
| prefill 1024 | 1228.1 tok/s | 1180.0 tok/s | -3.9% |
| prefill 2048 | 1220.1 tok/s | 1173.0 tok/s | -3.9% |
| time to first token, 1024-token prompt | 860.4 ms | 893.4 ms | +3.8% |

**Which families benefit.** Only Q3 benefits; it is 72% of the bytes and has a 0.7-bit entropy
gap. Q4 and Q5 would save at most 0.1-0.2 GiB each, BF16 and FP32 are 0.06 GiB, and none of them
were changed.

**When to use it.** It is worth using when memory is the limit: about 1.5 GiB more KV cache on a
16 GB card, for 6% of decode throughput and 4-15% of prefill. For throughput, use the compact
artifact.


## Layout

A Q3 code is a signed 3-bit integer in [-3, 3]; the converter never emits -4. Its entropy is about
2.30 bits, because -1, 0 and +1 make up 80% of the codes. P2E2 stores:

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

Specification and reference encoder/decoder: `tools/convert_compact_edrl_to_p2e2.py`. Device code:
`native/src/q3/p2e2.cuh` and `euhedral_q3_p2e2_embedding` (`native/src/embedding/kernels.cu`). Host:
`P2e2Layout`.

## Producing an artifact

```bash
python3 tools/convert_compact_edrl_to_p2e2.py \
    /mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl \
    /mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3_p2e2.edrl
```

The tool needs numpy. It transcodes every Q3 tensor whose K is a multiple of 1024: all 199 Q3
tensors of the model. Every other tensor is copied unchanged.

Each transcoded tensor is decoded back and compared byte for byte with its source before the output
is renamed into place. The manifest records the sizes and the SHA-256.

## Execution routes

| Route | P2E2 handling |
|---|---|
| Decode, one row, relaxed numerics (`euhedral_q3_decode_contiguous` shapes) | `euhedral_q3_p2e2_decode` reads the compressed tensor directly |
| Token embedding | `euhedral_q3_p2e2_embedding` gathers rows directly |
| Everything else: prefill regions, split-K and streamed FFN, 2-8 rows, exact numerics, scalar | `euhedral_q3_p2e2_expand` rebuilds the row-split tensor in a shared scratch buffer, then the unchanged row-split kernel runs |
| LM head with all-token logits (tests and direct contexts only; generation asks for the last token) | expanded in output-row chunks that fit the scratch, then each chunk is copied into place |

**Fused decode kernel.** It keeps `euhedral_q3_decode_contiguous`'s lane ownership, activations and
FP32 FMA chain. Each lane owns 32 contiguous codes of a 1024-code slice.

The codes come in pairs from a 256-entry shared table of float pairs. The table is indexed by the
pair's primary nibble and by its payload bits, placed at the pair's BIG positions and zero elsewhere.
So the 64% of pairs with no BIG code hit 9 broadcast entries in distinct banks.

Payload positions come from a warp prefix sum of each lane's BIG count. Each row and slice prefetches
32 payload words, which shuffles redistribute. A warp takes a slow path when any lane holds more than
16 BIG codes, or when the slice's units outrun the prefetched words.

**Scratch.** One device region, owned by `CudaGpuMemory`, is reused by every expansion. Each use waits
for the previous one through a CUDA event recorded on the stream that used it. Growing the region
first drains the device.

It is sized to the largest expansion seen: the FFN gate/up projection (73 MiB) in generation, gate/up
plus down (109 MiB) for 1024-row streamed prefill, and at most 128 MiB.

## Correctness

Compression is lossless, and every route is bitwise identical to the same route on the compact
artifact:

- **Converter:** every tensor round-trips byte for byte during conversion
  (`tools/test_p2e2_converter.py` covers the packing, extremes, chunk boundaries and corrupted planes).
- **Native kernels** (`native/tests/test_q3_p2e2.py`, synthetic tensors that force every path: dense
  BIG lanes, overflowing slices, empty rows):
  - decode equals `euhedral_q3_decode_contiguous` bit for bit;
  - expansion, of whole tensors and of row ranges, equals the row-split bytes;
  - the embedding equals `euhedral_q3_embedding` bit for bit.
  The host entry points' geometry checks and route decisions are tested too.
- **Full model** (`QwenP2e2CudaIntegrationTest`, `-Peuhedral.qwen.p2e2-artifact`): the P2E2 artifact
  reproduces the compact artifact's logits bit for bit in every case below. Since the outputs are
  identical, compression adds no drift.
  - an 80-token prefill with all-token logits (prefill regions, plus the chunked LM head);
  - six relaxed decode steps on the P2E2 kernel;
  - one decode step under exact numerics (expanded);
  - a 5-token prompt on the small-row kernels.
- **Existing kernels:** the PTX of every existing kernel in the q3, embedding and ffn modules is
  unchanged (`ptx_compare.py` against `main`).
- **Compact artifact unaffected:** the compact artifact runs as fast on this branch as on `main`.
  Six paired forks, all seven scenarios, stayed within ±0.4% with no consistent direction: decode
  +0.07% to +0.38%, prefill -0.08% to +0.11%.

## Characterization

**Resident weights.** The compact artifact uploads all 1,118 objects: 11.98 GiB.

| Family | Bytes | Codes / scales |
|---|---|---|
| Q3 | 8.59 GiB (text 8.45, MTP 0.13) | 7.93 GiB codes, 0.66 GiB scales |
| Q5 | 2.31 GiB | |
| Q4 | 0.91 GiB | |
| W8 | 0.12 GiB | |
| BF16 and FP32 | 0.06 GiB | |

The FFN `gate_up` and `down` projections alone are 54% of the model.

Three resident tensor families are never executed by text generation: `text/draft_head` (Q3, 0.25
GiB), `mtp/*` (0.21 GiB) and the vision tower (0.27 GiB). Not uploading them would save 0.73 GiB.
That is outside this layout and was not changed.

**Code statistics.** Order-0 entropy of the codes, and the symbol probabilities, measured over every
tensor:

| Format | Stored bits/code | Entropy | Huffman | Escape code | Ideal saving | Escape-code saving |
|---|---|---|---|---|---|---|
| Q3 | 3 | 2.31 | 2.36 | 2.40 (P2E2) | 1.83 GiB (23.1%) | 1.60 GiB (20.2%) |
| Q4 | 4 | 3.46 | 3.49 | 3.57 | 0.12 GiB (13.5%) | 0.09 GiB |
| Q5 | 5 | 4.55 | 4.57 | 4.78 | 0.20 GiB (9.0%) | 0.10 GiB |

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

## Decode kernel development

These are operator benchmarks of layer-10 weights against `euhedral_q3_decode_contiguous`. Weight
copies were rotated past L2, and every variant is bitwise identical. Times are relative to the
contiguous kernel:

| Variant | gate_up 34816x5120 | down 5120x17408 | Notes |
|---|---|---|---|
| bytes-only ceiling (primary decode only, dependent payload loads) | 1.03x | 0.78x | latency of the dependent payload load |
| pair table, raw window index | 0.57x | 0.59x | 3.6 shared wavefronts per lookup: bank conflicts |
| + masked, position-aligned window; payload words prefetched per slice | 0.59x | 0.58x | 2.1x instructions; spills |
| + slow path in a `#pragma unroll 1` loop | 0.29x | 0.28x | per-row arrays forced into local memory |
| + fully unrolled slow path | 0.87x | 0.76x | 128 registers, about 1.1 waves on down |
| 4 rows/warp, 5 CTAs/SM (96 registers) | 0.91x | 0.82x | |
| + `lop3`/`mad` for the table index (production) | **0.94x** | **0.85x** | 1.59x instructions, 59% issue |
| 2 rows/warp; 8 rows/warp; 6/9/10 CTAs/SM | 0.83-0.93x | 0.70-0.82x | rejected |
| software prefetch of the next slice | 0.92x | 0.82x | rejected: loads were not the limit |
| swizzled table (index nib + 17 * window) | 0.86x | 0.79x | rejected: BIG entries collide with the no-BIG entries' banks |
| SWAR rank prefix instead of POPC | 0.89x | 0.79x | rejected: more ALU than the POPC it removes |

**Why decode cannot be free.** On L2-resident data the contiguous kernel decodes about 2.85e12
codes/s, only about 1.4x its DRAM-fed rate. A decoder of a 20%-smaller layout therefore keeps up
only if it spends no more than about 1.3x the instructions per code.

Any variable-length code that keeps the FP32 FMA order, which bitwise identity requires, must place
each payload unit at its code's position. That costs a rank per pair: a POPC, a funnel shift, a mask
and a table lookup, about 11 instructions per pair against 7. The best kernel runs at 1.59x the
contiguous kernel's instructions.

## Prefill development

These were candidates for decoding inside the balanced tile engine's B producer at 512 rows, so that
no expanded tensor would be needed. All were bitwise identical; times are relative to the row-split
kernels:

| Variant | mixer 5120x6144 | gate_up | down (split-K) |
|---|---|---|---|
| decode in the B stage | 0.53x | 0.63x | 0.50x |
| + 6-bit field table | 0.59x | 0.59x | 0.55x |
| + 8-register state, 128-register cap | 0.60x | 0.58x | 0.57x |
| + one load per pass | about 0.62x | | |
| bound: no decode arithmetic | 0.61x | 0.60x | 0.58x |
| bound: no payload loads | 0.82x | 0.80x | 0.77x |
| bound: payload loads at fixed addresses | 0.70x | 0.62x | 0.66x |

Staging sits between the MMAs and the per-generation barrier. The extra loads and shuffles each K32
block needs cost more than the arithmetic, and even without the payload loads the engine stays
below 0.8x.

Expanding into the row-split layout instead runs at about the DRAM rate:

| Tensor | Expansion time |
|---|---|
| mixer | 18 us |
| gate_up | 160 us (790 GB/s read + write) |
| down | 73 us |
| LM head | 1.2 ms |

That is about 0.25 ms per layer per prefill chunk, on top of the unchanged kernels.

## General-purpose codecs: GDeflate and ANS

GDeflate (nvCOMP / DirectStorage) is Deflate laid out so that a warp decodes a 64 KB page in
parallel. nvCOMP decompresses whole pages into a separate buffer, so it could only replace the
expansion route; decode reads every Q3 weight once per token.

These are nvCOMP 5.3 measurements on layer-10 tensors; every decompression was exact. Ratios are
relative to the row-split tensor; P2E2 alone is 0.815:

| Input | Codec | Ratio | gate_up decompression | down decompression |
|---|---|---|---|---|
| row-split | GDeflate, entropy only (type 0) | 0.885 | 2.04 ms (35 GB/s) | 1.65 ms |
| row-split | GDeflate, maximum ratio (type 5) | 0.885 | 2.19 ms | 1.89 ms |
| row-split | nvCOMP ANS | 0.893 | 0.47 ms | 0.28 ms |
| P2E2 | GDeflate, type 0 | 0.765 | 1.97 ms | 1.69 ms |
| P2E2 | nvCOMP ANS | 0.771 | 0.45 ms | 0.24 ms |

For comparison:

- P2E2's decode GEMV of gate_up takes 94 us per token, and its expansion takes 160 us.
- zlib level 9 per 64 KB page agrees with nvCOMP on the codes: 0.91 of the packed planes. One code per
  byte reaches 0.79, and only with Huffman-only coding. The scale plane reaches 0.57, which nothing can
  read in place.

GDeflate alone compresses less than P2E2 and decompresses 12x slower than P2E2's expansion.
Stacked on P2E2 it would save another 6% of the Q3 bytes, but only through expansion, even for decode.
Rejected.

## Not done

- **Scale indices** (about 0.25 GiB; GDeflate shows the scale plane compresses 43%): a 10-bit
  dictionary index or a BF16 mantissa/exponent code would need a second table per tensor in every
  kernel. It is left for a later step.
- **Q4/Q5 escape codes:** about 0.1 GiB each.
- **Fused prefill:** see above.
- **Not uploading the unused draft head, MTP layer and vision tower:** 0.73 GiB, and independent of
  this layout.
