# Trellis (EXL3) weights

The engine serves Swift-1.5 (Qwen3.8-27B) from trellis checkpoints: the quantization module's
`calibrate --format trellis` output, packed into EDRL by `tools/convert_checkpoint.py --quantization trellis
--trellis-from DIR`. Each quantized linear is stored in Hadamard-rotated space as 16x16 trellis tiles decoded with the
procedural mul1 codebook, with input-side (`suh`) and output-side (`svh`) sign and scale vectors.

## Format

| Tensor | Storage |
|---|---|
| `<name>` | `TRELLIS` / `trellis-tile16-v1`: int16 words `[k/16, n/16, w]`, `w = 16 * bits` (1 to 8 bits in half steps) |
| `<name>.suh` | FP16 `[s, k]`: one input-side vector per segment |
| `<name>.svh` | FP16 `[n]` |
| `<name>.segments` | I32 `[s]`: segment ends, multiples of 128, the last equal to n |

An engine object that fuses several source linears (attention query/key with per-head gate rows, GDN qkv/z,
`gate_up`, the MTP `query_key_gate_value`) keeps one segment per source linear, each with its own `suh`. The companions
stay on the device; only payloads move to the host or disk tiers. The token embedding is Q3 and the LM and draft heads
are NVFP4, rounded by GPTQ in the calibration run (`--output-head BF16` makes a BF16-head artifact).

## Execution

Each trellis linear is a chain of `linear` table kernels inside the existing linear stage, in stream order:

- **Decode, VERIFY, DRAFT:** prepass (BF16 activations times `suh`, 128-point Hadamard per block, FP16 in the GEMM's
  fragment order), then a tensor-core `m16n8k16` GEMM that decodes the tiles in registers, then (when K is split) the
  epilogue that sums the split partials in split order and applies the output Hadamard and `svh`. Each output row's
  reduction order is fixed by the shape alone, so a batched VERIFY gives bitwise the result of one-row launches.
  Each warp streams its tiles through a four-slot ring of asynchronous copies and decodes a lane's eight weights from
  three ring words at offsets fixed by the tile width. Launches of up to 8 rows take the decoded weights as the MMA's A
  operand and the activations as B, one `m16n8k16` per k16 tile; larger launches take 16-row activation tiles as A.
  Both orientations give the same bits. The prepass and the epilogue run each 128-column block's Hadamard in one
  warp's registers, the same additions as the shared-memory butterfly of the scalar reference.
- **Many rows** (at least `euhedral.trellis.whole-k-rows`): the same GEMM, each block covering 32 rows (64 from
  `euhedral.trellis.whole-k-wide-rows`, half their split sums in registers, two blocks per SM) of a 128-column block
  and streaming the whole K range once, adding each split's sums in split order at the split's end. The block
  stages its activation rows in shared memory four k16 tiles at a time, so its warps share one copy, and finishes
  each row's Hadamard in one warp's registers. The results are bitwise those of the split launch, with no split
  partials.
- **Many rows, not a VERIFY, on sm_12xa** (`trellis_fp8/kernels.cu`, an architecture-specific module of the linear
  table): the same blocks on block-scaled FP8 tensor cores, one `m16n8k32` (`kind::mxf8f6f4`, E4M3 operands, FP32
  accumulation) per two k16 tiles, four times the rate of the FP32-accumulating `m16n8k16`. The prepass writes these
  launches' activations as E4M3 with one UE8M0 scale per 32 columns, the least power of two that brings the block's
  largest value within 448, in the order the MMA's A fragments read them; the decoded weights are rounded to E4M3
  (within ±3.46, no scale). Blocks stage 8 k16 tiles of activations and their scale bytes at a time; K is a multiple
  of 128, so the loop over a chunk is unrolled with its ring slots, staging and barrier fixed at compile time. Each
  row's result depends only on its own data, but not with the FP16 operands' bits, so a VERIFY keeps them. On other
  devices these launches take the FP32-accumulating blocks. On the held-out set, trellis 3.0 scores dNLL
  +0.0656 ± 0.0069, KL 0.0787, top-1 92.61% this way, and +0.0659 ± 0.0067, KL 0.0776, top-1 92.63% with FP16
  operands and FP32 accumulation throughout.
- **Measured, not kept:** FP16 operands with FP16 accumulation over runs of 8 k16 tiles (twice the FP32-accumulating
  rate; each run's sums stay under 195): prefill 2,185 tokens/s at 4096 tokens, KL 0.0761 on the held-out set. A
  fixed activation scale of 2^9 in place of the block scales: prefill 2,612 tokens/s at 4096 tokens, KL 0.0776, but
  activations past 0.875 (twice the largest prepass output over two calibration sequences) would saturate.
- **Prefill** (non-VERIFY launches above `euhedral.trellis.mma-max-rows` rows): the whole object decoded to BF16 in
  rotated space, one row-exact BF16 GEMM per segment, and the BF16 epilogue.
- **Exact numerics:** the scalar FP32 reference.

The prepass outputs fit FP16 with a wide margin: over two calibration sequences, the largest prepass output any of the
400 linears of a calibrated 3.5 bpw checkpoint meets is 0.44, for inputs up to |x| = 394
(`native/tests/fixtures/trellis-activation-extremes.json`, `TrellisActivationRangeCudaIntegrationTest`).

## Building an artifact

```bash
./gradlew :quantization:run --args="calibrate --model CHECKPOINT_DIR --format trellis --bits 3.5 \
    --tokens TOKENS.npy --draft-ids-from Q3_ARTIFACT.edrl --out EXL3_DIR"
python3 tools/convert_checkpoint.py --model CHECKPOINT_DIR --quantization trellis --trellis-from EXL3_DIR \
    --codes-from EXL3_DIR/heads --draft-ids-from Q3_ARTIFACT.edrl --out qwen3_8_27b_trellis_3_5.edrl
```

| Bits | Calibration | Artifact |
|---|---|---|
| 3.0 | 35 min 42 s | 11.0 GB |
| 3.5 | 50 min 47 s | 12.5 GB |

## Quality

KL(BF16 reference || artifact) and top-1 agreement on T1 to T4 (docs/QUALITY.md), with the chunk-level standard error:

| Artifact | T1 KL | T2 KL | T3 KL | T4 KL | T1 / T2 / T3 / T4 top-1 |
|---|---|---|---|---|---|
| trellis 3.0, NVFP4 heads | 0.0500 ± 0.0018 | 0.0559 ± 0.0064 | 0.1190 ± 0.0122 | 0.2206 ± 0.0300 | 0.887 / 0.910 / 0.952 / 0.922 |
| trellis 3.5, NVFP4 heads | 0.0321 ± 0.0013 | 0.0386 ± 0.0058 | 0.0731 ± 0.0077 | 0.1719 ± 0.0251 | 0.896 / 0.927 / 0.964 / 0.932 |

These artifacts store the token embedding as Q3 and score with NVFP4 attention caches, as the engine serves them. The
trellis linears themselves are exact: an artifact of the same 3.5 bpw checkpoint with a BF16 embedding and BF16 head,
scored with BF16 caches, measures T1 KL 0.0231 ± 0.0011 and T2 KL 0.0257 ± 0.0034, which is the checkpoint's KL in
exllamav3 1.6.1 (0.0226 and 0.0258).

## Speed

RTX 5070 Ti (16 GB) at a 350 W power limit (its default is 300 W), the benchmark module with 2 warmup and 3 measured
iterations, every weight on the device:

| Artifact | Prefill 512 tokens | Prefill 16384 tokens | Decode, MTP speculation | Decode, no speculation |
|---|---|---|---|---|
| trellis 3.0 | 2773 tokens/s | 2431 tokens/s (6.8 s) | 226.7 tokens/s | 66.1 tokens/s |
| trellis 3.5 | 2702 tokens/s | 2386 tokens/s (6.9 s) | 214.8 tokens/s | 59.8 tokens/s |

Decode is measured on `decode:32:256` (256 tokens after a 32-token prompt).

The defaults of the four properties are measured on this card, over the 3-bit model's object shapes with the weights
rotated past L2:

- `euhedral.trellis.target-ctas` (560): the thread blocks the tensor-core route's K split aims for. The trellis linears
  of one decode step take 14.41 ms at 140, 13.20 ms at 280, 13.01 ms at 560, 13.07 ms at 840 and 13.20 ms at 1120.
- `euhedral.trellis.whole-k-rows` (64): the fewest rows for which each block runs every K split itself. With split
  partials and with whole-K blocks of 32 rows, one step takes 48.6 and 59.4 ms at 48 rows, 56.5 and 64.9 ms at 56, and
  64.4 and 60.4 ms at 64.
- `euhedral.trellis.whole-k-wide-rows` (128): the fewest rows for which those blocks cover 64 rows. With blocks of 32
  and of 64 rows, one step takes 74.4 and 77.6 ms at 80 rows, 77.6 and 83.7 ms at 96, 102.9 and 101.7 ms at 128, and
  364.2 and 319.1 ms at 512.
- `euhedral.trellis.mma-max-rows` (512): the most rows a launch other than a verification keeps on the tensor-core route.
  The tensor-core route prefills 512-row chunks at 2773 tokens/s and the prefill route at 490, so every chunk of the
  default workspace stays on the tensor-core route and the residency plan reserves no prefill scratch.
