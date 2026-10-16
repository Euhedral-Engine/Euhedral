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
  Both orientations give the same bits.
- **Many rows** (at least `euhedral.trellis.whole-k-rows`): the same GEMM, each block covering 32 rows of a 128-column
  block and running every K split itself, adding the splits' sums in split order. The results are bitwise those of
  the split launch, with no split partials.
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
| trellis 3.0 | 996 tokens/s | 941 tokens/s (17.4 s) | 232.7 tokens/s | 65.7 tokens/s |
| trellis 3.5 | 988 tokens/s | 941 tokens/s (17.4 s) | 212.9 tokens/s | 59.6 tokens/s |

Decode is measured on `decode:32:256` (256 tokens after a 32-token prompt).

The defaults of the three properties are measured on this card, over the 3-bit model's object shapes with the weights
rotated past L2:

- `euhedral.trellis.target-ctas` (560): the thread blocks the tensor-core route's K split aims for. The trellis linears
  of one decode step take 14.41 ms at 140, 13.20 ms at 280, 13.01 ms at 560, 13.07 ms at 840 and 13.20 ms at 1120.
- `euhedral.trellis.whole-k-rows` (80): the fewest rows for which each block runs every K split itself. With split
  partials and with whole-K blocks, one step takes 64.5 and 67.8 ms at 64 rows, 83.0 and 78.6 ms at 80, and 108.4 and
  89.7 ms at 96.
- `euhedral.trellis.mma-max-rows` (512): the most rows a launch other than a verification keeps on the tensor-core route.
  The tensor-core route prefills 512-row chunks at 996 tokens/s and the prefill route at 490, so every chunk of the
  default workspace stays on the tensor-core route and the residency plan reserves no prefill scratch.
