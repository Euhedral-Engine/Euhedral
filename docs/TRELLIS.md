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

- **Decode, VERIFY, DRAFT:** prepass (BF16 activations times `suh`, 128-point Hadamard per block, FP16), then a
  tensor-core `m16n8k16` GEMM that decodes the tiles in registers, then (when K is split) the epilogue that sums the
  split partials in split order and applies the output Hadamard and `svh`. Each output row's reduction order is fixed by
  the shape alone, so a batched VERIFY gives bitwise the result of one-row launches.
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
| trellis 3.5, NVFP4 heads | 0.0321 ± 0.0013 | 0.0386 ± 0.0058 | 0.0731 ± 0.0077 | 0.1719 ± 0.0251 | 0.896 / 0.927 / 0.964 / 0.932 |
