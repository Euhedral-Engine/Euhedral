# Native Blackwell NVFP4

The first NVFP4 execution path (`native/src/nvfp4`, docs/NVFP4_RESIDENCY.md) never used FP4 tensor
cores. It is compiled for `compute_90` like every module. Its prefill tiles expand each weight to
BF16 and run `HMMA.16816.F32.BF16`, and decode is a scalar FP32 GEMV. This campaign adds a
Blackwell-only module (`native/src/nvfp4_native`) that runs block-scaled FP4 tensor-core MMA, and
measures it against that path and against compact Q3.

Measured 2026-10-01/02 on an RTX 5070 Ti (sm_120, 70 SMs, 16 GB) with CUDA 13.1 (NVRTC 13.1.80,
driver 615.71).

## Instruction contract

Verified against the installed toolchain and on the device, not from documentation.

**What the toolchain offers.**
- NVRTC 13.1 knows `kind::mxf4nvf4`, `kind::mxf8f6f4`, `.block_scale` and `.scale_vec::{1X,2X,4X}`.
- The NVVM intrinsic table in the installed Triton lists
  `mma.block.scale.m16n8k64.row.col.mxf4nvf4.scale.4x.f32.e2m1.e2m1.f32.ue4m3`.
- No CUTLASS or CuTe sources are installed.
- CCCL's `cuda::ptx` covers `tcgen05` only, which is sm_100 and not this GPU.

**The NVFP4 instruction:**

```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
    {d0..d3}, {a0..a3}, {b0, b1}, {c0..c3}, scaleA, {0, selA}, scaleB, {0, selB};
```

It compiles to SASS **`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`**.

| Item | Contract |
|---|---|
| Target | `sm_120a` or `sm_120f`. Plain `sm_120` is rejected by ptxas. `compute_90` PTX only "compiles" because NVRTC skips ptxas for virtual targets. |
| A and B | E2M1 only. Packed eight per 32-bit register, element j in nibble j, even K in the low nibble. |
| Accumulator | FP32 only (`.f16` is rejected). |
| Scales | UE4M3, one per 16 K values (`scale_vec::4X`): four per row or column per 64-K MMA. Decoded as E4M3 with the sign bit ignored, subnormals included; `0x7f`/`0xff` are NaN. |
| Tile | m16 n8 k64; 16384 FLOP per instruction. |
| A fragment | g = lane/4, t = lane%4. a0 row g, K 8t..8t+7; a1 row g+8; a2/a3 the same rows at K+32. |
| B fragment | b0 column g, K 8t..8t+7; b1 at K+32. |
| D fragment | d0/d1 row g, columns 2t, 2t+1; d2/d3 row g+8. |
| Scale A | Row r < 8 from lane 4r+2s, row r+8 from lane 4r+2s+1. selA = s ∈ {0,1}; ptxas rejects 2 and 3. Byte b scales K 16b..16b+15. |
| Scale B | Column n from lane 4n+s, selB = s ∈ {0..3}. Byte b scales K 16b..16b+15. |
| Error | Random codes and random scales (subnormals included), all selector pairs: max relative error 1.1e-7 against float64, which is FP32 accumulation. |

**Other forms.**
- `kind::mxf4nvf4.scale_vec::2X.ue8m0` (MXFP4) gives `OMMA.SF.16864.F32.E2M1.E2M1.E8`.
- `kind::mxf8f6f4` (E4M3 activations × E2M1 weights) gives `QMMA.SF.16832.F32.E4M3.E2M1.E8`. It accepts only one UE8M0 scale per 32 values. With UE4M3 it is rejected.
- **So NVFP4 weights with their per-16 E4M3 scales cannot be consumed natively with FP8 or BF16
  activations.** Native NVFP4 requires FP4 activations.

**The existing weight layout is consumed in place.** Codes are row-major, two per byte with the even
K low, K padded to 128. A B register is one aligned `u32` of a weight row. The scale plane is
`[rows][K/16]`, so a B scale register is one aligned `u32` of four scales. The FP32 global is
applied in the epilogue. No repacking is needed.

### Tensor-core throughput

Register-only loops with 8 independent accumulator chains per warp, 560 CTAs of 256 threads. Every
instruction was confirmed in the SASS.

| SASS | Dense TFLOPS |
|---|---|
| `HMMA.16816.F32.BF16` (the BF16-expansion path) | 98 |
| `HMMA.16816.F16` | 190 |
| `QMMA.16832.F32.E4M3.E4M3` | 197 |
| `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` | **815** |

## Kernels

`native/src/nvfp4_native/native.cuh`. Loaded by `euhedral_cuda_load_native_kernel`, which compiles a
cubin for `sm_<major><minor>a` and only on compute capability 12.x.

- **`quantize_rows<kTerms>`**: one CTA per row.
  - Per-row FP32 global = amax / (6 × 448).
  - Block scale = `e4m3_rn(block amax / (6 × global))`.
  - Codes = `e2m1_rn` with saturation (`F2FP.SATFINITE.E2M1.F32`).
  - With two terms, the residual x − term0 is quantized again with the same global, and each linear
    issues one MMA per term into the same accumulator.
- **`linear<kTerms, kPaired>`**:
  - CTA tile 128 × 128 and K tile 128, with `cp.async` stages (3 for one term, 2 for two, both
    67,584 bytes).
  - Code rows are padded to 80 bytes for conflict-free `ldmatrix`.
  - 8 warps of 64 × 32, epilogue × row global × weight global.
  - The paired variant puts 16 gate rows and their 16 up rows in each warp's slice. It applies
    SwiGLU to BF16-rounded gate and up, like the BF16 regions.
- **SASS:**

| Kernel | Main instructions | Registers |
|---|---|---|
| Linear | 32 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` per K tile (64 with two terms), `LDGSTS`, `LDSM.16.M88.4`; no weight conversion | 121 |
| Quantizer | `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`, `F2FP.SATFINITE.E4M3.F32` | 42 |

**Dispatch** (`CudaGpuMemory`, `native/src/host/nvfp4_linear.c`):

| Rows | NVFP4 linear (`linearNvfp4Bf16`) | Paired gate/up + SwiGLU (`nvfp4GateUpSwiGluBf16`) |
|---|---|---|
| 1 | the BF16-activation GEMV (`euhedral_nvfp4_decode`) when K is a multiple of 1024 and N of 16, as in every model shape; else the 64 × 64 BF16-expansion tile | not formed: decode and small views run gate/up as a linear |
| 2–63 | native skinny kernel when K is a multiple of 256 (`skinny_linear`, see "Decode-like row counts"), else the 128 × 128 tile | not formed (small views run gate/up as a linear) |
| 64 | native skinny kernel (same condition) | native paired tile |
| 65 and more | native 128 × 128 tile | native paired tile |

- **Thresholds:**
  - `NVFP4_NATIVE_MIN_ROWS` = 2 for linears;
  - `NVFP4_NATIVE_REGION_MIN_ROWS` = 64 for the paired region;
  - the host's `SKINNY_MAX_ROWS` = 64.
- Native linears need K to be a multiple of 128; otherwise the BF16-expansion kernels run.
- Quantized activations, plus FP32 split-K partials for skinny shapes, go into the shared,
  event-ordered scratch (`euhedral_cuda_nvfp4_native_scratch_bytes`).
- `EUHEDRAL_NVFP4_NATIVE`:
  - `0`: the BF16-expansion kernels;
  - `1`: one term;
  - two terms by default, the only mode that passes the drift harness.
- Exact numerics (`EUHEDRAL_EXACT=1`) always use the BF16-expansion kernels, which are the
  reference twin.

## Operator results

Real layer weights from both artifacts, rotated over 4 layers. Production entry points on one
stream, CUDA events. Native times include activation quantization. Times are in ms.

| Family | Rows | Compact Q3 | NVFP4→BF16 | Native, 1 term | Native, 2 terms |
|---|---|---|---|---|---|
| gate_up + SwiGLU | 64 | 0.502 | 0.558 | 0.190 | 0.262 |
| | 256 | 1.451 | 1.544 | 0.409 | 0.632 |
| | 512 | 2.812 | 3.029 | 0.805 | 1.211 |
| | 1024 | 5.552 | 5.919 | 1.573 | 2.405 |
| | 2048 | 21.611 | 11.761 | 3.166 | 4.707 |
| FFN down | 64 | 0.275 | 0.676 | 0.132 | 0.200 |
| | 256 | 0.719 | 1.073 | 0.260 | 0.466 |
| | 512 | 1.411 | 1.861 | 0.410 | 0.724 |
| | 1024 | 3.351 | 3.184 | 0.745 | 1.275 |
| | 2048 | 11.242 | 5.983 | 1.465 | 2.502 |
| GDN output | 64 | 0.162 | 0.237 | 0.048 | 0.074 |
| | 256 | 0.348 | 0.371 | 0.088 | 0.175 |
| | 512 | 0.593 | 0.610 | 0.139 | 0.242 |
| | 1024 | 1.032 | 1.114 | 0.240 | 0.405 |
| | 2048 | 1.959 | 2.085 | 0.509 | 0.878 |

- **One term:**
  - 3.7–4.6× the BF16-expansion path and 3.5–4.5× compact Q3 at 256–2048 rows.
  - The native tile reaches 225–290 TFLOPS, about 30% of `OMMA` peak.
  - Quantization costs 1–7% of native time.
- **Two terms:** 2.1–2.75× the BF16-expansion path.
- **Compact Q3 gate_up at 2048 rows** (21.6 ms) is off its own trend. The engine never forms
  2048-row quanta at the default 512-row chunk.

Before the change, the NVFP4 linear kernels were **93% of prefill GPU time**: 53% linears, 40%
gate_up+SwiGLU, GDN recurrence 2.7%, attention 1.9% (Nsight Systems, prefill 512 + 2048).

## End to end

Four-arm paired gate at commit 28e7070, when native linears started at 64 rows and fewer rows used
the BF16-expansion tile: 6 forks, each arm in a fresh JVM, start order rotated per fork, warmup 2,
3 iterations. Default 512-row prefill chunk. NVFP4 arms load executed objects only. Medians of
per-fork medians; "ahead" counts forks against the BF16-expansion arm.

The skinny route (c632d47) later changed the 64-row and smaller rows: prefill 64 went from 1178 to
1492 tok/s and TTFT for a 64-token prompt from 55.7 to 44.8 ms with two terms. See its gate under
"Decode-like row counts". The other rows are unaffected, because they form quanta of more than 64 rows.

| Scenario | Compact Q3 | NVFP4→BF16 | Native, 2 terms (default) | Native, 1 term |
|---|---|---|---|---|
| prefill 64 (tok/s) | 728 | 510 | 1154 (+126%, 6/6) | 1529 (+200%, 6/6) |
| prefill 256 | 1145 | 1025 | 2211 (+116%, 6/6) | 3197 (+212%, 6/6) |
| prefill 512 | 1229 | 1113 | 2506 (+125%, 6/6) | 3657 (+228%, 6/6) |
| prefill 1024 | 1225 | 1112 | 2504 (+125%, 6/6) | 3645 (+228%, 6/6) |
| prefill 2048 | 1219 | 1107 | 2477 (+124%, 6/6) | 3597 (+225%, 6/6) |
| TTFT, 64-token prompt (ms) | 88.4 | 126.9 | 55.6 (6/6) | 42.4 (6/6) |
| TTFT, 1024-token prompt (ms) | 861 | 944 | 434 (6/6) | 305 (6/6) |
| decode @64 (tok/s) | 62.9 | 51.3 | 51.2 (flat) | 51.2 (flat) |
| decode @1024 (tok/s) | 61.5 | 50.6 | 50.6 (flat) | 50.6 (flat) |
| 64 + 128 end to end (ms) | 2124 | 2623 | 2553 | 2539 |
| 1024 + 128 end to end (ms) | 2939 | 3474 | 2965 | 2836 |

- **Against compact Q3:** native prefill is 2.0–2.1× faster with two terms and 3.0× with one term at
  256–2048 tokens (6/6 forks each), and TTFT halves.
- **Decode is unchanged:** one row per token still runs the GEMV, and Q3 decodes 18% faster.

## Decode-like row counts

One-row decode streams every weight once per token, so its floor is bytes over DRAM bandwidth. The
NVFP4 GEMV already reaches it (gate_up: 100 MB in 0.128 ms, 780 GB/s). A tensor-core tile needs 16
rows, so the question is what extra rows cost.

**Skinny native kernel** (`skinny_linear`):
- 4 warps over 64 output columns and all M ≤ 64 rows.
- 256-value K tiles, up to 4 `cp.async` stages.
- K is split when 64-column tiles alone cannot occupy 140 CTAs; FP32 partials are summed with
  atomics and converted once.
- It takes 2–64 rows. One row stays on the GEMV, which is as fast and keeps BF16 activations.

**Operator times** (ms; real weights rotated over 4 layers; native includes quantization and split-K
clear and finish):

| Tensor (NVFP4 MB) | M | Q3 small-row route | NVFP4 GEMV / BF16 tile | Native 128×128, 1 term | Native skinny, 1 term | Native skinny, 2 terms |
|---|---|---|---|---|---|---|
| gate_up (100.3) | 1 | 0.095 | 0.128 | 0.165 | 0.128 | 0.129 |
| | 2 | 0.356 | 0.516 | 0.171 | 0.129 | 0.130 |
| | 4 | 0.531 | 0.518 | 0.166 | 0.129 | 0.130 |
| | 8 | 1.062 | 0.517 | 0.166 | 0.129 | 0.131 |
| | 16 | 0.403 | 0.523 | 0.174 | 0.131 | 0.132 |
| | 64 | 0.630 | 0.559 | 0.188 | 0.139 | 0.178 |
| down (50.1) | 1 | 0.046 | 0.068 | 0.117 | 0.071 | 0.077 |
| | 4 | 0.379 | 0.661 | 0.118 | 0.076 | 0.076 |
| | 16 | 0.328 | 0.663 | 0.120 | 0.072 | 0.082 |
| GDN output (17.7) | 1 | 0.013 | 0.024 | 0.035 | 0.028 | 0.031 |
| | 4 | 0.122 | 0.233 | 0.035 | 0.029 | 0.029 |
| | 16 | 0.114 | 0.233 | 0.041 | 0.028 | 0.030 |
| LM head (715.2) | 1 | 0.642 | 0.897 | 1.154 | 0.888 | 0.890 |
| | 4 | 3.625 | 3.306 | 1.159 | 0.893 | 0.895 |
| | 16 | 2.609 | 3.342 | 1.171 | 0.903 | 0.904 |

**Native NVFP4 is flat in M up to about 32 rows, at the weight-streaming floor.** Q3's small-row
route and the BF16-expansion tile cost 3–8× one row as soon as M ≥ 2.

**Whole-model estimate per decode step.**
- Linears weighted by layer count: 48 GDN layers (query_key, value_z, output), 16 attention layers
  (query_key, gate_value, output), 64 × (gate_up, down), and the LM head.
- Step time = the measured NVFP4 one-row step (19.52 ms at 51.24 tok/s) + linear(M) − linear(1).
- Linears are 94% of that step (18.3 ms).
- Attention and GDN growth with M is not included.

| M | Current NVFP4 route | Native, 1 term | Native, 2 terms |
|---|---|---|---|
| 1 | 19.5 ms (GEMV) | 19.5 ms (GEMV) | 19.5 ms (GEMV) |
| 2 | 121.4 ms | 20.9 ms (1.87× rows/s) | 21.6 ms (1.81×) |
| 4 | 120.1 ms | 21.1 ms (3.70×) | 21.8 ms (3.58×) |
| 8 | 121.1 ms | 20.8 ms (7.49×) | 21.8 ms (7.15×) |
| 16 | 120.7 ms | 22.1 ms (14.2×) | 21.8 ms (14.3×) |
| 32 | 121.9 ms | 21.8 ms (28.6×) | 23.7 ms (26.4×) |
| 64 | 125.4 ms | 23.6 ms (52.8×) | 31.3 ms (40.0×) |

**End to end, against the previous commit** (128×128 tile from 64 rows, BF16-expansion tile below):
paired gate, 6 forks, default two terms.

| Scenario | Before | After | Forks ahead |
|---|---|---|---|
| prefill 8 (tok/s) | 67.7 | 278.2 (+311%) | 6/6 |
| prefill 16 | 135.4 | 557.4 (+312%) | 6/6 |
| prefill 32 | 265.3 | 1003.1 (+278%) | 6/6 |
| prefill 64 | 1177.8 | 1492.5 (+27%) | 6/6 |
| TTFT, 16-token prompt (ms) | 119.4 | 29.1 | 6/6 |
| TTFT, 64-token prompt (ms) | 55.7 | 44.8 | 6/6 |
| decode @16 / @64 (tok/s) | 51.27 / 51.28 | 51.24 / 51.28 | flat |

**At M = 1 native loses 9%** (20.0 against 18.3 ms of linears), so one-row decode keeps the GEMV.
**From M = 2 native wins outright.** An MTP verification of 1 + 3 drafts, or a batch of 4, would
cost about 21 ms per step against 19.5 ms for one token.

## Numerics

**Operator level** (Gaussian activations, real-shape converter-quantized weights):
- Native output equals float64 over the same quantized operands within BF16 output rounding (max
  2.5e-3 of the output range).
- Against the BF16-activation result:

| Activation representation | Linear output relative RMS error |
|---|---|
| One term | 9.5% (activation quantization 9.5%) |
| Two terms | 0.88% (activation 0.87%) |

- Choosing the minimum-error scale among six candidates per block instead would only reach 8.8%.

**Model level** (`RelaxedNumericsDriftCudaIntegrationTest`, NVFP4 artifact, executed objects):
- Teacher-forced: exact arm = BF16-expansion kernels with exact numerics.
- Native changes prefill only, because decode runs the same one-row GEMV in both arms.
- "Floor" = native off: only the relaxed-order kernels differ.
- Exact against exact is bitwise zero.

| Prefix + steps | Arm | Median hidden error | Worst 1/8 window | KL mean | Top-1 | Harness |
|---|---|---|---|---|---|---|
| 256 + 384 | one term | 6.39% | 10.03% | 6.00e-3 | 95.1% | fails window bound (0.10) at the first window |
| 256 + 384 | two terms | 4.64% | 5.22% | 3.78e-3 | 97.4% | passes |
| 512 + 1024 | floor | 5.45% | 9.33% | 3.10e-3 | 97.9% | passes |
| 512 + 1024 | one term | 7.37% | 11.84% | 6.59e-3 | 97.0% | fails window bound |
| 512 + 1024 | two terms | 5.50% | 10.41% | 3.39e-3 | 97.6% | fails window bound late (position 1408), where the floor reaches 9.3% |

**Two terms sit at the floor; one term doubles KL.** Neither grows with position.
- A 1024-token single-quantum prefix does not fit beside 14 GB of weights and two sequences. 512 is
  the production chunk.
- No tolerance was changed.
