# Native Blackwell NVFP4

NVFP4 linears run on Blackwell's block-scaled FP4 tensor cores (`native/src/nvfp4_native`) from two rows: both operands are
NVFP4, the BF16 activations are quantized to two NVFP4 terms (the value and its quantized residual) and each linear issues one MMA
per term into the same accumulator. One row runs the BF16-activation GEMV (`native/src/nvfp4`, which also holds the 2 to 8 row
twins that verification uses). The engine requires a Blackwell GPU, so the native module (compiled for `sm_<major><minor>a`)
always loads. There are no options: the route, the two terms and the row thresholds are fixed. Exact numerics (oracle only,
`CudaGpuMemory.selectExactNumerics`) run the scalar reference (`native/src/reference/kernels.cu`).

Measured 2026-10-01/02 on an RTX 5070 Ti (sm_120, 70 SMs, 16 GB) with CUDA 13.1 (NVRTC 13.1.80, driver 615.71).

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
| Target | `sm_120a` or `sm_120f`. The `cvt` forms for E2M1 are rejected by ptxas for plain `sm_120`, which is why the other modules use exact software conversions. |
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
| `HMMA.16816.F32.BF16` | 98 |
| `HMMA.16816.F16` | 190 |
| `QMMA.16832.F32.E4M3.E4M3` | 197 |
| `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` | **815** |

## Kernels

`native/src/nvfp4_native/native.cuh`. Loaded by `euhedral_cuda_load_native_kernel`, which compiles a
cubin for `sm_<major><minor>a` and only on compute capability 12.x.

- **`euhedral_nvfp4n_quantize_rows`**: one CTA per row.
  - Per-row FP32 global = amax / (6 x 448).
  - Block scale = `e4m3_rn(block amax / (6 x global))`.
  - Codes = `e2m1_rn` with saturation (`F2FP.SATFINITE.E2M1.F32`).
  - Term 1 quantizes the residual x - term0 with the same global.
- **`euhedral_nvfp4n_linear_128x128`**:
  - CTA tile 128 x 128 and K tile 128, 256 threads, with two `cp.async` stages (67,584 bytes of shared memory).
  - Code rows are padded to 80 bytes for conflict-free `ldmatrix`.
  - 8 warps of 64 x 32, epilogue x row global x weight global.
  - SASS: 64 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` per K tile, `LDGSTS`, `LDSM.16.M88.4`; no weight conversion.
- **`euhedral_nvfp4n_gate_up_swiglu_128x64`**: the paired variant puts 16 gate rows and their 16 up rows in each warp's slice and
  applies SwiGLU to BF16-rounded gate and up.
- **`euhedral_nvfp4n_skinny_{16,32,64}`** and **`euhedral_nvfp4n_skinny_finish`**: the decode-like kernels (below).
- Every kernel has an `_sd4` twin for the SD4 scale-table layout ([NVFP4_COMPRESSED.md](NVFP4_COMPRESSED.md)); the twin computes,
  bit for bit, what the plain kernel computes on the tensor expanded to plain NVFP4 (`native/tests/test_nvfp4.py` and
  `test_nvfp4_native.py` check this with every table entry in use). The native kernels stage the index bytes with `cp.async`, half
  of plain's scale bytes, and look each B scale register (4 E4M3 codes) up in registers just before its MMAs: two `prmt` over the
  table's halves, merged on each index's bit 3.

**Dispatch** (`CudaGpuMemory`, `native/src/host/nvfp4_linear.c`):

| Rows | NVFP4 linear (`linearNvfp4Bf16`) | Paired gate/up + SwiGLU (`nvfp4GateUpSwiGluBf16`) |
|---|---|---|
| 1 | the BF16-activation GEMV (`euhedral_nvfp4_decode`) | not formed: decode and small views run gate/up as a linear |
| 2-63 | native skinny kernel when K is a multiple of 256, else the 128 x 128 tile | not formed (small views run gate/up as a linear) |
| 64 | native skinny kernel (same condition) | native paired tile |
| 65 and more | native 128 x 128 tile | native paired tile |

- **Thresholds:** `NVFP4_NATIVE_MIN_ROWS` = 2 for linears; `NVFP4_NATIVE_REGION_MIN_ROWS` = 64 for the paired region; the host's
  `SKINNY_MAX_ROWS` = 64.
- The native route needs K to be a multiple of 128 (the GEMV: K a multiple of 1024 and N of 16, as in every model shape).
  A shape the native route declines runs the decode kernels up to 8 rows, the scalar reference beyond.
- Verification quanta run row-exact (every row bit for bit as one-row decode) and never take the native route: they run the
  GEMV twins (`euhedral_nvfp4_decode_rows<M>`, 2 to 8 rows) as [MTP_CONTRACT.md](MTP_CONTRACT.md) section 6 requires.
- Quantized activations, plus FP32 split-K partials for skinny shapes, go into the shared, event-ordered scratch
  (`euhedral_cuda_nvfp4_native_scratch_bytes`).

## Operator results

Real layer weights, rotated over 4 layers. Production entry points on one stream, CUDA events. Times include activation
quantization (two terms), in ms.

| Family | Rows | Time |
|---|---|---|
| gate_up + SwiGLU | 64 | 0.262 |
| | 256 | 0.632 |
| | 512 | 1.211 |
| | 1024 | 2.405 |
| | 2048 | 4.707 |
| FFN down | 64 | 0.200 |
| | 256 | 0.466 |
| | 512 | 0.724 |
| | 1024 | 1.275 |
| | 2048 | 2.502 |
| GDN output | 64 | 0.074 |
| | 256 | 0.175 |
| | 512 | 0.242 |
| | 1024 | 0.405 |
| | 2048 | 0.878 |

## End to end

The `nvfp4` artifact, native two terms, default 512-row prefill chunk, NVFP4 arms load executed objects only. Paired gates of 6
forks, each arm in a fresh JVM, start order rotated per fork, warmup 2, 3 iterations; medians of per-fork medians.

| Scenario | Result |
|---|---|
| prefill 8 | 278 tok/s |
| prefill 16 | 557 tok/s |
| prefill 32 | 1003 tok/s |
| prefill 64 | 1492 tok/s |
| prefill 256 | 2211 tok/s |
| prefill 512 | 2506 tok/s |
| prefill 1024 | 2504 tok/s |
| prefill 2048 | 2477 tok/s |
| TTFT, 16-token prompt | 29.1 ms |
| TTFT, 64-token prompt | 44.8 ms |
| TTFT, 1024-token prompt | 434 ms |
| decode at 64 / 1024 tokens (one row, GEMV) | 51.2 / 50.6 tok/s |

Rows of 64 and fewer use the skinny kernel; larger rows form quanta of more than 64 rows, which the 128 x 128 tile serves.
These gates predate the producer-warp FA2 prefill attention ([PREFILL_MX.md](PREFILL_MX.md)), so the attention share of long-context prefill has changed.

## Decode-like row counts

One-row decode streams every weight once per token, so its floor is bytes over DRAM bandwidth. The
NVFP4 GEMV already reaches it (gate_up: 100 MB in 0.128 ms, 780 GB/s). A tensor-core tile needs 16
rows, so the question is what extra rows cost.

**Skinny native kernel** (`nvfp4n::skinny_linear`):
- 4 warps (128 threads) over 64 output columns and all M <= 64 rows; the 16, 32 and 64 row variants hold 1, 2 and 4 m16 fragments.
- 256-value K tiles, up to 4 `cp.async` stages.
- K is split when 64-column tiles alone cannot occupy 140 CTAs (two per SM); each split stores FP32 partials, and
  `euhedral_nvfp4n_skinny_finish` sums them in split order and converts once, so the result is deterministic.
- It takes 2-64 rows. One row stays on the GEMV, which keeps BF16 activations.

**Operator times** (ms; real weights rotated over 4 layers; native includes quantization and split-K finish):

| Tensor (NVFP4 MB) | M | GEMV (M = 1) | Native skinny, 2 terms |
|---|---|---|---|
| gate_up (100.3) | 1 | 0.128 | |
| | 2 | | 0.130 |
| | 4 | | 0.130 |
| | 8 | | 0.131 |
| | 16 | | 0.132 |
| | 64 | | 0.178 |
| down (50.1) | 1 | 0.068 | |
| | 4 | | 0.076 |
| | 16 | | 0.082 |
| GDN output (17.7) | 1 | 0.024 | |
| | 4 | | 0.029 |
| | 16 | | 0.030 |
| LM head (715.2) | 1 | 0.897 | |
| | 4 | | 0.895 |
| | 16 | | 0.904 |

**Native NVFP4 is flat in M up to 16 rows, at the weight-streaming floor** (gate_up 0.130-0.132 ms), and 0.178 ms at 64 rows.

**Whole-model estimate per decode step** (linears weighted by layer count: 48 GDN layers (query_key, value_z, output), 16
attention layers (query_key, gate_value, output), 64 x (gate_up, down), and the LM head; step time = the measured one-row step of
19.52 ms at 51.24 tok/s + linear(M) - linear(1); linears are 94% of that step, 18.3 ms; attention and GDN growth with M is not
included):

| M | Step |
|---|---|
| 1 | 19.5 ms (GEMV) |
| 2 | 21.6 ms |
| 4 | 21.8 ms |
| 8 | 21.8 ms |
| 16 | 21.8 ms |
| 32 | 23.7 ms |
| 64 | 31.3 ms |

## Numerics

**Operator level** (Gaussian activations, real-shape converter-quantized weights):
- Native output equals float64 over the same quantized operands within BF16 output rounding (max
  2.5e-3 of the output range).
- Against the BF16-activation result, two terms give a linear output relative RMS error of 0.88% (activation quantization 0.87%).

**Model level** (`RelaxedNumericsDriftCudaIntegrationTest`, NVFP4 artifact, executed objects):
- Teacher-forced against the exact-numerics oracle. Native changes prefill only, because decode runs the same one-row GEMV in
  both arms. "Floor" = native off: only the relaxed-order kernels differ. Exact against exact is bitwise zero.

| Prefix + steps | Arm | Median hidden error | Worst 1/8 window | KL mean | Top-1 | Harness |
|---|---|---|---|---|---|---|
| 256 + 384 | two terms | 4.64% | 5.22% | 3.78e-3 | 97.4% | passes |
| 512 + 1024 | floor | 5.45% | 9.33% | 3.10e-3 | 97.9% | passes |
| 512 + 1024 | two terms | 5.50% | 10.41% | 3.39e-3 | 97.6% | fails window bound late (position 1408), where the floor reaches 9.3% |

Two terms sit at the floor and do not grow with position. A 1024-token single-quantum prefix does not fit beside 14 GB of weights
and two sequences; 512 is the production chunk. No tolerance was changed.

## Rejected

- **One activation term** (the quantized value alone). The linear output relative RMS error is 9.5% (activation quantization
  9.5%) against 0.88% with two terms, and choosing the minimum-error scale among six candidates per block would only reach 8.8%.
  Model level, it doubles KL: 256 + 384: median hidden error 6.39%, worst window 10.03%, KL 6.00e-3, top-1 95.1%, failing the
  window bound (0.10) at the first window; 512 + 1024: 7.37%, 11.84%, 6.59e-3, 97.0%, failing the window bound. The saving was
  gate_up at 1024 rows in 1.573 ms instead of 2.405 ms.
- **Native FP4 for one row.** Native one-row decode costs 20.0 ms of linears per token against 18.3 ms for the GEMV (9% slower), so
  one row keeps the GEMV with BF16 activations.
- **Native FP4 drafting** (one row included, where the GEMV stays). Every NVFP4 linear in draft quanta ran native: no faster than the GEMV and acceptance unchanged
  ([MTP_VERIFIER.md](MTP_VERIFIER.md)).
- **Native FP4 verification** (every verifier row on the skinny kernel): about 4% faster than exact verification at depth 4
  against 3, but it changed every generated text (top-1 96.4%, KL 3-6e-3), so verification stays row-exact
  ([MTP_VERIFIER.md](MTP_VERIFIER.md)).
