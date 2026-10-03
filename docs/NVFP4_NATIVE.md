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
  - Buffer layout (`nvfp4n::ActivationLayout`): the code planes (`terms x rows` rows of K/2 bytes), then the scales **tile-major**,
    then one FP32 global per row. The 8 scale bytes of a row for K tile j and term t are at
    `scales + ((j * terms + t) * pad + row) * 8` with `pad` the rows rounded up to 128, so the scales of 128 rows of one tile and
    term are one contiguous KiB. A row-major plane would make every tile load 128 separate 8-byte pieces.
- **`euhedral_nvfp4n_linear_128x128`**: a TMA, warp-specialized tile.
  - CTA tile 128 x 128, K tile 128, 288 threads: eight consumer warps (2 along M x 4 along N, 64 x 32 each) and one producer
    warp whose first lane issues every load. Three stages of 26,624 bytes plus two scale slots (85,040 bytes of shared memory,
    one CTA per SM), guarded by an mbarrier pair per stage (full: the TMA transaction count; empty: one arrival per consumer warp).
  - Code tiles come from two tensor maps (activation planes `[K/2, rows, 2]`, weight rows `[K/2, N]`) with 64-byte boxes and
    `SWIZZLE_64B`, so every `ldmatrix` phase is conflict-free without padding. Out-of-range rows are zero-filled by the map.
  - Activation scales are tile-major (see the activation layout below): one 1 KiB bulk copy per term and K tile.
    Weight scales come from a third map in 16-byte boxes that cover two K tiles (four for the SD4 indices), fetched with the
    first tile of the group into one of two slots.
  - The consumers never issue a load. SASS: 64 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` per K tile and warp, `LDSM.16.M88.4`,
    `UBLKCP` and `SYNCS`; no weight conversion. The grid is `tiles_m * tiles_n` with M varying fastest, so the CTAs that run
    together share weight tiles. Epilogue: x row global x weight global.
  - Each output element accumulates its K tiles in order, two MMAs (one per term) per K step, as the scalar definition of the
    kernel (`test_nvfp4_native.py` checks it against float64 over the quantized operands).
  - Requirements (host dispatch): K a multiple of 256 (512 for SD4 weights, so the scale rows are multiples of 16 bytes), 16-byte
    aligned weights and scratch.
- **`euhedral_nvfp4n_gate_up_swiglu_128x64`**: the paired variant. A tile holds 64 gate rows and the matching 64 up rows (two boxes
    per operand); each warp's fragments 0 and 1 are 16 gate columns and fragments 2 and 3 the same up columns, and SwiGLU is applied
    to BF16-rounded gate and up.
- **`euhedral_nvfp4n_skinny_{16,32,64}`** and **`euhedral_nvfp4n_skinny_finish`**: the decode-like kernels (below).
- Every kernel has an `_sd4` twin for the SD4 scale-table layout ([NVFP4_COMPRESSED.md](NVFP4_COMPRESSED.md)); the twin computes,
  bit for bit, what the plain kernel computes on the tensor expanded to plain NVFP4 (`native/tests/test_nvfp4.py` and
  `test_nvfp4_native.py` check this with every table entry in use). The native kernels fetch the index bytes (half of plain's scale
  bytes, 16-byte boxes of four K tiles) and look each B scale register (4 E4M3 codes) up in registers just before its MMAs: two `prmt` over the
  table's halves, merged on each index's bit 3.

**Dispatch** (`CudaGpuMemory`, `native/src/host/nvfp4_linear.c`):

| Rows | NVFP4 linear (`linearNvfp4Bf16`) | Paired gate/up + SwiGLU (`nvfp4GateUpSwiGluBf16`) |
|---|---|---|
| 1 | the BF16-activation GEMV (`euhedral_nvfp4_decode`) | not formed: decode and small views run gate/up as a linear |
| 2-63 | native skinny kernel when K is a multiple of 256, else the tile when it qualifies | not formed (small views run gate/up as a linear) |
| 64 | native skinny kernel (same condition) | native paired tile |
| 65 and more | native 128 x 128 tile | native paired tile |

- **Thresholds:** `NVFP4_NATIVE_MIN_ROWS` = 2 for linears; `NVFP4_NATIVE_REGION_MIN_ROWS` = 64 for the paired region; the host's
  `SKINNY_MAX_ROWS` = 64.
- The native route needs K to be a multiple of 128, and for the tile kernels a multiple of 256 (512 for SD4 weights) with 16-byte
  aligned weights and scratch (the GEMV: K a multiple of 1024 and N of 16, as in every model shape).
  A shape the native route declines runs the decode kernels up to 8 rows, the scalar reference beyond.
- Verification quanta run row-exact (every row bit for bit as one-row decode) and never take the native route: they run the
  GEMV twins (`euhedral_nvfp4_decode_rows<M>`, 2 to 8 rows) as [MTP_CONTRACT.md](MTP_CONTRACT.md) section 6 requires.
- Quantized activations, plus FP32 split-K partials for skinny shapes, go into the shared, event-ordered scratch
  (`euhedral_cuda_nvfp4_native_scratch_bytes`).

## Operator results

Synthetic FFN-shaped tensors with random codes and scales, tile kernels only (activation quantization is below), one stream,
CUDA events, the average of 20 launches. TFLOPS count both activation terms (2 x 2 x rows x N x K).

| Family (N x K) | Rows | Plain | | SD4 | |
|---|---|---|---|---|---|
| | | us | TFLOPS | us | TFLOPS |
| gate_up + SwiGLU (34816 x 5120) | 256 | 300 | 608 | 311 | 587 |
| | 512 | 591 | 617 | 615 | 593 |
| | 1024 | 1183 | 617 | 1221 | 598 |
| | 2048 | 2339 | 624 | 2407 | 607 |
| FFN down (5120 x 17408) | 256 | 218 | 419 | 227 | 403 |
| | 512 | 323 | 565 | 336 | 543 |
| | 1024 | 531 | 687 | 555 | 658 |
| | 2048 | 1063 | 687 | 1114 | 655 |
| GDN output / attention output (5120 x 6144) | 256 | 80 | 404 | 83 | 387 |
| | 512 | 120 | 537 | 125 | 517 |
| | 1024 | 197 | 655 | 205 | 629 |
| | 2048 | 390 | 661 | 406 | 635 |
| GDN/attention input (12288 x 5120) | 256 | 102 | 634 | 106 | 610 |
| | 512 | 200 | 644 | 208 | 621 |
| | 1024 | 368 | 700 | 383 | 673 |
| | 2048 | 733 | 703 | 767 | 672 |

The block-scaled MMA loop alone peaks at 815 TFLOPS. Activation quantization (two terms) takes 6.1, 9.6 and 16 us for 512, 1024
and 2048 rows of K = 5120, and 16, 47 and 156 us for K = 17408.

## End to end

The `nvfp4` artifact, native two terms, default 512-row prefill chunk, one run (warmup 1, 3 iterations for the short prompts).

| Scenario | Result |
|---|---|
| prefill 64 | 1567 tok/s |
| prefill 256 | 3577 tok/s |
| prefill 512 | 4180 tok/s |
| prefill 1024 | 4186 tok/s |
| prefill 2048 | 4160 tok/s |
| TTFT, 16-token prompt | 38 ms |
| TTFT, 64-token prompt | 46 ms |
| TTFT, 1024-token prompt | 327 ms |

Chat corpus, MTP3, default 32768-token context (65536 for the 59K row), 128 generated tokens:

| Artifact | Prompt tokens | Prefill tok/s | TTFT |
|---|---|---|---|
| `nvfp4` | 3,964 | 3,874 | 1.13 s |
| | 15,930 | 3,254 | 5.19 s |
| | 31,906 | 2,762 | 11.98 s |
| | 59,111 | 2,145 | 29.0 s |
| `nvfp4-compressed` | 3,964 | 3,815 | 1.14 s |
| | 15,930 | 3,219 | 5.23 s |
| | 31,906 | 2,728 | 12.09 s |
| | 59,111 | 2,131 | 29.2 s |

Rows of 64 and fewer use the skinny kernel; larger rows form quanta of more than 64 rows, which the tile serves. In a 4K prefill
the two tile kernels take 68% of the kernel time (about 650 TFLOPS), the GDN recurrence 10%, FA2 attention 8%, and quantization
2.5%.

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
- **cp.async tile variants** (kept in the record because they located the limit). The tile is bound by its loads, not its MMAs:
  with the loads removed it ran at about 600 TFLOPS, with the MMAs removed it took nearly as long as the full kernel.
  - Row-major activation scales (8-byte pieces per row and K tile) cost a third of the load time: gate_up at 1024 rows took
    2.25 ms with them and 1.65 ms with tile-major scales.
  - Sixteen-byte B scale boxes on the cp.async tile: 1.61 ms. Burst-loading the scales of four K tiles into an eight-tile ring: no gain.
  - M-fastest rasterization on the cp.async tile: 1.65 ms, no change (DRAM was at 40%).
  - Three stages instead of two: 1.55 ms. A 128 x 256 tile: 1.52 ms (L2 traffic per FLOP down a third; the kernel was not L2-bound).
- **Persistent CTAs** on the TMA tile (one per SM looping over output tiles, so epilogues overlap the next tile's loads): gate_up at
  1024 rows 1.21 ms and down 0.56 ms against 1.18 and 0.53 ms for one CTA per tile.
- **Two stages instead of three** on the TMA tile: the same time; four stages do not fit the 99 KiB limit.
- **`cp.async.bulk` with the `.shared::cluster` destination** compiled to a helper call and 8 bytes of local memory per thread; the
  `.shared::cta` forms need none, which matters because the first launch of a kernel with local memory allocates device memory
  that a model filling the card may not have.
- **Larger tiles on `setmaxnreg`** (384 threads: a producer warpgroup at 40 registers, two consumer warpgroups at 232; the 288-thread
  block is capped at 168 registers because the allocation rounds to warpgroups). At 1024 rows, kernel only:

  | Tile | gate_up us | down us | GDN output us | input us |
  |---|---|---|---|---|
  | 128 x 128 (as shipped, 168 registers) | 1158 | 525 | 195 | 365 |
  | 128 x 256 | 1119 | 622 | 230 | 392 |
  | 256 x 128 | 1121 | 619 | 230 | 389 |

  The 64 x 64 warp tiles shorten the MMA loop only slightly and cost wave quantization: down at 1024 rows is 160 tiles on 70 SMs
  (3 waves, 76% used) against 320 tiles of 128 x 128 (5 waves, 91%); at 512 rows down takes 416 us and GDN output 153 us.
  Only gate_up gains (3% at 512 to 2048 rows, about 1% of a prefill), which does not pay for a second tile shape and its SD4 twin.
  A 192 x 128 tile would use 3 of 4 m-tiles' worth of a 512-row chunk (89%).
- **`setmaxnreg` on the 128 x 128 tile**: 1158 against 1165 us for gate_up, no change.
- **Persistent CTAs again, on the `setmaxnreg` layout**: the per-tile fixed cost falls (gate_up with K = 2560: 668 against 695 us)
  but the K = 20480 time rises (4206 against 4112 us) and K = 5120 is unchanged (1156 against 1163 us).
- **Where the time goes.** gate_up at 1024 rows takes 695, 1163, 2131 and 4112 us for K = 2560, 5120, 10240 and 20480: the
  asymptotic block-scaled MMA loop runs at 737 TFLOPS (90% of the 815 TFLOPS peak) and each kernel carries about 150 us of fixed
  cost (pipeline fill, epilogue and wave tails over 31 waves), which is the remaining headroom of the tile.
