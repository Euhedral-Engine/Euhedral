# Tensor cores on the RTX 5070 Ti (sm_120)

Consumer Blackwell has no `tcgen05`, no Tensor Memory and no `wgmma`. Its 5th-generation tensor
cores (4 per SM, 280 in total) are driven only by warp-level `mma.sync`, which gains new kinds:
FP8/FP6/FP4 operands and hardware block scaling.

This document measures every kind on the card and gives the rules that follow. Every number is in
[microbench/RESULTS.md](microbench/RESULTS.md) (`nvbench mma`, `nvbench mma num`).

## Throughput, measured

**Setup**
- Register-only loops, 4 independent accumulator chains per warp, 8–32 warps per SM. The best
  configuration is reported. SM clock was about 2.86 GHz.
- TFLOPS are dense-equivalent: 2·M·N·K per instruction. Sparse rows count the full K.
- "lat" is single-warp dependent latency in cycles.

| PTX form | SASS | TFLOPS | FLOP/clk/SM | lat |
|---|---|---|---|---|
| `m16n8k16 .f32.f16.f16.f32` | HMMA | 102.2 | 511 | 34.7 |
| `m16n8k16 .f32.bf16.bf16.f32` | HMMA | 102.5 | 510 | 34.7 |
| `m16n8k16 .f16.f16.f16.f16` | HMMA | **204.4** | 1019 | 29.2 |
| `m16n8k8 .f32.tf32.tf32.f32` | HMMA | 51.3 | 256 | 34.7 |
| `m16n8k32 .f32.e4m3.e4m3.f32` | QMMA | 204.9 | 1025 | 34.7 |
| `m16n8k32 .f16.e4m3.e4m3.f16` | QMMA | **408.8** | 2038 | 29.2 |
| `kind::f8f6f4 .f32.e4m3.e4m3.f32` | QMMA | 205.0 | 1021 | 34.7 |
| `kind::f8f6f4 .f16.e4m3.e4m3.f16` | QMMA | **407.9** | 2046 | 29.2 |
| `kind::f8f6f4 .f32.e3m2.e3m2.f32` (FP6) | QMMA | 203.9 | 1021 | 34.7 |
| `kind::f8f6f4 .f32.e2m1.e2m1.f32` (FP4 in 8-bit containers) | QMMA | 204.2 | 1021 | 34.7 |
| `kind::f8f6f4 .f32.e2m1.e4m3.f32` (mixed) | QMMA | 203.9 | 1021 | 34.7 |
| `kind::mxf8f6f4.block_scale.scale_vec::1X .f32.e4m3.e4m3.f32.ue8m0` | `QMMA.SF.16832.F32.E4M3.E4M3.E8` | **407.7** | 2039 | 29.2 |
| `kind::mxf8f6f4.block_scale.scale_vec::1X .f32.e2m1.e2m1.f32.ue8m0` | QMMA.SF | 205.0 | 1020 | 34.7 |
| `m16n8k64 kind::mxf4.block_scale.scale_vec::2X ... .ue8m0` (MXFP4) | OMMA.SF | **815.4** | 4058 | 29.2 |
| `m16n8k64 kind::mxf4nvf4.block_scale.scale_vec::4X ... .ue4m3` (NVFP4) | `OMMA.SF.16864...UE4M3.4X` | **815.4** | 4059 | 29.2 |
| `m16n8k64 kind::mxf4nvf4.block_scale.scale_vec::2X ... .ue8m0` | OMMA.SF | 815.5 | 4070 | 29.2 |
| `m16n8k32 .s32.s8.s8.s32` | IMMA | **407.8** | 2037 | 27.2 |
| `m16n8k64 .s32.s4.s4.s32` | emulated: `IMMA.16832.S8.S8` + unpack | 41.9 | 210 | 398.7 |
| `m16n8k256 .s32.b1.b1.s32.and.popc` | emulated | 251.5 | 1262 | 383.7 |
| `mma.sp::ordered_metadata m16n8k32 .f32.f16.f16.f32` (2:4) | HMMA.SP | 203.9 | 1019 | 34.7 |
| `mma.sp::ordered_metadata m16n8k64 .f32.e4m3.e4m3.f32` (2:4) | `QMMA.SP.16864` | 407.8 | 2041 | 34.7 |
| `mma.sp::ordered_metadata m16n8k64 kind::f8f6f4 e2m1` | QMMA.SP | 408.3 | 2039 | 34.7 |
| `mma.sp::ordered_metadata m16n8k128 kind::mxf4` | OMMA.SF.SP | **1617.4** | 8105 | 29.8 |
| `mma.sp::ordered_metadata m16n8k128 kind::mxf4nvf4 ... 4X .ue4m3` | `OMMA.SF.SP.168128...UE4M3.4X` | **1621.3** | 8132 | 29.8 |

**Per SM per clock**
- Peak rates are 4096 FLOP (FP4), 2048 (FP8 / INT8 / FP16-accumulate FP8), 1024 (FP16 accumulate,
  FP8 with FP32 accumulate), 512 (FP16/BF16 with FP32 accumulate) and 256 (TF32).
- These match the whitepaper's 5070 Ti table at 2452 MHz scaled to the observed clock
  ([whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf),
  p.52).
- Sustained loops ran at 2.80–2.85 GHz and 130–230 W. The power cap was not reached (see
  [system-tuning.md](system-tuning.md)).

**Agreement with earlier measurements.** docs/NVFP4_NATIVE.md measured BF16 98, FP16 190, E4M3 197
and NVFP4 815 TFLOPS with a different harness; these figures agree.

## The GeForce rules

1. **FP32 accumulation is half rate for every non-block-scaled kind.** This covers HMMA FP16/BF16 and
   QMMA FP8, FP6 and FP4 under `kind::f8f6f4`. FP16 accumulation is full rate.
2. **Block-scaled kinds accumulate in FP32 at full rate.**
   - `kind::mxf8f6f4` with E4M3 operands and unit UE8M0 scales (`0x7f7f7f7f`) runs at 408 TFLOPS,
     exactly 2× plain FP8 with FP32 accumulate.
   - FlashInfer found the same on an RTX 5090: 510 → 1014 TFLOPS, with bit-identical results and
     1.29–1.34× on real attention shapes
     ([flashinfer#5963](https://github.com/flashinfer-ai/flashinfer/issues/5963)).
   - The workstation RTX PRO 6000 is reported to run FP32 accumulate at full rate everywhere.
3. **E2M1 inside `mxf8f6f4` is half rate.** The block-scaled bypass needs E4M3 operands for FP8.
   FP4 needs the packed `mxf4` / `mxf4nvf4` kinds; these are always full rate.
   - Not measured: E4M3×E2M1 *mixed* under `mxf8f6f4`, and FP16-accumulate mixed kinds. The probe
     was written but not run.
4. **INT8 is full rate (408 TOPS) with exact integer accumulation.** INT4 and binary MMA no longer
   exist in hardware; ptxas emulates them through IMMA.S8 (`s4` measured at 42 TFLOPS, 400-cycle
   latency). Do not use them.
5. **FP6 saves bytes, not FLOPs.** E3M2/E2M3 sit in 8-bit containers at the FP8 rate.
6. **2:4 structured sparsity doubles every rate.** The sparse FP4 kinds are `sm_120a`/`sm_121a`
   only, with no `f` family form.

## Accumulator precision, measured

Two probes (`nvbench mma num`) feed one MMA whose only non-zero inputs are row 0 of A, column 0 of B
and C[0][0].

| Accumulator | Behaviour |
|---|---|
| FP32: f16, bf16, e4m3, `mxf8f6f4` e4m3, `mxf4`, `mxf4nvf4` | **24-bit significand, truncation.** C = 2^e plus three products of 1.0 is exact through e = 23. At e = 24 the result is +2 (round-to-nearest-even would give +4). At e ≥ 25 the result is +0. One product of 2^e plus three products of 1.0 inside the MMA behaves the same, so products are aligned without losing bits before the sum. |
| FP16: `.f16` forms | **IEEE FP16 with round-to-nearest.** Exact through 2^10; 2^11 + 3 gives 2^11 + 4. |

Consequences:
- **The full-rate MXFP8 path is genuinely FP32.** It isn't a reduced-precision accumulator in
  disguise. Within this probe it is indistinguishable from plain QMMA with FP32 accumulate.
- **FP32-accumulating tensor results truncate.** Tensor-core output differs from an FFMA chain, which
  rounds to nearest. That matters for Euhedral's bitwise contracts: an exact twin must use the same
  instruction sequence as its oracle. Also, a tensor-core MMA result for one row never depends on the
  other rows of the tile, so a multi-row kernel that issues the same MMAs as its one-row oracle is
  bitwise row-exact by construction.
- **FP16 accumulation loses 13 bits.** Use it only with periodic promotion into FP32 (see Prefill
  below). SageAttention2++ bounds FP16-accumulate overflow for FP8 PV at Pr·Vr ≤ 2047 per 32
  products ([arXiv 2505.21136](https://arxiv.org/html/2505.21136)).

## Instruction details that bite

- **Targets.**
  - `kind::*`, `.block_scale`, the FP6/FP4 types, the new `ldmatrix`/`stmatrix` shapes, the FP4/FP6
    `cvt` forms and `setmaxnreg` need `sm_120a`, or `sm_120f` from PTX 8.8 / CUDA 12.9
    ([NVIDIA blog](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)).
  - Euhedral's generic modules are `compute_90` PTX, so none of this is reachable from them. Only
    `nvfp4_native` is built for `sm_120a` (see [host-link-and-runtime.md](host-link-and-runtime.md)).
- **Qualifier order.** Both `mma.sync.aligned.m16n8k64.row.col.kind::...` (Euhedral) and CUTLASS's
  `mma.sync.aligned.kind::....m16n8k64.row.col...` are accepted.
- **Register layouts.** g = lane/4, t = lane%4.
  - m16n8k32 8-bit: A in 4 registers, bytes `[g][4t..4t+3]`, `[g+8][...]`, then `+16` in K. B in 2
    registers, `[4t..4t+3][g]`, then K + 16.
  - m16n8k64 FP4 (packed, nibble j = K 8t+j): A rows g / g+8, K 8t..8t+7, then K + 32. B column g.
  - D: d0, d1 at row g, columns 2t, 2t+1; d2, d3 at row g+8.
  - The NVFP4 scale-register contract (which lane supplies which row's scale, byte b ↔ K 16b..16b+15)
    is in docs/NVFP4_NATIVE.md "Instruction contract", measured on this card.
- **Containers.** Under `kind::f8f6f4`, E2M1 sits in bits 2–5 of each byte and FP6 in the low 6
  bits. The packed `mxf4` kinds use plain nibbles. `ldmatrix ... .b8x16.b4x16_p64` lands 4-bit data
  in the low bits, so CUTLASS shifts it left by 2 before an `f8f6f4` MMA
  ([mma_traits_sm120.hpp](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/atom/mma_traits_sm120.hpp)).
- **Mixing NVFP4 weights with FP8/BF16 activations is impossible in hardware.** `mxf8f6f4` takes only
  UE8M0 scales per 32, and NVFP4 has E4M3 per 16 (docs/NVFP4_NATIVE.md). Native NVFP4 needs FP4
  activations.
- **No TMA multicast on GeForce.** CUTLASS SM120 GEMMs run with cluster shape 1×1×1
  ([CUTLASS example 79a](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)).

## How to use them in Euhedral

### Decode (M = 1 to 4 rows)

- **Decode does not need tensor-core FLOPs.** One token of the 27B model is about 54 GFLOP per row.
  Even the half-rate BF16 path, 102 TFLOPS, could absorb roughly 3 TB/s of FP4 weights, which is
  3.5× DRAM.
- **What tensor cores buy decode is moving the FMAs off the CUDA cores.** Today's scalar GEMVs do
  1 FMA per weight per row on top of the dequant. At 4 rows they become ALU- and register-bound:
  about 600 GB/s, with a 223-register cliff at M = 4 (docs/MTP_VERIFIER.md).
- **Layout.** "Swap A and B": weights take the M=16 side, and the 1–4 activation rows sit in the n=8
  columns of B. Dequant then happens once per weight for any M ≤ 8, and the dot products are free.
- **Precision.** E2M1 × E4M3 scale products need at most 5 significant bits, so NVFP4 weights expand
  exactly into FP16 or BF16 fragments. Q3 codes (−4..3) are exact in every format, including E2M1.
- **Row-exactness.** It comes for free if the one-row oracle runs the same MMA sequence, as the
  native NVFP4 mode already does.
- **Prior art:**
  - Marlin-style offline fragment shuffling so each A fragment is one 128-bit load
    ([Marlin](https://arxiv.org/html/2408.11743)).
  - tilelang's W4A16 places E2M1 as BF16 subnormals and fixes the scale with one PRMT + IMAD; it is
    1.2–4.6% faster than Marlin at M = 1–32
    ([tilelang#3290](https://github.com/tile-ai/tilelang/pull/3290)).

### Prefill (compute-bound)

| Path | Peak on this card | Notes |
|---|---|---|
| BF16 HMMA, FP32 accumulate (Q3/Q4/Q5 today) | 102 | Achieved 57 TFLOPS in the FFN at 512 rows (docs/FRAME_MODEL.md) |
| FP16 HMMA, FP16 accumulate inside each 64-K Q3 group, then promotion to FP32 with the group scale | 204 | The scale is applied per group anyway; overflow and drift need the drift test |
| MXFP8 (`mxf8f6f4` E4M3, unit UE8M0) with FP8 activations | 408 | FP32 accumulate at full rate; activation quantization changes numerics |
| Two-term FP8 activations (hi + lo) through MXFP8 | 204 effective | About 7–8 significant bits, near BF16 |
| INT8 IMMA with two-term int8 activations (int16 split into hi/lo bytes) | 204 effective | Exact, order-independent integer accumulation per Q3 group; Q3 codes are already small integers |
| NVFP4 native, 1–2 activation terms (today) | 815 | Achieves 225–290 TFLOPS, about 30% (docs/NVFP4_NATIVE.md) |

- **The NVFP4 tile is the largest headroom.**
  - CUTLASS-style SM120 block-scaled GEMMs reach 83% of FP4 peak on an RTX PRO 6000 with TMA
    loads, a store warp, 2–4 stages, `setmaxnreg` 40/232, tile swizzling and a bank-conflict fix
    in the SFA layout
    ([Colfax](https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/)).
  - llama.cpp's MXFP4 MMA path raised prompt throughput on a 5070 Ti from 6461 to 7755 t/s
    ([llama.cpp#17906](https://github.com/ggml-org/llama.cpp/pull/17906)).
  - Euhedral's native tile uses `cp.async` and no TMA.

### Attention

- **Prefill attention is FA2-style HMMA FP16 with FP32 accumulate**, so it is capped at 102 TFLOPS.
- **FP8 alternatives:**
  - SageAttention2++ runs PV as `mma.f16.f8.f8.f16` (FP16 accumulate) with two-MMA FP16 partials
    folded into FP32 ([arXiv 2505.21136](https://arxiv.org/html/2505.21136)).
  - The MXFP8 bypass gives FP32 accumulation at the same 408-TFLOPS rate without the overflow
    analysis.
- **FP4:** SageAttention3 runs both QKᵀ and PV in NVFP4 with two-level scaling of P. It reaches
  1038 TOPS on a 5090, 5× FA2 ([arXiv 2505.11594](https://arxiv.org/html/2505.11594)).
- **Long-context decode attention is bandwidth- and parallelism-bound, not tensor-bound.** See
  [memory-system.md](memory-system.md).

### Sparsity

- 2:4 sparse FP4 runs at 1621 TFLOPS.
- For decode it would also cut weight bytes. Half the codes is 2 bits per weight; the metadata adds
  0.4–1 bit (depending on 2:4 vs the pairwise 4:8 form) and the scales add more. The total is
  roughly 3–3.5 bits per weight, against 4.5 for NVFP4.
- Pruning a dense 27B model to 2:4 costs quality that usually needs retraining. This is research,
  not a quick win.
