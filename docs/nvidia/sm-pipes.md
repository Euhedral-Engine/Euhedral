# SM execution pipes: CUDA cores, SFUs and conversions

**Layout.** Each of the 70 SMs has 4 partitions. Each partition has one warp scheduler issuing one
warp-instruction per clock, 32 FP32/INT32 lanes, one tensor core, load/store units and an SFU.

**What this means.**
- Issue tops out at 128 thread-instructions per clock per SM.
- Some instructions can only use part of that. Others run on separate pipes and overlap.

Logs: [microbench/RESULTS.md](microbench/RESULTS.md) (`nvbench alu`). Each probe runs 8 independent
dependency chains of one PTX instruction per thread at 1536 threads per SM. Rates are per SM per SM
clock.

## Measured rates

| Instruction | Thread-instr/clk/SM | Results/clk/SM | Notes |
|---|---|---|---|
| `fma.rn.f32` (FFMA) | 122.7 | 122.7 | The issue limit is 128 |
| `fma.rn.f32x2` (packed FP32, FFMA2) | 62.5 | 125 | Exists on sm_120 (needs an sm_100+/sm_120 target). Same FLOPs, **half the issue slots** |
| `add.rn.f32x2` | 63.0 | 126 | |
| `fma.rn.f16x2` (HFMA2) | 63.0 | 126 | **FP16 math is 1:1 with FP32.** The 12.9 programming guide's 256 is wrong for this part |
| `fma.rn.bf16x2` | 62.9 | 126 | |
| `max.f32` | 123.9 | 123.9 | |
| `add.u32` (IADD3) | 122.1 | 122.1 | Integer add is on the unified 128-lane path |
| `mad.lo.u32` (IMAD) | 62.9 | 62.9 | |
| `lop3.b32` | 63.1 | 63.1 | |
| `shf.l.wrap.b32` | 63.1 | 63.1 | |
| `prmt.b32` | 62.2 | 62.2 | |
| `dp4a.u32.u32` | 62.6 | 250 MACs | |
| `popc` | 16 | 16 | |
| `cvt.rn.f32.u32` (I2F) | 63.6 | 63.6 | |
| `cvt.rzi.u32.f32` (F2I) | 16 | 16 | |
| `cvt.rn.f16x2.f32` (pack) | 63.6 | 127 values | |
| `cvt.rn.satfinite.e4m3x2.f32` | about 32 | 64 values | |
| `cvt.rn.f16x2.e4m3x2` | 36.6 | 73 values | |
| `cvt.rn.f16x2.e2m1x2` (`F2FP.F16.E2M1.UNPACK_B`) | | **85 FP4 values** | In a probe of 4 cvt + 3 xor per 8 values |
| `cvt.rn.satfinite.e2m1x2.f32` | about 32 | 64 values | |
| `ex2`, `rsqrt`, `lg2`, `sin`, `tanh`, `rcp` `.approx` (MUFU) | 16 | 16 | |
| `ex2.approx.f16x2`, `ex2.approx.ftz.bf16x2`, `tanh.approx.f16x2` | 8 | 16 | Packed SFU ops give no extra results |
| `shfl.sync` | 16 | 16 | One warp shuffle per 2 clocks per SM |
| `redux.sync.add.u32` | 16 | 16 | |
| `fma.rn.f64` | 1.7 | 1.7 | 1/64 rate |

**Overlap tests (independent chains in one thread):**

| Mix | Thread-instr/clk/SM | Meaning |
|---|---|---|
| FFMA + LOP3 | 118 | Logic ops run on their own pipe, overlapping the FMAs up to the issue limit |
| FFMA + IMAD | 112 | IMAD mostly overlaps too |
| 4 FFMA + 1 EX2 | 67.5 | 13.5 `ex2` plus 54 FFMA per clock; the SFU and FMA pipes overlap imperfectly |

## What the numbers mean

**Integer throughput.** The whitepaper's "fully unified INT32" holds for adds, compares and min/max,
which run at 128 per clock. IMAD, LOP3, SHF and PRMT stay at 64 per clock, as on Ada. They run on a
pipe separate from FFMA, so a dequant loop mixing logic ops with FP32 FMAs can reach about 118 of 128
issue slots. The architecture research found the 12.9 throughput table says the same.

**FP16 on CUDA cores buys nothing.** HFMA2 delivers the same element rate as FFMA. Packed halves only
save registers and bandwidth.

**FFMA2 is an issue-slot tool, not a FLOP tool.**
- A GEMV that is issue-bound (dequant plus several rows of FMAs) can halve its FMA instruction count
  with `fma.rn.f32x2`, which leaves room for the dequant instructions.
- This applies to Euhedral's multi-row verify GEMVs.
- Packed FMA keeps IEEE round-to-nearest per element. Whether an `x2` rewrite of an FFMA chain is
  bitwise identical depends only on keeping the same operation order, so it fits the exact twins.

**The SFU is 16 per clock.**
- Softmax `exp`, SiLU's sigmoid, GDN gating (exp, softplus, sigmoid) and RMSNorm's `rsqrt` share it.
- Decode volumes are small: about 25M `exp` per token at 64K context across 16 attention layers, a
  few µs.
- Long prefill is different. At 64K the attention `exp` count is about 8·10^11, about 0.3 s at
  16 per clock per SM.
- FlashAttention-4 on B200 emulates part of `exp2` with an FMA polynomial for this reason. Here the
  FMA pipe can absorb a few `exp`s per clock alongside the SFU.

**Hardware FP4/FP8 conversion exists.**
- `F2FP.F16.E2M1.UNPACK_B` expands two E2M1 codes to an FP16 pair in one instruction. The probe got
  85 FP4 values per clock per SM, about 10× what NVFP4 decode at DRAM speed needs (about 8 values
  per clock per SM).
- Euhedral's KV codec emulates E2M1 in software (`attention/nvfp4_kv.cuh`) because its module is
  `compute_90` PTX. These `cvt` forms need an `sm_120a`/`sm_120f` target.

**Shuffles and warp reductions are 16 per clock (one warp-wide per 2 cycles).** Reductions with 5
shuffle levels per value add up quickly in multi-row kernels. Where possible, reduce through shared
memory, or arrange lanes so fewer levels are needed.

## For dequant-heavy kernels

1. **Budget about 128 thread-instructions per SM clock in total.** At 855 GB/s a 3.25-bit Q3 GEMV
   needs about 10.5 weights per clock per SM (at 2.86 GHz). That leaves roughly 12 instructions per
   weight for load, unpack, convert and every FMA across all rows. One row fits; four rows of FFMAs
   alone take a third of it.
2. **Move FMAs out of that budget.**
   - Use FFMA2 to halve FMA issue.
   - Or use tensor cores for the multiply-adds with weights in A and rows in B (see
     [tensor-cores.md](tensor-cores.md)).
   - Or move the unpack to the texture units (see [texture-units.md](texture-units.md)).
3. **Prefer IADD3/compare-based arithmetic where it replaces LOP3/SHF/PRMT,** since it runs at full
   width. Mix logic ops with FP work so both pipes are busy.
