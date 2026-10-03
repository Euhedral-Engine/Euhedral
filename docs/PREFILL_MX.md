# Prefill on block-scaled FP8 tensor cores

Item 2 of docs/nvidia/opportunities.md: get prefill off the half-rate BF16 tensor path. On GeForce Blackwell an
FP32-accumulating FP16/BF16 MMA runs at 102 TFLOPS, but the block-scaled FP8 kind (`mma.sync kind::mxf8f6f4`) runs at
408 TFLOPS with FP32 accumulation (docs/nvidia/tensor-cores.md). This log covers the design, what was kept, what was
not, and the measurements.

Setup: RTX 5070 Ti (sm_120, 70 SMs), driver 615.71, Q3 compact artifact (`qwen3_5_27b_compact_q3.edrl`), prefill
chunk 512 rows. Kernels: `native/src/q3_mx/kernels.cu`; host route `native/src/host/q3_mx.c`; Java route in
`CudaGpuMemory` (`invokeQ3Mx`).

## Where prefill time went

A 4096-token prefill (Nsight Systems, relaxed numerics, before the change), per run:

| Kernel | Share |
|---|---|
| gate/up SwiGLU (Q3) | 39% |
| FFN down (Q3, split-K) | 19% |
| GDN input projections, Q5 | 15% |
| Q3 linears (attention projections, GDN output) | 8% |
| GDN input projections, Q4 | 6% |
| FA2 attention | 4% |
| everything else (GDN recurrence, norms, convolution) | 9% |

The linears are 87% of the time and already ran at 66-69 TFLOPS, about two thirds of the BF16 peak, so only a
faster MMA kind could change the picture.

## Design

- **Weights are exact.** Q3 (-4..3), Q4 (-8..7) and Q5 (-16..15) codes are exact in E4M3.
- **Activations are exact.** A BF16 value has eight significant bits, and two E4M3 terms hold eight: x = 2^e (hi + lo),
  hi = E4M3(x / 2^e), lo = E4M3(x / 2^e - hi), one power-of-two block scale 2^e per 32 values
  (`euhedral_q3mx_quantize`). Elements within 2^-9 of their block's maximum are reproduced exactly; smaller ones
  keep an absolute error below 2^-17 of the block maximum.
- **The MMA applies the activation scale.** Scale A of `mxf8f6f4` is one UE8M0 byte per row and k32 block. Weights
  take unit scales; their FP16 group scale (per 64 codes) is applied in FP32 after the group's MMAs, one FFMA per
  accumulator. Group accumulators start at zero, so the whole group sum is exact products in FP32.
- **Two terms, so 204 TFLOPS effective** (the hi and lo MMAs of each k32 block).
- The relaxed BF16 route rounds code x scale to BF16 for every weight; the FP8 route does not. It is the more
  accurate of the two against the exact oracle.

Engine: 128 x 128 CTA tile, 8 warps of 64 x 32, 3-stage cp.async pipeline for the activation planes, two threads per
weight row expanding one group (32 codes) a group ahead, interleaved between the MMA items. Q3 expands by byte permute
(an E4M3 table indexed by the 3-bit code). Q4 and Q5 look up one E4M3 byte pair per nibble pair, and per pair of fifth
bits for Q5, in a 256 or 1024 entry shared table. The paired gate/up kernel orders each tile as 16 gate rows then 16
up rows per warp so SwiGLU runs in registers. 226 registers, no spills, one CTA per SM.

### The scale register contract (measured here)

`mma.sync.aligned.m16n8k32 ... kind::mxf8f6f4.block_scale.scale_vec::1X ... .ue8m0`, scale operand `{byte-id, thread-id}`:

| Item | Contract |
|---|---|
| Scale A | Row r < 8 from lane 4r + 2 tid, row r + 8 from lane 4r + 2 tid + 1; tid is 0 or 1 (ptxas rejects 2 and 3). The byte of the register selected by byte-id is the UE8M0 scale of that row for the whole k32 block. |
| Use here | tid = 0, byte-id = k32 block of the group (0 or 1): lanes with `lane & 3` of 0 and 1 load the two scale bytes of their row (one 16-bit load per 16-row tile and group). |
| Scale B | Unit (`0x7f7f7f7f`). |

## Measurements

### Screen (synthetic GEMM, FFN shapes, cold weights)

| Variant | gate/up 2048 rows |
|---|---|
| Production BF16 route | 69 TFLOPS |
| INT8 two-term (int16 activations as hi/lo bytes, exact integer accumulation per group, I2F promotion) | 99 |
| MXFP8 two-term, first version | 107 |
| + A fragment lookahead | 109 |
| + cp.async addresses set up once (they were recomputed with divisions every group) | 131 |
| MXFP8 with real weight expansion, not interleaved | 99 |
| MXFP8 with expansion interleaved between the MMA items (kept) | 118 |

INT8 was rejected: MXFP8 promotes with one FFMA per accumulator (the hardware applies the activation scale), INT8 needs
a shift, an I2F, a multiply and an FFMA, and MXFP8 keeps the exact BF16 activations while INT8 quantizes them to 16
bits. Skip toggles on the prototype showed the structure, not the tensor pipe, was the limit: the cp.async address
math cost 24%, weight expansion 27%, promotion 3%.

### Operator (quantize included, cold weights)

| Op | Rows | BF16 route | FP8 route | Speedup |
|---|---|---|---|---|
| gate/up SwiGLU | 512 | 2689 us | 1688 us | 1.59x |
| gate/up SwiGLU | 2048 | 10568 us | 6172 us | 1.71x |
| FFN down (K = 17408) | 512 | 1654 us | 1071 us | 1.54x |
| FFN down | 2048 | 5316 us | 3746 us | 1.42x |
| Q3 q/k/v (12288 x 5120) | 512 | 969 us | 671 us | 1.45x |
| Q5 (12288 x 5120) | 512 | 1048 us | 754 us | 1.39x |
| Q5 | 2048 | 3920 us | 2570 us | 1.53x |
| Q4 (6144 x 5120) | 512 | 523 us | 326 us | 1.60x |

### In-model: time to first token

One fork each, same build, route on against `EUHEDRAL_Q3_MX=0`, Q3 compact artifact, 2 measured iterations:

| Prompt | TTFT off | TTFT on | Change | Decode tok/s off / on |
|---|---|---|---|---|
| 4K | 3513 ms | 2584 ms | **-26.4%** | 61.51 / 61.49 |
| 16K | 15155 ms | 11307 ms | **-25.4%** | 60.53 / 60.56 |
| 32K | 33397 ms | 25742 ms | **-22.9%** | 58.98 / 58.99 |

Decode is untouched (the route needs 128 rows). Adding only Q3 (before Q4/Q5) gave -21%, -20% and -18%.

### Numerics

- Unit tests against FP64 (`native/tests/test_q3_mx.py`): the quantizer reproduces BF16 exactly within its range; Q3,
  Q4 and Q5 linears and the gate/up SwiGLU match to the BF16 output rounding.
- `RelaxedNumericsDriftCudaIntegrationTest` against the exact oracle, route on against off: KL 4.9e-3 against
  4.7e-3, hidden relative error 5.4e-2 against 5.5e-2, top-1 0.958 against 0.958. No growth with position.
- Exact numerics (`EUHEDRAL_EXACT=1`) and row-exact execution decline the route; the BF16 kernels run.

### A bug the unit tests missed

The first Q4/Q5 integration ran the Q3 kernel on Q4 weights (the kernel selection expression never matched when
patched in). It was fast, looked like a win, and failed the drift test (KL 0.34). The unit tests used the Q4/Q5
kernels directly, so only the in-model run exposed it. A dump of the real call replayed offline matched the BF16
route, which located the fault in the host launch rather than in the kernels or the data layout.

## Not done

- **FFN down tail:** at 512 rows the down projection has 160 CTAs on 70 SMs (2.3 waves) and runs at 85 TFLOPS against 108 for gate/up.
  Split-K or a 64-wide tile would help.
- **Quantization fused into the producer:** the down input is the gate/up output; quantizing in its epilogue would
  save a round trip (187 us of 3746 at 2048 rows).
- **Producer warps:** weight expansion still shares issue slots with the MMA warps (the no-expansion bound is 134 TFLOPS
  against 118).
- **P2E2 artifacts** expand into the shared scratch and decline the route.
- **FA2 prefill attention** is now 4-5% of TTFT; at 32K it grows (docs/nvidia/opportunities.md tier 3).
