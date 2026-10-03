# Prefill on block-scaled FP8 tensor cores

GeForce Blackwell runs a BF16 or FP16 MMA with FP32 accumulation at 102 TFLOPS, but the block-scaled FP8 kind
(`mma.sync kind::mxf8f6f4`) at 408 TFLOPS with FP32 accumulation (docs/nvidia/tensor-cores.md). Prefill linears run on
the second. This file describes the kernels, their measured performance, the numerics, and the variants that were
rejected.

Hardware for every measurement: RTX 5070 Ti (sm_120, 70 SMs), driver 615.71, desktop session running. Sources:
`native/src/q3_mx/kernels.cu` (kernels), `native/src/host/q3_mx.c` (host route), `CudaGpuMemory.invokeQ3Mx` (Java route).
Attention prefill is in `native/src/attention/nvfp4_prefill_fa2.cuh` (last section).

## Design

**Operands are exact.**
- Q3 (-4..3), Q4 (-8..7) and Q5 (-16..15) codes are exact in E4M3.
- A BF16 activation has eight significant bits, and two E4M3 terms hold eight: x = 2^e (hi + lo), hi = E4M3(x / 2^e), lo =
  E4M3(x / 2^e - hi), with one power-of-two block scale 2^e (UE8M0) per 32 values (`euhedral_q3mx_quantize`). Elements within
  2^-9 of their block's maximum are reproduced exactly; smaller ones keep an absolute error below 2^-17 of the block maximum.

**Scales.** The MMA applies the activation scale (scale A). Weights take unit scales; the FP16 scale of each 64-code weight
group is applied in FP32 after the group's MMAs, one FFMA per accumulator, so each group sum is a sum of exact products
accumulated in FP32. Two terms (hi, lo) make the logical rate half the MMA rate: 204 TFLOPS at most.

**Weights are expanded in the kernel.** Q3 expands by byte permute (an E4M3 table indexed by the 3-bit code). Q4 and Q5 look up
one E4M3 byte pair per nibble pair (and per pair of fifth bits for Q5) in a 256 or 1024 entry shared table.

**Engine.** 128 x 128 output tile per CTA, K in groups of 64, 384 threads:
- 8 consumer warps (64 x 32 each) only issue the MMAs and the group promotion;
- 4 producer warps stream the activation tiles (cp.async) and expand the weights into a ring of three stages, up to three groups
  ahead, handed over by mbarriers (the cp.async copies arrive on the barrier themselves);
- `setmaxnreg` moves registers from the producers to the consumers: 168 per thread at launch, 216 for consumers, 72 for producers;
- 75 KiB of shared memory, one CTA per SM.

The paired gate/up kernel orders each tile as 16 gate rows then 16 up rows per warp, so SwiGLU runs in registers on gate and up
rounded to BF16.

**Split-K.** A linear with few tiles (rows x outputs / 16384 CTAs on 70 SMs) fills its last wave poorly. The host picks a split of
1, 2 or 4 groups of K per CTA from a wave model (waves / splits, plus 4% per extra split, taken only when it is at least 8%
better). Splits write FP32 partials that `euhedral_q3mx_reduce` sums in split order and rounds to BF16. The paired gate/up
region never splits. A P2E2 tensor too large to expand at once runs in output-row chunks whose results must equal the whole
tensor's, so the host chooses the split from the whole tensor's rows (`euhedral_cuda_q3_mx_select_split_rows`).

**Routing.** From 16 rows (a smaller quantum has dedicated GEMV kernels): Q3 linears, FFN gate/up and down, Q4 and Q5 linears,
and the GDN input projections. P2E2 tensors expand into the shared scratch and the expansion reserves an activation region
behind the expanded weights. `EUHEDRAL_Q3_MX=0` disables the route; exact numerics and row-exact execution decline it.

### The scale register contract (measured)

`mma.sync.aligned.m16n8k32 ... kind::mxf8f6f4.block_scale.scale_vec::1X ... .ue8m0`, scale operand `{byte-id, thread-id}`:

| Item | Contract |
|---|---|
| Scale A | Row r < 8 from lane 4r + 2 tid, row r + 8 from lane 4r + 2 tid + 1; tid is 0 or 1 (ptxas rejects 2 and 3). The byte of the register selected by byte-id is the UE8M0 scale of that row for the whole k32 block. |
| Use here | tid = 0, byte-id = k32 block of the group (0 or 1): lanes with `lane & 3` of 0 and 1 load the two scale bytes of their row, one 16-bit load per 16-row tile and group. |
| Scale B | Unit (`0x7f7f7f7f`). |

## Performance

### Operators

Cold weights, activation quantization included, relaxed numerics. TFLOPS count 2 x rows x outputs x K.

| Operator (outputs x K) | Rows | Time | TFLOPS |
|---|---|---|---|
| FFN gate/up SwiGLU (34816 x 5120) | 256 | 742 us | 123 |
| | 512 | 1447 us | 126 |
| | 1024 | 2850 us | 128 |
| | 2048 | 5625 us | 130 |
| FFN down (5120 x 17408), split-K 4 / 2 / 1 / 1 | 256 | 438 us | 104 |
| | 512 | 854 us | 107 |
| | 1024 | 1684 us | 108 |
| | 2048 | 3619 us | 101 |
| Q3 q/k/v (12288 x 5120) | 512 | 610 us | 106 |
| | 2048 | 2214 us | 116 |
| Q5 GDN projection (12288 x 5120) | 512 | 609 us | 106 |
| | 2048 | 2149 us | 120 |
| Q4 GDN projection (6144 x 5120) | 512 | 276 us | 117 |
| | 2048 | 1031 us | 125 |

The quantize pass costs 8 us for 512 rows of width 5120 (24 us for the FFN down input, width 17408).

### Prefill, end to end

Q3 compact artifact, chunk 512, one fork, warmup 1 and 2 measured iterations. Decode tokens per second is the 128-token greedy
decode that follows the prompt.

| Prompt | Time to first token | Prefill | Decode |
|---|---|---|---|
| 1000 tokens | 0.56 s | 2015 tok/s | |
| 4K | 2.15 s | 1906 tok/s | 61.5 tok/s |
| 16K | 9.12 s | 1796 tok/s | 60.5 tok/s |
| 32K | 20.0 s | 1640 tok/s | 59.0 tok/s |

At 16K, kernel time is split gate/up 31%, Q3 linears 24%, FA2 attention 18%, Q5 12%, Q4 5%, GDN recurrence 4% and
quantization 1.3% (Nsight Systems, before the FA2 rewrite below).

### Prefill attention

`euhedral_attention_prefill_fa2_nvfp4` (FlashAttention-2 style over the NVFP4 cache, 16 query rows per CTA, one compute warp per
query head) now has two producer warps that expand each 32-key tile of K and V into a double buffer ahead of the compute warps,
handed over by mbarriers, with a lookup-table expansion. The arithmetic order is unchanged, so its output is bit-identical to the
single-role kernel kept as the test control (`reference_prefill_fa2.cuh`). One 512-row chunk with 24 query heads over 4 KV heads:

| Cache position | Time | TFLOPS (4 x rows x keys x 256 x heads) |
|---|---|---|
| 2048 | 664 us | 44 |
| 8192 | 1891 us | 56 |
| 16384 | 3529 us | 59 |
| 32768 | 6955 us | 60 |

Query-head groups above 6 use the 32-row tile kernel (the CTA needs 32 (G + 2) threads).

## Numerics

- `native/tests/test_q3_mx.py` compares the quantizer and the Q3, Q4 and Q5 linears, the split-K linear and the gate/up kernel with
  FP64 on exact products (they agree to the BF16 output rounding); the FA2 test compares the production kernel with the control bitwise.
- `RelaxedNumericsDriftCudaIntegrationTest` against the exact oracle, Q3 compact and P2E2 artifacts alike: KL 5.1e-3, hidden relative
  error 6.0e-2, top-1 agreement 0.961 over 8 windows of 48 positions, with no growth by position.
- The route is deterministic: 60 repeated launches of each kernel variant gave identical bits.
- A P2E2 artifact reproduces the compact artifact bitwise.

## Rejected

- **INT8 two-term** (activations as int16 split into two signed bytes, exact integer accumulation per group). The MXFP8 kind applies the
  activation scale in hardware, so promotion is one FFMA; INT8 needs a shift, an integer-to-float conversion, a scale multiply and an
  FFMA per accumulator, and it quantizes the activations to 16 bits where MXFP8 reproduces BF16 exactly. The screen reached 99 TFLOPS on
  gate/up before tuning.
- **Weight expansion inside the consumer warps** (interleaved between the MMAs). It reached 118 TFLOPS on gate/up; skipping the expansion
  bounds the structure at 134. Producer warps are the kept design.
- **FP16 arithmetic expansion of Q4/Q5 codes** (magic-number half2, then a hardware E4M3 conversion) and a **sign-magnitude byte-permute
  table** for Q5. Both were estimated at 3.5 to 4.25 instructions per code against about 2 for the shared pair table, and not built.
- **Fusing the quantization into the gate/up epilogue.** All quantization passes are 1.3% of a 16K prefill's kernel time; only the FFN
  down input could fuse (its producer is the gate/up kernel), and the planes would have to travel to the down call through the shared
  scratch, which another frame can overwrite between the two calls, so the down call would need a validity protocol or a plan-owned buffer.
- **Split-K for gate/up and for linears with 320 or more tiles.** The wave model shows no gain; measured splits of 2 and 4 on the down
  projection at 1024 rows were no faster than one.
- **A fixed split factor** (always 2, always 4): a factor above the wave model's choice slightly lowers the rate through the FP32 partials (the down projection at 512 rows ran 935 us at 2 and 950 us at 4).
- **FA2 with more compute warps per query head or two CTAs per SM.** Not built: each compute warp holds the 16 x 256 output accumulators
  and the query fragments (about 230 registers), so a CTA of 8 warps already fills the register file and the 67.6 KiB double buffer
  fills the shared memory. The remaining imbalance is that 6 compute warps share 4 schedulers (two schedulers carry two compute warps).
