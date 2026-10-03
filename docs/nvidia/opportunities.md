# Opportunities for Euhedral on the RTX 5070 Ti

This ranks what the research found by expected value for Euhedral's real workload: long prompts,
16K–64K context, MTP speculative decode with an exact verifier, and a 16 GB card. Every item lists
its evidence, where it lands in the code, and the cheapest first test, following the
screen-then-gate method: cheap synthetic screens first, full paired in-model gates only for finalists.

**How to read the gains.** They are upper bounds derived from measured rates and the existing docs,
not results. Nothing here has been gated in the model yet.

## Tier 1: largest expected effect

### 1. Long-context decode and verify attention at DRAM speed

**Status (2026-10-03): done.** The cause was not bytes in flight but one latency-bound warp per
scheduler; a three-warp CTA with the same arithmetic (bitwise identical) took 16K one-row attention from 156 to 38 µs
per layer, decode in the model +11.4% at 16K and +21.3% at 32K, and MTP2 decode +19% at 16K and +38% at 32K. Fusing the verifier rows, merging inside the decode kernel and sharing the query rotation were bounded with skip toggles and are not worth building (3%, 0.2%, 2.5%). See
[../ATTENTION_DECODE.md](../ATTENTION_DECODE.md).

**Evidence**
- At 16K, decode attention takes 2.83 ms per token (Q3, one row; docs/MTP_VERIFIER.md "Final
  profile at 16K"). It reads 288 MiB of NVFP4 KV (18 KiB per token across 16 layers). At the
  measured 855 GB/s that read takes 0.35 ms, so **attention runs about 8× below DRAM speed.**
- **Verify:** 6.09 ms (Q3, 3 rows) / 8.22 ms (NVFP4, 4 rows) at 16K. At 32K, about 10 ms per
  verification, 24–30% of the verify step.
- **Why:** the GQA decode kernel launches at most 4 × 64 single-warp CTAs, about 3.7 warps per SM.
  [memory-system.md](memory-system.md) measured that DRAM needs at least 8 KB in flight per SM, and
  reaches only 725 GB/s at 4 KB.
- Compute is trivial: 2·24·16K·256·2 FLOP × 16 layers ≈ 6 GFLOP per row.

**Change**
- **Split the key range across enough CTAs** (flash-decoding style: 70 × k CTAs, each streaming a
  KV chunk with ≥ 8 KB in flight). Combine the partial softmax states in a second small kernel or in
  the last CTA.
- **For verify, read each KV chunk once for all rows** while keeping each row's arithmetic identical
  to the one-row oracle. Order the per-row partial reductions the same way, so row-exactness holds.
  This is the "fused long-context verifier attention twin" already listed as a next target.

**Upper bound**
- Decode at 16K: save about 2.5 ms of a roughly 22 ms token, so +10%. At 32K–64K the gain grows
  with the KV size.
- Verify at 32K: roughly 8 of about 38 ms per verification, about +25% tok/s.

**First test:** an operator bench of the current GQA kernel vs a split-K variant at 16K/32K/64K with
cold KV pages, bitwise against the current kernel's per-row results.

### 2. Get prefill off the half-rate BF16 tensor path

**Evidence** ([tensor-cores.md](tensor-cores.md))
- Q3/Q4/Q5 prefill uses BF16 HMMA with FP32 accumulate: a 102 TFLOPS ceiling, 57 achieved in the FFN
  at 512 rows.
- Q3 time-to-first-token is about 17.5 s at 16K and 40 s at 32K. The linears dominate:
  27B × 2 × 16K ≈ 8.8·10^14 FLOP.

**Options**

| Option | Ceiling | Numerics |
|---|---|---|
| FP16 operands, FP16 accumulate within each 64-K Q3 group, promote to FP32 with the group scale (applied per group already) | 204 TFLOPS | Overflow bound to check; drift test |
| Two-term FP8 activations through MXFP8 (`mxf8f6f4` E4M3, unit UE8M0); Q3 codes are exact in E4M3 | 204 effective | Near BF16; FP32 accumulate |
| Two-term INT8 activations (int16 split into hi/lo bytes) through IMMA | 204 effective | Exact integer accumulation per group: deterministic, order-independent |
| One-term FP8 or INT8 activations | 408 | Larger drift; relaxed mode only |

- **Upper bound:** about 2× on the GEMM part of prefill, so Q3 TTFT at 16K could drop from about
  17.5 s toward 9–10 s.
- **First test:** a synthetic GEMM at the FFN shapes, 512–2048 rows, cold weights. Run the drift test
  (`RelaxedNumericsDrift`) only for the winner.

### 3. Native NVFP4 GEMM at more than 30% of peak

**Evidence**
- The native tile reaches 225–290 TFLOPS of an 815 TFLOPS OMMA peak (docs/NVFP4_NATIVE.md).
- CUTLASS-style SM120 block-scaled GEMMs reach 83% on an RTX PRO 6000 with:
  - TMA loads;
  - a dedicated load warp and a store warp;
  - 2–4 stages;
  - `setmaxnreg` 40/232;
  - 192×128 tiles;
  - an SFA layout that avoids bank conflicts
  ([Colfax](https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/)).
- Euhedral's tile uses `cp.async` and no TMA.

**Upper bound:** about 2–2.5× on NVFP4 GEMM time. NVFP4 TTFT at 16K is about 9.7 s today.

**First test:** a TMA plus warp-specialized variant of the native tile in the synthetic FFN bench.
`setmaxnreg` needs the `sm_120a` target the module already has.

## Tier 2: clear but smaller, or needs a structural change

### 4. CUDA graphs for decode and verify quanta

**Evidence**
- 772 launches per decode token. GPU busy is 16.85 of 18.76 ms at 1K (about 10% gaps). MTP steps
  have 0.8–1.2 ms of host gaps.
- Measured: a graph node costs 0.30 µs with PDL edges (0.39 µs plain), against 0.78 µs per stream
  launch.
- PDL is off for quanta at position ≥ 1024 today; a graph keeps PDL edges at any length.

**Upper bound:** about 10% at short context, a few percent at 16K+.

**Cost:** structural. Positions and lengths must come from device memory, or from per-token
`cuGraphExecKernelNodeSetParams`. A captured quantum becomes one frame for the lattice.

**Beyond:** a device-side `WHILE` loop costs 3.2 µs per iteration. That allows multi-token decode
with on-device greedy selection and no host boundary per token.

### 5. Verify GEMVs (2–4 rows) above 600 GB/s

**Evidence**
- One-row GEMVs reach 720–800 GB/s; the 3–4-row twins about 600 GB/s.
- There is a 223-register cliff at 4 rows. Linears are 58% of verify busy time.
- [sm-pipes.md](sm-pipes.md): the issue budget at DRAM speed is about 12 instructions per weight,
  and 4 rows of FFMAs use a third of it.

**Options, in order of risk**
1. **FFMA2** (`fma.rn.f32x2`). It halves FMA issue and keeps per-element rounding, so it stays
   bitwise if the operation order is unchanged. It needs an `sm_120f` build (item 7).
2. **Tensor-core GEMV:** weights as A (16 rows per MMA), rows as B (n = 8). Dequant happens once per
   weight for any M ≤ 8; FMAs cost nothing.
   - It is row-exact by construction, provided the one-row oracle uses the same MMAs. That changes
     the oracle, so it's a contract change.
   - FP32-accumulating MMAs truncate where FFMA chains round, so results differ from today's.
3. **Texture-decoded BC4/BC5 weights** ([texture-units.md](texture-units.md)). The decode costs no
   ALU, LSU or register work; a GEMV reached 756 GB/s in the first try. It is a new lossy 4-bit
   format: 23% more bytes than Q3, plus a full NLL gate.

**First test:** synthetic `decode_rows<M>` variants (M = 1–4) with cold weights, bitwise against
repeated M = 1.

### 6. Display on the iGPU, memory overclock, IOMMU passthrough

These are system changes; see [system-tuning.md](system-tuning.md). They need your action or go-ahead.

| Change | Expected effect |
|---|---|
| Desktop on the UHD 770 | 0.5–0.7 GiB of VRAM back. At 32K–64K NVFP4 each 128 MiB of host-backed weights costs 2–2.5%, so this is worth roughly 8–12% there. It also removes the 13% slow mode and the watchdog. |
| GDDR7 memory offset | DRAM-bound decode scales roughly with the bandwidth gained. Validate each step with `nvbench mem bw` (Error Detection and Replay hides instability as lost bandwidth) and with the NLL check. |
| `iommu=pt` | May lift H2D past 43.5 GB/s, toward the 57 GB/s D2H. That speeds the NVFP4 host weight stream, which bounds long-context quanta. |

### 7. Build the hot modules for `sm_120f` (keep `compute_90` as the fallback)

Everything except `nvfp4_native` is `compute_90` PTX today ([host-link-and-runtime.md](host-link-and-runtime.md)).
An `sm_120f` build unlocks:
- 256-bit loads: half the load instructions per byte.
- FFMA2: half the FMA instructions.
- Hardware FP4/FP8 conversion: the KV codec emulates E2M1 in software today.
- Block-scaled MMA in any module, which items 2 and 5 need.

**First test:** build both, prove neutrality with `ptx_compare.py` and the bitwise tests, then try
each instruction where the synthetic benches say the kernel is issue-bound.

### 8. L2 eviction priorities

**Evidence:** [memory-system.md](memory-system.md) measured that `createpolicy` `evict_first` on a
stream with `evict_last` on a hot set keeps the hot set at 2–3 TB/s through 1 GiB of streaming,
against 0.82–0.91 TB/s without hints.

**Change:** add `ld.global.L2::cache_hint` with an `evict_first` policy to the weight loads in the
decode and verify GEMVs. Optionally mark activations, residuals, norm vectors and per-quantum KV
pages `evict_last`.

**Upper bound:** whatever non-weight traffic misses L2 today. Measure that first with Nsight Compute
on a decode step. This one is cheap to try.

## Tier 3: niche, conditional, or later

- **Stream priorities** (supported, unused): a high-priority decode lane once concurrent requests
  share the GPU.
- **Green contexts** (8-SM partitions): run prefill for one request beside decode for another
  without the decode stream losing SMs. Decode itself is a chain, so splitting one sequence buys
  nothing.
- **Copy-engine D2D** (`cuMemcpyBatchAsync` with `PREFER_OVERLAP_WITH_COMPUTE`, 105–134 GB/s, no SMs):
  KV page compaction or defragmentation behind compute.
- **FP8 or FP4 prefill attention** (MXFP8 bypass, SageAttention3-style NVFP4): only once prefill
  attention, not the linears, dominates TTFT. That happens at very long prompts.
- **2:4 sparse FP4** (1621 TFLOPS, about 3–3.5 bits per weight): needs pruning with retraining-grade
  quality recovery.
- **BC5 holding gate and up together:** one texture fetch serves both SwiGLU inputs. It only makes
  sense if item 5's texture path wins.

## Investigated and rejected

| Idea | Evidence | Where |
|---|---|---|
| RT cores (kNN, ANN, indexing) | No search problem in exact inference. The only LLM result (JUNO++) is approximate attention; OptiX plumbing costs more than any gain | [rt-cores-and-fixed-function.md](rt-cores-and-fixed-function.md) |
| ROPs, rasterizer, depth test | Cross-API; reductions and top-k are cheap in CUDA | same |
| NVENC/NVDEC/OFA/JPEG as tensor codecs | 1–6 GB/s against 896 GB/s VRAM | same |
| Hardware Decompression Engine | Not present on GB203 | same |
| Compressible memory for weights, KV, activations | Measured: only zero/constant data compresses (5.9× zeros, 1.0× BF16/FP32/4-bit) | [memory-system.md](memory-system.md) |
| TEX as a second load pipe | Measured: LDG + TEX mixed is slower than either alone (shared L1TEX) | [texture-units.md](texture-units.md) |
| Clusters/DSMEM to share weight tiles | Measured: DSMEM 0.71 TB/s aggregate (27× below local shared memory); no TMA multicast on GeForce | [memory-system.md](memory-system.md) |
| INT4 or binary MMA | Emulated on sm_120: 42 TFLOPS, about 400-cycle latency | [tensor-cores.md](tensor-cores.md) |
| FP6 for speed | Runs at the FP8 rate; bytes only | same |
| FP16/BF16 CUDA-core math, packed SFU ops | Same element rate as FP32 | [sm-pipes.md](sm-pipes.md) |
| Second copy stream, batch API or zero-copy to raise H2D | All stop at about 43 GB/s; zero-copy beside DMA is slower | [host-link-and-runtime.md](host-link-and-runtime.md) |
| Hardware texture filtering as an activation LUT | 1/256 interpolation weights | [texture-units.md](texture-units.md) |
