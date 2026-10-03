# RTX 5070 Ti (Blackwell GB203): getting everything out of the card

This folder researches how to get the most performance out of the GPU that runs Euhedral. It covers
every unit on the chip, conventional and not: tensor cores, CUDA cores, SFUs, texture units, RT
cores, ROPs, media engines, copy engines and the memory system. Wherever possible each claim was
measured on this card instead of taken from a datasheet.

- **When:** measured 2026-10-02.
- **Hardware:** RTX 5070 Ti (sm_120, 70 SMs, 16 GB GDDR7, PCIe 5.0 x16).
- **Software:** driver 615.71, CUDA UMD 13.4, NVRTC 13.1.
- **Raw logs:** [microbench/RESULTS.md](microbench/RESULTS.md). The harness that produced them is in
  [microbench/](microbench/README.md) (C + NVRTC, built with `zig cc`).
- **Literature claims** carry their source link. A claim that was not verified on the device says so.

## Documents

| Document | Contents |
|---|---|
| [hardware.md](hardware.md) | Unit inventory, measured clocks and power, per-SM resources, what GeForce Blackwell lacks compared with B200 |
| [tensor-cores.md](tensor-cores.md) | Every `mma.sync` kind measured; the GeForce FP32-accumulate halving and the block-scaled bypass; accumulator precision; sparsity; uses for decode, prefill and attention |
| [sm-pipes.md](sm-pipes.md) | CUDA cores, SFUs, conversion instructions, which pipes overlap, packed FP32 |
| [memory-system.md](memory-system.md) | DRAM and bytes in flight, 256-bit loads, TMA bulk copies, L2 bandwidth, latency, residency control and hardware compression, DSMEM |
| [texture-units.md](texture-units.md) | Texture units as a hardware weight dequantizer (BC4/BC5) running a GEMV at DRAM speed; the measured decode palette; TEX as a second load pipe (refuted) |
| [rt-cores-and-fixed-function.md](rt-cores-and-fixed-function.md) | RT cores, ROPs and rasterizer, NVENC/NVDEC/OFA/JPEG, the absent decompression engine, GSP/AMP |
| [host-link-and-runtime.md](host-link-and-runtime.md) | PCIe, copy engines and the IOMMU; CUDA graphs, PDL, device-side loops and green contexts; compile targets |
| [system-tuning.md](system-tuning.md) | Clocks, power, memory overclocking, moving the desktop to the iGPU, the display watchdog |
| [opportunities.md](opportunities.md) | Ranked experiments for Euhedral, each mapped to code and a first cheap test |

## Headline findings

1. **On this GeForce card, FP32 accumulation runs at half rate, but block-scaled MMA is exempt.**
   - FP16 and BF16 with FP32 accumulate reach 102 TFLOPS; FP16 accumulate reaches 204. Euhedral's
     prefill uses the 102 path.
   - Plain FP8 with FP32 accumulate reaches 205 TFLOPS. The same FP8 operands through
     `kind::mxf8f6f4.block_scale` with unit scales reach **408 TFLOPS with FP32 accumulation**.
   - The block-scaled path keeps the same 24-bit, round-toward-zero accumulator in our precision probe.
2. **FP4 runs at 815 TFLOPS dense and 1621 with 2:4 sparsity.** INT8 runs at 408 TOPS with exact
   integer accumulation. INT4 and 1-bit MMA are emulated: 10× slower, with about 400-cycle latency.
3. **DRAM streams at 855 GB/s (95% of spec) once at least 8 KB per SM is in flight.**
   - A TMA bulk-copy ring reaches 855 GB/s from one issuing thread per CTA.
   - L2 delivers 3.55 TB/s across all 48 MiB.
   - L2 hit latency rises from 356 to about 700 cycles as the footprint grows. DRAM is about 770 cycles.
4. **L2 residency control works on GeForce.** `createpolicy` eviction hints, or the 30 MiB persisting
   window, keep a hot set at 2–3 TB/s through a 1 GiB streaming pass. Without them it falls to
   0.82–0.91 TB/s.
5. **The texture units can dequantize weights at DRAM speed.**
   - BC4 (4 bits per weight) via `tld4` decodes 16 values per clock per SM.
   - A BC4 GEMV streams at 614–756 GB/s with no ALU dequant work.
   - The hardware palette is not the D3D one. It was measured exactly and is documented.
6. **Hardware memory compression is enabled.** It reads zeros 5.9× faster, but does nothing for
   quantized weights, BF16 or FP32.
7. **Host link limits.**
   - From a THP arena: H2D 43.5 GB/s, D2H 57 GB/s.
   - Pinned memory on 4 KiB pages manages only 25 GB/s, because the IOMMU translates.
   - A second stream, the batch-copy API and zero-copy reads add nothing to H2D.
8. **Launch costs.** A stream launch costs 0.78 µs, a graph node 0.30–0.39 µs, and an iteration of an
   on-device `WHILE` loop 3.2 µs.
9. **Not worth it for this workload:**
   - RT cores, ROPs and the rasterizer, NVENC/NVDEC.
   - TEX as an extra load pipe.
   - DSMEM, which manages only 0.7 TB/s aggregate.
   - INT4 MMA.
   - Compressible memory for weights.
10. **The biggest no-code lever is to drive the desktop from the i9-14900K's UHD 770 iGPU**, which
    is disabled today. That frees 0.5–0.7 GiB of VRAM (each 128 MiB of host-backed NVFP4 costs
    2–2.5%), removes the 5–15% desktop contention, and turns off the kernel watchdog.

## Where Euhedral stands

These figures come from the existing docs (FRAME_MODEL.md, MTP_VERIFIER.md, NVFP4_NATIVE.md).

**Decode**
- The one-row decode GEMVs already stream at 720–800 GB/s. M=1 decode is DRAM-bound.
- The 4-row verify GEMVs run at about 600 GB/s.
- About 10% of a short-context decode token is host gaps (772 launches, no CUDA graphs).

**Prefill**
- Q3 prefill uses the half-rate BF16 tensor path.
- Native NVFP4 reaches about 30% of FP4 peak.

**Gaps**
- Long-context verify attention is 24–30% of verify time.
- The host weight stream bounds long-context NVFP4.

The ranked list in [opportunities.md](opportunities.md) starts from these gaps.

| # | Opportunity | Evidence | Expected effect |
|---|---|---|---|
| 1 | Display on the iGPU | 470–705 MiB desktop VRAM, two performance modes 13% apart | +0.5–0.7 GiB headroom; removes the slow mode |
| 2 | Long-context decode/verify attention at DRAM speed | Needs ≥ 8 KB in flight per SM; the GQA kernel has about 3.7 warps per SM | Verify attention is 24–30% of verify time at 16–32K |
| 3 | Prefill off the half-rate BF16 path | 102 vs 204 (FP16 accumulate) vs 408 (MXFP8 / INT8) TFLOPS | Up to 2× on the Q3 GEMM ceiling |
| 4 | Native NVFP4 GEMM pipelining | 30% of 815 TFLOPS today; CUTLASS-style SM120 GEMMs reach 83% elsewhere | Up to about 2.5× on NVFP4 GEMM time |
| 5 | CUDA graphs (plus PDL edges) for decode/verify quanta | 0.30 µs per graph node vs 0.78 µs per launch; 90% GPU busy | Up to about 10% at short context, less when long |
| 6 | Memory overclock | Decode is DRAM-bound | Roughly proportional to the bandwidth gained |
| 7 | L2 eviction hints on weight streams | 3× hot-set retention measured | Bounded by today's non-weight L2 misses |
| 8 | Tensor-core or texture-decoded verify GEMVs | Verify GEMVs are ALU/register-bound at about 600 GB/s | Toward 800 GB/s for M=2–4 |
