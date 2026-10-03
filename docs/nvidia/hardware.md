# RTX 5070 Ti hardware inventory

**What this is:** GB203-300, compute capability 12.0 (sm_120), on TSMC 4N.

**Sources:**
- The RTX Blackwell whitepaper
  ([PDF](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf),
  pp.49–53).
- Device attributes and measurements from this card
  ([microbench/RESULTS.md](microbench/RESULTS.md)).

## Units

| Unit | 5070 Ti | Full GB203 | Notes |
|---|---|---|---|
| GPCs / TPCs / SMs | 6 / 35 / 70 | 7 / 42 / 84 | GPC sizes are unequal; clusters top out at 12 |
| CUDA cores | 8,960 (128 per SM) | 10,752 | INT32 "unified", but only adds/compares are full width ([sm-pipes.md](sm-pipes.md)) |
| Tensor cores (5th gen) | 280 (4 per SM) | 336 | `mma.sync` only; no tcgen05, TMEM or wgmma |
| RT cores (4th gen) | 70 | 84 | |
| Texture units | 280 (4 per SM) | 336 | Decode BC formats; [texture-units.md](texture-units.md) |
| ROPs | 96 | 112 | |
| L2 | 48 MiB | 64 MiB | |
| L1/shared per SM | 128 KB | | Up to 100 KiB shared per SM, 99 KiB per block |
| Register file per SM | 256 KB (64K × 32-bit) | | |
| Memory | 16 GB GDDR7, 256-bit, 28 Gbps, 896 GB/s | | 8 × 32-bit controllers |
| Host link | PCIe 5.0 x16 | | |
| NVENC / NVDEC | 2 (9th gen) / 1 (6th gen) | | |
| Copy engines | 2 (`ASYNC_ENGINE_COUNT`) | | |
| TGP | 300 W default | | This card allows 250–350 W and is set to 350 W |

## Measured limits and attributes

**Execution**
- 1536 threads (48 warps) and 24 blocks per SM.
- 64K registers per SM and per block, 255 per thread.
- 1024 threads per block.

**Memory**
- Shared memory: 48 KiB per block by default, 99 KiB opt-in, 100 KiB per SM, 1 KiB reserved per
  block.
- L2 48 MiB. The persisting-L2 maximum is **30 MiB**; the maximum access-policy window is 128 MiB.
- Generic (compute data) compression supported. Virtual memory management and memory pools
  supported. Sparse CUDA arrays supported.
- Pageable memory access through HMM (without host page tables).
- No host-native atomics, no GPUDirect RDMA, no dma_buf.

**Textures**
- 2D up to 131072 × 65536.
- 1D linear up to 2^28 texels.

**Clusters and scheduling**
- Clusters: 8 portable, 12 non-portable.
- 4 memory sync domains.
- Stream priorities supported.
- Green-context SM partitions of 8 SMs minimum, alignment 8.

**Other**
- **`KERNEL_EXEC_TIMEOUT = 1`:** the display watchdog is armed, because this GPU drives the desktop.
- ECC off: GDDR7 on-die ECC is always on and not reported as ECC.
- FP32:FP64 ratio 64.

## Clocks and power, measured

| | Value |
|---|---|
| Rated boost | 2452 MHz |
| `CLOCK_RATE` attribute | 2588 MHz |
| V/F table maximum | 3105 MHz |
| Under load | 2.80–2.88 GHz in every test (memory streams, tensor loops, ALU loops) |
| Memory clock | 14,001 MHz maximum, 13,801 MHz under load, 405 MHz idle |
| Limiting reason under load | "Reliability" (the voltage/reliability cap), not power or temperature |
| Power, register-only tensor loops | 130–232 W against the 350 W limit |
| GPU temperature in those tests | 46–55 °C |

**Measurement consequences:**
- Peak-rate tables computed at 2452 MHz understate this card by about 15%. Per-clock rates are the
  stable quantity.
- Clocks move very fast on Blackwell. The whitepaper says clock changes are "1000x faster" than
  before, with separate core and memory rails. On this desktop-shared GPU, paired A/B measurements
  are mandatory (docs/FRAME_MODEL.md records two per-JVM performance modes 13% apart).

## Peak rates at the observed clock

At about 2.86 GHz, 70 SMs:

| Rate | Per SM per clock | Card |
|---|---|---|
| FP32 FMA (CUDA cores) | 128 FMA | 25.6 TFLOP-FMA/s (51 TFLOPS) |
| FP16/BF16 FMA (CUDA cores) | 128 elements | Same as FP32 |
| INT32 add | 128 | |
| IMAD / LOP3 / SHF / PRMT | 64 | |
| SFU (`ex2`, `rsqrt`, …) | 16 | 3.2 T/s |
| FP64 FMA | 2 | 0.8 TFLOPS |
| FP16/BF16 tensor, FP32 accumulate | 512 FLOP | 102 TFLOPS |
| FP16 tensor, FP16 accumulate | 1024 FLOP | 204 TFLOPS |
| FP8 tensor, FP32 accumulate (plain) | 1024 FLOP | 205 TFLOPS |
| FP8 tensor, FP16 accumulate, or block-scaled MXFP8 | 2048 FLOP | 408 TFLOPS |
| INT8 tensor | 2048 OP | 408 TOPS |
| FP4 tensor (MXFP4/NVFP4) | 4096 FLOP | 815 TFLOPS |
| DRAM (achieved) | 4.24 B | 855 GB/s |
| L2 (achieved) | 17.8 B | 3.55 TB/s |
| Texture BC4 gather | 16 values | 3.2 T values/s |

## Compared with other parts

**vs Ada (RTX 40):**
- **Same per clock:** FP16/BF16/TF32/FP8/INT8 tensor rates per SM per clock, the 128 KB L1/shared,
  48 warps per SM, 4 texture units per SM.
- **New:**
  - FP4/FP6 tensor kinds with block scaling.
  - TMA plus mbarrier transaction counts (present since sm_90).
  - Full-rate INT32 add/compare.
  - Point-sampled texture rate doubled per SM (whitepaper p.13).
  - Hardware FP4/FP6/FP8 conversions.
  - 256-bit global loads.
  - Packed FP32 FMA.
  - GDDR7.
  - A larger L2 per tier.

**vs B200 (sm_100):**
- **Missing:**
  - tcgen05, Tensor Memory, CTA-pair MMA;
  - TMA multicast and gather4/scatter4;
  - non-portable 16-CTA clusters;
  - the hardware Decompression Engine;
  - 228 KB shared memory per SM and 64 warps per SM.
- **Kept:** FP32 accumulation at half rate except for block-scaled kinds. Data-center parts and the
  RTX PRO 6000 run it at full rate.
- **The sm_120 programming model is Ampere/Hopper-style warp MMA, plus TMA and block scaling.**
  CUTLASS's SM120 kernels are the reference
  ([CUTLASS example 79](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm)).
