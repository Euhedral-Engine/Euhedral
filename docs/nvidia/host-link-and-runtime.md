# Host link, copy engines and the CUDA runtime

**Scope:**
- How data gets between host and device.
- How work gets onto the GPU.
- What the compile target leaves on the table.

Logs: [microbench/RESULTS.md](microbench/RESULTS.md) (`nvbench pcie`, `nvbench misc`, `nvbench attrs`).

## PCIe 5.0 x16 and the copy engines

**Device:**
- `ASYNC_ENGINE_COUNT = 2`.
- The link trains to Gen 5 under load and drops to Gen 1 at idle.

**Host:** i9-14900K. The IOMMU is in translated mode (Intel IOMMU on, no `iommu=pt` on the kernel
command line).

**1 GiB copies (best of 4):**

| Host memory | Chunk | Streams | H2D GB/s | D2H GB/s |
|---|---|---|---|---|
| `cuMemHostAlloc` (4 KiB pages) | 2 MiB – 1 GiB | 1–2 | 24.3–26.4 | 23.5–25.1 |
| THP arena (`posix_memalign` 2 MiB + `MADV_HUGEPAGE` + `MADV_COLLAPSE`, then `cuMemHostRegister`) | 2 MiB – 1 GiB | 1–2 | **43.2–43.9** | **55.6–57.1** |
| pageable `malloc` | 64 MiB – 1 GiB | 1 | 15.3–15.5 | 15.2–15.4 |

**Other transfer measurements:**

| Test | Result |
|---|---|
| Bidirectional, THP arena both ways | 54.1 GB/s total, 27 per direction |
| Zero-copy SM reads of the THP arena | 40–41 GB/s at 1–2 CTA/SM; collapses to 12.5–18.5 at 4–6 CTA/SM |
| Zero-copy kernel concurrent with a DMA H2D | 37.6–39.2 GB/s combined, below DMA alone |
| `cuMemcpyBatchAsync`, H2D, 16 × 64 MiB | 43.0 GB/s |
| `cuMemcpyDtoDAsync` | 387 GB/s (774 read+write): an SM copy kernel |
| `cuMemcpyBatchAsync` D2D with `CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE` | 105–134 GB/s on a copy engine, no SMs used |

**What this says:**

1. **The link can carry 57 GB/s; H2D stops at 43.5.**
   - H2D is the GPU *reading* host memory: non-posted requests that wait for completions through the
     IOMMU. D2H is posted writes.
   - The ceiling holds whatever does the reading:
     - a second stream;
     - chunk sizes from 2 MiB to 1 GiB;
     - the batch-copy API;
     - SM zero-copy loads, alone or alongside DMA.
   - That points at the host read path, not at engine count.
2. **4 KiB pinned pages cost 40% under the translating IOMMU.** Euhedral's single THP arena exists for
   this reason (docs/NVFP4_RESIDENCY.md "Platform").
   - **Untested lever:** boot with `intel_iommu=on iommu=pt` (passthrough). DMA would then skip
     translation, which should remove the THP dependence and may lift the 43.5 GB/s H2D ceiling. It
     needs a reboot and is a system change; see [system-tuning.md](system-tuning.md).
3. **Zero-copy needs few requests in flight.** More than 2 CTAs per SM of outstanding host reads makes
   it 3× slower. Euhedral's host-mapped embedding gathers only a few rows per token, so it is fine,
   but don't widen it.
4. **Device-to-device copies can be taken off the SMs.** Use the batch API with
   `PREFER_OVERLAP_WITH_COMPUTE`: 105–134 GB/s on a copy engine. That beats a 387 GB/s copy kernel
   only when the SMs have something better to do, for example KV-page compaction behind compute.
5. **Bidirectional traffic slows each direction.** If host-backed weights stream in while something
   streams out (KV offload, say), expect about 27 GB/s each way.

## Launch path

**2000 back-to-back tiny kernels (1 CTA × 128 threads):**

| Mechanism | Host cost | GPU time per kernel |
|---|---|---|
| `cuLaunchKernel` on a stream | 0.78 µs per launch | 0.78 µs (host-bound) |
| `cuLaunchKernelEx` with programmatic stream serialization (PDL) | 0.77 µs | 0.77 µs (host-bound) |
| Captured graph, plain edges | 3.5 µs per 2000-node graph | **0.39 µs per node** |
| Captured graph, PDL (programmatic) edges | 3.2 µs per graph | **0.30 µs per node** |
| `WHILE` conditional node, one-kernel body, condition set on the device | none | **3.23 µs per iteration** |

**Where Euhedral stands** (docs/FRAME_MODEL.md, docs/MTP_VERIFIER.md):
- About 772 launches per decode token.
- At 1K context the GPU is busy 16.85 of 18.76 ms (Q3, one row), so about 10% is host gaps and
  launch latency. MTP steps lose 0.8–1.2 ms per step (3–4%).
- **Programmatic dependent launch is applied only to quanta starting below position 1024**
  (`QwenExecutionContext`), so long-context decode runs without it.

**Implications:**
- **Graphs remove most of the launch cost.**
  - A graph node costs 0.30–0.39 µs against 0.78 µs of host submission per launch.
  - A captured decode quantum would cost one graph launch (a few µs of host time) plus about
    772 × 0.3 µs ≈ 0.23 ms of device-side launch latency.
  - The earlier estimate of "3–4% at most" (docs/MTP_VERIFIER.md) was for MTP steps; at short
    context the ceiling is about the 10% of host gaps.
  - Graph capture keeps PDL edges, so PDL could return at long context inside a graph.
- **Shape changes per token** (position, KV length) can be handled three ways:
  - device-resident counters, which kernels read instead of taking them as parameters;
  - `cuGraphExecKernelNodeSetParams` updates;
  - one graph per bucket of lengths.
- **A device-side token loop is viable.**
  - The `WHILE` node costs 3.2 µs per iteration. The body would be one decode quantum plus on-device
    greedy selection, which already exists (`sampling/`); the stop check would set the condition.
  - That removes the 100–400 µs host boundary per token at the cost of 3.2 µs.
  - Euhedral's model of frames and lanes would have to treat a captured quantum as one frame. The
    lattice still decides nothing centrally. This is a structural change, not a kernel change.

## Green contexts, priorities, sync domains

**Green contexts** (`cuDeviceGetDevResource` / `cuDevSmResourceSplitByCount` / `cuGreenCtxCreate`)
can split the 70 SMs into partitions:
- Minimum partition 8 SMs, alignment 8 (`minSmPartitionSize = 8`, `smCoscheduledAlignment = 8`).
- Uses for Euhedral:
  - keep a partition for the host-weight transfer path, or for zero-copy gathers;
  - run prefill of one request beside decode of another without the decode stream losing SMs.
- Decode itself is a chain (docs/FRAME_MODEL.md), so splitting one sequence across partitions buys
  nothing.

**Stream priorities** are supported (`STREAM_PRIORITIES_SUPPORTED = 1`) and unused; Euhedral creates
all lanes at default priority.
- A high-priority decode lane would let latency-critical decode kernels jump queued prefill or
  transfer work at CTA granularity.
- That matters once Euhedral serves concurrent requests.

**Memory sync domains:** the device reports 4. `CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN` lets kernels
whose fences don't need to order each other's traffic (for example a transfer or gather kernel
beside compute) avoid paying for each other's memory flushes.

## Compile targets

**Current setup** (docs/NVFP4_NATIVE.md, `native/src/host/cuda_kernel_loader.c`):
- Every module except `nvfp4_native` is compiled by NVRTC as `compute_90` PTX with
  `--std=c++14` and nothing else. The driver JITs it for sm_120.
- That PTX cannot use anything newer than sm_90.

**What the `compute_90` target loses:**

| Feature | Needs | Measured value on this card |
|---|---|---|
| Block-scaled `mma` (MXFP8 full-rate FP32 accumulate, MXFP4, NVFP4) | `sm_120a`/`sm_120f` | 408 / 815 TFLOPS |
| FP4/FP6 `cvt` (for example `F2FP.F16.E2M1.UNPACK_B`) | `sm_120a`/`sm_120f` | 85 FP4→FP16 values/clk/SM; the KV codec emulates it today |
| 256-bit loads `ld.global.v8.u32` (`LDG.E.ENL2.256`) | PTX 8.8, sm_100+ | Halves load instructions vs `LDG.128` |
| Packed FP32 `fma/add.rn.f32x2` | sm_100+ | Halves FMA issue slots |
| `setmaxnreg`, new `ldmatrix`/`stmatrix` shapes | `sm_120a`/`sm_120f` | |

**Recommendation:**
- Compile the hot modules twice: `sm_120f` (family-portable across 12.x) and `compute_90` as the
  fallback. Choose at load time by compute capability, as `nvfp4_native` already does.
- `sm_120a` is only required for sparse FP4 (`mma.sp` with `mxf4`/`mxf4nvf4`).
- A cubin also skips the driver JIT, which today happens once per process start.

**Other NVRTC options worth a measurement:**
- `--extra-device-vectorization`.
- `-lineinfo` for Nsight source correlation; it doesn't change code generation.
- Explicit `--maxrregcount` or `__launch_bounds__(…, minBlocks)` per kernel to steer the register
  cliffs seen in the verify GEMVs.
