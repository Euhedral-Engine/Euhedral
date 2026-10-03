# RT cores and the other fixed-function units

Beyond the SMs, tensor cores and texture units, GB203 carries:

| Unit | Count on the 5070 Ti |
|---|---|
| 4th-gen RT cores | 70 |
| ROPs | 96 |
| NVENC (9th gen) | 2 |
| NVDEC (6th gen) | 1 |
| Optical flow accelerator | present |
| Copy engines | 2 |
| GSP (RISC-V) | firmware processor |
| AMP (RISC-V) | firmware processor |

This document asks of each whether it can do useful work for LLM inference. The workload is
memory-bound decode GEMVs, compute-bound prefill, long-context attention and an exact speculative
verifier. Copy engines are covered in [host-link-and-runtime.md](host-link-and-runtime.md);
hardware memory compression is in [memory-system.md](memory-system.md).

**Short answer: none of these is a lever for this workload.** The texture units are the exception
([texture-units.md](texture-units.md)). The reasoning is below so the question doesn't need
re-asking.

## RT cores

**What the hardware does.**
- BVH traversal with box tests and watertight ray–triangle intersection, all in FP32.
- Blackwell adds:
  - 2× the triangle-test rate of Ada;
  - Triangle Cluster Intersection and Cluster Compression engines for Mega Geometry;
  - hardware spheres and linear swept spheres;
  - SER 2.0 thread reordering
  ([whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
  pp.17–25).
- Box and triangle tests per clock are not published.

**How compute code reaches them.**
- OptiX: headers from [NVIDIA/optix-dev](https://github.com/NVIDIA/optix-dev); `libnvoptix.so.1`
  ships with the driver and is installed on this machine. OptiX 9 needs driver R570+.
- Vulkan `VK_KHR_ray_query`, with CUDA external-memory/semaphore interop.
- Costs:
  - OptiX programs are single-threaded: no shared memory and no warp intrinsics.
  - A launch costs a few µs.
  - OptiX launches historically could not be captured into CUDA graphs.

**What has been mapped onto them.** The pattern is "points become primitives, queries become
rays": low-dimensional search and indexing.

| Work | Mapping | Result |
|---|---|---|
| [RTNN](https://arxiv.org/pdf/2201.01366) | 3-D kNN / radius search | 2.2–65× vs GPU neighbor search |
| [Arkade](https://arxiv.org/abs/2311.09168) | Non-Euclidean kNN | 1.6–200× vs shader cores |
| [JUNO](https://arxiv.org/pdf/2312.01712) | IVF-PQ ANN; each 2-D PQ subspace is a plane of sphere primitives | 2.2–8.5× vs FAISS on an RTX 4090 |
| [JUNO++](https://dl.acm.org/doi/full/10.1145/3768585) | The same retrieval picks top-k keys for sparse attention | 46% lower QKᵀ latency, "almost identical accuracy" |
| [RTIndeX](https://www.vldb.org/pvldb/vol16/p4268-schuhknecht.pdf) | Keys as triangles, lookups as rays | Competitive for read-only point lookups |
| [RTXRMQ](https://arxiv.org/abs/2306.03282) | Range minimum | ≤ 2.3× on small ranges, slower on large |
| [RTSpMSpM](https://dl.acm.org/doi/10.1145/3695053.3731072) | Sparse × sparse matmul | 1.85× vs cuSPARSE on a 4090 |
| [Graph BFS / triangle counting](https://xiaodongzhang1911.github.io/Zhang-papers/TR-25-2.pdf) | Graph algorithms | Loses to CUDA (OptiX overhead, BVH build) |

A [2026 survey](https://arxiv.org/html/2603.28771v1) lists the shared limits:
- 3-D only, FP32 only.
- About 9× memory blowup per value.
- Expensive switching between RT and CUDA execution.
- No speedup when nothing gets pruned.

**Mapping onto Euhedral:**
- **Dense GEMVs and GEMMs:** nothing to traverse.
- **Exact attention:** reads every key, so there is nothing to prune.
- **Top-k / argmax over 248K logits:** microseconds on CUDA cores.
- **Tokenization and n-gram drafting:** hash-table problems the CPU or plain CUDA handles.
- **The one LLM-adjacent result, JUNO++-style sparse attention, is approximate.** It cannot serve an
  exact verifier. It could only feed the MTP draft, and only at long context. It would need a
  per-layer index maintained as the KV cache grows, plus PQ codebooks.
- **Verdict:** do not pursue. A Quest-style per-page key summary (min/max per channel, scored by a
  tiny GEMV) gets the same pruning on CUDA cores without a second API.

## ROPs, rasterizer and depth test

**Reachable only through a graphics API** (Vulkan or OpenGL) with external-memory and semaphore
interop to CUDA.

**Classic tricks:**
- Blending as scatter-add.
- Depth test or occlusion queries for selection and k-th largest
  ([Govindaraju et al., SIGMOD 2004](http://web.cs.ucla.edu/~weiwang/paper/SIGMOD04.pdf)).
- Rasterizer coverage for interval problems.

**Recent evidence:** hardware-rasterized 3D Gaussian splatting with programmable blending beat atomics
10.4× in the backward pass, but FP32 blend targets ran at 0.27×
([arXiv 2505.18764](https://arxiv.org/abs/2505.18764)).

**Verdict: no.** For Euhedral:
- Scatter-adds (split-K reductions) are better done with L2 `red` atomics, or partials plus a reduce
  as today.
- Top-k and argmax are cheap.
- The rasterizer has no mapping.
- Cross-API synchronization would cost more than any of these save.

**Vulkan tensor access adds no throughput.** Cooperative vectors (`VK_NV_cooperative_vector`) and
cooperative matrix 2 drive the same tensor cores with fewer formats than CUDA (no NVFP4 operands).
They matter only to a Vulkan engine; llama.cpp's coopmat2 backend is the example
([llama.cpp#10206](https://github.com/ggml-org/llama.cpp/pull/10206)).

## NVENC, NVDEC, optical flow, JPEG

**Rates.** Per engine at 1080p, NVENC encodes about 1,000–1,100 fps (HEVC/AV1 P1) and NVDEC decodes
about 1,900–2,200 fps (H.264/HEVC)
([NVENC](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html),
[NVDEC](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-application-note/index.html)
application notes). That is roughly 4–6 GB/s of raw pixels.

**Video codecs as tensor compressors.**
- [LLM.265](https://arxiv.org/abs/2407.00467) (MICRO'25):
  - Llama-3-70B weights at 2.88 bits, close to 3-bit baselines.
  - KV cache at 2.9 bits, with perplexity 7.28 → 7.77.
  - Measured throughput only about 1 GB/s each way.
- [KVCodec](https://arxiv.org/abs/2602.09725) compresses 8-bit KV about 10× losslessly for network
  transfer.

**Other engines.**
- Optical flow accelerator: present on recent GeForce, no LLM mapping.
- JPEG decode engine: listed for some Blackwell parts, not advertised for GeForce. Check with
  `nvjpegGetHardwareDecoderInfo` if it ever matters.

**Verdict:** single-digit GB/s against 896 GB/s of VRAM and 43 GB/s of PCIe rules out anything
per-token. The only niche is archiving cold prefix KV when host RAM capacity, not speed, is the
limit.

## Decompression Engine

Blackwell's hardware LZ4/Snappy/Deflate engine (up to 600 GB/s, built into the copy engine and
driven through nvCOMP) exists only on B200, B300, GB200 and GB300
([nvCOMP FAQ](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html)). **GB203 does not
have it.** nvCOMP falls back to SM kernels. Euhedral's entropy-coding experiments
(docs/COMPRESSED_Q3.md) were right to treat decode cost as SM time.

## GSP and AMP

**Neither runs user code.**
- The GPU System Processor runs NVIDIA's signed RISC-V firmware.
- The AI Management Processor is a RISC-V context scheduler in front of the graphics pipeline. It
  works with Windows hardware-accelerated GPU scheduling and has no developer API.
