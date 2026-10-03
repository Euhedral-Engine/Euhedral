# Texture units as a weight decoder

The 5070 Ti has 280 texture units, 4 per SM. Compute kernels normally ignore them. They do three
things a dequantizing GEMV wants: fetch, decode block-compressed formats, and convert to float, all
outside the ALUs.

This document measures how far that goes. Logs: [microbench/RESULTS.md](microbench/RESULTS.md)
(`nvbench tex`). Kernels: [microbench/kernels/tex.cu](microbench/kernels/tex.cu).

## 1. TEX as a second load pipe: refuted for bandwidth

**Test:** 32-bit and 128-bit words through `ld.global.nc`, through `tex1Dfetch` on a linear-memory
texture, and half through each in the same thread.

| Words | Path | 16 MiB, reused (L1/L2) | 1 GiB, DRAM |
|---|---|---|---|
| 32-bit | LDG | 7.41 words/clk/SM | 853 GB/s |
| 32-bit | TEX (`TLD.LZ`) | 7.00 | 853 |
| 32-bit | half and half | 6.53 | 854 |
| 128-bit | LDG | 1.54 | 855 |
| 128-bit | TEX | 1.59 | 855 |
| 128-bit | half and half | 1.48 | 855 |

- **Mixing never beats either path alone.** LDG and TEX have separate issue queues (`lg_throttle` vs
  `tex_throttle` in the
  [Nsight Compute guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)), but they
  share the L1TEX data path, which is the limit here.
- **When TEX could still help:** a kernel bound on MIO *instruction issue*, where LDS, SHFL and LDG
  all queue together, could move loads to TEX to relieve that queue. Check the `mio_throttle` stall
  reason in Nsight Compute before trying.

## 2. Hardware conversion

An 8-bit unsigned-normalized linear texture returns 4 floats per fetch: 4.95 fetches, so 19.8
converted values, per clock per SM from cache, and DRAM speed (851 GB/s) when cold. Normalized reads
exist only for 8- and 16-bit integer formats, so this is a Q8/KV8 trick and not useful below 8 bits.

## 3. Block-compressed formats: hardware decode at DRAM speed

**Availability.** CUDA (11.5+) accepts BC1–BC7 CUDA arrays: `CU_AD_FORMAT_BC4_UNORM`, `_SNORM`,
`BC5`, `BC6H` and `BC7`.
- Extents are multiples of 4.
- Upload with `cuMemcpy2D`, where the pitch is the bytes in one row of 4×4 blocks and the height is
  the number of block rows.
- **Usable as weight formats:** BC4 and BC5. Each 4×4 block has two 8-bit endpoints and sixteen
  3-bit indices per channel: 4.0 bits per value, 8 levels per 16-value block.
- **Not usable:** BC1, BC6H and BC7 share one index across channels, so they can't hold independent
  weights.

### Throughput, measured

Values per clock per SM.

| Format (block bytes) | Fetch | Reused data | DRAM-cold | Cold compressed GB/s |
|---|---|---|---|---|
| BC4 (8 B) | point, 1 value | 4.77 | 4.26 | 426 |
| BC4 (8 B) | **gather `tld4`, 4 values** | **16.1** | **8.0** | **804, DRAM-bound** |
| BC1 (8 B) | point, RGBA | 19.0 | 17.1 | 424 |
| BC5 (16 B) | point, RG | 4.8 | 4.5 | 444 |
| BC5 (16 B) | gather, 4 values | 8.9 | 4.3 | 831 |
| BC7 (16 B) | point, RGBA | 9.5 | 9.2 | 445 |
| BC6H (16 B) | gather | 9.4 | 4.2 | 796 |

- **Texel rates:** 8-byte-block formats sample about 4.75 texels per clock per SM; 16-byte formats
  half that.
- **Gather (`TLD4.R`)** returns one channel of a 2×2 quad per instruction.
- **Speed:** the texture units can decode BC4 at 3.2 T values per second from cache. That is twice
  what DRAM can feed: 855 GB/s at 4 bits per value is 1.7 T values per second.
- **Gather order is confirmed:** w = (x0,y0), z = (x1,y0), x = (x0,y1), y = (x1,y1).

### Precision: not the D3D formula, but exact and reproducible

Decoding random blocks against the exact D3D10 interpolation gives a maximum error of 6.35 LSB of
1/255 for UNORM. The decoded values are not on an 8-bit grid.

The palette test (`nvbench tex precision`) shows the hardware computes

```
value = r0 + (r1 - r0) * W[index]
```

with a fixed weight table in 1/257 units:

| Mode | W for indices 2..7 | Exact D3D10 would be |
|---|---|---|
| r0 > r1 (8 levels) | 36, 72, 113, 144, 185, 221 (/257) | 1/7 .. 6/7 = 36.7, 73.4, 110.1, 146.9, 183.6, 220.3 |
| r0 ≤ r1 (6 levels + 0, 1) | 48, 96, 161, 209 (/257), then 0 and 1 | 1/5 .. 4/5 = 51.4, 102.8, 154.2, 205.6 |

- **Source of the mismatch:** NVIDIA expands endpoints to 16 bits and interpolates with an
  approximate reciprocal, ×36 instead of ×36.71
  ([fgiesen, "GPU BCn decoding"](https://fgiesen.wordpress.com/2021/10/04/gpu-bcn-decoding/)).
- **SNORM:** same structure; 8-level values ±{0.72094, 0.44188, 0.11625}.
- **For a quantizer:** the eight levels sit at a fixed, slightly non-uniform position grid between
  per-block endpoints. An encoder that targets this measured grid instead of the D3D one is exact
  with respect to what the GPU decodes.

### A GEMV with the texture unit as the dequantizer

**Kernel:** `bc4_gemv2` (cold weights, rotating copies past L2).
- The weight matrix is a BC4 SNORM texture, K wide and N tall.
- Each warp walks two output rows. Each lane issues one `tld4` per 2×2 quad: two weights from each
  of the two rows.
- The kernel does 4 FMAs per gather and one per-row scale at the end. There is no unpack code.

| K × N | µs | GB/s (4 bits/weight) |
|---|---|---|
| 5120 × 17408 (gate/up shape) | 58.9 | **756** |
| 12288 × 5120 | 48.2 | 652 |
| 17408 × 5120 (down shape) | 66.9 | 666 |
| 5120 × 5120 | 21.4 | 614 |

- **Comparison:** Euhedral's tuned one-row Q4/Q5 GEMVs stream at about 800 GB/s, and Q3 at
  720–800 GB/s (docs/FRAME_MODEL.md).
- **At M = 1 the texture path is roughly at parity,** and this first kernel is untuned: one row pair
  per warp and no split-K for the down shape.
- **Its value is what it frees.** The unpack and convert work leaves the ALUs, the LSU and the
  register file. Euhedral's multi-row verify GEMVs are bound by exactly those (about 600 GB/s, a
  223-register cliff at M = 4).
- (The `int4_alu` row in the log is a deliberately naive reference, not a fair baseline.)

### What a BC4/BC5 weight format would mean

**Format**
- 4.0 bits per weight: 3-bit indices over a 4×4 block of the matrix, which is 4 output rows × 4 K
  columns.
- Each block has 8-bit endpoints relative to a per-row (or per-tile) FP16 scale applied after the
  dot product.
- **Quality expectation:** between compact Q3 (3.25 bits, symmetric, per-64 FP16 scale) and NVFP4
  (4.5 bits). It is asymmetric with a small 16-value block, but the endpoint resolution is limited
  to 8 bits of the row range.
- **Not lossless from Q3:** the BC4 grid is non-uniform and its endpoints are 8-bit, so it has to be
  quantized from the BF16 checkpoint. Validate it with the NLL method (docs/NVFP4_COMPRESSED.md),
  never by weight MSE.

**Costs**
- 23% more bytes than Q3, so M = 1 decode would not get faster.
- Weights must live in CUDA arrays, which are opaque and tiled. Host-backed streaming would need
  `cuMemcpy2DAsync` into array slots.
- Texture objects become kernel parameters.

**Where it could win**
- **Multi-row verify GEMVs,** where today's dequant ALU work, not DRAM, is the limit.
- **BC5's two channels could hold gate and up rows together**, so one fetch serves both SwiGLU
  inputs.

**Prior art**
- Apple's 2025 foundation-model server stores weights as ASTC 6×6 textures (3.56 bits per weight)
  decoded by fixed-function hardware "at effectively zero compute cost," with low-rank adapters
  recovering quality ([arXiv 2507.13575](https://arxiv.org/pdf/2507.13575)).
- ASTC compression of CNN weights showed no accuracy loss
  ([Hot Chips 2017 poster](https://old.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.23-Posters-Pub/HC29.23.p10-Texture-Compression-Sharma-GeorgiaTech-v02.pdf)).
- No published work uses NVIDIA BCn formats for LLM weights. Desktop NVIDIA GPUs have no ASTC.

## 4. Other texture-unit ideas

**Exact lookup tables.**
- A small RG16F table texture (for example 64 entries mapping a 6-bit pair index to two FP16
  weights) stays resident in L1. It turns a lookup into one TEX instead of several LOP3/PRMT and a
  shared-memory load.
- This keeps a lossless format such as P2E2's symbol tables, but replaces ALU/MIO work with TEX
  work. That pays only if the kernel is limited by a narrow pipe (MIO or the 64/clk logic pipe),
  not by total issue.
- PTX `tex` can return `.f16x2` directly.

**Hardware filtering is not precise enough for activation lookup tables.** Filter weights are 1.8
fixed point, i.e. 1/256 steps
([NVIDIA forum](https://forums.developer.nvidia.com/t/accuracy-of-1d-linear-interpolation-by-cuda-texture-interpolation/28013)).

**Sparse (tiled) CUDA arrays are supported** (`SPARSE_CUDA_ARRAY_SUPPORTED = 1`). Texture-format
weights could be mapped page by page, like the host-backed staging slots.
