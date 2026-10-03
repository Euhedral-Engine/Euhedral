// Texture-path probes.
//  * tex_lin_*: tex1Dfetch on linear memory vs plain LDG, and both at once
//    (does the TEX pipe add load issue bandwidth to an LSU-bound kernel?).
//  * bc_*: block-compressed (BC4/BC5/BC7...) 2D arrays decoded by the texture
//    units: point fetch (1 texel) and gather (tld4: 4 texels, one channel).
typedef unsigned long long u64;
__device__ __forceinline__ u64 gtimer() {
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#define STAMP0(clk)                                     \
    if (clk && blockIdx.x == 0 && threadIdx.x == 0) {   \
        clk[0] = clock64();                             \
        clk[2] = gtimer();                              \
    }
#define STAMP1(clk)                                     \
    if (clk && blockIdx.x == 0 && threadIdx.x == 0) {   \
        clk[1] = clock64();                             \
        clk[3] = gtimer();                              \
    }

// ---- linear memory: 32-bit words, 8 in flight per thread ----------------------
extern "C" __global__ void ldg_u32(const unsigned *__restrict__ p, size_t n, int reps,
                                   unsigned *out, u64 *clk) {
    STAMP0(clk);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (int r = 0; r < reps; r++)
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i + 7 * stride < n;
             i += 8 * stride) {
            unsigned v[8];
#pragma unroll
            for (int u = 0; u < 8; u++)
                asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v[u]) : "l"(p + i + u * stride));
#pragma unroll
            for (int u = 0; u < 8; u++) acc += v[u];
        }
    if (acc == 0x9e3779b9u) out[0] = acc;
    STAMP1(clk);
}

extern "C" __global__ void tex_u32(cudaTextureObject_t t, size_t n, int reps, unsigned *out,
                                   u64 *clk) {
    STAMP0(clk);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (int r = 0; r < reps; r++)
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i + 7 * stride < n;
             i += 8 * stride) {
            unsigned v[8];
#pragma unroll
            for (int u = 0; u < 8; u++) v[u] = tex1Dfetch<unsigned>(t, (int)(i + u * stride));
#pragma unroll
            for (int u = 0; u < 8; u++) acc += v[u];
        }
    if (acc == 0x9e3779b9u) out[0] = acc;
    STAMP1(clk);
}

// Half the words through LDG, half through TEX, interleaved in one thread.
extern "C" __global__ void mix_u32(const unsigned *__restrict__ p, cudaTextureObject_t t, size_t n,
                                   int reps, unsigned *out, u64 *clk) {
    STAMP0(clk);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (int r = 0; r < reps; r++)
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i + 7 * stride < n;
             i += 8 * stride) {
            unsigned v[8];
#pragma unroll
            for (int u = 0; u < 8; u += 2) {
                asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v[u]) : "l"(p + i + u * stride));
                v[u + 1] = tex1Dfetch<unsigned>(t, (int)(i + (u + 1) * stride));
            }
#pragma unroll
            for (int u = 0; u < 8; u++) acc += v[u];
        }
    if (acc == 0x9e3779b9u) out[0] = acc;
    STAMP1(clk);
}

// 16-byte words through TEX (uint4 texel) vs LDG.128.
extern "C" __global__ void tex_v4(cudaTextureObject_t t, size_t n, int reps, unsigned *out,
                                  u64 *clk) {
    STAMP0(clk);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (int r = 0; r < reps; r++)
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i + 3 * stride < n;
             i += 4 * stride) {
            uint4 v[4];
#pragma unroll
            for (int u = 0; u < 4; u++) v[u] = tex1Dfetch<uint4>(t, (int)(i + u * stride));
#pragma unroll
            for (int u = 0; u < 4; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
        }
    if (acc == 0x9e3779b9u) out[0] = acc;
    STAMP1(clk);
}

extern "C" __global__ void ldg_v4(const uint4 *__restrict__ p, size_t n, int reps, unsigned *out,
                                  u64 *clk) {
    STAMP0(clk);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (int r = 0; r < reps; r++)
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i + 3 * stride < n;
             i += 4 * stride) {
            uint4 v[4];
#pragma unroll
            for (int u = 0; u < 4; u++) v[u] = __ldg(p + i + u * stride);
#pragma unroll
            for (int u = 0; u < 4; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
        }
    if (acc == 0x9e3779b9u) out[0] = acc;
    STAMP1(clk);
}

extern "C" __global__ void mix_v4(const uint4 *__restrict__ p, cudaTextureObject_t t, size_t n,
                                  int reps, unsigned *out, u64 *clk) {
    STAMP0(clk);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (int r = 0; r < reps; r++)
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i + 3 * stride < n;
             i += 4 * stride) {
            uint4 v[4];
            v[0] = __ldg(p + i);
            v[1] = tex1Dfetch<uint4>(t, (int)(i + stride));
            v[2] = __ldg(p + i + 2 * stride);
            v[3] = tex1Dfetch<uint4>(t, (int)(i + 3 * stride));
#pragma unroll
            for (int u = 0; u < 4; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
        }
    if (acc == 0x9e3779b9u) out[0] = acc;
    STAMP1(clk);
}

// ---- 8-bit normalized: tex returns 4 floats per fetch, conversion in the TMU ----
extern "C" __global__ void tex_unorm8x4(cudaTextureObject_t t, size_t n, int reps, float *out,
                                        u64 *clk) {
    STAMP0(clk);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0;
    for (int r = 0; r < reps; r++)
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i + 3 * stride < n;
             i += 4 * stride) {
            float4 v[4];
#pragma unroll
            for (int u = 0; u < 4; u++) v[u] = tex1Dfetch<float4>(t, (int)(i + u * stride));
#pragma unroll
            for (int u = 0; u < 4; u++) acc += v[u].x + v[u].y + v[u].z + v[u].w;
        }
    if (acc == 1234.5f) out[0] = acc;
    STAMP1(clk);
}

// ---- block-compressed 2D arrays -------------------------------------------------
// Each thread walks texel (x, y) positions covering the whole texture once per rep.
// Coordinates are unnormalized texel centers; point sampling.
// FETCH: 0 = tex2D<float> (1 value), 1 = tex2D<float2>, 2 = tex2D<float4>,
//        3 = tex2Dgather<float4> comp 0 (4 values from a 2x2 quad).
template <int FETCH>
__device__ __forceinline__ float fetch(cudaTextureObject_t t, float x, float y) {
    if (FETCH == 0) return tex2D<float>(t, x, y);
    if (FETCH == 1) {
        float2 v = tex2D<float2>(t, x, y);
        return v.x + v.y;
    }
    if (FETCH == 2) {
        float4 v = tex2D<float4>(t, x, y);
        return v.x + v.y + v.z + v.w;
    }
    float4 v = tex2Dgather<float4>(t, x, y, 0);
    return v.x + v.y + v.z + v.w;
}

// Threads of a warp take consecutive x positions (step 1 texel, or 2 for gather so
// each quad is fetched once); rows are distributed across the grid.
template <int FETCH>
__device__ void bc_walk(cudaTextureObject_t t, int w, int h, int reps, float *out, u64 *clk) {
    // w is a power of two; 32-bit shift/mask addressing keeps the ALU out of the way
    STAMP0(clk);
    const int step = FETCH == 3 ? 2 : 1;
    const int row_step = FETCH == 3 ? 2 : 1;
    const float off = FETCH == 3 ? 1.0f : 0.5f;
    unsigned cols = (unsigned)(w / step);
    unsigned shift = 31 - __clz(cols);
    unsigned total = cols * (unsigned)(h / row_step);
    unsigned stride = gridDim.x * blockDim.x;
    float acc = 0;
    for (int r = 0; r < reps; r++)
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i + 3 * stride < total; i += 4 * stride) {
            float v[4];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                unsigned j = i + u * stride;
                float x = (float)((j & (cols - 1)) * step) + off;
                float y = (float)((j >> shift) * row_step) + off;
                v[u] = fetch<FETCH>(t, x, y);
            }
            acc += v[0] + v[1] + v[2] + v[3];
        }
    if (acc == 1234.5f) out[0] = acc;
    STAMP1(clk);
}
extern "C" __global__ void bc_point1(cudaTextureObject_t t, int w, int h, int reps, float *out, u64 *clk) {
    bc_walk<0>(t, w, h, reps, out, clk);
}
extern "C" __global__ void bc_point2(cudaTextureObject_t t, int w, int h, int reps, float *out, u64 *clk) {
    bc_walk<1>(t, w, h, reps, out, clk);
}
extern "C" __global__ void bc_point4(cudaTextureObject_t t, int w, int h, int reps, float *out, u64 *clk) {
    bc_walk<2>(t, w, h, reps, out, clk);
}
extern "C" __global__ void bc_gather(cudaTextureObject_t t, int w, int h, int reps, float *out, u64 *clk) {
    bc_walk<3>(t, w, h, reps, out, clk);
}

// Decode every texel of a small texture (for precision checks against a CPU decoder).
extern "C" __global__ void bc_dump(cudaTextureObject_t t, int w, int h, int chans, float *out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= w || y >= h) return;
    float4 v = tex2D<float4>(t, x + 0.5f, y + 0.5f);
    float *o = out + ((size_t)y * w + x) * chans;
    o[0] = v.x;
    if (chans > 1) o[1] = v.y;
    if (chans > 2) o[2] = v.z;
    if (chans > 3) o[3] = v.w;
}

// Same decode via gather (checks tld4 returns the identical values and quad order).
extern "C" __global__ void bc_dump_gather(cudaTextureObject_t t, int w, int h, float *out) {
    int qx = blockIdx.x * blockDim.x + threadIdx.x, qy = blockIdx.y;
    if (qx * 2 >= w || qy * 2 >= h) return;
    float4 v = tex2Dgather<float4>(t, qx * 2 + 1.0f, qy * 2 + 1.0f, 0);
    float *o = out + ((size_t)qy * (w / 2) + qx) * 4;
    o[0] = v.x;
    o[1] = v.y;
    o[2] = v.z;
    o[3] = v.w;
}

// GEMV-shaped consumer: y[row] = sum_k W[row,k] * x[k] with W stored as a BC4
// texture of width K and height N (one weight row per texture row). Each warp
// owns one row; lanes gather 2x2 quads covering (row, row+1) x (k, k+1), so one
// tld4 serves two output rows. This is the "texture unit as dequantizer" kernel.
extern "C" __global__ void bc4_gemv2(cudaTextureObject_t t, const float *__restrict__ x, int K, int N,
                                     float scale, float *y, u64 *clk) {
    STAMP0(clk);
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int row = warp * 2;
    if (row < N) {
        float a0 = 0.f, a1 = 0.f;
        for (int k = lane * 2; k < K; k += 64) {
            // gather at the shared corner of texels (k,row),(k+1,row),(k,row+1),(k+1,row+1)
            float4 q = tex2Dgather<float4>(t, k + 1.0f, row + 1.0f, 0);
            float2 xv = *reinterpret_cast<const float2 *>(x + k);
            // tld4 order: w = (x0,y0), z = (x1,y0), x = (x0,y1), y = (x1,y1)
            a0 += q.w * xv.x + q.z * xv.y;
            a1 += q.x * xv.x + q.y * xv.y;
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            a0 += __shfl_xor_sync(0xffffffffu, a0, o);
            a1 += __shfl_xor_sync(0xffffffffu, a1, o);
        }
        if (lane == 0) {
            y[row] = a0 * scale;
            if (row + 1 < N) y[row + 1] = a1 * scale;
        }
    }
    STAMP1(clk);
}

// Reference: same GEMV over a packed 4-bit-per-weight buffer decoded on the ALUs
// (uint4 = 32 nibbles per lane per step; signed int4 * per-row scale). Matches the
// bytes moved by the BC4 path (4 bits/weight).
extern "C" __global__ void int4_gemv(const uint4 *__restrict__ w, const float *__restrict__ x, int K,
                                     int N, float scale, float *y, u64 *clk) {
    STAMP0(clk);
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp < N) {
        const uint4 *row = w + (size_t)warp * (K / 32);
        float acc = 0.f;
        for (int k = lane; k < K / 32; k += 32) {
            uint4 q = __ldg(row + k);
            const float *xv = x + k * 32;
            unsigned ws[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    int v = (int)((ws[j] >> (4 * n)) & 15u) - 8;
                    acc += (float)v * xv[j * 8 + n];
                }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[warp] = acc * scale;
    }
    STAMP1(clk);
}
