#pragma once
#include "formats.cuh"
// Balanced 128-row tile engine. Every warp loads a quarter of the A tile (32 rows x K32) and stages one
// N16 B tile per K32 generation into registers, so activation loads and weight dequantization are
// spread evenly instead of falling on two producer warps; the stages land in the other shared slot
// while the current one feeds the MMAs, with one CTA barrier per generation. Shared tiles use the
// k32_probe::b_index chunk swizzle. Same BF16 hi weights and K16 MMA order as the relaxed kernels, so
// results match them bit for bit. Requires in_features % 32 == 0 and 16-byte aligned activation rows.
namespace balanced {
using namespace k32_probe;
struct alignas(32) Storage { __nv_bfloat16 a[2][128 * 32]; __nv_bfloat16 b[2][64 * 32]; };
// K range [start, start + count) of rows of stride `width`; with `partial`, FP32 sums are written there
// (split-K) instead of rounded BF16 outputs to y.
template<class B>
static __device__ __forceinline__ void run(const unsigned short* x, const unsigned char* w, unsigned short* y,
        unsigned int rows, unsigned int width, unsigned int outputs, unsigned long long scale, Storage& s,
        unsigned int start = 0u, unsigned int count = 0xffffffffu, float* partial = nullptr) {
    if (count == 0xffffffffu) count = width;
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, mb = warp >> 1, nb = warp & 1u;
    const unsigned int tiles = (outputs + 63u) / 64u, row0 = (blockIdx.x / tiles) * 128u, col0 = (blockIdx.x % tiles) * 64u;
    const typename B::Layout layout = B::layout(w, width, outputs, scale);
    float acc[4][4][4] = {};
    uint4 av[4];
    typename B::Compact bv;
    const unsigned int generations = count / 32u;
    auto load = [&](unsigned int gen) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned int r = warp * 32u + (lane >> 2) + 8u * i, row = row0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (row < rows) v = *reinterpret_cast<const uint4*>(x + (unsigned long long)row * width + start + gen * 32u + (lane & 3u) * 8u);
            unsigned int f[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) { unsigned int lo = f[j] & 65535u, hi = f[j] >> 16;
                lo = (lo & 32767u) > 32640u ? 32767u : lo; hi = (hi & 32767u) > 32640u ? 32767u : hi; f[j] = lo | (hi << 16); }
            av[i] = make_uint4(f[0], f[1], f[2], f[3]);
        }
        B::prefetch(bv, layout, outputs, col0 + warp * 16u, start + gen * 32u, lane);
    };
    auto store = [&](unsigned int slot) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned int r = warp * 32u + (lane >> 2) + 8u * i;
            *reinterpret_cast<uint4*>(s.a[slot] + b_index(r, (lane & 3u) * 8u)) = av[i];
        }
        B::template stage<1>(s.b[slot] + warp * 16u * 32u, nullptr, bv, lane);
    };
    if (generations) { load(0); store(0); }
    __syncthreads();
    for (unsigned int gen = 0; gen < generations; ++gen) {
        const unsigned int slot = gen & 1u;
        if (gen + 1u < generations) load(gen + 1u);
        #pragma unroll
        for (unsigned int half = 0; half < 2; ++half) {
            unsigned int af[4][4], bf[2][4];
            #pragma unroll
            for (int m = 0; m < 4; ++m) q3::ldmatrix_x4(af[m], s.a[slot] + b_index(mb * 64u + m * 16u + (lane & 15u), half * 16u + (lane >> 4) * 8u));
            #pragma unroll
            for (int t = 0; t < 2; ++t) q3::ldmatrix_x4(bf[t], s.b[slot] + b_index(nb * 32u + t * 16u + (lane & 7u) + ((lane >> 4) << 3), half * 16u + ((lane >> 3) & 1u) * 8u));
            #pragma unroll
            for (int m = 0; m < 4; ++m)
                #pragma unroll
                for (int h = 0; h < 4; ++h) q3::mma_16816(acc[m][h], af[m], bf[h / 2][(h & 1) * 2], bf[h / 2][(h & 1) * 2 + 1]);
        }
        if (gen + 1u < generations) store(slot ^ 1u);
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < 4; ++m)
        #pragma unroll
        for (int h = 0; h < 4; ++h)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int row = row0 + mb * 64u + m * 16u + lane / 4u + 8u * (i >> 1);
                const unsigned int col = col0 + nb * 32u + h * 8u + 2u * (lane % 4u) + (i & 1u);
                if (row < rows && col < outputs) {
                    if (partial) partial[(unsigned long long)row * outputs + col] = acc[m][h][i];
                    else q3::write_bf16(y, row, col, outputs, acc[m][h][i]);
                }
            }
}
}
namespace balanced {
// Paired gate/up + SwiGLU: the CTA owns 128 rows x 32 SwiGLU columns (32 gate + 32 up weight rows).
// Warp w stages B tile w (0, 1: gate columns; 2, 3: up columns); warp (mb, fh) multiplies rows
// mb * 64.. and the fh-th 16 columns of both gate and up, so each thread pairs its own outputs.
template<class B>
static __device__ __forceinline__ void run_paired(const unsigned short* x, const unsigned char* w, unsigned short* y,
        unsigned int rows, unsigned int width, unsigned int outputs, unsigned long long scale, Storage& s,
        unsigned int start, unsigned int count) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, mb = warp >> 1, fh = warp & 1u;
    const unsigned int tiles = count / 32u, row0 = (blockIdx.x / tiles) * 128u, col0 = start + (blockIdx.x % tiles) * 32u;
    const typename B::Layout layout = B::layout(w, width, outputs, scale);
    float acc[4][2][2][4] = {};
    uint4 av[4];
    typename B::Compact bv;
    const unsigned int generations = width / 32u;
    const unsigned int bcol = col0 + (warp & 1u) * 16u + (warp >> 1) * (outputs / 2u);
    auto load = [&](unsigned int gen) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned int r = warp * 32u + (lane >> 2) + 8u * i, row = row0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (row < rows) v = *reinterpret_cast<const uint4*>(x + (unsigned long long)row * width + gen * 32u + (lane & 3u) * 8u);
            unsigned int f[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) { unsigned int lo = f[j] & 65535u, hi = f[j] >> 16;
                lo = (lo & 32767u) > 32640u ? 32767u : lo; hi = (hi & 32767u) > 32640u ? 32767u : hi; f[j] = lo | (hi << 16); }
            av[i] = make_uint4(f[0], f[1], f[2], f[3]);
        }
        B::prefetch(bv, layout, outputs, bcol, gen * 32u, lane);
    };
    auto store = [&](unsigned int slot) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned int r = warp * 32u + (lane >> 2) + 8u * i;
            *reinterpret_cast<uint4*>(s.a[slot] + b_index(r, (lane & 3u) * 8u)) = av[i];
        }
        B::template stage<1>(s.b[slot] + warp * 16u * 32u, nullptr, bv, lane);
    };
    if (generations) { load(0); store(0); }
    __syncthreads();
    for (unsigned int gen = 0; gen < generations; ++gen) {
        const unsigned int slot = gen & 1u;
        if (gen + 1u < generations) load(gen + 1u);
        #pragma unroll
        for (unsigned int half = 0; half < 2; ++half) {
            unsigned int af[4][4], bf[2][4];
            #pragma unroll
            for (int m = 0; m < 4; ++m) q3::ldmatrix_x4(af[m], s.a[slot] + b_index(mb * 64u + m * 16u + (lane & 15u), half * 16u + (lane >> 4) * 8u));
            #pragma unroll
            for (int pair = 0; pair < 2; ++pair) q3::ldmatrix_x4(bf[pair], s.b[slot] + b_index((pair * 2u + fh) * 16u + (lane & 7u) + ((lane >> 4) << 3), half * 16u + ((lane >> 3) & 1u) * 8u));
            #pragma unroll
            for (int m = 0; m < 4; ++m)
                #pragma unroll
                for (int pair = 0; pair < 2; ++pair)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) q3::mma_16816(acc[m][pair][h], af[m], bf[pair][h * 2], bf[pair][h * 2 + 1]);
        }
        if (gen + 1u < generations) store(slot ^ 1u);
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < 4; ++m)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int row = row0 + mb * 64u + m * 16u + lane / 4u + 8u * (i >> 1);
                const unsigned int col = col0 + fh * 16u + h * 8u + 2u * (lane % 4u) + (i & 1u);
                if (row < rows && col < outputs / 2u) {
                    float gate = q3::bf16_to_float(q3::float_to_bf16(acc[m][0][h][i])), up = q3::bf16_to_float(q3::float_to_bf16(acc[m][1][h][i]));
                    y[(unsigned long long)row * count + col - start] = __bfloat16_as_ushort(__float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up));
                }
            }
}
}
