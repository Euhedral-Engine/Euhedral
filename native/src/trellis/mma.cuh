// The trellis GEMM on tensor cores: FP16 activations times trellis tiles decoded in registers, FP32 accumulation.
#ifndef EUHEDRAL_TRELLIS_MMA_CUH
#define EUHEDRAL_TRELLIS_MMA_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "trellis/codebook.cuh"
#include "trellis/decode.cuh"

namespace trellis_mma {

__device__ __forceinline__ void mma_f16(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned int pack_half2(half2 value) { return *reinterpret_cast<unsigned int*>(&value); }

/// `MT` m16 row tiles starting at row `row_base` over the k16 tiles [first, last) of one 128-column block. Warp `warp`
/// owns the 16-column tile blockIdx.x * 8 + warp. The same instruction sequence runs for every m-tile, rows past
/// `rows` are zero, and k16 tiles are accumulated in increasing order, so a row's bits depend on nothing but its own
/// data and the K range.
template <int MT>
__device__ __forceinline__ void group(const unsigned short* __restrict__ act, const unsigned short* __restrict__ words,
        const unsigned short* __restrict__ svh, float* __restrict__ partials, unsigned short* __restrict__ out,
        int rows, int row_base, int k, int n, int w, int splits, int split, int first, int last,
        unsigned int (*ring)[kTrellisMaxWords], float (*tile)[128]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int grp = lane >> 2, q = lane & 3;
    const int nw = w >> 1, ka = w >> 4;
    const unsigned int mask = (w & 15) == 8 ? 0xAAAAu : 0u;
    const int tiles_n = n >> 4;
    const int tj = blockIdx.x * 8 + warp;

    // The ring window of each of this lane's eight positions is the same for every k16 tile.
    int start[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int at = frac_s(lane * 8 + j, ka, mask, w) - 16;
        if (at < 0) at += nw * 32;
        start[j] = at;
    }

    float acc[MT][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[m][h][i] = 0.0f;

    const unsigned int* source = reinterpret_cast<const unsigned int*>(words) + (size_t)tj * nw;
    const size_t stride = (size_t)tiles_n * nw;
    unsigned int next0 = 0, next1 = 0;
    if (first < last) {
        const unsigned int* from = source + (size_t)first * stride;
        next0 = lane < nw ? from[lane] : 0u;
        next1 = lane + 32 < nw ? from[lane + 32] : 0u;
    }
    unsigned int* mine = ring[warp];
    for (int ti = first; ti < last; ++ti) {
        mine[lane] = next0;
        mine[lane + 32] = next1;
        __syncwarp();
        if (ti + 1 < last) {
            const unsigned int* from = source + (size_t)(ti + 1) * stride;
            next0 = lane < nw ? from[lane] : 0u;
            next1 = lane + 32 < nw ? from[lane + 32] : 0u;
        }
        unsigned int b[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int half_index = 0; half_index < 2; ++half_index) {
                const int j = h * 4 + half_index * 2;
                b[h][half_index] = pack_half2(decode_mul1_2(frac_window(mine, nw, start[j]), frac_window(mine, nw, start[j + 1])));
            }
        __syncwarp();
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int row0 = row_base + m * 16 + grp, row1 = row0 + 8;
            const int at = ti * 16 + 2 * q;
            unsigned int a[4];
            a[0] = row0 < rows ? *reinterpret_cast<const unsigned int*>(act + (size_t)row0 * k + at) : 0u;
            a[1] = row1 < rows ? *reinterpret_cast<const unsigned int*>(act + (size_t)row1 * k + at) : 0u;
            a[2] = row0 < rows ? *reinterpret_cast<const unsigned int*>(act + (size_t)row0 * k + at + 8) : 0u;
            a[3] = row1 < rows ? *reinterpret_cast<const unsigned int*>(act + (size_t)row1 * k + at + 8) : 0u;
            mma_f16(acc[m][0], a, b[0][0], b[0][1]);
            mma_f16(acc[m][1], a, b[1][0], b[1][1]);
        }
    }

    const int column0 = blockIdx.x * 128;
    if (splits > 1) {
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int column = column0 + warp * 16 + h * 8 + 2 * q;
#pragma unroll
                for (int half_row = 0; half_row < 2; ++half_row) {
                    const int row = row_base + m * 16 + grp + half_row * 8;
                    if (row < rows)
                        *reinterpret_cast<float2*>(partials + ((size_t)split * rows + row) * n + column) =
                                make_float2(acc[m][h][2 * half_row], acc[m][h][2 * half_row + 1]);
                }
            }
        return;
    }
    // One split: the epilogue in place. Each m-tile goes through shared memory as 16 rows of 128 columns.
    const int half_block = threadIdx.x >> 7, t = threadIdx.x & 127;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int column = warp * 16 + h * 8 + 2 * q;
            tile[grp][column] = acc[m][h][0];
            tile[grp][column + 1] = acc[m][h][1];
            tile[grp + 8][column] = acc[m][h][2];
            tile[grp + 8][column + 1] = acc[m][h][3];
        }
        const float scale = __half2float(__ushort_as_half(svh[column0 + t]));
        for (int i = 0; i < 8; ++i) {
            const int r = i * 2 + half_block;
            trellis_hadamard128(tile[r], t);
            const int row = row_base + m * 16 + r;
            if (row < rows)
                out[(size_t)row * n + column0 + t] = __bfloat16_as_ushort(__float2bfloat16_rn(scale * tile[r][t]));
        }
    }
}

}  // namespace trellis_mma

// act f16 [s][rows][k]; words the payload [k/16][n/16][w]; ends the exclusive segment row ends; svh f16 [n]. With
// S == 1 the epilogue is applied here and `out` is written; with S > 1 the raw FP32 sums of split `blockIdx.y` go to
// partials[split][rows][n] (euhedral_trellis_epilogue finishes them). Split i covers the k16 tiles
// [i * T / S, (i + 1) * T / S), T = k / 16. Each CTA covers 128 columns of one segment: warp i owns the tile columns
// blockIdx.x * 128 + 16 i.. and decodes its tiles into its two m16n8k16 B fragments.
//   grid (n / 128, S), block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_trellis_mma(
        const unsigned short* __restrict__ act, const unsigned short* __restrict__ words, const int* __restrict__ ends,
        const unsigned short* __restrict__ svh, float* __restrict__ partials, unsigned short* __restrict__ out,
        int rows, int k, int n, int w, int splits, int s) {
    __shared__ unsigned int ring[8][kTrellisMaxWords];
    __shared__ float tile[16][128];
    const int column0 = blockIdx.x * 128;
    int g = 0;
    while (g + 1 < s && column0 >= ends[g]) ++g;
    const unsigned short* segment = act + (size_t)g * rows * k;
    const int tiles = k >> 4;
    const int first = (int)(((long long)blockIdx.y * tiles) / splits);
    const int last = (int)(((long long)(blockIdx.y + 1) * tiles) / splits);
    for (int row_base = 0; row_base < rows; row_base += 64) {
        const int remaining = (rows - row_base + 15) >> 4;
        if (remaining >= 3)
            trellis_mma::group<4>(segment, words, svh, partials, out, rows, row_base, k, n, w, splits, blockIdx.y, first, last, ring, tile);
        else if (remaining == 2)
            trellis_mma::group<2>(segment, words, svh, partials, out, rows, row_base, k, n, w, splits, blockIdx.y, first, last, ring, tile);
        else
            trellis_mma::group<1>(segment, words, svh, partials, out, rows, row_base, k, n, w, splits, blockIdx.y, first, last, ring, tile);
    }
}

#endif
