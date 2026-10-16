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

// Ring slots per warp. Over the 3.0 bpw model shapes, 3 or 4 slots keep the most bytes in flight; deeper rings cost
// occupancy.
constexpr int kStages = 4;

__device__ __forceinline__ void copy16(void* shared, const void* global) {
    const unsigned int address = (unsigned int)__cvta_generic_to_shared(shared);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(global));
}

__device__ __forceinline__ void copy_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void copy_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ unsigned int pack_half2(half2 value) { return *reinterpret_cast<unsigned int*>(&value); }

/// The tile ring of `W` uint16 words as each lane decodes it. A lane's eight positions lane * 8 + j are contiguous, so
/// their windows lie in `kWords` consecutive ring words from the lane's first window, and window j starts kOffset(j)
/// bits after window 0 whatever the lane: lane * 8 is 0 or 8 mod 16 and the half step mask sets only odd bits.
template <int W>
struct Ring {
    static constexpr int kWords32 = W / 2, kBits = W / 16;
    static constexpr bool kHalf = (W & 15) == 8;
    __host__ __device__ static constexpr int kOffset(int j) { return kBits * j + (kHalf ? (j + 1) / 2 : 0); }
    // The last window ends kOffset(7) + 16 bits after the first, which starts up to 31 bits into the first word.
    static constexpr int kWords = (31 + kOffset(7) + 16 + 31) / 32;
};

/// Window j of a lane's normalized words (see Ring): the 16 bits starting kOffset(j) bits into u, MSB first.
template <int W>
__device__ __forceinline__ unsigned int window(const unsigned int* u, int j) {
    const int offset = Ring<W>::kOffset(j);
    const int i = offset >> 5, o = offset & 31;
    if (o <= 16) return (u[i] >> (16 - o)) & 0xFFFFu;
    return __funnelshift_l(u[i + 1], u[i], o) >> 16;
}

/// `MT` m16 row tiles starting at row `row_base` over the k16 tiles [first, last) of one 128-column block, for tiles of
/// `W` words. Warp `warp` owns the 16-column tile blockIdx.x * 8 + warp. The same instruction sequence runs for every
/// m-tile, rows past `rows` are zero, and k16 tiles are accumulated in increasing order, so a row's bits depend on
/// nothing but its own data and the K range.
template <int W, int MT>
__device__ __forceinline__ void group(const unsigned short* __restrict__ act, const unsigned short* __restrict__ words,
        const unsigned short* __restrict__ svh, float* __restrict__ partials, unsigned short* __restrict__ out,
        int rows, int row_base, int k, int n, int splits, int split, int first, int last,
        unsigned int (*ring)[kStages][kTrellisMaxWords], float (*tile)[128]) {
    using R = Ring<W>;
    constexpr int nw = R::kWords32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int grp = lane >> 2, q = lane & 3;
    const int tiles_n = n >> 4;
    const int tj = blockIdx.x * 8 + warp;

    // This lane's ring words and the bit where its first window starts in the first of them, the same for every tile.
    int at = frac_s(lane * 8, W >> 4, R::kHalf ? 0xAAAAu : 0u, W) - 16;
    if (at < 0) at += nw * 32;
    const int shift = at & 31;
    int word[R::kWords];
#pragma unroll
    for (int i = 0; i < R::kWords; ++i) word[i] = ((at >> 5) + i) % nw;

    float acc[MT][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[m][h][i] = 0.0f;

    // Each warp streams its tiles through its own ring of kStages slots with 16-byte asynchronous copies, kStages - 1
    // tiles ahead of the one it decodes. The lanes of the 16-byte chunks of a tile copy them from `fetch`, which steps
    // one k16 tile at a time.
    const size_t stride = (size_t)tiles_n * W;
    const bool copies = lane < W / 8;
    const unsigned short* fetch = words + (size_t)first * stride + (size_t)tj * W + lane * 8;
#pragma unroll
    for (int i = 0; i < kStages - 1; ++i) {
        if (copies && first + i < last) copy16(&ring[warp][i][lane * 4], fetch);
        copy_commit();
        fetch += stride;
    }
    // This lane's A fragment elements of k16 tile `ti` are at a + 16 * ti plus a fixed offset per m-tile and half.
    const unsigned short* a_lane = act + (size_t)(row_base + grp) * k + 2 * q + (size_t)first * 16;
    int slot = 0;
    for (int ti = first; ti < last; ++ti) {
        copy_wait<kStages - 2>();
        __syncwarp();
        if (copies && ti + kStages - 1 < last) copy16(&ring[warp][slot == 0 ? kStages - 1 : slot - 1][lane * 4], fetch);
        copy_commit();
        fetch += stride;
        const unsigned int* mine = ring[warp][slot];
        slot = slot + 1 == kStages ? 0 : slot + 1;
        // The lane's words shifted so its first window starts at bit 0 (MSB first).
        unsigned int u[R::kWords];
#pragma unroll
        for (int i = 0; i < R::kWords; ++i) u[i] = mine[word[i]];
#pragma unroll
        for (int i = 0; i + 1 < R::kWords; ++i) u[i] = __funnelshift_l(u[i + 1], u[i], shift);
        u[R::kWords - 1] <<= shift;
        unsigned int b[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int half_index = 0; half_index < 2; ++half_index) {
                const int j = h * 4 + half_index * 2;
                b[h][half_index] = pack_half2(decode_mul1_2(window<W>(u, j), window<W>(u, j + 1)));
            }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int row0 = row_base + m * 16 + grp, row1 = row0 + 8;
            const unsigned short* a0 = a_lane + (size_t)(m * 16) * k;
            const unsigned short* a1 = a0 + (size_t)8 * k;
            unsigned int a[4];
            a[0] = row0 < rows ? *reinterpret_cast<const unsigned int*>(a0) : 0u;
            a[1] = row1 < rows ? *reinterpret_cast<const unsigned int*>(a1) : 0u;
            a[2] = row0 < rows ? *reinterpret_cast<const unsigned int*>(a0 + 8) : 0u;
            a[3] = row1 < rows ? *reinterpret_cast<const unsigned int*>(a1 + 8) : 0u;
            mma_f16(acc[m][0], a, b[0][0], b[0][1]);
            mma_f16(acc[m][1], a, b[1][0], b[1][1]);
        }
        a_lane += 16;
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

/// All rows in groups of up to 64 (four m16 tiles).
template <int W>
__device__ __forceinline__ void rows(const unsigned short* __restrict__ act, const unsigned short* __restrict__ words,
        const unsigned short* __restrict__ svh, float* __restrict__ partials, unsigned short* __restrict__ out,
        int rows, int k, int n, int splits, int first, int last, unsigned int (*ring)[kStages][kTrellisMaxWords],
        float (*tile)[128]) {
    for (int row_base = 0; row_base < rows; row_base += 64) {
        const int remaining = (rows - row_base + 15) >> 4;
        if (remaining >= 3)
            group<W, 4>(act, words, svh, partials, out, rows, row_base, k, n, splits, blockIdx.y, first, last, ring, tile);
        else if (remaining == 2)
            group<W, 2>(act, words, svh, partials, out, rows, row_base, k, n, splits, blockIdx.y, first, last, ring, tile);
        else
            group<W, 1>(act, words, svh, partials, out, rows, row_base, k, n, splits, blockIdx.y, first, last, ring, tile);
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
    __shared__ __align__(16) unsigned int ring[8][trellis_mma::kStages][kTrellisMaxWords];
    __shared__ float tile[16][128];
    const int column0 = blockIdx.x * 128;
    int g = 0;
    while (g + 1 < s && column0 >= ends[g]) ++g;
    const unsigned short* segment = act + (size_t)g * rows * k;
    const int tiles = k >> 4;
    const int first = (int)(((long long)blockIdx.y * tiles) / splits);
    const int last = (int)(((long long)(blockIdx.y + 1) * tiles) / splits);
    switch (w) {
#define EUHEDRAL_TRELLIS_MMA_WIDTH(W) \
    case W: \
        trellis_mma::rows<W>(segment, words, svh, partials, out, rows, k, n, splits, first, last, ring, tile); \
        break;
        EUHEDRAL_TRELLIS_MMA_WIDTH(16) EUHEDRAL_TRELLIS_MMA_WIDTH(24) EUHEDRAL_TRELLIS_MMA_WIDTH(32)
        EUHEDRAL_TRELLIS_MMA_WIDTH(40) EUHEDRAL_TRELLIS_MMA_WIDTH(48) EUHEDRAL_TRELLIS_MMA_WIDTH(56)
        EUHEDRAL_TRELLIS_MMA_WIDTH(64) EUHEDRAL_TRELLIS_MMA_WIDTH(72) EUHEDRAL_TRELLIS_MMA_WIDTH(80)
        EUHEDRAL_TRELLIS_MMA_WIDTH(88) EUHEDRAL_TRELLIS_MMA_WIDTH(96) EUHEDRAL_TRELLIS_MMA_WIDTH(104)
        EUHEDRAL_TRELLIS_MMA_WIDTH(112) EUHEDRAL_TRELLIS_MMA_WIDTH(120) EUHEDRAL_TRELLIS_MMA_WIDTH(128)
#undef EUHEDRAL_TRELLIS_MMA_WIDTH
    }
}

#endif
