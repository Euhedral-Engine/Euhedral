// The trellis prefill on block-scaled FP8 tensor cores (sm_12xa): the activations from euhedral_trellis_prepass's FP8
// layout, the decoded weights rounded to E4M3, one m16n8k32 (kind::mxf8f6f4, FP32 accumulation) per two k16 tiles at
// four times the rate of the FP32-accumulating m16n8k16. Each row's result depends only on its own data; it is not
// bitwise that of FP16 operands, so verifications keep euhedral_trellis_mma_rows.
#ifndef EUHEDRAL_TRELLIS_FP8_CUH
#define EUHEDRAL_TRELLIS_FP8_CUH

#include "trellis/mma.cuh"

namespace trellis_fp8 {

using namespace trellis_mma;

/// k16 tiles per staged activation chunk: 128 E4M3 bytes per row, the chunk slots' size, and four k32 scale bytes.
constexpr int kChunkTiles = 8;

/// The E4M3 m16n8k32 with UE8M0 block scales: `scale_a` holds each lane's row scale in every byte (lane 4r gives row
/// r, lane 4r + 1 row r + 8), the weights take unit scales.
__device__ __forceinline__ void mma_e4m3(
        float* d, const unsigned int* a, unsigned int b0, unsigned int b1, unsigned int scale_a) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(scale_a), "r"(0x7f7f7f7fu));
}

/// The four 8x8 b16 matrices at the rows `address` names for lanes 8i..8i + 7 (matrix i).
__device__ __forceinline__ void load_matrix_x4(unsigned int* a, const void* address) {
    const unsigned int shared = (unsigned int)__cvta_generic_to_shared(address);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(shared));
}

/// Two FP16 values as two E4M3 bytes (round to nearest even, satfinite), the first in the low byte.
__device__ __forceinline__ unsigned int e4m3x2(unsigned int f16x2) {
    unsigned short e;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(e) : "r"(f16x2));
    return e;
}

/// A 4-byte copy of the first `bytes` (0 or 4) of `global`, zero-filling the rest.
__device__ __forceinline__ void copy4_fill(void* shared, const void* global, int bytes) {
    const unsigned int address = (unsigned int)__cvta_generic_to_shared(shared);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(address), "l"(global), "r"(bytes));
}

/// group_splits' accumulation from the FP8 layout (euhedral_trellis_prepass layout 2): `act` the segment's E4M3 rows
/// (k bytes each), `aux` its UE8M0 scale rows (k / 32 bytes each), `shared_aux` the kernel's unsigned int[2][SR]: each
/// chunk's four scale bytes per row. The whole K range in chunks of kChunkTiles tiles; K is a multiple of 128, so the
/// loop over a chunk is unrolled and its ring slots, staging and barrier are compile-time. The sums go to the first
/// RegP row tiles' `regs` and to `sum` from row 8 RegP, as accumulate's last split.
struct Fp8Accumulation {
    __device__ static const void* segment(const unsigned short* act, int g, int rows, int k, int s) {
        return reinterpret_cast<const unsigned char*>(act) + (size_t)g * rows * k;
    }
    __device__ static const void* aux(const unsigned short* act, int g, int rows, int k, int s) {
        return reinterpret_cast<const unsigned char*>(act) + (size_t)s * rows * k + (size_t)g * rows * (k >> 5);
    }
    template <int W, int P, int SR, int RegP>
    __device__ static void run(float (&acc)[P][4], const void* activations, const unsigned short* __restrict__ words,
            int rows, int row_base, int k, int n, int splits, unsigned int (*ring)[kStages][kTrellisMaxWords],
            ChunkSlots<SR>* chunks, float (*sum)[128], float (*regs)[4], const void* aux, void* shared_aux) {
        static_assert(P % 2 == 0, "wide tiles");
        const unsigned char* act = static_cast<const unsigned char*>(activations);
        const unsigned char* scale_rows = static_cast<const unsigned char*>(aux);
        unsigned int (*scales)[SR] = static_cast<unsigned int (*)[SR]>(shared_aux);
        using R = Ring<W>;
        constexpr int nw = R::kWords32;
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const int grp = lane >> 2, q = lane & 3;
        const int tiles_n = n >> 4, last = k >> 4;
        const int tj = blockIdx.x * 8 + warp;

        int at = frac_s(lane * 8, W >> 4, R::kHalf ? 0xAAAAu : 0u, W) - 16;
        if (at < 0) at += nw * 32;
        const int shift = at & 31;
        int word[R::kWords];
#pragma unroll
        for (int i = 0; i < R::kWords; ++i) word[i] = ((at >> 5) + i) % nw;

        const size_t stride = (size_t)tiles_n * W;
        const bool copies = lane < W / 8;
        const unsigned short* fetch = words + (size_t)tj * W + lane * 8;
        const int chunk_row = threadIdx.x >> 3, chunk_segment = threadIdx.x & 7;
        // The chunk that starts at k16 tile `tile` into slot `slot`: 16-byte segments of the rows' E4M3 bytes and each
        // row's four scale bytes, rows past `rows` and chunks past K zero.
        const auto stage = [&](int tile, int slot) {
            const int from = tile < last ? tile : 0;
#pragma unroll
            for (int r = chunk_row; r < SR; r += 32) {
                const bool live = row_base + r < rows && tile < last;
                copy16_fill(&(*chunks)[slot][r][chunk_segment ^ (r & 7)],
                        act + (size_t)chunk_live_row(row_base + r, rows) * k + (size_t)from * 16 + chunk_segment * 16,
                        live ? 16 : 0);
            }
            if (threadIdx.x < SR) {
                const int r = threadIdx.x;
                const bool live = row_base + r < rows && tile < last;
                copy4_fill(&scales[slot][r],
                        scale_rows + (size_t)chunk_live_row(row_base + r, rows) * (k >> 5) + (from >> 1),
                        live ? 4 : 0);
            }
        };
#pragma unroll
        for (int i = 0; i < kStages - 1; ++i) {
            if (copies) copy16(&ring[warp][i][lane * 4], fetch);
            if (i == 0) stage(0, 0);
            copy_commit();
            fetch += stride;
        }
        for (int t0 = 0; t0 < last; t0 += kChunkTiles) {
            const int slot = (t0 / kChunkTiles) & 1;
            unsigned int b[2][2], scale[P / 2];
#pragma unroll
            for (int j = 0; j < kChunkTiles; ++j) {
                const int ti = t0 + j;
                copy_wait<kStages - 2>();
                __syncwarp();
                // Tile ti + 3 into the slot tile ti - 1 used; the next chunk when its first tile is 3 ahead.
                if (copies && ti + kStages - 1 < last)
                    copy16(&ring[warp][(j + kStages - 1) % kStages][lane * 4], fetch);
                if (j + kStages - 1 == kChunkTiles) stage(t0 + kChunkTiles, slot ^ 1);
                copy_commit();
                // The chunk is complete in every thread's groups: one barrier makes it visible. Chunk c + 2 reuses
                // c's slot from its staging in chunk c + 1, after every warp passed chunk c + 1's barrier.
                if (j == 0) {
                    __syncthreads();
#pragma unroll
                    for (int m = 0; m < P / 2; ++m) scale[m] = scales[slot][m * 16 + grp + 8 * (q & 1)];
                }
                fetch += stride;
                const unsigned int* mine = ring[warp][j % kStages];
                unsigned int u[R::kWords];
#pragma unroll
                for (int i = 0; i < R::kWords; ++i) u[i] = mine[word[i]];
#pragma unroll
                for (int i = 0; i + 1 < R::kWords; ++i) u[i] = __funnelshift_l(u[i + 1], u[i], shift);
                u[R::kWords - 1] <<= shift;
                // d[h], d[h + 2]: column 8h + g at k 2q, 2q + 1 and 2q + 8, 2q + 9 (accumulate's wide B fragment),
                // as the four E4M3 bytes of the k32 B register of this tile (b[j & 1]).
                unsigned int d[4];
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int half_index = 0; half_index < 2; ++half_index) {
                        const int jj = h * 4 + half_index * 2;
                        d[h + 2 * half_index] = pack_half2(decode_mul1_2(window<W>(u, jj), window<W>(u, jj + 1)));
                    }
#pragma unroll
                for (int h = 0; h < 2; ++h) b[j & 1][h] = e4m3x2(d[h]) | e4m3x2(d[h + 2]) << 16;
                if (j & 1) {
                    // k32 block `pair` of the chunk, one ldmatrix: matrix i = lane / 8 is rows 8 (i & 1).. of the 16
                    // at tile 2 pair + (i >> 1), each tile one 16-byte segment, swizzled by the row.
                    const int pair = j >> 1, matrix = lane >> 3;
#pragma unroll
                    for (int m = 0; m < P / 2; ++m) {
                        const int row = m * 16 + (lane & 7) + 8 * (matrix & 1);
                        unsigned int a[4];
                        load_matrix_x4(a, &(*chunks)[slot][row][(2 * pair + (matrix >> 1)) ^ (lane & 7)]);
                        const unsigned int block_scale = __byte_perm(scale[m], 0u, 0x1111u * pair);
                        mma_e4m3(acc[2 * m], a, b[0][0], b[1][0], block_scale);
                        mma_e4m3(acc[2 * m + 1], a, b[0][1], b[1][1], block_scale);
                    }
                }
            }
        }
#pragma unroll
        for (int p = 0; p < P; ++p)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (p < RegP) regs[p][i] = acc[p][i];
                else sum[acc_row<false>(p, i) - 8 * RegP][acc_column<false>(p, i)] = acc[p][i];
            }
    }
};

}  // namespace trellis_fp8

// euhedral_trellis_mma_rows on FP8 operands (trellis_fp8::Fp8Accumulation): launches of many rows other than
// verifications. act is the prepass's FP8 layout of the s segments: E4M3 [s][rows][k], then UE8M0 [s][rows][k / 32].
//   grid (n / 128, ceil(rows / kSplitRows)), block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_trellis_mma_rows_fp8(
        const unsigned short* __restrict__ act, const unsigned short* __restrict__ words, const int* __restrict__ ends,
        const unsigned short* __restrict__ svh, unsigned short* __restrict__ out, int rows, int k, int n, int w,
        int splits, int s) {
    __shared__ __align__(16) unsigned int ring[8][trellis_mma::kStages][kTrellisMaxWords];
    __shared__ float sum[trellis_mma::kSplitRows][128];
    __shared__ trellis_mma::ChunkSlots<trellis_mma::kSplitRows> chunks;
    __shared__ unsigned int scales[2][trellis_mma::kSplitRows];
    trellis_mma::whole_k<trellis_mma::kSplitRows, trellis_fp8::Fp8Accumulation>(
            act, words, ends, svh, out, rows, k, n, w, splits, s, ring, sum, &chunks, scales);
}

// euhedral_trellis_mma_rows_fp8 in blocks of kSplitRows64 rows, two per SM.
//   grid (n / 128, ceil(rows / kSplitRows64)), block 256.
extern "C" __global__ __launch_bounds__(256, 2) void euhedral_trellis_mma_rows64_fp8(
        const unsigned short* __restrict__ act, const unsigned short* __restrict__ words, const int* __restrict__ ends,
        const unsigned short* __restrict__ svh, unsigned short* __restrict__ out, int rows, int k, int n, int w,
        int splits, int s) {
    __shared__ __align__(16) unsigned int ring[8][trellis_mma::kStages][kTrellisMaxWords];
    __shared__ float sum[32][128];
    __shared__ trellis_mma::ChunkSlots<trellis_mma::kSplitRows64> chunks;
    __shared__ unsigned int scales[2][trellis_mma::kSplitRows64];
    trellis_mma::whole_k<trellis_mma::kSplitRows64, trellis_fp8::Fp8Accumulation>(
            act, words, ends, svh, out, rows, k, n, w, splits, s, ring, sum, &chunks, scales);
}

#endif
