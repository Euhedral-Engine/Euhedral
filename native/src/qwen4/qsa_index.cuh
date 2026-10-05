#pragma once
#include "qwen4/common.cuh"

// QSA indexer: block scores and top-k block selection (docs/FLASH_NEXT_QSA.md).
//
// Query row p (position p) sees nb = (p + 1) / 4 complete blocks. The score of block j is
//   sum over the 4 indexer heads h of relu(q_h . key_j) / sqrt(128)
// with FP32 arithmetic on BF16 operands, and the query keeps min(512, nb) blocks of the highest score.

namespace q4 {

static __device__ __forceinline__ void mma_bf16_16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
        unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

}  // namespace q4

// Scores of a tile of rows against the pooled block keys on BF16 tensor cores with FP32 accumulation.
//   q       rotated indexer queries, row row_begin + r at q + (row_begin + r) * q_row_stride, 4 heads of 128 values
//   keys    pooled, normalized, rotated block keys [block][128]
//   scores  [tile_rows][score_stride] FP32; entry (r, j) is written for j < nb of row row_begin + r only
// A warp computes the scores of 4 rows x 4 heads (the 16 rows of an m16n8k16 tile) against 8 blocks per step; the
// four heads of a row sit in rows 4 r .. 4 r + 3 of the tile, so the relu-sum over heads is a shuffle across the
// lanes that differ in their row bits.
//   grid (ceil(blocks_total / 64), ceil(tile_rows / 32)), block 256: 8 warps = 32 rows, 8 n-tiles = 64 blocks.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_qsa_scores(
        const unsigned short* __restrict__ q, const unsigned short* __restrict__ keys, float* __restrict__ scores,
        unsigned int row_begin, unsigned int tile_rows, unsigned int start, unsigned int q_row_stride,
        unsigned int score_stride, unsigned int blocks_total) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int g = lane >> 2, tig = lane & 3u;
    const unsigned int row0 = blockIdx.y * 32u + warp * 4u;  // first tile row of this warp
    if (row0 >= tile_rows) return;
    const unsigned int block0 = blockIdx.x * 64u;
    // A fragments: tile row m = 4 * lrow + head. Rows g and g + 8 of the fragment are (lrow g >> 2, head g & 3) and
    // (lrow 2 + (g >> 2), head g & 3).
    unsigned a[8][4];
    const unsigned int head = g & 3u;
    const unsigned int rowA = row0 + (g >> 2), rowB = row0 + 2u + (g >> 2);
#pragma unroll
    for (int ks = 0; ks < 8; ks++) {
        const unsigned int dim = ks * 16u + 2u * tig;
        a[ks][0] = rowA < tile_rows ? *reinterpret_cast<const unsigned*>(q + (unsigned long long)(row_begin + rowA) * q_row_stride + head * 128u + dim) : 0u;
        a[ks][1] = rowB < tile_rows ? *reinterpret_cast<const unsigned*>(q + (unsigned long long)(row_begin + rowB) * q_row_stride + head * 128u + dim) : 0u;
        a[ks][2] = rowA < tile_rows ? *reinterpret_cast<const unsigned*>(q + (unsigned long long)(row_begin + rowA) * q_row_stride + head * 128u + dim + 8u) : 0u;
        a[ks][3] = rowB < tile_rows ? *reinterpret_cast<const unsigned*>(q + (unsigned long long)(row_begin + rowB) * q_row_stride + head * 128u + dim + 8u) : 0u;
    }
    // Blocks visible to the rows this lane writes: nb = (start + row + 1) / 4.
    const unsigned int blocksA = (start + row_begin + rowA + 1u) >> 2, blocksB = (start + row_begin + rowB + 1u) >> 2;
    const float inverse_root = 1.0f / 11.313708498984761f;  // 1 / sqrt(128)
    for (unsigned int n = 0; n < 8u; n++) {
        const unsigned int key_block = block0 + n * 8u + g;
        const unsigned short* key = keys + (unsigned long long)key_block * 128u;
        const bool real = key_block < blocks_total;
        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const unsigned int dim = ks * 16u + 2u * tig;
            const unsigned b0 = real ? *reinterpret_cast<const unsigned*>(key + dim) : 0u;
            const unsigned b1 = real ? *reinterpret_cast<const unsigned*>(key + dim + 8u) : 0u;
            q4::mma_bf16_16816(c, a[ks][0], a[ks][1], a[ks][2], a[ks][3], b0, b1);
        }
        float sum[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float v = fmaxf(c[i], 0.0f);
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            sum[i] = v;
        }
        if (head == 0u) {
            const unsigned int column = block0 + n * 8u + 2u * tig;
            if (rowA < tile_rows) {
                float* out = scores + (unsigned long long)rowA * score_stride;
                if (column < blocksA) out[column] = sum[0] * inverse_root;
                if (column + 1u < blocksA) out[column + 1u] = sum[1] * inverse_root;
            }
            if (rowB < tile_rows) {
                float* out = scores + (unsigned long long)rowB * score_stride;
                if (column < blocksB) out[column] = sum[2] * inverse_root;
                if (column + 1u < blocksB) out[column + 1u] = sum[3] * inverse_root;
            }
        }
    }
}

namespace q4 {

// Rank of this thread among the flagged threads of the block (the number of flagged threads before it) and the
// number flagged in all. Two barriers; `warp_counts` has one slot per warp.
static __device__ __forceinline__ unsigned int scan_flag(bool flag, unsigned int* warp_counts, unsigned int& total) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const unsigned int ballot = __ballot_sync(0xffffffffu, flag);
    __syncthreads();
    if (lane == 0) warp_counts[warp] = __popc(ballot);
    __syncthreads();
    unsigned int before = 0, all = 0;
    for (unsigned int w = 0; w < warps; w++) {
        const unsigned int count = warp_counts[w];
        before += w < warp ? count : 0u;
        all += count;
    }
    total = all;
    return before + __popc(ballot & ((1u << lane) - 1u));
}

// Order-preserving key of a non-negative score: its bit pattern.
static __device__ __forceinline__ unsigned int score_key(float score) {
    return __float_as_uint(fmaxf(score, 0.0f));
}

}  // namespace q4

// Top-k block selection of one row per CTA. The row keeps k = min(budget, nb) blocks: the k highest scores, and among
// equal scores the lower block ids. Radix selection finds the k-th highest score in four 8-bit passes over the
// row's scores; one in-order pass then compacts the blocks scoring above it and the first of those equal to it, so
// the ids come out ascending. A row with nb <= budget keeps every block and reads no score.
//   scores  [tile_rows][score_stride], row row_begin + r at index r
//   ids     [rows][budget] int32, counts [rows] int32 (the number of ids written), both by chunk row
//   grid tile_rows, block 1024.
extern "C" __global__ __launch_bounds__(1024) void euhedral_q4_qsa_select(
        const float* __restrict__ scores, int* __restrict__ ids, int* __restrict__ counts, unsigned int row_begin,
        unsigned int start, unsigned int score_stride, unsigned int budget) {
    __shared__ unsigned int histogram[256];
    __shared__ unsigned int warp_counts[32];
    __shared__ unsigned int state[2];
    const unsigned int tid = threadIdx.x, threads = blockDim.x;
    const unsigned int row = row_begin + blockIdx.x;
    const unsigned int nb = (start + row + 1u) >> 2;
    int* out = ids + (unsigned long long)row * budget;
    if (nb <= budget) {
        for (unsigned int i = tid; i < nb; i += threads) out[i] = (int)i;
        if (tid == 0) counts[row] = (int)nb;
        return;
    }
    const float* s = scores + (unsigned long long)blockIdx.x * score_stride;
    unsigned int prefix = 0, remaining = budget;
    for (int pass = 0; pass < 4; pass++) {
        const int shift = 24 - 8 * pass;
        for (unsigned int i = tid; i < 256u; i += threads) histogram[i] = 0u;
        __syncthreads();
        for (unsigned int e = tid; e < nb; e += threads) {
            const unsigned int key = q4::score_key(s[e]);
            if (pass == 0 || ((key ^ prefix) >> (shift + 8)) == 0u) atomicAdd(&histogram[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned int accumulated = 0, digit = 0;
            for (int d = 255; d >= 0; d--) {
                const unsigned int count = histogram[d];
                if (accumulated + count >= remaining) {
                    digit = (unsigned int)d;
                    break;
                }
                accumulated += count;
            }
            state[0] = digit;
            state[1] = accumulated;
        }
        __syncthreads();
        prefix |= state[0] << shift;
        remaining -= state[1];
        __syncthreads();
    }
    // prefix is the k-th highest key; `remaining` blocks scoring exactly it still have to be taken, lowest ids first.
    unsigned int taken_ties = 0, written = 0;
    for (unsigned int base = 0; base < nb && written < budget; base += threads) {
        const unsigned int e = base + tid;
        const unsigned int key = e < nb ? q4::score_key(s[e]) : 0u;
        const bool greater = e < nb && key > prefix, tie = e < nb && key == prefix;
        unsigned int tie_total, selected_total;
        const unsigned int tie_rank = q4::scan_flag(tie, warp_counts, tie_total);
        const bool keep = greater || (tie && taken_ties + tie_rank < remaining);
        const unsigned int rank = q4::scan_flag(keep, warp_counts, selected_total);
        if (keep) out[written + rank] = (int)e;
        taken_ties += tie_total;
        written += selected_total;
    }
    if (tid == 0) counts[row] = (int)budget;
}
