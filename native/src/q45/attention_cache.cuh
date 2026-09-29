#pragma once
#include "strategies/prefill.cuh"

namespace q45 {
// Queries and gates retain their established row stride. Value columns are
// rounded once at their producer and written directly to reserved cache rows.
struct ValueCacheWriteback {
    unsigned short* cache;
    unsigned int query_width;
    unsigned long long start;
    template<int ROWS, int COLS, int THREADS>
    __device__ __forceinline__ void write(unsigned short* output, const float* result,
            unsigned int rows, unsigned int width, unsigned int row_start, unsigned int col_start,
            unsigned int thread) const {
        for (unsigned int i = thread; i < ROWS * COLS; i += THREADS) {
            unsigned int row = row_start + i / COLS, col = col_start + i % COLS;
            if (row >= rows || col >= width) continue;
            if (col < query_width) write_bf16(output, row, col, width, result[i]);
            else cache[(start + row) * (width - query_width) + col - query_width] = float_to_bf16(result[i]);
        }
    }
};
}

extern "C" __global__ __launch_bounds__(128) void euhedral_attention_value_cache_bf16(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned short* cache, unsigned int query_width, unsigned long long start) {
    using Tile = q45::Prefill64;
    __shared__ q45::PrefillShared<Tile, 80> staging;
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * 64];
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * 64];
    q45::tiled_prefill<5, Tile, 80, 64, q45::MmaPingPongLeaf<Tile>, q45::ValueCacheWriteback>(
            input, weights, output, rows, in_features, out_features, staging, b_hi, b_lo,
            {cache, query_width, start});
}
