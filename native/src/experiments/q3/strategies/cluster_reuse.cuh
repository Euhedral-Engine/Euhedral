#pragma once
#include "q3/strategies/prefill.cuh"
#include <cooperative_groups.h>

namespace q3 {

// A cluster is a CM-by-CN array of CTAs: n varies fastest in cluster rank.
// Each CTA exclusively owns its Tile::kRows x Tile::kCols output tile, its
// accumulator fragments and its result staging. For each K group, (m,0)
// produces A[m], and (0,n) decodes/stages B[n]. Other CTAs borrow these tiles
// through distributed shared memory, copying their contents to CTA-private
// shared tiles before WMMA. The producer policy belongs to this strategy,
// never to the load/decode primitives; it can be replaced independently.
//
// Phase 1: owners write their own shared tiles; __syncthreads then cluster
// sync publishes them to remote readers. Phase 2: consumers copy remote tiles
// into CTA-local shared memory; __syncthreads makes these copies visible to
// WMMA. Phase 3: after all WMMA reads finish, cluster sync retires the remote
// borrows and permits producers to overwrite their tiles in the next K group.
// No CTA returns before the final cluster sync; remote shared addresses never
// escape that collective. All CTAs participate even at partial matrix edges.
// This deliberately keeps the 64-wide K stage and the original CTA WMMA order.
template<class Tile, int CM, int CN>
static __device__ __forceinline__ void cluster_prefill(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset, __nv_bfloat16* a, __nv_bfloat16* b_hi,
        __nv_bfloat16* b_lo, float* result) {
    constexpr int kThreads = Tile::kWarps * 32;
    static_assert(CM >= 1 && CN >= 1 && CM * CN <= 8, "cluster size must fit the target");
    const auto cluster = cooperative_groups::this_cluster();
    const unsigned int m = cluster.block_index().y, n = cluster.block_index().x;
    const unsigned int row_start = blockIdx.y * Tile::kRows;
    const unsigned int out_start = blockIdx.x * Tile::kCols;
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const Layout w(weights, in_features, scale_offset);
    Accumulators<Tile::kFrags> acc;
    acc.fill();
    for (unsigned int base = 0; base < in_features; base += kGroup) {
        if (n == 0) stage_activation_tile<Tile::kRows, kGroup, kThreads>(
                a, input, rows, in_features, row_start, base, threadIdx.x);
        if (m == 0) stage_weight_tile<Tile::kCols, Tile::kWarps>(
                b_hi, b_lo, w, out_start, out_features, base, warp, lane);
        __syncthreads();
        cluster.sync();
        if (n != 0) {
            const __nv_bfloat16* remote_a = cluster.map_shared_rank(a, m * CN);
            for (unsigned int i = threadIdx.x; i < Tile::kRows * kGroup; i += kThreads) a[i] = remote_a[i];
        }
        if (m != 0) {
            const unsigned int producer = n;
            const __nv_bfloat16* remote_hi = cluster.map_shared_rank(b_hi, producer);
            const __nv_bfloat16* remote_lo = cluster.map_shared_rank(b_lo, producer);
            for (unsigned int i = threadIdx.x; i < Tile::kCols * kGroup; i += kThreads) {
                b_hi[i] = remote_hi[i];
                b_lo[i] = remote_lo[i];
            }
        }
        __syncthreads();
        consume_mma_tile<Tile, kGroup, kGroup, kGroup>(acc, a, b_hi, b_lo, warp);
        cluster.sync();
    }
    store_accumulators<Tile>(result, acc, warp);
    __syncthreads();
    write_output_tile<Tile::kRows, Tile::kCols, kThreads>(
            output, result, rows, out_features, row_start, out_start, threadIdx.x);
}

}  // namespace q3
