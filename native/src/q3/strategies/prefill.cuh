#pragma once
#include "../primitives/activation.cuh"
#include "../primitives/staging.cuh"
#include "../primitives/prefetch.cuh"
#include "../primitives/mma.cuh"
#include "../primitives/mma_leaf.cuh"
#include "../primitives/writeback.cuh"

namespace q3 {
// The two trivial regions use the same bytes in different phases of one CTA.
template<class Tile, int A_STRIDE = kGroup>
union alignas(32) PrefillShared {
    struct Activation {
        __nv_bfloat16 values[Tile::kRows * A_STRIDE];
    };
    struct Result {
        float values[Tile::kRows * Tile::kCols];
    };
    Activation a;
    Result result;
    struct ActivateA {};
    struct ActivateResult {};
    PrefillShared() = default;
    __device__ PrefillShared(ActivateA) : a() {}
    __device__ PrefillShared(ActivateResult) : result() {}
};

// Tiled MMA prefill (leaf selected per kernel). The CTA owns Tile::kRows x Tile::kCols outputs and consumes
// K one G64 group at a time:
//   decode prefetched A and B into shared -> barrier
//   each compute warp prefetches the next compact group while consuming this
//   decoded tile with the MMA leaf in the original K order -> retirement barrier
// then stores accumulators to shared, barriers, and writes BF16 output. The two
// per-step barriers are the single-buffer reuse edges: staged tiles are visible
// only after the first, and may be overwritten only after the second.
template<class Tile, int A_STRIDE = kGroup, int B_STRIDE = kGroup, class Leaf = WmmaLeaf<Tile>>
static __device__ __forceinline__ void tiled_prefill(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset, PrefillShared<Tile, A_STRIDE>& staging,
        __nv_bfloat16* b_hi, __nv_bfloat16* b_lo) {
    using Shared = PrefillShared<Tile, A_STRIDE>;
    using Activation = typename Shared::Activation;
    using Result = typename Shared::Result;
    static_assert(__is_trivial(Activation), "A wrapper must be trivial");
    static_assert(__is_trivial(Result), "result wrapper must be trivial");
    static_assert(__is_trivially_constructible(Shared), "shared union must default-construct trivially");
    constexpr int kThreads = Tile::kWarps * 32;
    const unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned int output_tiles = (out_features + Tile::kCols - 1u) / Tile::kCols;
    const unsigned int row_start = (blockIdx.x / output_tiles) * Tile::kRows;
    const unsigned int out_start = (blockIdx.x % output_tiles) * Tile::kCols;
    const Layout w(weights, in_features, scale_offset);
    typename Leaf::Acc acc;
    Leaf::fill(acc);
    // Reconstruct the union with A selected; never name the inactive member.
    if (threadIdx.x == 0) new (static_cast<void*>(&staging)) Shared(typename Shared::ActivateA{});
    __syncthreads();
    __nv_bfloat16* a = staging.a.values;
    // BF16 pointers may be only two-byte aligned; keep the original single-
    // stage path for those inputs and for odd row strides. The fast path uses
    // one compact register stage plus the existing decoded shared stage.
    if ((reinterpret_cast<unsigned long long>(input) & 3ull) == 0ull && (in_features & 1u) == 0u) {
        CompactPrefetch<Tile> next;
        if (in_features != 0)
            prefetch_compact_tile(next, input, w, rows, in_features, out_features,
                    row_start, out_start, 0, threadIdx.x, warp, lane);
        for (unsigned int base = 0; base < in_features; base += kGroup) {
            stage_prefetched_activation<Tile, A_STRIDE>(a, next, threadIdx.x);
            stage_prefetched_weights<Tile, B_STRIDE>(b_hi, b_lo, next, warp, lane);
            __syncthreads();
            if (base + kGroup < in_features)
                prefetch_compact_tile(next, input, w, rows, in_features, out_features,
                        row_start, out_start, base + kGroup, threadIdx.x, warp, lane);
            Leaf::template consume<kGroup, A_STRIDE, B_STRIDE>(acc, a, b_hi, b_lo, warp);
            __syncthreads();
        }
    } else {
        for (unsigned int base = 0; base < in_features; base += kGroup) {
            stage_activation_tile<Tile::kRows, kGroup, kThreads, A_STRIDE>(
                    a, input, rows, in_features, row_start, base, threadIdx.x);
            stage_weight_tile<Tile::kCols, Tile::kWarps, B_STRIDE>(
                    b_hi, b_lo, w, out_start, out_features, base, warp, lane);
            __syncthreads();
            Leaf::template consume<kGroup, A_STRIDE, B_STRIDE>(acc, a, b_hi, b_lo, warp);
            __syncthreads();
        }
    }
    // Every thread has completed its final A read before replacing that member.
    if (threadIdx.x == 0) new (static_cast<void*>(&staging)) Shared(typename Shared::ActivateResult{});
    __syncthreads();
    float* result = staging.result.values;
    Leaf::store(result, acc, warp);
    __syncthreads();
    write_output_tile<Tile::kRows, Tile::kCols, kThreads>(
            output, result, rows, out_features, row_start, out_start, threadIdx.x);
}

// Tile geometries used by the current routes. 2 x 2 warps with one or two
// fragments per warp give 32 x 32 and 64 x 32 CTA tiles. Host dispatch in
// q3_linear_bf16.c sizes grids with the same constants.
using Prefill32 = WarpTile<2, 2, 1>;
using Prefill64 = WarpTile<2, 2, 2>;


}  // namespace q3
