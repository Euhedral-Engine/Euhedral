#pragma once
#include "../primitives/staging.cuh"
#include "../primitives/prefetch.cuh"
#include "../primitives/writeback.cuh"
// The MMA leaves and warp tiles are format-independent: they consume staged
// BF16 A and B hi/lo tiles, so Q4/Q5 share them with Q3.
#include "../../q3/primitives/mma_leaf.cuh"

namespace q45 {
using q3::WarpTile;
using q3::WmmaLeaf;
using q3::MmaSyncLeaf;
using q3::MmaPingPongLeaf;

// The activation tile and the FP32 result tile use the same bytes in different
// phases of one CTA.
template<class Tile, int A_STRIDE>
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

// Tiled MMA prefill. The CTA owns Tile::kRows x Tile::kCols outputs and consumes
// K one G64 group at a time:
//   stage prefetched A and decoded B hi/lo into shared -> barrier
//   prefetch the next compact group into registers while the MMA leaf consumes
//   this tile in the original K order -> retirement barrier
// then stores accumulators to shared, barriers, and writes BF16 output. The two
// per-step barriers are the single-buffer reuse edges. A 2-byte-aligned input
// base takes the sequential staging path, which stages identical tiles.
struct MatrixWriteback {
    template<int ROWS, int COLS, int THREADS>
    __device__ __forceinline__ void write(unsigned short* output, const float* result,
            unsigned int rows, unsigned int width, unsigned int row, unsigned int col, unsigned int thread) const {
        write_output_tile<ROWS, COLS, THREADS>(output, result, rows, width, row, col, thread);
    }
};

template<int BITS, class Tile, int A_STRIDE, int B_STRIDE, class Leaf, class Writeback = MatrixWriteback>
static __device__ __forceinline__ void tiled_prefill(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        PrefillShared<Tile, A_STRIDE>& staging, __nv_bfloat16* b_hi, __nv_bfloat16* b_lo,
        unsigned int block, Writeback writeback = Writeback{}) {
    using Shared = PrefillShared<Tile, A_STRIDE>;
    static_assert(__is_trivial(typename Shared::Activation), "A wrapper must be trivial");
    static_assert(__is_trivial(typename Shared::Result), "result wrapper must be trivial");
    static_assert(__is_trivially_constructible(Shared), "shared union must default-construct trivially");
    constexpr int kThreads = Tile::kWarps * 32;
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    // Quotient-and-remainder form: out_features + kCols - 1 could wrap for valid ABI sizes.
    const unsigned int output_tiles = out_features / Tile::kCols + (out_features % Tile::kCols != 0u);
    const unsigned int row_start = (block / output_tiles) * Tile::kRows;
    const unsigned int out_start = (block % output_tiles) * Tile::kCols;
    const Layout<BITS> w(weights, in_features, out_features);
    typename Leaf::Acc acc;
    Leaf::fill(acc);
    if (threadIdx.x == 0) new (static_cast<void*>(&staging)) Shared(typename Shared::ActivateA{});
    __syncthreads();
    __nv_bfloat16* a = staging.a.values;
    if ((reinterpret_cast<unsigned long long>(input) & 3ull) == 0ull) {
        CompactPrefetch<Tile, BITS> next;
        prefetch_compact_tile(next, input, w, rows, in_features, out_features,
                row_start, out_start, 0, threadIdx.x, warp, lane);
        for (unsigned int base = 0; base < in_features; base += kGroup) {
            stage_prefetched_activation<Tile, BITS, A_STRIDE>(a, next, threadIdx.x);
            stage_prefetched_weights<Tile, BITS, B_STRIDE>(b_hi, b_lo, next, warp, lane);
            __syncthreads();
            if (base + kGroup < in_features)
                prefetch_compact_tile(next, input, w, rows, in_features, out_features,
                        row_start, out_start, base + kGroup, threadIdx.x, warp, lane);
            Leaf::template consume<kGroup, A_STRIDE, B_STRIDE>(acc, a, b_hi, b_lo, warp);
            __syncthreads();
        }
    } else {
        for (unsigned int base = 0; base < in_features; base += kGroup) {
            stage_activation_tile<Tile::kRows, kThreads, A_STRIDE>(
                    a, input, rows, in_features, row_start, base, threadIdx.x);
            stage_weight_tile<BITS, Tile::kCols, Tile::kWarps, B_STRIDE>(
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
    writeback.template write<Tile::kRows, Tile::kCols, kThreads>(
            output, result, rows, out_features, row_start, out_start, threadIdx.x);
}

// 2 x 2 warps with one or two fragments per warp give 32 x 32 and 64 x 32 CTA
// tiles. Host dispatch in qwen_layer_ops.c sizes grids with the same constants.
using Prefill32 = WarpTile<2, 2, 1>;
using Prefill64 = WarpTile<2, 2, 2>;

}  // namespace q45
