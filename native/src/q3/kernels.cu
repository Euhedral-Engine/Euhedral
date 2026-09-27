#include "strategies/scalar.cuh"
#include "strategies/decode.cuh"
#include "strategies/prefill.cuh"

// MMA leaves per CTA AUTO tile. WmmaLeaf remains the portable reference; the
// explicit leaves' FP32 accumulators are tested bitwise against it (see
// mma_leaf.cuh).
// The 32-row tile is fastest with the explicit ldmatrix/mma.sync leaf; the
// 64-row tile additionally benefits from fragment ping-pong, except the down
// projection at 64 rows, which AUTO routes to euhedral_q3_prefill_64_wmma.
#ifndef Q3_PREFILL_LEAF
#define Q3_PREFILL_LEAF MmaSyncLeaf
#endif
#ifndef Q3_PREFILL64_LEAF
#define Q3_PREFILL64_LEAF MmaPingPongLeaf
#endif

extern "C" __global__ __launch_bounds__(128) void euhedral_q3_linear_bf16(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    __shared__ float partial[128];
    q3::scalar_reference(input, weights, output, rows, in_features, out_features, scale_offset, partial);
}

#define EUHEDRAL_Q3_DECODE_KERNEL(R) \
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_decode_##R( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features, unsigned long long scale_offset) { \
    q3::cooperative_decode<R>(input, weights, output, rows, in_features, out_features, scale_offset); \
}
EUHEDRAL_Q3_DECODE_KERNEL(1)
EUHEDRAL_Q3_DECODE_KERNEL(2)
EUHEDRAL_Q3_DECODE_KERNEL(4)
#undef EUHEDRAL_Q3_DECODE_KERNEL

// 32 token rows x 32 outputs; general Q3 prefill route.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    using Tile = q3::Prefill32;
    constexpr int A_STRIDE = 80;
    constexpr int B_STRIDE = Tile::kRows == 32 ? 80 : 64;
    __shared__ q3::PrefillShared<Tile, A_STRIDE> staging;
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * B_STRIDE];
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * B_STRIDE];
    q3::tiled_prefill<Tile, A_STRIDE, B_STRIDE, q3::Q3_PREFILL_LEAF<Tile>>(input, weights, output, rows, in_features, out_features, scale_offset,
            staging, b_hi, b_lo);
}

// 64 token rows x 32 outputs; shape-gated route for the large MLP projections.
// Reuses each decoded weight tile across twice as many rows. Optional symbol:
// host dispatch falls back to euhedral_q3_prefill when it is absent.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill_64(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    using Tile = q3::Prefill64;
    constexpr int A_STRIDE = 80;
    constexpr int B_STRIDE = Tile::kRows == 32 ? 80 : 64;
    __shared__ q3::PrefillShared<Tile, A_STRIDE> staging;
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * B_STRIDE];
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * B_STRIDE];
    q3::tiled_prefill<Tile, A_STRIDE, B_STRIDE, q3::Q3_PREFILL64_LEAF<Tile>>(input, weights, output, rows, in_features, out_features, scale_offset,
            staging, b_hi, b_lo);
}

// 64-row tile with the WMMA reference leaf. Optional symbol: AUTO routes shapes
// where the explicit leaf measured slower here (q3_prefill_policy.h).
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill_64_wmma(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    using Tile = q3::Prefill64;
    constexpr int A_STRIDE = 80;
    constexpr int B_STRIDE = Tile::kRows == 32 ? 80 : 64;
    __shared__ q3::PrefillShared<Tile, A_STRIDE> staging;
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * B_STRIDE];
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * B_STRIDE];
    q3::tiled_prefill<Tile, A_STRIDE, B_STRIDE, q3::WmmaLeaf<Tile>>(input, weights, output, rows, in_features, out_features, scale_offset,
            staging, b_hi, b_lo);
}
