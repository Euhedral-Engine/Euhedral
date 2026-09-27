#include "strategies/scalar.cuh"
#include "strategies/decode.cuh"
#include "strategies/prefill.cuh"

namespace q3 {
// A is dead after the final K-step barrier; result is first written after it.
// Both arrays occupy the same bytes for Prefill32 and Prefill64.
template<class Tile>
union alignas(32) PrefillShared {
    __nv_bfloat16 a[Tile::kRows * kGroup];
    float result[Tile::kRows * Tile::kCols];
};
}  // namespace q3

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
    __shared__ q3::PrefillShared<Tile> staging;
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * q3::kGroup];
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * q3::kGroup];
    q3::tiled_prefill<Tile>(input, weights, output, rows, in_features, out_features, scale_offset,
            staging.a, b_hi, b_lo, staging.result);
}

// 64 token rows x 32 outputs; shape-gated route for the large MLP projections.
// Reuses each decoded weight tile across twice as many rows. Optional symbol:
// host dispatch falls back to euhedral_q3_prefill when it is absent.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill_64(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    using Tile = q3::Prefill64;
    __shared__ q3::PrefillShared<Tile> staging;
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * q3::kGroup];
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * q3::kGroup];
    q3::tiled_prefill<Tile>(input, weights, output, rows, in_features, out_features, scale_offset,
            staging.a, b_hi, b_lo, staging.result);
}
