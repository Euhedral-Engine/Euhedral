#include "strategies/scalar.cuh"
#include "strategies/decode.cuh"
#include "strategies/decode_wide.cuh"
#include "strategies/decode_contiguous.cuh"
#include "strategies/prefill.cuh"
#include "strategies/k32_prefill.cuh"
#include "pdl.cuh"

// Production prefill entry points consume BF16 hi weights only (one MMA per weight); each has an
// _exact twin that consumes hi + lo and reproduces the scalar reference bit for bit.
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
    euhedral_pdl_begin(); \
    q3::cooperative_decode<R>(input, weights, output, rows, in_features, out_features, scale_offset); \
}
EUHEDRAL_Q3_DECODE_KERNEL(1)
EUHEDRAL_Q3_DECODE_KERNEL(2)
EUHEDRAL_Q3_DECODE_KERNEL(4)
#undef EUHEDRAL_Q3_DECODE_KERNEL

// Single-row decode with contiguous lane ownership (strategies/decode_contiguous.cuh): each lane
// owns 32 whole codes. Not bitwise: FP32 accumulation is reordered relative to the exact oracle
// kernels below. 128 threads, 16 rows per CTA. Optional symbol: host dispatch selects it for the
// shapes in euhedral_q3_decode_contiguous_shape unless exact decode is selected.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_decode_contiguous(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features, unsigned long long scale_offset) {
    q3::contiguous_decode(input, weights, output, in_features, out_features, scale_offset);
}

// Single-row decode with warp-local wide streaming (strategies/decode_wide.cuh): every lane loads
// 16 contiguous weight bytes and each warp stages its own rows. Bitwise identical to
// euhedral_q3_decode_1. Optional symbol: host dispatch selects it for the shapes in
// euhedral_q3_decode_wide_shape with a 16-byte aligned weight base.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_decode_wide(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features, unsigned long long scale_offset) {
    __shared__ q3::DecodeWideShared shared;
    q3::wide_decode(input, weights, output, in_features, out_features, scale_offset, shared);
}

// 32 token rows x 32 outputs; general Q3 prefill route. Shared strides of 88 elements (176 bytes)
// keep the 16-byte ldmatrix rows conflict-free; residency drops from 6 to 5 CTA/SM.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    using Tile = q3::Prefill32;
    constexpr int A_STRIDE = 88;
    constexpr int B_STRIDE = 88;
    __shared__ q3::PrefillShared<Tile, A_STRIDE> staging;
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * B_STRIDE];
    __shared__ __align__(32) __nv_bfloat16 b_lo[8];
    q3::tiled_prefill<Tile, A_STRIDE, B_STRIDE, q3::MmaSyncLeaf<Tile, false, 1>>(input, weights, output, rows, in_features, out_features, scale_offset,
            staging, b_hi, b_lo);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill_exact(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    using Tile = q3::Prefill32;
    constexpr int A_STRIDE = 88;
    constexpr int B_STRIDE = 88;
    __shared__ q3::PrefillShared<Tile, A_STRIDE> staging;
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * B_STRIDE];
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * B_STRIDE];
    q3::tiled_prefill<Tile, A_STRIDE, B_STRIDE, q3::Q3_PREFILL_LEAF<Tile>>(input, weights, output, rows, in_features, out_features, scale_offset,
            staging, b_hi, b_lo);
}

// Same 32-row kernel with 104-element strides. It removes the same conflicts but lowers residency to
// 4 CTA/SM, which pays only where the grid is one or three row tiles (measured on the GDN output
// mixer). Optional symbol: host dispatch selects it for that shape only and otherwise uses the 88 kernel.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill_s104(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    using Tile = q3::Prefill32;
    constexpr int A_STRIDE = 104;
    constexpr int B_STRIDE = 104;
    __shared__ q3::PrefillShared<Tile, A_STRIDE> staging;
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * B_STRIDE];
    __shared__ __align__(32) __nv_bfloat16 b_lo[8];
    q3::tiled_prefill<Tile, A_STRIDE, B_STRIDE, q3::MmaSyncLeaf<Tile, false, 1>>(input, weights, output, rows, in_features, out_features, scale_offset,
            staging, b_hi, b_lo);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill_s104_exact(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    using Tile = q3::Prefill32;
    constexpr int A_STRIDE = 104;
    constexpr int B_STRIDE = 104;
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
    __shared__ __align__(32) __nv_bfloat16 b_lo[8];
    q3::tiled_prefill<Tile, A_STRIDE, B_STRIDE, q3::MmaPingPongLeaf<Tile, 1>>(input, weights, output, rows, in_features, out_features, scale_offset,
            staging, b_hi, b_lo);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill_64_exact(
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

// Exact operator/row shapes measured for the K32 compact-B-only schedule.
// Host AUTO selects this optional symbol only when its policy and load both pass.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill_64_k32_cb(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    __shared__ k32_probe::Shared storage;
    k32_probe::run<false, false, false, true, 1>(input, weights, output, rows, in_features,
            out_features, scale_offset, nullptr, storage);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill_64_k32_cb_exact(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    __shared__ k32_probe::Shared storage;
    k32_probe::run<false, false, false, true>(input, weights, output, rows, in_features,
            out_features, scale_offset, nullptr, storage);
}
