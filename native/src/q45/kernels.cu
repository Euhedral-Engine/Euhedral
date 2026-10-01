#include "strategies/decode.cuh"
#include "strategies/decode_wide.cuh"
#include "strategies/decode_contiguous.cuh"
#include "strategies/prefill.cuh"
#include "attention_cache.cuh"
#include "pdl.cuh"

// Q4/Q5 kernels, structured like the Q3 module: format layout and code
// unpacking are Q4/Q5-specific; the cooperative decode, compact K prefetch,
// padded shared strides and MMA leaves follow the Q3 designs. Every route keeps
// the original per-element FP32 order; the explicit MMA leaves are tested
// bitwise against the WMMA reference leaf on the target.
// P: B operand parts. Production entry points consume BF16 hi only (P = 1); the _exact twins
// consume hi + lo (P = 2), reproducing the original kernels bit for bit.
namespace q45 {
template<class Tile, int P> using PrefillLeaf = q3::MmaSyncLeaf<Tile, false, P>;
template<class Tile, int P> using Prefill64Leaf = q3::MmaPingPongLeaf<Tile, P>;
}

#define EUHEDRAL_Q45_DECODE_KERNEL(B, R) \
extern "C" __global__ __launch_bounds__(128) void euhedral_q##B##_decode_##R( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features) { \
    euhedral_pdl_begin(); \
    q45::cooperative_decode<B, R>(input, weights, output, rows, in_features, out_features); \
}
EUHEDRAL_Q45_DECODE_KERNEL(4, 1)
EUHEDRAL_Q45_DECODE_KERNEL(4, 2)
EUHEDRAL_Q45_DECODE_KERNEL(4, 4)
EUHEDRAL_Q45_DECODE_KERNEL(5, 1)
EUHEDRAL_Q45_DECODE_KERNEL(5, 2)
EUHEDRAL_Q45_DECODE_KERNEL(5, 4)
#undef EUHEDRAL_Q45_DECODE_KERNEL

// Single-row decode with warp-local wide streaming (strategies/decode_wide.cuh). Bitwise identical
// to euhedral_q*_decode_1. Optional symbols: host dispatch selects them for the shapes in
// euhedral_q45_decode_wide_shape with a 16-byte aligned weight base.
#define EUHEDRAL_Q45_DECODE_WIDE_KERNEL(B) \
extern "C" __global__ __launch_bounds__(128) void euhedral_q##B##_decode_wide( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features) { \
    __shared__ q45::DecodeWideShared<B> shared; \
    q45::wide_decode<B>(input, weights, output, in_features, out_features, shared); \
}
EUHEDRAL_Q45_DECODE_WIDE_KERNEL(4)
EUHEDRAL_Q45_DECODE_WIDE_KERNEL(5)
#undef EUHEDRAL_Q45_DECODE_WIDE_KERNEL

// Single-row decode with contiguous lane ownership (strategies/decode_contiguous.cuh); not bitwise.
// 128 threads, 8 rows per CTA. Host dispatch selects it unless exact numerics are selected.
#define EUHEDRAL_Q45_DECODE_CONTIGUOUS_KERNEL(B) \
extern "C" __global__ __launch_bounds__(128) void euhedral_q##B##_decode_contiguous( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features) { \
    q45::contiguous_decode<B>(input, weights, output, in_features, out_features); \
}
EUHEDRAL_Q45_DECODE_CONTIGUOUS_KERNEL(4)
EUHEDRAL_Q45_DECODE_CONTIGUOUS_KERNEL(5)
#undef EUHEDRAL_Q45_DECODE_CONTIGUOUS_KERNEL

// Shared strides of 72 elements (144 bytes) spread the eight ldmatrix rows of
// one phase across distinct banks; the earlier 64 and 80 element strides made
// the fragment loads conflict.
// Prefill entry points: _prefill is the 32 x 32 tile, _prefill_64 the 64 x 32
// tile that reuses each decoded weight tile across twice as many rows.
#define EUHEDRAL_Q45_PREFILL_KERNEL(B, NAME, TILE, A_STRIDE, B_STRIDE, LEAF, P) \
extern "C" __global__ __launch_bounds__(128) void euhedral_q##B##_##NAME( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features) { \
    using Tile = q45::TILE; \
    __shared__ q45::PrefillShared<Tile, A_STRIDE> staging; \
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * B_STRIDE]; \
    __shared__ __align__(32) __nv_bfloat16 b_lo[P > 1 ? Tile::kCols * B_STRIDE : 8]; \
    q45::tiled_prefill<B, Tile, A_STRIDE, B_STRIDE, q45::LEAF<Tile, P>>( \
            input, weights, output, rows, in_features, out_features, staging, b_hi, b_lo, blockIdx.x); \
}
EUHEDRAL_Q45_PREFILL_KERNEL(4, prefill, Prefill32, 72, 72, PrefillLeaf, 1)
EUHEDRAL_Q45_PREFILL_KERNEL(5, prefill, Prefill32, 72, 72, PrefillLeaf, 1)
EUHEDRAL_Q45_PREFILL_KERNEL(4, prefill_64, Prefill64, 72, 72, Prefill64Leaf, 1)
EUHEDRAL_Q45_PREFILL_KERNEL(5, prefill_64, Prefill64, 72, 72, Prefill64Leaf, 1)
EUHEDRAL_Q45_PREFILL_KERNEL(4, prefill_exact, Prefill32, 72, 72, PrefillLeaf, 2)
EUHEDRAL_Q45_PREFILL_KERNEL(5, prefill_exact, Prefill32, 72, 72, PrefillLeaf, 2)
EUHEDRAL_Q45_PREFILL_KERNEL(4, prefill_64_exact, Prefill64, 72, 72, Prefill64Leaf, 2)
EUHEDRAL_Q45_PREFILL_KERNEL(5, prefill_64_exact, Prefill64, 72, 72, Prefill64Leaf, 2)
#undef EUHEDRAL_Q45_PREFILL_KERNEL

// One launch for a Q4 and a Q5 projection of the same activations: the first
// Q4 tile blocks run the Q4 weights, the rest run the Q5 weights. Each block is
// the same 64-row tile kernel as euhedral_q4_prefill_64 / euhedral_q5_prefill_64,
// so the outputs are bitwise identical to two launches. It removes the idle
// tail of the small Q4 grid; the host uses it only where that wins.
namespace q45 {
template<int P>
static __device__ __forceinline__ void prefill_64_grouped(
        const unsigned short* input, const unsigned char* q4_weights, unsigned short* q4_output,
        const unsigned char* q5_weights, unsigned short* q5_output, unsigned int rows,
        unsigned int in_features, unsigned int q4_features, unsigned int q5_features,
        PrefillShared<Prefill64, 72>& staging, __nv_bfloat16* b_hi, __nv_bfloat16* b_lo) {
    using Tile = Prefill64;
    const unsigned int row_tiles = rows / Tile::kRows + (rows % Tile::kRows != 0u);
    const unsigned int q4_blocks = row_tiles * (q4_features / Tile::kCols + (q4_features % Tile::kCols != 0u));
    if (blockIdx.x < q4_blocks)
        tiled_prefill<4, Tile, 72, 72, Prefill64Leaf<Tile, P>>(
                input, q4_weights, q4_output, rows, in_features, q4_features, staging, b_hi, b_lo, blockIdx.x);
    else
        tiled_prefill<5, Tile, 72, 72, Prefill64Leaf<Tile, P>>(
                input, q5_weights, q5_output, rows, in_features, q5_features, staging, b_hi, b_lo,
                blockIdx.x - q4_blocks);
}
}  // namespace q45
#define EUHEDRAL_Q45_GROUPED(NAME, P) \
extern "C" __global__ __launch_bounds__(128) void NAME( \
        const unsigned short* input, const unsigned char* q4_weights, unsigned short* q4_output, \
        const unsigned char* q5_weights, unsigned short* q5_output, unsigned int rows, \
        unsigned int in_features, unsigned int q4_features, unsigned int q5_features) { \
    __shared__ q45::PrefillShared<q45::Prefill64, 72> staging; \
    __shared__ __align__(32) __nv_bfloat16 b_hi[q45::Prefill64::kCols * 72]; \
    __shared__ __align__(32) __nv_bfloat16 b_lo[P > 1 ? q45::Prefill64::kCols * 72 : 8]; \
    q45::prefill_64_grouped<P>(input, q4_weights, q4_output, q5_weights, q5_output, rows, in_features, \
            q4_features, q5_features, staging, b_hi, b_lo); \
}
EUHEDRAL_Q45_GROUPED(euhedral_q45_prefill_64_grouped, 1)
EUHEDRAL_Q45_GROUPED(euhedral_q45_prefill_64_grouped_exact, 2)
#undef EUHEDRAL_Q45_GROUPED
