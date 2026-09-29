#include "strategies/decode.cuh"
#include "strategies/prefill.cuh"
#include "attention_cache.cuh"

// Q4/Q5 kernels, structured like the Q3 module: format layout and code
// unpacking are Q4/Q5-specific; the cooperative decode, compact K prefetch,
// padded shared strides and MMA leaves follow the Q3 designs. Every route keeps
// the original per-element FP32 order; the explicit MMA leaves are tested
// bitwise against the WMMA reference leaf on the target.
#ifndef Q45_PREFILL_LEAF
#define Q45_PREFILL_LEAF MmaSyncLeaf
#endif
#ifndef Q45_PREFILL64_LEAF
#define Q45_PREFILL64_LEAF MmaPingPongLeaf
#endif

#define EUHEDRAL_Q45_DECODE_KERNEL(B, R) \
extern "C" __global__ __launch_bounds__(128) void euhedral_q##B##_decode_##R( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features) { \
    q45::cooperative_decode<B, R>(input, weights, output, rows, in_features, out_features); \
}
EUHEDRAL_Q45_DECODE_KERNEL(4, 1)
EUHEDRAL_Q45_DECODE_KERNEL(4, 2)
EUHEDRAL_Q45_DECODE_KERNEL(4, 4)
EUHEDRAL_Q45_DECODE_KERNEL(5, 1)
EUHEDRAL_Q45_DECODE_KERNEL(5, 2)
EUHEDRAL_Q45_DECODE_KERNEL(5, 4)
#undef EUHEDRAL_Q45_DECODE_KERNEL

// Prefill entry points: _prefill is the 32 x 32 tile, _prefill_64 the 64 x 32
// tile that reuses each decoded weight tile across twice as many rows.
#define EUHEDRAL_Q45_PREFILL_KERNEL(B, NAME, TILE, A_STRIDE, B_STRIDE, LEAF) \
extern "C" __global__ __launch_bounds__(128) void euhedral_q##B##_##NAME( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features) { \
    using Tile = q45::TILE; \
    __shared__ q45::PrefillShared<Tile, A_STRIDE> staging; \
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * B_STRIDE]; \
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * B_STRIDE]; \
    q45::tiled_prefill<B, Tile, A_STRIDE, B_STRIDE, q45::LEAF<Tile>>( \
            input, weights, output, rows, in_features, out_features, staging, b_hi, b_lo); \
}
EUHEDRAL_Q45_PREFILL_KERNEL(4, prefill, Prefill32, 80, 80, Q45_PREFILL_LEAF)
EUHEDRAL_Q45_PREFILL_KERNEL(5, prefill, Prefill32, 80, 80, Q45_PREFILL_LEAF)
EUHEDRAL_Q45_PREFILL_KERNEL(4, prefill_64, Prefill64, 80, 64, Q45_PREFILL64_LEAF)
EUHEDRAL_Q45_PREFILL_KERNEL(5, prefill_64, Prefill64, 80, 64, Q45_PREFILL64_LEAF)
#undef EUHEDRAL_Q45_PREFILL_KERNEL
