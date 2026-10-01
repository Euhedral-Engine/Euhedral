#include "strategies/cluster_reuse.cuh"

// Explicit opt-in kernels only: production dispatch and launch geometry are unchanged.
// Every CTA in a fixed-size cluster participates, including zero-filled edge CTAs.
#define EUHEDRAL_Q3_CLUSTER_KERNEL(ROWS, TILE, CM, CN) \
extern "C" __global__ __launch_bounds__(128) __cluster_dims__(CN, CM, 1) \
void euhedral_q3_cluster_##ROWS##_##CM##x##CN( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features, unsigned long long scale_offset) { \
    using Tile = q3::TILE; \
    __shared__ __align__(32) __nv_bfloat16 a[Tile::kRows * q3::kGroup]; \
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * q3::kGroup]; \
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * q3::kGroup]; \
    __shared__ __align__(32) float result[Tile::kRows * Tile::kCols]; \
    q3::cluster_prefill<Tile, CM, CN>(input, weights, output, rows, in_features, out_features, \
            scale_offset, a, b_hi, b_lo, result); \
}
#define EUHEDRAL_Q3_CLUSTER_TILES(ROWS, TILE) \
    EUHEDRAL_Q3_CLUSTER_KERNEL(ROWS, TILE, 1, 1) \
    EUHEDRAL_Q3_CLUSTER_KERNEL(ROWS, TILE, 2, 1) \
    EUHEDRAL_Q3_CLUSTER_KERNEL(ROWS, TILE, 4, 1) \
    EUHEDRAL_Q3_CLUSTER_KERNEL(ROWS, TILE, 1, 2) \
    EUHEDRAL_Q3_CLUSTER_KERNEL(ROWS, TILE, 1, 4) \
    EUHEDRAL_Q3_CLUSTER_KERNEL(ROWS, TILE, 2, 2)
EUHEDRAL_Q3_CLUSTER_TILES(32, Prefill32)
EUHEDRAL_Q3_CLUSTER_TILES(64, Prefill64)
#undef EUHEDRAL_Q3_CLUSTER_TILES
#undef EUHEDRAL_Q3_CLUSTER_KERNEL
