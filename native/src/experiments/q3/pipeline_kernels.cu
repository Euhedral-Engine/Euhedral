#include "strategies/pipelined_fragments.cuh"

// Opt-in only; each CTA retains four MMA warps and adds one producer per branch.
// Nullable observations: [CTA][generation][4 branches * 6 + 4 consumers * 5].
// Branch: useful-start, accepted, pre-publish, DSM fragments, generation, slot.
// Consumer: acquired, MMA-done, pre-release, A address, B address.
#define EUHEDRAL_Q3_PIPELINE_KERNEL(ROWS, TILE, CM, CN) \
extern "C" __global__ __launch_bounds__(256) __cluster_dims__(CN, CM, 1) \
void euhedral_q3_pipeline_##ROWS##_##CM##x##CN( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features, unsigned long long scale_offset, \
        unsigned long long* observations) { \
    using Tile = q3::TILE; \
    using Node = q3::PipelinedFragmentNode<Tile>; \
    __shared__ __align__(32) __nv_bfloat16 a[Tile::kRows * q3::kGroup * Node::kSlots]; \
    __shared__ __align__(32) __nv_bfloat16 hi[Tile::kCols * q3::kGroup * Node::kSlots]; \
    __shared__ __align__(32) __nv_bfloat16 lo[Tile::kCols * q3::kGroup * Node::kSlots]; \
    __shared__ __align__(32) float result[Tile::kRows * Tile::kCols]; \
    __shared__ q3::FragmentSlot states[Node::kBranches * Node::kSlots]; \
    const Node node{states, a, hi, lo}; \
    q3::pipelined_fragment_prefill<Tile, CM, CN>(input, weights, output, rows, in_features, out_features, \
            scale_offset, node, result, observations); \
}
#define EUHEDRAL_Q3_PIPELINE_TILES(ROWS, TILE) \
    EUHEDRAL_Q3_PIPELINE_KERNEL(ROWS, TILE, 1, 1) \
    EUHEDRAL_Q3_PIPELINE_KERNEL(ROWS, TILE, 2, 1) \
    EUHEDRAL_Q3_PIPELINE_KERNEL(ROWS, TILE, 4, 1) \
    EUHEDRAL_Q3_PIPELINE_KERNEL(ROWS, TILE, 1, 2) \
    EUHEDRAL_Q3_PIPELINE_KERNEL(ROWS, TILE, 1, 4) \
    EUHEDRAL_Q3_PIPELINE_KERNEL(ROWS, TILE, 2, 2)
EUHEDRAL_Q3_PIPELINE_TILES(32, Prefill32)
EUHEDRAL_Q3_PIPELINE_TILES(64, Prefill64)
#undef EUHEDRAL_Q3_PIPELINE_TILES
#undef EUHEDRAL_Q3_PIPELINE_KERNEL
