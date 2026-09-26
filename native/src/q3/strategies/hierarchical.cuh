#pragma once
#include "prefill.cuh"
#include <cooperative_groups.h>

namespace q3 {

// Pure logical M/N node: owns one output region and identifies the child
// producing each reusable K tile. Physical CTA and DSM mapping live below.
template<class Tile, int CN>
struct OutputRegion {
    const unsigned int m, n, row, col;
    __device__ __forceinline__ OutputRegion(unsigned int m_, unsigned int n_,
            unsigned int row_, unsigned int col_) : m(m_), n(n_), row(row_), col(col_) {}
    __device__ __forceinline__ bool produces_a() const { return n == 0; }
    __device__ __forceinline__ bool produces_b() const { return m == 0; }
};

// Warp-scoped views: each points at the first row/column of one 16x64
// fragment band in this K group; the leaf's legal indexing is confined to
// 16 rows/columns and the current 16-wide K slice. These are logical bounds,
// not hardware-enforced pointer bounds.
struct AFragmentView {
    static constexpr int kRows = 16, kK = kGroup;
    const __nv_bfloat16* first_row;
    bool borrowed;
};
struct BFragmentView {
    static constexpr int kColumns = 16, kK = kGroup;
    const __nv_bfloat16* first_hi;
    const __nv_bfloat16* first_lo;
    bool borrowed;
};
template<int M_FRAGS>
struct KAccess {
    AFragmentView a[M_FRAGS];
    BFragmentView b;
};

// CTA-local slots are narrowed again before reaching a warp leaf: each warp
// can stage only its own 16x16 A and hi/lo B execution fragments.
template<int M_FRAGS>
struct WarpScratch {
    __nv_bfloat16* a[M_FRAGS];
    __nv_bfloat16* hi;
    __nv_bfloat16* lo;
};

// One hardware binding for OutputRegion/KAccess. Different mappings may
// supply the same leaf contract without changing the logical region or WMMA.
template<class Tile, int CM, int CN>
struct ClusterPlacement {
    const cooperative_groups::cluster_group cluster;
    __device__ __forceinline__ ClusterPlacement() : cluster(cooperative_groups::this_cluster()) {}
    __device__ __forceinline__ OutputRegion<Tile, CN> region() const {
        return OutputRegion<Tile, CN>(cluster.block_index().y, cluster.block_index().x,
                blockIdx.y * Tile::kRows, blockIdx.x * Tile::kCols);
    }
    __device__ __forceinline__ KAccess<Tile::kFrags> borrow(const OutputRegion<Tile, CN>& r,
            __nv_bfloat16* a, __nv_bfloat16* hi, __nv_bfloat16* lo, unsigned int warp) const {
        const __nv_bfloat16* owner_a = r.produces_a() ? a : cluster.map_shared_rank(a, r.m * CN);
        const __nv_bfloat16* owner_hi = r.produces_b() ? hi : cluster.map_shared_rank(hi, r.n);
        const __nv_bfloat16* owner_lo = r.produces_b() ? lo : cluster.map_shared_rank(lo, r.n);
        KAccess<Tile::kFrags> access;
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; ++m)
            access.a[m] = {owner_a + Tile::row(warp, m) * kGroup, !r.produces_a()};
        access.b = {owner_hi + Tile::col(warp) * kGroup,
                    owner_lo + Tile::col(warp) * kGroup, !r.produces_b()};
        return access;
    }
    __device__ __forceinline__ WarpScratch<Tile::kFrags> scratch(
            __nv_bfloat16* a, __nv_bfloat16* hi, __nv_bfloat16* lo, unsigned int warp) const {
        WarpScratch<Tile::kFrags> view;
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; ++m)
            view.a[m] = a + (warp * Tile::kFrags + m) * 256;
        view.hi = hi + warp * 256;
        view.lo = lo + warp * 256;
        return view;
    }
};

// A producer owns a whole execution tile for the current K region. Borrowers
// receive a remote read-only capability, not a copy of the producer's tile.
// The source lifetime ends only after the second cluster barrier.
template<class Tile, int CN>
static __device__ __forceinline__ void stage_owned_k(
        const OutputRegion<Tile, CN>& region, __nv_bfloat16* a,
        __nv_bfloat16* b_hi, __nv_bfloat16* b_lo, const unsigned short* input,
        const Layout& packed, unsigned int rows, unsigned int width, unsigned int outputs,
        unsigned int base, unsigned int warp, unsigned int lane) {
    if (region.produces_a())
        stage_activation_tile<Tile::kRows, kGroup, Tile::kWarps * 32>(
                a, input, rows, width, region.row, base, threadIdx.x);
    if (region.produces_b())
        stage_weight_tile<Tile::kCols, Tile::kWarps>(
                b_hi, b_lo, packed, region.col, outputs, base, warp, lane);
    __syncthreads();
}

// Keep the DSM-to-warp-fragment transfer as a small device boundary: the
// pinned NVRTC compiler crashes when both remote hi/lo transfers and their
// WMMA consumers are inlined into one function.
static __device__ __noinline__ void stage_remote_b_fragment(
        __nv_bfloat16* scratch, const __nv_bfloat16* first_column,
        unsigned int k, unsigned int lane) {
    for (unsigned int i = lane; i < 256; i += 32)
        scratch[i] = first_column[(i / 16) * kGroup + k + i % 16];
    __syncwarp();
}

// Warp leaf: only a current 16x16 fragment crosses from a remote owner into
// CTA-local shared memory for WMMA. Producer CTAs consume their own shared tile
// directly. Borrowers overlay warp-private scratch on the otherwise-unused
// producer slots (A for n!=0, B for m!=0), never a full CTA-private A/B copy.
// Rows of A and columns of B stay in the same WMMA order as tiled_prefill.
template<class Tile>
static __device__ __forceinline__ void consume_borrowed_k(
        Accumulators<Tile::kFrags>& acc, const KAccess<Tile::kFrags>& access,
        const WarpScratch<Tile::kFrags>& scratch, unsigned int lane) {
    const bool remote_a = access.a[0].borrowed, remote_b = access.b.borrowed;
    namespace wm = nvcuda::wmma;
    #pragma unroll
    for (unsigned int k = 0; k < kGroup; k += 16) {
        wm::fragment<wm::matrix_a, 16, 16, 16, __nv_bfloat16, wm::row_major> af[Tile::kFrags];
        wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::col_major> bf;
        if (remote_a) {
            #pragma unroll
            for (int m = 0; m < Tile::kFrags; ++m) {
                __nv_bfloat16* fragment = scratch.a[m];
                for (unsigned int i = lane; i < 256; i += 32)
                    fragment[i] = access.a[m].first_row[(i / 16) * kGroup + k + i % 16];
            }
            __syncwarp();
        }
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; ++m)
            wm::load_matrix_sync(af[m], remote_a
                    ? scratch.a[m]
                    : access.a[m].first_row + k,
                    remote_a ? 16 : kGroup);

        if (remote_b) {
            stage_remote_b_fragment(scratch.hi, access.b.first_hi, k, lane);
            stage_remote_b_fragment(scratch.lo, access.b.first_lo, k, lane);
        }
        wm::load_matrix_sync(bf, remote_b ? scratch.hi
                : access.b.first_hi + k, remote_b ? 16 : kGroup);
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; ++m) wm::mma_sync(acc.frag[m], af[m], bf, acc.frag[m]);

        wm::load_matrix_sync(bf, remote_b ? scratch.lo
                : access.b.first_lo + k, remote_b ? 16 : kGroup);
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; ++m) wm::mma_sync(acc.frag[m], af[m], bf, acc.frag[m]);
    }
}

// Warp partial C flows upward to a CTA-owned result tile, then through the
// existing bounds-aware BF16 output transform. No global intermediate C tile.
template<class Tile>
static __device__ __forceinline__ void aggregate_output(
        Accumulators<Tile::kFrags>& acc, float* result, unsigned short* output,
        unsigned int rows, unsigned int outputs, unsigned int row, unsigned int col,
        unsigned int warp) {
    store_accumulators<Tile>(result, acc, warp);
    __syncthreads();
    write_output_tile<Tile::kRows, Tile::kCols, Tile::kWarps * 32>(
            output, result, rows, outputs, row, col, threadIdx.x);
}

template<class Tile, int CM, int CN>
static __device__ __forceinline__ void hierarchical_prefill(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int width, unsigned int outputs, unsigned long long scale_offset,
        __nv_bfloat16* a, __nv_bfloat16* b_hi, __nv_bfloat16* b_lo, float* result) {
    static_assert(CM >= 1 && CN >= 1 && CM * CN <= 8, "cluster size must fit the target");
    static_assert(Tile::kWarps * Tile::kFrags * 256 <= Tile::kRows * kGroup,
            "borrowed A fragments must fit the unowned A slot");
    static_assert(Tile::kWarps * 256 <= Tile::kCols * kGroup,
            "borrowed B fragments must fit the unowned B slot");
    const ClusterPlacement<Tile, CM, CN> placement;
    const auto region = placement.region();
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const Layout packed(weights, width, scale_offset);
    Accumulators<Tile::kFrags> acc;
    acc.fill();
    for (unsigned int base = 0; base < width; base += kGroup) {
        stage_owned_k(region, a, b_hi, b_lo, input, packed, rows, width, outputs, base, warp, lane);
        placement.cluster.sync();
        const KAccess<Tile::kFrags> access = placement.borrow(region, a, b_hi, b_lo, warp);
        const WarpScratch<Tile::kFrags> scratch = placement.scratch(a, b_hi, b_lo, warp);
        consume_borrowed_k<Tile>(acc, access, scratch, lane);
        placement.cluster.sync();
    }
    aggregate_output<Tile>(acc, result, output, rows, outputs, region.row, region.col, warp);
}

}  // namespace q3
