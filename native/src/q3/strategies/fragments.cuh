#pragma once
#include "hierarchical.cuh"

namespace q3 {

struct KGeneration {
    unsigned int value;
    unsigned int slot;
};
struct PublishedK {
    unsigned int value;
    unsigned int slot;
};
template<int M_FRAGS>
struct FragmentLease {
    KAccess<M_FRAGS> access;
    unsigned int generation;
    unsigned int slot;
};
struct ClusterKRegion {
    const __nv_bfloat16* a;
    const __nv_bfloat16* hi;
    const __nv_bfloat16* lo;
    bool remote_a;
    bool remote_b;
};
// A branch node owns exactly one reusable A fragment band: M_FRAGS 16x64
// fragments. Its consumers are the warp row's N siblings, which borrow read-only.
template<class Tile>
struct AFragmentNode {
    static constexpr int kNBranches = Tile::kCols / 16;
    __nv_bfloat16* storage;
    unsigned int branch;

    __device__ __forceinline__ unsigned int accept(KGeneration generation,
            const __nv_bfloat16* source, bool remote, unsigned int lane_thread) const {
        constexpr unsigned int kThreads = Tile::kWarps * 32;
        const unsigned int slot_offset = generation.slot * Tile::kRows * kGroup;
        unsigned int fetched = 0;
        if (remote || generation.slot != 0) {
            #pragma unroll
            for (int m = 0; m < Tile::kFrags; ++m) {
                const unsigned int row = Tile::row(branch * kNBranches, m);
                for (unsigned int i = lane_thread; i < 16 * kGroup; i += kThreads)
                    storage[slot_offset + row * kGroup + i] = source[row * kGroup + i];
                if (remote) ++fetched; // one logical DSM fragment read for this M subfragment
            }
        }
        return fetched;
    }

    __device__ __forceinline__ AFragmentView borrow(KGeneration generation, int m) const {
        const unsigned int offset = generation.slot * Tile::kRows * kGroup
                + Tile::row(branch * kNBranches, m) * kGroup;
        return {storage + offset, false};
    }
};

// A branch node owns one 16x64 high/low B execution fragment. Its consumers
// are the warp column's M siblings; each borrows the same immutable view.
template<class Tile>
struct BFragmentNode {
    __nv_bfloat16* hi_storage;
    __nv_bfloat16* lo_storage;
    unsigned int branch;

    __device__ __forceinline__ unsigned int accept(KGeneration generation,
            const __nv_bfloat16* source_hi, const __nv_bfloat16* source_lo,
            bool remote, unsigned int lane_thread) const {
        constexpr unsigned int kThreads = Tile::kWarps * 32;
        const unsigned int offset = generation.slot * Tile::kCols * kGroup
                + branch * 16 * kGroup;
        if (remote || generation.slot != 0) {
            for (unsigned int i = lane_thread; i < 16 * kGroup; i += kThreads) {
                const unsigned int source_offset = Tile::col(branch) * kGroup + i;
                hi_storage[offset + i] = source_hi[source_offset];
                lo_storage[offset + i] = source_lo[source_offset];
            }
        }
        return remote ? 1u : 0u;
    }

    __device__ __forceinline__ BFragmentView borrow(KGeneration generation) const {
        const unsigned int offset = generation.slot * Tile::kCols * kGroup
                + branch * 16 * kGroup;
        return {hi_storage + offset, lo_storage + offset, false};
    }
};

// Lifecycle owner and router. Fragment storage is partitioned into branch-sized
// regions; no branch node owns unrelated siblings' fragments.
template<class Tile>
struct FragmentNode {
    static constexpr int kSlots = 1;
    static constexpr int kNBranches = Tile::kCols / 16;
    static constexpr int kMBranches = Tile::kWarps / kNBranches;

    __nv_bfloat16* a;
    __nv_bfloat16* hi;
    __nv_bfloat16* lo;
    unsigned long long* published_generation;

    __device__ __forceinline__ static unsigned long long stamp(unsigned int generation, unsigned int slot) {
        return (static_cast<unsigned long long>(generation) << 32) | slot;
    }
    __device__ __forceinline__ AFragmentNode<Tile> a_branch(unsigned int branch) const {
        return {a, branch};
    }
    __device__ __forceinline__ BFragmentNode<Tile> b_branch(unsigned int branch) const {
        return {hi, lo, branch};
    }
    __device__ __forceinline__ void begin(KGeneration generation) const {
        if (threadIdx.x == 0) *published_generation = ~stamp(generation.value, generation.slot);
        __syncthreads();
    }
    __device__ __forceinline__ PublishedK publish(KGeneration generation) const {
        __syncthreads();
        if (threadIdx.x == 0) *published_generation = stamp(generation.value, generation.slot);
        __syncthreads();
        return {generation.value, generation.slot};
    }
    __device__ __forceinline__ FragmentLease<Tile::kFrags> acquire(
            PublishedK published, unsigned int warp) const {
        if (*published_generation != stamp(published.value, published.slot)) asm volatile("trap;");
        KGeneration generation{published.value, published.slot};
        KAccess<Tile::kFrags> access;
        const AFragmentNode<Tile> a_node = a_branch(warp / kNBranches);
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; ++m)
            access.a[m] = a_node.borrow(generation, m);
        access.b = b_branch(warp % kNBranches).borrow(generation);
        return {access, published.value, published.slot};
    }
    __device__ __forceinline__ void release(const FragmentLease<Tile::kFrags>& lease) const {
        __syncthreads();
        if (threadIdx.x == 0) *published_generation = ~stamp(lease.generation, lease.slot);
        __syncthreads();
    }
};

// The placement routes one cluster-owned K tile to the intermediate node.
// Logical fragment branches remain defined by Tile; physical DSM ranks do not.
template<class Tile, int CM, int CN>
struct ClusterFragmentRoute {
    const ClusterPlacement<Tile, CM, CN>& placement;
    __device__ __forceinline__ ClusterKRegion accept(
            const OutputRegion<Tile, CN>& region, __nv_bfloat16* a,
            __nv_bfloat16* hi, __nv_bfloat16* lo) const {
        return {
            region.produces_a() ? a : placement.cluster.map_shared_rank(a, region.m * CN),
            region.produces_b() ? hi : placement.cluster.map_shared_rank(hi, region.n),
            region.produces_b() ? lo : placement.cluster.map_shared_rank(lo, region.n),
            !region.produces_a(), !region.produces_b()
        };
    }
};

template<class Tile>
static __device__ __forceinline__ void consume_fragment_node(
        Accumulators<Tile::kFrags>& acc, const FragmentLease<Tile::kFrags>& lease) {
    namespace wm = nvcuda::wmma;
    const auto& access = lease.access;
    #pragma unroll
    for (unsigned int k = 0; k < kGroup; k += 16) {
        wm::fragment<wm::matrix_a, 16, 16, 16, __nv_bfloat16, wm::row_major> af[Tile::kFrags];
        wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::col_major> bf;
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; ++m)
            wm::load_matrix_sync(af[m], access.a[m].first_row + k, kGroup);
        wm::load_matrix_sync(bf, access.b.first_hi + k, kGroup);
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; ++m) wm::mma_sync(acc.frag[m], af[m], bf, acc.frag[m]);
        wm::load_matrix_sync(bf, access.b.first_lo + k, kGroup);
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; ++m) wm::mma_sync(acc.frag[m], af[m], bf, acc.frag[m]);
    }
}

template<class Tile, int CM, int CN>
static __device__ __forceinline__ void fragment_node_prefill(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int width, unsigned int outputs, unsigned long long scale_offset,
        __nv_bfloat16* a, __nv_bfloat16* hi, __nv_bfloat16* lo, float* result,
        unsigned long long* generation_state, unsigned long long* observations) {
    static_assert(CM >= 1 && CN >= 1 && CM * CN <= 8, "cluster size must fit the target");
    const ClusterPlacement<Tile, CM, CN> placement;
    const auto region = placement.region();
    const FragmentNode<Tile> node{a, hi, lo, generation_state};
    const ClusterFragmentRoute<Tile, CM, CN> route{placement};
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned int cta_index = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int report_stride = 2 + 2 * Tile::kWarps;
    const Layout packed(weights, width, scale_offset);
    Accumulators<Tile::kFrags> acc;
    acc.fill();
    unsigned int fetched_a = 0, fetched_b = 0;
    for (unsigned int base = 0, generation = 0; base < width; base += kGroup, ++generation) {
        stage_owned_k(region, a, hi, lo, input, packed, rows, width, outputs, base, warp, lane);
        placement.cluster.sync(); // parent publishes the K region to DSM
        const KGeneration current{generation, 0};
        node.begin(current);
        const auto source = route.accept(region, a, hi, lo);
        #pragma unroll
        for (int branch = 0; branch < FragmentNode<Tile>::kMBranches; ++branch)
            fetched_a += node.a_branch(branch).accept(current, source.a, source.remote_a, threadIdx.x);
        #pragma unroll
        for (int branch = 0; branch < FragmentNode<Tile>::kNBranches; ++branch)
            fetched_b += node.b_branch(branch).accept(current, source.hi, source.lo,
                                                       source.remote_b, threadIdx.x);
        const PublishedK published = node.publish(current);
        const FragmentLease<Tile::kFrags> lease = node.acquire(published, warp);
        if (observations && lane == 0) {
            const unsigned long long at = static_cast<unsigned long long>(cta_index) * report_stride + 2 + 2 * warp;
            observations[at] = reinterpret_cast<unsigned long long>(lease.access.a[0].first_row);
            observations[at + 1] = reinterpret_cast<unsigned long long>(lease.access.b.first_hi);
        }
        consume_fragment_node<Tile>(acc, lease);
        node.release(lease); // all descendant warps retire before the node can be reused
        placement.cluster.sync(); // parent K region may now be overwritten
    }
    aggregate_output<Tile>(acc, result, output, rows, outputs, region.row, region.col, warp);
    if (observations && threadIdx.x == 0) {
        const unsigned long long at = static_cast<unsigned long long>(cta_index) * report_stride;
        observations[at] = fetched_a;
        observations[at + 1] = fetched_b;
    }
}

} // namespace q3
