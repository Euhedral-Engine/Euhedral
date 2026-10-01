#pragma once
#include "fragments.cuh"
#include <cuda/ptx>

namespace q3 {

// The protocol owns identities and borrower completion, not a K extent or a
// physical producer warp. All methods are collective over the calling warp.
// Exactly one producer warp calls begin/publish for a branch generation;
// each registered borrower warp calls acquire/release once, in generation order.
struct FragmentSlot {
    cuda::std::uint64_t ready;
    cuda::std::uint64_t released;
    cuda::std::uint64_t parent_ready;
    unsigned int generation;
};

template<class Tile, int SLOTS = 2>
struct PipelinedFragmentNode {
    static_assert(SLOTS >= 2, "temporal fragment storage needs multiple slots");
    static constexpr int kSlots = SLOTS;
    static constexpr int kNBranches = Tile::kCols / 16;
    static constexpr int kMBranches = Tile::kWarps / kNBranches;
    static constexpr int kBranches = kMBranches + kNBranches;
    FragmentSlot* states;
    __nv_bfloat16* a;
    __nv_bfloat16* hi;
    __nv_bfloat16* lo;

    __device__ __forceinline__ FragmentSlot& state(unsigned int branch, KGeneration g) const {
        return states[branch * kSlots + g.slot];
    }
    __device__ __forceinline__ static unsigned int phase(KGeneration g) {
        return (g.value / kSlots) & 1u;
    }
    // Bootstrap caller supplies the exact descendants, including parent DSM
    // borrowers when local execution storage aliases the parent-owned region.
    __device__ __forceinline__ void initialize(unsigned int branch, unsigned int borrowers) const {
        for (int slot = 0; slot < kSlots; ++slot) {
            auto& s = states[branch * kSlots + slot];
            cuda::ptx::mbarrier_init(&s.ready, 1);
            cuda::ptx::mbarrier_init(&s.released, borrowers);
            cuda::ptx::mbarrier_init(&s.parent_ready, 1);
            s.generation = ~0u;
        }
    }
    __device__ __forceinline__ void begin(unsigned int branch, KGeneration g, unsigned int lane) const {
        auto& s = state(branch, g);
        if (g.value >= kSlots) {
            while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire,
                    cuda::ptx::scope_cluster, &s.released, phase(g) ^ 1u)) {}
        }
        if (lane == 0) s.generation = g.value;
        __syncwarp();
    }
    __device__ __forceinline__ PublishedK publish(unsigned int branch, KGeneration g, unsigned int lane) const {
        // All fragment writes precede the leader's release notification.
        __syncwarp();
        if (lane == 0) cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release,
                cuda::ptx::scope_cta, cuda::ptx::space_shared, &state(branch, g).ready);
        return {g.value, g.slot};
    }
    __device__ __forceinline__ void acquire(unsigned int branch, PublishedK p) const {
        KGeneration g{p.value, p.slot};
        auto& s = state(branch, g);
        while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire,
                cuda::ptx::scope_cta, &s.ready, phase(g))) {}
        if (s.generation != g.value) asm volatile("trap;");
    }
    __device__ __forceinline__ void release(unsigned int branch, KGeneration g, unsigned int lane) const {
        // Every lane has finished its shared reads before one warp arrival.
        __syncwarp();
        if (lane == 0) cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release,
                cuda::ptx::scope_cluster, cuda::ptx::space_shared, &state(branch, g).released);
    }
    // Accept exactly this branch from a parent generation. The parent source
    // pointers already select its slot; an owner aliases storage without a copy.
    __device__ __forceinline__ unsigned int accept(unsigned int branch, KGeneration g,
            const ClusterKRegion& source, unsigned int lane) const {
        if (branch < kMBranches) {
            if (source.remote_a) {
                #pragma unroll
                for (int m = 0; m < Tile::kFrags; ++m) {
                    unsigned int row = Tile::row(branch * kNBranches, m);
                    for (unsigned int i = lane; i < 16 * kGroup; i += 32)
                        a[g.slot * Tile::kRows * kGroup + row * kGroup + i] = source.a[row * kGroup + i];
                }
            }
            return source.remote_a ? Tile::kFrags : 0;
        }
        const unsigned int col = (branch - kMBranches) * 16;
        if (source.remote_b) {
            for (unsigned int i = lane; i < 16 * kGroup; i += 32) {
                const unsigned int offset = col * kGroup + i;
                hi[g.slot * Tile::kCols * kGroup + offset] = source.hi[offset];
                lo[g.slot * Tile::kCols * kGroup + offset] = source.lo[offset];
            }
        }
        return source.remote_b ? 1 : 0;
    }
    __device__ __forceinline__ FragmentLease<Tile::kFrags> borrow(KGeneration g, unsigned int warp) const {
        KAccess<Tile::kFrags> access;
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; ++m)
            access.a[m] = {a + g.slot * Tile::kRows * kGroup + Tile::row(warp, m) * kGroup, false};
        const unsigned int offset = g.slot * Tile::kCols * kGroup + Tile::col(warp) * kGroup;
        access.b = {hi + offset, lo + offset, false};
        return {access, g.value, g.slot};
    }
};

// Cross-SM comparable diagnostic timestamps; never synchronization state.
static __device__ __forceinline__ unsigned long long fragment_time() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) : : "memory");
    return value;
}

// First physical strategy: one producer warp per A/B branch, followed by the
// existing Tile::kWarps consumers. No placement rule lives in the slot protocol.
// K64 is a production/consumption policy here, not the generation identity API.
template<class Tile, int CM, int CN>
static __device__ __forceinline__ void pipelined_fragment_prefill(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int width, unsigned int outputs, unsigned long long scale_offset,
        const PipelinedFragmentNode<Tile>& node, float* result, unsigned long long* observations) {
    using Node = PipelinedFragmentNode<Tile>;
    constexpr int kThreads = (Tile::kWarps + Node::kBranches) * 32;
    constexpr int kTraceWords = Node::kBranches * 6 + Tile::kWarps * 5;
    static_assert(CM >= 1 && CN >= 1 && CM * CN <= 8, "cluster size must fit the target");
    const ClusterPlacement<Tile, CM, CN> placement;
    const auto region = placement.region();
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned int generations = (width + kGroup - 1) / kGroup;
    const unsigned long long cta = static_cast<unsigned long long>(blockIdx.y) * gridDim.x + blockIdx.x;
    const Layout packed(weights, width, scale_offset);
    if (threadIdx.x < Node::kBranches) {
        const unsigned int branch = threadIdx.x;
        const bool is_a = branch < Node::kMBranches;
        const unsigned int descendants = is_a ? Node::kNBranches : Node::kMBranches;
        const unsigned int remote_readers = is_a
                ? (region.produces_a() ? CN - 1 : 0) : (region.produces_b() ? CM - 1 : 0);
        node.initialize(branch, descendants + remote_readers);
    }
    // Bootstrap: all local barriers initialized and every DSM CTA is alive.
    placement.cluster.sync();
    if (warp < Node::kBranches) {
        const unsigned int branch = warp;
        const bool is_a = branch < Node::kMBranches;
        const bool owner = is_a ? region.produces_a() : region.produces_b();
        const unsigned int owner_rank = is_a ? region.m * CN : region.n;
        for (unsigned int generation = 0; generation < generations; ++generation) {
            const KGeneration g{generation, generation % Node::kSlots};
            node.begin(branch, g, lane);
            unsigned long long* trace = observations
                    ? observations + (cta * generations + generation) * kTraceWords + branch * 6 : nullptr;
            const unsigned int a_offset = g.slot * Tile::kRows * kGroup;
            const unsigned int b_offset = g.slot * Tile::kCols * kGroup;
            auto& state = node.state(branch, g);
            if (!owner) {
                // Only this branch waits on its parent publication, locally.
                while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire,
                        cuda::ptx::scope_cluster, &state.parent_ready, Node::phase(g))) {}
            }
            if (trace && lane == 0) trace[0] = fragment_time();
            if (owner) {
                const unsigned int base = generation * kGroup;
                if (is_a) {
                    #pragma unroll
                    for (int m = 0; m < Tile::kFrags; ++m) {
                        unsigned int row = Tile::row(branch * Node::kNBranches, m);
                        stage_activation_tile<16, kGroup, 32>(node.a + a_offset + row * kGroup,
                                input, rows, width, region.row + row, base, lane);
                    }
                } else {
                    const unsigned int col = (branch - Node::kMBranches) * 16;
                    stage_weight_tile<16, 1>(node.hi + b_offset + col * kGroup,
                            node.lo + b_offset + col * kGroup, packed, region.col + col,
                            outputs, base, 0, lane);
                }
                // The producing warp joins only itself before notifying each
                // actual DSM child. It never waits for another parent branch.
                __syncwarp();
                if (lane == 0) {
                    const unsigned int peers = is_a ? CN : CM;
                    for (unsigned int peer = 1; peer < peers; ++peer) {
                        const unsigned int rank = is_a ? region.m * CN + peer : peer * CN + region.n;
                        auto* ready = placement.cluster.map_shared_rank(&state.parent_ready, rank);
                        cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release,
                                cuda::ptx::scope_cluster, cuda::ptx::space_cluster, ready);
                    }
                }
            }
            const ClusterKRegion source{
                owner ? node.a + a_offset : placement.cluster.map_shared_rank(node.a + a_offset, owner_rank),
                owner ? node.hi + b_offset : placement.cluster.map_shared_rank(node.hi + b_offset, owner_rank),
                owner ? node.lo + b_offset : placement.cluster.map_shared_rank(node.lo + b_offset, owner_rank),
                is_a && !owner, !is_a && !owner};
            const unsigned int fetched = node.accept(branch, g, source, lane);
            __syncwarp();
            if (trace && lane == 0) {
                trace[1] = fragment_time();
                trace[3] = fetched;
                trace[4] = g.value;
                trace[5] = g.slot;
            }
            if (!owner) {
                // Copy completed: release the parent immediately, not after MMA.
                // Local descendants retain the independently owned local slot.
                if (lane == 0) {
                    auto* released = placement.cluster.map_shared_rank(&state.released, owner_rank);
                    cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release,
                            cuda::ptx::scope_cluster, cuda::ptx::space_cluster, released);
                }
            }
            // Stamp immediately before publication so consumer timestamps cannot
            // race ahead of the diagnostic stamp. No consumer reads this trace.
            if (trace && lane == 0) trace[2] = fragment_time();
            node.publish(branch, g, lane);
        }
    } else {
        const unsigned int consumer = warp - Node::kBranches;
        const unsigned int a_branch = consumer / Node::kNBranches;
        const unsigned int b_branch = Node::kMBranches + consumer % Node::kNBranches;
        Accumulators<Tile::kFrags> acc;
        acc.fill();
        for (unsigned int generation = 0; generation < generations; ++generation) {
            const KGeneration g{generation, generation % Node::kSlots};
            const PublishedK published{g.value, g.slot};
            node.acquire(a_branch, published);
            node.acquire(b_branch, published);
            const auto lease = node.borrow(g, consumer);
            unsigned long long* trace = observations
                    ? observations + (cta * generations + generation) * kTraceWords
                            + Node::kBranches * 6 + consumer * 5 : nullptr;
            if (trace && lane == 0) {
                trace[0] = fragment_time();
                trace[3] = reinterpret_cast<unsigned long long>(lease.access.a[0].first_row);
                trace[4] = reinterpret_cast<unsigned long long>(lease.access.b.first_hi);
            }
            consume_fragment_node<Tile>(acc, lease);
            __syncwarp();
            if (trace && lane == 0) {
                trace[1] = fragment_time();
                trace[2] = fragment_time();
            }
            node.release(a_branch, g, lane);
            node.release(b_branch, g, lane);
        }
        store_accumulators<Tile>(result, acc, consumer);
    }
    // Terminal DSM lifetime join: no CTA may exit while a peer still touches its
    // shared state. This is outside the K loop and also joins local result writes.
    placement.cluster.sync();
    write_output_tile<Tile::kRows, Tile::kCols, kThreads>(
            output, result, rows, outputs, region.row, region.col, threadIdx.x);
}

} // namespace q3
