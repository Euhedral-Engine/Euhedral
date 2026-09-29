#pragma once
#include "q3/strategies/prefill.cuh"

namespace k32_probe {
using Tile = q3::Prefill64;
using Leaf = q3::MmaSyncLeaf<Tile>;

// The stage (including mbarriers) and result have disjoint CTA lifetimes.
union alignas(32) Shared {
    struct Stage {
        __nv_bfloat16 a[2][2][32 * 32];       // [slot][M branch][local row, K]
        __nv_bfloat16 b[2][2][2][16 * 32];    // [slot][N branch][hi/lo][column, K]
        alignas(8) unsigned long long ready[4][2];
        alignas(8) unsigned long long release[4][2];
    };
    struct Result { float values[64 * 32]; };
    Stage stage;
    Result result;
    struct ActivateStage {};
    struct ActivateResult {};
    Shared() = default;
    __device__ Shared(ActivateStage) : stage() {}
    __device__ Shared(ActivateResult) : result() {}
};
static_assert(sizeof(Shared::Stage) == 16512, "K32 stage budget changed");
static_assert(sizeof(Shared) == 16512, "K32 shared overlay changed");
static_assert(__is_trivial(Shared::Stage), "stage must have trivial storage");
static_assert(__is_trivial(Shared::Result), "result must have trivial storage");

static __device__ __forceinline__ unsigned int address(const unsigned long long* value) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(value));
}
static __device__ __forceinline__ void init(unsigned long long* value, unsigned int count) {
    unsigned int a = address(value);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(count) : "memory");
}
static __device__ __forceinline__ void invalidate(unsigned long long* value) {
    unsigned int a = address(value);
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(a) : "memory");
}
static __device__ __forceinline__ void arrive(unsigned long long* value) {
    unsigned int a = address(value);
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
}
static __device__ __forceinline__ void wait(unsigned long long* value, unsigned int parity) {
    unsigned int a = address(value);
    asm volatile("{ .reg .pred p; K32_WAIT_%=: "
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1; "
                 "@!p bra K32_WAIT_%=; }"
                 :: "r"(a), "r"(parity) : "memory");
}

// Adjacent lanes read contiguous 16-byte stripes. One warp owns the whole
// M32xK32 pack; both N sibling warps subsequently borrow that single copy.
static __device__ __forceinline__ void produce_a(
        __nv_bfloat16* a, const unsigned short* input,
        unsigned int rows, unsigned int width, unsigned int first_row,
        unsigned int m_branch, unsigned int base, unsigned int lane) {
    #pragma unroll
    for (unsigned int pass = 0; pass < 4; ++pass) {
        unsigned int local_row = (lane >> 2) + pass * 8;
        unsigned int local_k = (lane & 3u) * 8u;
        unsigned int row = first_row + m_branch * 16u + (local_row & 15u)
                + (local_row >= 16u ? 32u : 0u);
        unsigned int fields[4] = {0, 0, 0, 0};
        if (row < rows) {
            const unsigned short* src = input + (unsigned long long)row * width + base + local_k;
            // Raw aligned vector load, avoiding C++ object/type punning.
            asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(fields[0]), "=r"(fields[1]), "=r"(fields[2]), "=r"(fields[3])
                    : "l"(src) : "memory");
        }
        // All 65536 raw BF16 encodings were checked against production's
        // bf16_to_float -> __float2bfloat16 conversion in the K16 campaign.
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int pair = fields[j];
            unsigned int lo = pair & 65535u, hi = pair >> 16;
            lo = (lo & 32767u) > 32640u ? 32767u : lo;
            hi = (hi & 32767u) > 32640u ? 32767u : hi;
            fields[j] = lo | (hi << 16);
        }
        unsigned int dst = static_cast<unsigned int>(__cvta_generic_to_shared(a + local_row * 32 + local_k));
        asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::
                "r"(dst), "r"(fields[0]), "r"(fields[1]), "r"(fields[2]), "r"(fields[3]) : "memory");
    }
}

// Each N16 branch is decoded by its own producer warp. K32 uses three packed
// words and one scale per column; the two K16 halves share this acquisition.
static __device__ __forceinline__ void produce_b(
        __nv_bfloat16* hi, __nv_bfloat16* lo, const q3::Layout& w,
        unsigned int outputs, unsigned int first_col, unsigned int base,
        unsigned int lane) {
    unsigned int sublane = lane & 7u;
    unsigned int first_word = (base & 32u) ? 3u : 0u;
    #pragma unroll
    for (unsigned int j = 0; j < 4; ++j) {
        unsigned int col = (lane >> 3) + 4u * j;
        unsigned int word = 0, scale_bits = 0;
        if (first_col + col < outputs) {
            unsigned long long g = w.group(first_col + col, base / 64u);
            if (sublane < 3u)
                word = reinterpret_cast<const unsigned int*>(w.group_bytes(g))[first_word + sublane];
            if (sublane == 0) scale_bits = w.scales[g];
        }
        float scale = q3::fp16_to_float((unsigned short)__shfl_sync(0xffffffffu, scale_bits, 0, 8));
        #pragma unroll
        for (unsigned int half = 0; half < 2; ++half) {
            unsigned int bit = half * 48u + sublane * 6u;
            unsigned int first = __shfl_sync(0xffffffffu, word, bit >> 5, 8);
            unsigned int second = __shfl_sync(0xffffffffu, word, (bit >> 5) + 1, 8);
            unsigned int codes = static_cast<unsigned int>(
                    ((static_cast<unsigned long long>(second) << 32) | first) >> (bit & 31u)) & 63u;
            q3::stage_split_pair(hi, lo, col * 32u + half * 16u + sublane * 2u, codes, scale);
        }
    }
}

struct CompactB {
    unsigned int words[4];
    unsigned int scales[4];
};

static __device__ __forceinline__ void prefetch_compact_b(
        CompactB& next, const q3::Layout& w, unsigned int outputs,
        unsigned int first_col, unsigned int base, unsigned int lane) {
    unsigned int sublane = lane & 7u;
    unsigned int first_word = (base & 32u) ? 3u : 0u;
    #pragma unroll
    for (unsigned int j = 0; j < 4; ++j) {
        unsigned int col = (lane >> 3) + 4u * j;
        unsigned int word = 0, scale_bits = 0;
        if (first_col + col < outputs) {
            unsigned long long g = w.group(first_col + col, base / 64u);
            if (sublane < 3u)
                word = reinterpret_cast<const unsigned int*>(w.group_bytes(g))[first_word + sublane];
            if (sublane == 0u) scale_bits = w.scales[g];
        }
        next.words[j] = word;
        next.scales[j] = scale_bits;
    }
}

static __device__ __forceinline__ void stage_compact_b(
        __nv_bfloat16* hi, __nv_bfloat16* lo, const CompactB& next,
        unsigned int lane) {
    unsigned int sublane = lane & 7u;
    #pragma unroll
    for (unsigned int j = 0; j < 4; ++j) {
        unsigned int col = (lane >> 3) + 4u * j;
        float scale = q3::fp16_to_float((unsigned short)__shfl_sync(
                0xffffffffu, next.scales[j], 0, 8));
        #pragma unroll
        for (unsigned int half = 0; half < 2; ++half) {
            unsigned int bit = half * 48u + sublane * 6u;
            unsigned int first = __shfl_sync(0xffffffffu, next.words[j], bit >> 5, 8);
            unsigned int second = __shfl_sync(0xffffffffu, next.words[j], (bit >> 5) + 1, 8);
            unsigned int codes = static_cast<unsigned int>(
                    ((static_cast<unsigned long long>(second) << 32) | first) >> (bit & 31u)) & 63u;
            q3::stage_split_pair(hi, lo, col * 32u + half * 16u + sublane * 2u, codes, scale);
        }
    }
}

template<bool EARLY_A, bool EARLY_B, bool DUMP, bool COMPACT_B = false>
static __device__ __forceinline__ void run(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int width, unsigned int outputs,
        unsigned long long scale_offset, float* dump, Shared& storage) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int m_branch = warp >> 1, n_branch = warp & 1u;
    const bool owns_a = warp == 0u || warp == 3u;
    const unsigned int owned_branch = owns_a ? m_branch : 2u + n_branch;
    const unsigned int output_tiles = (outputs + 31u) / 32u;
    const unsigned int first_row = (blockIdx.x / output_tiles) * 64u;
    const unsigned int first_col = (blockIdx.x % output_tiles) * 32u;
    const q3::Layout weights_layout(weights, width, scale_offset);
    typename Leaf::Acc acc;
    Leaf::fill(acc);

    if (threadIdx.x == 0) new (static_cast<void*>(&storage)) Shared(Shared::ActivateStage{});
    __syncthreads();
    Shared::Stage& stage = storage.stage;
    if (threadIdx.x < 8u) {
        unsigned int branch = threadIdx.x >> 1, slot = threadIdx.x & 1u;
        init(&stage.ready[branch][slot], 32u);
        init(&stage.release[branch][slot], 64u);
    }
    __syncthreads();

    const unsigned int generations = width / 32u;
    CompactB compact_next{};
    if (generations) {
        if (owns_a) produce_a(stage.a[0][m_branch], input, rows, width,
                first_row, m_branch, 0, lane);
        else produce_b(stage.b[0][n_branch][0], stage.b[0][n_branch][1],
                weights_layout, outputs, first_col + n_branch * 16u, 0, lane);
        arrive(&stage.ready[owned_branch][0]);
    }
    for (unsigned int gen = 0; gen < generations; ++gen) {
        // The early variant produces n+1 before borrowing/consuming n and
        // publishes readiness immediately, without doing its current MMA.
        if (gen + 1u < generations && (owns_a ? EARLY_A : (EARLY_B && !COMPACT_B))) {
            unsigned int next = gen + 1u;
            if (next >= 2u)
                wait(&stage.release[owned_branch][next & 1u], ((next - 2u) >> 1) & 1u);
            if (owns_a) produce_a(stage.a[next & 1u][m_branch], input, rows, width,
                    first_row, m_branch, next * 32u, lane);
            else produce_b(stage.b[next & 1u][n_branch][0], stage.b[next & 1u][n_branch][1],
                    weights_layout, outputs, first_col + n_branch * 16u, next * 32u, lane);
            arrive(&stage.ready[owned_branch][next & 1u]);
        }
        // A and B have separate acquire edges; there is no steady CTA or
        // pair-wide rendezvous and no publication after unrelated current MMA.
        wait(&stage.ready[m_branch][gen & 1u], (gen >> 1) & 1u);
        wait(&stage.ready[2u + n_branch][gen & 1u], (gen >> 1) & 1u);

        __nv_bfloat16* a = stage.a[gen & 1u][m_branch];
        __nv_bfloat16* hi = stage.b[gen & 1u][n_branch][0];
        __nv_bfloat16* lo = stage.b[gen & 1u][n_branch][1];
        #pragma unroll
        for (unsigned int half = 0; half < 2; ++half) {
            unsigned int af[2][4], bf[2][4];
            #pragma unroll
            for (unsigned int m = 0; m < 2; ++m)
                q3::ldmatrix_x4(af[m], a + (m * 16u + (lane & 15u)) * 32u
                        + half * 16u + (lane >> 4) * 8u);
            unsigned int col = (lane & 7u) + ((lane >> 4) << 3);
            unsigned int index = col * 32u + half * 16u + ((lane >> 3) & 1u) * 8u;
            q3::ldmatrix_x4(bf[0], hi + index);
            q3::ldmatrix_x4(bf[1], lo + index);
            if (half == 1u) {
                // All of this warp's LDSM borrows have completed. MMA below
                // touches only fragment registers, so the slots can be reused
                // once both borrower warps have arrived, independently per branch.
                __syncwarp();
                arrive(&stage.release[m_branch][gen & 1u]);
                arrive(&stage.release[2u + n_branch][gen & 1u]);
            }
            Leaf::step(acc, af, bf);
            if (COMPACT_B && !owns_a && half == 0u && gen + 1u < generations)
                prefetch_compact_b(compact_next, weights_layout, outputs,
                        first_col + n_branch * 16u, (gen + 1u) * 32u, lane);
        }
        if (COMPACT_B && !owns_a && gen + 1u < generations) {
            const unsigned int next = gen + 1u;
            if (next >= 2u)
                wait(&stage.release[owned_branch][next & 1u], ((next - 2u) >> 1) & 1u);
            stage_compact_b(stage.b[next & 1u][n_branch][0],
                    stage.b[next & 1u][n_branch][1], compact_next, lane);
            arrive(&stage.ready[owned_branch][next & 1u]);
        }
        if (gen + 1u < generations && !(owns_a ? EARLY_A : (EARLY_B || COMPACT_B))) {
            const unsigned int next = gen + 1u;
            if (next >= 2u)
                wait(&stage.release[owned_branch][next & 1u], ((next - 2u) >> 1) & 1u);
            if (owns_a) produce_a(stage.a[next & 1u][m_branch], input, rows, width,
                    first_row, m_branch, next * 32u, lane);
            else produce_b(stage.b[next & 1u][n_branch][0], stage.b[next & 1u][n_branch][1],
                    weights_layout, outputs, first_col + n_branch * 16u, next * 32u, lane);
            arrive(&stage.ready[owned_branch][next & 1u]);
        }
    }
    __syncthreads();
    if (threadIdx.x < 8u) {
        unsigned int branch = threadIdx.x >> 1, slot = threadIdx.x & 1u;
        invalidate(&stage.ready[branch][slot]);
        invalidate(&stage.release[branch][slot]);
    }
    __syncthreads();
    if (threadIdx.x == 0) new (static_cast<void*>(&storage)) Shared(Shared::ActivateResult{});
    __syncthreads();
    float* result = storage.result.values;
    Leaf::store(result, acc, warp);
    __syncthreads();
    if (DUMP) for (unsigned int i = threadIdx.x; i < 64u * 32u; i += 128u)
        dump[(unsigned long long)blockIdx.x * 64u * 32u + i] = result[i];
    q3::write_output_tile<64, 32, 128>(
            output, result, rows, outputs, first_row, first_col, threadIdx.x);
}
}  // namespace k32_probe
