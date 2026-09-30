#include "q3/strategies/k32_prefill.cuh"
#include "qwen_ffn_streamed.cu"

// Region B: pair the independent N16 branches as gate/up columns and apply SwiGLU in the
// epilogue. CB helpers and the K32/A/compact-B/MMA schedule are reused; the schedule is duplicated
// here, so the A/B index math in `paired_gate_up` must stay in sync with k32_prefill.cuh. The template
// flags mirror the CB leaf signature; only <false,false,false,true> is instantiated, and folding
// them changes generated SASS, so they stay until the kernel is requalified.
namespace qwen_ffn {
using namespace k32_probe;
template<bool EARLY_A, bool EARLY_B, bool DUMP, bool COMPACT_B = false>
static __device__ __forceinline__ void paired_gate_up(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int width, unsigned int outputs,
        unsigned long long scale_offset, float* dump, Shared& storage) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int m_branch = warp >> 1, n_branch = warp & 1u;
    const bool owns_a = warp == 0u || warp == 3u;
    const unsigned int owned_branch = owns_a ? m_branch : 2u + n_branch;
    const unsigned int output_tiles = (outputs + 31u) / 32u;
    const unsigned int first_row = (blockIdx.x / output_tiles) * 64u;
    const unsigned int first_col = (blockIdx.x % output_tiles) * 16u;
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
        else produce_b<true>(stage.b[0][n_branch][0], stage.b[0][n_branch][1],
                weights_layout, outputs, first_col + n_branch * (outputs / 2u), 0, lane);
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
            else produce_b<true>(stage.b[next & 1u][n_branch][0], stage.b[next & 1u][n_branch][1],
                    weights_layout, outputs, first_col + n_branch * (outputs / 2u), next * 32u, lane);
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
                q3::ldmatrix_x4(af[m], a + (m * 16u + (lane & 15u)) * kAStride
                        + half * 16u + (lane >> 4) * 8u);
            unsigned int col = (lane & 7u) + ((lane >> 4) << 3);
            unsigned int index = b_index(col, half * 16u + ((lane >> 3) & 1u) * 8u);
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
                        first_col + n_branch * (outputs / 2u), (gen + 1u) * 32u, lane);
        }
        if (COMPACT_B && !owns_a && gen + 1u < generations) {
            const unsigned int next = gen + 1u;
            if (next >= 2u)
                wait(&stage.release[owned_branch][next & 1u], ((next - 2u) >> 1) & 1u);
            stage_compact_b<true>(stage.b[next & 1u][n_branch][0],
                    stage.b[next & 1u][n_branch][1], compact_next, lane);
            arrive(&stage.ready[owned_branch][next & 1u]);
        }
        if (gen + 1u < generations && !(owns_a ? EARLY_A : (EARLY_B || COMPACT_B))) {
            const unsigned int next = gen + 1u;
            if (next >= 2u)
                wait(&stage.release[owned_branch][next & 1u], ((next - 2u) >> 1) & 1u);
            if (owns_a) produce_a(stage.a[next & 1u][m_branch], input, rows, width,
                    first_row, m_branch, next * 32u, lane);
            else produce_b<true>(stage.b[next & 1u][n_branch][0], stage.b[next & 1u][n_branch][1],
                    weights_layout, outputs, first_col + n_branch * (outputs / 2u), next * 32u, lane);
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
    for (unsigned int i = threadIdx.x; i < 64u * 16u; i += 128u) {
        unsigned int row = first_row + i / 16u;
        unsigned int col = first_col + i % 16u;
        if (row < rows && col < outputs / 2u) {
            unsigned int at = (i / 16u) * 32u + i % 16u;
            // Emulate the eliminated gate/up BF16 store/load before the existing SwiGLU expression.
            float gate = q3::bf16_to_float(q3::float_to_bf16(result[at]));
            float up = q3::bf16_to_float(q3::float_to_bf16(result[at + 16u]));
            output[(unsigned long long)row * (outputs / 2u) + col] =
                    __bfloat16_as_ushort(__float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up));
        }
    }
}

} // namespace qwen_ffn

extern "C" __global__ __launch_bounds__(128) void euhedral_q3_gate_up_swiglu_bf16(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int width, unsigned int outputs, unsigned long long scale_offset) {
    __shared__ k32_probe::Shared storage;
    qwen_ffn::paired_gate_up<false, false, false, true>(input, weights, output, rows, width, outputs,
            scale_offset, nullptr, storage);
}
