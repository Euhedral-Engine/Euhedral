#pragma once
// GDN decay and gate control: alpha = exp(g) and beta per (row, value head), with and without the
// fused BF16 projections.
#include "reductions.cuh"
extern "C" __global__ void euhedral_gdn_control_fp32(
        const float* aProjection, const float* bProjection, const float* aLog, const float* dtBias,
        float* alphaOutput, float* betaOutput, uint32_t rows, uint32_t heads) {
    euhedral_pdl_begin();
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<uint64_t>(rows) * heads) return;
    const uint32_t head = static_cast<uint32_t>(index % heads);
    const float shifted = aProjection[index] + dtBias[head];
    const float softplus = fmaxf(shifted, 0.0f) + log1pf(expf(-fabsf(shifted)));
    alphaOutput[index] = expf(-expf(aLog[head]) * softplus);
    betaOutput[index] = qwen_gdn_sigmoid(bProjection[index]);
}

// One CTA owns both FP32 projections for one (row, head). Activation loads are shared,
// while each projection retains the original 128-lane FMA stripes and addition tree.
// The decay factor alpha = exp(g) is produced once per (row, head) for every recurrence consumer.
extern "C" __global__ __launch_bounds__(128) void euhedral_gdn_project_control_fp32(
        const __nv_bfloat16* input, const __nv_bfloat16* aWeight, const __nv_bfloat16* bWeight,
        const float* aLog, const float* dtBias, float* alphaOutput, float* betaOutput,
        uint32_t rows, uint32_t width, uint32_t heads) {
    euhedral_pdl_begin();
    const uint64_t index = blockIdx.x;
    if (index >= static_cast<uint64_t>(rows) * heads) return;
    const uint32_t row = static_cast<uint32_t>(index / heads);
    const uint32_t head = static_cast<uint32_t>(index % heads);
    const uint64_t activationBase = static_cast<uint64_t>(row) * width;
    const uint64_t weightBase = static_cast<uint64_t>(head) * width;
    // Each thread's operands are loaded in batches before its in-order FMA chain.
    constexpr uint32_t kBatch = 10;
    const uint32_t stride = blockDim.x;
    float a = 0.0f, b = 0.0f;
    uint32_t k = threadIdx.x;
    for (; k + (kBatch - 1) * stride < width; k += kBatch * stride) {
        __nv_bfloat16 xs[kBatch], as[kBatch], bs[kBatch];
#pragma unroll
        for (uint32_t i = 0; i < kBatch; i++) {
            xs[i] = input[activationBase + k + i * stride];
            as[i] = aWeight[weightBase + k + i * stride];
            bs[i] = bWeight[weightBase + k + i * stride];
        }
#pragma unroll
        for (uint32_t i = 0; i < kBatch; i++) {
            const float x = __bfloat162float(xs[i]);
            a = fmaf(x, __bfloat162float(as[i]), a);
            b = fmaf(x, __bfloat162float(bs[i]), b);
        }
    }
    for (; k < width; k += stride) {
        const float x = __bfloat162float(input[activationBase + k]);
        a = fmaf(x, __bfloat162float(aWeight[weightBase + k]), a);
        b = fmaf(x, __bfloat162float(bWeight[weightBase + k]), b);
    }
    __shared__ float partialA[128], partialB[128];
    partialA[threadIdx.x] = a;
    partialB[threadIdx.x] = b;
    __syncthreads();
    for (uint32_t stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partialA[threadIdx.x] += partialA[threadIdx.x + stride];
            partialB[threadIdx.x] += partialB[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float shifted = partialA[0] + dtBias[head];
        const float softplus = fmaxf(shifted, 0.0f) + log1pf(expf(-fabsf(shifted)));
        alphaOutput[index] = expf(-expf(aLog[head]) * softplus);
        betaOutput[index] = qwen_gdn_sigmoid(partialB[0]);
    }
}

// Prefill variant: one CTA owns R rows x H heads, so each activation row and each weight row is
// loaded once per CTA instead of once per (row, head). Every output keeps the 128-lane FMA stripes
// and the addition tree above (strides 64 and 32 in shared memory, then 16 .. 1 within a warp), so
// results match euhedral_gdn_project_control_fp32 bit for bit. Grid: ceil(rows / R) * (heads / H).
template<int R, int H>
static __device__ __forceinline__ void gdn_project_control_tile(
        const __nv_bfloat16* input, const __nv_bfloat16* aWeight, const __nv_bfloat16* bWeight,
        const float* aLog, const float* dtBias, float* alphaOutput, float* betaOutput,
        uint32_t rows, uint32_t width, uint32_t heads) {
    constexpr int kValues = 2 * R * H;
    const uint32_t headGroups = heads / H;
    const uint32_t firstRow = (blockIdx.x / headGroups) * R, firstHead = (blockIdx.x % headGroups) * H;
    float sums[kValues] = {};
    for (uint32_t k = threadIdx.x; k < width; k += 128u) {
        float xs[R], as[H], bs[H];
#pragma unroll
        for (int r = 0; r < R; r++)
            xs[r] = firstRow + r < rows ? __bfloat162float(input[static_cast<uint64_t>(firstRow + r) * width + k]) : 0.0f;
#pragma unroll
        for (int h = 0; h < H; h++) {
            as[h] = __bfloat162float(aWeight[static_cast<uint64_t>(firstHead + h) * width + k]);
            bs[h] = __bfloat162float(bWeight[static_cast<uint64_t>(firstHead + h) * width + k]);
        }
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int h = 0; h < H; h++) {
                sums[2 * (r * H + h)] = fmaf(xs[r], as[h], sums[2 * (r * H + h)]);
                sums[2 * (r * H + h) + 1] = fmaf(xs[r], bs[h], sums[2 * (r * H + h) + 1]);
            }
    }
    __shared__ float partial[kValues][128];
#pragma unroll
    for (int v = 0; v < kValues; v++) partial[v][threadIdx.x] = sums[v];
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < kValues * 64u; i += 128u) partial[i / 64u][i % 64u] += partial[i / 64u][i % 64u + 64u];
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < kValues * 32u; i += 128u) partial[i / 32u][i % 32u] += partial[i / 32u][i % 32u + 32u];
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    for (uint32_t pair = warp; pair < static_cast<uint32_t>(R * H); pair += 4u) {
        float a = partial[2 * pair][lane], b = partial[2 * pair + 1][lane];
#pragma unroll
        for (int stride = 16; stride; stride >>= 1) {
            a += __shfl_down_sync(0xffffffffu, a, stride);
            b += __shfl_down_sync(0xffffffffu, b, stride);
        }
        const uint32_t row = firstRow + pair / H, head = firstHead + pair % H;
        if (lane == 0 && row < rows) {
            const uint64_t index = static_cast<uint64_t>(row) * heads + head;
            const float shifted = a + dtBias[head];
            const float softplus = fmaxf(shifted, 0.0f) + log1pf(expf(-fabsf(shifted)));
            alphaOutput[index] = expf(-expf(aLog[head]) * softplus);
            betaOutput[index] = qwen_gdn_sigmoid(b);
        }
    }
}

extern "C" __global__ __launch_bounds__(128) void euhedral_gdn_project_control_8x4_fp32(
        const __nv_bfloat16* input, const __nv_bfloat16* aWeight, const __nv_bfloat16* bWeight,
        const float* aLog, const float* dtBias, float* alphaOutput, float* betaOutput,
        uint32_t rows, uint32_t width, uint32_t heads) {
    gdn_project_control_tile<8, 4>(input, aWeight, bWeight, aLog, dtBias, alphaOutput, betaOutput, rows, width, heads);
}
