#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "pdl.cuh"

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;


__device__ __forceinline__ float qwen_gdn_reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

/// Sums 128 values held four per lane (`lane`, `lane + 32`, `lane + 64`, `lane + 96`) with the addition
/// tree of `qwen_gdn_reduce_sum`: stride 64 pairs `e0 + e2` and `e1 + e3`, stride 32 joins them, then
/// shuffles finish strides 16..1. Every lane receives lane 0's result.
__device__ __forceinline__ float qwen_gdn_warp_sum_128(float e0, float e1, float e2, float e3) {
    float sum = __fadd_rn(__fadd_rn(e0, e2), __fadd_rn(e1, e3));
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, 16));
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, 8));
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, 4));
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, 2));
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, 1));
    return __shfl_sync(0xffffffffu, sum, 0);
}

__device__ __forceinline__ float qwen_gdn_sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ float qwen_gdn_silu(float value) {
    return value * qwen_gdn_sigmoid(value);
}

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
    const uint64_t index = blockIdx.x;
    if (index >= static_cast<uint64_t>(rows) * heads) return;
    const uint32_t row = static_cast<uint32_t>(index / heads);
    const uint32_t head = static_cast<uint32_t>(index % heads);
    const uint64_t activationBase = static_cast<uint64_t>(row) * width;
    const uint64_t weightBase = static_cast<uint64_t>(head) * width;
    float a = 0.0f, b = 0.0f;
    for (uint32_t k = threadIdx.x; k < width; k += blockDim.x) {
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

/// Rows are independent apart from a sliding window of the previous `kernelSize - 1` inputs, so
/// blockIdx.y splits them into blocks of `QWEN_GDN_CONV_ROWS` (at least the largest window): a block
/// seeds its window from the stored state (first block) or from the preceding input rows, which
/// hold the same values the window would. Each output keeps the original tap order. The new state
/// is the last window of inputs; the first block writes it after reading the old state, so no
/// other CTA touches a channel's state.
#define QWEN_GDN_CONV_ROWS 32u
extern "C" __global__ void euhedral_gdn_convolution_bf16(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* valueZ,
        const __nv_bfloat16* convolutionWeights, __nv_bfloat16* convolutionState,
        __nv_bfloat16* output, uint32_t rows, uint32_t queryKeyWidth,
        uint32_t valueWidth, uint32_t convolutionWidth, uint32_t kernelSize) {
    euhedral_pdl_begin();
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= convolutionWidth) return;
    const uint32_t historyWidth = kernelSize - 1;
    const uint32_t first = blockIdx.y * QWEN_GDN_CONV_ROWS;
    if (first >= rows) return;
    const uint32_t last = min(rows, first + QWEN_GDN_CONV_ROWS);
    auto input = [&](uint32_t row) {
        return channel < queryKeyWidth
                ? __bfloat162float(queryKey[static_cast<uint64_t>(row) * queryKeyWidth + channel])
                : __bfloat162float(valueZ[static_cast<uint64_t>(row) * valueWidth * 2 + channel - queryKeyWidth]);
    };
    float history[31];
    for (uint32_t i = 0; i < historyWidth; i++) {
        history[i] = first == 0
                ? __bfloat162float(convolutionState[static_cast<uint64_t>(channel) * historyWidth + i])
                : input(first - historyWidth + i);
    }
    for (uint32_t row = first; row < last; row++) {
        float sum = 0.0f;
        for (uint32_t tap = 0; tap < kernelSize; tap++) {
            const float value = tap < historyWidth ? history[tap] : input(row);
            sum = fmaf(value, __bfloat162float(convolutionWeights[static_cast<uint64_t>(tap) * convolutionWidth + channel]), sum);
        }
        output[static_cast<uint64_t>(row) * convolutionWidth + channel] = __float2bfloat16_rn(qwen_gdn_silu(sum));
        if (historyWidth > 0) {
            for (uint32_t i = 0; i + 1 < historyWidth; i++) history[i] = history[i + 1];
            history[historyWidth - 1] = input(row);
        }
    }
    if (first != 0) return;
    // The first block owns the state. With one block its window already ends at the last row;
    // otherwise rows >= QWEN_GDN_CONV_ROWS > historyWidth and the window is the last inputs.
    for (uint32_t i = 0; i < historyWidth; i++) {
        const float value = last == rows ? history[i] : input(rows - historyWidth + i);
        convolutionState[static_cast<uint64_t>(channel) * historyWidth + i] = __float2bfloat16_rn(value);
    }
}

/// Columns of one value head owned by a warp. Its lanes hold four state elements per column, so the
/// warp reuses one query/key load, one normalization and one alpha/beta pair across all columns.
#define QWEN_GDN_WARP_COLUMNS 8

/// One warp owns `QWEN_GDN_WARP_COLUMNS` contiguous value columns of one value head. Recurrent state
/// stays in registers across every row and is loaded and stored once. Grid: valueHeads * 16 warps,
/// launched as 32-thread CTAs. Reductions use `qwen_gdn_warp_sum_128`, so no barrier is needed.
extern "C" __global__ __launch_bounds__(32) void euhedral_gdn_recurrence_bf16(
        const __nv_bfloat16* convolved, const float* alpha, const float* beta,
        float* recurrentState, __nv_bfloat16* output, uint32_t rows,
        uint32_t keyHeads, uint32_t valueHeads, uint32_t keyHeadDim,
        uint32_t valueHeadDim, float outputScale) {
    euhedral_pdl_begin();
    constexpr uint32_t columns = QWEN_GDN_WARP_COLUMNS;
    if (blockDim.x != 32 || keyHeadDim != 128 || valueHeadDim != 128) return;
    const uint32_t tilesPerHead = valueHeadDim / columns;
    const uint32_t valueHead = blockIdx.x / tilesPerHead;
    const uint32_t column0 = (blockIdx.x % tilesPerHead) * columns;
    if (valueHead >= valueHeads) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t queryKeyWidth = 2 * keyHeads * keyHeadDim;
    const uint32_t convolvedWidth = queryKeyWidth + valueHeads * valueHeadDim;
    const uint32_t keyHead = valueHead / (valueHeads / keyHeads);
    float state[columns][4];
#pragma unroll
    for (uint32_t c = 0; c < columns; c++) {
        const uint64_t base = (static_cast<uint64_t>(valueHead) * valueHeadDim + column0 + c) * keyHeadDim;
#pragma unroll
        for (uint32_t j = 0; j < 4; j++) state[c][j] = recurrentState[base + lane + 32 * j];
    }
    for (uint32_t row = 0; row < rows; row++) {
        const uint64_t rowOffset = static_cast<uint64_t>(row) * convolvedWidth;
        const uint64_t queryBase = rowOffset + keyHead * keyHeadDim + lane;
        const uint64_t keyBase = rowOffset + keyHeads * keyHeadDim + keyHead * keyHeadDim + lane;
        float query[4], key[4];
#pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            query[j] = __bfloat162float(convolved[queryBase + 32 * j]);
            key[j] = __bfloat162float(convolved[keyBase + 32 * j]);
        }
        const float keyInverse = rsqrtf(qwen_gdn_warp_sum_128(__fmul_rn(key[0], key[0]), __fmul_rn(key[1], key[1]),
                __fmul_rn(key[2], key[2]), __fmul_rn(key[3], key[3])) + 1.0e-6f);
        const float queryInverse = rsqrtf(qwen_gdn_warp_sum_128(__fmul_rn(query[0], query[0]),
                __fmul_rn(query[1], query[1]), __fmul_rn(query[2], query[2]), __fmul_rn(query[3], query[3])) + 1.0e-6f);
        float normalizedKey[4], normalizedQuery[4];
#pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            normalizedKey[j] = key[j] * keyInverse;
            normalizedQuery[j] = query[j] * queryInverse;
        }
        const float rowAlpha = alpha[static_cast<uint64_t>(row) * valueHeads + valueHead];
        const float rowBeta = beta[static_cast<uint64_t>(row) * valueHeads + valueHead];
        float stateKeyDot[columns], value[columns];
#pragma unroll
        for (uint32_t c = 0; c < columns; c++) {
            value[c] = __bfloat162float(convolved[rowOffset + queryKeyWidth + valueHead * valueHeadDim + column0 + c]);
            stateKeyDot[c] = qwen_gdn_warp_sum_128(__fmul_rn(state[c][0], normalizedKey[0]),
                    __fmul_rn(state[c][1], normalizedKey[1]), __fmul_rn(state[c][2], normalizedKey[2]),
                    __fmul_rn(state[c][3], normalizedKey[3]));
        }
        float outputSum[columns];
#pragma unroll
        for (uint32_t c = 0; c < columns; c++) {
            // Explicit FMA: the compiler already fuses this product-subtract; pinning it keeps the rounding independent of the driver.
            const float delta = rowBeta * __fmaf_rn(-rowAlpha, stateKeyDot[c], value[c]);
#pragma unroll
            for (uint32_t j = 0; j < 4; j++) state[c][j] = rowAlpha * state[c][j] + delta * normalizedKey[j];
            outputSum[c] = qwen_gdn_warp_sum_128(__fmul_rn(state[c][0], normalizedQuery[0]),
                    __fmul_rn(state[c][1], normalizedQuery[1]), __fmul_rn(state[c][2], normalizedQuery[2]),
                    __fmul_rn(state[c][3], normalizedQuery[3]));
        }
        // Lane c publishes column c; the select chain keeps the array in registers.
        float mine = outputSum[0];
#pragma unroll
        for (uint32_t c = 1; c < columns; c++) mine = lane == c ? outputSum[c] : mine;
        if (lane < columns) {
            output[static_cast<uint64_t>(row) * valueHeads * valueHeadDim + valueHead * valueHeadDim + column0 + lane]
                    = __float2bfloat16_rn(mine * outputScale);
        }
    }
#pragma unroll
    for (uint32_t c = 0; c < columns; c++) {
        const uint64_t base = (static_cast<uint64_t>(valueHead) * valueHeadDim + column0 + c) * keyHeadDim;
#pragma unroll
        for (uint32_t j = 0; j < 4; j++) recurrentState[base + lane + 32 * j] = state[c][j];
    }
}

extern "C" __global__ void euhedral_gdn_gated_rms_norm_bf16(
        const __nv_bfloat16* recurrent, const __nv_bfloat16* valueZ,
        const __nv_bfloat16* normWeight, __nv_bfloat16* output,
        uint32_t rows, uint32_t valueHeads, uint32_t headDim, float epsilon) {
    euhedral_pdl_begin();
    const uint32_t rowHead = blockIdx.x;
    const uint32_t row = rowHead / valueHeads;
    const uint32_t head = rowHead % valueHeads;
    if (row >= rows) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t width = valueHeads * headDim;
    const uint64_t recurrentOffset = static_cast<uint64_t>(row) * width + static_cast<uint64_t>(head) * headDim;
    const uint64_t zOffset = static_cast<uint64_t>(row) * width * 2 + width + static_cast<uint64_t>(head) * headDim;
    float partial = 0.0f;
    for (uint32_t i = lane; i < headDim; i += blockDim.x) {
        const float value = __bfloat162float(recurrent[recurrentOffset + i]);
        partial = fmaf(value, value, partial);
    }
    __shared__ float scratch[128];
    const float sum = qwen_gdn_reduce_sum(partial, scratch);
    const float inverse = rsqrtf(sum / static_cast<float>(headDim) + epsilon);
    for (uint32_t i = lane; i < headDim; i += blockDim.x) {
        const float x = __bfloat162float(recurrent[recurrentOffset + i]);
        const float z = __bfloat162float(valueZ[zOffset + i]);
        output[recurrentOffset + i] = __float2bfloat16_rn(x * inverse * __bfloat162float(normWeight[i]) * qwen_gdn_silu(z));
    }
}
