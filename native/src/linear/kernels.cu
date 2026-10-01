#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "common/pdl.cuh"

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;


__device__ __forceinline__ int qwen_quantized_value(
        const uint8_t* weights, uint32_t row, uint32_t k, uint32_t groups, uint32_t outputs, uint32_t bits) {
    const uint32_t group = k / 64;
    const uint32_t lane = k & 63;
    const uint64_t groupIndex = static_cast<uint64_t>(row) * groups + group;
    const uint8_t packed = weights[groupIndex * 32 + (lane >> 1)];
    int value = (packed >> ((lane & 1) * 4)) & 0x0f;
    if (bits == 5) {
        const uint64_t codeBytes = static_cast<uint64_t>(outputs) * groups * 32;
        const uint64_t highOffset = (codeBytes + 255) & ~255ull;
        const uint8_t high = weights[highOffset + groupIndex * 8 + (lane >> 3)];
        value |= ((high >> (lane & 7)) & 1) << 4;
        if (value >= 16) value -= 32;
    } else if (value >= 8) {
        value -= 16;
    }
    return value;
}

__device__ __forceinline__ float qwen_quantized_scale(
        const uint8_t* weights, uint32_t row, uint32_t group, uint32_t groups, uint32_t outputs, uint32_t bits) {
    const uint64_t codeBytes = static_cast<uint64_t>(outputs) * groups * 32;
    const uint64_t codePlaneBytes = (codeBytes + 255) & ~255ull;
    const uint64_t highBytes = bits == 5 ? static_cast<uint64_t>(outputs) * groups * 8 : 0;
    const uint64_t highPlaneBytes = (highBytes + 255) & ~255ull;
    const uint64_t scaleOffset = codePlaneBytes + highPlaneBytes;
    const uint64_t scaleIndex = static_cast<uint64_t>(row) * groups + group;
    return __half2float(reinterpret_cast<const __half*>(weights + scaleOffset)[scaleIndex]);
}

__device__ __forceinline__ float qwen_reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

extern "C" __global__ void euhedral_linear_quantized_bf16(
        const __nv_bfloat16* input,
        const uint8_t* weights,
        __nv_bfloat16* output,
        uint32_t rows,
        uint32_t inFeatures,
        uint32_t outFeatures,
        uint32_t bits) {
    // Block-stride over outputs: rows * outFeatures may exceed the grid limit, so
    // the host caps the grid and each block computes every gridDim.x-th output.
    // The loop bound is block-uniform, so the reduction barriers stay uniform.
    const uint64_t count = static_cast<uint64_t>(rows) * outFeatures;
    const uint32_t groups = inFeatures / 64;
    __shared__ float scratch[128];
    for (uint64_t outputIndex = blockIdx.x; outputIndex < count; outputIndex += gridDim.x) {
        const uint32_t row = static_cast<uint32_t>(outputIndex / outFeatures);
        const uint32_t outputColumn = static_cast<uint32_t>(outputIndex % outFeatures);
        const __nv_bfloat16* activation = input + static_cast<uint64_t>(row) * inFeatures;
        float partial = 0.0f;
        for (uint32_t k = threadIdx.x; k < inFeatures; k += blockDim.x) {
            const int q = qwen_quantized_value(weights, outputColumn, k, groups, outFeatures, bits);
            const float scale = qwen_quantized_scale(weights, outputColumn, k / 64, groups, outFeatures, bits);
            partial = fmaf(__bfloat162float(activation[k]), static_cast<float>(q) * scale, partial);
        }
        const float sum = qwen_reduce_sum(partial, scratch);
        if (threadIdx.x == 0) output[outputIndex] = __float2bfloat16_rn(sum);
    }
}

extern "C" __global__ void euhedral_linear_bf16_to_float(
        const __nv_bfloat16* input,
        const __nv_bfloat16* weights,
        float* output,
        uint32_t rows,
        uint32_t inFeatures,
        uint32_t outFeatures) {
    euhedral_pdl_begin();
    const uint64_t outputIndex = static_cast<uint64_t>(blockIdx.x);
    if (outputIndex >= static_cast<uint64_t>(rows) * outFeatures) return;
    const uint32_t row = static_cast<uint32_t>(outputIndex / outFeatures);
    const uint32_t outputColumn = static_cast<uint32_t>(outputIndex % outFeatures);
    const __nv_bfloat16* activation = input + static_cast<uint64_t>(row) * inFeatures;
    const __nv_bfloat16* weight = weights + static_cast<uint64_t>(outputColumn) * inFeatures;
    // In-order FMA chain per thread; operands are loaded in batches so the chain does not wait on
    // one dependent load per element.
    constexpr uint32_t kBatch = 20;
    const uint32_t stride = blockDim.x;
    float partial = 0.0f;
    uint32_t k = threadIdx.x;
    for (; k + (kBatch - 1) * stride < inFeatures; k += kBatch * stride) {
        __nv_bfloat16 a[kBatch], w[kBatch];
#pragma unroll
        for (uint32_t i = 0; i < kBatch; i++) {
            a[i] = activation[k + i * stride];
            w[i] = weight[k + i * stride];
        }
#pragma unroll
        for (uint32_t i = 0; i < kBatch; i++) partial = fmaf(__bfloat162float(a[i]), __bfloat162float(w[i]), partial);
    }
    for (; k < inFeatures; k += stride) {
        partial = fmaf(__bfloat162float(activation[k]), __bfloat162float(weight[k]), partial);
    }
    __shared__ float scratch[128];
    const float sum = qwen_reduce_sum(partial, scratch);
    if (threadIdx.x == 0) output[outputIndex] = sum;
}
