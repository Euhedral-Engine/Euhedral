#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "common/pdl.cuh"

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;


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
