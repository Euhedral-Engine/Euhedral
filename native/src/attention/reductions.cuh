#pragma once
// Shared helpers of the BF16 attention leaves.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using uint32_t = unsigned int;
using uint64_t = unsigned long long;

__device__ __forceinline__ float attention_reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}
