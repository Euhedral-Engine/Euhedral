#pragma once
// Shared GDN device helpers: the 128-lane reduction trees and activations.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "common/pdl.cuh"

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
