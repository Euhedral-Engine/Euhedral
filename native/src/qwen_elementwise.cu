#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "pdl.cuh"

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;


__device__ __forceinline__ float euhedral_silu(float value) {
    return value / (1.0f + expf(-value));
}

extern "C" __global__ void euhedral_residual_add_bf16(
        const __nv_bfloat16* residual, const __nv_bfloat16* delta,
        __nv_bfloat16* output, uint32_t elementCount) {
    euhedral_pdl_begin();
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elementCount) {
        output[index] = __float2bfloat16_rn(
                __bfloat162float(residual[index]) + __bfloat162float(delta[index]));
    }
}

extern "C" __global__ void euhedral_swiglu_bf16(
        const __nv_bfloat16* gateUp, __nv_bfloat16* output,
        uint32_t rows, uint32_t intermediateSize) {
    euhedral_pdl_begin();
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(rows) * intermediateSize;
    if (index >= count) return;
    const uint32_t row = static_cast<uint32_t>(index / intermediateSize);
    const uint32_t column = static_cast<uint32_t>(index % intermediateSize);
    const uint64_t rowOffset = static_cast<uint64_t>(row) * intermediateSize * 2;
    const float gate = __bfloat162float(gateUp[rowOffset + column]);
    const float up = __bfloat162float(gateUp[rowOffset + intermediateSize + column]);
    output[index] = __float2bfloat16_rn(euhedral_silu(gate) * up);
}

// Keep the residual BF16 rounding and the original 128-lane RMS reduction tree.
// The residual remains live for the FFN skip path; only its independent completion disappears.
extern "C" __global__ __launch_bounds__(128) void euhedral_residual_rms_norm_bf16(
        const __nv_bfloat16* residual, const __nv_bfloat16* delta, const __nv_bfloat16* weight,
        __nv_bfloat16* hidden, unsigned short* normalized, uint32_t rows, uint32_t width, float epsilon) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint64_t base = static_cast<uint64_t>(row) * width;
    __shared__ float partial[128];
    float sum = 0.0f;
    for (uint32_t col = threadIdx.x; col < width; col += blockDim.x) {
        const __nv_bfloat16 rounded = __float2bfloat16_rn(
                __bfloat162float(residual[base + col]) + __bfloat162float(delta[base + col]));
        hidden[base + col] = rounded;
        const float value = __bfloat162float(rounded);
        sum += value * value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(partial[0] / (float)width + epsilon);
    for (uint32_t col = threadIdx.x; col < width; col += blockDim.x) {
        const float value = __bfloat162float(hidden[base + col]) * inverse * (__bfloat162float(weight[col]) + 1.0f);
        // RMSNorm's existing store is intentionally not the CUDA NaN-canonicalizing intrinsic.
        uint32_t bits = __float_as_uint(value);
        bits += 0x7fffu + ((bits >> 16) & 1u);
        normalized[base + col] = (unsigned short)(bits >> 16);
    }
}
