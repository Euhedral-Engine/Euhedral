#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace q45 {
// Conversions keep the original Q4/Q5 kernels' intrinsics so every route stays
// bitwise identical to them (including NaN canonicalization by __half2float).
static __device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float((unsigned int)value << 16);
}
static __device__ __forceinline__ float scale_to_float(unsigned short value) {
    return __half2float(__ushort_as_half(value));
}
static __device__ __forceinline__ unsigned short float_to_bf16(float value) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}
// Write one FP32 result as BF16 (round-to-nearest-even). Bounds are the caller's.
static __device__ __forceinline__ void write_bf16(
        unsigned short* output, unsigned int row, unsigned int col, unsigned int out_features, float value) {
    output[(unsigned long long)row * out_features + col] = float_to_bf16(value);
}

}  // namespace q45
