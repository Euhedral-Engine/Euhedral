#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace q3 {
static __device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float((unsigned int)value << 16);
}
static __device__ __forceinline__ float fp16_to_float(unsigned short value) {
    float converted = __half2float(__ushort_as_half(value));
    // The intrinsic canonicalizes NaNs; retain the original sign and payload.
    if ((value & 0x7c00u) == 0x7c00u && (value & 0x03ffu) != 0u)
        return __uint_as_float(((unsigned int)(value & 0x8000u) << 16) | 0x7f800000u
                | ((unsigned int)(value & 0x03ffu) << 13));
    return converted;
}
static __device__ __forceinline__ unsigned short float_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (unsigned short)(bits >> 16);
}


}  // namespace q3
