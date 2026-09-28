#pragma once
#include "../numeric.cuh"

namespace q45 {
// Sign-extend a BITS-wide two's-complement code (Q4 [-8, 7], Q5 [-16, 15]).
// SCOPE: thread-local; pure.
template<int BITS>
static __device__ __forceinline__ int sign_code(unsigned int value) {
    return (int)value - ((value & (1u << (BITS - 1))) ? (1 << BITS) : 0);
}

// Unpack slot 0 or 1 of a code pair: bits 0-7 hold the two low nibbles and,
// for Q5, bits 8 and 9 hold the two fifth bits. SCOPE: thread-local; pure.
template<int BITS>
static __device__ __forceinline__ int unpack_code(unsigned int pair, unsigned int slot) {
    unsigned int value = (pair >> (slot * 4)) & 15u;
    if (BITS == 5) value |= ((pair >> (8 + slot)) & 1u) << 4;
    return sign_code<BITS>(value);
}

// Two orders exist and are separate numerical contracts; do not interchange them.

// Dequantized weight code * scale, used when a weight is staged for MMA.
static __device__ __forceinline__ float apply_scale(int code, float scale) {
    return (float)code * scale;
}

// SIMT decode contribution in the original FP32 order: (x * code) * scale.
static __device__ __forceinline__ float scaled_product(float x, int code, float scale) {
    return x * (float)code * scale;
}

}  // namespace q45
