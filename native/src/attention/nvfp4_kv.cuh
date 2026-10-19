#pragma once
// Group-16 4-bit key/value cache codec for D256 attention heads (the NVFP4 cache format: 4-bit codes, one E4M3 scale
// per 16 values).
//
// Storage contract: device page tables address 256-token pages. Within each page,
// token-major KV heads contain 144-byte rows: 128 packed-code bytes followed by
// 16 scale bytes. The low nibble is the even element, and each E4M3 scale covers
// one contiguous group of 16 represented elements.
// Every represented value is exactly magnitude(code) * e4m3(scale), applied to the row AFTER the
// normalized Sylvester H256 rotation. The codes are an INT4 midrise grid: sign bit 3, magnitude m + 0.5 for the low
// three bits m (0.5 to 7.5), the scale chosen from the group's largest value over 7.5. Queries are rotated by the same orthogonal transform, so
// q.k is preserved; attention outputs are rotated back with the same (self-inverse) transform.
//
// Encoding takes the nearest magnitude, ties (the integers) to the larger, saturating at 7.5.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// EUHEDRAL_KV_CODEC: 1 (the default) is the INT4 midrise grid. Measurement builds set another through
// EUHEDRAL_CUDA_KV_CODEC: 0 is E2M1 (magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6, round to nearest even, the scale over 6),
// 2 and 3 the INT3 and INT2 midrise grids (m < 4 and m < 2). A module that includes this file may pin its own.
#ifndef EUHEDRAL_KV_CODEC
#define EUHEDRAL_KV_CODEC 1
#endif

namespace nvfp4kv {

constexpr unsigned int kMidriseLevels = EUHEDRAL_KV_CODEC == 1 ? 8u : EUHEDRAL_KV_CODEC == 2 ? 4u : 2u;

constexpr int kHeadDim = 256;
constexpr int kGroup = 16;
constexpr int kGroups = kHeadDim / kGroup;
constexpr int kCodeBytes = kHeadDim / 2;
constexpr float kMaxFinite = EUHEDRAL_KV_CODEC == 0 ? 6.0f : kMidriseLevels - 0.5f;
constexpr float kScaleMinimum = 1.0f / 512.0f;
constexpr float kScaleMaximum = 448.0f;

// The magnitude of the low three code bits.
__device__ __forceinline__ float code_magnitude(unsigned int code) {
    const unsigned int magnitude = code & 7u;
    if (EUHEDRAL_KV_CODEC != 0) return static_cast<float>(magnitude) + 0.5f;
    // E2M1 bias is one. Codes 0 and 1 are zero and the 0.5 subnormal;
    // normal codes map directly to the FP32 exponent and leading fraction bit.
    // A dynamically indexed local table otherwise emits eight local stores per
    // expansion in the attention loop under NVRTC.
    const unsigned int bits = magnitude < 2u ? magnitude * 0x3f000000u
            : (((magnitude >> 1) + 126u) << 23) | ((magnitude & 1u) << 22);
    return __uint_as_float(bits);
}

// The nearest code, saturating. Midrise grids: ties (integers) go to the larger magnitude. E2M1: ties go to the
// even code (0, 1.0, 2.0, 4.0 are even).
__device__ __forceinline__ unsigned int encode_code(float value) {
    const float a = fabsf(value);
    unsigned int code;
    if (EUHEDRAL_KV_CODEC != 0) {
        code = min(static_cast<unsigned int>(a), kMidriseLevels - 1u);  // nearest (m + 0.5), saturating
        return code | ((__float_as_uint(value) >> 28) & 8u);
    }
    if (a <= 0.25f) code = 0;
    else if (a < 0.75f) code = 1;
    else if (a <= 1.25f) code = 2;
    else if (a < 1.75f) code = 3;
    else if (a <= 2.5f) code = 4;
    else if (a < 3.5f) code = 5;
    else if (a <= 5.0f) code = 6;
    else code = 7;
    // NaN never reaches here: inputs are finite rotated BF16 values divided by a positive scale.
    return code | ((__float_as_uint(value) >> 28) & 8u);
}

__device__ __forceinline__ float decode_code(unsigned int code) {
    const float magnitude = code_magnitude(code);
    return (code & 8u) ? -magnitude : magnitude;
}

__device__ __forceinline__ float e4m3_decode(unsigned char bits) {
    __nv_fp8_e4m3 encoded;
    encoded.__x = bits;
    return static_cast<float>(encoded);
}

struct Group16 {
    unsigned int codes_lo;  // elements 0..7
    unsigned int codes_hi;  // elements 8..15
    unsigned char scale;
};

// Quantizes 16 contiguous FP32 values (already rotated) into codes plus one E4M3 scale.
__device__ __forceinline__ Group16 quantize_group16(const float* source) {
    float max_abs = 0.0f;
#pragma unroll
    for (int i = 0; i < kGroup; i++) max_abs = fmaxf(max_abs, fabsf(source[i]));
    Group16 result{0u, 0u, 0};
    if (max_abs == 0.0f) return result;
    const float raw = __fdiv_rn(max_abs, kMaxFinite);
    const float bounded = fminf(kScaleMaximum, fmaxf(kScaleMinimum, raw));
    __nv_fp8_e4m3 scale(bounded);  // round-to-nearest, saturating finite
    result.scale = scale.__x;
    const float represented = static_cast<float>(scale);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        result.codes_lo |= encode_code(__fdiv_rn(source[i], represented)) << (4 * i);
        result.codes_hi |= encode_code(__fdiv_rn(source[8 + i], represented)) << (4 * i);
    }
    return result;
}

// Expands eight packed codes into FP16 pairs. Every code magnitude times a legal E4M3 cache scale is exactly
// representable in FP16 (at most eight significant bits, 2^-10 to 3360), so this is exact.
__device__ __forceinline__ void dequantize8_f16(unsigned int codes, float scale, __half2 (&out)[4]) {
#pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        const float lo = decode_code((codes >> (8 * pair)) & 0xFu) * scale;
        const float hi = decode_code((codes >> (8 * pair + 4)) & 0xFu) * scale;
        out[pair] = __floats2half2_rn(lo, hi);
    }
}

// One warp owns one D256 row; lane l holds dimensions l + 32*r in values[r]. The transform is the
// normalized Sylvester H256 (entries +-1/16), applied in FP32 with a fixed butterfly order.
__device__ __forceinline__ void hadamard256(float (&values)[8], unsigned int lane) {
#pragma unroll
    for (int stride = 1; stride <= 16; stride <<= 1) {
#pragma unroll
        for (int r = 0; r < 8; r++) {
            const float value = values[r];
            const float peer = __shfl_xor_sync(0xffffffffu, value, stride);
            values[r] = (lane & stride) == 0 ? __fadd_rn(value, peer) : __fsub_rn(peer, value);
        }
    }
#pragma unroll
    for (int span = 1; span < 8; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 8; base += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; offset++) {
                const float low = values[base + offset];
                const float high = values[base + offset + span];
                values[base + offset] = __fadd_rn(low, high);
                values[base + offset + span] = __fsub_rn(low, high);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 8; r++) values[r] = __fmul_rn(values[r], 0.0625f);
}

// Rotates and quantizes one D256 row owned by a warp. `scratch` holds 256 floats for this warp.
__device__ __forceinline__ void quantize_row(
        const __nv_bfloat16* source, unsigned char* codes, unsigned char* scales, float* scratch,
        unsigned int lane) {
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; r++) values[r] = __bfloat162float(source[lane + 32 * r]);
    hadamard256(values, lane);
#pragma unroll
    for (int r = 0; r < 8; r++) scratch[lane + 32 * r] = values[r];
    __syncwarp();
    if (lane < kGroups) {
        const Group16 quantized = quantize_group16(scratch + lane * kGroup);
        *reinterpret_cast<uint2*>(codes + lane * (kGroup / 2)) = make_uint2(quantized.codes_lo, quantized.codes_hi);
        scales[lane] = quantized.scale;
    }
    __syncwarp();
}

}  // namespace nvfp4kv
