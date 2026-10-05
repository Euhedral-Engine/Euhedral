#pragma once
// Shared helpers of the Flash-Next (qwen4_exp) kernels. Every kernel rounds to BF16 where the upstream PyTorch
// model does: an elementwise op on a BF16 tensor computes in FP32 and rounds its result to BF16, so a chain of
// ops is a chain of roundings, reproduced here step by step (docs/FLASH_NEXT_EXECUTION.md).
namespace q4 {

static __device__ __forceinline__ float bf(unsigned short value) {
    return __uint_as_float((unsigned int)value << 16);
}

// Round to nearest even; NaN stays NaN.
static __device__ __forceinline__ unsigned short bfr(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((bits >> 16) | 0x40u);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (unsigned short)(bits >> 16);
}

// The value `bfr` would give, as a float: the result of a BF16 op, kept in a register.
static __device__ __forceinline__ float round_bf(float value) {
    return bf(bfr(value));
}

static __device__ __forceinline__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

static __device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

// Sum of the 32 lanes in a fixed tree.
static __device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

// Block-wide sum for blocks of up to 1024 threads; every thread returns the total. `scratch` has 32 floats.
static __device__ __forceinline__ float block_sum(float value, float* scratch) {
    value = warp_sum(value);
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    const unsigned int warps = (blockDim.x + 31u) >> 5;
    float total = lane < warps ? scratch[lane] : 0.0f;
    total = warp_sum(total);
    return total;
}

}  // namespace q4
