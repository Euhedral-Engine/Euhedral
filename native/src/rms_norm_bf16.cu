#include "pdl.cuh"
static __device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float((unsigned int)value << 16);
}
static __device__ __forceinline__ unsigned short float_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (unsigned short)(bits >> 16);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_rms_norm_bf16(
        const unsigned short* input, const unsigned short* weight, unsigned short* output,
        unsigned int rows, unsigned int width, float epsilon, float weight_offset) {
    euhedral_pdl_begin();
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partial[128];
    float sum = 0.0f;
    for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
        float value = bf16_to_float(input[(unsigned long long)row * width + col]);
        sum += value * value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(partial[0] / (float)width + epsilon);
    for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
        unsigned long long index = (unsigned long long)row * width + col;
        float value = bf16_to_float(input[index]) * inverse * (bf16_to_float(weight[col]) + weight_offset);
        output[index] = float_to_bf16(value);
    }
}
