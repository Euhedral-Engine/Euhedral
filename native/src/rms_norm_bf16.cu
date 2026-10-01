#include "pdl.cuh"
static __device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float((unsigned int)value << 16);
}
static __device__ __forceinline__ unsigned short float_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (unsigned short)(bits >> 16);
}
// Each thread owns the columns col = thread + 128 * i of its row, accumulated in order, and the
// 128 partials reduce through the fixed tree below. Loads are issued in batches of kBatch before
// their squares are accumulated, so the row is not one serial chain of dependent loads.
// gridDim.y slices one row's output across CTAs (decode normalizes a single row): every slice
// recomputes the same reduction in the same order, then writes only its own columns.
static constexpr unsigned int kBatch = 20;
extern "C" __global__ __launch_bounds__(128) void euhedral_rms_norm_bf16(
        const unsigned short* input, const unsigned short* weight, unsigned short* output,
        unsigned int rows, unsigned int width, float epsilon, float weight_offset) {
    euhedral_pdl_begin();
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned short* in = input + (unsigned long long)row * width;
    unsigned short* out = output + (unsigned long long)row * width;
    const unsigned int stride = blockDim.x;
    __shared__ float partial[128];
    float sum = 0.0f;
    unsigned int col = threadIdx.x;
    for (; col + (kBatch - 1) * stride < width; col += kBatch * stride) {
        unsigned short values[kBatch];
#pragma unroll
        for (unsigned int i = 0; i < kBatch; i++) values[i] = in[col + i * stride];
#pragma unroll
        for (unsigned int i = 0; i < kBatch; i++) {
            float value = bf16_to_float(values[i]);
            sum += value * value;
        }
    }
    for (; col < width; col += stride) {
        float value = bf16_to_float(in[col]);
        sum += value * value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int s = 64; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    float inverse = rsqrtf(partial[0] / (float)width + epsilon);
    unsigned int span = (width + gridDim.y - 1) / gridDim.y;
    unsigned int begin = blockIdx.y * span, end = min(width, begin + span);
    for (col = begin + threadIdx.x; col < end; col += stride) {
        float value = bf16_to_float(in[col]) * inverse * (bf16_to_float(weight[col]) + weight_offset);
        out[col] = float_to_bf16(value);
    }
}
