#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "q3/layout.cuh"
#include "q3/numeric.cuh"
#include "nvfp4/nvfp4.cuh"

// Scalar numerical references: one CTA per output element, threads stride K, a 128-thread tree
// reduction, FP32 throughout. They are the oracle that exact numerics select (see
// euhedral_cuda_select_exact_numerics) and that the production kernels are measured against; no
// production route launches them.

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// Q3G64_F16S. One CTA per (row, output).
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_reference(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    __shared__ float partial[128];
    const unsigned int linear = blockIdx.x;
    const unsigned int row = linear / out_features, out = linear % out_features;
    if (row >= rows) return;
    const q3::Layout w(weights, in_features, scale_offset);
    float sum = 0.0f;
    for (unsigned int k = threadIdx.x; k < in_features; k += blockDim.x) {
        const unsigned long long g = w.group(out, k / q3::kGroup);
        // Code index within the group; reads byte (index * 3) / 8 + 1, one past the group for code 63.
        const unsigned char* bytes = w.group_bytes(g);
        const unsigned int bit = (k % q3::kGroup) * 3u;
        const unsigned int word = (unsigned int)bytes[bit >> 3] | ((unsigned int)bytes[(bit >> 3) + 1u] << 8);
        const unsigned int field = (word >> (bit & 7u)) & 7u;
        const int code = (int)field - (field >= 4u ? 8 : 0);
        // The scale index is formed in 32-bit arithmetic, which wraps once a single tensor reaches
        // 2^32 groups (>= 104 GiB).
        const unsigned int scale_index = out * w.groups + k / q3::kGroup;
        const float scale = q3::fp16_to_float(w.scales[scale_index]);
        sum = sum + q3::bf16_to_float(input[(unsigned long long)row * in_features + k]) * (float)code * scale;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) q3::write_bf16(output, row, out, out_features, partial[0]);
}

// Q4 and Q5 (`bits`). The host caps the grid, so each block computes every gridDim.x-th output.
__device__ __forceinline__ int qwen_quantized_value(
        const uint8_t* weights, uint32_t row, uint32_t k, uint32_t groups, uint32_t outputs, uint32_t bits) {
    const uint32_t group = k / 64;
    const uint32_t lane = k & 63;
    const uint64_t groupIndex = static_cast<uint64_t>(row) * groups + group;
    const uint8_t packed = weights[groupIndex * 32 + (lane >> 1)];
    int value = (packed >> ((lane & 1) * 4)) & 0x0f;
    if (bits == 5) {
        const uint64_t codeBytes = static_cast<uint64_t>(outputs) * groups * 32;
        const uint64_t highOffset = (codeBytes + 255) & ~255ull;
        const uint8_t high = weights[highOffset + groupIndex * 8 + (lane >> 3)];
        value |= ((high >> (lane & 7)) & 1) << 4;
        if (value >= 16) value -= 32;
    } else if (value >= 8) {
        value -= 16;
    }
    return value;
}

__device__ __forceinline__ float qwen_quantized_scale(
        const uint8_t* weights, uint32_t row, uint32_t group, uint32_t groups, uint32_t outputs, uint32_t bits) {
    const uint64_t codeBytes = static_cast<uint64_t>(outputs) * groups * 32;
    const uint64_t codePlaneBytes = (codeBytes + 255) & ~255ull;
    const uint64_t highBytes = bits == 5 ? static_cast<uint64_t>(outputs) * groups * 8 : 0;
    const uint64_t highPlaneBytes = (highBytes + 255) & ~255ull;
    const uint64_t scaleOffset = codePlaneBytes + highPlaneBytes;
    const uint64_t scaleIndex = static_cast<uint64_t>(row) * groups + group;
    return __half2float(reinterpret_cast<const __half*>(weights + scaleOffset)[scaleIndex]);
}

__device__ __forceinline__ float reference_reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

extern "C" __global__ void euhedral_q45_reference(
        const __nv_bfloat16* input,
        const uint8_t* weights,
        __nv_bfloat16* output,
        uint32_t rows,
        uint32_t inFeatures,
        uint32_t outFeatures,
        uint32_t bits) {
    // Block-stride over outputs: rows * outFeatures may exceed the grid limit, so
    // the host caps the grid and each block computes every gridDim.x-th output.
    // The loop bound is block-uniform, so the reduction barriers stay uniform.
    const uint64_t count = static_cast<uint64_t>(rows) * outFeatures;
    const uint32_t groups = inFeatures / 64;
    __shared__ float scratch[128];
    for (uint64_t outputIndex = blockIdx.x; outputIndex < count; outputIndex += gridDim.x) {
        const uint32_t row = static_cast<uint32_t>(outputIndex / outFeatures);
        const uint32_t outputColumn = static_cast<uint32_t>(outputIndex % outFeatures);
        const __nv_bfloat16* activation = input + static_cast<uint64_t>(row) * inFeatures;
        float partial = 0.0f;
        for (uint32_t k = threadIdx.x; k < inFeatures; k += blockDim.x) {
            const int q = qwen_quantized_value(weights, outputColumn, k, groups, outFeatures, bits);
            const float scale = qwen_quantized_scale(weights, outputColumn, k / 64, groups, outFeatures, bits);
            partial = fmaf(__bfloat162float(activation[k]), static_cast<float>(q) * scale, partial);
        }
        const float sum = reference_reduce_sum(partial, scratch);
        if (threadIdx.x == 0) output[outputIndex] = __float2bfloat16_rn(sum);
    }
}


// NVFP4 (`sd4` selects the table-indexed scale layout): weight = e2m1(code) * e4m3(scale), the FP32
// global scale applied to the sum, activations in BF16.
template <bool kSd4>
static __device__ __forceinline__ void nvfp4_reference(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features) {
    __shared__ float scratch[128];
    __shared__ float scale_table[16];
    const nvfp4::LayoutT<kSd4> w(weights, in_features, out_features);
    if (kSd4 && threadIdx.x < 16u) scale_table[threadIdx.x] = nvfp4::e4m3_to_float(w.table.lookup(threadIdx.x));
    __syncthreads();
    const unsigned long long count = (unsigned long long)rows * out_features;
    for (unsigned long long index = blockIdx.x; index < count; index += gridDim.x) {
        const unsigned int row = (unsigned int)(index / out_features), column = (unsigned int)(index % out_features);
        float partial = 0.0f;
        for (unsigned int k = threadIdx.x; k < in_features; k += blockDim.x) {
            const unsigned int code = (w.codes[(unsigned long long)column * w.row_bytes + k / 2u] >> ((k & 1u) * 4u)) & 15u;
            const unsigned int block = k / nvfp4::kBlock;
            const unsigned int pair = w.scale_bits(column, block / 2u);
            const float scale = nvfp4::scale<kSd4>(scale_table, pair, block & 1u);
            partial = fmaf(q3::bf16_to_float(input[(unsigned long long)row * in_features + k]),
                    nvfp4::e2m1_value(code) * scale, partial);
        }
        const float sum = reference_reduce_sum(partial, scratch);
        if (threadIdx.x == 0) q3::write_bf16(output, row, column, out_features, sum * w.global);
    }
}
extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_reference(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features) {
    nvfp4_reference<false>(input, weights, output, rows, in_features, out_features);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_reference_sd4(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features) {
    nvfp4_reference<true>(input, weights, output, rows, in_features, out_features);
}
