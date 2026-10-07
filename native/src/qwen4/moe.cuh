#pragma once
#include "qwen4/common.cuh"

// Routing and the shared expert of the Flash-Next MoE block (Qwen4ExpTextSparseMoeBlock).

// The router's selection for each row from its BF16 logits, as Qwen4ExpTextTopKRouter computes it:
//   p = softmax(float(logits)), the k largest p (descending; equal values: the LOWER expert first, a deterministic
//   rule where torch.topk's tie order is unspecified), renormalized by their sum in FP32 and rounded to BF16.
// ids[r][t] is the t-th expert of row r and weights[r][t] its BF16 weight bits.
//   grid rows, block 256; experts <= 1024.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_router_bf16(
        const unsigned short* __restrict__ logits, int* __restrict__ ids, unsigned short* __restrict__ weights,
        unsigned int rows, unsigned int experts, unsigned int k) {
    __shared__ float probability[1024];
    __shared__ float scratch[32];
    __shared__ float best_value[8];
    __shared__ int best_index[8];
    const unsigned int row = blockIdx.x;
    if (row >= rows || experts > 1024u) return;
    const unsigned short* in = logits + (unsigned long long)row * experts;
    float maximum = -3.0e38f;
    for (unsigned int i = threadIdx.x; i < experts; i += blockDim.x) maximum = fmaxf(maximum, q4::bf(in[i]));
    // Block maximum through the warp tree and the scratch.
    for (int offset = 16; offset > 0; offset >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    if ((threadIdx.x & 31u) == 0) scratch[threadIdx.x >> 5] = maximum;
    __syncthreads();
    maximum = scratch[0];
    for (unsigned int w = 1; w < (blockDim.x + 31u) / 32u; w++) maximum = fmaxf(maximum, scratch[w]);
    __syncthreads();
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < experts; i += blockDim.x) {
        const float e = expf(q4::bf(in[i]) - maximum);
        probability[i] = e;
        sum += e;
    }
    sum = q4::block_sum(sum, scratch);
    for (unsigned int i = threadIdx.x; i < experts; i += blockDim.x) probability[i] = probability[i] / sum;
    __syncthreads();
    float selected[16];
    for (unsigned int t = 0; t < k; t++) {
        float value = -1.0f;
        int index = 0x7fffffff;
        for (unsigned int i = threadIdx.x; i < experts; i += blockDim.x) {
            const float p = probability[i];
            if (p > value) {  // ascending scan per thread: the first (lowest) index keeps a tie
                value = p;
                index = (int)i;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value = __shfl_xor_sync(0xffffffffu, value, offset);
            const int other_index = __shfl_xor_sync(0xffffffffu, index, offset);
            if (other_value > value || (other_value == value && other_index < index)) {
                value = other_value;
                index = other_index;
            }
        }
        if ((threadIdx.x & 31u) == 0) {
            best_value[threadIdx.x >> 5] = value;
            best_index[threadIdx.x >> 5] = index;
        }
        __syncthreads();
        value = best_value[0];
        index = best_index[0];
        for (unsigned int w = 1; w < (blockDim.x + 31u) / 32u; w++) {
            if (best_value[w] > value || (best_value[w] == value && best_index[w] < index)) {
                value = best_value[w];
                index = best_index[w];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            ids[(unsigned long long)row * k + t] = index;
            probability[index] = -2.0f;  // out of the next rounds
        }
        selected[t] = value;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (unsigned int t = 0; t < k; t++) total += selected[t];
        for (unsigned int t = 0; t < k; t++) weights[(unsigned long long)row * k + t] = q4::bfr(selected[t] / total);
    }
}

// out = bf16(bf16(silu(gate)) * up): the SwiGLU of Qwen4ExpTextMLP on separate gate and up outputs.
extern "C" __global__ void euhedral_q4_swiglu_bf16(
        const unsigned short* __restrict__ gate, const unsigned short* __restrict__ up,
        unsigned short* __restrict__ output, unsigned int count) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = q4::bfr(q4::round_bf(q4::silu(q4::bf(gate[i]))) * q4::bf(up[i]));
}

// The shared expert's SwiGLU (as euhedral_q4_swiglu_bf16) in rows of `padded` values: columns at and past `cols` are 0,
// the input of a down projection whose weights are padded to `padded` columns (euhedral_q4_nvfp4_pad_k).
//   grid (ceil(padded / 256), rows), block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_swiglu_padded_bf16(
        const unsigned short* __restrict__ gate, const unsigned short* __restrict__ up,
        unsigned short* __restrict__ output, unsigned int cols, unsigned int padded) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= padded) return;
    const unsigned long long row = blockIdx.y;
    unsigned short value = 0;
    if (c < cols) {
        const unsigned long long i = row * cols + c;
        value = q4::bfr(q4::round_bf(q4::silu(q4::bf(gate[i]))) * q4::bf(up[i]));
    }
    output[row * padded + c] = value;
}

// Copies a plain NVFP4 tensor (nvfp4/nvfp4.cuh) of `rows` rows of `k` values into one of `padded` values per row: the
// new columns have code 0 and scale 0, so every product they add is exactly 0. `k` and `padded` are multiples of 128.
//   grid rows (+ 1: the last block copies the global scale), block 128.
extern "C" __global__ __launch_bounds__(128) void euhedral_q4_nvfp4_pad_k(
        const unsigned char* __restrict__ source, unsigned char* __restrict__ target, unsigned int rows,
        unsigned int k, unsigned int padded) {
    const unsigned long long row = blockIdx.x;
    const unsigned int row_bytes = k / 2u, padded_bytes = padded / 2u, row_scales = k / 16u, padded_scales = padded / 16u;
    const unsigned long long source_scales = ((unsigned long long)rows * row_bytes + 255ull) & ~255ull;
    const unsigned long long target_scales = ((unsigned long long)rows * padded_bytes + 255ull) & ~255ull;
    if (row == rows) {
        if (threadIdx.x < 4) {
            const unsigned long long source_global = (source_scales + (unsigned long long)rows * row_scales + 255ull) & ~255ull;
            const unsigned long long target_global =
                    (target_scales + (unsigned long long)rows * padded_scales + 255ull) & ~255ull;
            target[target_global + threadIdx.x] = source[source_global + threadIdx.x];
        }
        return;
    }
    for (unsigned int b = threadIdx.x; b < padded_bytes; b += blockDim.x)
        target[row * padded_bytes + b] = b < row_bytes ? source[row * row_bytes + b] : 0;
    for (unsigned int b = threadIdx.x; b < padded_scales; b += blockDim.x)
        target[target_scales + row * padded_scales + b] = b < row_scales ? source[source_scales + row * row_scales + b] : 0;
}

// The MoE block's result: out = bf16(routed + bf16(bf16(sigmoid(gate)) * shared)), `gate` being the shared expert
// gate's raw BF16 projection, one value per row.
//   grid (ceil(width / 256), rows), block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_moe_finish_bf16(
        const unsigned short* __restrict__ routed, const unsigned short* __restrict__ shared,
        const unsigned short* __restrict__ gate, unsigned short* __restrict__ output,
        unsigned int rows, unsigned int width) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y;
    if (d >= width || row >= rows) return;
    const unsigned long long at = (unsigned long long)row * width + d;
    const float scale = q4::round_bf(q4::sigmoid(q4::bf(gate[row])));
    output[at] = q4::bfr(q4::bf(routed[at]) + q4::round_bf(scale * q4::bf(shared[at])));
}
