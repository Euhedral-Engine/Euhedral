#pragma once
#include "qwen4/common.cuh"

// BF16 linear: out[r][j] = bf16(sum_k in[r][k] * w[j][k]) with FP32 accumulation, the arithmetic of a BF16
// torch.nn.Linear without bias. A warp owns one output column and up to 8 rows: the weight row is read once per
// 8 rows, 16 bytes per lane per step, so one-row decode streams the weights at memory speed.
//   grid  (ceil(n / 8), ceil(rows / 8)), block 256; k must be a multiple of 8 and all pointers 16-byte aligned.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_linear_bf16(
        const unsigned short* __restrict__ input, const unsigned short* __restrict__ weights,
        unsigned short* __restrict__ output, unsigned int rows, unsigned int k, unsigned int n) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int column = blockIdx.x * 8u + warp;
    if (column >= n) return;
    const unsigned int row0 = blockIdx.y * 8u;
    const unsigned int active = min(8u, rows - row0);
    float accumulator[8];
#pragma unroll
    for (int r = 0; r < 8; r++) accumulator[r] = 0.0f;
    const uint4* weight_row = reinterpret_cast<const uint4*>(weights + (unsigned long long)column * k);
    const unsigned int vectors = k >> 3;
    for (unsigned int v = lane; v < vectors; v += 32u) {
        const uint4 packed = weight_row[v];
        float w[8];
        w[0] = __uint_as_float(packed.x << 16); w[1] = __uint_as_float(packed.x & 0xffff0000u);
        w[2] = __uint_as_float(packed.y << 16); w[3] = __uint_as_float(packed.y & 0xffff0000u);
        w[4] = __uint_as_float(packed.z << 16); w[5] = __uint_as_float(packed.z & 0xffff0000u);
        w[6] = __uint_as_float(packed.w << 16); w[7] = __uint_as_float(packed.w & 0xffff0000u);
#pragma unroll
        for (int r = 0; r < 8; r++) {
            if ((unsigned int)r < active) {
                const uint4 x = reinterpret_cast<const uint4*>(input + (unsigned long long)(row0 + r) * k)[v];
                float a = accumulator[r];
                a = fmaf(w[0], __uint_as_float(x.x << 16), a); a = fmaf(w[1], __uint_as_float(x.x & 0xffff0000u), a);
                a = fmaf(w[2], __uint_as_float(x.y << 16), a); a = fmaf(w[3], __uint_as_float(x.y & 0xffff0000u), a);
                a = fmaf(w[4], __uint_as_float(x.z << 16), a); a = fmaf(w[5], __uint_as_float(x.z & 0xffff0000u), a);
                a = fmaf(w[6], __uint_as_float(x.w << 16), a); a = fmaf(w[7], __uint_as_float(x.w & 0xffff0000u), a);
                accumulator[r] = a;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 8; r++) {
        const float total = q4::warp_sum(accumulator[r]);
        if (lane == 0 && (unsigned int)r < active) output[(unsigned long long)(row0 + r) * n + column] = q4::bfr(total);
    }
}

// Embedding rows: out[r] = table[ids[r]] for rows of `width` BF16 values (a multiple of 8, 16-byte aligned rows).
// The table may be device memory or host memory mapped into the device's address space.
//   grid rows, block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_embedding_bf16(
        const unsigned short* __restrict__ table, const int* __restrict__ ids, unsigned short* __restrict__ output,
        unsigned int rows, unsigned int width, unsigned int vocabulary) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int token = (unsigned int)ids[row];
    const uint4* source = reinterpret_cast<const uint4*>(table + (unsigned long long)min(token, vocabulary - 1u) * width);
    uint4* destination = reinterpret_cast<uint4*>(output + (unsigned long long)row * width);
    for (unsigned int i = threadIdx.x; i < (width >> 3); i += blockDim.x) destination[i] = source[i];
}
