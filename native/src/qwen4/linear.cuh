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

// BF16 linear on m16n8k16 tensor cores (the scheme of dflash's linear, docs/DFLASH2.md): out = bf16(sum_k in * w)
// with FP32 accumulation, in a different summation order from euhedral_q4_linear_bf16, which stays the exact route.
// A CTA's four warps own 32 output columns over the whole K (KS = 1), or 8 columns with K in four contiguous parts
// of whole 32-wide chunks that add in warp order (KS = 4, for outputs of at most 1536). Per 32-wide K chunk a lane
// loads 16 bytes of its column's weight row and of its rows' activations and feeds two MMAs the same permutation of K
// on both operands. A row's accumulation is fixed by the shape (k, n, KS) alone: MT, the row tiles of 16 that reuse
// a weight load, never changes its bits, so decode and prefill rows agree.
//   grid (n / (KS == 1 ? 32 : 8), ceil(rows / (16 * MT))), block 128; k a multiple of 32, n of 32 (KS 1) or 8.
namespace q4tc {

static __device__ __forceinline__ void mma_bf16(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ uint4 load_row(const unsigned short* row, unsigned int k, bool valid) {
    if (!valid) return make_uint4(0, 0, 0, 0);
    return __ldg(reinterpret_cast<const uint4*>(row + k));
}

template <int MT, int UNROLL, int KS>
static __device__ __forceinline__ void linear(
        const unsigned short* __restrict__ x, const unsigned short* __restrict__ w, unsigned short* __restrict__ y,
        unsigned int rows, unsigned int k, unsigned int n) {
    const unsigned int warp = threadIdx.x / 32u, lane = threadIdx.x % 32u;
    const unsigned int g = lane / 4u, q = lane % 4u;
    const unsigned int part = KS == 1 ? 0u : warp;
    const unsigned int column0 = KS == 1 ? blockIdx.x * 32u + warp * 8u : blockIdx.x * 8u;
    const unsigned int rowBase = blockIdx.y * 16u * MT;
    const unsigned int begin = KS == 1 ? 0u : k / 32u * part / KS * 32u;
    const unsigned int end = KS == 1 ? k : k / 32u * (part + 1u) / KS * 32u;
    const unsigned short* weightRow = w + (unsigned long long)(column0 + g) * k;
    const unsigned short* rowPointer[MT][2];
    bool rowValid[MT][2];
#pragma unroll
    for (int t = 0; t < MT; t++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const unsigned int row = rowBase + t * 16u + g + h * 8u;
            rowValid[t][h] = row < rows;
            rowPointer[t][h] = x + (unsigned long long)(rowValid[t][h] ? row : 0u) * k;
        }
    float acc[MT][4];
#pragma unroll
    for (int t = 0; t < MT; t++) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
    unsigned int chunk = begin;
    for (; chunk + 32u * UNROLL <= end; chunk += 32u * UNROLL) {
        uint4 b[UNROLL];
        uint4 a[UNROLL][MT][2];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const unsigned int at = chunk + u * 32u + q * 8u;
            b[u] = __ldg(reinterpret_cast<const uint4*>(weightRow + at));
#pragma unroll
            for (int t = 0; t < MT; t++)
#pragma unroll
                for (int h = 0; h < 2; h++) a[u][t][h] = load_row(rowPointer[t][h], at, rowValid[t][h]);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; u++)
#pragma unroll
            for (int t = 0; t < MT; t++) {
                const unsigned int step0[4] = {a[u][t][0].x, a[u][t][1].x, a[u][t][0].y, a[u][t][1].y};
                mma_bf16(acc[t], step0, b[u].x, b[u].y);
                const unsigned int step1[4] = {a[u][t][0].z, a[u][t][1].z, a[u][t][0].w, a[u][t][1].w};
                mma_bf16(acc[t], step1, b[u].z, b[u].w);
            }
    }
    for (; chunk < end; chunk += 32u) {
        const unsigned int at = chunk + q * 8u;
        const uint4 b = __ldg(reinterpret_cast<const uint4*>(weightRow + at));
#pragma unroll
        for (int t = 0; t < MT; t++) {
            const uint4 a0 = load_row(rowPointer[t][0], at, rowValid[t][0]);
            const uint4 a1 = load_row(rowPointer[t][1], at, rowValid[t][1]);
            const unsigned int step0[4] = {a0.x, a1.x, a0.y, a1.y};
            mma_bf16(acc[t], step0, b.x, b.y);
            const unsigned int step1[4] = {a0.z, a1.z, a0.w, a1.w};
            mma_bf16(acc[t], step1, b.z, b.w);
        }
    }
    if (KS > 1) {
        __shared__ float partial[KS][MT][32][4];
#pragma unroll
        for (int t = 0; t < MT; t++)
#pragma unroll
            for (int i = 0; i < 4; i++) partial[part][t][lane][i] = acc[t][i];
        __syncthreads();
        if (warp != 0) return;
#pragma unroll
        for (int t = 0; t < MT; t++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float sum = partial[0][t][lane][i];
#pragma unroll
                for (int v = 1; v < KS; v++) sum += partial[v][t][lane][i];
                acc[t][i] = sum;
            }
    }
#pragma unroll
    for (int t = 0; t < MT; t++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            if (!rowValid[t][h]) continue;
            const unsigned int row = rowBase + t * 16u + g + h * 8u;
            const unsigned int pair = (unsigned int)q4::bfr(acc[t][2 * h]) | ((unsigned int)q4::bfr(acc[t][2 * h + 1]) << 16);
            *reinterpret_cast<unsigned int*>(y + (unsigned long long)row * n + column0 + 2u * q) = pair;
        }
}

}  // namespace q4tc

extern "C" __global__ __launch_bounds__(128) void euhedral_q4_linear_tc_bf16(
        const unsigned short* x, const unsigned short* w, unsigned short* y, unsigned int rows, unsigned int k,
        unsigned int n) {
    q4tc::linear<1, 4, 1>(x, w, y, rows, k, n);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_q4_linear_tc_split_bf16(
        const unsigned short* x, const unsigned short* w, unsigned short* y, unsigned int rows, unsigned int k,
        unsigned int n) {
    q4tc::linear<1, 4, 4>(x, w, y, rows, k, n);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_q4_linear_tc_rows_bf16(
        const unsigned short* x, const unsigned short* w, unsigned short* y, unsigned int rows, unsigned int k,
        unsigned int n) {
    q4tc::linear<4, 2, 1>(x, w, y, rows, k, n);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_q4_linear_tc_rows_split_bf16(
        const unsigned short* x, const unsigned short* w, unsigned short* y, unsigned int rows, unsigned int k,
        unsigned int n) {
    q4tc::linear<4, 2, 4>(x, w, y, rows, k, n);
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
