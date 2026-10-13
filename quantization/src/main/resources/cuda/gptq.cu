// GPTQ for the artifact formats (tools/gptq/gptq.py, tools/euhedral_artifacts/grouped.py and nvfp4.py).
//
// Elementwise arithmetic rounds as the Python reference does: division, multiplication and subtraction each round to
// nearest (no fused multiply-add), codes round half to even. The weighted error sums that rank candidate scales add
// in index order (PyTorch's reduction order is not reproduced), so a scale choice can differ from the reference only
// where two candidates' errors tie to within rounding.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kE4m3Codes = 127;
constexpr int kNvfp4Block = 16;
constexpr int kSearchBelow = 6;
constexpr int kSearchAbove = 2;
constexpr int kSd4Above = 40;

__device__ __forceinline__ float e2m1_value(int index) {
    return index < 5 ? index * 0.5f : index == 5 ? 3.0f : index == 6 ? 4.0f : 6.0f;
}

// E4M3 finite magnitudes by code 0..126: (code & 7) / 8 * 2^-6 below 8, else (1 + (code & 7) / 8) * 2^((code >> 3) - 7).
__device__ __forceinline__ float e4m3_value(int code) {
    return code < 8 ? (code & 7) * (1.0f / 8.0f) * 0.015625f : (1.0f + (code & 7) * 0.125f) * exp2f((float)((code >> 3) - 7));
}

// round_to_table over the ascending E4M3 magnitudes: searchsorted (left), clamped to 1..126, nearest with ties to the
// even index, saturating at the top and flushing at zero.
__device__ int round_to_e4m3(float value) {
    int lo = 0, hi = kE4m3Codes;  // first index with table >= value
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (e4m3_value(mid) < value) lo = mid + 1; else hi = mid;
    }
    int upper = min(max(lo, 1), kE4m3Codes - 1), lower = upper - 1;
    float below = __fsub_rn(value, e4m3_value(lower)), above = __fsub_rn(e4m3_value(upper), value);
    int index = (above < below || (above == below && (upper & 1) == 0)) ? upper : lower;
    if (value >= e4m3_value(kE4m3Codes - 1)) index = kE4m3Codes - 1;
    if (value <= 0.0f) index = 0;
    return index;
}

// nearest_scale_codes: the block's largest magnitude / 6 / global scale, rounded to an E4M3 code.
__device__ int nearest_scale(const float* block, float global_scale) {
    float amax = 0.0f;
    for (int i = 0; i < kNvfp4Block; i++) amax = fmaxf(amax, fabsf(block[i]));
    return round_to_e4m3(__fdiv_rn(__fdiv_rn(amax, 6.0f), global_scale));
}

// weighted_block_error_torch for one block under the absolute scale `decode`.
__device__ float block_error(const float* block, const float* weights, float decode) {
    float sum = 0.0f;
    for (int i = 0; i < kNvfp4Block; i++) {
        float magnitude = fabsf(block[i]);
        float q = decode > 0.0f ? __fdiv_rn(magnitude, decode) : 0.0f;
        float e = q < 2.5f ? fminf(__fmul_rn(rintf(__fmul_rn(q, 2.0f)), 0.5f), 2.0f)
                : q < 3.5f ? 3.0f : q < 5.0f ? 4.0f : 6.0f;
        float d = __fsub_rn(magnitude, __fmul_rn(e, decode));
        sum = __fadd_rn(sum, __fmul_rn(__fmul_rn(d, d), weights[i]));
    }
    return sum;
}

// _binary16_scales: signed binary16, never zero for a nonzero group.
__device__ __half binary16(float raw, float max_abs) {
    __half scale = __float2half_rn(fminf(fmaxf(raw, -65504.0f), 65504.0f));
    if (__half2float(scale) == 0.0f && max_abs > 0.0f) scale = __float2half_rn(raw < 0.0f ? -5.9604644775390625e-08f : 5.9604644775390625e-08f);
    return scale;
}

__device__ float grouped_trial(const float* x, const float* w, int group, int qmin, int qmax, float decode, float* numerator,
                               float* denominator) {
    float error = 0.0f, num = 0.0f, den = 0.0f;
    for (int i = 0; i < group; i++) {
        float code = decode != 0.0f ? rintf(__fdiv_rn(x[i], decode)) : 0.0f;
        code = fminf(fmaxf(code, (float)qmin), (float)qmax);
        float d = __fsub_rn(__fmul_rn(code, decode), x[i]);
        error = __fadd_rn(error, __fmul_rn(w[i], __fmul_rn(d, d)));
        num = __fadd_rn(num, __fmul_rn(__fmul_rn(w[i], x[i]), code));
        den = __fadd_rn(den, __fmul_rn(__fmul_rn(w[i], code), code));
    }
    *numerator = num;
    *denominator = den;
    return error;
}

}  // namespace

// search_group_codes_torch for one group per row: values [rows, ld] (the group's `group` columns first), weights
// [group]. Writes the binary16 scale bits and its FP32 decode per row.
extern "C" __global__ void grouped_choose(const float* values, int ld, const float* weights, int rows, int group, int qmin,
                                          int qmax, unsigned short* scale_bits, float* decode_out, int out_stride) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* x = values + (long long)row * ld;
    float max_abs = 0.0f, largest = 0.0f;
    for (int i = 0; i < group; i++) {
        float a = fabsf(x[i]);
        if (a > max_abs) { max_abs = a; largest = x[i]; }
    }
    float num, den;
    __half best = binary16(__fdiv_rn(max_abs, (float)qmax), max_abs);
    float best_error = grouped_trial(x, weights, group, qmin, qmax, __half2float(best), &num, &den);
    float negative_end = largest > 0.0f ? -1.0f : 1.0f;
    int ends[2] = {min(qmax, -qmin), max(qmax, -qmin)};
    int count = qmax == -qmin ? 1 : 2;
    for (int e = 0; e < count; e++) {
        int end = ends[e];
        float sign = end == -qmin ? negative_end : 1.0f;
        for (int step = -9; step <= 9; step++) {
            float denominator_scalar = (float)((double)end + 0.1 * (double)step);
            float start = __fdiv_rn(__fmul_rn(sign, max_abs), denominator_scalar);
            __half first = binary16(start, max_abs);
            grouped_trial(x, weights, group, qmin, qmax, __half2float(first), &num, &den);
            float refit = den > 0.0f ? __fdiv_rn(num, den) : start;
            __half scale = binary16(refit, max_abs);
            float error = grouped_trial(x, weights, group, qmin, qmax, __half2float(scale), &num, &den);
            if (error < best_error) { best = scale; best_error = error; }
        }
    }
    scale_bits[(long long)row * out_stride] = __half_as_ushort(best);
    decode_out[(long long)row * out_stride] = __half2float(best);
}

// The NVFP4 block scale of one 16-column block per row: values [rows, ld], weights [16]. With `table` (16 E4M3
// codes, SD4) the least-error entry at most `below` codes under the nearest scale, the first on ties; without it the
// least-error code from SEARCH_BELOW under to SEARCH_ABOVE over the nearest (search_nvfp4_rows_torch). Writes the
// stored code (table index for SD4) and the absolute decode e4m3 * global scale.
extern "C" __global__ void nvfp4_choose(const float* values, int ld, const float* weights, int rows, float global_scale,
                                        const unsigned char* table, int below, unsigned char* code_out, float* decode_out,
                                        int out_stride) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* x = values + (long long)row * ld;
    int nearest = nearest_scale(x, global_scale);
    int stored, code;
    if (table != nullptr) {
        stored = 0;
        code = table[0];
        float best_error = __int_as_float(0x7f800000);
        for (int entry = 0; entry < 16; entry++) {
            int candidate = table[entry];
            if (candidate < nearest - below) continue;
            float error = block_error(x, weights, __fmul_rn(e4m3_value(candidate), global_scale));
            if (error < best_error) { best_error = error; stored = entry; code = candidate; }
        }
    } else {
        code = nearest;
        float best_error = block_error(x, weights, __fmul_rn(e4m3_value(nearest), global_scale));
        for (int offset = -kSearchBelow; offset <= kSearchAbove; offset++) {
            if (offset == 0) continue;
            int candidate = min(max(nearest + offset, 0), kE4m3Codes - 1);
            float error = block_error(x, weights, __fmul_rn(e4m3_value(candidate), global_scale));
            if (error < best_error) { best_error = error; code = candidate; }
        }
        stored = code;
    }
    code_out[(long long)row * out_stride] = (unsigned char)stored;
    decode_out[(long long)row * out_stride] = __fmul_rn(e4m3_value(code), global_scale);
}

// sd4_costs_torch with weights for the blocks of `values` [rows, k] (FP32): costs[nearest * 127 + code] += the block's
// weighted error under each code from `below` under to SD4_ABOVE over its nearest scale (clamped codes measured, codes
// outside the table cost nothing), energy[nearest] += its weighted energy, counts[nearest] += 1.
extern "C" __global__ void sd4_costs(const float* values, long long blocks, int k, float global_scale, const float* weights,
                                     int below, double* costs, double* energy, unsigned long long* counts) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= blocks) return;
    const float* x = values + index * kNvfp4Block;
    const float* w = weights + (index * kNvfp4Block) % k;
    int nearest = nearest_scale(x, global_scale);
    for (int offset = -below; offset <= kSd4Above; offset++) {
        int code = nearest + offset;
        if (code < 0 || code >= kE4m3Codes) continue;
        float error = block_error(x, w, __fmul_rn(e4m3_value(code), global_scale));
        atomicAdd(costs + nearest * kE4m3Codes + code, (double)error);
    }
    float total = 0.0f;
    for (int i = 0; i < kNvfp4Block; i++) total = __fadd_rn(total, __fmul_rn(__fmul_rn(x[i], x[i]), w[i]));
    atomicAdd(energy + nearest, (double)total);
    atomicAdd(counts + nearest, 1ull);
}

// One run of GPTQ columns (tools/gptq/kernels.py): block [rows, width] holds columns begin.. of the permuted weights;
// columns first..last-1 round with their group's decode (decodes[row * groups + group[j]]), their codes and values go
// to codes / decoded at weight column column_of[j], their error to errors[:, j], and the error times row begin+j of the
// upper factor (element (i, c) at upper[i * upper_row + c * upper_column]) leaves the later columns of the block.
// 32 rows per CTA, four threads per row; format 0 is grouped (qmin..qmax), 1 is NVFP4.
extern "C" __global__ void segment(float* block, float* errors, const float* upper, const float* decodes,
                                   const int* group, const int* column_of, short* codes, float* decoded, int rows,
                                   int width, long long k, long long upper_row, long long upper_column, int begin,
                                   int first, int last, int groups, int format, float qmin, float qmax) {
    extern __shared__ float tile[];  // [32][width + 1]
    const int stride = width + 1;
    const int local = threadIdx.x >> 2, lane = threadIdx.x & 3;
    const int row = blockIdx.x * 32 + local;
    const bool active = row < rows;
    float* mine = tile + local * stride;
    for (int c = lane; c < width; c += 4) mine[c] = active ? block[(long long)row * width + c] : 0.0f;
    __syncwarp();
    for (int j = first; j < last; j++) {
        float x = mine[j];
        float decode = active ? decodes[(long long)row * groups + group[j]] : 0.0f;
        float value;
        int stored;
        if (format == 0) {
            float code = decode != 0.0f ? rintf(__fdiv_rn(x, decode)) : 0.0f;
            code = fminf(fmaxf(code, qmin), qmax);
            value = __fmul_rn(code, decode);
            stored = (int)code;
        } else {
            float magnitude = fabsf(x);
            float scaled = decode > 0.0f ? __fdiv_rn(magnitude, decode) : 0.0f;
            int upper_index = (scaled > 0.0f) + (scaled > 0.5f) + (scaled > 1.0f) + (scaled > 1.5f) + (scaled > 2.0f)
                    + (scaled > 3.0f) + (scaled > 4.0f) + (scaled > 6.0f);
            upper_index = min(max(upper_index, 1), 7);
            int lower_index = upper_index - 1;
            float below = __fsub_rn(scaled, e2m1_value(lower_index));
            float above = __fsub_rn(e2m1_value(upper_index), scaled);
            int index = (above < below || (above == below && (upper_index & 1) == 0)) ? upper_index : lower_index;
            if (scaled >= 6.0f) index = 7;
            if (scaled <= 0.0f) index = 0;
            bool negative = x < 0.0f && index > 0;
            value = __fmul_rn(__fmul_rn(e2m1_value(index), negative ? -1.0f : 1.0f), decode);
            stored = index | (negative ? 8 : 0);
        }
        const float* factor = upper + (long long)(begin + j) * upper_row;
        float error = __fdiv_rn(__fsub_rn(x, value), factor[(long long)(begin + j) * upper_column]);
        if (active && lane == 0) {
            long long at = (long long)row * k + column_of[j];
            codes[at] = (short)stored;
            decoded[at] = value;
            errors[(long long)row * width + j] = error;
        }
        for (int c = lane; c < width; c += 4) {
            if (c >= j) mine[c] = __fsub_rn(mine[c], __fmul_rn(error, factor[(long long)(begin + c) * upper_column]));
        }
        __syncwarp();
    }
    if (active) for (int c = lane; c < width; c += 4) block[(long long)row * width + c] = mine[c];
}

// The group's current values for its scale choice (_choose): position[i] is weight column group*size+i's permuted
// position; those inside [begin, end) come from block, later ones from w.
extern "C" __global__ void gather_group(const float* w, long long k, const float* block, int width, int begin, int end,
                                        const int* position, int size, int rows, float* out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    for (int i = 0; i < size; i++) {
        int at = position[i];
        out[(long long)row * size + i] = at >= begin && at < end ? block[(long long)row * width + (at - begin)]
                                                                 : w[(long long)row * k + at];
    }
}

extern "C" __global__ void bf16_to_float_permuted(const __nv_bfloat16* in, long long rows, long long k, const int* order,
                                                  float* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * k) return;
    long long row = i / k, column = i % k;
    out[i] = __bfloat162float(in[row * k + order[column]]);
}

extern "C" __global__ void float_to_bf16(const float* in, long long count, __nv_bfloat16* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __float2bfloat16_rn(in[i]);
}

// out[i, j] = h[order[i], order[j]] (symmetric, n x n).
extern "C" __global__ void permute_symmetric(const float* h, const int* order, int n, float* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n * n) return;
    long long r = i / n, c = i % n;
    out[i] = h[(long long)order[r] * n + order[c]];
}

extern "C" __global__ void read_diagonal(const float* a, int n, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[(long long)i * n + i];
}

extern "C" __global__ void write_diagonal(float* a, int n, const float* values) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[(long long)i * n + i] = values[i];
}

// Copies the lower triangle of a column-major n x n matrix over its upper triangle (potri leaves only one).
extern "C" __global__ void mirror_lower(float* a, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n * n) return;
    long long r = i % n, c = i / n;  // column-major (r, c)
    if (r < c) a[c * n + r] = a[r * n + c];
}

extern "C" __global__ void amax_bf16(const __nv_bfloat16* x, long long count, unsigned int* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    for (; i < count; i += (long long)gridDim.x * blockDim.x) local = fmaxf(local, fabsf(__bfloat162float(x[i])));
    for (int offset = 16; offset > 0; offset >>= 1) local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, offset));
    if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(local));
}

// dst [rows, width] = src[:, begin:begin + width] of a [rows, k] matrix.
extern "C" __global__ void slice_columns(const float* src, long long k, int begin, int width, long long rows, float* dst) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    long long row = i / width, column = i % width;
    dst[i] = src[row * k + begin + column];
}

// Zeroes the columns listed in `columns` of a [rows, k] matrix.
extern "C" __global__ void zero_columns(float* a, long long k, long long rows, const int* columns, int count) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * count) return;
    a[(i / count) * k + columns[i % count]] = 0.0f;
}

extern "C" __global__ void fill_bytes(unsigned char* a, long long count, unsigned char value) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) a[i] = value;
}
