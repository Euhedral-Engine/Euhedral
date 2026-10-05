#pragma once
#include "qwen4/common.cuh"
#include "attention/nvfp4_kv.cuh"
#include "attention/nvfp4_pipe.cuh"

// QSA attention over the NVFP4 KV pages (docs/FLASH_NEXT_QSA.md).
//
// Query row p attends the tokens of its selected blocks and the incomplete tail block: with nb = (p + 1) / 4 and the
// c selected block ids b_0 < ... < b_{c-1}, the key set is
//   { 4 b_i + u : i < c, u < 4 }  ++  { 4 nb, ..., p }                       (n = 4 c + (p + 1 - 4 nb) keys)
// listed in this order as the "virtual" keys 0 .. n - 1. Rows with every block selected (ids = null) are the dense
// causal prefix, where virtual key t is token t. The softmax is over exactly these keys.
//
// The cache holds the keys and values Hadamard-rotated (nvfp4_kv.cuh), so queries are rotated by the same transform
// and the output is rotated back.

namespace q4qsa {

constexpr int D = 256, KT = 16, STRIDE = D + 8;
constexpr unsigned kRawBytes = KT * 144;                        // one 16-token plane of cache rows
constexpr unsigned kWarpSharedBytes = KT * STRIDE * 2 + 4 * kRawBytes;  // expanded tile + double-buffered raw K and V

static __device__ __forceinline__ void mma16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const void* p) {
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
static __device__ __forceinline__ void ldsm_x4_t(unsigned (&r)[4], const void* p) {
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
static __device__ __forceinline__ unsigned pack_half2(float low, float high) {
    __half2 h = __floats2half2_rn(low, high);
    return *reinterpret_cast<unsigned*>(&h);
}

static __device__ __forceinline__ const unsigned char* cache_row(
        const unsigned char* const* pages, unsigned int token, unsigned int head, unsigned int heads) {
    return pages[token >> 8] + (((token & 255u) * heads + head) * 144u);
}

// The cache token of virtual key t of a row: selected block slot t / 4 (ids == null: the block itself), or the tail.
static __device__ __forceinline__ unsigned int token_of(
        const int* ids, unsigned int count, unsigned int nb, unsigned int t) {
    const unsigned int slot = t >> 2;
    if (slot < count) return (ids != nullptr ? (unsigned int)ids[slot] : slot) * 4u + (t & 3u);
    return 4u * nb + (t - 4u * count);
}

// Starts the copy of the raw rows (144 bytes: nine 16-byte chunks) of the 16 virtual keys at `base` of one head into
// the K plane and the V plane of `raw`.
static __device__ __forceinline__ void issue_tile(unsigned char* raw, const unsigned char* const* key_pages,
        const unsigned char* const* value_pages, const int* ids, unsigned int count, unsigned int nb, unsigned int base,
        unsigned int end, unsigned int head, unsigned int heads, unsigned int lane) {
#pragma unroll
    for (int j = 0; j < 9; j++) {
        const unsigned int chunk = lane + 32u * j;  // 0 .. 287: plane, key, chunk
        const unsigned int plane = chunk >= 144u ? 1u : 0u, within = chunk - plane * 144u;
        const unsigned int key = within / 9u, offset = within - key * 9u;
        if (base + key < end) {
            const unsigned int token = token_of(ids, count, nb, base + key);
            const unsigned char* source = cache_row(plane == 0u ? key_pages : value_pages, token, head, heads);
            nvfp4pipe::cp_async16(raw + plane * kRawBytes + key * 144u + offset * 16u, source + offset * 16u);
        }
    }
}

// Expands the 16 raw rows of one plane into FP16 rows of `tile` (exact products, nvfp4_pipe.cuh); keys at or past
// `valid` become zero rows.
static __device__ __forceinline__ void expand_tile(__half* tile, const unsigned char* raw, const unsigned* pairs,
        unsigned int valid, unsigned int lane) {
#pragma unroll
    for (unsigned int it = 0; it < 8u; it++) {
        const unsigned int key = 2u * it + (lane >> 4), group = lane & 15u;
        unsigned int out[8];
        if (key < valid) {
            const unsigned char* row = raw + key * 144u;
            const uint2 codes = *reinterpret_cast<const uint2*>(row + group * 8u);
            const unsigned scale = nvfp4pipe::scale_pair(row[128u + group]);
            const unsigned words[2] = {codes.x, codes.y};
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int b = 0; b < 4; b++) out[h * 4 + b] = nvfp4pipe::half2_multiply(pairs[(words[h] >> (8 * b)) & 0xFFu], scale);
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) out[i] = 0u;
        }
        uint4* destination = reinterpret_cast<uint4*>(tile + key * STRIDE + group * 16u);
        destination[0] = make_uint4(out[0], out[1], out[2], out[3]);
        destination[1] = make_uint4(out[4], out[5], out[6], out[7]);
    }
}

// Rotates the 256 values of an attention output row back (the transform is its own inverse) and writes the two
// outputs of the layer: core (upstream's attention output, BF16) and gated = bf16(core * bf16(sigmoid(gate))),
// the input of the output projection. `values[r]` holds the unrotated value l + 32 r of lane l.
static __device__ __forceinline__ void write_row(float (&values)[8], unsigned int lane, unsigned int row,
        unsigned int head, unsigned int query_heads, const unsigned short* gate, unsigned int gate_row_stride,
        unsigned int gate_head_stride, unsigned short* core, unsigned short* gated) {
    nvfp4kv::hadamard256(values, lane);
#pragma unroll
    for (int r = 0; r < 8; r++) {
        const unsigned int column = lane + 32u * r;
        const unsigned short rounded = q4::bfr(values[r]);
        const unsigned long long at = (unsigned long long)row * query_heads * D + (unsigned long long)head * D + column;
        if (core != nullptr) core[at] = rounded;
        const float g = q4::round_bf(q4::sigmoid(q4::bf(
                gate[(unsigned long long)row * gate_row_stride + (unsigned long long)head * gate_head_stride + column])));
        gated[at] = q4::bfr(q4::bf(rounded) * g);
    }
}

}  // namespace q4qsa

// Sparse attention of `rows` query rows (chunk rows at positions start + row) over their selected tokens.
// One warp owns one (row, KV head, key split) unit and runs a flash-attention loop over 16-key tiles: the group of
// query heads of the KV head (at most 16) is the m16 dimension, so each tile is expanded once for the whole group.
// Scores and probabilities are FP16 operands on tensor cores with FP32 accumulation, the online softmax is FP32.
//   q          BF16 queries (norm + RoPE applied), head h of row r at q + r * q_row_stride + h * q_head_stride
//   gate       BF16 gate values, the same layout from the gate base pointer
//   ids/counts per-row selected block ids ([row][budget]) and their counts; ids == null: every block of the row
//   splits == 1: the finished rows are written to core (may be null) and gated, [rows][query_heads * 256];
//   splits  > 1: unnormalized partial (256 values, max, sum) per (row, head, split) to `partial`,
//                [row][head][split][258] floats, for euhedral_q4_qsa_merge.
//   grid ceil(rows * key_heads * splits / warps), block 32 * warps; dynamic shared memory kWarpSharedBytes per warp.
extern "C" __global__ __launch_bounds__(128) void euhedral_q4_qsa_attention(
        const unsigned short* __restrict__ q, const unsigned short* __restrict__ gate,
        const unsigned char* const* key_pages, const unsigned char* const* value_pages, const int* __restrict__ ids,
        const int* __restrict__ counts, float* __restrict__ partial, unsigned short* __restrict__ core,
        unsigned short* __restrict__ gated, unsigned int rows, unsigned int query_heads, unsigned int key_heads,
        unsigned int start, unsigned int splits, unsigned int budget, unsigned int q_row_stride,
        unsigned int q_head_stride, unsigned int gate_row_stride, unsigned int gate_head_stride) {
    using namespace q4qsa;
    extern __shared__ __align__(16) unsigned char shared[];
    __shared__ __align__(16) unsigned pairs[256];
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const unsigned int g = lane >> 2, tig = lane & 3u;
    for (unsigned int i = threadIdx.x; i < 256u; i += blockDim.x) pairs[i] = nvfp4pipe::e2m1_pair_bits(i);
    __syncthreads();
    const unsigned long long unit = (unsigned long long)blockIdx.x * warps + warp;
    if (unit >= (unsigned long long)rows * key_heads * splits) return;
    const unsigned int split = (unsigned int)(unit % splits);
    const unsigned int kh = (unsigned int)((unit / splits) % key_heads);
    const unsigned int row = (unsigned int)(unit / ((unsigned long long)splits * key_heads));
    const unsigned int group = query_heads / key_heads;
    const unsigned int position = start + row;
    const unsigned int nb = (position + 1u) >> 2;
    const unsigned int count = ids != nullptr ? (unsigned int)counts[row] : nb;
    const int* row_ids = ids != nullptr ? ids + (unsigned long long)row * budget : nullptr;
    const unsigned int n = 4u * count + (position + 1u - 4u * nb);
    const unsigned int span = (((n + splits - 1u) / splits) + (KT - 1)) & ~(unsigned int)(KT - 1);
    const unsigned int begin = split * span, end = min(n, begin + span);
    const unsigned int tiles = begin < end ? (end - begin + KT - 1) / KT : 0u;

    unsigned char* mine = shared + (unsigned long long)warp * kWarpSharedBytes;
    __half* tile = reinterpret_cast<__half*>(mine);
    unsigned char* raw = mine + KT * STRIDE * 2;  // [buffer][K, V][16][144]

    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float m0 = -__int_as_float(0x7f800000), m1 = m0, l0 = 0.0f, l1 = 0.0f;

    if (tiles > 0) {
        // The first tile's copy overlaps the query rotation below.
        issue_tile(raw, key_pages, value_pages, row_ids, count, nb, begin, end, kh, key_heads, lane);
        nvfp4pipe::cp_async_commit();
    }
    // Rotated queries as FP16 A fragments: rows are the query heads of the group (rows >= group are zero).
    unsigned qa[16][4];
    {
        for (unsigned int h = 0; h < 16u; h++) {
            float values[8];
            if (h < group) {
                const unsigned short* source = q + (unsigned long long)row * q_row_stride
                        + (unsigned long long)(kh * group + h) * q_head_stride;
#pragma unroll
                for (int r = 0; r < 8; r++) values[r] = q4::bf(source[lane + 32 * r]);
                nvfp4kv::hadamard256(values, lane);
            } else {
#pragma unroll
                for (int r = 0; r < 8; r++) values[r] = 0.0f;
            }
#pragma unroll
            for (int r = 0; r < 8; r++) tile[h * STRIDE + lane + 32 * r] = __float2half_rn(values[r]);
        }
        __syncwarp();
#pragma unroll
        for (int k = 0; k < 16; k++) ldsm_x4(qa[k], tile + (lane & 15u) * STRIDE + k * 16 + (lane >> 4) * 8);
        __syncwarp();
    }

    for (unsigned int index = 0; index < tiles; index++) {
        const unsigned int base = begin + index * KT;
        if (index + 1u < tiles)
            issue_tile(raw + ((index + 1u) & 1u) * 2u * kRawBytes, key_pages, value_pages, row_ids, count, nb,
                    base + KT, end, kh, key_heads, lane);
        nvfp4pipe::cp_async_commit();
        nvfp4pipe::cp_async_wait<1>();
        __syncwarp();
        const unsigned char* current = raw + (index & 1u) * 2u * kRawBytes;
        const unsigned int valid = end - base;  // keys of this tile before `end`
        expand_tile(tile, current, pairs, valid, lane);
        __syncwarp();
        // S = Q K^T: the 16 x 16 scores of the group's heads against the tile's keys.
        float s[2][4];
#pragma unroll
        for (int i = 0; i < 2; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
#pragma unroll
        for (int k = 0; k < 16; k++) {
            unsigned b[4];
            ldsm_x4(b, tile + ((lane & 7u) + ((lane >> 4) << 3)) * STRIDE + k * 16 + ((lane >> 3) & 1u) * 8);
            mma16816(s[0], qa[k], b[0], b[1]);
            mma16816(s[1], qa[k], b[2], b[3]);
        }
        // Mask, scale (1/sqrt(256)) and the online softmax of rows g (heads) and g + 8; a quad of lanes shares a row.
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const bool ok = 8u * i + 2u * tig + e < valid;
                s[i][e] = ok ? s[i][e] * 0.0625f : -__int_as_float(0x7f800000);
                s[i][2 + e] = ok ? s[i][2 + e] * 0.0625f : -__int_as_float(0x7f800000);
                mx0 = fmaxf(mx0, s[i][e]);
                mx1 = fmaxf(mx1, s[i][2 + e]);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float c0 = isfinite(m0) ? expf(m0 - mx0) : 0.0f, c1 = isfinite(m1) ? expf(m1 - mx1) : 0.0f;
        float sum0 = 0.0f, sum1 = 0.0f;
        unsigned pa[4];  // P as the A fragment of one k16 step over the tile's 16 keys
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const float p00 = isfinite(s[i][0]) ? expf(s[i][0] - mx0) : 0.0f;
            const float p01 = isfinite(s[i][1]) ? expf(s[i][1] - mx0) : 0.0f;
            const float p10 = isfinite(s[i][2]) ? expf(s[i][2] - mx1) : 0.0f;
            const float p11 = isfinite(s[i][3]) ? expf(s[i][3] - mx1) : 0.0f;
            sum0 += p00 + p01;
            sum1 += p10 + p11;
            pa[2 * i] = pack_half2(p00, p01);
            pa[2 * i + 1] = pack_half2(p10, p11);
        }
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        l0 = l0 * c0 + sum0;
        l1 = l1 * c1 + sum1;
        m0 = mx0;
        m1 = mx1;
#pragma unroll
        for (int j = 0; j < 32; j++) {
            o[j][0] *= c0;
            o[j][1] *= c0;
            o[j][2] *= c1;
            o[j][3] *= c1;
        }
        __syncwarp();  // every lane has read the keys: the tile buffer takes the values
        expand_tile(tile, current + kRawBytes, pairs, valid, lane);
        __syncwarp();
        // O += P V over the 256 dimensions: V^T fragments by transposed ldmatrix.
#pragma unroll
        for (int jp = 0; jp < 16; jp++) {
            unsigned b[4];
            ldsm_x4_t(b, tile + ((lane & 7u) + (((lane >> 3) & 1u) << 3)) * STRIDE + jp * 16 + (lane >> 4) * 8);
            mma16816(o[2 * jp], pa, b[0], b[1]);
            mma16816(o[2 * jp + 1], pa, b[2], b[3]);
        }
        __syncwarp();  // the buffers are free for the next tile's copies and expansion
    }
    nvfp4pipe::cp_async_wait<0>();
    __syncwarp();

    if (splits > 1u) {
#pragma unroll
        for (int half = 0; half < 2; half++) {
            const unsigned int h = g + 8u * half;
            if (h >= group) continue;
            float* destination = partial
                    + (((unsigned long long)row * query_heads + (kh * group + h)) * splits + split) * 258u;
#pragma unroll
            for (int j = 0; j < 32; j++)
                *reinterpret_cast<float2*>(destination + 8 * j + 2 * tig) = make_float2(o[j][2 * half], o[j][2 * half + 1]);
            if (tig == 0) {
                destination[256] = half == 0 ? m0 : m1;
                destination[257] = half == 0 ? l0 : l1;
            }
        }
        return;
    }
    // One split: normalize the rows through shared memory and finish them one head at a time.
    const float inverse0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inverse1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    float* rows_buffer = reinterpret_cast<float*>(mine);  // 16 x 256 floats
#pragma unroll
    for (int j = 0; j < 32; j++) {
        const unsigned int column = 8u * j + 2u * tig;
        *reinterpret_cast<float2*>(rows_buffer + g * D + column) = make_float2(o[j][0] * inverse0, o[j][1] * inverse0);
        *reinterpret_cast<float2*>(rows_buffer + (g + 8u) * D + column) = make_float2(o[j][2] * inverse1, o[j][3] * inverse1);
    }
    __syncwarp();
    for (unsigned int h = 0; h < group; h++) {
        float values[8];
#pragma unroll
        for (int r = 0; r < 8; r++) values[r] = rows_buffer[h * D + lane + 32 * r];
        write_row(values, lane, row, kh * group + h, query_heads, gate, gate_row_stride, gate_head_stride, core, gated);
    }
}

// Combines the key splits of euhedral_q4_qsa_attention and finishes the rows like its one-split path.
//   partial [row][head][split][258] floats; grid ceil(rows * query_heads / 4), block 128 (a warp per row and head).
extern "C" __global__ __launch_bounds__(128) void euhedral_q4_qsa_merge(
        const float* __restrict__ partial, const unsigned short* __restrict__ gate, unsigned short* __restrict__ core,
        unsigned short* __restrict__ gated, unsigned int rows, unsigned int query_heads, unsigned int splits,
        unsigned int gate_row_stride, unsigned int gate_head_stride) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int unit = blockIdx.x * 4u + (threadIdx.x >> 5);
    if (unit >= rows * query_heads) return;
    const unsigned int row = unit / query_heads, head = unit % query_heads;
    const float* records = partial + (unsigned long long)unit * splits * 258u;
    float maximum = -__int_as_float(0x7f800000);
    for (unsigned int s = 0; s < splits; s++) maximum = fmaxf(maximum, records[(unsigned long long)s * 258u + 256u]);
    float total = 0.0f;
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; r++) values[r] = 0.0f;
    for (unsigned int s = 0; s < splits; s++) {
        const float* record = records + (unsigned long long)s * 258u;
        const float m = record[256];
        if (!isfinite(m)) continue;
        const float weight = expf(m - maximum);
        total += weight * record[257];
#pragma unroll
        for (int r = 0; r < 8; r++) values[r] += weight * record[lane + 32 * r];
    }
    const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
#pragma unroll
    for (int r = 0; r < 8; r++) values[r] *= inverse;
    q4qsa::write_row(values, lane, row, head, query_heads, gate, gate_row_stride, gate_head_stride, core, gated);
}

// Appends BF16 K and V rows (NVFP4 rotated rows, the dense engine's codec) to the sequence's cache pages: row r goes to
// position start + r. K and V rows have `key_heads` heads of 256 values at the given row strides.
//   grid ceil(rows * key_heads / 4), block 128.
extern "C" __global__ __launch_bounds__(128) void euhedral_q4_qsa_kv_append(
        const unsigned short* __restrict__ keys, const unsigned short* __restrict__ values,
        unsigned char* const* key_pages, unsigned char* const* value_pages, unsigned int rows,
        unsigned int key_heads, unsigned int key_row_stride, unsigned int value_row_stride, unsigned int start) {
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned int unit = blockIdx.x * 4u + warp;
    if (unit >= rows * key_heads) return;
    const unsigned int row = unit / key_heads, head = unit % key_heads;
    const unsigned int position = start + row;
    const unsigned int offset = ((position & 255u) * key_heads + head) * 144u;
    unsigned char* key = key_pages[position >> 8] + offset;
    unsigned char* value = value_pages[position >> 8] + offset;
    __shared__ float scratch[4][256];
    nvfp4kv::quantize_row(reinterpret_cast<const __nv_bfloat16*>(keys + (unsigned long long)row * key_row_stride + head * 256u),
            key, key + 128, scratch[warp], lane);
    nvfp4kv::quantize_row(reinterpret_cast<const __nv_bfloat16*>(values + (unsigned long long)row * value_row_stride + head * 256u),
            value, value + 128, scratch[warp], lane);
}
