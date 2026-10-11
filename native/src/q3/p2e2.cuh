#pragma once
#include "contiguous.cuh"
#include "common/pdl.cuh"

// P2E2: a lossless, smaller persistent layout of Q3G64_F16S tensors (layout "row-split-p2e2-v1",
// docs/COMPRESSED_Q3.md). The signed 3-bit codes take 2.3 bits of entropy: -1, 0 and +1 cover 80% of
// them. P2E2 stores every code as a 2-bit primary symbol t (0: -1, 1: 0, 2: +1, 3: BIG) and every BIG
// code's value as a 2-bit payload unit p (0: -3, 1: -2, 2: +2, 3: +3) in one stream in row-major
// order. One u32 per row gives the unit index of the row's first BIG code; a warp that walks a row's
// 1024-code slices in order derives every later slice's from its own counts. The FP16 scale plane
// is the row-split one, byte for byte. Rows of K codes (K a multiple of 1024):
//   primary  at 0                          rows * K/16 u32, code j of a row at word j/16, bits 2(j%16)
//   row base at align256(rows * K/4)       rows u32
//   scales   at align256(row base + 4 rows) rows * K/64 FP16
//   payload  at align256(scales + K/32 rows) 16 units per u32 (LSB first), then kPadWords zero words
// Kernels may read up to kPadWords words past the last unit; they never write the tensor.
namespace q3 {
namespace p2e2 {
static constexpr unsigned int kSlice = 1024u;
static constexpr unsigned int kPadWords = 80u;

static __host__ __device__ __forceinline__ unsigned long long align256(unsigned long long v) {
    return (v + 255ull) & ~255ull;
}

struct Layout {
    const unsigned int* primary;
    const unsigned int* row_base;
    const unsigned short* scales;
    const unsigned int* payload;
    unsigned int words, groups;
    __device__ __forceinline__ Layout(const unsigned char* w, unsigned int in_features, unsigned int rows)
            : words(in_features / 16u), groups(in_features / 64u) {
        const unsigned long long base = align256((unsigned long long)rows * in_features / 4u);
        const unsigned long long scale = align256(base + 4ull * rows);
        const unsigned long long payload_offset = align256(scale + (unsigned long long)rows * groups * 2u);
        primary = reinterpret_cast<const unsigned int*>(w);
        row_base = reinterpret_cast<const unsigned int*>(w + base);
        scales = reinterpret_cast<const unsigned short*>(w + scale);
        payload = reinterpret_cast<const unsigned int*>(w + payload_offset);
    }
};

static __device__ __forceinline__ int big_code(unsigned int p) { return p == 0 ? -3 : p == 1 ? -2 : p == 2 ? 2 : 3; }

// BIG flags of 16 primary symbols, one bit at the even position of each symbol.
static __device__ __forceinline__ unsigned int big_bits(unsigned int word) { return word & (word >> 1) & 0x55555555u; }

// Inclusive warp prefix of `value`; adds each lane's predecessors.
static __device__ __forceinline__ unsigned int inclusive_scan(unsigned int value, unsigned int lane) {
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const unsigned int up = __shfl_up_sync(0xffffffffu, value, d);
        if (lane >= (unsigned int)d) value += up;
    }
    return value;
}

static __device__ __forceinline__ unsigned int and3(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int d;
    asm("lop3.b32 %0, %1, %2, %3, 0x80;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
static __device__ __forceinline__ unsigned int mad_u32(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int d;
    asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// ---------------------------------------------------------------------------------------------------
// Single-row decode, bitwise identical to contiguous_decode: the same lane ownership (a lane owns 32
// contiguous codes of a 1024-code slice), the same activations, and the same FP32 FMA chain over
// the same code values. A pair of codes is looked up in a shared table of float pairs indexed by its
// primary nibble and its payload bits placed at the pair's BIG positions (zero elsewhere, so pairs
// without a BIG code, 64% of them, hit 9 broadcast entries in distinct banks).
namespace decode {
static constexpr int kRows = 4;
// The most BIG codes (of a lane's 32) the fast path decodes: 62 bits of units behind the window's two zero bits.
static constexpr unsigned int kFastBig = 31u;
struct Table { float2 pair[256]; };

static __device__ __forceinline__ void fill(Table& table) {
    for (unsigned int i = threadIdx.x; i < 256u; i += blockDim.x) {
        const unsigned int nib = i & 15u, wa = i >> 4, t0 = nib & 3u, t1 = nib >> 2;
        const float c0 = t0 == 3u ? (float)big_code(wa & 3u) : (float)t0 - 1.0f;
        const float c1 = t1 == 3u ? (float)big_code(wa >> 2) : (float)t1 - 1.0f;
        table.pair[i] = make_float2(c0, c1);
    }
}

template<int I>
static __device__ __forceinline__ unsigned int nibble_offset(unsigned int word) {
    constexpr int s = 4 * (I & 7);
    return s >= 3 ? (word >> (s - 3)) & 0x78u : (word << (3 - s)) & 0x78u;
}

// Fast path: (lo, hi) holds this half's payload units behind two zero bits. Shifting right by twice the
// number of BIG codes up to code 2i leaves code 2i's unit (if BIG) at bits 0-1 and code 2i + 1's at
// bits 2-3; the doubled BIG mask of the pair clears the rest. A lane's window is its units shifted left
// by two in 64 bits, so it holds kFastBig units: the first half reads it from bit 0, the second half from
// bit 2 * (BIG codes in the first half), and both fit while the lane has at most kFastBig BIG codes.
template<int I>
static __device__ __forceinline__ float2 pair(const char* table, unsigned int word, unsigned int big2,
        unsigned int lo, unsigned int hi) {
    constexpr int s = 4 * (I & 7);
    const unsigned int shift = __popc(big2 & ((1u << (s + 2)) - 1u));
    const unsigned int wa = and3(__funnelshift_rc(lo, hi, shift), big2 >> s, 15u);
    return *reinterpret_cast<const float2*>(table + mad_u32(wa, 128u, nibble_offset<I>(word)));
}

// Accumulates M activation rows against one decoded lane-row: row t's chain is the one-row chain
// (pairs in order, x[2i] then x[2i + 1]), so each row is bit for bit what a one-row decode computes.
template<int M, int I>
struct Dot {
    static __device__ __forceinline__ void run(const char* table, const float (&x)[M][32], uint2 p, unsigned int bx,
            unsigned int by, unsigned int xl, unsigned int xh, unsigned int yl, unsigned int yh, float (&sum)[M]) {
        const float2 f = I < 8 ? pair<I>(table, p.x, bx, xl, xh) : pair<I>(table, p.y, by, yl, yh);
        #pragma unroll
        for (int t = 0; t < M; t++) {
            sum[t] = fmaf(x[t][2 * I], f.x, sum[t]);
            sum[t] = fmaf(x[t][2 * I + 1], f.y, sum[t]);
        }
        Dot<M, I + 1>::run(table, x, p, bx, by, xl, xh, yl, yh, sum);
    }
};
template<int M>
struct Dot<M, 16> {
    static __device__ __forceinline__ void run(const char*, const float (&)[M][32], uint2, unsigned int, unsigned int,
            unsigned int, unsigned int, unsigned int, unsigned int, float (&)[M]) {}
};

// Any lane-row: units consumed one BIG code at a time from a 64-bit window (up to 32 units).
template<int M, int I>
struct DotAny {
    static __device__ __forceinline__ void run(const Table& table, const float (&x)[M][32], uint2 p,
            unsigned long long q, unsigned int used, float (&sum)[M]) {
        const unsigned int word = I < 8 ? p.x : p.y, nib = (word >> (4 * (I & 7))) & 15u;
        unsigned int wa = 0;
        if ((nib & 3u) == 3u) wa |= (unsigned int)(q >> (2 * used++)) & 3u;
        if ((nib >> 2) == 3u) wa |= ((unsigned int)(q >> (2 * used++)) & 3u) << 2;
        const float2 f = table.pair[nib | wa << 4];
        #pragma unroll
        for (int t = 0; t < M; t++) {
            sum[t] = fmaf(x[t][2 * I], f.x, sum[t]);
            sum[t] = fmaf(x[t][2 * I + 1], f.y, sum[t]);
        }
        DotAny<M, I + 1>::run(table, x, p, q, used, sum);
    }
};
template<int M>
struct DotAny<M, 16> {
    static __device__ __forceinline__ void run(const Table&, const float (&)[M][32], uint2, unsigned long long,
            unsigned int, float (&)[M]) {}
};
}  // namespace decode

// M activation rows (1 to 4) against each weight row. Row t of the output is bit for bit what the contiguous
// one-row kernel gives for it alone: the weights are decoded once and every row runs the same FMA chain.
// Requirements (checked by host dispatch): in_features a multiple of 1024, out_features a multiple of
// 4 * kRows, a 16-byte aligned input and a 16-byte aligned tensor.
template<int M>
static __device__ __forceinline__ void p2e2_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int in_features, unsigned int out_features) {
    using namespace decode;
    __shared__ Table table;
    fill(table);
    __syncthreads();
    const char* table_bytes = reinterpret_cast<const char*>(&table);
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int first_row = (blockIdx.x * (blockDim.x >> 5) + warp) * kRows;
    const Layout w(weights, in_features, out_features);
    const unsigned int slices = in_features / kSlice;
    const uint2* primary = reinterpret_cast<const uint2*>(w.primary + (unsigned long long)first_row * w.words) + lane;
    const unsigned short* scale_rows = w.scales + (unsigned long long)first_row * w.groups + (lane >> 1);
    float sums[M][kRows] = {};
    euhedral_pdl_begin();
    unsigned int base[kRows];
    #pragma unroll
    for (int r = 0; r < kRows; r++) base[r] = w.row_base[first_row + r];
    for (unsigned int slice = 0; slice < slices; slice++) {
        uint2 prim[kRows];
        float sc[kRows];
        unsigned int window[kRows];
        #pragma unroll
        for (int r = 0; r < kRows; r++) {
            prim[r] = primary[(unsigned long long)r * (w.words / 2u) + slice * 32u];
            sc[r] = __half2float(__ushort_as_half(scale_rows[(unsigned long long)r * w.groups + slice * 16u]));
            window[r] = w.payload[(base[r] >> 4) + lane];
        }
        float x[M][32];
        #pragma unroll
        for (int t = 0; t < M; t++) {
            const uint4* activation = reinterpret_cast<const uint4*>(
                    input + (unsigned long long)t * in_features + slice * kSlice + 32u * lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint4 v = activation[i];
                const unsigned int pairs[4] = {v.x, v.y, v.z, v.w};
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    x[t][i * 8 + j * 2] = __uint_as_float(pairs[j] << 16);
                    x[t][i * 8 + j * 2 + 1] = __uint_as_float(pairs[j] & 0xffff0000u);
                }
            }
        }
        unsigned int bx[kRows], by[kRows], cx[kRows], count[kRows], before[kRows], total[kRows];
        #pragma unroll
        for (int r = 0; r < kRows; r++) {
            const unsigned int x0 = big_bits(prim[r].x), y0 = big_bits(prim[r].y);
            bx[r] = x0 * 3u;
            by[r] = y0 * 3u;
            cx[r] = __popc(x0);
            count[r] = cx[r] + __popc(y0);
        }
        // Units before this lane in the slice, two rows per 16-bit half (a slice has at most 1024).
        #pragma unroll
        for (int r = 0; r < kRows; r += 2) {
            const unsigned int packed = count[r] | count[r + 1] << 16;
            const unsigned int inclusive = inclusive_scan(packed, lane);
            const unsigned int all = __shfl_sync(0xffffffffu, inclusive, 31);
            before[r] = (inclusive - packed) & 0xffffu;
            before[r + 1] = (inclusive - packed) >> 16;
            total[r] = all & 0xffffu;
            total[r + 1] = all >> 16;
        }
        unsigned int lo[kRows], hi[kRows];
        bool any = false;
        #pragma unroll
        for (int r = 0; r < kRows; r++) {
            const unsigned int unit = (base[r] & 15u) + before[r];
            const unsigned int word = unit >> 4, shift = (unit & 15u) * 2u;
            const unsigned int a = __shfl_sync(0xffffffffu, window[r], word & 31u);
            const unsigned int b = __shfl_sync(0xffffffffu, window[r], (word + 1u) & 31u);
            const unsigned int c = __shfl_sync(0xffffffffu, window[r], (word + 2u) & 31u);
            const unsigned int l = __funnelshift_r(a, b, shift), h = __funnelshift_r(b, c, shift);
            lo[r] = l << 2;
            hi[r] = __funnelshift_l(l, h, 2);
            any |= count[r] > kFastBig || word > 29u;
        }
        if (!__any_sync(0xffffffffu, any)) {
            #pragma unroll
            for (int r = 0; r < kRows; r++) {
                const unsigned int sy = 2u * cx[r];
                const unsigned int yl = __funnelshift_rc(lo[r], hi[r], sy), yh = sy >= 32u ? 0u : hi[r] >> sy;
                float dot[M] = {};
                Dot<M, 0>::run(table_bytes, x, prim[r], bx[r], by[r], lo[r], hi[r], yl, yh, dot);
                #pragma unroll
                for (int t = 0; t < M; t++) sums[t][r] = fmaf(dot[t], sc[r], sums[t][r]);
            }
        } else {
            // A lane whose 32 codes are all BIG, or a slice whose units outrun the 32 prefetched words.
            #pragma unroll
            for (int r = 0; r < kRows; r++) {
                const unsigned int unit = base[r] + before[r];
                const unsigned int* q = w.payload + (unit >> 4);
                const unsigned int shift = (unit & 15u) * 2u;
                const unsigned long long window64 = (unsigned long long)__funnelshift_r(q[1], q[2], shift) << 32
                        | __funnelshift_r(q[0], q[1], shift);
                float dot[M] = {};
                DotAny<M, 0>::run(table, x, prim[r], window64, 0u, dot);
                #pragma unroll
                for (int t = 0; t < M; t++) sums[t][r] = fmaf(dot[t], sc[r], sums[t][r]);
            }
        }
        #pragma unroll
        for (int r = 0; r < kRows; r++) base[r] += total[r];
    }
    #pragma unroll
    for (int t = 0; t < M; t++) {
        #pragma unroll
        for (int r = 0; r < kRows; r++) {
            #pragma unroll
            for (int distance = 16; distance; distance >>= 1)
                sums[t][r] += __shfl_xor_sync(0xffffffffu, sums[t][r], distance);
        }
        float mine = sums[t][0];
        #pragma unroll
        for (int r = 1; r < kRows; r++) mine = lane == (unsigned int)r ? sums[t][r] : mine;
        if (lane < (unsigned int)kRows)
            write_bf16(output + (unsigned long long)t * out_features, 0, first_row + lane, out_features, mine);
    }
}

// ---------------------------------------------------------------------------------------------------
// Expansion back to the row-split layout (24-byte code groups, then the 256-aligned FP16 scale
// plane), byte for byte. One warp per row walks its slices in order; each lane rebuilds the three
// original words of its 32 codes from 6-bit pair fields looked up by primary nibble and the payload
// bits that start at the pair's first BIG unit.
namespace expand {
struct Table { unsigned char field[256]; };

static __device__ __forceinline__ void fill(Table& table) {
    for (unsigned int i = threadIdx.x; i < 256u; i += blockDim.x) {
        const unsigned int nib = i & 15u, window = i >> 4, t0 = nib & 3u, t1 = nib >> 2;
        unsigned int used = 0;
        int c0 = (int)t0 - 1, c1 = (int)t1 - 1;
        if (t0 == 3u) { c0 = big_code(window & 3u); used = 2; }
        if (t1 == 3u) c1 = big_code((window >> used) & 3u);
        table.field[i] = (unsigned char)(((unsigned int)c0 & 7u) | (((unsigned int)c1 & 7u) << 3));
    }
}

// Original 6-bit field of pair I; `rank` counts the lane's BIG codes before the pair.
template<int I>
static __device__ __forceinline__ unsigned int field(const Table& table, uint2 p, unsigned int bx, unsigned int by,
        unsigned int count_x, unsigned int lo, unsigned int hi) {
    constexpr int s = 4 * (I & 7);
    const unsigned int rank = I < 8 ? __popc(bx & ((1u << s) - 1u)) : count_x + __popc(by & ((1u << s) - 1u));
    const unsigned int b = 2u * rank;
    const unsigned int window = (b < 32u ? __funnelshift_r(lo, hi, b) : hi >> (b - 32u)) & 15u;
    return table.field[(((I < 8 ? p.x : p.y) >> s) & 15u) | window << 4];
}

template<int I>
static __device__ __forceinline__ void place(unsigned int f, unsigned int& w0, unsigned int& w1, unsigned int& w2) {
    constexpr int b = 6 * I;
    if (b + 6 <= 32) {
        w0 |= f << b;
    } else if (b < 32) {
        w0 |= f << b;
        w1 |= f >> (32 - b);
    } else if (b + 6 <= 64) {
        w1 |= f << (b - 32);
    } else if (b < 64) {
        w1 |= f << (b - 32);
        w2 |= f >> (64 - b);
    } else {
        w2 |= f << (b - 64);
    }
}

template<int I>
struct Rebuild {
    static __device__ __forceinline__ void run(const Table& table, uint2 p, unsigned int bx, unsigned int by,
            unsigned int count_x, unsigned int lo, unsigned int hi, unsigned int& w0, unsigned int& w1, unsigned int& w2) {
        place<I>(field<I>(table, p, bx, by, count_x, lo, hi), w0, w1, w2);
        Rebuild<I + 1>::run(table, p, bx, by, count_x, lo, hi, w0, w1, w2);
    }
};
template<>
struct Rebuild<16> {
    static __device__ __forceinline__ void run(const Table&, uint2, unsigned int, unsigned int, unsigned int, unsigned int,
            unsigned int, unsigned int&, unsigned int&, unsigned int&) {}
};
}  // namespace expand

// Writes rows [first_row, first_row + count) of a tensor of `rows` rows as the row-split tensor of
// `count` rows into `out` (codes, then the 256-aligned scale plane).
static __device__ __forceinline__ void p2e2_expand(
        const unsigned char* weights, unsigned char* out, unsigned int rows, unsigned int in_features,
        unsigned int first_row, unsigned int count) {
    using namespace expand;
    __shared__ Table table;
    fill(table);
    __syncthreads();
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int local = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (local >= count) return;
    const unsigned int row = first_row + local;
    const Layout w(weights, in_features, rows);
    const uint2* primary = reinterpret_cast<const uint2*>(w.primary + (unsigned long long)row * w.words) + lane;
    unsigned int* codes = reinterpret_cast<unsigned int*>(out) + (unsigned long long)local * w.groups * 6u + 3u * lane;
    unsigned int base = w.row_base[row];
    for (unsigned int slice = 0; slice < in_features / kSlice; slice++) {
        const uint2 p = primary[slice * 32u];
        const unsigned int bx = big_bits(p.x), by = big_bits(p.y);
        const unsigned int count_x = __popc(bx), count = count_x + __popc(by);
        const unsigned int inclusive = inclusive_scan(count, lane);
        const unsigned int unit = base + inclusive - count;
        const unsigned int* q = w.payload + (unit >> 4);
        const unsigned int shift = (unit & 15u) * 2u;
        const unsigned int lo = __funnelshift_r(q[0], q[1], shift), hi = __funnelshift_r(q[1], q[2], shift);
        unsigned int w0 = 0, w1 = 0, w2 = 0;
        Rebuild<0>::run(table, p, bx, by, count_x, lo, hi, w0, w1, w2);
        codes[slice * 96u] = w0;
        codes[slice * 96u + 1u] = w1;
        codes[slice * 96u + 2u] = w2;
        base += __shfl_sync(0xffffffffu, inclusive, 31);
    }
    const unsigned long long scale_plane = align256((unsigned long long)count * w.groups * 24u);
    const unsigned int* scale_in = reinterpret_cast<const unsigned int*>(w.scales + (unsigned long long)row * w.groups);
    unsigned int* scale_out = reinterpret_cast<unsigned int*>(out + scale_plane) + (unsigned long long)local * (w.groups / 2u);
    for (unsigned int i = lane; i < w.groups / 2u; i += 32u) scale_out[i] = scale_in[i];
}
}  // namespace p2e2
}  // namespace q3
