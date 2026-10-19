// The trellis GEMM on tensor cores: FP16 activations times trellis tiles decoded in registers, FP32 accumulation. The
// whole-K blocks (group_splits, whole_k) are generic in how a block accumulates its K range, so the FP8 prefill
// (trellis/fp8.cuh) shares them.
#ifndef EUHEDRAL_TRELLIS_MMA_CUH
#define EUHEDRAL_TRELLIS_MMA_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "trellis/codebook.cuh"
#include "trellis/decode.cuh"

namespace trellis_mma {

__device__ __forceinline__ void mma_f16(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Ring slots per warp. Over the 3.0 bpw model shapes, 3 or 4 slots keep the most bytes in flight; deeper rings cost
// occupancy.
constexpr int kStages = 4;

__device__ __forceinline__ void copy16(void* shared, const void* global) {
    const unsigned int address = (unsigned int)__cvta_generic_to_shared(shared);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(global));
}

__device__ __forceinline__ void copy_commit() { asm volatile("cp.async.commit_group;\n" ::); }

/// A 16-byte copy of the first `bytes` (0 or 16) of `global`, zero-filling the rest.
__device__ __forceinline__ void copy16_fill(void* shared, const void* global, int bytes) {
    const unsigned int address = (unsigned int)__cvta_generic_to_shared(shared);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(address), "l"(global), "r"(bytes));
}

/// Rows of one CTA of euhedral_trellis_mma_rows.
constexpr int kSplitRows = 32;
/// k16 tiles per activation chunk the CTAs of euhedral_trellis_mma_rows stage in shared memory: 64 FP16 elements, eight
/// 16-byte segments per row.
constexpr int kChunkTiles = 4;
/// Rows of one CTA of euhedral_trellis_mma_rows64.
constexpr int kSplitRows64 = 64;
/// Two chunk slots of `Rows` rows; segment s of row r is stored at s ^ (r % 8), so the eight rows a warp's lanes read
/// at once fall in different banks.
template <int Rows>
using ChunkSlots = uint4[2][Rows][8];

template <int N>
__device__ __forceinline__ void copy_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// The stored tile widths (TrellisLayout.wordsValid): X(w) for each.
#define EUHEDRAL_TRELLIS_WIDTHS(X) \
    X(16) X(24) X(32) X(40) X(48) X(56) X(64) X(72) X(80) X(88) X(96) X(104) X(112) X(120) X(128)

__device__ __forceinline__ unsigned int pack_half2(half2 value) { return *reinterpret_cast<unsigned int*>(&value); }

/// The tile ring of `W` uint16 words as each lane decodes it. A lane's eight positions lane * 8 + j are contiguous, so
/// their windows lie in `kWords` consecutive ring words from the lane's first window, and window j starts kOffset(j)
/// bits after window 0 whatever the lane: lane * 8 is 0 or 8 mod 16 and the half step mask sets only odd bits.
template <int W>
struct Ring {
    static constexpr int kWords32 = W / 2, kBits = W / 16;
    static constexpr bool kHalf = (W & 15) == 8;
    __host__ __device__ static constexpr int kOffset(int j) { return kBits * j + (kHalf ? (j + 1) / 2 : 0); }
    // The last window ends kOffset(7) + 16 bits after the first, which starts up to 31 bits into the first word.
    static constexpr int kWords = (31 + kOffset(7) + 16 + 31) / 32;
};

/// Window j of a lane's normalized words (see Ring): the 16 bits starting kOffset(j) bits into u, MSB first.
template <int W>
__device__ __forceinline__ unsigned int window(const unsigned int* u, int j) {
    const int offset = Ring<W>::kOffset(j);
    const int i = offset >> 5, o = offset & 31;
    if (o <= 16) return (u[i] >> (16 - o)) & 0xFFFFu;
    return __funnelshift_l(u[i + 1], u[i], o) >> 16;
}

/// A row of the activations to address: `row`, or the last row for rows past the launch (their copies read nothing).
__device__ __forceinline__ int chunk_live_row(int row, int rows) { return row < rows ? row : rows - 1; }

/// Where acc[p][i] of a lane lands: row (from the group's first row) and column (in the 128-column block). Narrow
/// tiles take the decoded weights as the m16n8k16 A operand (16 output columns by k16) and 8 activation rows as B, so
/// acc[p] holds columns g and g + 8 of the warp's 16 for rows 8p + 2q and + 1 (g = lane / 4, q = lane % 4). Wide tiles
/// take 16 activation rows as A and each 8-column half h of the weights as B: acc[2m + h] holds rows 16m + g and + 8
/// for columns 8h + 2q and + 1. The products are bitwise the same either way.
template <bool Narrow>
__device__ __forceinline__ int acc_row(int p, int i) {
    const int lane = threadIdx.x & 31;
    return Narrow ? 8 * p + 2 * (lane & 3) + (i & 1) : 16 * (p >> 1) + (lane >> 2) + 8 * (i >> 1);
}

template <bool Narrow>
__device__ __forceinline__ int acc_column(int p, int i) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    return warp * 16 + (Narrow ? (lane >> 2) + 8 * (i >> 1) : 8 * (p & 1) + 2 * (lane & 3) + (i & 1));
}

/// Adds to acc the products of the `8 P` rows from `row_base` over the k16 tiles [first, last) of one 128-column block,
/// for tiles of `W` words, in Narrow or wide tiles (acc_row; wide needs an even P). Warp `warp` owns the 16-column tile
/// blockIdx.x * 8 + warp. The same instruction sequence runs for every row tile, rows past `rows` are zero, and k16
/// tiles are accumulated in increasing order, so a row's bits depend on nothing but its own data and the K range.
template <int W, int P, bool Narrow, bool Staged = false, int SR = kSplitRows, int RegP = 0>
__device__ __forceinline__ void accumulate(float (&acc)[P][4], const unsigned short* __restrict__ act,
        const unsigned short* __restrict__ words, int rows, int row_base, int k, int n, int first, int last,
        unsigned int (*ring)[kStages][kTrellisMaxWords], ChunkSlots<SR>* chunks = nullptr, int splits = 1,
        float (*sum)[128] = nullptr, float (*regs)[4] = nullptr) {
    using R = Ring<W>;
    constexpr int nw = R::kWords32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int grp = lane >> 2, q = lane & 3;
    const int tiles_n = n >> 4;
    const int tj = blockIdx.x * 8 + warp;

    // This lane's ring words and the bit where its first window starts in the first of them, the same for every tile.
    int at = frac_s(lane * 8, W >> 4, R::kHalf ? 0xAAAAu : 0u, W) - 16;
    if (at < 0) at += nw * 32;
    const int shift = at & 31;
    int word[R::kWords];
#pragma unroll
    for (int i = 0; i < R::kWords; ++i) word[i] = ((at >> 5) + i) % nw;

    // Each warp streams its tiles through its own ring of kStages slots with 16-byte asynchronous copies, kStages - 1
    // tiles ahead of the one it decodes. The lanes of the 16-byte chunks of a tile copy them from `fetch`, which steps
    // one k16 tile at a time.
    const size_t stride = (size_t)tiles_n * W;
    const bool copies = lane < W / 8;
    const unsigned short* fetch = words + (size_t)first * stride + (size_t)tj * W + lane * 8;
    // Staged (wide tiles of SR rows): the CTA copies the activation chunk that starts at a tile together with that
    // tile's weights, 16-byte segments of 32 rows per pass of its threads, rows past `rows` and tiles past `last` zero.
    const int chunk_row = threadIdx.x >> 3, chunk_segment = threadIdx.x & 7;
    const auto stage = [&](int tile) {
        if (!Staged || (tile - first) % kChunkTiles != 0 || tile >= last) return;
        const int chunk = (tile - first) / kChunkTiles;
#pragma unroll
        for (int r = chunk_row; r < SR; r += 32) {
            const bool live = row_base + r < rows && tile + (chunk_segment >> 1) < last;
            copy16_fill(&(*chunks)[chunk & 1][r][chunk_segment ^ (r & 7)],
                    act + (size_t)chunk_live_row(row_base + r, rows) * k + (size_t)tile * 16 + chunk_segment * 8,
                    live ? 16 : 0);
        }
    };
#pragma unroll
    for (int i = 0; i < kStages - 1; ++i) {
        if (copies && first + i < last) copy16(&ring[warp][i][lane * 4], fetch);
        stage(first + i);
        copy_commit();
        fetch += stride;
    }
    // The activations are in fragment order (euhedral_trellis_prepass): this lane's four A elements of a row in k16
    // tile `ti` are the 8 bytes at a + 16 * ti.
    const unsigned short* a_lane = act + (size_t)(row_base + grp) * k + 4 * q + (size_t)first * 16;
    int slot = 0;
    // Staged: [first, last) is the whole K range and the sums of each of the `splits` ranges go to `sum` at its end.
    int split = 0, split_end = (int)((long long)(last - first) / splits) + first;
    for (int ti = first; ti < last; ++ti) {
        copy_wait<kStages - 2>();
        __syncwarp();
        if (copies && ti + kStages - 1 < last) copy16(&ring[warp][slot == 0 ? kStages - 1 : slot - 1][lane * 4], fetch);
        stage(ti + kStages - 1);
        copy_commit();
        // A chunk is complete in every thread's groups once the tile it starts at is: one barrier makes it visible.
        // Chunk c + 2 reuses c's slot from tile 4c + 5 on, after every warp passed tile 4c + 4's barrier.
        if (Staged && (ti - first) % kChunkTiles == 0) __syncthreads();
        fetch += stride;
        const unsigned int* mine = ring[warp][slot];
        slot = slot + 1 == kStages ? 0 : slot + 1;
        // The lane's words shifted so its first window starts at bit 0 (MSB first).
        unsigned int u[R::kWords];
#pragma unroll
        for (int i = 0; i < R::kWords; ++i) u[i] = mine[word[i]];
#pragma unroll
        for (int i = 0; i + 1 < R::kWords; ++i) u[i] = __funnelshift_l(u[i + 1], u[i], shift);
        u[R::kWords - 1] <<= shift;
        // Columns g and g + 8 by k 2q, 2q + 1 and 2q + 8, 2q + 9 (positions 0-1, 4-5, 2-3, 6-7): a narrow tile's A
        // fragment, and d[h], d[h + 2] the B fragment of a wide tile's half h.
        unsigned int d[4];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int half_index = 0; half_index < 2; ++half_index) {
                const int j = h * 4 + half_index * 2;
                d[h + 2 * half_index] = pack_half2(decode_mul1_2(window<W>(u, j), window<W>(u, j + 1)));
            }
        if (Narrow) {
#pragma unroll
            for (int p = 0; p < P; ++p) {
                const int row = row_base + p * 8 + grp;
                const uint2 x =
                        row < rows ? *reinterpret_cast<const uint2*>(a_lane + (size_t)(p * 8) * k) : make_uint2(0u, 0u);
                mma_f16(acc[p], d, x.x, x.y);
            }
        } else {
#pragma unroll
            for (int m = 0; m < P / 2; ++m) {
                const int row0 = row_base + m * 16 + grp, row1 = row0 + 8;
                uint2 low, high;
                if (Staged) {
                    const int chunk = (ti - first) / kChunkTiles, tile = (ti - first) % kChunkTiles;
                    const int segment = (2 * tile + (q >> 1)) ^ grp;
                    low = reinterpret_cast<const uint2*>(&(*chunks)[chunk & 1][m * 16 + grp][segment])[q & 1];
                    high = reinterpret_cast<const uint2*>(&(*chunks)[chunk & 1][m * 16 + grp + 8][segment])[q & 1];
                } else {
                    const unsigned short* a0 = a_lane + (size_t)(m * 16) * k;
                    low = row0 < rows ? *reinterpret_cast<const uint2*>(a0) : make_uint2(0u, 0u);
                    high = row1 < rows ? *reinterpret_cast<const uint2*>(a0 + (size_t)8 * k) : make_uint2(0u, 0u);
                }
                const unsigned int a[4] = {low.x, high.x, low.y, high.y};
                mma_f16(acc[2 * m], a, d[0], d[2]);
                mma_f16(acc[2 * m + 1], a, d[1], d[3]);
            }
        }
        a_lane += 16;
        if (Staged && ti + 1 == split_end) {
            // Each lane adds to the sum elements only it touches, in split order, and the next split starts from zero.
            // The first RegP row tiles' sums stay in registers, the rest in `sum` from row 8 RegP.
#pragma unroll
            for (int p = 0; p < P; ++p)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float& at = p < RegP ? regs[p][i] : sum[acc_row<false>(p, i) - 8 * RegP][acc_column<false>(p, i)];
                    at = split == 0 ? acc[p][i] : at + acc[p][i];
                    acc[p][i] = 0.0f;
                }
            ++split;
            split_end = (int)((long long)(split + 1) * (last - first) / splits) + first;
        }
    }
}

template <int P>
__device__ __forceinline__ void clear(float (&acc)[P][4]) {
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[p][i] = 0.0f;
}

/// The raw FP32 sums of split `split` to partials[split][rows][n].
template <int P, bool Narrow>
__device__ __forceinline__ void store_partials(const float (&acc)[P][4], float* __restrict__ partials, int rows,
        int row_base, int n, int split) {
    const int column0 = blockIdx.x * 128;
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
        for (int i = 0; i < 4; i += Narrow ? 1 : 2) {
            const int row = row_base + acc_row<Narrow>(p, i);
            float* at = partials + ((size_t)split * rows + row) * n + column0 + acc_column<Narrow>(p, i);
            if (row >= rows) continue;
            // A wide tile's i and i + 1 are adjacent columns.
            if (Narrow) *at = acc[p][i];
            else *reinterpret_cast<float2*>(at) = make_float2(acc[p][i], acc[p][i + 1]);
        }
}

/// The epilogue in place: out = bf16(svh * H128(sums)), 16 rows (8 for one narrow tile) at a time through `tile`.
template <int P, bool Narrow>
__device__ __forceinline__ void finish(const float (&acc)[P][4], const unsigned short* __restrict__ svh,
        unsigned short* __restrict__ out, int rows, int row_base, int n, float (*tile)[128]) {
    constexpr int kSlice = P == 1 ? 1 : 2;  // row tiles per pass of 16 rows
    const int column0 = blockIdx.x * 128;
    const int half_block = threadIdx.x >> 7, t = threadIdx.x & 127;
    const float scale = __half2float(__ushort_as_half(svh[column0 + t]));
#pragma unroll
    for (int p0 = 0; p0 < P; p0 += kSlice) {
        __syncthreads();
#pragma unroll
        for (int p = p0; p < p0 + kSlice; ++p)
#pragma unroll
            for (int i = 0; i < 4; ++i) tile[acc_row<Narrow>(p, i) - 8 * p0][acc_column<Narrow>(p, i)] = acc[p][i];
        for (int i = 0; i < 4 * kSlice; ++i) {
            const int r = i * 2 + half_block;
            trellis_hadamard128(tile[r], t);
            const int row = row_base + 8 * p0 + r;
            if (row < rows)
                out[(size_t)row * n + column0 + t] = __bfloat16_as_ushort(__float2bfloat16_rn(scale * tile[r][t]));
        }
    }
}

/// One row group of the split kernel, `8 P` rows in narrow or wide tiles: split `blockIdx.y` of the K range, finished
/// here when there is one split.
template <int W, int P, bool Narrow>
__device__ __forceinline__ void group(const unsigned short* __restrict__ act, const unsigned short* __restrict__ words,
        const unsigned short* __restrict__ svh, float* __restrict__ partials, unsigned short* __restrict__ out,
        int rows, int row_base, int k, int n, int splits, int first, int last,
        unsigned int (*ring)[kStages][kTrellisMaxWords], float (*tile)[128]) {
    float acc[P][4];
    clear(acc);
    accumulate<W, P, Narrow>(acc, act, words, rows, row_base, k, n, first, last, ring);
    if (splits > 1) store_partials<P, Narrow>(acc, partials, rows, row_base, n, blockIdx.y);
    else finish<P, Narrow>(acc, svh, out, rows, row_base, n, tile);
}

/// All rows: up to 8 in one narrow tile (one m16n8k16 per k16 tile), more in groups of up to 32 rows of wide tiles.
template <int W>
__device__ __forceinline__ void rows(const unsigned short* __restrict__ act, const unsigned short* __restrict__ words,
        const unsigned short* __restrict__ svh, float* __restrict__ partials, unsigned short* __restrict__ out,
        int rows, int k, int n, int splits, int first, int last, unsigned int (*ring)[kStages][kTrellisMaxWords],
        float (*tile)[128]) {
    if (rows <= 8) {
        group<W, 1, true>(act, words, svh, partials, out, rows, 0, k, n, splits, first, last, ring, tile);
        return;
    }
    for (int row_base = 0; row_base < rows; row_base += 32) {
        if (rows - row_base > 16)
            group<W, 4, false>(act, words, svh, partials, out, rows, row_base, k, n, splits, first, last, ring, tile);
        else
            group<W, 2, false>(act, words, svh, partials, out, rows, row_base, k, n, splits, first, last, ring, tile);
    }
}

/// One row group with every split of the K range in order, in one pass over K: split i accumulates from zero and its
/// sums are added to the running sums in order 0..S-1, as euhedral_trellis_epilogue adds the partials of the split
/// kernel. The sums of the first RegP row tiles stay in registers and the rest go to `sum` (shared, [32][128]); the
/// register sums then pass through the chunk slots, which the K loop no longer needs. Then the epilogue, one row per
/// warp at a time. Wide tiles of `8 P` rows staged in chunks of SR rows.
/// The whole-K accumulation of FP16 activations in fragment order: every split of the K range in order, FP32 sums, the
/// bits of the split kernel. `aux` and `shared_aux` are unused.
struct SplitAccumulation {
    __device__ static const void* segment(const unsigned short* act, int g, int rows, int k, int s) {
        return act + (size_t)g * rows * k;
    }
    __device__ static const void* aux(const unsigned short* act, int g, int rows, int k, int s) { return nullptr; }
    template <int W, int P, int SR, int RegP>
    __device__ static void run(float (&acc)[P][4], const void* act, const unsigned short* __restrict__ words, int rows,
            int row_base, int k, int n, int splits, unsigned int (*ring)[kStages][kTrellisMaxWords],
            ChunkSlots<SR>* chunks, float (*sum)[128], float (*regs)[4], const void* aux, void* shared_aux) {
        accumulate<W, P, false, true, SR, RegP>(acc, static_cast<const unsigned short*>(act), words, rows, row_base, k,
                n, 0, k >> 4, ring, chunks, splits, sum, regs);
    }
};

template <int W, int P, int SR, int RegP, class A>
__device__ __forceinline__ void group_splits(const void* act, const unsigned short* __restrict__ words,
        const unsigned short* __restrict__ svh, unsigned short* __restrict__ out, int rows, int row_base, int k, int n,
        int splits, unsigned int (*ring)[kStages][kTrellisMaxWords], float (*sum)[128], ChunkSlots<SR>* chunks,
        const void* aux, void* shared_aux) {
    float acc[P][4], regs[RegP > 0 ? RegP : 1][4];
    clear(acc);
    A::template run<W, P, SR, RegP>(
            acc, act, words, rows, row_base, k, n, splits, ring, chunks, sum, regs, aux, shared_aux);
    __syncthreads();
    float (*low)[128] = reinterpret_cast<float (*)[128]>(chunks);
    if (RegP > 0) {
#pragma unroll
        for (int p = 0; p < RegP; ++p)
#pragma unroll
            for (int i = 0; i < 4; ++i) low[acc_row<false>(p, i)][acc_column<false>(p, i)] = regs[p][i];
        __syncthreads();
    }
    // The epilogue, one row per warp at a time: the Hadamard in registers, then svh and BF16, 8 bytes per lane.
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int column = blockIdx.x * 128 + lane * 4;
    const uint2 scales = *reinterpret_cast<const uint2*>(svh + column);
    const float scale[4] = {__half2float(__ushort_as_half((unsigned short)scales.x)),
            __half2float(__ushort_as_half((unsigned short)(scales.x >> 16))),
            __half2float(__ushort_as_half((unsigned short)scales.y)),
            __half2float(__ushort_as_half((unsigned short)(scales.y >> 16)))};
    for (int r = warp; r < P * 8; r += 8) {
        const int row = row_base + r;
        if (row >= rows) break;
        const float* from = r < 8 * RegP ? low[r] : sum[r - 8 * RegP];
        const float4 loaded = *reinterpret_cast<const float4*>(from + lane * 4);
        float v[4] = {loaded.x, loaded.y, loaded.z, loaded.w};
        trellis_hadamard128_warp(v, lane);
        unsigned short bits[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) bits[j] = __bfloat16_as_ushort(__float2bfloat16_rn(scale[j] * v[j]));
        *reinterpret_cast<uint2*>(out + (size_t)row * n + column) =
                make_uint2(bits[0] | (unsigned int)bits[1] << 16, bits[2] | (unsigned int)bits[3] << 16);
    }
}

#define EUHEDRAL_TRELLIS_GROUP_ARGS act, words, svh, out, rows, row_base, k, n, splits, ring, sum, chunks, aux, shared_aux

template <int W, class A>
__device__ __forceinline__ void row_group(const void* act, const unsigned short* __restrict__ words,
        const unsigned short* __restrict__ svh, unsigned short* __restrict__ out, int rows, int row_base, int k, int n,
        int splits, unsigned int (*ring)[kStages][kTrellisMaxWords], float (*sum)[128], ChunkSlots<kSplitRows>* chunks,
        const void* aux, void* shared_aux) {
    // Wide tiles: 16 or 32 rows.
    if (rows - row_base > 16) group_splits<W, 4, kSplitRows, 0, A>(EUHEDRAL_TRELLIS_GROUP_ARGS);
    else group_splits<W, 2, kSplitRows, 0, A>(EUHEDRAL_TRELLIS_GROUP_ARGS);
}

/// 64-row blocks: 64 rows keep half their split sums in registers; a last block of at most 32 rows keeps them all in
/// `sum`. (An overload of row_group by the chunk slots' rows.)
template <int W, class A>
__device__ __forceinline__ void row_group(const void* act, const unsigned short* __restrict__ words,
        const unsigned short* __restrict__ svh, unsigned short* __restrict__ out, int rows, int row_base, int k, int n,
        int splits, unsigned int (*ring)[kStages][kTrellisMaxWords], float (*sum)[128],
        ChunkSlots<kSplitRows64>* chunks, const void* aux, void* shared_aux) {
    if (rows - row_base > 32) group_splits<W, 8, kSplitRows64, 4, A>(EUHEDRAL_TRELLIS_GROUP_ARGS);
    else if (rows - row_base > 16) group_splits<W, 4, kSplitRows64, 0, A>(EUHEDRAL_TRELLIS_GROUP_ARGS);
    else group_splits<W, 2, kSplitRows64, 0, A>(EUHEDRAL_TRELLIS_GROUP_ARGS);
}

#undef EUHEDRAL_TRELLIS_GROUP_ARGS

/// The whole-K kernels' body: blocks of SR rows (kSplitRows or kSplitRows64) from blockIdx.y * SR, accumulating by
/// A, whose `shared_aux` the kernel declares.
template <int SR, class A>
__device__ __forceinline__ void whole_k(const unsigned short* __restrict__ act, const unsigned short* __restrict__ words,
        const int* __restrict__ ends, const unsigned short* __restrict__ svh, unsigned short* __restrict__ out,
        int rows, int k, int n, int w, int splits, int s, unsigned int (*ring)[kStages][kTrellisMaxWords],
        float (*sum)[128], ChunkSlots<SR>* chunks, void* shared_aux) {
    const int column0 = blockIdx.x * 128;
    int g = 0;
    while (g + 1 < s && column0 >= ends[g]) ++g;
    const void* segment = A::segment(act, g, rows, k, s);
    const void* aux = A::aux(act, g, rows, k, s);
    const int row_base = blockIdx.y * SR;
    switch (w) {
#define EUHEDRAL_TRELLIS_MMA_WIDTH(W) \
    case W: \
        row_group<W, A>(segment, words, svh, out, rows, row_base, k, n, splits, ring, sum, chunks, aux, shared_aux); \
        break;
        EUHEDRAL_TRELLIS_WIDTHS(EUHEDRAL_TRELLIS_MMA_WIDTH)
#undef EUHEDRAL_TRELLIS_MMA_WIDTH
    }
}

}  // namespace trellis_mma

// act f16 [s][rows][k]; words the payload [k/16][n/16][w]; ends the exclusive segment row ends; svh f16 [n]. With
// S == 1 the epilogue is applied here and `out` is written; with S > 1 the raw FP32 sums of split `blockIdx.y` go to
// partials[split][rows][n] (euhedral_trellis_epilogue finishes them). Split i covers the k16 tiles
// [i * T / S, (i + 1) * T / S), T = k / 16. Each CTA covers 128 columns of one segment: warp i owns the tile columns
// blockIdx.x * 128 + 16 i.. and decodes its tiles into its two m16n8k16 B fragments.
//   grid (n / 128, S), block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_trellis_mma(
        const unsigned short* __restrict__ act, const unsigned short* __restrict__ words, const int* __restrict__ ends,
        const unsigned short* __restrict__ svh, float* __restrict__ partials, unsigned short* __restrict__ out,
        int rows, int k, int n, int w, int splits, int s) {
    __shared__ __align__(16) unsigned int ring[8][trellis_mma::kStages][kTrellisMaxWords];
    __shared__ float tile[16][128];
    const int column0 = blockIdx.x * 128;
    int g = 0;
    while (g + 1 < s && column0 >= ends[g]) ++g;
    const unsigned short* segment = act + (size_t)g * rows * k;
    const int tiles = k >> 4;
    const int first = (int)(((long long)blockIdx.y * tiles) / splits);
    const int last = (int)(((long long)(blockIdx.y + 1) * tiles) / splits);
    switch (w) {
#define EUHEDRAL_TRELLIS_MMA_WIDTH(W) \
    case W: \
        trellis_mma::rows<W>(segment, words, svh, partials, out, rows, k, n, splits, first, last, ring, tile); \
        break;
        EUHEDRAL_TRELLIS_WIDTHS(EUHEDRAL_TRELLIS_MMA_WIDTH)
#undef EUHEDRAL_TRELLIS_MMA_WIDTH
    }
}

// The split kernel's result without partials, for launches of many rows: each CTA covers 128 columns of one segment
// and the kSplitRows rows from blockIdx.y * kSplitRows, and runs the S splits of the K range itself, adding their sums
// in the epilogue's order, so every output is bitwise that of euhedral_trellis_mma with S splits and
// euhedral_trellis_epilogue.
//   grid (n / 128, ceil(rows / kSplitRows)), block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_trellis_mma_rows(
        const unsigned short* __restrict__ act, const unsigned short* __restrict__ words, const int* __restrict__ ends,
        const unsigned short* __restrict__ svh, unsigned short* __restrict__ out, int rows, int k, int n, int w,
        int splits, int s) {
    __shared__ __align__(16) unsigned int ring[8][trellis_mma::kStages][kTrellisMaxWords];
    __shared__ float sum[trellis_mma::kSplitRows][128];
    __shared__ trellis_mma::ChunkSlots<trellis_mma::kSplitRows> chunks;
    trellis_mma::whole_k<trellis_mma::kSplitRows, trellis_mma::SplitAccumulation>(
            act, words, ends, svh, out, rows, k, n, w, splits, s, ring, sum, &chunks, nullptr);
}

// euhedral_trellis_mma_rows with blocks of kSplitRows64 rows, two per SM: each decoded tile serves twice the rows.
//   grid (n / 128, ceil(rows / kSplitRows64)), block 256.
extern "C" __global__ __launch_bounds__(256, 2) void euhedral_trellis_mma_rows64(
        const unsigned short* __restrict__ act, const unsigned short* __restrict__ words, const int* __restrict__ ends,
        const unsigned short* __restrict__ svh, unsigned short* __restrict__ out, int rows, int k, int n, int w,
        int splits, int s) {
    __shared__ __align__(16) unsigned int ring[8][trellis_mma::kStages][kTrellisMaxWords];
    __shared__ float sum[32][128];
    __shared__ trellis_mma::ChunkSlots<trellis_mma::kSplitRows64> chunks;
    trellis_mma::whole_k<trellis_mma::kSplitRows64, trellis_mma::SplitAccumulation>(
            act, words, ends, svh, out, rows, k, n, w, splits, s, ring, sum, &chunks, nullptr);
}

#endif
