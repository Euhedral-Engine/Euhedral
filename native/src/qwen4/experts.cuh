#pragma once
#include "nvfp4/nvfp4.cuh"
#include "qwen4/common.cuh"

// Routed-expert math of Flash-Next on a wave of resident experts (docs/FLASH_NEXT_EXPERTS.md).
//
// An expert's weights stay NVFP4 (row-split-k128-v1, nvfp4.cuh) inside its cache slot: gate_up [2I x H] at one
// record offset and down [H x I] at another. Both projections run on BF16 tensor cores as the dense decode kernels
// do: a weight (E2M1 code times its E4M3 block scale) is exact in BF16, every product with a BF16 activation is
// exact in FP32, the weight rows are the MMA's 16 M rows and the tokens its 8 N columns. A token's result therefore
// depends only on its own activations and on the tile's fixed K order, never on the tokens beside it: the same
// (token, expert) pair gives the same bits whatever the wave, the work item or the column it sits in.
//
// A work item is one expert and up to 8 of its pairs (a pair is a token row and a routing weight). The host builds
// the items of a wave in ascending expert order; a kernel's grid is (tiles of output rows, items).
namespace q4 {

// {slot index in the wave, first pair, pair count (1..8), unused}.
struct ExpertItem {
    int expert, begin, count, unused;
};

// A pair: the token's row in the activation matrix and its BF16 routing weight (low 16 bits).
struct ExpertPair {
    int token;
    unsigned int weight;
};

static __device__ __forceinline__ void load_pair_table(unsigned int* table) {
    for (unsigned int i = threadIdx.x; i < 256u; i += blockDim.x) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(nvfp4::e2m1_value(i & 15u), nvfp4::e2m1_value(i >> 4));
        table[i] = *reinterpret_cast<const unsigned int*>(&v);
    }
}

// The MMAs of one 16-row weight fragment over one chunk of 128 K values: the lane's 16 code bytes of rows g and g + 8
// (`cg`, `ch`), their block scales (`sg`, `sh`: two E4M3 codes each) and the 32 activations of its token (`xs`).
static __device__ __forceinline__ void mma_chunk(float (&acc)[4], const uint4 cg, const uint4 ch, const unsigned int sg,
        const unsigned int sh, const unsigned int (&xs)[16], const unsigned int* pair_table) {
    const unsigned int wg[4] = {cg.x, cg.y, cg.z, cg.w}, wh[4] = {ch.x, ch.y, ch.z, ch.w};
#pragma unroll
    for (int half = 0; half < 2; half++) {
        const float fg = nvfp4::e4m3_to_float(half ? sg >> 8 : sg & 0xffu);
        const float fh = nvfp4::e4m3_to_float(half ? sh >> 8 : sh & 0xffu);
        const __nv_bfloat162 bg = __floats2bfloat162_rn(fg, fg), bh = __floats2bfloat162_rn(fh, fh);
#pragma unroll
        for (int s = half * 4; s < half * 4 + 4; s++) {
            const unsigned int word_g = wg[s >> 1], word_h = wh[s >> 1];
            const unsigned int shift = (s & 1) * 16u;
            unsigned int pg0 = pair_table[(word_g >> shift) & 0xffu], pg1 = pair_table[(word_g >> (shift + 8u)) & 0xffu];
            unsigned int ph0 = pair_table[(word_h >> shift) & 0xffu], ph1 = pair_table[(word_h >> (shift + 8u)) & 0xffu];
            __nv_bfloat162 vg0 = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&pg0), bg);
            __nv_bfloat162 vg1 = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&pg1), bg);
            __nv_bfloat162 vh0 = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&ph0), bh);
            __nv_bfloat162 vh1 = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&ph1), bh);
            const unsigned int a0 = *reinterpret_cast<unsigned int*>(&vg0), a1 = *reinterpret_cast<unsigned int*>(&vh0);
            const unsigned int a2 = *reinterpret_cast<unsigned int*>(&vg1), a3 = *reinterpret_cast<unsigned int*>(&vh1);
            asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                    "{%0,%1,%2,%3};\n"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(xs[2 * s]), "r"(xs[2 * s + 1]));
        }
    }
}

// F weight fragments (16 rows each, row r of fragment f is `rows[f] + r`) of one tensor against the 8 token columns
// whose activation rows start at `x_row` (null: a column without a token, zeros). The warp handles the chunks
// kw, kw + W, ... of 128 K values in order; lane (g, q) loads 16 code bytes of rows g and g + 8 and the 32
// activations of token g at K offset 32 q of a chunk.
template <int F, int W>
static __device__ __forceinline__ void tile_mma(float (&acc)[F][4], const nvfp4::Layout& w, const unsigned int (&rows)[F],
        const unsigned short* x_row, const unsigned int chunks, const unsigned int kw, const unsigned int* pair_table) {
    const unsigned int q = threadIdx.x & 3u;
#pragma unroll
    for (int f = 0; f < F; f++) acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.0f;
#pragma unroll 2
    for (unsigned int c = kw; c < chunks; c += (unsigned int)W) {
        const unsigned int k0 = c * 128u + 32u * q;
        uint4 xa[4];
#pragma unroll
        for (int i = 0; i < 4; i++)
            xa[i] = x_row != nullptr ? reinterpret_cast<const uint4*>(x_row + k0)[i] : make_uint4(0, 0, 0, 0);
        const unsigned int xs[16] = {xa[0].x, xa[0].y, xa[0].z, xa[0].w, xa[1].x, xa[1].y, xa[1].z, xa[1].w,
                                     xa[2].x, xa[2].y, xa[2].z, xa[2].w, xa[3].x, xa[3].y, xa[3].z, xa[3].w};
        const unsigned int g = (threadIdx.x & 31u) >> 2;
#pragma unroll
        for (int f = 0; f < F; f++) {
            const unsigned int rg = rows[f] + g, rh = rg + 8u;
            const uint4 cg = *reinterpret_cast<const uint4*>(w.codes + (unsigned long long)rg * w.row_bytes + k0 / 2u);
            const uint4 ch = *reinterpret_cast<const uint4*>(w.codes + (unsigned long long)rh * w.row_bytes + k0 / 2u);
            mma_chunk(acc[f], cg, ch, w.scale_bits(rg, k0 / 32u), w.scale_bits(rh, k0 / 32u), xs, pair_table);
        }
    }
}

// Writes a warp's accumulators of F fragments to shared memory as partial[row][token].
template <int F>
static __device__ __forceinline__ void store_partial(float (*partial)[8], const float (&acc)[F][4]) {
    const unsigned int lane = threadIdx.x & 31u, g = lane >> 2, q = lane & 3u;
#pragma unroll
    for (int f = 0; f < F; f++) {
        partial[16 * f + g][2 * q] = acc[f][0];
        partial[16 * f + g][2 * q + 1] = acc[f][1];
        partial[16 * f + g + 8][2 * q] = acc[f][2];
        partial[16 * f + g + 8][2 * q + 1] = acc[f][3];
    }
}

// act[pair][j] = bf16(bf16(silu(gate[j])) * up[j]) for every pair of every item, with gate, up = chunk(gate_up(x), 2),
// each BF16-rounded once after FP32 accumulation, x[token] the BF16 activation row of the pair's token. A CTA
// computes 16 P RW act columns of one item: every warp owns P gate fragments and the P matching up fragments, the W
// warps of a row slab split K (chunk c on warp c mod W, summed in warp order), RW slabs side by side.
//   grid (inter / (16 P RW), items), block 32 W RW; hidden a multiple of 128, inter a multiple of 16 P RW.
template <int P, int W, int RW>
static __device__ __forceinline__ void gate_up_swiglu(const unsigned long long* __restrict__ slots,
        const ExpertItem* __restrict__ items, const ExpertPair* __restrict__ pairs,
        const unsigned short* __restrict__ x, unsigned short* __restrict__ act, unsigned int record_offset,
        unsigned int hidden, unsigned int inter) {
    __shared__ unsigned int pair_table[256];
    __shared__ float partial[RW][W][32 * P][8];
    load_pair_table(pair_table);
    __syncthreads();
    const ExpertItem item = items[blockIdx.y];
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, g = lane >> 2;
    const unsigned int kw = warp % W, rw = warp / W;
    const nvfp4::Layout w(reinterpret_cast<const unsigned char*>(slots[item.expert]) + record_offset, hidden, 2u * inter);
    const unsigned int row0 = blockIdx.x * 16u * P * RW + 16u * P * rw;
    unsigned int rows[2 * P];
#pragma unroll
    for (int p = 0; p < P; p++) {
        rows[p] = row0 + 16u * p;
        rows[P + p] = inter + row0 + 16u * p;
    }
    const unsigned short* x_row =
            (int)g < item.count ? x + (unsigned long long)pairs[item.begin + g].token * hidden : nullptr;
    float acc[2 * P][4];
    tile_mma<2 * P, W>(acc, w, rows, x_row, hidden / 128u, kw, pair_table);
    store_partial<2 * P>(partial[rw][kw], acc);
    __syncthreads();
    for (unsigned int index = kw * 32u + lane; index < 16u * P * 8u; index += 32u * W) {
        const unsigned int t = index & 7u, r = index >> 3;
        if ((int)t >= item.count) continue;
        float gate = partial[rw][0][r][t], up = partial[rw][0][16 * P + r][t];
#pragma unroll
        for (int v = 1; v < W; v++) {
            gate += partial[rw][v][r][t];
            up += partial[rw][v][16 * P + r][t];
        }
        const float gate_bf = round_bf(gate * w.global), up_bf = round_bf(up * w.global);
        act[(unsigned long long)(item.begin + t) * inter + row0 + r] = bfr(round_bf(silu(gate_bf)) * up_bf);
    }
}

// weighted[pair][j] = bf16(bf16(down(act[pair]))[j] * weight[pair]) for every pair of every item: the expert's output
// row scaled by the pair's BF16 routing weight, each op rounded to BF16 as the BF16 tensors upstream are. A CTA computes
// 16 R RW output rows of one item (R fragments per warp, K split over W warps, RW row slabs).
//   grid (hidden / (16 R RW), items), block 32 W RW; inter a multiple of 128, hidden a multiple of 16 R RW.
template <int R, int W, int RW>
static __device__ __forceinline__ void down_weighted(const unsigned long long* __restrict__ slots,
        const ExpertItem* __restrict__ items, const ExpertPair* __restrict__ pairs,
        const unsigned short* __restrict__ act, unsigned short* __restrict__ weighted, unsigned int record_offset,
        unsigned int inter, unsigned int hidden) {
    __shared__ unsigned int pair_table[256];
    __shared__ float partial[RW][W][16 * R][8];
    load_pair_table(pair_table);
    __syncthreads();
    const ExpertItem item = items[blockIdx.y];
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, g = lane >> 2;
    const unsigned int kw = warp % W, rw = warp / W;
    const nvfp4::Layout w(reinterpret_cast<const unsigned char*>(slots[item.expert]) + record_offset, inter, hidden);
    const unsigned int row0 = blockIdx.x * 16u * R * RW + 16u * R * rw;
    unsigned int rows[R];
#pragma unroll
    for (int f = 0; f < R; f++) rows[f] = row0 + 16u * f;
    const unsigned short* x_row = (int)g < item.count ? act + (unsigned long long)(item.begin + g) * inter : nullptr;
    float acc[R][4];
    tile_mma<R, W>(acc, w, rows, x_row, inter / 128u, kw, pair_table);
    store_partial<R>(partial[rw][kw], acc);
    __syncthreads();
    for (unsigned int index = kw * 32u + lane; index < 16u * R * 8u; index += 32u * W) {
        const unsigned int t = index & 7u, r = index >> 3;
        if ((int)t >= item.count) continue;
        float sum = partial[rw][0][r][t];
#pragma unroll
        for (int v = 1; v < W; v++) sum += partial[rw][v][r][t];
        const float y = round_bf(sum * w.global);
        const float scale = bf((unsigned short)pairs[item.begin + t].weight);
        weighted[(unsigned long long)(item.begin + t) * hidden + row0 + r] = bfr(y * scale);
    }
}

}  // namespace q4

// Gate_up and SwiGLU of every item of a wave: arguments are the wave's slot addresses (device addresses of the expert
// records), the items and pairs of the wave, the activations [tokens][hidden], the output act [pairs][inter] and the
// record offset of the expert's gate_up tensor.
//   grid (inter / 32, items), block 128.
extern "C" __global__ __launch_bounds__(128) void euhedral_q4_expert_gate_up_swiglu_bf16(
        const unsigned long long* __restrict__ slots, const q4::ExpertItem* __restrict__ items,
        const q4::ExpertPair* __restrict__ pairs, const unsigned short* __restrict__ x, unsigned short* __restrict__ act,
        unsigned int record_offset, unsigned int hidden, unsigned int inter) {
    q4::gate_up_swiglu<2, 4, 1>(slots, items, pairs, x, act, record_offset, hidden, inter);
}

// The down projection of every item with the routing weight applied: arguments as above, act [pairs][inter] in and
// weighted [pairs][hidden] out, the record offset of the expert's down tensor.
//   grid (hidden / 256, items), block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_expert_down_bf16(
        const unsigned long long* __restrict__ slots, const q4::ExpertItem* __restrict__ items,
        const q4::ExpertPair* __restrict__ pairs, const unsigned short* __restrict__ act,
        unsigned short* __restrict__ weighted, unsigned int record_offset, unsigned int inter, unsigned int hidden) {
    q4::down_weighted<2, 1, 8>(slots, items, pairs, act, weighted, record_offset, inter, hidden);
}

// out[t][c] = bf16(... bf16(bf16(out[t][c] + weighted[p0][c]) + weighted[p1][c]) ...) over the pairs p0, p1, ... of
// token t listed in `token_pair[token_offset[t] .. token_offset[t + 1])`, in that order. `zero_first` starts every
// token from zero (and writes tokens without pairs): the first wave of a layer. A token without pairs is otherwise
// left alone.
//   grid (ceil(hidden / 8 / 256), tokens), block 256; hidden a multiple of 8.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_expert_combine_bf16(
        const unsigned short* __restrict__ weighted, const unsigned int* __restrict__ token_offset,
        const unsigned int* __restrict__ token_pair, unsigned short* __restrict__ out, unsigned int hidden,
        unsigned int zero_first) {
    const unsigned int token = blockIdx.y;
    const unsigned int begin = token_offset[token], end = token_offset[token + 1];
    const unsigned int column = (blockIdx.x * blockDim.x + threadIdx.x) * 8u;
    if (column >= hidden || (begin == end && zero_first == 0u)) return;
    uint4* destination = reinterpret_cast<uint4*>(out + (unsigned long long)token * hidden + column);
    float total[8];
    if (zero_first != 0u) {
#pragma unroll
        for (int i = 0; i < 8; i++) total[i] = 0.0f;
    } else {
        const uint4 packed = *destination;
        const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (int i = 0; i < 8; i++) total[i] = q4::bf((unsigned short)(words[i >> 1] >> (16 * (i & 1))));
    }
    for (unsigned int i = begin; i < end; i++) {
        const uint4 packed = *reinterpret_cast<const uint4*>(
                weighted + (unsigned long long)token_pair[i] * hidden + column);
        const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (int j = 0; j < 8; j++)
            total[j] = q4::round_bf(total[j] + q4::bf((unsigned short)(words[j >> 1] >> (16 * (j & 1)))));
    }
    unsigned int result[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
        result[i] = (unsigned int)q4::bfr(total[2 * i]) | ((unsigned int)q4::bfr(total[2 * i + 1]) << 16);
    *destination = make_uint4(result[0], result[1], result[2], result[3]);
}
