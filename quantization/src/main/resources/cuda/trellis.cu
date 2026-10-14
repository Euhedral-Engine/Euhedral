// Trellis (EXL3) quantization kernels. Portions ported from exllamav3 1.6.1 (MIT License, Copyright (c) 2025
// Turboderp): codebook.cuh (mul1 decode), quantize_tiles_optimized.cuh (the dense Viterbi that runs on sm_120 for
// K = 2..6), quantize_tiles_frac_kernel.cuh (alternating K and K + 1 bit steps), pack.cu / frac.cu (MSB-first
// packing) and exl3_dq.cuh (the circular bit window of the decoder).
//
// A 256-weight tile is a tail-biting ring of 16 * K bits. Weight i is quantized to a 16-bit state whose low K bits are
// new; the codebook value of a state is the mul1 product, summed bytewise, scaled in fp16. Costs are fp16; ties go to
// the lowest predecessor, so the result is exactly the reference's.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

#define H_INF __ushort_as_half(0x7c00)

namespace {

constexpr int kTile = 256;

/// Two mul1 codebook entries from the precomputed products x0 = state0 * 0x83DCD12D, x1 = state1 * 0x83DCD12D.
__device__ inline half2 decode_mul1_product_2(uint32_t x0, uint32_t x1) {
    const uint32_t acc = 0x6400u;
    uint32_t sum0 = __dp4a(x0, 0x01010101u, acc);
    uint32_t sum1 = __dp4a(x1, 0x01010101u, acc);
    half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
    half h0 = __ushort_as_half((uint16_t)sum0);
    half h1 = __ushort_as_half((uint16_t)sum1);
    return __hfma2(__halves2half2(h0, h1), k_inv_h2, k_bias_h2);
}

__device__ inline half2 decode_mul1_2(uint32_t x0, uint32_t x1) {
    return decode_mul1_product_2(x0 * 0x83DCD12Du, x1 * 0x83DCD12Du);
}

__device__ inline half decode_mul1(uint32_t x) {
    x *= 0x83DCD12Du;
    uint32_t sum = __dp4a(x, 0x01010101u, 0x6400u);
    return __hfma(__ushort_as_half((uint16_t)sum), __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
}

__device__ __forceinline__ uint32_t warp_min(uint32_t value) {
    return __reduce_min_sync(0xffffffff, value);
}

__host__ __device__ constexpr int viterbi_threads(int K) { return K == 5 || K == 6 ? 512 : 1024; }

/// Dense tail-biting Viterbi of one tile for K = 2..6 (one block per tile: the grid must equal the tile count). Costs live in shared memory, the history
/// is one byte per (position, state) in `temp_edges`, [256, 65536 >> K] per tile.
template <int K>
__device__ __forceinline__ void viterbi_body(
        const float* __restrict__ input_tiles_ptr, float* __restrict__ output_tiles_ptr,
        uint16_t* __restrict__ output_indices_ptr, uint16_t* __restrict__ temp_edges_ptr) {
    constexpr int L = kTile;
    constexpr int NT = viterbi_threads(K);
    constexpr int NW = NT / 32;
    constexpr int Kr = 16 - K;
    constexpr int max_q = 1 << K;
    constexpr int edges = 65536 >> K;
    constexpr int history_bytes = L * edges;
    const int tile_idx = blockIdx.x;
    const int thread = threadIdx.x;
    const float* input_tile = input_tiles_ptr + L * tile_idx;
    float* output_tile = output_tiles_ptr + L * tile_idx;
    uint16_t* output_indices = output_indices_ptr + L * tile_idx;
    uint8_t* temp_edges = reinterpret_cast<uint8_t*>(temp_edges_ptr) + (int64_t)history_bytes * tile_idx;

    extern __shared__ uint8_t shbuf[];
    half* sh_input_tile = reinterpret_cast<half*>(shbuf);
    int* sh_idx = reinterpret_cast<int*>(shbuf + L * sizeof(half) + 64);
    half* temp_costs = reinterpret_cast<half*>(shbuf + L * sizeof(half) + 64 + 128);
    half* temp_costs_inc = temp_costs + edges;
    for (int i = thread; i < L; i += NT) sh_input_tile[i] = __float2half_rn(input_tile[i]);
    __syncthreads();
    auto ring = [&](int i, int roll) {
        int ri = i + roll;
        if (ri >= L) ri -= L;
        return ri;
    };

    // Best final cost; ties resolve by the bit-reversed rank of the state, as in the reference.
    auto argmin_cost = [&]() {
        uint32_t best = 0x7c00ffffu;
        for (int e = thread; e < edges; e += NT) {
            unsigned v = e & 1023;
            unsigned rank = ((__brev(v >> 5) >> 27) << 10) | ((__brev(v & 31) >> 27) << 5) | (e >> 10);
            unsigned key = ((uint32_t)__half_as_ushort(temp_costs[e]) << 16) | rank;
            best = min(best, key);
        }
        best = warp_min(best);
        if ((thread & 31) == 0) ((uint32_t*)sh_idx)[thread >> 5] = best;
        __syncthreads();
        if (thread < 32) {
            best = thread < NW ? ((uint32_t*)sh_idx)[thread] : 0x7c00ffffu;
            best = warp_min(best);
        }
        unsigned rank = best & 65535;
        unsigned v = ((__brev(rank >> 10) >> 27) << 5) | (__brev((rank >> 5) & 31) >> 27);
        return best >= 0x7c000000u ? 0 : (int)(((rank & 31) << 10) | v);
    };

    auto backward = [&](int roll, bool write, int edge) {
        if (thread == 0) {
            for (int i = L - 1; i >= 0; --i) {
                const int ri = ring(i, roll);
                const int branch = temp_edges[edges * ri + edge];
                const int prev_edge = ((branch & (max_q - 1)) << (16 - 2 * K)) | (edge >> K);
                const int encoded = (prev_edge << K) | edge;
                edge = prev_edge;
                if (write) {
                    output_indices[ri] = (uint16_t)encoded;
                    output_tile[ri] = __half2float(decode_mul1(encoded));
                } else if (ri == 0)
                    break;
            }
        }
        if (thread == 0) sh_idx[0] = edge;
        __syncthreads();
        return sh_idx[0];
    };

    // Two passes: the first from an open start, whose best end state then pins the second.
#define SOLVE()                                           \
    forward(L / 2, -1);                                   \
    int end_state = backward(L / 2, false, argmin_cost()); \
    forward(0, end_state);                                \
    backward(0, true, end_state)

    if constexpr (K <= 5) {
        constexpr int NP = edges / (2 * NT);
        constexpr int NC = max_q;
        // Decoded values persist in registers across both passes.
        half2 values[NP][NC];
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            int out = 2 * (thread + j * NT);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                int state = (c << Kr) | out;
                values[j][c] = decode_mul1_2(state, state + 1);
            }
        }
        auto forward = [&](int roll, int pre_state) {
            for (int i = 0; i < L; ++i) {
                int ri = ring(i, roll);
                half* t = temp_costs;
                temp_costs = temp_costs_inc;
                temp_costs_inc = t;
                half2 w = __half2half2(sh_input_tile[ri]);
#pragma unroll
                for (int j = 0; j < NP; ++j) {
                    int out = 2 * (thread + j * NT);
                    half2 best;
                    int b0 = 0x7fffffff, b1 = 0x7fffffff;
                    uint32_t best0 = 0x7c00ffffu, best1 = 0x7c00ffffu;
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        int state = (c << Kr) | out;
                        int pred = state >> K;
                        half2 d = __hsub2(values[j][c], w);
                        half2 err;
                        if (i == 0) {
                            err = __hmul2(d, d);
                            if (pre_state >= 0 && pred != pre_state) err = __half2half2(H_INF);
                        } else
                            err = __hfma2(d, d, __half2half2(temp_costs_inc[pred]));
                        if constexpr (K <= 3) {
                            // Strict less-than in candidate order: the lowest predecessor wins a tie.
                            if (c == 0) best = __half2half2(H_INF);
                            if (c == 0 || __hlt(__low2half(err), __low2half(best))) {
                                best = __halves2half2(__low2half(err), __high2half(best));
                                b0 = pred;
                            }
                            if (c == 0 || __hlt(__high2half(err), __high2half(best))) {
                                best = __halves2half2(__low2half(best), __high2half(err));
                                b1 = pred;
                            }
                        } else {
                            // Packed (cost, predecessor) minimum, K = 4 and 5.
                            best0 = min(best0, ((uint32_t)__half_as_ushort(__low2half(err)) << 16) | pred);
                            best1 = min(best1, ((uint32_t)__half_as_ushort(__high2half(err)) << 16) | pred);
                        }
                    }
                    if constexpr (K > 3) {
                        best = __halves2half2(__ushort_as_half(best0 >> 16), __ushort_as_half(best1 >> 16));
                        b0 = best0 & 65535;
                        b1 = best1 & 65535;
                    }
                    reinterpret_cast<half2*>(temp_costs)[out / 2] = best;
                    if (pre_state >= 0 || ri < L / 2) {
                        ((uint16_t*)temp_edges)[(edges * ri + out) / 2] =
                                ((b0 >> (16 - 2 * K)) & 255) | (((b1 >> (16 - 2 * K)) & 255) << 8);
                    }
                }
                __syncthreads();
            }
        };
        SOLVE();
    } else {
        // K = 6: no register cache; the codebook values are recomputed (the reference reads an equal fp16 table).
        auto step = [&](int i, int ri, int pre_state, bool write) {
            for (int out_edge_idx = 2 * thread; out_edge_idx < edges; out_edge_idx += 2 * NT) {
                const half2 w2 = __half2half2(sh_input_tile[ri]);
                int in_edge_idx = out_edge_idx >> K;
                half2 decoded2 = decode_mul1_2(out_edge_idx, out_edge_idx + 1);
                half2 dh2 = __hsub2(decoded2, w2);
                half2 min_err2;
                if (i == 0) {
                    min_err2 = __hmul2(dh2, dh2);
                    if (pre_state >= 0 && in_edge_idx != pre_state) min_err2 = __half2half2(H_INF);
                } else
                    min_err2 = __hfma2(dh2, dh2, __half2half2(temp_costs_inc[in_edge_idx]));
                uint32_t best0 = ((uint32_t)__half_as_ushort(__low2half(min_err2)) << 16) | in_edge_idx;
                uint32_t best1 = ((uint32_t)__half_as_ushort(__high2half(min_err2)) << 16) | in_edge_idx;
#pragma unroll
                for (int k = 1; k < max_q; ++k) {
                    const int state0 = (k << Kr) | out_edge_idx;
                    in_edge_idx = state0 >> K;
                    decoded2 = decode_mul1_2(state0, state0 + 1);
                    dh2 = __hsub2(decoded2, w2);
                    half2 err2;
                    if (i == 0) {
                        err2 = __hmul2(dh2, dh2);
                        if (pre_state >= 0 && in_edge_idx != pre_state) err2 = __half2half2(H_INF);
                    } else
                        err2 = __hfma2(dh2, dh2, __half2half2(temp_costs_inc[in_edge_idx]));
                    best0 = min(best0, ((uint32_t)__half_as_ushort(__low2half(err2)) << 16) | in_edge_idx);
                    best1 = min(best1, ((uint32_t)__half_as_ushort(__high2half(err2)) << 16) | in_edge_idx);
                }
                min_err2 = __halves2half2(__ushort_as_half(best0 >> 16), __ushort_as_half(best1 >> 16));
                int min_in_edge0 = best0 & 65535;
                int min_in_edge1 = best1 & 65535;
                reinterpret_cast<half2*>(temp_costs)[out_edge_idx >> 1] = min_err2;
                if (write) {
                    ((uint16_t*)temp_edges)[(edges * ri + out_edge_idx) / 2] =
                            ((min_in_edge0 >> (16 - 2 * K)) & 255) | (((min_in_edge1 >> (16 - 2 * K)) & 255) << 8);
                }
            }
            __syncthreads();
        };
        auto forward = [&](int roll, int pre_state) {
            for (int i = 0; i < L; ++i) {
                int ri = ring(i, roll);
                half* t = temp_costs;
                temp_costs = temp_costs_inc;
                temp_costs_inc = t;
                step(i, ri, pre_state, pre_state >= 0 || ri < L / 2);
            }
        };
        SOLVE();
    }
}

constexpr int kFracThreads = 512;

/// One step of the fractional Viterbi: in-nodes are the top 16 - KIN bits of the window, out-nodes the top
/// 16 - KOUT bits of the next. History is two bytes per (position, node).
template <int KIN, int KOUT>
__device__ __forceinline__ void frac_step(
        const half w, const half* __restrict__ costs_in, half* __restrict__ costs_out,
        uint16_t* __restrict__ hist_row, const bool first, const int pre_state, const bool write_hist,
        const int thread) {
    constexpr int Kr = 16 - KOUT;
    constexpr int max_q = 1 << KOUT;
    constexpr int edges_out = 65536 >> KOUT;
    const half2 w2 = __half2half2(w);
    const half2 inf2 = __half2half2(H_INF);
    for (int out_edge_idx = 2 * thread; out_edge_idx < edges_out; out_edge_idx += 2 * kFracThreads) {
        int in_edge_idx = out_edge_idx >> KIN;
        uint32_t product0 = out_edge_idx * 0x83DCD12Du;
        uint32_t product1 = product0 + 0x83DCD12Du;
        half2 decoded2 = decode_mul1_product_2(product0, product1);
        half2 dh2 = __hsub2(decoded2, w2);
        half2 min_err2;
        if (first) {
            min_err2 = __hmul2(dh2, dh2);
            if (pre_state >= 0 && in_edge_idx != pre_state) min_err2 = inf2;
        } else
            min_err2 = __hfma2(dh2, dh2, __half2half2(costs_in[in_edge_idx]));
        int min_in_edge0 = in_edge_idx;
        int min_in_edge1 = in_edge_idx;
#pragma unroll
        for (int k = 1; k < max_q; ++k) {
            const int state0 = (k << Kr) | out_edge_idx;
            in_edge_idx = state0 >> KIN;
            constexpr uint32_t product_step = 0x83DCD12Du << Kr;
            product0 += product_step;
            product1 += product_step;
            decoded2 = decode_mul1_product_2(product0, product1);
            dh2 = __hsub2(decoded2, w2);
            half2 err2;
            if (first) {
                err2 = __hmul2(dh2, dh2);
                if (pre_state >= 0 && in_edge_idx != pre_state) err2 = inf2;
            } else
                err2 = __hfma2(dh2, dh2, __half2half2(costs_in[in_edge_idx]));
            if (__hlt(__low2half(err2), __low2half(min_err2))) {
                min_err2 = __halves2half2(__low2half(err2), __high2half(min_err2));
                min_in_edge0 = in_edge_idx;
            }
            if (__hlt(__high2half(err2), __high2half(min_err2))) {
                min_err2 = __halves2half2(__low2half(min_err2), __high2half(err2));
                min_in_edge1 = in_edge_idx;
            }
        }
        reinterpret_cast<half2*>(costs_out)[out_edge_idx >> 1] = min_err2;
        if (write_hist) {
            hist_row[out_edge_idx] = (uint16_t)min_in_edge0;
            hist_row[out_edge_idx + 1] = (uint16_t)min_in_edge1;
        }
    }
}

/// Fractional-rate Viterbi, KA bits per position plus one where bit (position mod 16) of MASK is set. Costs
/// (2 * edges_max halves per tile) and history (256 * edges_max uint16 per tile) are in global scratch.
template <int KA, uint32_t MASK>
__device__ __forceinline__ void frac_body(
        const float* __restrict__ input_tiles_ptr, float* __restrict__ output_tiles_ptr,
        uint16_t* __restrict__ output_indices_ptr, half* __restrict__ temp_costs_ptr,
        uint16_t* __restrict__ temp_edges_ptr) {
    constexpr int L = kTile;
    constexpr int NT = kFracThreads;
    constexpr int NW = NT / 32;
    constexpr int KB = KA + 1;
    constexpr int edges_max = 65536 >> KA;
    constexpr int edges_last = 65536 >> (KA + (MASK & 1));

    extern __shared__ uint8_t shbuf[];
    uint8_t* sh = shbuf;
    const int tile_idx = blockIdx.x;
    const int thread = threadIdx.x;
    const float* input_tile = input_tiles_ptr + L * tile_idx;
    float* output_tile = output_tiles_ptr + L * tile_idx;
    uint16_t* output_indices = output_indices_ptr + L * tile_idx;
    uint16_t* temp_edges = temp_edges_ptr + (size_t)L * edges_max * tile_idx;

    half* sh_input_tile = (half*)sh;
    sh += L * sizeof(half);
    int* sh_idx = (int*)sh;

    half* temp_costs = temp_costs_ptr + (size_t)2 * edges_max * tile_idx;
    half* temp_costs_inc = temp_costs + edges_max;

    for (int i = thread; i < L; i += NT) sh_input_tile[i] = __float2half_rn(input_tile[i]);
    __syncthreads();

    auto ring = [&](int i, int roll) {
        int ri = i + roll;
        if (ri >= L) ri -= L;
        return ri;
    };

    auto forward = [&](int roll, int pre_state) {
        for (int i = 0; i < L; ++i) {
            const int ri = ring(i, roll);
            half* t = temp_costs;
            temp_costs = temp_costs_inc;
            temp_costs_inc = t;
            const bool first = i == 0;
            const bool write_hist = pre_state >= 0 || ri < L / 2;
            uint16_t* hist_row = temp_edges + (size_t)edges_max * ri;
            const half w = sh_input_tile[ri];
            const int din = (MASK >> (ri & 15)) & 1, dout = (MASK >> ((ri + 1) & 15)) & 1;
            switch (din * 2 + dout) {
                case 0: frac_step<KA, KA>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, thread); break;
                case 1: frac_step<KA, KB>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, thread); break;
                case 2: frac_step<KB, KA>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, thread); break;
                default: frac_step<KB, KB>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, thread); break;
            }
            __syncthreads();
        }
    };

    auto argmin_cost = [&]() {
        uint32_t best = 0x7c00ffffu;
        for (int e = thread; e < edges_last; e += NT) {
            unsigned v = e & 1023;
            unsigned rank = ((__brev(v >> 5) >> 27) << 10) | ((__brev(v & 31) >> 27) << 5) | (e >> 10);
            unsigned key = ((uint32_t)__half_as_ushort(temp_costs[e]) << 16) | rank;
            best = min(best, key);
        }
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) best = min(best, __shfl_xor_sync(0xffffffff, best, offset));
        if ((thread & 31) == 0) ((uint32_t*)sh_idx)[thread >> 5] = best;
        __syncthreads();
        if (thread < 32) {
            best = thread < NW ? ((uint32_t*)sh_idx)[thread] : 0x7c00ffffu;
#pragma unroll
            for (int offset = 16; offset; offset >>= 1) best = min(best, __shfl_xor_sync(0xffffffff, best, offset));
        }
        unsigned rank = best & 65535;
        unsigned v = ((__brev(rank >> 10) >> 27) << 5) | (__brev((rank >> 5) & 31) >> 27);
        return best >= 0x7c000000u ? 0 : (int)(((rank & 31) << 10) | v);
    };

    auto backward = [&](int roll, bool write, int edge) {
        if (thread == 0) {
            for (int i = L - 1; i >= 0; --i) {
                const int ri = ring(i, roll);
                const int prev_edge = (int)temp_edges[(size_t)edges_max * ri + edge];
                const int kin = KA + ((MASK >> (ri & 15)) & 1);
                const int encoded = ((prev_edge << kin) | edge) & 0xFFFF;
                edge = prev_edge;
                if (write) {
                    output_indices[ri] = (uint16_t)encoded;
                    output_tile[ri] = __half2float(decode_mul1(encoded));
                } else if (ri == 0)
                    break;
            }
        }
        if (thread == 0) sh_idx[0] = edge;
        __syncthreads();
        return sh_idx[0];
    };

    forward(L / 2, -1);
    int end_state = backward(L / 2, false, argmin_cost());
    forward(0, end_state);
    backward(0, true, end_state);
}

/// Integer-rate packing of one tile (one block of 128 threads per tile, grid == tile count): K low bits of each state, MSB first, 16 spans of 16 weights, 16-bit halves of
/// each 32-bit word swapped (pack.cu).
template <int K>
__device__ __forceinline__ void pack_body(uint16_t* __restrict__ g_packed, const uint16_t* __restrict__ g_unpacked) {
    constexpr int packed_size = 256 * K / 16;
    __shared__ uint16_t s_unpacked[256];
    __shared__ uint16_t s_packed[packed_size];
    int t = threadIdx.x;
    g_packed += blockIdx.x * packed_size;
    g_unpacked += blockIdx.x * 256;
    ((uint32_t*)s_unpacked)[t] = ((const uint32_t*)g_unpacked)[t];
    __syncthreads();
    const int spans = 16;
    const int len = 256 / spans;
    if (t < spans) {
        int i = len * t;
        int j = K * t;
        int k = 32;
        uint32_t buf = 0;
        for (int n = 0; n < len; ++n) {
            uint32_t v = (uint32_t)s_unpacked[i];
            v &= ((1 << K) - 1);
            k -= K;
            buf |= (v << k);
            if (k <= 16) {
                s_packed[j] = (uint16_t)(buf >> 16);
                buf <<= 16;
                k += 16;
                j++;
            }
            i++;
        }
    }
    __syncthreads();
    if (t < packed_size / 2) ((uint32_t*)g_packed)[t] = __byte_perm(((uint32_t*)s_packed)[t], 0, 0x1032);
}

constexpr int kFracMaxWords = 64;

__device__ __forceinline__ int frac_d(int i, int ka, uint32_t mask) { return ka + ((mask >> (i & 15)) & 1); }

/// Ring position where window i ends.
__device__ __forceinline__ int frac_s(int i, int ka, uint32_t mask, int bpb) {
    int s = (i >> 4) * bpb;
    for (int j = 0; j <= (i & 15); ++j) s += frac_d(j, ka, mask);
    return s;
}

/// 16-bit window starting at ring bit `start`, from MSB-first 32-bit words.
__device__ __forceinline__ uint32_t frac_window(const uint32_t* words, int nw, int start) {
    const int R = nw * 32;
    start %= R;
    if (start < 0) start += R;
    const int wi = start >> 5, o = start & 31;
    const uint64_t v = ((uint64_t)words[wi] << 32) | (uint64_t)words[(wi + 1) % nw];
    return (uint32_t)((v >> (48 - o)) & 0xFFFFu);
}

/// Fractional packing, one thread per tile (frac.cu); `bpb` is the uint16 words per tile.
__device__ __forceinline__ void pack_frac_tile(
        uint16_t* __restrict__ g_packed, const uint16_t* __restrict__ g_unpacked, int tile, int ka, uint32_t mask,
        int bpb) {
    const int nw = bpb / 2;
    uint32_t words[kFracMaxWords];
    for (int w = 0; w < nw; ++w) words[w] = 0;
    const uint16_t* idx = g_unpacked + (size_t)tile * 256;
    int pos = 0;
    for (int i = 0; i < 256; ++i) {
        const int d = frac_d(i, ka, mask);
        const uint32_t v = (uint32_t)idx[i] & ((1u << d) - 1);
        for (int b = d - 1; b >= 0; --b, ++pos)
            if ((v >> b) & 1) words[pos >> 5] |= 1u << (31 - (pos & 31));
    }
    uint32_t* out = (uint32_t*)(g_packed + (size_t)tile * bpb);
    for (int w = 0; w < nw; ++w) out[w] = words[w];
}

/// Decodes tile `tile` of packed words (blockDim must be 256: one thread per weight), integer or fractional: the value
/// at tensor-core position t is the codebook value of the 16-bit window ending at ring position S(t) (frac.cu unpack,
/// exl3_dq.cuh window).
__device__ __forceinline__ float decode_tile_value(
        const uint16_t* __restrict__ g_packed, size_t tile, int ka, uint32_t mask, int bpb) {
    __shared__ uint32_t words[kFracMaxWords];
    const int t = threadIdx.x;
    const int nw = bpb / 2;
    __syncthreads();
    for (int w = t; w < nw; w += blockDim.x) words[w] = ((const uint32_t*)(g_packed + tile * bpb))[w];
    __syncthreads();
    return __half2float(decode_mul1(frac_window(words, nw, frac_s(t, ka, mask, bpb) - 16)));
}

__device__ __forceinline__ void decode_tile(
        const uint16_t* __restrict__ g_packed, float* __restrict__ g_decoded, int ka, uint32_t mask, int bpb) {
    g_decoded[(size_t)blockIdx.x * 256 + threadIdx.x] = decode_tile_value(g_packed, blockIdx.x, ka, mask, bpb);
}

/// Row-major element (r * 16 + c) of a 16x16 tile at tensor-core position `pos`: for t = pos / 8, r0 = (t % 4) * 2 and
/// c0 = t / 4, the eight positions of t are rows r0, r0 + 1, r0 + 8, r0 + 9 at columns c0, then c0 + 8.
__device__ __forceinline__ int tensor_core_element(int pos) {
    const int t = pos >> 3, j = pos & 7;
    const int r0 = (t & 3) * 2;
    const int row = r0 + ((j & 3) >> 1) * 8 + (j & 1);
    const int col = (t >> 2) + (j >> 2) * 8;
    return row * 16 + col;
}

/// Decodes tile (blockIdx.x) of a `[k / 16, n / 16]` grid into the row-major float32 matrix `w` `[k, n]`.
__device__ __forceinline__ void reconstruct_tile(const uint16_t* __restrict__ g_packed, float* __restrict__ w, int n,
        int ka, uint32_t mask, int bpb) {
    const int tiles_n = n >> 4;
    const int ti = blockIdx.x / tiles_n, tj = blockIdx.x % tiles_n;
    const float v = decode_tile_value(g_packed, blockIdx.x, ka, mask, bpb);
    const int e = tensor_core_element(threadIdx.x);
    w[(size_t)(ti * 16 + (e >> 4)) * n + tj * 16 + (e & 15)] = v;
}

}  // namespace

#define VITERBI(K)                                                                                          \
    extern "C" __global__ __launch_bounds__(viterbi_threads(K), 1) void viterbi_k##K(                        \
            const float* tiles, int count, uint16_t* indices, float* decoded, uint16_t* scratch) {           \
        viterbi_body<K>(tiles, decoded, indices, scratch);                                                   \
    }
VITERBI(2)
VITERBI(3)
VITERBI(4)
VITERBI(5)
VITERBI(6)

#define VITERBI_FRAC(KA)                                                                                     \
    extern "C" __global__ __launch_bounds__(kFracThreads, 2) void viterbi_frac_k##KA(                        \
            const float* tiles, int count, uint16_t* indices, float* decoded, uint16_t* scratch) {           \
        constexpr size_t edges_max = 65536 >> KA;                                                            \
        half* costs = reinterpret_cast<half*>(scratch);                                                      \
        uint16_t* history = scratch + (size_t)count * 2 * edges_max;                                         \
        frac_body<KA, 0xaaaau>(tiles, decoded, indices, costs, history);                                     \
    }
VITERBI_FRAC(2)
VITERBI_FRAC(3)

#define PACK(K)                                                                                              \
    extern "C" __global__ __launch_bounds__(128) void pack_k##K(                                             \
            const uint16_t* indices, int count, uint16_t* words) {                                           \
        pack_body<K>(words, indices);                                                                        \
    }
PACK(2)
PACK(3)
PACK(4)
PACK(5)
PACK(6)

#define PACK_FRAC(KA)                                                                                        \
    extern "C" __global__ __launch_bounds__(128) void pack_frac_k##KA(                                       \
            const uint16_t* indices, int count, uint16_t* words) {                                           \
        const int tile = blockIdx.x * 128 + threadIdx.x;                                                     \
        if (tile < count) pack_frac_tile(words, indices, tile, KA, 0xaaaau, 16 * KA + 8);                   \
    }
PACK_FRAC(2)
PACK_FRAC(3)

#define DECODE(K)                                                                                            \
    extern "C" __global__ __launch_bounds__(256) void decode_k##K(                                           \
            const uint16_t* words, int count, float* decoded) {                                              \
        decode_tile(words, decoded, K, 0u, 16 * K);                                                          \
    }
DECODE(2)
DECODE(3)
DECODE(4)
DECODE(5)
DECODE(6)

#define DECODE_FRAC(KA)                                                                                      \
    extern "C" __global__ __launch_bounds__(256) void decode_frac_k##KA(                                     \
            const uint16_t* words, int count, float* decoded) {                                              \
        decode_tile(words, decoded, KA, 0xaaaau, 16 * KA + 8);                                               \
    }
DECODE_FRAC(2)
DECODE_FRAC(3)

#define RECONSTRUCT(K)                                                                                       \
    extern "C" __global__ __launch_bounds__(256) void reconstruct_k##K(                                      \
            const uint16_t* words, int n, float* w) {                                                        \
        reconstruct_tile(words, w, n, K, 0u, 16 * K);                                                        \
    }
RECONSTRUCT(2)
RECONSTRUCT(3)
RECONSTRUCT(4)
RECONSTRUCT(5)
RECONSTRUCT(6)

#define RECONSTRUCT_FRAC(KA)                                                                                 \
    extern "C" __global__ __launch_bounds__(256) void reconstruct_frac_k##KA(                                \
            const uint16_t* words, int n, float* w) {                                                        \
        reconstruct_tile(words, w, n, KA, 0xaaaau, 16 * KA + 8);                                             \
    }
RECONSTRUCT_FRAC(2)
RECONSTRUCT_FRAC(3)

namespace {

constexpr float kInvSqrt128 = 0.08838834764831845f;

/// Walsh-Hadamard butterflies below are Sylvester natural order, H[i][j] = (-1)^popcount(i & j).
}  // namespace

/// x[b * 128 + i, c] <- sum_j H[i][j] x[b * 128 + j, c] / sqrt(128) for every 128-row block b and column c. Block:
/// 32 columns x 8 row lanes; grid (ceil(cols / 32), rows / 128). The shared tile is padded against bank conflicts.
extern "C" __global__ __launch_bounds__(256) void hadamard128_rows(float* x, int rows, int cols) {
    __shared__ float s[128][33];
    const int lane = threadIdx.x & 31, sub = threadIdx.x >> 5;
    const int c = blockIdx.x * 32 + lane;
    float* base = x + (size_t)blockIdx.y * 128 * cols;
    for (int r = sub; r < 128; r += 8) s[r][lane] = c < cols ? base[(size_t)r * cols + c] : 0.f;
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
        for (int p = sub; p < 64; p += 8) {
            const int i = ((p / h) * 2 * h) + (p % h);
            const float a = s[i][lane], b = s[i + h][lane];
            s[i][lane] = a + b;
            s[i + h][lane] = a - b;
        }
        __syncthreads();
    }
    if (c < cols)
        for (int r = sub; r < 128; r += 8) base[(size_t)r * cols + c] = s[r][lane] * kInvSqrt128;
}

/// x[r, b * 128 + i] <- sum_j H[i][j] x[r, b * 128 + j] / sqrt(128) for every row r and 128-column block b. Block: 128
/// threads on one segment; grid (cols / 128, rows).
extern "C" __global__ __launch_bounds__(128) void hadamard128_columns(float* x, int rows, int cols) {
    __shared__ float s[128];
    float* base = x + (size_t)blockIdx.y * cols + (size_t)blockIdx.x * 128;
    const int t = threadIdx.x;
    s[t] = base[t];
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
        if (t < 64) {
            const int i = ((t / h) * 2 * h) + (t % h);
            const float a = s[i], b = s[i + h];
            s[i] = a + b;
            s[i + h] = a - b;
        }
        __syncthreads();
    }
    base[t] = s[t] * kInvSqrt128;
}

/// Gathers the 16-row strip of `w` `[k, n]` starting at `first_row` into n / 16 tiles of 256 values in tensor-core
/// order (`inverse` false), or scatters the tiles back into the strip (`inverse` true). One thread per value.
extern "C" __global__ __launch_bounds__(256) void strip_tiles(
        float* w, int n, int first_row, float* tiles, int inverse) {
    const int tile = blockIdx.x, pos = threadIdx.x;
    const int e = tensor_core_element(pos);
    float* element = w + (size_t)(first_row + (e >> 4)) * n + tile * 16 + (e & 15);
    float* slot = tiles + (size_t)tile * 256 + pos;
    if (inverse) *element = *slot;
    else *slot = *element;
}

/// w[r, :] *= s[r] for fp16 s; grid (ceil(n / 256), k).
extern "C" __global__ void scale_rows_fp16(float* w, const __half* s, int k, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) w[(size_t)blockIdx.y * n + c] *= __half2float(s[blockIdx.y]);
}

/// w[:, c] *= s[c] for fp16 s; grid (ceil(n / 256), k).
extern "C" __global__ void scale_columns_fp16(float* w, const __half* s, int k, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) w[(size_t)blockIdx.y * n + c] *= __half2float(s[c]);
}

/// out[c, r] = bf16(w[r, c]) (round to nearest even) for w `[k, n]`, out `[n, k]`. Block 256 as 32 x 8; grid
/// (ceil(n / 32), ceil(k / 32)).
extern "C" __global__ __launch_bounds__(256) void transpose_to_bf16(
        const float* w, int k, int n, __nv_bfloat16* out) {
    __shared__ float s[32][33];
    const int x = threadIdx.x & 31, y = threadIdx.x >> 5;
    for (int j = y; j < 32; j += 8) {
        const int r = blockIdx.y * 32 + j, c = blockIdx.x * 32 + x;
        if (r < k && c < n) s[j][x] = w[(size_t)r * n + c];
    }
    __syncthreads();
    for (int j = y; j < 32; j += 8) {
        const int c = blockIdx.x * 32 + j, r = blockIdx.y * 32 + x;
        if (r < k && c < n) out[(size_t)c * k + r] = __float2bfloat16_rn(s[x][j]);
    }
}

// ---- Quantizer kernels: Hessian preparation, block LDL, regularization, scale search, LDLQ strips, refit ----

namespace {

/// Sum of `v` over the block (all threads must call); the result is valid in thread 0. `shared` holds 32 doubles.
__device__ inline double block_sum(double v, double* shared) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    const int warps = (blockDim.x + 31) >> 5;
    v = threadIdx.x < warps ? shared[threadIdx.x] : 0.0;
    if (warp == 0)
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

}  // namespace

extern "C" __global__ void fill_floats(float* x, long long count, float value) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
            i += (long long)gridDim.x * blockDim.x)
        x[i] = value;
}

extern "C" __global__ void copy_floats(const float* src, float* dst, long long count) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
            i += (long long)gridDim.x * blockDim.x)
        dst[i] = src[i];
}

extern "C" __global__ void float_to_half(const float* x, long long count, __half* out) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
            i += (long long)gridDim.x * blockDim.x)
        out[i] = __float2half_rn(x[i]);
}

/// y += alpha * x
extern "C" __global__ void axpy_floats(float* y, const float* x, long long count, float alpha) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
            i += (long long)gridDim.x * blockDim.x)
        y[i] += alpha * x[i];
}

extern "C" __global__ void subtract_floats(const float* a, const float* b, float* out, long long count) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
            i += (long long)gridDim.x * blockDim.x)
        out[i] = a[i] - b[i];
}

/// out[0] += sum(a * b), in double.
extern "C" __global__ __launch_bounds__(256) void sum_products(
        const float* a, const float* b, long long count, double* out) {
    __shared__ double shared[32];
    double sum = 0.0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
            i += (long long)gridDim.x * blockDim.x)
        sum += (double)a[i] * (double)b[i];
    sum = block_sum(sum, shared);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

/// out[0] = mean of v[i * stride] for i < count (one block of 1024 threads).
extern "C" __global__ __launch_bounds__(1024) void strided_mean(
        const float* v, int count, long long stride, float* out) {
    __shared__ double shared[32];
    double sum = 0.0;
    for (int i = threadIdx.x; i < count; i += blockDim.x) sum += v[(long long)i * stride];
    sum = block_sum(sum, shared);
    if (threadIdx.x == 0) out[0] = (float)(sum / count);
}

/// h[i, i] += factor * mean[0] for the k x k matrix h.
extern "C" __global__ void add_diagonal(float* h, int k, const float* mean, float factor) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) h[(long long)i * k + i] += factor * mean[0];
}

/// store != 0: d = diag(h); otherwise diag(h) = d.
extern "C" __global__ void diagonal_io(float* h, int k, float* d, int store) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    if (store) d[i] = h[(long long)i * k + i];
    else h[(long long)i * k + i] = d[i];
}

/// mode 0: upper = lower; 1: lower = upper; 2: strict upper = 0. Grid (ceil(k / 32), ceil(k / 8)), block 32 x 8.
extern "C" __global__ __launch_bounds__(256) void triangle(float* h, int k, int mode) {
    const int c = blockIdx.x * 32 + (threadIdx.x & 31), r = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (r >= k || c >= r) return;
    if (mode == 0) h[(long long)c * k + r] = h[(long long)r * k + c];
    else if (mode == 1) h[(long long)r * k + c] = h[(long long)c * k + r];
    else h[(long long)c * k + r] = 0.f;
}

/// x[r, c] *= s[axis == 0 ? r : c], or /= with divide; grid (ceil(cols / 256), rows).
extern "C" __global__ void scale_floats(float* x, int rows, int cols, const float* s, int axis, int divide) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    const float v = s[axis == 0 ? blockIdx.y : c];
    float* p = x + (long long)blockIdx.y * cols + c;
    *p = divide ? *p / v : *p * v;
}

/// out[c, r] = float(w[r, c]) for the bf16 w `[n, k]`, out `[k, n]`. Block 32 x 8; grid (ceil(k / 32), ceil(n / 32)).
extern "C" __global__ __launch_bounds__(256) void bf16_to_float_transposed(
        const __nv_bfloat16* w, int n, int k, float* out) {
    __shared__ float s[32][33];
    const int x = threadIdx.x & 31, y = threadIdx.x >> 5;
    for (int j = y; j < 32; j += 8) {
        const int r = blockIdx.y * 32 + j, c = blockIdx.x * 32 + x;
        if (r < n && c < k) s[j][x] = __bfloat162float(w[(long long)r * k + c]);
    }
    __syncthreads();
    for (int j = y; j < 32; j += 8) {
        const int c = blockIdx.x * 32 + j, r = blockIdx.y * 32 + x;
        if (r < n && c < k) out[(long long)c * n + r] = s[x][j];
    }
}

/// inv of each 16 x 16 lower-triangular diagonal block of the row-major lower factor l, by forward substitution
/// (one thread per inverse column). Grid k / 16, block 16.
extern "C" __global__ __launch_bounds__(16) void invert_diagonal_blocks(const float* l, int k, float* inv) {
    __shared__ float d[16][16];
    const int b = blockIdx.x, j = threadIdx.x;
    for (int i = 0; i < 16; i++) d[i][j] = l[(long long)(16 * b + i) * k + 16 * b + j];
    __syncthreads();
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        float acc = i == j ? 1.f : 0.f;
#pragma unroll
        for (int m = 0; m < i; m++) acc -= d[i][m] * x[m];
        x[i] = acc / d[i][i];
        inv[(long long)b * 256 + i * 16 + j] = x[i];
    }
}

/// l[:, 16b : 16b + 16] <- l[:, 16b : 16b + 16] * inv[b] for the rows of and below block b; the diagonal block itself
/// becomes 0 (identity with a zeroed diagonal). Rows above are zero already. Grid (ceil(k / 128), k / 16), block 128.
extern "C" __global__ __launch_bounds__(128) void right_multiply_blocks(float* l, int k, const float* inv) {
    __shared__ float s[256];
    const int b = blockIdx.y;
    for (int i = threadIdx.x; i < 256; i += 128) s[i] = inv[(long long)b * 256 + i];
    __syncthreads();
    const int r = blockIdx.x * 128 + threadIdx.x;
    if (r < 16 * b || r >= k) return;
    float* row = l + (long long)r * k + 16 * b;
    if (r < 16 * b + 16) {
#pragma unroll
        for (int c = 0; c < 16; c++) row[c] = 0.f;
        return;
    }
    float v[16];
#pragma unroll
    for (int m = 0; m < 16; m++) v[m] = row[m];
#pragma unroll
    for (int c = 0; c < 16; c++) {
        float acc = 0.f;
#pragma unroll
        for (int m = 0; m < 16; m++) acc += v[m] * s[m * 16 + c];
        row[c] = acc;
    }
}

/// out[c] = sqrt(sum_r w[r, c]^2 / k) for the row-major w `[k, n]`. Block 32 x 8; grid ceil(n / 32).
extern "C" __global__ __launch_bounds__(256) void column_rms(const float* w, int k, int n, float* out) {
    __shared__ double s[8][33];
    const int lane = threadIdx.x & 31, sub = threadIdx.x >> 5;
    const int c = blockIdx.x * 32 + lane;
    double sum = 0.0;
    if (c < n)
        for (int r = sub; r < k; r += 8) {
            const double v = w[(long long)r * n + c];
            sum += v * v;
        }
    s[sub][lane] = sum;
    __syncthreads();
    if (sub == 0 && c < n) {
        double total = 0.0;
        for (int i = 0; i < 8; i++) total += s[i][lane];
        out[c] = (float)sqrt(total / k);
    }
}

/// out[r] = sqrt(sum_c w[r, c]^2 / n). Grid k, block 256.
extern "C" __global__ __launch_bounds__(256) void row_rms(const float* w, int k, int n, float* out) {
    __shared__ double shared[32];
    double sum = 0.0;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const double v = w[(long long)blockIdx.x * n + c];
        sum += v * v;
    }
    sum = block_sum(sum, shared);
    if (threadIdx.x == 0) out[blockIdx.x] = (float)sqrt(sum / n);
}

/// Output channel scales of exllamav3's regularize: from the column RMS and its mean, the divisor of the weight
/// (sign, times the normalised scale when applied) and the stored sv (the divisor, 0 for channels without energy).
extern "C" __global__ void output_scales(const float* rms, const float* mean, int n, const float* signs,
        int apply, float* divisor, float* stored) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const bool normalised = mean[0] > 1e-30f;
    float s = normalised ? rms[c] / mean[0] : rms[c];
    const bool zero = fabsf(s) < 1e-30f;
    float d = signs[c];
    if (apply || !normalised) {
        if (zero) s = 0.1f;
        d = signs[c] * s + 1e-10f;
    }
    divisor[c] = d;
    stored[c] = zero ? 0.f : d;
}

/// su = sign * rms / -codebook_scale + 1e-10, rms below 1e-30 read as 0.1.
extern "C" __global__ void input_scales(const float* rms, const float* signs, int k, float* su) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= k) return;
    float s = rms[r];
    if (fabsf(s) < 1e-30f) s = 0.1f;
    su[r] = signs[r] * s / (-1.24371088f) + 1e-10f;
}

/// out[tile] = mean of squares of tile (blockIdx.y, blockIdx.x) of the row-major w with row length n. Block 256.
extern "C" __global__ __launch_bounds__(256) void tile_mean_square(const float* w, int n, float* out) {
    __shared__ double shared[32];
    const int row = threadIdx.x >> 4, col = threadIdx.x & 15;
    const double v = w[(long long)(blockIdx.y * 16 + row) * n + blockIdx.x * 16 + col];
    const double sum = block_sum(v * v, shared);
    if (threadIdx.x == 0) out[blockIdx.y * gridDim.x + blockIdx.x] = (float)(sum / 256.0);
}

/// Sample tile s (grid, block 256): tile (tile_k[s], tile_n[s]) of w in tensor-core order, times scale.
extern "C" __global__ __launch_bounds__(256) void gather_tiles(
        const float* w, int n, const int* tile_k, const int* tile_n, float scale, float* out) {
    const int e = tensor_core_element(threadIdx.x);
    const float v = w[(long long)(tile_k[blockIdx.x] * 16 + (e >> 4)) * n + tile_n[blockIdx.x] * 16 + (e & 15)];
    out[(long long)blockIdx.x * 256 + threadIdx.x] = v * scale;
}

/// Candidate c, sample j (every `stride`th of the samples) times scales[c]: out `[cands, per, 256]`.
extern "C" __global__ void candidate_tiles(
        const float* samples, int stride, int per, const float* scales, long long count, float* out) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
            i += (long long)gridDim.x * blockDim.x) {
        const long long tile = i >> 8;
        const long long cand = tile / per, j = tile % per;
        out[i] = samples[(j * stride) * 256 + (i & 255)] * scales[cand];
    }
}

/// out[c] = mean((q[c] / scales[c] - sample)^2) over candidate c's tiles. Grid cands, block 256.
extern "C" __global__ __launch_bounds__(256) void candidate_mse(
        const float* q, const float* samples, int stride, int per, const float* scales, float* out) {
    __shared__ double shared[32];
    const float s = scales[blockIdx.x];
    double sum = 0.0;
    const long long total = (long long)per * 256;
    for (long long i = threadIdx.x; i < total; i += blockDim.x) {
        const long long j = i >> 8;
        const float d = q[(long long)blockIdx.x * total + i] / s - samples[(j * stride) * 256 + (i & 255)];
        sum += (double)d * d;
    }
    sum = block_sum(sum, shared);
    if (threadIdx.x == 0) out[blockIdx.x] = (float)(sum / (double)total);
}

/// The 16-row strip of w (plus p when given) at first_row as n / 16 tiles in tensor-core order. Grid n / 16, block 256.
extern "C" __global__ __launch_bounds__(256) void strip_gather_sum(
        const float* w, const float* p, int n, int first_row, float* tiles) {
    const int e = tensor_core_element(threadIdx.x);
    const long long at = (long long)(first_row + (e >> 4)) * n + blockIdx.x * 16 + (e & 15);
    tiles[(long long)blockIdx.x * 256 + threadIdx.x] = p ? w[at] + p[at] : w[at];
}

/// e[(e_row + r) * n + c] = w[(first_row + r) * n + c] - q[(first_row + r) * n + c] for the 16 rows of a strip.
extern "C" __global__ void strip_error(
        const float* w, const float* q, int n, int first_row, float* e, int e_row) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 16 * n) return;
    e[(long long)e_row * n + i] = w[(long long)first_row * n + i] - q[(long long)first_row * n + i];
}

/// out[c] = sum_r a[r, c] * b[r, c] + extra * sum_r a[r, c]^2. Block 32 x 8; grid ceil(n / 32).
extern "C" __global__ __launch_bounds__(256) void column_dots(
        const float* a, const float* b, float extra, int k, int n, float* out) {
    __shared__ double s[8][33];
    const int lane = threadIdx.x & 31, sub = threadIdx.x >> 5;
    const int c = blockIdx.x * 32 + lane;
    double sum = 0.0;
    if (c < n)
        for (int r = sub; r < k; r += 8) {
            const double x = a[(long long)r * n + c];
            sum += x * b[(long long)r * n + c] + (double)extra * x * x;
        }
    s[sub][lane] = sum;
    __syncthreads();
    if (sub == 0 && c < n) {
        double total = 0.0;
        for (int i = 0; i < 8; i++) total += s[i][lane];
        out[c] = (float)total;
    }
}

/// out[r] = sum_c a[r, c] * b[r, c]. Grid k, block 256.
extern "C" __global__ __launch_bounds__(256) void row_dots(
        const float* a, const float* b, int k, int n, float* out) {
    __shared__ double shared[32];
    double sum = 0.0;
    for (int c = threadIdx.x; c < n; c += blockDim.x)
        sum += (double)a[(long long)blockIdx.x * n + c] * b[(long long)blockIdx.x * n + c];
    sum = block_sum(sum, shared);
    if (threadIdx.x == 0) out[blockIdx.x] = (float)sum;
}

/// num[c] <- dn[c] > 1e-30 ? num[c] / dn[c] : 1
extern "C" __global__ void column_ratio(float* num, const float* dn, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) num[c] = dn[c] > 1e-30f ? num[c] / fmaxf(dn[c], 1e-30f) : 1.f;
}

/// a[r, c] *= h[r, c] (+ damping on the diagonal of h); grid (ceil(k / 256), k).
extern "C" __global__ void hessian_product(float* a, const float* h, int k, float damping) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    const long long at = (long long)blockIdx.y * k + c;
    a[at] *= h[at] + (c == blockIdx.y ? damping : 0.f);
}

/// Solution of the input-side system, kept where finite and positive and 1 elsewhere (all 1 when the factorization
/// failed).
extern "C" __global__ void clean_row_scales(float* r, const int* info, int k, int* failures) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    if (i == 0 && info[0] != 0) atomicAdd(failures, 1);
    const float v = r[i];
    r[i] = info[0] == 0 && isfinite(v) && v > 0.f ? v : 1.f;
}

/// x *= s, or /= s with divide.
extern "C" __global__ void scale_scalar(float* x, long long count, float s, int divide) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
            i += (long long)gridDim.x * blockDim.x)
        x[i] = divide ? x[i] / s : x[i] * s;
}
