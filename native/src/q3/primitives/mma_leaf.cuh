#pragma once
#include "mma.cuh"
#include "writeback.cuh"

namespace q3 {
// MMA leaves consume one staged K tile into per-warp FP32 accumulators and
// later store them to the shared result tile. Every leaf reads the same shared
// A (row-major, LDA) and B hi/lo (column-major, LDB) tiles and issues, for
// every accumulator element, the same sequence of BF16 MMA steps:
//   for each 16-wide k: acc += A * B_hi; acc += A * B_lo
// PTX does not specify MMA accumulation rounding across instruction shapes, so
// FP32 accumulator equality with WmmaLeaf is a tested property of the target
// (test_explicit_mma_leaves_match_wmma_bitwise), not a language guarantee.

// WMMA leaf: the portable fragment API.
template<class Tile>
struct WmmaLeaf {
    using Acc = Accumulators<Tile::kFrags>;
    static __device__ __forceinline__ void fill(Acc& acc) { acc.fill(); }
    template<int K_TILE, int LDA, int LDB>
    static __device__ __forceinline__ void consume(Acc& acc, const __nv_bfloat16* a, const __nv_bfloat16* b_hi,
            const __nv_bfloat16* b_lo, unsigned int warp) {
        consume_mma_tile<Tile, K_TILE, LDA, LDB>(acc, a, b_hi, b_lo, warp);
    }
    static __device__ __forceinline__ void store(float* result, Acc& acc, unsigned int warp) {
        store_accumulators<Tile>(result, acc, warp);
    }
};

// Explicit ldmatrix/mma.sync leaf. Each warp's 16x16 output band is two
// m16n8k16 tiles (n halves); the fragment registers follow the PTX ISA layouts.
// SCOPE: warp-collective. BORROWS: shared A and B tiles, read-only; the same
// staging and barrier contract as consume_mma_tile.
struct MmaFragments {
    unsigned int a[4];      // A 16x16: rows 0-7/8-15 x k 0-7/8-15
    unsigned int b[2][4];   // [hi/lo]: n 0-7 (k 0-7, 8-15), n 8-15 (k 0-7, 8-15)
};

static __device__ __forceinline__ void ldmatrix_x4(unsigned int (&r)[4], const __nv_bfloat16* p) {
    unsigned int address = (unsigned int)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

static __device__ __forceinline__ void mma_16816(float (&c)[4], const unsigned int (&a)[4],
        unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// PINGPONG keeps two fragment register sets and loads step k + 16 before
// issuing the MMAs of step k; the MMA sequence is unchanged.
template<class Tile, bool PINGPONG = false>
struct MmaSyncLeaf {
    struct Acc {
        float c[Tile::kFrags][2][4];  // [m][n half][fragment element]
    };
    static __device__ __forceinline__ void fill(Acc& acc) {
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; m++)
            #pragma unroll
            for (int h = 0; h < 2; h++)
                #pragma unroll
                for (int i = 0; i < 4; i++) acc.c[m][h][i] = 0.0f;
    }

    // Load the B hi/lo fragments for one 16-wide k step. Lane l addresses
    // column n = (l & 7) + 8 * (l >> 4) at k offset 8 * ((l >> 3) & 1).
    template<int LDB>
    static __device__ __forceinline__ void load_b(unsigned int (&b)[2][4], const __nv_bfloat16* b_hi,
            const __nv_bfloat16* b_lo, unsigned int warp, unsigned int k, unsigned int lane) {
        unsigned int offset = (Tile::col(warp) + (lane & 7u) + ((lane >> 4) << 3)) * LDB + k + ((lane >> 3) & 1u) * 8u;
        ldmatrix_x4(b[0], b_hi + offset);
        ldmatrix_x4(b[1], b_lo + offset);
    }
    // Load one A fragment. Lane l addresses row l & 15 at k offset 8 * (l >> 4).
    template<int LDA>
    static __device__ __forceinline__ void load_a(unsigned int (&a)[4], const __nv_bfloat16* tile,
            unsigned int warp, int m, unsigned int k, unsigned int lane) {
        ldmatrix_x4(a, tile + (Tile::row(warp, m) + (lane & 15u)) * LDA + k + (lane >> 4) * 8u);
    }
    // Same per-element order as the WMMA leaf: hi for every fragment, then lo.
    static __device__ __forceinline__ void step(Acc& acc, const unsigned int (&a)[Tile::kFrags][4],
            const unsigned int (&b)[2][4]) {
        #pragma unroll
        for (int part = 0; part < 2; part++)
            #pragma unroll
            for (int m = 0; m < Tile::kFrags; m++) {
                mma_16816(acc.c[m][0], a[m], b[part][0], b[part][1]);
                mma_16816(acc.c[m][1], a[m], b[part][2], b[part][3]);
            }
    }

    template<int K_TILE, int LDA, int LDB>
    static __device__ __forceinline__ void consume(Acc& acc, const __nv_bfloat16* a, const __nv_bfloat16* b_hi,
            const __nv_bfloat16* b_lo, unsigned int warp) {
        const unsigned int lane = threadIdx.x & 31u;
        if (PINGPONG) {
        unsigned int af[2][Tile::kFrags][4], bf[2][2][4];
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; m++) load_a<LDA>(af[0][m], a, warp, m, 0, lane);
        load_b<LDB>(bf[0], b_hi, b_lo, warp, 0, lane);
        #pragma unroll
        for (int s = 0; s < K_TILE / 16; s++) {
            if (s + 1 < K_TILE / 16) {
                #pragma unroll
                for (int m = 0; m < Tile::kFrags; m++) load_a<LDA>(af[(s + 1) & 1][m], a, warp, m, (s + 1) * 16, lane);
                load_b<LDB>(bf[(s + 1) & 1], b_hi, b_lo, warp, (s + 1) * 16, lane);
            }
            step(acc, af[s & 1], bf[s & 1]);
        }
        } else {
        #pragma unroll
        for (int s = 0; s < K_TILE / 16; s++) {
            unsigned int af[Tile::kFrags][4], bf[2][4];
            #pragma unroll
            for (int m = 0; m < Tile::kFrags; m++) load_a<LDA>(af[m], a, warp, m, s * 16, lane);
            load_b<LDB>(bf, b_hi, b_lo, warp, s * 16, lane);
            step(acc, af, bf);
        }
        }
    }

    // Store to the row-major shared result tile (leading dimension Tile::kCols).
    // Element i of an m16n8 accumulator is row lane / 4 + 8 * (i >> 1), column
    // 2 * (lane % 4) + (i & 1).
    static __device__ __forceinline__ void store(float* result, Acc& acc, unsigned int warp) {
        const unsigned int lane = threadIdx.x & 31u;
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; m++)
            #pragma unroll
            for (int h = 0; h < 2; h++)
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    unsigned int row = Tile::row(warp, m) + lane / 4u + 8u * (i >> 1);
                    unsigned int col = Tile::col(warp) + h * 8u + 2u * (lane % 4u) + (i & 1);
                    result[row * Tile::kCols + col] = acc.c[m][h][i];
                }
    }
};
template<class Tile>
using MmaPingPongLeaf = MmaSyncLeaf<Tile, true>;


}  // namespace q3
