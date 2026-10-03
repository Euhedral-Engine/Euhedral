// Q3 prefill on the block-scaled FP8 tensor path (sm_120a). Module root.
//
// GeForce Blackwell runs FP32-accumulating FP16/BF16 MMA at half rate (102 TFLOPS) but the block-scaled FP8 kind
// (mma.sync kind::mxf8f6f4, E4M3 x E4M3, FP32 accumulate) at full rate (408 TFLOPS), docs/nvidia/tensor-cores.md.
// Q3 codes (-4..3) are exact in E4M3. A BF16 activation has eight significant bits, which two E4M3 terms hold
// exactly: x = 2^e (hi + lo), one UE8M0 block scale 2^e per 32 values, hi = E4M3(x / 2^e), lo = E4M3 of the
// residual. The MMA applies the activation scale (scale A), the weights take unit scales and the FP16 group
// scale of each 64-code weight group is applied in FP32 after the group's four MMAs, so
//   y = sum over groups of w_scale * (sum over k in group of (hi + lo) * 2^e * code)
// has exact operands and FP32 accumulation. The relaxed BF16 route rounds code * scale to BF16 first.
//
// Flow: euhedral_q3mx_quantize (BF16 rows -> hi, lo, scale planes), then a 128 x 128 tile GEMM, 8 warps of 64 x 32.
// A tiles arrive by cp.async; the compact Q3 weight groups (24 bytes) are expanded to E4M3 bytes by two threads
// per row, one group ahead of the MMAs.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace q3mx {
constexpr int kStages = 3, kTile = 128 * 64;
constexpr int kStageBytes = 3 * kTile + 128 * 4;  // A hi, A lo, expanded W, FP32 weight-group scales per tile row
constexpr unsigned kSharedBytes = kStages * kStageBytes;

// 16-byte chunk c of a 64-byte tile row, XOR-swizzled so the eight rows of an ldmatrix phase hit distinct banks.
__device__ __forceinline__ unsigned swizzle(unsigned row, unsigned chunk) { return row * 64u + ((chunk ^ ((row >> 1) & 3u)) << 4); }
__device__ __forceinline__ unsigned shared_address(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
    const unsigned size = valid ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(shared_address(dst)), "l"(src), "r"(size));
}
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(shared_address(p)));
}
// Block-scaled E4M3 MMA, FP32 accumulate. Scale A: lane 4r supplies row r, lane 4r + 1 row r + 8, byte KB of the
// register (this k32 block). Scale B: unit.
template<int KB>
__device__ __forceinline__ void mma_e4m3(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1, unsigned scale_a) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%11, 0}, %12, {0, 0};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(scale_a), "n"(KB), "r"(0x7f7f7f7fu));
}
// E4M3 byte of each signed 3-bit code (two's complement), as the byte table of prmt: index = code & 7.
constexpr unsigned kCodeBytesLow = 0x44403800u;   // 0, 1, 2, 3
constexpr unsigned kCodeBytesHigh = 0xB8C0C4C8u;  // -4, -3, -2, -1

// Four E4M3 bytes (codes 4 j .. 4 j + 3 of 32 consecutive codes held as 96 bits in three words).
template<int J>
__device__ __forceinline__ unsigned expand_word(unsigned w0, unsigned w1, unsigned w2) {
    // Twelve bits (four codes) at bit 12 J of the 96-bit stream.
    constexpr int bit = 12 * J, word = bit >> 5, shift = bit & 31;
    const unsigned lo = word == 0 ? w0 : word == 1 ? w1 : w2, hi = word == 0 ? w1 : word == 1 ? w2 : 0u;
    const unsigned v = __funnelshift_r(lo, hi, shift) & 0xFFFu;
    const unsigned selector = (v & 0x7u) | ((v & 0x38u) << 1) | ((v & 0x1C0u) << 2) | ((v & 0xE00u) << 3);
    return __byte_perm(kCodeBytesLow, kCodeBytesHigh, selector);
}

// One tile: PAIRED = 0 writes out[row][col] for 128 weight rows; PAIRED = 1 treats the weight rows as gate rows
// followed by up rows, pairs 64 of each per tile and writes silu(gate) * up. `width` is the output row length.
template<int PAIRED>
__device__ __forceinline__ void gemm(const unsigned char* a_hi, const unsigned char* a_lo, const unsigned char* a_scales,
        const unsigned char* codes, unsigned short* out, unsigned rows, unsigned weight_rows, unsigned k,
        unsigned long long scale_offset) {
    extern __shared__ __align__(128) unsigned char smem[];
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, t = lane & 3u;
    const unsigned tiles_m = (rows + 127u) / 128u;
    const unsigned tm = blockIdx.x % tiles_m, tn = blockIdx.x / tiles_m;
    const unsigned row0 = tm * 128u;
    const unsigned wm = warp >> 2, wn = warp & 3u;
    const unsigned groups = k / 64u, scale_stride = k / 32u;
    const unsigned short* weight_scales = reinterpret_cast<const unsigned short*>(codes + scale_offset);
    auto a_tile = [&](unsigned stage, unsigned plane) { return smem + stage * kStageBytes + plane * kTile; };
    auto w_scale_row = [&](unsigned stage) { return reinterpret_cast<float*>(smem + stage * kStageBytes + 3 * kTile); };

    // Activation copies: each thread owns rows r0 and r0 + 64 of the hi and lo planes, chunk c.
    const unsigned copy_row = tid >> 2, copy_chunk = tid & 3u, copy_offset = swizzle(copy_row, copy_chunk);
    const unsigned char* a_source[4];
    bool a_valid[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned r = row0 + copy_row + 64u * (i & 1);
        a_valid[i] = r < rows;
        a_source[i] = (i < 2 ? a_hi : a_lo) + (unsigned long long)(a_valid[i] ? r : 0u) * k + copy_chunk * 16u;
    }
    auto issue_activations = [&](unsigned group, unsigned stage) {
#pragma unroll
        for (int i = 0; i < 4; i++) copy16(a_tile(stage, i >> 1) + (i & 1) * 4096u + copy_offset, a_source[i] + group * 64u, a_valid[i]);
    };
    // Weight producer: thread tid expands half (tid & 1) of tile row tid >> 1.
    const unsigned produce_row = tid >> 1, half = tid & 1u;
    unsigned weight_row;
    if (PAIRED) {
        const unsigned pair_half = weight_rows / 2u, i = produce_row & 31u, w = produce_row >> 5;
        weight_row = (i < 16u ? tn * 64u + w * 16u + i : pair_half + tn * 64u + w * 16u + (i - 16u));
    } else {
        weight_row = tn * 128u + produce_row;
    }
    const unsigned char* weight_group = codes + ((unsigned long long)weight_row * groups) * 24u + half * 12u;
    const unsigned short* weight_scale = weight_scales + (unsigned long long)weight_row * groups;
    unsigned raw[3], raw_scale = 0;
    auto load_weights = [&](unsigned group) {
        const unsigned* p = reinterpret_cast<const unsigned*>(weight_group + (unsigned long long)group * 24u);
        raw[0] = p[0]; raw[1] = p[1]; raw[2] = p[2];
        if (half == 0) raw_scale = weight_scale[group];
    };
    // Expansion of the group fetched one group ago into the next stage, spread over the MMA loop below: word J of
    // the two 16-byte chunks per step, a chunk stored once its four words exist.
    unsigned expanded[4];
    auto expand_step = [&](int j, unsigned stage) {
        switch (j) {
            case 0: expanded[0] = expand_word<0>(raw[0], raw[1], raw[2]); break;
            case 1: expanded[1] = expand_word<1>(raw[0], raw[1], raw[2]); break;
            case 2: expanded[2] = expand_word<2>(raw[0], raw[1], raw[2]); break;
            case 3: expanded[3] = expand_word<3>(raw[0], raw[1], raw[2]); break;
            case 4: expanded[0] = expand_word<4>(raw[0], raw[1], raw[2]); break;
            case 5: expanded[1] = expand_word<5>(raw[0], raw[1], raw[2]); break;
            case 6: expanded[2] = expand_word<6>(raw[0], raw[1], raw[2]); break;
            default: expanded[3] = expand_word<7>(raw[0], raw[1], raw[2]); break;
        }
        if (j == 3 || j == 7) {
            unsigned char* tile = a_tile(stage, 2);
            *reinterpret_cast<uint4*>(tile + swizzle(produce_row, 2u * half + (j == 7 ? 1u : 0u))) =
                    make_uint4(expanded[0], expanded[1], expanded[2], expanded[3]);
        }
        if (j == 7 && half == 0) w_scale_row(stage)[produce_row] = __half2float(__ushort_as_half((unsigned short)raw_scale));
    };
    // Activation scales: lane (g, t < 2) supplies row g + 8 t of each 16-row tile, two bytes (k32 blocks) per group.
    unsigned next_scale[4];
    auto load_scales = [&](unsigned group) {
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            const unsigned r = row0 + wm * 64u + mt * 16u + g + 8u * (t & 1u);
            next_scale[mt] = r < rows ? (unsigned)*reinterpret_cast<const unsigned short*>(a_scales + (unsigned long long)r * scale_stride + group * 2u)
                                      : 0x7f7fu;
        }
    };

    float accumulated[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) accumulated[i][j][e] = 0.0f;

    // Prologue: activations of groups 0 and 1, expanded weights of group 0, raw weights of group 1.
    for (unsigned s = 0; s < kStages - 1; s++) {
        if (s < groups) issue_activations(s, s);
        asm volatile("cp.async.commit_group;");
    }
    load_weights(0);
#pragma unroll
    for (int j = 0; j < 8; j++) expand_step(j, 0);
    if (groups > 1) load_weights(1);
    load_scales(0);
    for (unsigned group = 0; group < groups; group++) {
        const unsigned stage = group % kStages;
        unsigned scale_a[4];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) scale_a[mt] = next_scale[mt];
        asm volatile("cp.async.wait_group %0;" ::"n"(kStages - 2));
        __syncthreads();
        if (group + kStages - 1 < groups) issue_activations(group + kStages - 1, (group + kStages - 1) % kStages);
        asm volatile("cp.async.commit_group;");
        if (group + 1 < groups) load_scales(group + 1);
        // Weights of the next group: expand the registers fetched one group ago, then fetch two ahead.
        const unsigned char* tile_hi = a_tile(stage, 0);
        const unsigned char* tile_lo = a_tile(stage, 1);
        const unsigned char* tile_w = a_tile(stage, 2);
        float acc[4][4][4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
        unsigned b[2][4][2];
#pragma unroll
        for (int kk = 0; kk < 2; kk++)
#pragma unroll
            for (int np = 0; np < 2; np++) {
                unsigned r[4];
                const unsigned n_row = wn * 32u + np * 16u + (lane & 7u) + (lane >> 4) * 8u, chunk = 2u * kk + ((lane >> 3) & 1u);
                ldmatrix_x4(r, tile_w + swizzle(n_row, chunk));
                b[kk][np * 2][0] = r[0]; b[kk][np * 2][1] = r[1]; b[kk][np * 2 + 1][0] = r[2]; b[kk][np * 2 + 1][1] = r[3];
            }
        unsigned a[2][4];
        auto load_a = [&](int i, unsigned (&dst)[4]) {
            const int part = i >> 3, kk = (i >> 2) & 1, mt = i & 3;
            const unsigned a_row = wm * 64u + mt * 16u + (lane & 7u) + ((lane >> 3) & 1u) * 8u, chunk = 2u * kk + (lane >> 4);
            ldmatrix_x4(dst, (part == 0 ? tile_hi : tile_lo) + swizzle(a_row, chunk));
        };
        load_a(0, a[0]);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i + 1 < 16) load_a(i + 1, a[(i + 1) & 1]);
            const int kk = (i >> 2) & 1, mt = i & 3;
            if (i < 8 && group + 1 < groups) expand_step(i, (group + 1) % kStages);
            if (i == 8 && group + 2 < groups) load_weights(group + 2);
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                if (kk == 0) mma_e4m3<0>(acc[mt][nt], a[i & 1], b[kk][nt][0], b[kk][nt][1], scale_a[mt]);
                else mma_e4m3<1>(acc[mt][nt], a[i & 1], b[kk][nt][0], b[kk][nt][1], scale_a[mt]);
            }
        }
        // Promote with the weight group scales.
        const float* scales_now = w_scale_row(stage);
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const float2 w = *reinterpret_cast<const float2*>(scales_now + wn * 32u + nt * 8u + 2u * t);
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                accumulated[mt][nt][0] = fmaf(acc[mt][nt][0], w.x, accumulated[mt][nt][0]);
                accumulated[mt][nt][1] = fmaf(acc[mt][nt][1], w.y, accumulated[mt][nt][1]);
                accumulated[mt][nt][2] = fmaf(acc[mt][nt][2], w.x, accumulated[mt][nt][2]);
                accumulated[mt][nt][3] = fmaf(acc[mt][nt][3], w.y, accumulated[mt][nt][3]);
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const unsigned r = row0 + wm * 64u + mt * 16u + g + 8u * (e >> 1);
            if (r >= rows) continue;
            if (PAIRED) {
                const unsigned width = weight_rows / 2u;
#pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    const unsigned c = tn * 64u + wn * 16u + nt * 8u + 2u * t + (e & 1);
                    const float gate = __bfloat162float(__float2bfloat16_rn(accumulated[mt][nt][e]));
                    const float up = __bfloat162float(__float2bfloat16_rn(accumulated[mt][nt + 2][e]));
                    out[(unsigned long long)r * width + c] = __bfloat16_as_ushort(__float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up));
                }
            } else {
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    const unsigned c = tn * 128u + wn * 32u + nt * 8u + 2u * t + (e & 1);
                    out[(unsigned long long)r * weight_rows + c] = __bfloat16_as_ushort(__float2bfloat16_rn(accumulated[mt][nt][e]));
                }
            }
        }
}
}  // namespace q3mx

// Quantizes BF16 rows into the hi and lo E4M3 planes (rows x width bytes each) and one UE8M0 scale byte per
// 32 values (rows x width / 32). One thread per block of 32. width % 128 == 0.
extern "C" __global__ __launch_bounds__(256) void euhedral_q3mx_quantize(const unsigned short* x, unsigned char* hi,
        unsigned char* lo, unsigned char* scales, unsigned rows, unsigned width) {
    const unsigned long long blocks_per_row = width / 32u, index = (unsigned long long)blockIdx.x * 256u + threadIdx.x;
    if (index >= (unsigned long long)rows * blocks_per_row) return;
    const unsigned long long row = index / blocks_per_row, block = index % blocks_per_row;
    const uint4* source = reinterpret_cast<const uint4*>(x + row * width + block * 32u);
    uint4 packed[4] = {source[0], source[1], source[2], source[3]};
    float v[32];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned w[4] = {packed[i].x, packed[i].y, packed[i].z, packed[i].w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            v[i * 8 + j * 2] = __uint_as_float(w[j] << 16);
            v[i * 8 + j * 2 + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
        }
    }
#pragma unroll
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[i]));
    // Smallest e with amax <= 448 * 2^e: the exponent of amax minus 8, plus one when its mantissa exceeds 1.75.
    const unsigned bits = __float_as_uint(amax);
    int biased = amax == 0.0f ? 127 : (int)(bits >> 23) - 8 + ((bits & 0x7FFFFFu) > 0x600000u ? 1 : 0);
    biased = min(max(biased, 1), 253);
    const float inverse = __uint_as_float((unsigned)(254 - biased) << 23);
    unsigned char out_hi[32], out_lo[32];
#pragma unroll
    for (int i = 0; i < 32; i += 2) {
        const float a = v[i] * inverse, b = v[i + 1] * inverse;
        const unsigned short high = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
        const float2 back = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(high, __NV_E4M3)));
        const unsigned short low = __nv_cvt_float2_to_fp8x2(make_float2(a - back.x, b - back.y), __NV_SATFINITE, __NV_E4M3);
        out_hi[i] = high & 0xFF; out_hi[i + 1] = high >> 8;
        out_lo[i] = low & 0xFF; out_lo[i + 1] = low >> 8;
    }
    uint4* high_destination = reinterpret_cast<uint4*>(hi + row * width + block * 32u);
    uint4* low_destination = reinterpret_cast<uint4*>(lo + row * width + block * 32u);
#pragma unroll
    for (int i = 0; i < 2; i++) {
        high_destination[i] = *reinterpret_cast<const uint4*>(out_hi + 16 * i);
        low_destination[i] = *reinterpret_cast<const uint4*>(out_lo + 16 * i);
    }
    scales[row * blocks_per_row + block] = (unsigned char)biased;
}

// Linear: out[rows][weight_rows] = x * W^T. Grid = ceil(rows / 128) * (weight_rows / 128), 256 threads,
// q3mx::kSharedBytes dynamic shared memory. weight_rows % 128 == 0, k % 128 == 0.
extern "C" __global__ __launch_bounds__(256, 1) void euhedral_q3mx_linear_128x128(const unsigned char* a_hi,
        const unsigned char* a_lo, const unsigned char* a_scales, const unsigned char* codes, unsigned short* out,
        unsigned rows, unsigned weight_rows, unsigned k, unsigned long long scale_offset) {
    q3mx::gemm<0>(a_hi, a_lo, a_scales, codes, out, rows, weight_rows, k, scale_offset);
}

// Gate/up SwiGLU: weight_rows = gate rows followed by up rows; out[rows][weight_rows / 2]. 64 outputs per tile:
// grid = ceil(rows / 128) * (weight_rows / 128).
extern "C" __global__ __launch_bounds__(256, 1) void euhedral_q3mx_gate_up_swiglu_128x64(const unsigned char* a_hi,
        const unsigned char* a_lo, const unsigned char* a_scales, const unsigned char* codes, unsigned short* out,
        unsigned rows, unsigned weight_rows, unsigned k, unsigned long long scale_offset) {
    q3mx::gemm<1>(a_hi, a_lo, a_scales, codes, out, rows, weight_rows, k, scale_offset);
}
