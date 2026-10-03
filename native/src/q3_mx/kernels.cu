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
// Q4 and Q5 (native/src/q45 layout: nibble plane, fifth-bit plane, scale plane) run the same engine; their codes (-8..7, -16..15)
// are exact in E4M3 too. Q3 expands by byte permute; Q4/Q5 look up one E4M3 pair per nibble pair (and fifth-bit pair).
//
// Flow: euhedral_q3mx_quantize (BF16 rows -> hi, lo, scale planes), then a 128 x 128 tile GEMM: 8 consumer warps of
// 64 x 32 and 4 producer warps. A tiles arrive by cp.async; the compact weight groups are expanded to E4M3 bytes
// by the producers, up to three groups ahead of the MMAs.
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

// Word J (four codes) of a 32-code half group from its raw words: Q3 three words of a 3-bit stream; Q4 four words of
// nibbles (code K at nibble K); Q5 the same plus the fifth bits of the half (bit K of raw[4]).
template<int BITS, int J>
__device__ __forceinline__ unsigned expand_any(const unsigned (&raw)[5], const unsigned short* pair_table) {
    if (BITS == 3) return expand_word<J>(raw[0], raw[1], raw[2]);
    const unsigned word = raw[J >> 1], shift = 16u * (J & 1);
    unsigned first = (word >> shift) & 0xFFu, second = (word >> (shift + 8u)) & 0xFFu;
    if (BITS == 5) {
        const unsigned fifth = raw[4] >> (4 * J);
        first |= (fifth & 3u) << 8;
        second |= ((fifth >> 2) & 3u) << 8;
    }
    return (unsigned)pair_table[first] | ((unsigned)pair_table[second] << 16);
}

// One tile: PAIRED = 0 writes out[row][col] for 128 weight rows; PAIRED = 1 treats the weight rows as gate rows
// followed by up rows, pairs 64 of each per tile and writes silu(gate) * up. `weight_rows` is the number of weight rows.
//
// Warp-specialized: warps 0-7 only issue the MMAs and the promotion; warps 8-11 stream the next groups' activations
// (cp.async) and expand the weights into a ring of three stages handed over by mbarriers. The consumers run with the
// registers the producers give up (setmaxnreg), so an SM holds one 384-thread CTA.
constexpr unsigned kConsumerThreads = 256, kProducerThreads = 128, kConsumerRegisters = 216, kProducerRegisters = 72;

__device__ __forceinline__ void mbarrier_init(unsigned long long* mb, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(shared_address(mb)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive(unsigned long long* mb) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(shared_address(mb)) : "memory");
}
// Arrives when this thread's earlier cp.async copies have landed; the barrier's count includes these arrivals.
__device__ __forceinline__ void mbarrier_arrive_after_copies(unsigned long long* mb) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(shared_address(mb)) : "memory");
}
__device__ __forceinline__ void mbarrier_wait(unsigned long long* mb, unsigned parity) {
    asm volatile("{\n .reg .pred p;\n WAIT_LOOP:\n mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n"
                 " @!p bra WAIT_LOOP;\n}" ::"r"(shared_address(mb)), "r"(parity) : "memory");
}

template<int BITS, int PAIRED>
__device__ __forceinline__ void gemm(const unsigned char* a_hi, const unsigned char* a_lo, const unsigned char* a_scales,
        const unsigned char* codes, unsigned short* out, unsigned rows, unsigned weight_rows, unsigned k,
        unsigned long long high_offset, unsigned long long scale_offset, float* partials) {
    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ unsigned short pair_table[BITS == 5 ? 1024 : BITS == 4 ? 256 : 1];
    __shared__ unsigned long long barriers[2 * kStages];  // full[0..2], empty[0..2]
    const unsigned tid = threadIdx.x;
    if (BITS != 3) {
        for (unsigned i = tid; i < (BITS == 5 ? 1024u : 256u); i += kConsumerThreads + kProducerThreads) {
            int c[2];
            for (int s = 0; s < 2; s++) {
                const unsigned n = (i >> (4 * s)) & 15u;
                c[s] = BITS == 5 ? (int)n - (int)(((i >> (8 + s)) & 1u) << 4) : (int)n - (int)((n >> 3) << 4);
            }
            pair_table[i] = __nv_cvt_float2_to_fp8x2(make_float2((float)c[0], (float)c[1]), __NV_SATFINITE, __NV_E4M3);
        }
    }
    if (tid == 0)
        for (int s = 0; s < kStages; s++) {
            mbarrier_init(&barriers[s], kProducerThreads + kProducerThreads);  // 128 copy arrivals + 128 store arrivals
            mbarrier_init(&barriers[kStages + s], kConsumerThreads);
        }
    __syncthreads();
    const unsigned tiles_m = (rows + 127u) / 128u;
    const unsigned tm = blockIdx.x % tiles_m, tn = blockIdx.x / tiles_m;
    const unsigned row0 = tm * 128u;
    const unsigned groups = k / 64u, scale_stride = k / 32u, group_count = groups / gridDim.y, base_group = blockIdx.y * group_count;
    const unsigned short* weight_scales = reinterpret_cast<const unsigned short*>(codes + scale_offset);
    auto a_tile = [&](unsigned stage, unsigned plane) { return smem + stage * kStageBytes + plane * kTile; };
    auto w_scale_row = [&](unsigned stage) { return reinterpret_cast<float*>(smem + stage * kStageBytes + 3 * kTile); };
    unsigned long long* full = barriers;
    unsigned long long* empty = barriers + kStages;

    if (tid >= kConsumerThreads) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kProducerRegisters));
        const unsigned pt = tid - kConsumerThreads;
        // Activations: 1024 sixteen-byte chunks per group, eight per thread: rows (pt >> 2) + 32 i of the hi (i < 4) and lo planes.
        const unsigned a_row = pt >> 2, a_chunk = pt & 3u, a_offset = swizzle(a_row, a_chunk);
        const unsigned char* a_source[8];
        bool a_valid[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned r = row0 + a_row + 32u * (i & 3);
            a_valid[i] = r < rows;
            a_source[i] = (i < 4 ? a_hi : a_lo) + (unsigned long long)(a_valid[i] ? r : 0u) * k + a_chunk * 16u;
        }
        // Weights: two (row, half) units per thread, 32 codes each.
        constexpr unsigned kCodeBytes = BITS == 3 ? 24u : 32u, kHalfBytes = kCodeBytes / 2u;
        const unsigned half = pt & 1u;
        unsigned prow[2];
        const unsigned char* weight_group[2];
        const unsigned char* high_group[2];
        const unsigned short* weight_scale[2];
#pragma unroll
        for (int u = 0; u < 2; u++) {
            prow[u] = (pt >> 1) + 64u * u;
            unsigned weight_row;
            if (PAIRED) {
                const unsigned pair_half = weight_rows / 2u, i = prow[u] & 31u, w = prow[u] >> 5;
                weight_row = (i < 16u ? tn * 64u + w * 16u + i : pair_half + tn * 64u + w * 16u + (i - 16u));
            } else {
                weight_row = tn * 128u + prow[u];
            }
            const unsigned long long first_group = (unsigned long long)weight_row * groups;
            weight_group[u] = codes + first_group * kCodeBytes + half * kHalfBytes;
            high_group[u] = codes + high_offset + first_group * 8u + half * 4u;
            weight_scale[u] = weight_scales + first_group;
        }
        unsigned raw[2][5], raw_scale[2] = {0, 0};
        auto load_weights = [&](unsigned group) {
#pragma unroll
            for (int u = 0; u < 2; u++) {
                if (BITS == 3) {
                    const unsigned* p = reinterpret_cast<const unsigned*>(weight_group[u] + (unsigned long long)(base_group + group) * 24u);
                    raw[u][0] = p[0]; raw[u][1] = p[1]; raw[u][2] = p[2];
                } else {
                    const uint4 p = *reinterpret_cast<const uint4*>(weight_group[u] + (unsigned long long)(base_group + group) * 32u);
                    raw[u][0] = p.x; raw[u][1] = p.y; raw[u][2] = p.z; raw[u][3] = p.w;
                    if (BITS == 5) raw[u][4] = *reinterpret_cast<const unsigned*>(high_group[u] + (unsigned long long)(base_group + group) * 8u);
                }
                if (half == 0) raw_scale[u] = weight_scale[u][base_group + group];
            }
        };
        load_weights(0);
        for (unsigned group = 0; group < group_count; group++) {
            const unsigned stage = group % kStages;
            if (group >= kStages) mbarrier_wait(&empty[stage], ((group / kStages) - 1) & 1u);
#pragma unroll
            for (int i = 0; i < 8; i++)
                copy16(a_tile(stage, i >> 2) + (i & 3) * 2048u + a_offset, a_source[i] + (base_group + group) * 64u, a_valid[i]);
            mbarrier_arrive_after_copies(&full[stage]);
            unsigned char* tile = a_tile(stage, 2);
#pragma unroll
            for (int u = 0; u < 2; u++) {
                unsigned words[8];
                words[0] = expand_any<BITS, 0>(raw[u], pair_table); words[1] = expand_any<BITS, 1>(raw[u], pair_table);
                words[2] = expand_any<BITS, 2>(raw[u], pair_table); words[3] = expand_any<BITS, 3>(raw[u], pair_table);
                words[4] = expand_any<BITS, 4>(raw[u], pair_table); words[5] = expand_any<BITS, 5>(raw[u], pair_table);
                words[6] = expand_any<BITS, 6>(raw[u], pair_table); words[7] = expand_any<BITS, 7>(raw[u], pair_table);
                *reinterpret_cast<uint4*>(tile + swizzle(prow[u], 2u * half)) = make_uint4(words[0], words[1], words[2], words[3]);
                *reinterpret_cast<uint4*>(tile + swizzle(prow[u], 2u * half + 1u)) = make_uint4(words[4], words[5], words[6], words[7]);
                if (half == 0) w_scale_row(stage)[prow[u]] = __half2float(__ushort_as_half((unsigned short)raw_scale[u]));
            }
            mbarrier_arrive(&full[stage]);
            if (group + 1 < group_count) load_weights(group + 1);
        }
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(kConsumerRegisters));
    const unsigned lane = tid & 31u, warp = tid >> 5, g = lane >> 2, t = lane & 3u;
    const unsigned wm = warp >> 2, wn = warp & 3u;
    // Activation scales: lane (g, t < 2) supplies row g + 8 t of each 16-row tile, two bytes (k32 blocks) per group.
    unsigned next_scale[4];
    auto load_scales = [&](unsigned group) {
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            const unsigned r = row0 + wm * 64u + mt * 16u + g + 8u * (t & 1u);
            next_scale[mt] = r < rows ? (unsigned)*reinterpret_cast<const unsigned short*>(a_scales + (unsigned long long)r * scale_stride + (base_group + group) * 2u)
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
    load_scales(0);
    for (unsigned group = 0; group < group_count; group++) {
        const unsigned stage = group % kStages;
        unsigned scale_a[4];
#pragma unroll
        for (int mt = 0; mt < 4; mt++) scale_a[mt] = next_scale[mt];
        if (group + 1 < group_count) load_scales(group + 1);
        mbarrier_wait(&full[stage], (group / kStages) & 1u);
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
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                if (kk == 0) mma_e4m3<0>(acc[mt][nt], a[i & 1], b[kk][nt][0], b[kk][nt][1], scale_a[mt]);
                else mma_e4m3<1>(acc[mt][nt], a[i & 1], b[kk][nt][0], b[kk][nt][1], scale_a[mt]);
            }
        }
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
        mbarrier_arrive(&empty[stage]);
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
                    if (partials != nullptr)
                        partials[((unsigned long long)blockIdx.y * rows + r) * weight_rows + c] = accumulated[mt][nt][e];
                    else
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

// Linear: out[rows][weight_rows] = x * W^T. Grid = (ceil(rows / 128) * (weight_rows / 128), splits), 384 threads,
// q3mx::kSharedBytes dynamic shared memory. weight_rows % 128 == 0, k % 128 == 0. Q3: the codes then the scales at
// scale_offset. With splits > 1 (gridDim.y, dividing k / 64) each split writes FP32 partials[split][rows][weight_rows] and
// euhedral_q3mx_reduce sums them in split order. Q4/Q5: the nibble plane, the fifth-bit plane at high_offset (Q5), the scales at scale_offset.
#define EUHEDRAL_Q3MX_KERNEL(NAME, BITS, PAIRED) \
extern "C" __global__ __launch_bounds__(384, 1) void NAME(const unsigned char* a_hi, const unsigned char* a_lo, \
        const unsigned char* a_scales, const unsigned char* codes, unsigned short* out, unsigned rows, \
        unsigned weight_rows, unsigned k, unsigned long long high_offset, unsigned long long scale_offset, float* partials) { \
    q3mx::gemm<BITS, PAIRED>(a_hi, a_lo, a_scales, codes, out, rows, weight_rows, k, high_offset, scale_offset, partials); \
}
EUHEDRAL_Q3MX_KERNEL(euhedral_q3mx_linear_128x128, 3, 0)
// Gate/up SwiGLU: weight_rows = gate rows followed by up rows; out[rows][weight_rows / 2]. 64 outputs per tile.
EUHEDRAL_Q3MX_KERNEL(euhedral_q3mx_gate_up_swiglu_128x64, 3, 1)
EUHEDRAL_Q3MX_KERNEL(euhedral_q4mx_linear_128x128, 4, 0)
EUHEDRAL_Q3MX_KERNEL(euhedral_q5mx_linear_128x128, 5, 0)
#undef EUHEDRAL_Q3MX_KERNEL

// Sums `splits` FP32 partial planes of `count` values in split order and rounds to BF16. count % 4 == 0.
extern "C" __global__ __launch_bounds__(256) void euhedral_q3mx_reduce(const float* partials, unsigned short* out,
        unsigned count, unsigned splits) {
    const unsigned i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= count) return;
    float4 sum = *reinterpret_cast<const float4*>(partials + i);
    for (unsigned s = 1; s < splits; s++) {
        const float4 part = *reinterpret_cast<const float4*>(partials + (unsigned long long)s * count + i);
        sum.x += part.x; sum.y += part.y; sum.z += part.z; sum.w += part.w;
    }
    const unsigned low = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(sum.x)) | ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(sum.y)) << 16);
    const unsigned high = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(sum.z)) | ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(sum.w)) << 16);
    *reinterpret_cast<uint2*>(out + i) = make_uint2(low, high);
}
