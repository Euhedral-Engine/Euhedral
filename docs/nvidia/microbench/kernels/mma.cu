// Tensor-core throughput: every warp issues NACC independent mma.sync chains on
// register operands. Compile one variant per program with -DV_<name>, so an
// unsupported kind fails alone. Operands are hashed per lane (realistic toggling,
// no NaN/Inf encodings for the types used).
typedef unsigned long long u64;
#ifndef NACC
#define NACC 4
#endif

__device__ __forceinline__ u64 gtimer() {
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned hsh(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// Constrain random words to finite, moderate values per element type.
#define F16X2(x) (((x) & 0x83ff83ffu) | 0x38003800u)   /* |v| in [0.5, 1) */
#define E4M3X4(x) (((x) & 0x87878787u) | 0x30303030u)  /* exp field 6-7 */
#define E3M2X4(x) ((x) & 0x3f3f3f3fu)                  /* 6-bit codes in 8-bit containers */
#define E2M1X4(x) ((x) & 0x0f0f0f0fu)                  /* 4-bit codes in 8-bit containers */
#define E2M1X8(x) (x)                                  /* packed nibbles, all finite */
#define S8X4(x) (x)

#define PROLOGUE                                                    \
    if (clk && blockIdx.x == 0 && threadIdx.x == 0) {               \
        clk[0] = clock64();                                         \
        clk[2] = gtimer();                                          \
    }                                                               \
    unsigned seed = hsh(blockIdx.x * 1024 + threadIdx.x + 1);       \
    unsigned r0 = hsh(seed), r1 = hsh(r0), r2 = hsh(r1), r3 = hsh(r2), \
             r4 = hsh(r3), r5 = hsh(r4), r6 = hsh(r5), r7 = hsh(r6);
#define EPILOGUE(SUMEXPR)                                           \
    out[blockIdx.x * blockDim.x + threadIdx.x] = (SUMEXPR);         \
    if (clk && blockIdx.x == 0 && threadIdx.x == 0) {               \
        clk[1] = clock64();                                         \
        clk[3] = gtimer();                                          \
    }

// ---- D = 4 x f32, A = 4 regs, B = 2 regs --------------------------------------
#define KERNEL_F32(NAME, INSTR, AF, BF)                                                    \
    extern "C" __global__ void NAME(int iters, float *out, u64 *clk) {                     \
        PROLOGUE                                                                           \
        unsigned a0 = AF(r0), a1 = AF(r1), a2 = AF(r2), a3 = AF(r3), b0 = BF(r4), b1 = BF(r5); \
        float d[NACC][4];                                                                  \
        _Pragma("unroll") for (int j = 0; j < NACC; j++) d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0.f; \
        for (int i = 0; i < iters; i++) {                                                  \
            _Pragma("unroll") for (int j = 0; j < NACC; j++) asm volatile(                 \
                INSTR " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"             \
                : "+f"(d[j][0]), "+f"(d[j][1]), "+f"(d[j][2]), "+f"(d[j][3])               \
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));                   \
        }                                                                                  \
        float s = 0;                                                                       \
        _Pragma("unroll") for (int j = 0; j < NACC; j++) s += d[j][0] + d[j][1] + d[j][2] + d[j][3]; \
        EPILOGUE(s)                                                                        \
    }

// ---- D = 2 x f16x2 ------------------------------------------------------------
#define KERNEL_F16ACC(NAME, INSTR, AF, BF)                                                 \
    extern "C" __global__ void NAME(int iters, float *out, u64 *clk) {                     \
        PROLOGUE                                                                           \
        unsigned a0 = AF(r0), a1 = AF(r1), a2 = AF(r2), a3 = AF(r3), b0 = BF(r4), b1 = BF(r5); \
        unsigned d[NACC][2];                                                               \
        _Pragma("unroll") for (int j = 0; j < NACC; j++) d[j][0] = d[j][1] = 0u;           \
        for (int i = 0; i < iters; i++) {                                                  \
            _Pragma("unroll") for (int j = 0; j < NACC; j++) asm volatile(                 \
                INSTR " {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"                         \
                : "+r"(d[j][0]), "+r"(d[j][1])                                             \
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));                   \
        }                                                                                  \
        unsigned s = 0;                                                                    \
        _Pragma("unroll") for (int j = 0; j < NACC; j++) s ^= d[j][0] ^ d[j][1];           \
        EPILOGUE(__uint_as_float(s))                                                       \
    }

// ---- D = 4 x s32 ----------------------------------------------------------------
#define KERNEL_S32(NAME, INSTR, AF, BF)                                                    \
    extern "C" __global__ void NAME(int iters, float *out, u64 *clk) {                     \
        PROLOGUE                                                                           \
        unsigned a0 = AF(r0), a1 = AF(r1), a2 = AF(r2), a3 = AF(r3), b0 = BF(r4), b1 = BF(r5); \
        int d[NACC][4];                                                                    \
        _Pragma("unroll") for (int j = 0; j < NACC; j++) d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0; \
        for (int i = 0; i < iters; i++) {                                                  \
            _Pragma("unroll") for (int j = 0; j < NACC; j++) asm volatile(                 \
                INSTR " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"             \
                : "+r"(d[j][0]), "+r"(d[j][1]), "+r"(d[j][2]), "+r"(d[j][3])               \
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));                   \
        }                                                                                  \
        int s = 0;                                                                         \
        _Pragma("unroll") for (int j = 0; j < NACC; j++) s += d[j][0] ^ d[j][1] ^ d[j][2] ^ d[j][3]; \
        EPILOGUE((float)s)                                                                 \
    }

// ---- block-scaled: D = 4 x f32 plus one scale register per operand --------------
#define KERNEL_BS(NAME, INSTR, AF, BF, SCALE)                                              \
    extern "C" __global__ void NAME(int iters, float *out, u64 *clk) {                     \
        PROLOGUE                                                                           \
        unsigned a0 = AF(r0), a1 = AF(r1), a2 = AF(r2), a3 = AF(r3), b0 = BF(r4), b1 = BF(r5); \
        unsigned sa = SCALE, sb = SCALE;                                                   \
        float d[NACC][4];                                                                  \
        _Pragma("unroll") for (int j = 0; j < NACC; j++) d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0.f; \
        for (int i = 0; i < iters; i++) {                                                  \
            _Pragma("unroll") for (int j = 0; j < NACC; j++) asm volatile(                 \
                INSTR " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};" \
                : "+f"(d[j][0]), "+f"(d[j][1]), "+f"(d[j][2]), "+f"(d[j][3])               \
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb)); \
        }                                                                                  \
        float s = 0;                                                                       \
        _Pragma("unroll") for (int j = 0; j < NACC; j++) s += d[j][0] + d[j][1] + d[j][2] + d[j][3]; \
        EPILOGUE(s)                                                                        \
    }

// ---- 2:4 sparse: A = 4 regs (compressed), B = 4 regs, metadata + selector ----------
#define KERNEL_SP(NAME, INSTR, AF, BF)                                                     \
    extern "C" __global__ void NAME(int iters, float *out, u64 *clk) {                     \
        PROLOGUE                                                                           \
        unsigned a0 = AF(r0), a1 = AF(r1), a2 = AF(r2), a3 = AF(r3);                       \
        unsigned b0 = BF(r4), b1 = BF(r5), b2 = BF(r6), b3 = BF(r7);                       \
        unsigned meta = 0x44444444u; /* indices {0,1} in every group of four */            \
        float d[NACC][4];                                                                  \
        _Pragma("unroll") for (int j = 0; j < NACC; j++) d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0.f; \
        for (int i = 0; i < iters; i++) {                                                  \
            _Pragma("unroll") for (int j = 0; j < NACC; j++) asm volatile(                 \
                INSTR " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;" \
                : "+f"(d[j][0]), "+f"(d[j][1]), "+f"(d[j][2]), "+f"(d[j][3])               \
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3),  \
                  "r"(meta));                                                              \
        }                                                                                  \
        float s = 0;                                                                       \
        _Pragma("unroll") for (int j = 0; j < NACC; j++) s += d[j][0] + d[j][1] + d[j][2] + d[j][3]; \
        EPILOGUE(s)                                                                        \
    }

#define KERNEL_SPBS(NAME, INSTR, AF, BF, SCALE)                                            \
    extern "C" __global__ void NAME(int iters, float *out, u64 *clk) {                     \
        PROLOGUE                                                                           \
        unsigned a0 = AF(r0), a1 = AF(r1), a2 = AF(r2), a3 = AF(r3);                       \
        unsigned b0 = BF(r4), b1 = BF(r5), b2 = BF(r6), b3 = BF(r7);                       \
        unsigned meta = 0x44444444u, sa = SCALE, sb = SCALE;                               \
        float d[NACC][4];                                                                  \
        _Pragma("unroll") for (int j = 0; j < NACC; j++) d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0.f; \
        for (int i = 0; i < iters; i++) {                                                  \
            _Pragma("unroll") for (int j = 0; j < NACC; j++) asm volatile(                 \
                INSTR " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0, %13, {0, 0}, %14, {0, 0};" \
                : "+f"(d[j][0]), "+f"(d[j][1]), "+f"(d[j][2]), "+f"(d[j][3])               \
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3),  \
                  "r"(meta), "r"(sa), "r"(sb));                                            \
        }                                                                                  \
        float s = 0;                                                                       \
        _Pragma("unroll") for (int j = 0; j < NACC; j++) s += d[j][0] + d[j][1] + d[j][2] + d[j][3]; \
        EPILOGUE(s)                                                                        \
    }

#define UE8M0_ONE 0x7f7f7f7fu
#define UE4M3_ONE 0x38383838u

#ifdef V_f16_f32
KERNEL_F32(k, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", F16X2, F16X2)
#endif
#ifdef V_f16_f16
KERNEL_F16ACC(k, "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16", F16X2, F16X2)
#endif
#ifdef V_bf16_f32
KERNEL_F32(k, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", F16X2, F16X2)
#endif
#ifdef V_tf32_f32
KERNEL_F32(k, "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", F16X2, F16X2)
#endif
#ifdef V_e4m3_f32
KERNEL_F32(k, "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", E4M3X4, E4M3X4)
#endif
#ifdef V_e4m3_f16
KERNEL_F16ACC(k, "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16", E4M3X4, E4M3X4)
#endif
#ifdef V_f8f6f4_e4m3_f32
KERNEL_F32(k, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32", E4M3X4, E4M3X4)
#endif
#ifdef V_f8f6f4_e4m3_f16
KERNEL_F16ACC(k, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16", E4M3X4, E4M3X4)
#endif
#ifdef V_f8f6f4_e3m2_f32
KERNEL_F32(k, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e3m2.f32", E3M2X4, E3M2X4)
#endif
#ifdef V_f8f6f4_e2m1_f32
KERNEL_F32(k, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32", E2M1X4, E2M1X4)
#endif
#ifdef V_f8f6f4_e2m1xe4m3_f32
KERNEL_F32(k, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e4m3.f32", E2M1X4, E4M3X4)
#endif
#ifdef V_mxf8f6f4_e4m3
KERNEL_BS(k, "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0", E4M3X4, E4M3X4, UE8M0_ONE)
#endif
#ifdef V_mxf8f6f4_e2m1
KERNEL_BS(k, "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0", E2M1X4, E2M1X4, UE8M0_ONE)
#endif
#ifdef V_mxf4_2x
KERNEL_BS(k, "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", E2M1X8, E2M1X8, UE8M0_ONE)
#endif
#ifdef V_mxf4nvf4_4x
KERNEL_BS(k, "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3", E2M1X8, E2M1X8, UE4M3_ONE)
#endif
#ifdef V_mxf4nvf4_2x_ue8m0
KERNEL_BS(k, "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", E2M1X8, E2M1X8, UE8M0_ONE)
#endif
#ifdef V_s8_s32
KERNEL_S32(k, "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", S8X4, S8X4)
#endif
#ifdef V_s4_s32
KERNEL_S32(k, "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32", S8X4, S8X4)
#endif
#ifdef V_b1_and
KERNEL_S32(k, "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc", S8X4, S8X4)
#endif
#ifdef V_sp_f16_f32
KERNEL_SP(k, "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32", F16X2, F16X2)
#endif
#ifdef V_sp_e4m3_f32
KERNEL_SP(k, "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32", E4M3X4, E4M3X4)
#endif
#ifdef V_sp_f8f6f4_e2m1
KERNEL_SP(k, "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32", E2M1X4, E2M1X4)
#endif
#ifdef V_sp_mxf4nvf4_4x
KERNEL_SPBS(k, "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3", E2M1X8, E2M1X8, UE4M3_ONE)
#endif
#ifdef V_sp_mxf4_2x
KERNEL_SPBS(k, "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0", E2M1X8, E2M1X8, UE8M0_ONE)
#endif
