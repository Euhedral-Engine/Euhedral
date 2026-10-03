// CUDA-core / SFU / conversion throughput. One variant per program (-DV_<name>).
// Each thread runs CH independent dependency chains of exactly one PTX
// instruction (packed variants count elements separately on the host).
typedef unsigned long long u64;
#define CH 8

__device__ __forceinline__ u64 gtimer() {
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

#if defined(V_ffma)
#define T float
#define INIT(j) (1.0f + (j) * 1e-3f + (float)threadIdx.x * 1e-6f)
#define OP(x) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(0.999f), "f"(1e-3f))
#define FOLD(x) __float_as_uint(x)
#elif defined(V_ffma2)
#define T unsigned long long
#define INIT(j) (0x3f8000003f800000ull + (j) + threadIdx.x)
#define OP(x) asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(x) : "l"(0x3f7fbe773f7fbe77ull), "l"(0x3a83126f3a83126full))
#define FOLD(x) ((unsigned)(x) ^ (unsigned)((x) >> 32))
#elif defined(V_fadd2)
#define T unsigned long long
#define INIT(j) (0x3f8000003f800000ull + (j) + threadIdx.x)
#define OP(x) asm volatile("add.rn.f32x2 %0, %0, %1;" : "+l"(x) : "l"(0x3a83126f3a83126full))
#define FOLD(x) ((unsigned)(x) ^ (unsigned)((x) >> 32))
#elif defined(V_hfma2)
#define T unsigned
#define INIT(j) (0x3c003c00u + (j) + threadIdx.x)
#define OP(x) asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(0x3bff3bffu), "r"(0x14001400u))
#define FOLD(x) (x)
#elif defined(V_bfma2)
#define T unsigned
#define INIT(j) (0x3f803f80u + (j) + threadIdx.x)
#define OP(x) asm volatile("fma.rn.bf16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(0x3f7f3f7fu), "r"(0x3a833a83u))
#define FOLD(x) (x)
#elif defined(V_fmax)
#define T float
#define INIT(j) ((float)(j) + threadIdx.x)
#define OP(x) asm volatile("max.f32 %0, %0, %1;" : "+f"(x) : "f"(3.0f))
#define FOLD(x) __float_as_uint(x)
#elif defined(V_imad)
#define T unsigned
#define INIT(j) (threadIdx.x * 7u + (j))
#define OP(x) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(0x9e3779b1u), "r"(12345u))
#define FOLD(x) (x)
#elif defined(V_iadd)
// two adds per OP forming a Fibonacci pair (ptxas cannot fold repeated constant adds)
#define T unsigned long long
#define INIT(j) (((unsigned long long)(threadIdx.x + 3u) << 32) | (unsigned)(j + 1))
#define OP(x) asm volatile("{ .reg .u32 lo, hi; mov.b64 {lo, hi}, %0; add.u32 lo, lo, hi; add.u32 hi, hi, lo; mov.b64 %0, {lo, hi}; }" : "+l"(x))
#define FOLD(x) ((unsigned)(x) ^ (unsigned)((x) >> 32))
#elif defined(V_mix_ffma_lop3)
// one FFMA and one LOP3 per OP on independent chains: do the FP32 and integer pipes overlap?
#define T unsigned long long
#define INIT(j) (((unsigned long long)__float_as_uint(1.0f + (j) * 1e-3f) << 32) | (threadIdx.x * 7u + (j)))
#define OP(x) asm volatile("{ .reg .u32 lo; .reg .f32 hi; .reg .b32 hb; mov.b64 {lo, hb}, %0; mov.b32 hi, hb;"   \
                           " fma.rn.f32 hi, hi, 0f3F7FBE77, 0f3A83126F; lop3.b32 lo, lo, 0x9e3779b1, 0x85ebca6b, 0x96;" \
                           " mov.b32 hb, hi; mov.b64 %0, {lo, hb}; }" : "+l"(x))
#define FOLD(x) ((unsigned)(x) ^ (unsigned)((x) >> 32))
#elif defined(V_mix_ffma_imad)
#define T unsigned long long
#define INIT(j) (((unsigned long long)__float_as_uint(1.0f + (j) * 1e-3f) << 32) | (threadIdx.x * 7u + (j)))
#define OP(x) asm volatile("{ .reg .u32 lo; .reg .f32 hi; .reg .b32 hb; mov.b64 {lo, hb}, %0; mov.b32 hi, hb;"   \
                           " fma.rn.f32 hi, hi, 0f3F7FBE77, 0f3A83126F; mad.lo.u32 lo, lo, 0x9e3779b1, 12345;" \
                           " mov.b32 hb, hi; mov.b64 %0, {lo, hb}; }" : "+l"(x))
#define FOLD(x) ((unsigned)(x) ^ (unsigned)((x) >> 32))
#elif defined(V_mix_ffma_ex2)
#define T unsigned long long
#define INIT(j) (((unsigned long long)__float_as_uint(1.0f + (j) * 1e-3f) << 32) | __float_as_uint(0.5f + (j) * 1e-3f))
#define OP(x) asm volatile("{ .reg .f32 lo, hi; .reg .b32 lb, hb; mov.b64 {lb, hb}, %0; mov.b32 hi, hb; mov.b32 lo, lb;" \
                           " fma.rn.f32 hi, hi, 0f3F7FBE77, 0f3A83126F; fma.rn.f32 hi, hi, 0f3F7FBE77, 0f3A83126F;"      \
                           " fma.rn.f32 hi, hi, 0f3F7FBE77, 0f3A83126F; fma.rn.f32 hi, hi, 0f3F7FBE77, 0f3A83126F;"      \
                           " ex2.approx.ftz.f32 lo, lo;"                                                                   \
                           " mov.b32 hb, hi; mov.b32 lb, lo; mov.b64 %0, {lb, hb}; }" : "+l"(x))
#define FOLD(x) ((unsigned)(x) ^ (unsigned)((x) >> 32))
#elif defined(V_lop3)
#define T unsigned
#define INIT(j) (threadIdx.x * 7u + (j))
#define OP(x) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(x) : "r"(0x9e3779b1u), "r"(0x85ebca6bu))
#define FOLD(x) (x)
#elif defined(V_shf)
#define T unsigned
#define INIT(j) (threadIdx.x * 7u + (j))
#define OP(x) asm volatile("shf.l.wrap.b32 %0, %0, %1, 5;" : "+r"(x) : "r"(0x9e3779b1u))
#define FOLD(x) (x)
#elif defined(V_prmt)
#define T unsigned
#define INIT(j) (threadIdx.x * 7u + (j))
#define OP(x) asm volatile("prmt.b32 %0, %0, %1, 0x5140;" : "+r"(x) : "r"(0x9e3779b1u))
#define FOLD(x) (x)
#elif defined(V_dp4a)
#define T unsigned
#define INIT(j) (threadIdx.x * 7u + (j))
#define OP(x) asm volatile("dp4a.u32.u32 %0, %1, %2, %0;" : "+r"(x) : "r"(0x01020304u), "r"(0x05060708u))
#define FOLD(x) (x)
#elif defined(V_popc)
#define T unsigned
#define INIT(j) (threadIdx.x * 7u + (j) + 0xffff)
#define OP(x) asm volatile("{ .reg .u32 t; popc.b32 t, %0; xor.b32 %0, %0, t; }" : "+r"(x))
#define FOLD(x) (x)
#elif defined(V_i2f)
#define T unsigned
#define INIT(j) (threadIdx.x * 7u + (j))
#define OP(x) asm volatile("{ .reg .f32 t; cvt.rn.f32.u32 t, %0; mov.b32 %0, t; }" : "+r"(x))
#define FOLD(x) (x)
#elif defined(V_f2i)
#define T unsigned
#define INIT(j) (0x4b000000u + threadIdx.x + (j))
#define OP(x) asm volatile("{ .reg .f32 t; mov.b32 t, %0; cvt.rzi.u32.f32 %0, t; }" : "+r"(x))
#define FOLD(x) (x)
#elif defined(V_f2f16x2)
#define T unsigned
#define INIT(j) (0x3f800000u + threadIdx.x + (j))
#define OP(x) asm volatile("{ .reg .f32 t; mov.b32 t, %0; cvt.rn.f16x2.f32 %0, t, t; }" : "+r"(x))
#define FOLD(x) (x)
#elif defined(V_f2e4m3x2)
#define T unsigned
#define INIT(j) (0x3f800000u + threadIdx.x + (j))
#define OP(x) asm volatile("{ .reg .f32 t; .reg .b16 h; mov.b32 t, %0; cvt.rn.satfinite.e4m3x2.f32 h, t, t; cvt.u32.u16 %0, h; }" : "+r"(x))
#define FOLD(x) (x)
#elif defined(V_e4m3x2_f16x2)
#define T unsigned
#define INIT(j) (0x38383838u + threadIdx.x + (j))
#define OP(x) asm volatile("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %0; cvt.rn.f16x2.e4m3x2 %0, lo; }" : "+r"(x))
#define FOLD(x) (x)
#elif defined(V_e2m1x2_f16x2)
// 4 cvt per op: expands 8 FP4 codes to 8 FP16 values.
#define T unsigned
#define INIT(j) (0x12345678u + threadIdx.x + (j))
#define OP(x) asm volatile("{ .reg .b8 b0, b1, b2, b3; .reg .b32 h0, h1, h2, h3; mov.b32 {b0, b1, b2, b3}, %0;"   \
                           " cvt.rn.f16x2.e2m1x2 h0, b0; cvt.rn.f16x2.e2m1x2 h1, b1;"                       \
                           " cvt.rn.f16x2.e2m1x2 h2, b2; cvt.rn.f16x2.e2m1x2 h3, b3;"                       \
                           " xor.b32 h0, h0, h1; xor.b32 h2, h2, h3; xor.b32 %0, h0, h2; }" : "+r"(x))
#define FOLD(x) (x)
#elif defined(V_f2e2m1x2)
#define T unsigned
#define INIT(j) (0x3f800000u + threadIdx.x + (j))
#define OP(x) asm volatile("{ .reg .f32 t; .reg .b8 b; mov.b32 t, %0; cvt.rn.satfinite.e2m1x2.f32 b, t, t; cvt.u32.u8 %0, b; }" : "+r"(x))
#define FOLD(x) (x)
#elif defined(V_ex2)
#define T float
#define INIT(j) (0.5f + (j) * 1e-3f)
#define OP(x) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(x))
#define FOLD(x) __float_as_uint(x)
#elif defined(V_rcp)
#define T float
#define INIT(j) (1.5f + (j) * 1e-3f)
#define OP(x) asm volatile("rcp.approx.ftz.f32 %0, %0; mul.f32 %0, %0, 0f3F800347;" : "+f"(x))
#define FOLD(x) __float_as_uint(x)
#elif defined(V_rsqrt)
#define T float
#define INIT(j) (1.5f + (j) * 1e-3f)
#define OP(x) asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(x))
#define FOLD(x) __float_as_uint(x)
#elif defined(V_lg2)
#define T float
#define INIT(j) (1.5f + (j) * 1e-3f)
#define OP(x) asm volatile("lg2.approx.ftz.f32 %0, %0;" : "+f"(x))
#define FOLD(x) __float_as_uint(x)
#elif defined(V_sin)
#define T float
#define INIT(j) (0.5f + (j) * 1e-3f)
#define OP(x) asm volatile("sin.approx.ftz.f32 %0, %0;" : "+f"(x))
#define FOLD(x) __float_as_uint(x)
#elif defined(V_tanh)
#define T float
#define INIT(j) (0.5f + (j) * 1e-3f)
#define OP(x) asm volatile("tanh.approx.f32 %0, %0;" : "+f"(x))
#define FOLD(x) __float_as_uint(x)
#elif defined(V_ex2_f16x2)
#define T unsigned
#define INIT(j) (0x38003800u + (j))
#define OP(x) asm volatile("ex2.approx.f16x2 %0, %0;" : "+r"(x))
#define FOLD(x) (x)
#elif defined(V_ex2_bf16x2)
#define T unsigned
#define INIT(j) (0x3f003f00u + (j))
#define OP(x) asm volatile("ex2.approx.ftz.bf16x2 %0, %0;" : "+r"(x))
#define FOLD(x) (x)
#elif defined(V_tanh_f16x2)
#define T unsigned
#define INIT(j) (0x38003800u + (j))
#define OP(x) asm volatile("tanh.approx.f16x2 %0, %0;" : "+r"(x))
#define FOLD(x) (x)
#elif defined(V_shfl)
#define T unsigned
#define INIT(j) (threadIdx.x * 7u + (j))
#define OP(x) asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1f, 0xffffffff;" : "+r"(x))
#define FOLD(x) (x)
#elif defined(V_redux)
#define T unsigned
#define INIT(j) (threadIdx.x * 7u + (j))
#define OP(x) asm volatile("redux.sync.add.u32 %0, %0, 0xffffffff;" : "+r"(x))
#define FOLD(x) (x)
#elif defined(V_dfma)
#define T double
#define INIT(j) (1.0 + (j) * 1e-3)
#define OP(x) asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(x) : "d"(0.999), "d"(1e-3))
#define FOLD(x) ((unsigned)__double_as_longlong(x))
#endif

extern "C" __global__ void k(int iters, unsigned *out, u64 *clk) {
    if (clk && blockIdx.x == 0 && threadIdx.x == 0) {
        clk[0] = clock64();
        clk[2] = gtimer();
    }
    T x[CH];
#pragma unroll
    for (int j = 0; j < CH; j++) x[j] = INIT(j);
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < CH; j++) OP(x[j]);
    }
    unsigned s = 0;
#pragma unroll
    for (int j = 0; j < CH; j++) s ^= FOLD(x[j]);
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    if (clk && blockIdx.x == 0 && threadIdx.x == 0) {
        clk[1] = clock64();
        clk[3] = gtimer();
    }
}
