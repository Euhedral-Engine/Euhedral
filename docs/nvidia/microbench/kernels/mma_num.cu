// Accumulator-precision probe: one mma.sync where only A row 0 / B column 0 /
// C[0][0] are non-zero and live in lane 0's registers. The host chooses the
// element bit patterns; lane 0 returns D[0][0].
// Fragment facts used (PTX ISA fragment layouts):
//   m16n8k16 f16/bf16: lane 0 holds A[0][0..1] in a0, A[0][8..9] in a2; B[0..1][0] in b0, B[8..9][0] in b1
//   m16n8k32 8-bit:    lane 0 holds A[0][0..3] in a0, A[0][16..19] in a2; B[0..3][0] in b0, B[16..19][0] in b1
//   m16n8k64 4-bit:    lane 0 holds A[0][0..7] in a0, A[0][32..39] in a2; B[0..7][0] in b0, B[32..39][0] in b1
//   D: lane 0 d0 = D[0][0] (for f16 accumulators: low half of d0)
extern "C" __global__ void k(const unsigned *in, float *out) {
    unsigned lane = threadIdx.x;
    unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
    float c0 = 0.f;
    unsigned c16 = 0;
    if (lane == 0) {
        a0 = in[0];
        a2 = in[1];
        b0 = in[2];
        b1 = in[3];
        c0 = __uint_as_float(in[4]);
        c16 = in[4] & 0xffffu;
    }
    unsigned sa = in[5], sb = in[6];
    float d0 = c0, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    unsigned h0 = c16, h1 = 0;
    (void)sa;
    (void)sb;
    (void)h0;
    (void)h1;
#if defined(V_f16_f32)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(V_bf16_f32)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(V_f16_f16)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(h0), "+r"(h1) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    asm("{ .reg .b16 h; cvt.u16.u32 h, %1; cvt.f32.f16 %0, h; }" : "=f"(d0) : "r"(h0));
#elif defined(V_e4m3_f32)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(V_e4m3_f16)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(h0), "+r"(h1) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    asm("{ .reg .b16 h; cvt.u16.u32 h, %1; cvt.f32.f16 %0, h; }" : "=f"(d0) : "r"(h0));
#elif defined(V_mxf8f6f4_e4m3)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#elif defined(V_mxf4nvf4_4x)
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#elif defined(V_mxf4_2x)
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#elif defined(V_s8_s32)
    int i0 = (int)in[4], i1 = 0, i2 = 0, i3 = 0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(i0), "+r"(i1), "+r"(i2), "+r"(i3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    d0 = (float)i0;
#endif
    if (lane == 0) out[0] = d0;
}
