#pragma once
#include <cuda_fp16.h>
// Pipeline helpers shared by the tensor-core attention kernels (decode and prefill): cp.async, shared-memory mbarriers
// for warp-specialized hand-offs (named barriers would cap the SM at one resident CTA when nine ids are used), and
// the exact-product FP16 expansion of the NVFP4 cache.
namespace nvfp4pipe {
static __device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
static __device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
static __device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> static __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
static __device__ __forceinline__ void mbarrier_init(unsigned long long* mb, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(mb)), "r"(count) : "memory");
}
static __device__ __forceinline__ void mbarrier_arrive(unsigned long long* mb) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(mb)) : "memory");
}
static __device__ __forceinline__ void mbarrier_wait(unsigned long long* mb, unsigned parity) {
    asm volatile("{\n .reg .pred p;\n WAIT_LOOP:\n mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n"
                 " @!p bra WAIT_LOOP;\n}" ::"r"(smem_u32(mb)), "r"(parity) : "memory");
}
// FP16 pair for a byte holding two E2M1 codes (low nibble first), before the scale: one table lookup plus one
// FP16 multiply by the group's scale replaces the software expansion. Every product (E2M1 x E4M3: at most five
// significant bits, 2^-10 to 2688) is exact in FP16, so the multiply equals the FP32 product rounded to FP16.
static __device__ __forceinline__ unsigned e2m1_pair_bits(unsigned byte) {
    unsigned pair = 0;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const unsigned code = (byte >> (4 * i)) & 15u, magnitude = code & 7u;
        const unsigned bits = (magnitude < 2u ? magnitude * 0x3800u : ((((magnitude >> 1) + 14u) << 10) | ((magnitude & 1u) << 9)))
                | ((code & 8u) << 12);
        pair |= bits << (16 * i);
    }
    return pair;
}
static __device__ __forceinline__ unsigned half2_multiply(unsigned a, unsigned b) {
    const __half2 product = __hmul2(*reinterpret_cast<const __half2*>(&a), *reinterpret_cast<const __half2*>(&b));
    return *reinterpret_cast<const unsigned*>(&product);
}
// An E4M3 scale byte as two FP16 copies (hardware conversion, exact).
static __device__ __forceinline__ unsigned scale_pair(unsigned byte) {
    unsigned pair;
    asm("{ .reg .b16 b; cvt.u16.u32 b, %1; cvt.rn.f16x2.e4m3x2 %0, b; }" : "=r"(pair) : "r"(byte * 0x0101u));
    return pair;
}

}  // namespace nvfp4pipe
