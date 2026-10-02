#pragma once
// Scale table of an SD4 NVFP4 tensor (row-split-k128-sd4-v1, nvfp4.cuh): 16 E4M3 codes, in registers.
namespace nvfp4 {
struct ScaleTable {
    uint4 codes;
    // E4M3 codes of up to four packed 4-bit indices (nibble i -> byte i). prmt selects among 8 bytes,
    // so the low and high halves of the table are looked up separately and merged on each index's
    // bit 3.
    __device__ __forceinline__ unsigned int lookup(unsigned int indices) const {
        const unsigned int select = indices & 0x7777u;
        const unsigned int low = __byte_perm(codes.x, codes.y, select), high = __byte_perm(codes.z, codes.w, select);
        const unsigned int upper = __byte_perm(0u, 0xffffffffu, (indices >> 1) & 0x4444u);
        return (low & ~upper) | (high & upper);
    }
};
}  // namespace nvfp4
