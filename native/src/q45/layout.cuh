#pragma once

namespace q45 {
// Persistent Q4/Q5 layout (unchanged): per G64 group, 32 low-nibble code bytes;
// for Q5 a separate plane of 8 fifth-bit bytes per group; then one FP16 scale
// per group. Each plane starts on a 256-byte boundary.
static constexpr unsigned int kGroup = 64;
static constexpr unsigned int kCodeBytes = 32;
static constexpr unsigned int kHighBytes = 8;

static __device__ __forceinline__ unsigned long long align256(unsigned long long n) {
    return (n + 255ull) & ~255ull;
}

template<int BITS>
struct Layout {
    static_assert(BITS == 4 || BITS == 5, "packed Q4/Q5 layout only");
    const unsigned char* codes;
    const unsigned char* high;
    const unsigned short* scales;
    unsigned int groups;

    __device__ __forceinline__ Layout(
            const unsigned char* weights, unsigned int in_features, unsigned int out_features)
            : codes(weights), groups(in_features / kGroup) {
        const unsigned long long code_plane = align256((unsigned long long)out_features * groups * kCodeBytes);
        const unsigned long long high_plane =
                BITS == 5 ? align256((unsigned long long)out_features * groups * kHighBytes) : 0ull;
        high = weights + code_plane;
        scales = (const unsigned short*)(weights + code_plane + high_plane);
    }

    // Linear group index for output row `out` and K group `k_group`.
    __device__ __forceinline__ unsigned long long group(unsigned int out, unsigned int k_group) const {
        return (unsigned long long)out * groups + k_group;
    }
    // Groups are 32 (codes) and 8 (fifth bits) bytes, so both planes can be
    // read as 32-bit words, and consecutive groups are contiguous words.
    __device__ __forceinline__ const unsigned int* code_words(unsigned long long g) const {
        return (const unsigned int*)(codes + g * kCodeBytes);
    }
    __device__ __forceinline__ const unsigned int* high_words(unsigned long long g) const {
        return (const unsigned int*)(high + g * kHighBytes);
    }
};

}  // namespace q45
