#pragma once
#include "../layout.cuh"
#include "decode.cuh"

namespace q45 {
// G64 word set: one warp-wide round of 32-bit loads for one group.
// Lanes 0..7 hold code words 0..7; for Q5 lanes 8..9 hold the fifth-bit words;
// lane kGroupScaleLane holds the FP16 scale (zero-extended); other lanes hold 0.
// SCOPE: lane-local loads. BORROWS: the group's code, fifth-bit and scale bytes.
template<int BITS>
struct GroupWords {
    static constexpr unsigned int kScaleLane = BITS == 5 ? 10u : 8u;
};

template<int BITS>
static __device__ __forceinline__ unsigned int load_group_word(
        const Layout<BITS>& w, unsigned long long g, unsigned int lane) {
    if (lane < 8u) return w.code_words(g)[lane];
    if (BITS == 5 && lane < 10u) return w.high_words(g)[lane - 8u];
    if (lane == GroupWords<BITS>::kScaleLane) return (unsigned int)w.scales[g];
    return 0u;
}

// Hand a G64 word set to load ownership: the returned pair holds the codes for
// K = 2*lane and 2*lane + 1 within the group (unpack_code slot 0 and 1).
// SCOPE: warp-collective.
template<int BITS>
static __device__ __forceinline__ unsigned int group_pair(unsigned int word, unsigned int lane) {
    unsigned int code = __shfl_sync(0xffffffffu, word, lane >> 2);
    unsigned int pair = (code >> ((lane & 3u) * 8u)) & 0xffu;
    if (BITS == 5) {
        unsigned int high = __shfl_sync(0xffffffffu, word, 8u + (lane >> 4));
        pair |= ((high >> ((lane * 2u) & 31u)) & 3u) << 8;
    }
    return pair;
}

// Broadcast the scale of a G64 word set. SCOPE: warp-collective.
template<int BITS>
static __device__ __forceinline__ float group_scale(unsigned int word) {
    return scale_to_float((unsigned short)__shfl_sync(0xffffffffu, word, GroupWords<BITS>::kScaleLane));
}

// K128 word set: both adjacent G64 groups starting at the even group g, in a
// single round of loads. Lanes 0..15 hold code words (group g, then g + 1);
// for Q5 lanes 16..19 hold fifth-bit words (g: 16..17, g + 1: 18..19); lane
// kScaleLane holds scales[g] | scales[g + 1] << 16; other lanes hold 0.
// The scale pair is 4-byte aligned: g is even (in_features % 128 == 0) and the
// scale plane is 256-byte aligned.
template<int BITS>
struct K128Words {
    static constexpr unsigned int kScaleLane = BITS == 5 ? 20u : 16u;
};

template<int BITS>
static __device__ __forceinline__ unsigned int load_k128_word(
        const Layout<BITS>& w, unsigned long long g, unsigned int lane) {
    if (lane < 16u) return w.code_words(g)[lane];
    if (BITS == 5 && lane < 20u) return w.high_words(g)[lane - 16u];
    if (lane == K128Words<BITS>::kScaleLane) return *(const unsigned int*)(w.scales + g);
    return 0u;
}

// Code for K offset lane + 32 * stripe within group g + half of a K128 set
// (K-stripe ownership). SCOPE: warp-collective.
template<int BITS>
static __device__ __forceinline__ int k128_code(unsigned int word, int half, int stripe, unsigned int lane) {
    unsigned int code = __shfl_sync(0xffffffffu, word, half * 8 + stripe * 4 + (lane >> 3));
    unsigned int value = (code >> ((lane & 7u) * 4u)) & 15u;
    if (BITS == 5) {
        unsigned int high = __shfl_sync(0xffffffffu, word, 16 + half * 2 + stripe);
        value |= ((high >> lane) & 1u) << 4;
    }
    return sign_code<BITS>(value);
}

template<int BITS>
static __device__ __forceinline__ float k128_scale(unsigned int word, int half) {
    return scale_to_float((unsigned short)(__shfl_sync(0xffffffffu, word, K128Words<BITS>::kScaleLane) >> (16 * half)));
}

}  // namespace q45
