static __device__ __forceinline__ float euhedral_half_to_float(unsigned short value) {
    unsigned int sign = ((unsigned int)value & 0x8000u) << 16;
    unsigned int exponent = ((unsigned int)value >> 10) & 0x1fu;
    unsigned int mantissa = (unsigned int)value & 0x3ffu;
    unsigned int bits;

    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign;
        } else {
            int unbiased = -14;
            while ((mantissa & 0x400u) == 0u) {
                mantissa <<= 1;
                --unbiased;
            }
            mantissa &= 0x3ffu;
            bits = sign | ((unsigned int)(unbiased + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    return __uint_as_float(bits);
}

static __device__ __forceinline__ unsigned short euhedral_float_to_bfloat16(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        return (unsigned short)((bits >> 16) | 0x40u);
    }
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ __launch_bounds__(128) void euhedral_q3_embedding(
        const int* token_ids,
        const unsigned char* weights,
        unsigned short* output,
        unsigned int token_count,
        unsigned int vocabulary_size,
        unsigned int hidden_size,
        unsigned long long scale_offset) {
    unsigned int groups_per_row = ((hidden_size + 127u) / 128u) * 2u;
    unsigned int active_groups = hidden_size / 64u;
    unsigned int groups_per_block = (active_groups + 3u) / 4u;
    unsigned int token = (unsigned int)blockIdx.x / groups_per_block;
    unsigned int group = ((unsigned int)blockIdx.x % groups_per_block) * 4u
            + ((unsigned int)threadIdx.x >> 5u);
    if (token >= token_count || group >= active_groups) {
        return;
    }

    int row = token_ids[token];
    if (row < 0 || (unsigned int)row >= vocabulary_size) {
        return;
    }

    unsigned int lane = (unsigned int)threadIdx.x & 31u;
    unsigned long long row_group = (unsigned long long)(unsigned int)row * groups_per_row + group;
    const unsigned int* group_words = (const unsigned int*)(weights + row_group * 24ull);
    unsigned int word = lane < 6u ? group_words[lane] : 0u;
    unsigned int first_word_lane = (lane * 6u) >> 5u;
    unsigned int second_word_lane = first_word_lane + 1u;
    unsigned int low = __shfl_sync(0xffffffffu, word, first_word_lane);
    unsigned int high = __shfl_sync(0xffffffffu, word, second_word_lane);
    unsigned int shift = (lane * 6u) & 31u;
    unsigned long long pair = ((unsigned long long)high << 32) | low;
    unsigned int codes = (unsigned int)(pair >> shift) & 0x3fu;

    int first = (int)(codes & 7u);
    int second = (int)((codes >> 3u) & 7u);
    if ((first & 4) != 0) {
        first -= 8;
    }
    if ((second & 4) != 0) {
        second -= 8;
    }

    const unsigned short* scales = (const unsigned short*)(weights + scale_offset);
    unsigned short scale_bits = lane == 0u ? scales[row_group] : 0u;
    scale_bits = __shfl_sync(0xffffffffu, scale_bits, 0);
    float scale = euhedral_half_to_float(scale_bits);
    unsigned long long output_index = (unsigned long long)token * hidden_size + group * 64ull + lane * 2ull;
    output[output_index] = euhedral_float_to_bfloat16((float)first * scale);
    output[output_index + 1ull] = euhedral_float_to_bfloat16((float)second * scale);
}

// Embedding gather from a P2E2 tensor (q3/p2e2.cuh, docs/COMPRESSED_Q3.md); this module has no
// include paths, so the layout arithmetic is repeated here. One warp per token walks the row's
// 1024-code slices in order; each lane decodes its 32 codes (half a group) and writes
// bf16(code * scale) exactly as euhedral_q3_embedding does. Requires hidden_size % 1024 == 0.
static __device__ __forceinline__ unsigned long long euhedral_align256(unsigned long long v) {
    return (v + 255ull) & ~255ull;
}

static __device__ __forceinline__ int euhedral_p2e2_big(unsigned int p) {
    return p == 0u ? -3 : p == 1u ? -2 : p == 2u ? 2 : 3;
}

extern "C" __global__ __launch_bounds__(128) void euhedral_q3_p2e2_embedding(
        const int* token_ids,
        const unsigned char* weights,
        unsigned short* output,
        unsigned int token_count,
        unsigned int vocabulary_size,
        unsigned int hidden_size) {
    __shared__ unsigned char fields[256];
    for (unsigned int i = threadIdx.x; i < 256u; i += blockDim.x) {
        unsigned int t0 = i & 3u, t1 = (i >> 2) & 3u, window = i >> 4, used = 0u;
        int c0 = (int)t0 - 1, c1 = (int)t1 - 1;
        if (t0 == 3u) { c0 = euhedral_p2e2_big(window & 3u); used = 2u; }
        if (t1 == 3u) c1 = euhedral_p2e2_big((window >> used) & 3u);
        fields[i] = (unsigned char)(((unsigned int)c0 & 7u) | (((unsigned int)c1 & 7u) << 3));
    }
    __syncthreads();
    unsigned int token = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    unsigned int lane = threadIdx.x & 31u;
    if (token >= token_count) return;
    int row = token_ids[token];
    if (row < 0 || (unsigned int)row >= vocabulary_size) return;

    unsigned long long rows = vocabulary_size, words = hidden_size / 16u, groups = hidden_size / 64u;
    unsigned long long base_plane = euhedral_align256(rows * hidden_size / 4u);
    unsigned long long scale_plane = euhedral_align256(base_plane + 4ull * rows);
    unsigned long long payload_plane = euhedral_align256(scale_plane + rows * groups * 2ull);
    const unsigned int* primary = (const unsigned int*)weights + (unsigned long long)(unsigned int)row * words;
    const unsigned short* scales = (const unsigned short*)(weights + scale_plane) + (unsigned long long)(unsigned int)row * groups;
    const unsigned int* payload = (const unsigned int*)(weights + payload_plane);
    unsigned int base = ((const unsigned int*)(weights + base_plane))[row];

    for (unsigned int slice = 0; slice < hidden_size / 1024u; slice++) {
        unsigned int p[2] = {primary[slice * 64u + 2u * lane], primary[slice * 64u + 2u * lane + 1u]};
        unsigned int big[2] = {p[0] & (p[0] >> 1) & 0x55555555u, p[1] & (p[1] >> 1) & 0x55555555u};
        unsigned int count_x = __popc(big[0]), count = count_x + __popc(big[1]);
        unsigned int inclusive = count;
        for (int d = 1; d < 32; d <<= 1) {
            unsigned int up = __shfl_up_sync(0xffffffffu, inclusive, d);
            if (lane >= (unsigned int)d) inclusive += up;
        }
        unsigned int unit = base + inclusive - count;
        const unsigned int* q = payload + (unit >> 4);
        unsigned int shift = (unit & 15u) * 2u;
        unsigned int lo = __funnelshift_r(q[0], q[1], shift), hi = __funnelshift_r(q[1], q[2], shift);
        float scale = euhedral_half_to_float(scales[slice * 16u + (lane >> 1)]);
        unsigned int packed[16];
        for (unsigned int i = 0; i < 16u; i++) {
            unsigned int half = i >> 3, s = 4u * (i & 7u);
            unsigned int rank = (half ? count_x : 0u) + __popc(big[half] & ((1u << s) - 1u));
            unsigned int b = 2u * rank;
            unsigned int window = (b < 32u ? __funnelshift_r(lo, hi, b) : hi >> (b - 32u)) & 15u;
            unsigned int codes = fields[((p[half] >> s) & 15u) | (window << 4)];
            int first = (int)(codes & 7u), second = (int)((codes >> 3u) & 7u);
            if ((first & 4) != 0) first -= 8;
            if ((second & 4) != 0) second -= 8;
            packed[i] = (unsigned int)euhedral_float_to_bfloat16((float)first * scale)
                    | (unsigned int)euhedral_float_to_bfloat16((float)second * scale) << 16;
        }
        uint4* out = (uint4*)(output + (unsigned long long)token * hidden_size + slice * 1024u + lane * 32u);
        for (unsigned int i = 0; i < 4u; i++)
            out[i] = make_uint4(packed[4u * i], packed[4u * i + 1u], packed[4u * i + 2u], packed[4u * i + 3u]);
        base += __shfl_sync(0xffffffffu, inclusive, 31);
    }
}
