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
