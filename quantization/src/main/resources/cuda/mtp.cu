// Kernels of the MTP drafter tools (quantization/README.md).

namespace {
constexpr int kThreads = 256;
}

// One block per row: the first index of the largest of `vocabulary` BF16 logits (greedy decode's token).
extern "C" __global__ void argmax_rows(const unsigned short* logits, int vocabulary, int* out) {
    __shared__ float best_value[kThreads];
    __shared__ int best_index[kThreads];
    const unsigned short* row = logits + (long long)blockIdx.x * vocabulary;
    float value = __uint_as_float(0xff800000u);
    int index = 0x7fffffff;
    for (int i = threadIdx.x; i < vocabulary; i += kThreads) {
        float v = __uint_as_float(static_cast<unsigned int>(row[i]) << 16);
        if (v > value) { value = v; index = i; }
    }
    best_value[threadIdx.x] = value;
    best_index[threadIdx.x] = index;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = best_value[threadIdx.x + stride];
            int other_index = best_index[threadIdx.x + stride];
            if (other > best_value[threadIdx.x]
                    || (other == best_value[threadIdx.x] && other_index < best_index[threadIdx.x])) {
                best_value[threadIdx.x] = other;
                best_index[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = best_index[0];
}
