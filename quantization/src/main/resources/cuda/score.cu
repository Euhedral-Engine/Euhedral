// Teacher-forced scoring against a llama.cpp BF16 reference (tools/compare_reference.py, on the device): for each row,
// the engine's logits (BF16 [vocabulary]) against the reference's 16-bit log-probabilities. The reference row holds
// two FP32 values (scale, minimum) then a uint16 code per token: log p = scale * code + minimum (in float64). Writes,
// per row: the candidate's NLL of the forced token, the reference's, KL(reference || candidate) over the reference
// entries above its -16 floor, top-1 agreement (first maximum on both sides), and delta-p of the forced token.
#include <cuda_bf16.h>

namespace {

constexpr int kThreads = 256;
constexpr double kFloor = -16.0;
constexpr double kLowest = -1.7976931348623157e308;  // below every finite value

__device__ __forceinline__ float logit(const unsigned short* row, int i) {
    return __uint_as_float(static_cast<unsigned int>(row[i]) << 16);
}

// Block reduction of (value, index): the larger value, the smaller index on ties.
__device__ void reduce_max(double& value, int& index, double* shared_value, int* shared_index) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_down_sync(0xffffffffu, value, offset);
        int other_index = __shfl_down_sync(0xffffffffu, index, offset);
        if (other > value || (other == value && other_index < index)) { value = other; index = other_index; }
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) { shared_value[warp] = value; shared_index[warp] = index; }
    __syncthreads();
    if (warp == 0) {
        value = lane < kThreads / 32 ? shared_value[lane] : kLowest;
        index = lane < kThreads / 32 ? shared_index[lane] : 0x7fffffff;
        for (int offset = 16; offset > 0; offset >>= 1) {
            double other = __shfl_down_sync(0xffffffffu, value, offset);
            int other_index = __shfl_down_sync(0xffffffffu, index, offset);
            if (other > value || (other == value && other_index < index)) { value = other; index = other_index; }
        }
        if (lane == 0) { shared_value[0] = value; shared_index[0] = index; }
    }
    __syncthreads();
    value = shared_value[0];
    index = shared_index[0];
    __syncthreads();
}

__device__ double reduce_sum(double value, double* shared) {
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < kThreads / 32 ? shared[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
        if (lane == 0) shared[0] = value;
    }
    __syncthreads();
    value = shared[0];
    __syncthreads();
    return value;
}

}  // namespace

// One block per row. logits: [rows, vocabulary] BF16; reference: [rows, width] uint16 (width = 2 * ceil(vocabulary / 2)
// + 4); targets: [rows]; out: [rows, 5] float64 (nll, reference nll, kl, agree, delta-p).
extern "C" __global__ void score_rows(const unsigned short* logits, const unsigned short* reference, int width,
                                      int vocabulary, const int* targets, double* out) {
    __shared__ double shared_value[kThreads / 32];
    __shared__ int shared_index[kThreads / 32];
    const unsigned short* q = logits + (long long)blockIdx.x * vocabulary;
    const unsigned short* row = reference + (long long)blockIdx.x * width;
    const float scale = __uint_as_float(row[0] | (static_cast<unsigned int>(row[1]) << 16));
    const float minimum = __uint_as_float(row[2] | (static_cast<unsigned int>(row[3]) << 16));
    const unsigned short* codes = row + 4;

    double q_max = kLowest, p_max = kLowest;
    int q_arg = 0x7fffffff, p_arg = 0x7fffffff;
    for (int i = threadIdx.x; i < vocabulary; i += kThreads) {
        double value = logit(q, i);
        if (value > q_max) { q_max = value; q_arg = i; }
        double p = (double)scale * (double)codes[i] + (double)minimum;
        if (p > p_max) { p_max = p; p_arg = i; }
    }
    reduce_max(q_max, q_arg, shared_value, shared_index);
    reduce_max(p_max, p_arg, shared_value, shared_index);
    double sum = 0.0;
    for (int i = threadIdx.x; i < vocabulary; i += kThreads) sum += exp((double)logit(q, i) - q_max);
    const double log_z = q_max + log(reduce_sum(sum, shared_value));
    double kl = 0.0;
    for (int i = threadIdx.x; i < vocabulary; i += kThreads) {
        double p = (double)scale * (double)codes[i] + (double)minimum;
        if (p > kFloor) kl += exp(p) * (p - ((double)logit(q, i) - log_z));
    }
    kl = reduce_sum(kl, shared_value);
    if (threadIdx.x == 0) {
        int target = targets[blockIdx.x];
        double q_target = (double)logit(q, target) - log_z;
        double p_target = (double)scale * (double)codes[target] + (double)minimum;
        double* result = out + 5LL * blockIdx.x;
        result[0] = -q_target;
        result[1] = -p_target;
        result[2] = kl;
        result[3] = q_arg == p_arg ? 1.0 : 0.0;
        result[4] = exp(q_target) - exp(p_target);
    }
}
