// Statistics of the residual rotation (Rotate.java).

// Adds, over every group of `group` consecutive values of `w` (groups of a row's input columns), the excess kurtosis
// m4 / m2^2 - 3 into sums[0] and the number of groups with nonzero variance into sums[1].
extern "C" __global__ void group_kurtosis(const float* w, long long groups, int group, double* sums) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double kurtosis = 0.0, counted = 0.0;
    if (g < groups) {
        const float* x = w + g * group;
        double mean = 0.0;
        for (int i = 0; i < group; i++) mean += x[i];
        mean /= group;
        double m2 = 0.0, m4 = 0.0;
        for (int i = 0; i < group; i++) {
            double d = x[i] - mean, d2 = d * d;
            m2 += d2;
            m4 += d2 * d2;
        }
        m2 /= group;
        m4 /= group;
        if (m2 > 0.0) {
            kurtosis = m4 / (m2 * m2) - 3.0;
            counted = 1.0;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        kurtosis += __shfl_down_sync(0xffffffffu, kurtosis, offset);
        counted += __shfl_down_sync(0xffffffffu, counted, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(sums, kurtosis);
        atomicAdd(sums + 1, counted);
    }
}
