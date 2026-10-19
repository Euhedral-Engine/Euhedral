package io.euhedral_execution.inference.core.gpu;

/// Independent mathematical reference for the persistent NVFP4 cache representation: 4-bit codes on the INT4 midrise
/// grid (magnitudes m + 0.5, the nearest with integers going to the larger, saturating at 7.5) and one E4M3 scale per
/// 16 values, chosen from the group's largest value over 7.5. H256 is evaluated from (-1)^popcount(i & j), not the
/// device
/// butterfly tree.
final class Nvfp4KvReference {
    /// The largest code magnitude.
    static final double MAXIMUM = 7.5;
    private static final double[] SCALES = new double[127];

    static {
        for (int bits = 0; bits < SCALES.length; bits++)
            SCALES[bits] = bits < 8 ? bits / 512.0 : Math.scalb(1 + (bits & 7) / 8.0, (bits >> 3) - 7);
    }

    private Nvfp4KvReference() {}

    static double[] represented(short[] input, int offset) {
        double[] original = new double[256];
        for (int i = 0; i < 256; i++) original[i] = CudaGpuOperationsIntegrationTest.bf16ToFloat(input[offset + i]);
        double[] rotated = transform(original);
        for (int base = 0; base < 256; base += 16) {
            double maximum = 0;
            for (int d = 0; d < 16; d++) maximum = Math.max(maximum, Math.abs(rotated[base + d]));
            if (maximum == 0) continue;
            double scale = SCALES[nearest(Math.clamp((float) maximum / (float) MAXIMUM, 1.0 / 512, 448), SCALES)];
            for (int d = 0; d < 16; d++) {
                float divided = (float) rotated[base + d] / (float) scale;
                double magnitude = Math.min(Math.floor(Math.abs(divided)), 7) + 0.5;
                rotated[base + d] = Math.copySign(magnitude, divided) * scale;
            }
        }
        return transform(rotated);
    }

    private static int nearest(double value, double[] choices) {
        // Saturate first: very large finite values can round all candidate
        // distances to the same number and otherwise select an even interior code.
        value = Math.clamp(value, choices[0], choices[choices.length - 1]);
        int best = 0;
        double distance = Double.POSITIVE_INFINITY;
        for (int i = 0; i < choices.length; i++) {
            double candidate = Math.abs(value - choices[i]);
            if (candidate < distance || (candidate == distance && (i & 1) == 0)) {
                best = i;
                distance = candidate;
            }
        }
        return best;
    }

    private static double[] transform(double[] input) {
        double[] output = new double[256];
        for (int i = 0; i < 256; i++) {
            double sum = 0;
            for (int j = 0; j < 256; j++) sum += (Integer.bitCount(i & j) & 1) == 0 ? input[j] : -input[j];
            output[i] = sum / 16;
        }
        return output;
    }
}
