package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.E2M1;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bf;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bits;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.e4m3;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.r;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.silu;

import io.euhedral_execution.inference.core.model_loader.artifact.Nvfp4Layout;
import java.util.SplittableRandom;

/// CPU reference of one routed expert, written from upstream's Qwen4ExpTextExperts: weights expanded exactly
/// (`e2m1 * sign * (e4m3 * global)` in FP32), linears accumulated in double and rounded to BF16 once, then
/// `bf16(bf16(silu(gate)) * up)`, `bf16(down(act))`, `bf16(y * routing weight)`, and the per-row routed sum as a
/// sequence of BF16 additions in ascending expert order.
final class Qwen4ExpertReference {

    static final int HIDDEN = 2560;
    static final int INTER = 640;
    static final int GATE_UP_OFFSET = 0;
    static final int DOWN_OFFSET = 1_843_456;
    static final int RECORD_BYTES = 2_768_896;

    private Qwen4ExpertReference() {}

    /// One expert's expanded weights: `gateUp` `[2 * INTER][HIDDEN]` and `down` `[HIDDEN][INTER]`.
    record Expert(float[] gateUp, float[] down) {}

    /// Expands the NVFP4 tensor of `rows x k` at `offset` of a record, exactly.
    static float[] expand(byte[] record, int offset, int rows, int k) {
        long scaleAt = offset + Nvfp4Layout.scaleOffset(rows, k);
        int globalAt = (int) (offset + Nvfp4Layout.globalScaleOffset(rows, k));
        float global = Float.intBitsToFloat((record[globalAt] & 0xff)
                | (record[globalAt + 1] & 0xff) << 8
                | (record[globalAt + 2] & 0xff) << 16
                | (record[globalAt + 3] & 0xff) << 24);
        float[] values = new float[rows * k];
        for (int row = 0; row < rows; row++) {
            for (int i = 0; i < k; i++) {
                int byteValue = record[offset + row * (k / 2) + (i >> 1)] & 0xff;
                int nibble = (i & 1) != 0 ? byteValue >> 4 : byteValue & 15;
                float magnitude = E2M1[nibble & 7];
                float scale = e4m3(record[(int) scaleAt + row * (k / 16) + (i >> 4)] & 0xff) * global;
                float value = magnitude * scale;
                values[row * k + i] = (nibble & 8) != 0 ? -value : value;
            }
        }
        return values;
    }

    static Expert expand(byte[] record) {
        return new Expert(
                expand(record, GATE_UP_OFFSET, 2 * INTER, HIDDEN), expand(record, DOWN_OFFSET, HIDDEN, INTER));
    }

    /// act of one row: `[INTER]` BF16 bits.
    static short[] act(Expert expert, short[] x, int rowOffset) {
        short[] act = new short[INTER];
        double[] sums = new double[2];
        for (int j = 0; j < INTER; j++) {
            for (int part = 0; part < 2; part++) {
                int base = (part * INTER + j) * HIDDEN;
                double sum = 0;
                for (int i = 0; i < HIDDEN; i++) sum += (double) bf(x[rowOffset + i]) * expert.gateUp[base + i];
                sums[part] = r(sum);
            }
            act[j] = bits(r(silu((float) sums[0])) * (float) sums[1]);
        }
        return act;
    }

    /// weighted of one row: `[HIDDEN]` BF16 bits from act and the BF16 routing weight bits.
    static short[] weighted(Expert expert, short[] act, short weight) {
        short[] weighted = new short[HIDDEN];
        float w = bf(weight);
        for (int j = 0; j < HIDDEN; j++) {
            double sum = 0;
            for (int i = 0; i < INTER; i++) sum += (double) bf(act[i]) * expert.down[j * INTER + i];
            weighted[j] = bits(r(sum) * w);
        }
        return weighted;
    }

    /// Adds `weighted` into `sum` as a BF16 tensor add: `sum = bf16(sum + weighted)`.
    static void accumulate(short[] sum, short[] weighted) {
        for (int j = 0; j < sum.length; j++) sum[j] = bits(bf(sum[j]) + bf(weighted[j]));
    }

    /// Synthetic expert record: random codes, E4M3 scales of 0.5..2 and a global scale that keeps a unit-variance
    /// activation's outputs near 1.
    static byte[] randomRecord(SplittableRandom random) {
        byte[] record = new byte[RECORD_BYTES];
        tensor(random, record, GATE_UP_OFFSET, 2 * INTER, HIDDEN, 0.012f + 0.004f * random.nextFloat());
        tensor(random, record, DOWN_OFFSET, HIDDEN, INTER, 0.024f + 0.008f * random.nextFloat());
        return record;
    }

    private static void tensor(SplittableRandom random, byte[] record, int offset, int rows, int k, float global) {
        int scaleAt = (int) (offset + Nvfp4Layout.scaleOffset(rows, k));
        int globalAt = (int) (offset + Nvfp4Layout.globalScaleOffset(rows, k));
        for (int i = 0; i < rows * k / 2; i++) record[offset + i] = (byte) random.nextInt(256);
        for (int i = 0; i < rows * k / 16; i++) record[scaleAt + i] = (byte) (0x30 + random.nextInt(0x11));
        int bitsOfGlobal = Float.floatToRawIntBits(global);
        for (int b = 0; b < 4; b++) record[globalAt + b] = (byte) (bitsOfGlobal >>> (8 * b));
    }

    /// BF16 normal values.
    static short[] randomBf16(SplittableRandom random, int count, double scale) {
        short[] values = new short[count];
        for (int i = 0; i < count; i++) {
            double u = random.nextDouble(), v = random.nextDouble();
            values[i] =
                    bits((float) (Math.sqrt(-2 * Math.log(Math.max(u, 1e-12))) * Math.cos(2 * Math.PI * v) * scale));
        }
        return values;
    }
}
