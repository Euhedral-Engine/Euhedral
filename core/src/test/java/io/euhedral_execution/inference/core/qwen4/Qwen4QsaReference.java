package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bf;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bits;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.r;

import java.util.Arrays;

/// CPU references of the QSA operators, written from Qwen4ExpTextQSAIndexer and Qwen4ExpTextAttention of
/// the pinned
/// upstream revision (modular_qwen4_exp.py). Tensors are BF16 where upstream's are, every elementwise
/// operation
/// rounds to BF16, reductions run in double.
final class Qwen4QsaReference {

    static final int BLOCK = 4;
    static final int BUDGET_BLOCKS = 512;
    static final int INDEX_DIM = 128;
    static final int INDEX_HEADS = 4;
    static final float THETA = 1.0e7f;
    static final float EPSILON = 1.0e-6f;

    private Qwen4QsaReference() {}

    /// cos and sin of RoPE index `index` at `position` rounded to BF16, as Qwen4ExpTextRotaryEmbedding
    /// builds them.
    static float[] cosSin(int position, int index, float theta, int rotary) {
        double power = Math.pow(theta, (2.0 * index) / rotary);
        float inverse = 1.0f / (float) power;
        float angle = (float) position * inverse;
        return new float[] {r(Math.cos(angle)), r(Math.sin(angle))};
    }

    /// Qwen4ExpTextRMSNorm of `width` BF16 values at `offset` with the `(1 + weight)` scale, as BF16
    /// bits.
    static short[] headNorm(short[] x, int offset, int width, short[] weight, float epsilon) {
        double sum = 0;
        for (int i = 0; i < width; i++) sum += (double) bf(x[offset + i]) * bf(x[offset + i]);
        float inverse = (float) (1.0 / Math.sqrt(sum / width + epsilon));
        short[] out = new short[width];
        for (int i = 0; i < width; i++) out[i] = bits(bf(x[offset + i]) * inverse * (1.0f + bf(weight[i])));
        return out;
    }

    /// apply_rotary_pos_emb on the first `rotary` values of a BF16 head (all ops rounded to BF16).
    static short[] rope(short[] head, int position, float theta, int rotary) {
        short[] out = head.clone();
        int half = rotary / 2;
        for (int i = 0; i < half; i++) {
            float[] cs = cosSin(position, i, theta, rotary);
            float x1 = bf(head[i]), x2 = bf(head[i + half]);
            out[i] = bits(r(x1 * cs[0]) + r(-x2 * cs[1]));
            out[i + half] = bits(r(x2 * cs[0]) + r(x1 * cs[1]));
        }
        return out;
    }

    /// norm then RoPE of one head: the indexer's q_layernorm + RoPE or the attention's q_norm + RoPE.
    static short[] normRope(short[] x, int offset, int width, short[] weight, int position, int rotary) {
        short[] normed = headNorm(x, offset, width, weight, EPSILON);
        return rotary == 0 ? normed : rope(normed, position, THETA, rotary);
    }

    /// The pooled, normalized, rotated key of block `block` of `raw` ([token][128] BF16): the mean of the
    /// four raw keys
    /// in FP32 rounded to BF16, k_layernorm, RoPE at the position of the block's first token.
    static short[] blockKey(short[][] raw, int block, short[] kNorm) {
        short[] pooled = new short[INDEX_DIM];
        for (int d = 0; d < INDEX_DIM; d++) {
            float sum = 0;
            for (int i = 0; i < BLOCK; i++) sum += bf(raw[BLOCK * block + i][d]);
            pooled[d] = bits(sum / BLOCK);
        }
        short[] normed = headNorm(pooled, 0, INDEX_DIM, kNorm, EPSILON);
        return rope(normed, BLOCK * block, THETA, 64);
    }

    /// Scores of the first `blocks` block keys for the four query heads `q[4][128]` of one row: the sum
    /// over heads of
    /// relu(q . key) divided by sqrt(128), in double.
    static double[] scores(short[][] q, short[][] keys, int blocks) {
        double[] scores = new double[blocks];
        for (int j = 0; j < blocks; j++) {
            double sum = 0;
            for (int h = 0; h < INDEX_HEADS; h++) {
                double dot = 0;
                for (int d = 0; d < INDEX_DIM; d++) dot += (double) bf(q[h][d]) * bf(keys[j][d]);
                sum += Math.max(dot, 0);
            }
            scores[j] = sum / Math.sqrt(INDEX_DIM);
        }
        return scores;
    }

    /// The ids a row selects: the top min(budget, n) scores, ties to the lower id, ascending.
    static int[] select(double[] scores, int budget) {
        int n = scores.length, k = Math.min(budget, n);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(
                order, (a, b) -> scores[a] != scores[b] ? Double.compare(scores[b], scores[a]) : Integer.compare(a, b));
        int[] ids = new int[k];
        for (int i = 0; i < k; i++) ids[i] = order[i];
        Arrays.sort(ids);
        return ids;
    }

    /// The k-th highest score (k = min(budget, n)); a block scoring above it must be selected and one
    /// scoring below it
    /// must not.
    static double kthScore(double[] scores, int budget) {
        double[] sorted = scores.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length - Math.min(budget, sorted.length)];
    }

    /// What a selection check found: blocks the selection has that the reference's does not, and the
    /// largest score by
    /// which a selected block falls short of the k-th score or an omitted block exceeds it.
    record Check(int differing, double gap) {}

    /// Checks a GPU selection against reference scores up to `epsilon`: right count, ascending distinct
    /// ids in range,
    /// no selected block more than epsilon below the k-th score, no omitted block more than epsilon above
    /// it.
    static Check checkSelection(double[] scores, int[] ids, int count, int budget, double epsilon, String what) {
        int n = scores.length, k = Math.min(budget, n);
        if (count != k) throw new AssertionError(what + ": selected " + count + " blocks, expected " + k);
        double kth = kthScore(scores, budget);
        double gap = 0;
        boolean[] chosen = new boolean[n];
        for (int i = 0; i < count; i++) {
            int id = ids[i];
            if (id < 0 || id >= n || (i > 0 && id <= ids[i - 1]))
                throw new AssertionError(what + ": ids not ascending in range at " + i + ": " + id);
            chosen[id] = true;
            gap = Math.max(gap, kth - scores[id]);
            if (scores[id] < kth - epsilon)
                throw new AssertionError(
                        what + ": selected block " + id + " scores " + scores[id] + " below k-th " + kth);
        }
        for (int j = 0; j < n; j++)
            if (!chosen[j]) {
                gap = Math.max(gap, scores[j] - kth);
                if (scores[j] > kth + epsilon)
                    throw new AssertionError(
                            what + ": omitted block " + j + " scores " + scores[j] + " above k-th " + kth);
            }
        int[] expected = select(scores, budget);
        int differ = 0;
        for (int id : expected) if (!chosen[id]) differ++;
        return new Check(differ, gap);
    }

    /// Normalized Walsh-Hadamard transform of 256 values (entries +-1/16): the cache's rotation, its own
    /// inverse.
    static double[] hadamard(double[] x) {
        double[] v = x.clone();
        for (int stride = 1; stride < 256; stride <<= 1)
            for (int base = 0; base < 256; base += 2 * stride)
                for (int i = base; i < base + stride; i++) {
                    double a = v[i], b = v[i + stride];
                    v[i] = a + b;
                    v[i + stride] = a - b;
                }
        for (int i = 0; i < 256; i++) v[i] /= 16.0;
        return v;
    }

    static double e4m3(int code) {
        int exponent = (code >> 3) & 15, mantissa = code & 7;
        if (exponent == 0) return mantissa / 8.0 * Math.pow(2, -6);
        return (1.0 + mantissa / 8.0) * Math.pow(2, exponent - 7);
    }

    static final double[] E2M1 = {0, 0.5, 1, 1.5, 2, 3, 4, 6};

    /// The represented values (unrotated: the inverse rotation applied) of one 144-byte cache row.
    static double[] decodeRow(byte[] page, int offset) {
        double[] rotated = new double[256];
        for (int d = 0; d < 256; d++) {
            int code = (page[offset + d / 2] >> (4 * (d & 1))) & 15;
            double magnitude = E2M1[code & 7];
            rotated[d] = (code & 8) != 0 ? -magnitude : magnitude;
            rotated[d] *= e4m3(page[offset + 128 + d / 16] & 0xff);
        }
        return hadamard(rotated);
    }

    /// Qwen4ExpTextAttention's softmax attention of one query head (256 BF16 values at `qOffset` of `q`)
    /// over the
    /// virtual key list of row `position`: the tokens of the selected blocks (`ids[0 ..< count]`, null:
    /// all complete
    /// blocks) then the tail. `keys` and `values` are `[token][256]` effective (unrotated) values of the
    /// KV head.
    static double[] attend(
            short[] q, int qOffset, double[][] keys, double[][] values, int position, int[] ids, int count) {
        int nb = (position + 1) / BLOCK;
        int blocks = ids == null ? nb : count;
        int n = BLOCK * blocks + (position + 1 - BLOCK * nb);
        int[] tokens = new int[n];
        for (int slot = 0; slot < blocks; slot++)
            for (int u = 0; u < BLOCK; u++) tokens[BLOCK * slot + u] = BLOCK * (ids == null ? slot : ids[slot]) + u;
        for (int t = BLOCK * blocks; t < n; t++) tokens[t] = BLOCK * nb + (t - BLOCK * blocks);
        double[] logits = new double[n];
        double max = Double.NEGATIVE_INFINITY;
        for (int t = 0; t < n; t++) {
            double dot = 0;
            for (int d = 0; d < 256; d++) dot += (double) bf(q[qOffset + d]) * keys[tokens[t]][d];
            logits[t] = dot / 16.0;
            max = Math.max(max, logits[t]);
        }
        double sum = 0;
        double[] out = new double[256];
        for (int t = 0; t < n; t++) {
            double p = Math.exp(logits[t] - max);
            sum += p;
            for (int d = 0; d < 256; d++) out[d] += p * values[tokens[t]][d];
        }
        for (int d = 0; d < 256; d++) out[d] /= sum;
        return out;
    }

    /// Relative RMS error of `actual` against `expected` (BF16 bits against doubles).
    static double relativeRms(double[] expected, short[] actual) {
        double error = 0, norm = 0;
        for (int i = 0; i < expected.length; i++) {
            double d = bf(actual[i]) - expected[i];
            error += d * d;
            norm += expected[i] * expected[i];
        }
        return Math.sqrt(error / Math.max(norm, 1e-300));
    }

    /// Relative RMS error of two BF16 tensors.
    static double relativeRms(short[] expected, short[] actual) {
        double[] e = new double[expected.length];
        for (int i = 0; i < e.length; i++) e[i] = bf(expected[i]);
        return relativeRms(e, actual);
    }

    /// The largest |difference| in units of the expected tensor's RMS.
    static double maxDifferenceInRms(short[] expected, short[] actual) {
        double norm = 0;
        for (short value : expected) norm += (double) bf(value) * bf(value);
        double rms = Math.sqrt(norm / expected.length);
        double worst = 0;
        for (int i = 0; i < expected.length; i++)
            worst = Math.max(worst, Math.abs(bf(expected[i]) - bf(actual[i])) / rms);
        return worst;
    }
}
