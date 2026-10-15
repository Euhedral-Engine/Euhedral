package io.euhedral_execution.inference.core.sampling;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.function.IntPredicate;
import org.junit.jupiter.api.Test;

class TokenSamplerKeyedTest {
    private static final GenerationConfig PUBLISHED = new GenerationConfig(1.0f, 20, 0.95f, 1234L, false);

    @Test
    void aPositionSamplesTheSameTokenInAnyOrder() {
        float[] row = new float[64];
        for (int i = 0; i < row.length; i++) row[i] = (float) Math.log(1 + i);
        var forward = new TokenSampler(PUBLISHED, 64);
        var backward = new TokenSampler(PUBLISHED, 64);
        int[] a = new int[200], b = new int[200];
        for (int p = 0; p < 200; p++) a[p] = forward.selectToken(row.clone(), 1000L + p);
        for (int p = 199; p >= 0; p--) b[p] = backward.selectToken(row.clone(), 1000L + p);
        assertArrayEquals(a, b);
    }

    @Test
    void drawsAreUniform() {
        int[] bins = new int[10];
        for (long p = 0; p < 100_000; p++) {
            double u = TokenSampler.unitDraw(77L, p);
            assertTrue(u >= 0 && u < 1);
            bins[(int) (u * 10)]++;
        }
        double chi = 0;
        for (int n : bins) chi += Math.pow(n - 10_000, 2) / 10_000;
        assertTrue(chi < 30, "chi-square " + chi);
    }

    @Test
    void keyedSamplingFollowsTheDistribution() {
        var config = new GenerationConfig(1.0f, 0, 1.0f, 5L, false);
        var sampler = new TokenSampler(config, 2);
        float[] row = {0f, (float) Math.log(3)};
        int ones = 0;
        for (long p = 0; p < 20_000; p++) ones += sampler.selectToken(row.clone(), p);
        assertEquals(0.75, ones / 20_000.0, 0.015);
    }

    @Test
    void greedyIgnoresThePosition() {
        var sampler = new TokenSampler(GenerationConfig.greedy(1L), 3);
        assertEquals(1, sampler.selectToken(new float[] {0f, 2f, 1f}, 5));
        assertEquals(1, sampler.selectToken(new float[] {0f, 2f, 1f}, 6));
    }

    @Test
    void aConstrainedKeyedDrawMatchesTheMaskedRow() {
        var config = new GenerationConfig(1.0f, 20, 0.9f, 9L, false);
        var random = new java.util.SplittableRandom(3);
        for (int trial = 0; trial < 200; trial++) {
            float[] row = new float[100];
            for (int i = 0; i < row.length; i++) row[i] = (float) random.nextGaussian();
            IntPredicate allowed = id -> id % 3 != 0;
            float[] masked = row.clone();
            for (int i = 0; i < masked.length; i++) if (!allowed.test(i)) masked[i] = Float.NEGATIVE_INFINITY;
            assertEquals(
                    new TokenSampler(config, 100).selectToken(masked, trial),
                    new TokenSampler(config, 100).selectToken(row.clone(), allowed, trial));
        }
    }
}
