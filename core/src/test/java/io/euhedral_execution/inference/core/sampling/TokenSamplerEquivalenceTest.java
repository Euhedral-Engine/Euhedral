package io.euhedral_execution.inference.core.sampling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The single-pass selection loops must choose exactly what the original implementation chose, for the
/// same rows in the same order, including its random draws, tie-breaking, and failures.
class TokenSamplerEquivalenceTest {

    private static final float[] TEMPERATURES = {
        1.0f, 0.7f, 2.5f, 1.0e-3f, 1.0e-30f, Float.MIN_VALUE, Float.MIN_NORMAL, 3.0e38f
    };
    private static final float[] TOP_PS = {1.0f, 0.95f, 0.5f, 1.0e-4f};

    @Test
    void greedySelectionMatchesTheOriginalOnEveryRowShape() {
        SplittableRandom random = new SplittableRandom(11);
        for (int vocabulary : new int[] {1, 2, 3, 7, 64, 1000, 4099}) {
            var config = GenerationConfig.greedy(5L);
            var sampler = new TokenSampler(config, vocabulary);
            var reference = new ReferenceTokenSampler(config, vocabulary);
            for (int row = 0; row < 300; row++) {
                float[] logits = row(random, vocabulary, row);
                assertEquals(outcome(() -> reference.selectToken(logits)), outcome(() -> sampler.selectToken(logits)));
            }
        }
    }

    @Test
    void stochasticSelectionMatchesTheOriginalDrawForDraw() {
        SplittableRandom random = new SplittableRandom(23);
        int compared = 0;
        for (int vocabulary : new int[] {1, 2, 5, 33, 600, 2049}) {
            int[] topKs = {0, 1, 2, 5, 20, vocabulary - 1, vocabulary, vocabulary + 3};
            for (float temperature : TEMPERATURES) {
                for (int topK : topKs) {
                    if (topK < 0) continue;
                    for (float topP : TOP_PS) {
                        long seed = random.nextLong();
                        var config = new GenerationConfig(temperature, topK, topP, seed, false);
                        var sampler = new TokenSampler(config, vocabulary);
                        var reference = new ReferenceTokenSampler(config, vocabulary);
                        for (int row = 0; row < 12; row++) {
                            float[] logits = row(random, vocabulary, row);
                            String expected = outcome(() -> reference.selectToken(logits));
                            assertEquals(
                                    expected,
                                    outcome(() -> sampler.selectToken(logits)),
                                    () -> "vocabulary " + vocabulary + " " + config);
                            compared++;
                        }
                    }
                }
            }
        }
        assertTrue(compared > 10_000);
    }

    @Test
    void stochasticSelectionMatchesTheOriginalAtTheServedVocabulary() {
        int vocabulary = 248_320;
        SplittableRandom random = new SplittableRandom(31);
        var config = new GenerationConfig(1.0f, 20, 0.95f, 99L, false);
        var sampler = new TokenSampler(config, vocabulary);
        var reference = new ReferenceTokenSampler(config, vocabulary);
        for (int row = 0; row < 40; row++) {
            float[] logits = row(random, vocabulary, row);
            assertEquals(outcome(() -> reference.selectToken(logits)), outcome(() -> sampler.selectToken(logits)));
        }
    }

    /// Realistic, tied, special-valued, and sparse rows, cycling by row index.
    private static float[] row(SplittableRandom random, int vocabulary, int index) {
        float[] logits = new float[vocabulary];
        switch (index % 6) {
            case 0 -> {
                for (int i = 0; i < vocabulary; i++) logits[i] = (float) (random.nextGaussian() * 3.0);
            }
            case 1 -> {
                // Few distinct values: ties everywhere, including at the maximum and the top-k cutoff.
                for (int i = 0; i < vocabulary; i++) logits[i] = random.nextInt(4) * 0.5f;
            }
            case 2 -> {
                for (int i = 0; i < vocabulary; i++) logits[i] = special(random);
            }
            case 3 -> {
                // Mostly unselectable: fewer candidates than top-k, sometimes none.
                for (int i = 0; i < vocabulary; i++)
                    logits[i] = random.nextInt(8) == 0 ? (float) random.nextGaussian() : Float.NEGATIVE_INFINITY;
                if (random.nextInt(4) == 0) logits[random.nextInt(vocabulary)] = Float.NaN;
            }
            case 4 -> {
                for (int i = 0; i < vocabulary; i++) logits[i] = random.nextBoolean() ? 0.0f : -0.0f;
            }
            default -> {
                List<Float> extremes = new ArrayList<>(List.of(
                        Float.MAX_VALUE, -Float.MAX_VALUE, Float.MIN_VALUE, -Float.MIN_VALUE, Float.MIN_NORMAL));
                for (int i = 0; i < vocabulary; i++)
                    logits[i] = random.nextInt(3) == 0
                            ? extremes.get(random.nextInt(extremes.size()))
                            : (float) (random.nextGaussian() * 1.0e3);
            }
        }
        return logits;
    }

    private static float special(SplittableRandom random) {
        return switch (random.nextInt(9)) {
            case 0 -> Float.NaN;
            case 1 -> Float.POSITIVE_INFINITY;
            case 2 -> Float.NEGATIVE_INFINITY;
            case 3 -> -0.0f;
            case 4 -> 0.0f;
            case 5 -> Float.MIN_VALUE;
            case 6 -> -Float.MIN_VALUE;
            default -> (float) random.nextGaussian();
        };
    }

    /// The selected token, or the failure the original reported.
    private static String outcome(java.util.function.IntSupplier selection) {
        try {
            return "token " + selection.getAsInt();
        } catch (IllegalArgumentException failure) {
            return "rejected: " + failure.getMessage();
        }
    }
}
