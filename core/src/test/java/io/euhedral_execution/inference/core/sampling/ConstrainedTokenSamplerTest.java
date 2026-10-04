package io.euhedral_execution.inference.core.sampling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntPredicate;
import org.junit.jupiter.api.Test;

/// A constrained selection tests few tokens yet selects what selection over the masked row selects.
class ConstrainedTokenSamplerTest {
    private static final int VOCABULARY = 3000;

    @Test
    void greedySelectionEqualsTheMaskedArgmax() {
        var random = new SplittableRandom(1);
        for (int trial = 0; trial < 400; trial++) {
            float[] row = row(random);
            IntPredicate allowed = predicate(random);
            int expected = maskedSelection(GenerationConfig.greedy(5), row, allowed);
            assertEquals(
                    expected,
                    new TokenSampler(GenerationConfig.greedy(5), VOCABULARY).selectToken(row.clone(), allowed));
        }
    }

    @Test
    void topKTopPSamplingEqualsTheMaskedRowDrawForDraw() {
        var random = new SplittableRandom(2);
        for (int trial = 0; trial < 200; trial++) {
            float[] row = row(random);
            IntPredicate allowed = predicate(random);
            var config = new GenerationConfig(0.7f + random.nextFloat(), 20, 0.95f, random.nextLong(), false);
            var masked = new TokenSampler(config, VOCABULARY);
            var lazy = new TokenSampler(config, VOCABULARY);
            float[] maskedRow = mask(row, allowed);
            for (int draw = 0; draw < 5; draw++)
                assertEquals(masked.selectToken(maskedRow), lazy.selectToken(row.clone(), allowed), "trial " + trial);
        }
    }

    @Test
    void withoutTopPTheDrawStaysAmongTheMaskedTopK() {
        var random = new SplittableRandom(3);
        var config = new GenerationConfig(1.0f, 20, 1.0f, 9, false);
        var sampler = new TokenSampler(config, VOCABULARY);
        for (int trial = 0; trial < 200; trial++) {
            float[] row = row(random);
            IntPredicate allowed = predicate(random);
            float[] masked = mask(row, allowed);
            int selected = sampler.selectToken(row.clone(), allowed);
            assertTrue(allowed.test(selected));
            int better = 0;
            for (int id = 0; id < VOCABULARY; id++)
                if (masked[id] > masked[selected] || (masked[id] == masked[selected] && id < selected)) better++;
            assertTrue(better < 20, "selected outside the top 20 allowed tokens");
        }
    }

    @Test
    void anUnconstrainingPredicateIsTestedOnlyOnTheBestCandidates() {
        var random = new SplittableRandom(4);
        AtomicInteger tested = new AtomicInteger();
        IntPredicate everything = id -> {
            tested.incrementAndGet();
            return true;
        };
        new TokenSampler(new GenerationConfig(1.0f, 20, 0.95f, 1, false), VOCABULARY)
                .selectToken(row(random), everything);
        assertEquals(20, tested.get());
        tested.set(0);
        new TokenSampler(GenerationConfig.greedy(1), VOCABULARY).selectToken(row(random), everything);
        assertEquals(1, tested.get());
    }

    @Test
    void aPredicateThatRejectsEverythingFails() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new TokenSampler(GenerationConfig.greedy(1), VOCABULARY)
                        .selectToken(row(new SplittableRandom(5)), id -> false));
    }

    private static int maskedSelection(GenerationConfig config, float[] row, IntPredicate allowed) {
        return new TokenSampler(config, VOCABULARY).selectToken(mask(row, allowed));
    }

    private static float[] mask(float[] row, IntPredicate allowed) {
        float[] masked = row.clone();
        for (int id = 0; id < masked.length; id++) if (!allowed.test(id)) masked[id] = Float.NEGATIVE_INFINITY;
        return masked;
    }

    /// Coarse values make ties common; a few NaN, negative infinities and signed zeros.
    private static float[] row(SplittableRandom random) {
        float[] row = new float[VOCABULARY];
        for (int id = 0; id < row.length; id++) {
            int kind = random.nextInt(100);
            row[id] = kind == 0
                    ? Float.NaN
                    : kind == 1 ? Float.NEGATIVE_INFINITY : kind == 2 ? -0.0f : random.nextInt(40) / 4.0f - 5.0f;
        }
        return row;
    }

    /// From permissive (most tokens) to restrictive (a handful), so both the ranked and the full paths run.
    private static IntPredicate predicate(SplittableRandom random) {
        int modulus = 1 + random.nextInt(random.nextBoolean() ? 4 : 600);
        int residue = random.nextInt(modulus);
        return id -> id % modulus == residue;
    }
}
