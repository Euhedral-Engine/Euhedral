package io.euhedral_execution.inference.core.sampling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import org.junit.jupiter.api.Test;

class TokenSamplerTest {

    @Test
    void greedySelectionReturnsTheMaximumLogitAndBreaksTiesByLowestTokenId() {
        TokenSampler sampler = new TokenSampler(GenerationConfig.greedy(7L), 4);
        TokenSampler nonpositiveTemperature = new TokenSampler(new GenerationConfig(-0.5f, 0, 1.0f, 7L, false), 2);

        assertEquals(2, sampler.selectToken(new float[] {-2.0f, -1.0f, 3.0f, 0.0f}));
        assertEquals(0, sampler.selectToken(new float[] {1.0f, 1.0f, -3.0f, Float.NEGATIVE_INFINITY}));
        assertEquals(1, nonpositiveTemperature.selectToken(new float[] {-1.0f, 2.0f}));
    }

    @Test
    void explicitGreedyModeUsesArgmaxEvenWhenTemperatureAndFiltersWouldSample() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 1, 0.1f, 7L, true), 3);

        assertEquals(1, sampler.selectToken(new float[] {-2.0f, 4.0f, 3.0f}));
    }

    @Test
    void topKOneAlwaysSelectsTheHighestLogit() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 1, 1.0f, 31L, false), 3);

        assertEquals(1, sampler.selectToken(new float[] {-2.0f, 4.0f, 3.0f}));
    }

    @Test
    void topKTreatsSignedZeroAsATieAndPrefersTheLowerTokenId() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 1, 1.0f, 31L, false), 2);

        assertEquals(0, sampler.selectToken(new float[] {-0.0f, 0.0f}));
    }

    @Test
    void topKBreaksATieByTokenIdAtTheRetentionCutoff() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 2, 1.0f, 31L, false), 4);

        for (int i = 0; i < 32; i++) {
            int tokenId = sampler.selectToken(new float[] {5.0f, 4.0f, 4.0f, 0.0f});
            assertTrue(tokenId == 0 || tokenId == 1);
        }
    }

    @Test
    void topKRetainsOnlyTheHighestKTokensWithStableTies() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 2, 1.0f, 2L, false), 5);
        float[] logits = {3.0f, 2.0f, 2.0f, 4.0f, 0.0f};

        for (int i = 0; i < 32; i++) {
            int tokenId = sampler.selectToken(logits);
            assertTrue(tokenId == 0 || tokenId == 3);
        }
    }

    @Test
    void topKThenTopPRestrictsSamplingToTheFilteredNucleus() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 2, 0.75f, 2L, false), 4);
        float[] logits = {4.0f, 3.0f, 2.0f, 1.0f};
        boolean selectedSecond = false;

        for (int i = 0; i < 64; i++) {
            int tokenId = sampler.selectToken(logits);
            assertTrue(tokenId == 0 || tokenId == 1);
            selectedSecond |= tokenId == 1;
        }
        assertTrue(selectedSecond);
    }

    @Test
    void topPNearOneRetainsTheTailUntilTheThresholdIsMet() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 0, 0.999999f, 42L, false), 3);
        boolean[] selected = new boolean[3];

        for (int i = 0; i < 96; i++) selected[sampler.selectToken(new float[] {0.0f, 0.0f, 0.0f})] = true;

        assertTrue(selected[0]);
        assertTrue(selected[1]);
        assertTrue(selected[2]);
    }

    @Test
    void samplingApproximatelyFollowsTheNormalizedUniformDistribution() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 0, 1.0f, 42L, false), 4);
        int[] counts = new int[4];

        for (int i = 0; i < 8192; i++) counts[sampler.selectToken(new float[] {0.0f, 0.0f, 0.0f, 0.0f})]++;

        for (int count : counts) assertTrue(count > 500 && count < 2_500);
    }

    @Test
    void topPHandlesLargeAlternatingVocabularyOrdering() {
        float[] logits = new float[248_320];
        for (int tokenId = 0; tokenId < logits.length; tokenId++) {
            logits[tokenId] = tokenId % 2 == 0 ? -tokenId : tokenId;
        }
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 0, 0.5f, 29L, false), logits.length);

        assertEquals(logits.length - 1, sampler.selectToken(logits));
    }

    @Test
    void topPKeepsTheSmallestPrefixThatMeetsTheProbabilityThreshold() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 0, 0.5f, 17L, false), 3);
        TokenSampler largerNucleus = new TokenSampler(new GenerationConfig(1.0f, 0, 0.8f, 17L, false), 3);

        for (int i = 0; i < 32; i++) assertEquals(0, sampler.selectToken(new float[] {10.0f, 9.0f, 0.0f}));
        boolean selectedSecond = false;
        for (int i = 0; i < 32; i++) {
            int tokenId = largerNucleus.selectToken(new float[] {10.0f, 9.0f, 0.0f});
            assertTrue(tokenId == 0 || tokenId == 1);
            selectedSecond |= tokenId == 1;
        }
        assertTrue(selectedSecond);
    }

    @Test
    void topPResolvesAnExactThresholdTieByLowerTokenId() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 0, 0.5f, 17L, false), 2);

        for (int i = 0; i < 32; i++) assertEquals(0, sampler.selectToken(new float[] {0.0f, 0.0f}));
    }

    @Test
    void temperatureChangesTheFilteredSamplingDistribution() {
        float[] logits = {2.0f, 1.0f, 0.0f};
        TokenSampler cold = new TokenSampler(new GenerationConfig(0.1f, 0, 0.8f, 1L, false), logits.length);
        TokenSampler warm = new TokenSampler(new GenerationConfig(10.0f, 0, 0.8f, 1L, false), logits.length);

        for (int i = 0; i < 32; i++) assertEquals(0, cold.selectToken(logits));
        boolean selectedOtherToken = false;
        for (int i = 0; i < 32; i++) selectedOtherToken |= warm.selectToken(logits) != 0;
        assertTrue(selectedOtherToken);
    }

    @Test
    void independentSamplersWithTheSameSeedProduceTheSameSequence() {
        float[] logits = {0.1f, 0.4f, 0.9f, 0.2f};
        TokenSampler first = new TokenSampler(new GenerationConfig(1.0f, 0, 1.0f, 1234L, false), logits.length);
        TokenSampler second = new TokenSampler(new GenerationConfig(1.0f, 0, 1.0f, 1234L, false), logits.length);
        int[] firstSequence = new int[64];
        int[] secondSequence = new int[64];
        for (int i = 0; i < firstSequence.length; i++) {
            firstSequence[i] = first.selectToken(logits);
            secondSequence[i] = second.selectToken(logits);
        }

        assertEquals(Arrays.toString(firstSequence), Arrays.toString(secondSequence));
    }

    @Test
    void differentSeedsCanProduceDifferentSequences() {
        float[] logits = {0.0f, 0.0f, 0.0f, 0.0f};
        TokenSampler first = new TokenSampler(new GenerationConfig(1.0f, 0, 1.0f, 3L, false), logits.length);
        TokenSampler second = new TokenSampler(new GenerationConfig(1.0f, 0, 1.0f, 11L, false), logits.length);
        int[] firstSequence = new int[64];
        int[] secondSequence = new int[64];
        for (int i = 0; i < firstSequence.length; i++) {
            firstSequence[i] = first.selectToken(logits);
            secondSequence[i] = second.selectToken(logits);
        }

        assertNotEquals(Arrays.toString(firstSequence), Arrays.toString(secondSequence));
    }

    @Test
    void greedySelectionRejectsRowsWithNoSelectableLogits() {
        TokenSampler sampler = new TokenSampler(GenerationConfig.greedy(9L), 2);

        assertThrows(
                IllegalArgumentException.class,
                () -> sampler.selectToken(new float[] {Float.NaN, Float.NEGATIVE_INFINITY}));
    }

    @Test
    void stochasticSelectionRejectsRowsWithNoSelectableLogits() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 1, 1.0f, 9L, false), 2);

        assertThrows(
                IllegalArgumentException.class,
                () -> sampler.selectToken(new float[] {Float.NaN, Float.NEGATIVE_INFINITY}));
    }

    @Test
    void topKLargerThanSelectableVocabularyKeepsAllSelectableTokens() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 4, 1.0f, 9L, false), 3);

        assertEquals(2, sampler.selectToken(new float[] {Float.NaN, Float.NEGATIVE_INFINITY, 5.0f}));
    }

    @Test
    void topPHandlesUnderflowedTailProbabilities() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 0, 0.999999f, 9L, false), 3);

        for (int i = 0; i < 16; i++) assertEquals(0, sampler.selectToken(new float[] {0.0f, -10_000.0f, -20_000.0f}));
    }

    @Test
    void samplingAcceptsPositiveInfinityAndExcludesNanAndNegativeInfinity() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 0, 1.0f, 5L, false), 4);

        for (int i = 0; i < 32; i++) {
            int tokenId = sampler.selectToken(
                    new float[] {Float.NaN, Float.POSITIVE_INFINITY, 100.0f, Float.POSITIVE_INFINITY});
            assertTrue(tokenId == 1 || tokenId == 3);
        }
    }

    @Test
    void topPSamplesPositiveInfinitiesWithEqualWeight() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 0, 0.75f, 42L, false), 3);
        boolean selectedFirstInfinity = false;
        boolean selectedSecondInfinity = false;

        for (int i = 0; i < 64; i++) {
            int tokenId = sampler.selectToken(new float[] {Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, 5.0f});
            assertTrue(tokenId == 0 || tokenId == 1);
            selectedFirstInfinity |= tokenId == 0;
            selectedSecondInfinity |= tokenId == 1;
        }

        assertTrue(selectedFirstInfinity);
        assertTrue(selectedSecondInfinity);
    }

    @Test
    void topPAtPositiveInfinityCutoffPrefersTheFirstTiedToken() {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 0, 0.5f, 42L, false), 3);

        for (int i = 0; i < 32; i++) {
            assertEquals(0, sampler.selectToken(new float[] {Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, 5.0f}));
        }
    }

    @Test
    void selectionRejectsVocabularyMismatches() {
        TokenSampler sampler = new TokenSampler(GenerationConfig.greedy(9L), 2);

        assertThrows(IllegalArgumentException.class, () -> sampler.selectToken(new float[] {1.0f}));
    }

    @Test
    void greedySelectionSupportsTheRealQwenVocabularySize() {
        float[] logits = new float[248_320];
        logits[248_319] = 2.0f;
        TokenSampler sampler = new TokenSampler(GenerationConfig.greedy(0L), logits.length);

        assertEquals(248_319, sampler.selectToken(logits));
    }

    private static int[] keyedSequence(long seed, float[] logits, int length, long first) {
        TokenSampler sampler = new TokenSampler(new GenerationConfig(1.0f, 0, 1.0f, seed, false), logits.length);
        int[] sequence = new int[length];
        for (int i = 0; i < length; i++) sequence[i] = sampler.selectToken(logits, first + i);
        return sequence;
    }

    @Test
    void keyedDrawsOfNeighbouringPositionsAreIndependent() {
        float[] logits = {0.0f, 0.0f};
        int[] sequence = keyedSequence(1234L, logits, 4001, 0);
        int[][] pairs = new int[2][2];
        for (int i = 0; i + 1 < sequence.length; i += 2) pairs[sequence[i]][sequence[i + 1]]++;
        // 2000 pairs over four cells: each expects 500; a correlated stream would empty off-diagonal cells.
        for (int[] row : pairs)
            for (int count : row) assertTrue(count > 400 && count < 600, Arrays.deepToString(pairs));
    }

    @Test
    void keyedDrawsDependOnTheSeed() {
        float[] logits = {0.0f, 0.0f, 0.0f, 0.0f};
        assertNotEquals(
                Arrays.toString(keyedSequence(1234L, logits, 64, 0)),
                Arrays.toString(keyedSequence(1235L, logits, 64, 0)));
        assertEquals(
                Arrays.toString(keyedSequence(1234L, logits, 64, 0)),
                Arrays.toString(keyedSequence(1234L, logits, 64, 0)));
    }

    @Test
    void keyedCallsIgnoreTheSequentialStreamAndCallOrder() {
        float[] logits = {0.0f, 0.0f, 0.0f, 0.0f};
        GenerationConfig config = new GenerationConfig(1.0f, 0, 1.0f, 1234L, false);
        int[] expected = keyedSequence(1234L, logits, 64, 0);
        TokenSampler interleaved = new TokenSampler(config, logits.length);
        int[] actual = new int[64];
        for (int i = 63; i >= 0; i--) {
            interleaved.selectToken(logits);
            actual[i] = interleaved.selectToken(logits, (long) i);
            interleaved.selectToken(logits, token -> true);
        }
        assertEquals(Arrays.toString(expected), Arrays.toString(actual));
    }
}
