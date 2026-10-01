package io.euhedral_execution.inference.core.sampling;

import java.util.Objects;
import java.util.SplittableRandom;

/// Test oracle: TokenSampler as it was before its selection loops were optimized, kept verbatim.
/// Create one sampler per generation; instances are not thread-safe and must not be shared concurrently.
/// Sampling excludes NaN and negative infinity; positive infinities share probability mass equally.
final class ReferenceTokenSampler {

    private final GenerationConfig config;
    private final int vocabularySize;
    private final SplittableRandom random;

    ReferenceTokenSampler(GenerationConfig config, int vocabularySize) {
        this.config = Objects.requireNonNull(config, "config");
        if (vocabularySize <= 0) throw new IllegalArgumentException("vocabularySize must be positive");
        this.vocabularySize = vocabularySize;
        this.random = new SplittableRandom(config.seed());
    }

    public int vocabularySize() {
        return this.vocabularySize;
    }

    public int selectToken(float[] logits) {
        Objects.requireNonNull(logits, "logits");
        if (logits.length != this.vocabularySize) {
            throw new IllegalArgumentException(
                    "expected " + this.vocabularySize + " vocabulary logits, got " + logits.length);
        }
        if (this.config.greedy() || this.config.temperature() <= 0.0f) return argmax(logits);

        // Scale scores before applying the filters; top-k ties prefer the lower token ID.
        double[] scaledScores = new double[this.vocabularySize];
        int[] candidateIds = new int[this.vocabularySize];
        int candidateCount = 0;
        for (int tokenId = 0; tokenId < logits.length; tokenId++) {
            float logit = logits[tokenId];
            if (Float.isNaN(logit) || logit == Float.NEGATIVE_INFINITY) continue;
            scaledScores[tokenId] = (double) logit / this.config.temperature();
            candidateIds[candidateCount++] = tokenId;
        }
        if (candidateCount == 0) throw new IllegalArgumentException("logit row has no selectable token");

        int topK = this.config.topK();
        if (topK > 0 && topK < candidateCount) {
            candidateIds = retainTopK(candidateIds, candidateCount, topK, scaledScores);
            candidateCount = topK;
        }

        // Top-p uses the normalized top-k distribution, and draw renormalizes the retained prefix.
        boolean applyTopP = this.config.topP() < 1.0f;
        if (applyTopP) sortByPriority(candidateIds, candidateCount, scaledScores);

        double[] probabilities = normalizedProbabilities(candidateIds, candidateCount, scaledScores);
        int retainedCount = candidateCount;
        if (applyTopP) {
            double cumulativeProbability = 0.0;
            for (int i = 0; i < candidateCount; i++) {
                cumulativeProbability += probabilities[i];
                if (cumulativeProbability >= this.config.topP()) {
                    retainedCount = i + 1;
                    break;
                }
            }
        }

        return draw(candidateIds, probabilities, retainedCount);
    }

    private static int argmax(float[] logits) {
        int bestTokenId = -1;
        float bestLogit = Float.NEGATIVE_INFINITY;
        for (int tokenId = 0; tokenId < logits.length; tokenId++) {
            float logit = logits[tokenId];
            if (Float.isNaN(logit) || logit == Float.NEGATIVE_INFINITY) continue;
            if (bestTokenId < 0 || logit > bestLogit) {
                bestTokenId = tokenId;
                bestLogit = logit;
            }
        }
        if (bestTokenId < 0) throw new IllegalArgumentException("logit row has no selectable token");
        return bestTokenId;
    }

    private static int[] retainTopK(int[] candidateIds, int candidateCount, int topK, double[] scores) {
        int[] heap = new int[topK];
        int heapSize = 0;
        for (int i = 0; i < candidateCount; i++) {
            int tokenId = candidateIds[i];
            if (heapSize < topK) {
                heap[heapSize] = tokenId;
                siftUpWorstFirst(heap, heapSize, scores);
                heapSize++;
            } else if (comparePriority(tokenId, heap[0], scores) < 0) {
                heap[0] = tokenId;
                siftDownWorstFirst(heap, heapSize, 0, scores);
            }
        }
        return heap;
    }

    private static void siftUpWorstFirst(int[] heap, int index, double[] scores) {
        while (index > 0) {
            int parent = (index - 1) >>> 1;
            if (comparePriority(heap[parent], heap[index], scores) >= 0) return;
            swap(heap, parent, index);
            index = parent;
        }
    }

    private static void siftDownWorstFirst(int[] heap, int heapSize, int index, double[] scores) {
        while (true) {
            int left = (index << 1) + 1;
            if (left >= heapSize) return;
            int right = left + 1;
            int worseChild = right < heapSize && comparePriority(heap[left], heap[right], scores) < 0 ? right : left;
            if (comparePriority(heap[index], heap[worseChild], scores) >= 0) return;
            swap(heap, index, worseChild);
            index = worseChild;
        }
    }

    private static void sortByPriority(int[] tokenIds, int count, double[] scores) {
        // A worst-first heap produces best-first order without recursive stack growth.
        for (int root = (count >>> 1) - 1; root >= 0; root--) {
            siftDownWorstFirst(tokenIds, count, root, scores);
        }
        for (int end = count - 1; end > 0; end--) {
            swap(tokenIds, 0, end);
            siftDownWorstFirst(tokenIds, end, 0, scores);
        }
    }

    private static int comparePriority(int leftTokenId, int rightTokenId, double[] scores) {
        double leftScore = scores[leftTokenId];
        double rightScore = scores[rightTokenId];
        if (leftScore > rightScore) return -1;
        if (leftScore < rightScore) return 1;
        return Integer.compare(leftTokenId, rightTokenId);
    }

    private static double[] normalizedProbabilities(int[] tokenIds, int count, double[] scores) {
        double[] probabilities = new double[count];
        int positiveInfinityCount = 0;
        double maximumScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) {
            double score = scores[tokenIds[i]];
            if (score == Double.POSITIVE_INFINITY) positiveInfinityCount++;
            if (score > maximumScore) maximumScore = score;
        }

        if (positiveInfinityCount > 0) {
            double probability = 1.0 / positiveInfinityCount;
            for (int i = 0; i < count; i++) {
                if (scores[tokenIds[i]] == Double.POSITIVE_INFINITY) probabilities[i] = probability;
            }
            return probabilities;
        }

        double totalWeight = 0.0;
        for (int i = 0; i < count; i++) {
            probabilities[i] = Math.exp(scores[tokenIds[i]] - maximumScore);
            totalWeight += probabilities[i];
        }
        if (!(totalWeight > 0.0) || !Double.isFinite(totalWeight)) {
            throw new IllegalArgumentException("logit row has no normalizable probability mass");
        }
        for (int i = 0; i < count; i++) probabilities[i] /= totalWeight;
        return probabilities;
    }

    private int draw(int[] tokenIds, double[] probabilities, int retainedCount) {
        double retainedMass = 0.0;
        int lastPositiveProbability = -1;
        for (int i = 0; i < retainedCount; i++) {
            retainedMass += probabilities[i];
            if (probabilities[i] > 0.0) lastPositiveProbability = tokenIds[i];
        }
        if (!(retainedMass > 0.0) || !Double.isFinite(retainedMass) || lastPositiveProbability < 0) {
            throw new IllegalArgumentException("sampling filters removed all probability mass");
        }

        double draw = this.random.nextDouble() * retainedMass;
        double cumulativeMass = 0.0;
        for (int i = 0; i < retainedCount; i++) {
            cumulativeMass += probabilities[i];
            if (probabilities[i] > 0.0 && draw < cumulativeMass) return tokenIds[i];
        }
        return lastPositiveProbability;
    }

    private static void swap(int[] values, int left, int right) {
        int value = values[left];
        values[left] = values[right];
        values[right] = value;
    }
}
