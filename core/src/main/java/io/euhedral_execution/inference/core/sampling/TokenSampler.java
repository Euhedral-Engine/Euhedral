package io.euhedral_execution.inference.core.sampling;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.IntPredicate;

/// Selects one token from a single vocabulary-logit row using a generation-local random stream.
/// Create one sampler per generation; instances are not thread-safe and must not be shared concurrently.
/// Sampling excludes NaN and negative infinity; positive infinities share probability mass equally.
///
/// Selection runs once per generated token on the host-serial token boundary, so each row is scanned
/// once, and stochastic sampling reuses this generation's scratch rows instead of allocating them.
public final class TokenSampler {
    /// Fewest best candidates a constrained selection ranks before it tests them.
    static final int MIN_RANKED = 64;

    private final GenerationConfig config;
    private final int vocabularySize;
    private final SplittableRandom random;
    private double[] scaledScores;
    private int[] candidateIds;
    private double[] probabilities;
    private int[] ranked;

    public TokenSampler(GenerationConfig config, int vocabularySize) {
        this.config = Objects.requireNonNull(config, "config");
        if (vocabularySize <= 0) throw new IllegalArgumentException("vocabularySize must be positive");
        this.vocabularySize = vocabularySize;
        this.random = new SplittableRandom(config.seed());
    }

    public int vocabularySize() {
        return this.vocabularySize;
    }

    /// Whether every selection is the argmax of the logits.
    public boolean greedy() {
        return this.config.greedy() || this.config.temperature() <= 0.0f;
    }

    public int selectToken(float[] logits) {
        Objects.requireNonNull(logits, "logits");
        if (logits.length != this.vocabularySize) {
            throw new IllegalArgumentException(
                    "expected " + this.vocabularySize + " vocabulary logits, got " + logits.length);
        }
        if (greedy()) return argmax(logits);

        // Scale scores before applying the filters; top-k ties prefer the lower token ID. Only the
        // candidates' entries of the reused score row are written and read.
        ensureScratch();
        double[] scaledScores = this.scaledScores;
        int[] candidateIds = this.candidateIds;
        int topK = this.config.topK();
        int candidateCount = topK > 0 && topK < this.vocabularySize
                ? scaleTopK(logits, topK, scaledScores, candidateIds)
                : scaleAll(logits, scaledScores, candidateIds);
        if (candidateCount == 0) throw new IllegalArgumentException("logit row has no selectable token");

        return drawFrom(candidateIds, candidateCount, this.config.topP() < 1.0f);
    }

    /// Selects as [#selectToken(float[])] would from the row with every token `allowed` rejects removed, testing
    /// as few tokens as it can: the best candidates are tested in order until enough are allowed (one for a
    /// greedy selection, `topK` when sampling), and every token only when they are not, in which case the
    /// disallowed entries of `logits` are overwritten with negative infinity.
    ///
    /// A greedy selection, and sampling with top-p below one, select exactly what the masked row selects. With
    /// top-p at one the same candidates and probabilities are drawn in priority order rather than in the
    /// masked row's heap order, so a seed may select another token of the same distribution.
    public int selectToken(float[] logits, IntPredicate allowed) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(allowed, "allowed");
        if (logits.length != this.vocabularySize) {
            throw new IllegalArgumentException(
                    "expected " + this.vocabularySize + " vocabulary logits, got " + logits.length);
        }
        int topK = this.config.topK();
        int needed = greedy() ? 1 : topK > 0 && topK < this.vocabularySize ? topK : 0;
        if (needed > 0) {
            int limit = Math.min(this.vocabularySize, Math.max(MIN_RANKED, needed * 4));
            if (this.ranked == null || this.ranked.length < limit) this.ranked = new int[limit];
            int[] candidates = this.ranked;
            int rankedCount = rankBest(logits, limit, candidates);
            int found = 0;
            for (int index = 0; index < rankedCount && found < needed; index++)
                if (allowed.test(candidates[index])) candidates[found++] = candidates[index];
            // Every selectable token was ranked when fewer than the limit exist, so the allowed ones are all found.
            if (found == needed || (rankedCount < limit && found > 0)) {
                if (greedy()) return candidates[0];
                ensureScratch();
                for (int index = 0; index < found; index++)
                    this.scaledScores[candidates[index]] =
                            (double) logits[candidates[index]] / this.config.temperature();
                // Ranked order is priority order, which a top-p sort keeps.
                System.arraycopy(candidates, 0, this.candidateIds, 0, found);
                return drawFrom(this.candidateIds, found, this.config.topP() < 1.0f);
            }
        }
        for (int tokenId = 0; tokenId < logits.length; tokenId++)
            if (!allowed.test(tokenId)) logits[tokenId] = Float.NEGATIVE_INFINITY;
        return selectToken(logits);
    }

    private void ensureScratch() {
        if (this.scaledScores == null) {
            this.scaledScores = new double[this.vocabularySize];
            this.candidateIds = new int[this.vocabularySize];
            this.probabilities = new double[this.vocabularySize];
        }
    }

    /// Normalizes the candidates' scaled scores, keeps the top-p prefix when `applyTopP`, and draws.
    private int drawFrom(int[] candidateIds, int candidateCount, boolean applyTopP) {
        double[] scaledScores = this.scaledScores;
        // Top-p uses the normalized top-k distribution, and draw renormalizes the retained prefix.
        if (applyTopP) sortByPriority(candidateIds, candidateCount, scaledScores);

        double[] probabilities =
                normalizedProbabilities(candidateIds, candidateCount, scaledScores, this.probabilities);
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

    /// Writes the `limit` best selectable tokens to `ranked`, best first (higher logit, then lower token ID),
    /// and returns how many there are: fewer than `limit` only when the row has fewer selectable tokens.
    private static int rankBest(float[] logits, int limit, int[] ranked) {
        int size = 0;
        for (int tokenId = 0; tokenId < logits.length; tokenId++) {
            float logit = logits[tokenId];
            if (Float.isNaN(logit) || logit == Float.NEGATIVE_INFINITY) continue;
            if (size < limit) {
                ranked[size] = tokenId;
                siftUpRanked(ranked, size++, logits);
            } else if (logit > logits[ranked[0]]) {
                // A later token with an equal logit has the higher ID and ranks lower: it never enters.
                ranked[0] = tokenId;
                siftDownRanked(ranked, size, 0, logits);
            }
        }
        for (int end = size - 1; end > 0; end--) {
            swap(ranked, 0, end);
            siftDownRanked(ranked, end, 0, logits);
        }
        return size;
    }

    /// Worst-first order of the ranking heap: a lower logit, or an equal logit and a higher ID, is worse.
    private static boolean ranksBelow(int left, int right, float[] logits) {
        return logits[left] < logits[right] || (logits[left] == logits[right] && left > right);
    }

    private static void siftUpRanked(int[] heap, int index, float[] logits) {
        while (index > 0) {
            int parent = (index - 1) >>> 1;
            if (!ranksBelow(heap[index], heap[parent], logits)) return;
            swap(heap, parent, index);
            index = parent;
        }
    }

    private static void siftDownRanked(int[] heap, int size, int index, float[] logits) {
        while (true) {
            int left = (index << 1) + 1;
            if (left >= size) return;
            int right = left + 1;
            int worse = right < size && ranksBelow(heap[right], heap[left], logits) ? right : left;
            if (!ranksBelow(heap[worse], heap[index], logits)) return;
            swap(heap, index, worse);
            index = worse;
        }
    }

    /// One comparison per logit. NaN never compares greater, negative infinity never beats the initial
    /// negative infinity, and a strict comparison keeps the lowest token ID among equal maxima.
    private static int argmax(float[] logits) {
        int bestTokenId = -1;
        float bestLogit = Float.NEGATIVE_INFINITY;
        for (int tokenId = 0; tokenId < logits.length; tokenId++) {
            if (logits[tokenId] > bestLogit) {
                bestLogit = logits[tokenId];
                bestTokenId = tokenId;
            }
        }
        if (bestTokenId < 0) throw new IllegalArgumentException("logit row has no selectable token");
        return bestTokenId;
    }

    /// Scales every selectable logit and lists the candidates in token-ID order.
    private int scaleAll(float[] logits, double[] scaledScores, int[] candidateIds) {
        int candidateCount = 0;
        for (int tokenId = 0; tokenId < logits.length; tokenId++) {
            float logit = logits[tokenId];
            if (Float.isNaN(logit) || logit == Float.NEGATIVE_INFINITY) continue;
            scaledScores[tokenId] = (double) logit / this.config.temperature();
            candidateIds[candidateCount++] = tokenId;
        }
        return candidateCount;
    }

    /// Retains the `topK` best candidates in one pass, leaving them in `candidateIds` exactly as a
    /// worst-first heap built over every candidate in token-ID order would. With no more selectable
    /// tokens than `topK`, every candidate is retained in token-ID order instead.
    ///
    /// A later candidate replaces the heap's worst entry only when it scores strictly higher: equal
    /// scores keep the lower token ID, which is already in the heap. Dividing a non-NaN float by a
    /// positive, finite float temperature preserves both its order and its ties in double precision
    /// across the whole float range, so that test is made on the raw logit and only retained candidates
    /// are scaled.
    private int scaleTopK(float[] logits, int topK, double[] scaledScores, int[] heap) {
        double temperature = this.config.temperature();
        int heapSize = 0;
        int tokenId = 0;
        for (; tokenId < logits.length && heapSize < topK; tokenId++) {
            float logit = logits[tokenId];
            if (Float.isNaN(logit) || logit == Float.NEGATIVE_INFINITY) continue;
            scaledScores[tokenId] = (double) logit / temperature;
            heap[heapSize] = tokenId;
            siftUpWorstFirst(heap, heapSize, scaledScores);
            heapSize++;
        }
        // Only when another candidate follows the first topK does the heap order apply.
        boolean exceeded = false;
        for (; tokenId < logits.length && !exceeded; tokenId++) {
            float logit = logits[tokenId];
            if (Float.isNaN(logit) || logit == Float.NEGATIVE_INFINITY) continue;
            exceeded = true;
            replaceWorst(logits, tokenId, temperature, scaledScores, heap, heapSize);
        }
        if (!exceeded) {
            Arrays.sort(heap, 0, heapSize);
            return heapSize;
        }
        // NaN and negative infinity never compare greater than a retained logit.
        float worstLogit = logits[heap[0]];
        for (; tokenId < logits.length; tokenId++) {
            if (logits[tokenId] > worstLogit) {
                replaceWorst(logits, tokenId, temperature, scaledScores, heap, heapSize);
                worstLogit = logits[heap[0]];
            }
        }
        return topK;
    }

    /// Replaces the heap's worst candidate with `tokenId` when that scores strictly higher.
    private static void replaceWorst(
            float[] logits, int tokenId, double temperature, double[] scaledScores, int[] heap, int heapSize) {
        if (!(logits[tokenId] > logits[heap[0]])) return;
        scaledScores[tokenId] = (double) logits[tokenId] / temperature;
        heap[0] = tokenId;
        siftDownWorstFirst(heap, heapSize, 0, scaledScores);
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

    private static double[] normalizedProbabilities(
            int[] tokenIds, int count, double[] scores, double[] probabilities) {
        Arrays.fill(probabilities, 0, count, 0.0);
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
