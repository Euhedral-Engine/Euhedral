package io.euhedral_execution.inference.core.qwen4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The n-gram id function against a direct translation of upstream's tensor program (`_shift_right_ignore_eos`
/// with a cumulative maximum of end-of-sequence positions), on whole streams and on the chunkings an engine uses.
class Qwen4NgramIdsTest {

    private static final int EOS = 248044;
    private static final long[] MULTIPLIERS = {1_234_567_891L, 987_654_321_987L, 31_415_926_535L};

    private static long[] vocab(int heads) {
        long[] sizes = new long[heads];
        for (int i = 0; i < heads; i++) sizes[i] = 20_000_003L + 6L * i;
        return sizes;
    }

    private static long[] offsets(long[] sizes) {
        long[] offsets = new long[sizes.length];
        long total = 0;
        for (int i = 0; i < sizes.length; i++) {
            offsets[i] = total;
            total += sizes[i];
        }
        return offsets;
    }

    /// Upstream's forward over one whole token stream with the two-token EOS context in front.
    private static long[] upstream(int[] tokens, long[] multipliers, long[] sizes, long[] offsets, int headsPerNgram) {
        int context = 2;
        int[] history = new int[context + tokens.length];
        Arrays.fill(history, 0, context, EOS);
        System.arraycopy(tokens, 0, history, context, tokens.length);
        int length = history.length;
        int[][] shifted = new int[3][];
        for (int shift = 0; shift < 3; shift++) shifted[shift] = shiftRightIgnoreEos(history, shift);
        int heads = 2 * headsPerNgram;
        long[] ids = new long[tokens.length * heads];
        for (int ngram = 2; ngram <= 3; ngram++) {
            int start = (ngram - 2) * headsPerNgram;
            for (int p = context; p < length; p++) {
                long mixed = (long) shifted[0][p] * multipliers[0];
                for (int position = 1; position < ngram; position++)
                    mixed ^= (long) shifted[position][p] * multipliers[position];
                for (int h = 0; h < headsPerNgram; h++)
                    ids[(p - context) * heads + start + h] =
                            Math.floorMod(mixed, sizes[start + h]) + offsets[start + h];
            }
        }
        return ids;
    }

    private static int[] shiftRightIgnoreEos(int[] tokens, int shift) {
        if (shift == 0) return tokens;
        int length = tokens.length;
        int[] previousEos = new int[length]; // the last EOS position before p, or -1 (cummax, shifted right by one)
        int running = -1;
        for (int p = 0; p < length; p++) {
            previousEos[p] = running;
            if (tokens[p] == EOS) running = p;
        }
        int[] shifted = new int[length];
        for (int p = 0; p < length; p++) {
            int segmentStart = previousEos[p] + 1;
            int source = p - shift;
            boolean valid = p - segmentStart >= shift && source >= 0;
            shifted[p] = valid ? tokens[Math.max(source, 0)] : EOS;
        }
        return shifted;
    }

    private static Qwen4NgramIds ids(int headsPerNgram) {
        long[] sizes = vocab(2 * headsPerNgram);
        return new Qwen4NgramIds(3, headsPerNgram, MULTIPLIERS, sizes, offsets(sizes), EOS);
    }

    private static int[] stream(SplittableRandom rng, int count, double eosRate) {
        int[] tokens = new int[count];
        for (int i = 0; i < count; i++) tokens[i] = rng.nextDouble() < eosRate ? EOS : rng.nextInt(248320);
        return tokens;
    }

    @Test
    void wholeStreamsMatchUpstream() {
        SplittableRandom rng = new SplittableRandom(21);
        int headsPerNgram = 8;
        long[] sizes = vocab(2 * headsPerNgram);
        long[] offsets = offsets(sizes);
        for (double eosRate : new double[] {0.0, 0.1, 0.5}) {
            for (int count : new int[] {1, 2, 3, 40}) {
                int[] tokens = stream(rng, count, eosRate);
                long[] expected = upstream(tokens, MULTIPLIERS, sizes, offsets, headsPerNgram);
                Qwen4NgramIds function = ids(headsPerNgram);
                long[] actual = new long[count * function.heads()];
                function.compute(function.newContext(), tokens, 0, count, actual);
                assertArrayEquals(expected, actual, "rate " + eosRate + " count " + count);
            }
        }
    }

    @Test
    void chunkingAndDecodeDoNotChangeTheIds() {
        SplittableRandom rng = new SplittableRandom(22);
        int headsPerNgram = 8;
        long[] sizes = vocab(2 * headsPerNgram);
        long[] offsets = offsets(sizes);
        int[] tokens = stream(rng, 97, 0.15);
        long[] expected = upstream(tokens, MULTIPLIERS, sizes, offsets, headsPerNgram);
        for (int[] chunks :
                new int[][] {{97}, {1, 96}, {2, 1, 94}, {13, 1, 1, 1, 40, 41}, {1, 1, 1, 1, 93}, {50, 47}}) {
            Qwen4NgramIds function = ids(headsPerNgram);
            Qwen4NgramIds.Context context = function.newContext();
            long[] actual = new long[tokens.length * function.heads()];
            long[] chunkIds = new long[tokens.length * function.heads()];
            int at = 0;
            for (int chunk : chunks) {
                function.compute(context, tokens, at, chunk, chunkIds);
                System.arraycopy(chunkIds, 0, actual, at * function.heads(), chunk * function.heads());
                at += chunk;
            }
            assertEquals(tokens.length, at);
            assertArrayEquals(expected, actual, Arrays.toString(chunks));
        }
    }

    @Test
    void aSequenceStartAndAnEosBoundaryHideEarlierTokens() {
        Qwen4NgramIds function = ids(2);
        long[] first = new long[function.heads()];
        long[] second = new long[function.heads()];
        // The first token of a sequence, and the first token after EOS, see only EOS before them: the ids do not
        // depend on what came earlier.
        function.compute(function.newContext(), new int[] {7}, 0, 1, first);
        Qwen4NgramIds.Context context = function.newContext();
        function.compute(context, new int[] {100, 200, EOS}, 0, 3, new long[3 * function.heads()]);
        function.compute(context, new int[] {7}, 0, 1, second);
        assertArrayEquals(first, second);
        // The token after the first one sees it as its predecessor.
        long[] next = new long[function.heads()];
        function.compute(context, new int[] {8}, 0, 1, next);
        assertFalse(Arrays.equals(first, next));
    }
}
