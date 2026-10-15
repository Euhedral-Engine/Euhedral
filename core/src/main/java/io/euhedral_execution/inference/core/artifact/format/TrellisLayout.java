package io.euhedral_execution.inference.core.artifact.format;

/// The arithmetic of a trellis object's payload and its `.segments` companion: which tile widths exist, what they
/// mean in bits, and which segment ends a linear may carry.
///
/// A payload is `[k/16, n/16, w]` int16 words, one 16x16 tile per `w` words. `w = 16*K` stores K bits per weight;
/// `w = 16*K + 8` stores the half-step layout in between.
public final class TrellisLayout {

    /// The most segments (distinct input scale vectors) one object carries.
    public static final int MAX_SEGMENTS = 8;

    /// The Hadamard block: every segment boundary, `n` and `k` are multiples of it.
    public static final int BLOCK = 128;

    private TrellisLayout() {}

    /// Whether `w` words per tile is a stored width: 16*K for K in 1..8, or 16*K + 8 for K in 1..7.
    public static boolean wordsValid(int w) {
        if (w < 16 || w > 128) return false;
        if (w % 16 == 0) return true;
        return w % 16 == 8 && w <= 120;
    }

    /// Bits per weight of a tile of `w` words.
    public static double bits(int w) {
        return w / 16.0;
    }

    /// Checks the segment ends `ends` of an object of `n` rows whose `suh` has `suhRows` rows.
    ///
    /// @throws IllegalArgumentException naming the first violated rule
    public static void validateSegments(int[] ends, long n, long suhRows) {
        if (ends.length < 1 || ends.length > MAX_SEGMENTS)
            throw new IllegalArgumentException("segment count " + ends.length + " is outside 1.." + MAX_SEGMENTS);
        if (ends.length != suhRows)
            throw new IllegalArgumentException("segment count " + ends.length + " differs from suh rows " + suhRows);
        int previous = 0;
        for (int index = 0; index < ends.length; index++) {
            int end = ends[index];
            if (end <= previous)
                throw new IllegalArgumentException("segment ends are not strictly increasing at " + index);
            if (end % BLOCK != 0)
                throw new IllegalArgumentException("segment end " + end + " is not a multiple of " + BLOCK);
            previous = end;
        }
        if (previous != n) throw new IllegalArgumentException("last segment end " + previous + " is not n=" + n);
    }

    /// The index of the segment that holds `row`: the first whose end is above it.
    public static int segmentOf(int[] ends, long row) {
        for (int index = 0; index < ends.length; index++) if (row < ends[index]) return index;
        throw new IllegalArgumentException("row " + row + " is beyond the last segment");
    }

    /// The segment ends in the little-endian int32 payload `bytes`.
    public static int[] readEnds(java.lang.foreign.MemorySegment payload) {
        int[] ends = new int[Math.toIntExact(payload.byteSize() / Integer.BYTES)];
        for (int index = 0; index < ends.length; index++)
            ends[index] = payload.get(
                    java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED.withOrder(java.nio.ByteOrder.LITTLE_ENDIAN),
                    (long) index * Integer.BYTES);
        return ends;
    }
}
