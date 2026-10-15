package io.euhedral_execution.inference.core.generation;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.sampling.TokenSampler;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;
import java.util.function.IntPredicate;

/// Bridges Qwen BF16 logits to the independent host-side token sampler.
/// Keep one instance per generation so its seeded random state and scratch row are request-local.
public final class LogitsSampler {

    /// Two adjacent BF16 logits as the device wrote them.
    private static final ValueLayout.OfInt BF16_PAIR =
            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static final ValueLayout.OfShort BF16 = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final TokenSampler sampler;
    private final float[] scratch;

    public LogitsSampler(GenerationConfig config, int vocabularySize) {
        this.sampler = new TokenSampler(config, vocabularySize);
        this.scratch = new float[vocabularySize];
    }

    /// Samples the final logits row that a successful quantum copied into `logits`, or returns the token
    /// that its greedy device selection chose.
    public int selectToken(HostLogits logits, IntPredicate allowed) {
        Objects.requireNonNull(logits, "logits");
        requireVocabulary(logits.vocabularySize());
        if (logits.hasSelection()) {
            if (allowed != null || !greedy()) throw new IllegalStateException("device selection is greedy only");
            return logits.selectedToken();
        }
        return select(logits.row(), allowed);
    }

    /// As [#selectToken(HostLogits, IntPredicate)], drawing with the sampler's draw for `position` (the position
    /// the selected token occupies) instead of its sequential stream. A device selection is greedy and has no draw.
    public int selectToken(HostLogits logits, IntPredicate allowed, long position) {
        Objects.requireNonNull(logits, "logits");
        requireVocabulary(logits.vocabularySize());
        if (logits.hasSelection()) {
            if (allowed != null || !greedy()) throw new IllegalStateException("device selection is greedy only");
            return logits.selectedToken();
        }
        return select(logits.row(), allowed, position);
    }

    /// Selects row `row` of the verification copy ([HostLogits#queueRows]) at `position`. No vocabulary
    /// constraint is applied: constrained sessions do not verify speculatively.
    public int selectRow(HostLogits logits, int row, long position) {
        Objects.requireNonNull(logits, "logits");
        requireVocabulary(logits.vocabularySize());
        return select(logits.row(row), null, position);
    }

    /// Each verified row's selection: the device's greedy choices, or each row sampled at its own position
    /// (`firstPosition` is the position the first row's token occupies). No vocabulary constraint is applied.
    public int[] selectRows(HostLogits logits, long firstPosition) {
        Objects.requireNonNull(logits, "logits");
        requireVocabulary(logits.vocabularySize());
        if (logits.selectsOnDevice()) {
            if (!greedy()) throw new IllegalStateException("device selection is greedy only");
            return logits.selectedTokens();
        }
        int[] tokens = new int[logits.rowCount()];
        for (int row = 0; row < tokens.length; row++) tokens[row] = selectRow(logits, row, firstPosition + row);
        return tokens;
    }

    /// Whether this sampler selects the argmax, which a device selection reproduces exactly.
    public boolean greedy() {
        return this.sampler.greedy();
    }

    /// Copies only the final vocabulary row and does not close or otherwise claim the logits allocation.
    public int selectToken(DeviceLogits logits, ExecutionGpu gpu) {
        return selectToken(logits, gpu, null);
    }

    /// Applies a request-local vocabulary constraint before temperature, top-k, or top-p sampling.
    public int selectToken(DeviceLogits logits, ExecutionGpu gpu, IntPredicate allowed) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(gpu, "gpu");
        requireVocabulary(logits.vocabularySize());
        long rowByteSize = Math.multiplyExact((long) logits.vocabularySize(), Short.BYTES);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment row = arena.allocate(rowByteSize, Short.BYTES);
            logits.copyFinalTokenRowToHost(gpu, row);
            return select(row, allowed);
        }
    }

    private void requireVocabulary(int vocabularySize) {
        if (vocabularySize != this.sampler.vocabularySize()) {
            throw new IllegalArgumentException(
                    "sampler expects " + this.sampler.vocabularySize() + " logits, got " + vocabularySize);
        }
    }

    /// Converts a BF16 row exactly to FP32 in the reusable scratch row; a constraint is tested only on the
    /// candidates selection needs ([TokenSampler#selectToken(float[], IntPredicate)]).
    private int select(MemorySegment row, IntPredicate allowed) {
        float[] hostLogits = this.scratch;
        convertBf16(row, hostLogits);
        return allowed == null ? this.sampler.selectToken(hostLogits) : this.sampler.selectToken(hostLogits, allowed);
    }

    private int select(MemorySegment row, IntPredicate allowed, long position) {
        float[] hostLogits = this.scratch;
        convertBf16(row, hostLogits);
        return allowed == null
                ? this.sampler.selectToken(hostLogits, position)
                : this.sampler.selectToken(hostLogits, allowed, position);
    }

    /// BF16 is the upper half of FP32, so each value converts exactly by a shift. The row is read two
    /// values per load; it is read once, from memory that the device just wrote.
    static void convertBf16(MemorySegment row, float[] out) {
        int pairs = out.length & ~1;
        for (int tokenId = 0; tokenId < pairs; tokenId += 2) {
            int pair = row.get(BF16_PAIR, (long) tokenId * Short.BYTES);
            out[tokenId] = Float.intBitsToFloat(pair << 16);
            out[tokenId + 1] = Float.intBitsToFloat(pair & 0xFFFF0000);
        }
        if (pairs != out.length) {
            out[pairs] = Float.intBitsToFloat(Short.toUnsignedInt(row.get(BF16, (long) pairs * Short.BYTES)) << 16);
        }
    }
}
