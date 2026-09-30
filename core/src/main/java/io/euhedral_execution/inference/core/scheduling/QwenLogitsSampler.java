package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.sampling.TokenSampler;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.function.IntPredicate;

/// Bridges Qwen BF16 logits to the independent host-side token sampler.
/// Keep one instance per generation so its seeded random state and scratch row are request-local.
public final class QwenLogitsSampler {

    private final TokenSampler sampler;
    private final float[] scratch;

    public QwenLogitsSampler(GenerationConfig config, int vocabularySize) {
        this.sampler = new TokenSampler(config, vocabularySize);
        this.scratch = new float[vocabularySize];
    }

    /// Samples the final logits row that a successful quantum copied into `logits`.
    public int selectToken(QwenHostLogits logits, IntPredicate allowed) {
        Objects.requireNonNull(logits, "logits");
        requireVocabulary(logits.vocabularySize());
        return select(logits.row(), allowed);
    }

    /// Copies only the final vocabulary row and does not close or otherwise claim the logits allocation.
    public int selectToken(QwenDeviceLogits logits, ExecutionGpu gpu) {
        return selectToken(logits, gpu, null);
    }

    /// Applies a request-local vocabulary constraint before temperature, top-k, or top-p sampling.
    public int selectToken(QwenDeviceLogits logits, ExecutionGpu gpu, IntPredicate allowed) {
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

    /// Converts a BF16 row exactly to FP32 in the reusable scratch row, masking disallowed tokens.
    private int select(MemorySegment row, IntPredicate allowed) {
        float[] hostLogits = this.scratch;
        for (int tokenId = 0; tokenId < hostLogits.length; tokenId++) {
            short bf16Bits = row.get(ValueLayout.JAVA_SHORT, (long) tokenId * Short.BYTES);
            hostLogits[tokenId] = allowed != null && !allowed.test(tokenId)
                    ? Float.NEGATIVE_INFINITY
                    : Float.intBitsToFloat(Short.toUnsignedInt(bf16Bits) << 16);
        }
        return this.sampler.selectToken(hostLogits);
    }
}
