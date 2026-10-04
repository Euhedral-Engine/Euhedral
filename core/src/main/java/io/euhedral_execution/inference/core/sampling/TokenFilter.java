package io.euhedral_execution.inference.core.sampling;

import java.util.function.IntPredicate;

/// Which tokens a selection may choose. `test` answers for one token; [#maskDisallowed] for the whole row, which
/// an implementation that knows its allowed set at once can do without testing every token.
public interface TokenFilter extends IntPredicate {

    /// Overwrites every disallowed entry of `logits` with negative infinity.
    default void maskDisallowed(float[] logits) {
        for (int tokenId = 0; tokenId < logits.length; tokenId++)
            if (!test(tokenId)) logits[tokenId] = Float.NEGATIVE_INFINITY;
    }
}
