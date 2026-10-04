package io.euhedral_execution.inference.core.tokenizer;

import java.util.function.IntPredicate;

/// A request-local restriction on which tokens generation may select next.
///
/// `allows` is queried for candidate tokens and must not change state; `accept` commits the token sampling
/// selected. Both run on the worker that selects the token, one generation step after another.
public interface TokenConstraint extends IntPredicate {

    boolean allows(int tokenId);

    /// Commits a selected token; throws when the token was not allowed.
    void accept(int tokenId);

    @Override
    default boolean test(int tokenId) {
        return allows(tokenId);
    }
}
