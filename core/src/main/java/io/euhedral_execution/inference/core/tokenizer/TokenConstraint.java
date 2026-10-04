package io.euhedral_execution.inference.core.tokenizer;

import io.euhedral_execution.inference.core.sampling.TokenFilter;

/// A request-local restriction on which tokens generation may select next.
///
/// `allows` is queried for candidate tokens and must not change state; `accept` commits the token sampling
/// selected. Both run on the worker that selects the token, one generation step after another. A constraint
/// holding native state frees it in `close`, after the generation ended.
public interface TokenConstraint extends TokenFilter, AutoCloseable {

    boolean allows(int tokenId);

    /// Commits a selected token; throws when the token was not allowed.
    void accept(int tokenId);

    @Override
    default boolean test(int tokenId) {
        return allows(tokenId);
    }

    @Override
    default void close() {}
}
