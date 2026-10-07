package io.euhedral_execution.inference.core.prefix;

import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;

/// Runs a piece of prefix-cache host work as one frame and completes with its result.
public interface PrefixFrames {
    <T> CompletableFuture<T> run(Supplier<T> work);
}
