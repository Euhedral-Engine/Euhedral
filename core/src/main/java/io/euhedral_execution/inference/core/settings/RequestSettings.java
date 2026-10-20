package io.euhedral_execution.inference.core.settings;

import java.time.Duration;
import java.util.Objects;

/// Limits on one generation request.
///
/// @param defaultMaxTokens tokens generated when a request names no limit
/// @param maxQueued generation requests waiting for the engine before new ones are refused; zero refuses any that
///     cannot start at once
/// @param maxBodySize largest request body, in bytes
/// @param timeout longest a request may run
public record RequestSettings(int defaultMaxTokens, int maxQueued, long maxBodySize, Duration timeout) {

    public RequestSettings {
        if (defaultMaxTokens < 1) throw new IllegalArgumentException("default-max-tokens must be positive");
        if (maxQueued < 0) throw new IllegalArgumentException("max-queued must not be negative");
        if (maxBodySize < 1) throw new IllegalArgumentException("max-body-size must be positive");
        Objects.requireNonNull(timeout, "timeout");
        if (timeout.isZero() || timeout.isNegative()) throw new IllegalArgumentException("timeout must be positive");
    }

    public static RequestSettings defaults() {
        return new RequestSettings(4096, 16, 1L << 20, Duration.ofMinutes(30));
    }
}
