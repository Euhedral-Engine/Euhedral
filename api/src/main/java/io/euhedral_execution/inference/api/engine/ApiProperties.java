package io.euhedral_execution.inference.api.engine;

import java.time.Duration;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/// HTTP adaptation limits bound from `euhedral.api.*`.
///
/// `defaultMaxTokens` applies when a request sets neither `max_tokens` nor `max_completion_tokens`;
/// it is further capped by the remaining context. One generation runs at a time (the engine sizes device
/// memory for one sequence); generations beyond `maxQueuedGenerations` waiting are rejected with 503
/// rather than queued unbounded.
@ConfigurationProperties("euhedral.api")
public record ApiProperties(
        @DefaultValue("4096") int defaultMaxTokens,
        @DefaultValue("16") int maxQueuedGenerations,
        @DefaultValue("1048576") int maxRequestBytes,
        @DefaultValue("30m") Duration requestTimeout) {

    public ApiProperties {
        if (defaultMaxTokens <= 0)
            throw new IllegalArgumentException("euhedral.api.default-max-tokens must be positive");
        if (maxQueuedGenerations < 0)
            throw new IllegalArgumentException("euhedral.api.max-queued-generations must not be negative");
        if (maxRequestBytes <= 0) throw new IllegalArgumentException("euhedral.api.max-request-bytes must be positive");
        if (requestTimeout == null || requestTimeout.isNegative() || requestTimeout.isZero())
            throw new IllegalArgumentException("euhedral.api.request-timeout must be positive");
    }
}
