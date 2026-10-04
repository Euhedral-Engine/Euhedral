package io.euhedral_execution.inference.api.metrics;

import io.euhedral_execution.inference.api.chat.Finish;
import io.euhedral_execution.inference.api.chat.TokenUsage;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.DistributionSummary;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import java.time.Duration;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import org.springframework.stereotype.Component;

/// The server's request metrics, exported at `/metrics` (Prometheus text format, `euhedral_` prefix).
///
/// Everything is recorded once per request, when it is refused or ends, or once per speculative verification
/// from the session's timing callbacks; nothing is recorded per token. Meters are created on first use and cached,
/// so recording allocates nothing once a label combination has been seen.
@Component
public class ServerMetrics {
    private final MeterRegistry registry;
    private final ConcurrentHashMap<String, Counter> counters = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Timer> timers = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, DistributionSummary> summaries = new ConcurrentHashMap<>();

    public ServerMetrics(MeterRegistry registry) {
        this.registry = registry;
    }

    /// Why a request ended without a generation's answer.
    public enum Outcome {
        /// Answered (any finish reason).
        SUCCESS,
        /// Refused while planning: invalid, unsupported, or over the context.
        INVALID,
        /// Refused for capacity or shutdown.
        REJECTED,
        /// The client left or the request timed out.
        CANCELLED,
        /// The server or the model's output failed.
        FAILED
    }

    /// Gauges of the generation slot and queue, read at each scrape.
    public void bindQueue(Supplier<Number> active, Supplier<Number> queued) {
        Gauge.builder("euhedral.generations.active", active)
                .description("Generations running")
                .register(this.registry);
        Gauge.builder("euhedral.generations.queued", queued)
                .description("Requests waiting for the generation slot")
                .register(this.registry);
    }

    public void request(String api, Outcome outcome) {
        counter("euhedral.requests", "Requests by API surface and outcome", "api", api, "outcome", name(outcome))
                .increment();
    }

    /// A request whose JSON schema or grammar could not be enforced.
    public void schemaRejected(String api) {
        counter("euhedral.schema.rejections", "Requests refused because a schema cannot be enforced", "api", api)
                .increment();
    }

    /// A finished output that failed its constraint's check (`invalid_tool_call`, `invalid_structured_output`).
    public void constrainedFailure(String reason) {
        counter("euhedral.constrained.failures", "Constrained outputs that failed their final check", "reason", reason)
                .increment();
    }

    public void toolCalls(String api, int calls) {
        if (calls > 0)
            counter("euhedral.tool.calls", "Tool calls returned", "api", api).increment(calls);
    }

    /// One answered generation: its finish, token counts and timings. `firstTokenNanos` is 0 when it produced
    /// no token.
    public void generation(
            String api, Finish finish, long queuedNanos, long startedNanos, long firstTokenNanos, long endedNanos) {
        TokenUsage usage = finish.usage();
        counter("euhedral.finishes", "Generations by finish reason", "api", api, "reason", name(finish.reason()))
                .increment();
        counter("euhedral.prompt.tokens", "Prompt tokens").increment(usage.promptTokens());
        counter("euhedral.prompt.cached.tokens", "Prompt tokens restored from the prefix cache")
                .increment(usage.cachedPromptTokens());
        int prefilled = usage.promptTokens() - usage.cachedPromptTokens();
        counter("euhedral.prompt.prefilled.tokens", "Prompt tokens prefilled").increment(prefilled);
        counter("euhedral.completion.tokens", "Generated tokens, reasoning included")
                .increment(usage.completionTokens());
        if (usage.reasoningTokens() != null)
            counter("euhedral.reasoning.tokens", "Generated reasoning tokens").increment(usage.reasoningTokens());
        timer("euhedral.request.duration", "From arrival to the last token", api)
                .record(endedNanos - queuedNanos, TimeUnit.NANOSECONDS);
        timer("euhedral.queue.wait", "From arrival to the start of generation", api)
                .record(startedNanos - queuedNanos, TimeUnit.NANOSECONDS);
        if (firstTokenNanos == 0) return;
        long toFirst = firstTokenNanos - startedNanos;
        timer("euhedral.time.to.first.token", "From the start of generation to the first token", api)
                .record(toFirst, TimeUnit.NANOSECONDS);
        if (prefilled > 0 && toFirst > 0)
            summary("euhedral.prefill.tokens.per.second", "Prefilled tokens per second up to the first token")
                    .record(prefilled * 1e9 / toFirst);
        long decoding = endedNanos - firstTokenNanos;
        if (usage.completionTokens() > 1 && decoding > 0)
            summary("euhedral.decode.tokens.per.second", "Generated tokens per second after the first")
                    .record((usage.completionTokens() - 1) * 1e9 / decoding);
    }

    /// One speculative verification that accepted `accepted` of its drafted tokens.
    public void speculativeStep(int accepted) {
        counter("euhedral.speculative.verifications", "Speculative verification steps")
                .increment();
        counter(
                        "euhedral.speculative.accepted",
                        "Verification steps by the number of drafted tokens accepted",
                        "drafts",
                        Integer.toString(accepted))
                .increment();
        counter("euhedral.speculative.accepted.tokens", "Drafted tokens accepted")
                .increment(accepted);
    }

    private Counter counter(String name, String description, String... tags) {
        String key = name + String.join("\u0000", tags);
        Counter counter = this.counters.get(key);
        if (counter == null)
            counter = this.counters.computeIfAbsent(
                    key,
                    ignored -> Counter.builder(name)
                            .description(description)
                            .tags(tags)
                            .register(this.registry));
        return counter;
    }

    private Timer timer(String name, String description, String api) {
        String key = name + "\u0000" + api;
        Timer timer = this.timers.get(key);
        if (timer == null)
            timer = this.timers.computeIfAbsent(
                    key,
                    ignored -> Timer.builder(name)
                            .description(description)
                            .tag("api", api)
                            .publishPercentileHistogram()
                            .minimumExpectedValue(Duration.ofMillis(10))
                            .maximumExpectedValue(Duration.ofMinutes(10))
                            .register(this.registry));
        return timer;
    }

    private DistributionSummary summary(String name, String description) {
        DistributionSummary summary = this.summaries.get(name);
        if (summary == null)
            summary = this.summaries.computeIfAbsent(
                    name,
                    ignored -> DistributionSummary.builder(name)
                            .description(description)
                            .publishPercentileHistogram()
                            .minimumExpectedValue(1.0)
                            .maximumExpectedValue(100_000.0)
                            .register(this.registry));
        return summary;
    }

    private static String name(Enum<?> value) {
        return value.name().toLowerCase(java.util.Locale.ROOT);
    }
}
