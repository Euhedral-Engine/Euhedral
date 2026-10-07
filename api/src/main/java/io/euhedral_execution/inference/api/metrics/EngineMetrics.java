package io.euhedral_execution.inference.api.metrics;

import io.euhedral_execution.inference.core.InferenceEngine;
import io.euhedral_execution.inference.core.prefix.PrefixCacheStats;
import io.micrometer.core.instrument.FunctionCounter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.binder.MeterBinder;
import java.util.function.ToDoubleFunction;

/// The engine's and the prefix cache's state, read from the engine at each scrape: nothing is recorded on the
/// engine's paths for them.
public final class EngineMetrics implements MeterBinder {
    private final InferenceEngine engine;
    private final String modelId;
    private final int contextTokens;

    public EngineMetrics(InferenceEngine engine, String modelId, int contextTokens) {
        this.engine = engine;
        this.modelId = modelId;
        this.contextTokens = contextTokens;
    }

    @Override
    public void bindTo(MeterRegistry registry) {
        InferenceEngine engine = this.engine;
        Gauge.builder("euhedral.engine.info", () -> 1)
                .description("The served model and artifact")
                .tag("model", this.modelId)
                .tag(
                        "artifact",
                        engine.description().artifactName() == null
                                ? "unknown"
                                : engine.description().artifactName())
                .tag("speculation", engine.description().speculation())
                .tag("speculative_depth", Integer.toString(engine.description().speculativeDepth()))
                .register(registry);
        Gauge.builder("euhedral.engine.context.tokens", () -> this.contextTokens)
                .description("Longest prompt plus completion a request may use")
                .register(registry);
        Gauge.builder(
                        "euhedral.engine.workers",
                        () -> engine.config().workerCpus().cardinality())
                .description("Worker CPUs of the engine's lattice")
                .register(registry);
        Gauge.builder("euhedral.engine.device.allocated.bytes", engine, InferenceEngine::allocatedDeviceBytes)
                .description("Device memory the engine allocated: weights, KV cache, sequence state, workspaces")
                .baseUnit("bytes")
                .register(registry);
        Gauge.builder(
                        "euhedral.engine.device.free.bytes",
                        engine,
                        value -> value.deviceMemoryInfo().freeBytes())
                .description("Free device memory reported by the driver")
                .baseUnit("bytes")
                .register(registry);
        Gauge.builder("euhedral.engine.host.backed.weight.bytes", engine, InferenceEngine::hostBackedWeightBytes)
                .description("Weights held in pinned host memory and streamed to the device")
                .baseUnit("bytes")
                .register(registry);
        if (engine.prefixCacheStats() == null) return;
        cacheCounter(registry, "euhedral.prefix.lookups", "Prefix cache lookups", PrefixCacheStats::lookups);
        cacheCounter(
                registry,
                "euhedral.prefix.hits",
                "Prefix cache lookups that restored a prefix",
                PrefixCacheStats::hits);
        cacheCounter(
                registry,
                "euhedral.prefix.restored.tokens",
                "Prompt tokens restored instead of prefilled",
                PrefixCacheStats::reusedTokens);
        cacheCounter(registry, "euhedral.prefix.captures", "Checkpoints stored", PrefixCacheStats::captured);
        cacheCounter(
                registry,
                "euhedral.prefix.skipped",
                "Checkpoints not stored for lack of room",
                PrefixCacheStats::skipped);
        cacheCounter(registry, "euhedral.prefix.failed", "Checkpoints whose copies failed", PrefixCacheStats::failed);
        cacheCounter(registry, "euhedral.prefix.evictions", "Checkpoints evicted", PrefixCacheStats::evictions);
        cacheCounter(registry, "euhedral.prefix.restores", "Restores completed", PrefixCacheStats::restores);
        cacheCounter(
                registry,
                "euhedral.prefix.capture.seconds",
                "Time spent storing checkpoints",
                stats -> stats.captureNanos() / 1e9);
        cacheCounter(
                registry,
                "euhedral.prefix.restore.seconds",
                "Time spent restoring prefixes",
                stats -> stats.restoreNanos() / 1e9);
        cacheGauge(registry, "euhedral.prefix.nodes", "Checkpoints held", PrefixCacheStats::nodes);
        cacheGauge(registry, "euhedral.prefix.used.bytes", "Pinned arena bytes in use", PrefixCacheStats::usedBytes);
        cacheGauge(registry, "euhedral.prefix.capacity.bytes", "Pinned arena bytes", PrefixCacheStats::totalBytes);
    }

    private void cacheCounter(
            MeterRegistry registry, String name, String description, ToDoubleFunction<PrefixCacheStats> value) {
        FunctionCounter.builder(name, this.engine, engine -> value.applyAsDouble(engine.prefixCacheStats()))
                .description(description)
                .register(registry);
    }

    private void cacheGauge(
            MeterRegistry registry, String name, String description, ToDoubleFunction<PrefixCacheStats> value) {
        Gauge.builder(name, this.engine, engine -> value.applyAsDouble(engine.prefixCacheStats()))
                .description(description)
                .register(registry);
    }
}
