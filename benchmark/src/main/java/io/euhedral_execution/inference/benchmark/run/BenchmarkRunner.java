package io.euhedral_execution.inference.benchmark.run;

import com.fasterxml.jackson.databind.JsonNode;
import io.euhedral_execution.inference.benchmark.config.BenchmarkOptions;
import io.euhedral_execution.inference.benchmark.config.Scenario;
import io.euhedral_execution.inference.benchmark.measure.IterationTiming;
import io.euhedral_execution.inference.benchmark.measure.Metrics;
import io.euhedral_execution.inference.benchmark.prompt.PromptMaterial;
import io.euhedral_execution.inference.benchmark.result.BenchmarkResult;
import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.InferenceRunSnapshot;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.io.IOException;
import java.time.Clock;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/// Loads one engine, then runs each scenario's warmup rows followed by its measured rows. Every
/// iteration uses a fresh session and sequence.
public final class BenchmarkRunner {
    private BenchmarkRunner() {}

    /// A loaded engine. Implementations must run [#generate] in a new session and call
    /// `markEntry`/`markReturn` immediately around the session's generate call, outside session
    /// creation and close.
    public interface Target extends AutoCloseable {
        InferenceRunSnapshot snapshot(GenerationConfig generation);

        List<Integer> generate(String prompt, int maxNewTokens, GenerationConfig generation, IterationTiming timing)
                throws Exception;

        boolean isEos(int tokenId);

        /// Prompt tokens per prefill quantum of the target's model.
        default int prefillChunkTokens() {
            return InferenceConfig.PREFILL_CHUNK_TOKENS;
        }

        /// Returns `{freeBytes, totalBytes}` for the device.
        /// {free, total, allocated, peak allocated, retained workspace} device bytes.
        long[] memory();

        /// Restarts the peak of the allocated bytes [#memory] reports.
        default void resetPeakMemory() {}

        @Override
        void close();
    }

    public interface TargetFactory {
        Target open() throws IOException;
    }

    public record Context(
            String runId, BenchmarkResult.Fork fork, BenchmarkResult.Provenance provenance, Clock clock) {}

    public static void run(
            BenchmarkOptions options,
            Map<Scenario, List<PromptMaterial>> prompts,
            TargetFactory factory,
            Context context,
            Consumer<BenchmarkResult> sink)
            throws IOException, InterruptedException {
        try (Target target = factory.open()) {
            // Parse the serialized snapshot so the row holds exactly what is stored (e.g. floats as JSON numbers).
            JsonNode engine = BenchmarkResult.JSON.readTree(
                    target.snapshot(options.generation().toConfig()).toJson());
            for (Scenario scenario : options.scenarios()) {
                List<PromptMaterial> scenarioPrompts = prompts.get(scenario);
                if (scenarioPrompts == null || scenarioPrompts.isEmpty())
                    throw new IllegalStateException("no prompt prepared for " + scenario.name());
                for (int index = 0; index < options.warmup(); index++)
                    sink.accept(iteration(
                            options,
                            scenario,
                            scenarioPrompts.get(index % scenarioPrompts.size()),
                            target,
                            engine,
                            context,
                            true,
                            index));
                for (int index = 0; index < options.iterations(); index++)
                    sink.accept(iteration(
                            options,
                            scenario,
                            scenarioPrompts.get(index % scenarioPrompts.size()),
                            target,
                            engine,
                            context,
                            false,
                            index));
            }
        }
    }

    static BenchmarkResult iteration(
            BenchmarkOptions options,
            Scenario scenario,
            PromptMaterial prompt,
            Target target,
            JsonNode engine,
            Context context,
            boolean warmup,
            int index)
            throws InterruptedException {
        if (options.gpuMemory()) target.resetPeakMemory();
        long[] before = options.gpuMemory() ? target.memory() : null;
        var timing = new IterationTiming(
                Math.ceilDiv(prompt.actualTokens(), target.prefillChunkTokens()), scenario.requestedNewTokens());
        List<Integer> tokens = null;
        Throwable failure = null;
        try {
            tokens = target.generate(
                    prompt.text(),
                    scenario.requestedNewTokens(),
                    options.generation().toConfig(),
                    timing);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw interrupted;
        } catch (Exception executionFailure) {
            failure = executionFailure;
        }
        long[] after = options.gpuMemory() ? target.memory() : null;
        Metrics.Outcome outcome = Metrics.evaluate(scenario, timing, tokens, target::isEos, failure);
        if (!BenchmarkResult.FAILED.equals(outcome.status())
                && !java.util.Objects.equals(outcome.work().promptTokens(), prompt.actualTokens()))
            outcome = new Metrics.Outcome(
                    BenchmarkResult.FAILED, "prompt_token_count_mismatch", outcome.work(), outcome.timings(), null);
        return new BenchmarkResult(
                BenchmarkResult.SCHEMA,
                BenchmarkResult.SCHEMA_VERSION,
                BenchmarkResult.IMPLEMENTATION,
                context.provenance(),
                context.runId(),
                context.fork(),
                context.clock().instant().toString(),
                new BenchmarkResult.ScenarioRecord(
                        scenario.name(),
                        scenario.kind().label(),
                        scenario.targetPromptTokens(),
                        scenario.requestedNewTokens(),
                        prompt.generator(),
                        options.promptSeed(),
                        prompt.sha256()),
                warmup,
                index,
                outcome.status(),
                outcome.reason(),
                outcome.work(),
                outcome.timings(),
                outcome.throughput(),
                engine,
                before == null
                        ? null
                        : new BenchmarkResult.GpuMemory(
                                before[0],
                                after[0],
                                before[1],
                                at(before, 2),
                                at(after, 3),
                                at(after, 2),
                                at(after, 4)));
    }

    private static Long at(long[] values, int index) {
        return values.length > index ? values[index] : null;
    }
}
