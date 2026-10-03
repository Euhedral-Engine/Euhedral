package io.euhedral_execution.inference.benchmark.run;

import io.euhedral_execution.inference.benchmark.measure.IterationTiming;
import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.InferenceEngine;
import io.euhedral_execution.inference.core.InferenceRunSnapshot;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.List;
import java.util.function.Consumer;

/// End-to-end target: a real [InferenceEngine] running Qwen sessions through Euhedral and CUDA.
public final class EngineTarget implements BenchmarkRunner.Target {
    private static final Consumer<String> DISCARD = ignored -> {};

    private final InferenceEngine engine;

    private EngineTarget(InferenceEngine engine) {
        this.engine = engine;
    }

    public static BenchmarkRunner.TargetFactory factory(
            Path artifact,
            Path tokenizer,
            Path cudaLibrary,
            BitSet workerCpus,
            int maxContextTokens,
            Duration shutdown) {
        return () -> {
            try {
                var engine = InferenceEngine.load(
                        new InferenceConfig(artifact, tokenizer, cudaLibrary, workerCpus, maxContextTokens, shutdown));
                System.out.println("loaded: " + engine.allocatedDeviceBytes() + " device bytes, "
                        + engine.hostBackedWeightBytes() + " host-backed weight bytes");
                return new EngineTarget(engine);
            } catch (InferenceEngine.StartupFailure failure) {
                try {
                    failure.close();
                } catch (RuntimeException | Error cleanup) {
                    failure.addSuppressed(cleanup);
                }
                throw failure;
            }
        };
    }

    @Override
    public InferenceRunSnapshot snapshot(GenerationConfig generation) {
        return this.engine.snapshot(generation);
    }

    @Override
    public List<Integer> generate(String prompt, int maxNewTokens, GenerationConfig generation, IterationTiming timing)
            throws Exception {
        try (var session = this.engine.createSession(generation)) {
            timing.markEntry();
            List<Integer> tokens = session.generate(prompt, maxNewTokens, DISCARD, null, timing);
            timing.markReturn();
            return tokens;
        }
    }

    @Override
    public boolean isEos(int tokenId) {
        return this.engine.tokenizer().isGenerationEosToken(tokenId);
    }

    @Override
    public long[] memory() {
        var info = this.engine.deviceMemoryInfo();
        return new long[] {
            info.freeBytes(),
            info.totalBytes(),
            this.engine.allocatedDeviceBytes(),
            this.engine.peakAllocatedDeviceBytes(),
            this.engine.retainedWorkspaceBytes()
        };
    }

    @Override
    public void resetPeakMemory() {
        this.engine.resetPeakAllocatedDeviceBytes();
    }

    @Override
    public void close() {
        this.engine.close();
    }
}
