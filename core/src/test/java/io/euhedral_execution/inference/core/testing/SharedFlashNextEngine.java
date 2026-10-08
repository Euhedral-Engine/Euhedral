package io.euhedral_execution.inference.core.testing;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.InferenceEngine;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;

/// The Flash-Next text engine (one worker on CPU 0, the artifact's tokenizer) that CUDA tests of one JVM share, in the
/// [ModelGroup#FLASH_NEXT_ENGINE] scope.
public final class SharedFlashNextEngine {
    private SharedFlashNextEngine() {}

    private static final class Holder implements AutoCloseable {
        private InferenceEngine engine;
        private int context;

        synchronized InferenceEngine get(int context) {
            if (this.engine != null && (this.context != context || this.engine.isClosed())) close();
            if (this.engine == null) {
                try {
                    this.engine = InferenceEngine.load(config(context));
                } catch (java.io.IOException failure) {
                    throw new java.io.UncheckedIOException(failure);
                }
                this.context = context;
            }
            return this.engine;
        }

        @Override
        public synchronized void close() {
            InferenceEngine closing = this.engine;
            this.engine = null;
            if (closing != null) closing.close();
        }
    }

    /// The engine over the artifact for `context` tokens, loaded on first use (and again when a test closed it, as the
    /// tests that check closing do; they run last).
    public static InferenceEngine get(int context) {
        config(context); // assumptions first, before the registry opens anything
        return Shared.get(ModelGroup.FLASH_NEXT_ENGINE, "flash-next-engine", Holder::new)
                .get(context);
    }

    private static InferenceConfig config(int context) {
        Path artifact = Path.of(System.getProperty(
                "euhedral.qwen4.artifact", "/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl"));
        assumeTrue(Files.isRegularFile(artifact), "no Flash-Next artifact at " + artifact);
        Path tokenizers =
                Path.of(System.getProperty("euhedral.qwen4.tokenizer-dir", "/mnt/shared/qwen38-flash-next/nvfp4"));
        assumeTrue(Files.isRegularFile(tokenizers.resolve("tokenizer.json")), "no tokenizer at " + tokenizers);
        BitSet cpus = new BitSet();
        cpus.set(0);
        return new InferenceConfig(
                artifact,
                tokenizers,
                Path.of(System.getProperty("euhedral.cuda.library")),
                cpus,
                context,
                Duration.ofSeconds(30));
    }
}
