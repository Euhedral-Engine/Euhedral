package io.euhedral_execution.inference.core;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.testing.Shared;
import java.nio.file.Path;
import java.util.function.Supplier;

/// Inference engines that CUDA tests of one JVM share. An engine is kept for the tests that ask for the same key;
/// asking for another key closes it first, because two engines do not fit the device. Tests are ordered by the
/// engine they need (see [io.euhedral_execution.inference.core.testing.ModelGroup]), so a switch happens once.
///
/// A test reads what it changed (cache counters, allocated bytes) and does not assume a fresh engine.
final class SharedEngines {
    private SharedEngines() {}

    /// An engine, and the device handle it opened.
    record Held(InferenceEngine engine, RecordingBootstrap bootstrap) implements AutoCloseable {
        ExecutionGpu gpu() {
            return this.bootstrap.gpu;
        }

        @Override
        public void close() {
            this.engine.close();
        }
    }

    /// The engine kept under `key`, loaded from `config` on first use. A configuration that assumes away a test
    /// (a missing artifact) throws the assumption failure from the supplier.
    static Held get(String key, Supplier<InferenceConfig> config) {
        return Shared.get(key, key, () -> {
            var bootstrap = new RecordingBootstrap();
            return new Held(InferenceEngine.load(config.get(), bootstrap), bootstrap);
        });
    }

    /// A bootstrap that keeps the device handle the engine opens, for tests that read device state.
    static final class RecordingBootstrap extends InferenceEngine.Bootstrap {
        private volatile CudaGpuMemory gpu;

        @Override
        ExecutionGpu openGpu(Path path) {
            ExecutionGpu opened = super.openGpu(path);
            this.gpu = (CudaGpuMemory) opened;
            return opened;
        }
    }
}
