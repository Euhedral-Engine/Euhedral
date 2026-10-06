package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.ModelArchitecture;
import io.euhedral_execution.inference.core.model_loader.qwen4.HostBudget;
import io.euhedral_execution.inference.core.model_loader.qwen4.HostMemoryGpu;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Mode;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4TestArtifact;
import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The engine-level entry: architecture detection and the load from an [InferenceConfig] alone.
class Qwen4StorageTest {

    @TempDir
    Path directory;

    static InferenceConfig config(Path artifact, int context) {
        BitSet cpus = new BitSet();
        cpus.set(0);
        return new InferenceConfig(
                artifact, Path.of("unused"), Path.of("unused"), cpus, context, Duration.ofSeconds(5));
    }

    /// A GPU of `freeBytes`, host memory of `pinnable` bytes.
    static final class FakeEnvironment implements Qwen4Storage.Environment {
        final HostMemoryGpu gpu = new HostMemoryGpu();
        final long free;
        boolean closed;

        FakeEnvironment(long free) {
            this.free = free;
        }

        @Override
        public ExecutionGpu openGpu(InferenceConfig config) {
            return this.gpu;
        }

        @Override
        public long freeDeviceBytes(ExecutionGpu gpu) {
            return this.free;
        }

        @Override
        public void closeGpu(ExecutionGpu gpu) {
            this.closed = true;
        }

        @Override
        public HostBudget hostBudget() {
            return HostBudget.ofAvailable(64L << 30);
        }
    }

    @Test
    void detectsTheArchitectureFromTheHeader() throws IOException {
        Path path = this.directory.resolve("mini.edrl");
        Qwen4TestArtifact.write(path, Qwen4TestArtifact.miniConfig(), 1);
        assertEquals(ModelArchitecture.QWEN4_EXP, ModelArchitecture.detect(path));
    }

    @Test
    void loadsFromTheConfigurationAloneAndClosesEverything() throws IOException {
        Path path = this.directory.resolve("mini.edrl");
        Qwen4TestArtifact.write(path, Qwen4TestArtifact.miniConfig(), 1);
        var environment = new FakeEnvironment(4L << 30);
        try (Qwen4Storage storage = Qwen4Storage.load(config(path, 2048), environment, Qwen4Mode.TEXT)) {
            assertTrue(storage.plan().fits());
            assertEquals(2048, storage.plan().maxContextTokens());
            assertTrue(storage.model().expertCache().slotCount() > 0);
            assertTrue(environment.gpu.liveDeviceBytes() > 0);
        }
        assertTrue(environment.closed);
        assertEquals(0, environment.gpu.liveDeviceAllocations());
        assertEquals(0, environment.gpu.livePinnedAllocations());
    }

    @Test
    void anImpossibleContextClosesTheGpuAndExplains() throws IOException {
        Path path = this.directory.resolve("mini.edrl");
        Qwen4TestArtifact.write(path, Qwen4TestArtifact.miniConfig(), 1);
        var environment = new FakeEnvironment(256L << 20);
        var failure = assertThrows(
                IOException.class, () -> Qwen4Storage.load(config(path, 2048), environment, Qwen4Mode.TEXT));
        assertTrue(failure.getMessage().contains("cannot be placed"), failure.getMessage());
        assertTrue(environment.closed);
        assertEquals(0, environment.gpu.liveDeviceAllocations());
    }
}
