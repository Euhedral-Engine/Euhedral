package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.Objects;

/// Owns the device allocations made by model loading, including non-compact artifacts.
/// The GPU is borrowed and must outlive this model. Close only after all model executions stop.
public final class QwenModel implements AutoCloseable {
    private final Allocations allocations;
    private final QwenWeights weights;

    private QwenModel(Allocations allocations, QwenWeights weights) {
        this.allocations = allocations;
        this.weights = Objects.requireNonNull(weights);
    }

    public static QwenModel load(Path path, QwenArtifact artifact, GpuMemory gpu) throws IOException {
        return load(path, artifact, gpu, WeightResidency.ALL);
    }

    public static QwenModel load(Path path, QwenArtifact artifact, GpuMemory gpu, WeightResidency residency)
            throws IOException {
        return load(gpu, memory -> QwenWeightLoader.load(path, artifact, memory, residency));
    }

    public static QwenModel loadFirstLayer(Path path, QwenArtifact artifact, GpuMemory gpu) throws IOException {
        return load(gpu, memory -> QwenWeightLoader.loadFirstLayer(path, artifact, memory));
    }

    public QwenWeights weights() {
        return this.weights;
    }

    @FunctionalInterface
    interface Loader {
        QwenWeights load(GpuMemory memory) throws IOException;
    }

    static QwenModel load(GpuMemory memory, Loader loader) throws IOException {
        var allocations = new Allocations(Objects.requireNonNull(memory));
        try {
            return new QwenModel(allocations, loader.load(allocations));
        } catch (IOException | RuntimeException | Error failure) {
            try {
                allocations.close();
            } catch (RuntimeException | Error cleanup) {
                throw new LoadFailure(failure, cleanup, allocations);
            }
            throw failure;
        }
    }

    /// A failed upload whose device cleanup also failed. Keep the borrowed GPU open and retry close.
    public static final class LoadFailure extends IOException implements AutoCloseable {
        private final Allocations allocations;

        private LoadFailure(Throwable failure, Throwable cleanup, Allocations allocations) {
            super("model loading failed and device allocations still require cleanup", failure);
            addSuppressed(cleanup);
            this.allocations = allocations;
        }

        @Override
        public void close() {
            this.allocations.close();
        }
    }

    @Override
    public void close() {
        this.allocations.close();
    }

    private static final class Allocations implements GpuMemory, AutoCloseable {
        private final GpuMemory gpu;
        private final LinkedHashSet<Long> live = new LinkedHashSet<>();

        private Allocations(GpuMemory gpu) {
            this.gpu = gpu;
        }

        public long allocate(long bytes) {
            long address = this.gpu.allocate(bytes);
            this.live.add(address);
            return address;
        }

        public void free(long address) {
            this.gpu.free(address);
            this.live.remove(address);
        }

        public void copyHostToDevice(long address, MemorySegment source, long bytes) {
            this.gpu.copyHostToDevice(address, source, bytes);
        }

        public void copyDeviceToHost(MemorySegment destination, long address, long bytes) {
            this.gpu.copyDeviceToHost(destination, address, bytes);
        }

        public void copyDeviceToDevice(long destination, long source, long bytes) {
            this.gpu.copyDeviceToDevice(destination, source, bytes);
        }

        public synchronized void close() {
            Throwable failure = null;
            for (long address : this.live.reversed().stream().toList()) {
                try {
                    free(address);
                } catch (RuntimeException | Error cleanup) {
                    if (failure == null) failure = cleanup;
                    else if (failure != cleanup) failure.addSuppressed(cleanup);
                }
            }
            if (failure instanceof RuntimeException exception) throw exception;
            if (failure instanceof Error error) throw error;
        }
    }
}
