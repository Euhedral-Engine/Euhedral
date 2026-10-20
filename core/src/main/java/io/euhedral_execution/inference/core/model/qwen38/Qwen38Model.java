package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightStaging;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.loader.ObjectLoader;
import io.euhedral_execution.inference.core.model.qwen38.loader.WeightLoader;
import io.euhedral_execution.inference.core.model.qwen38.loader.Weights;
import io.euhedral_execution.inference.core.model.qwen38.offload.DiskWindow;
import io.euhedral_execution.inference.core.state.KvHostPages;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// Owns the device allocations made by model loading.
/// The GPU is borrowed and must outlive this model. Close only after all model executions stop.
public final class Qwen38Model implements AutoCloseable {
    private static final Logger LOG = LoggerFactory.getLogger(Qwen38Model.class);

    private final Allocations allocations;
    private final Weights weights;
    private final WeightStaging staging;
    /// The pinned window disk-backed weights are read into, and the artifact they are read from; null without any.
    private final DiskWindow window;
    private final Path artifactPath;
    /// The host pages the sequences' attention caches keep past their device share, or null; the model owns them.
    private volatile KvHostPages kvHostPages;

    private Qwen38Model(
            Allocations allocations, Weights weights, WeightStaging staging, DiskWindow window, Path artifactPath) {
        this.allocations = allocations;
        this.weights = Objects.requireNonNull(weights);
        this.staging = staging;
        this.window = window;
        this.artifactPath = artifactPath;
    }

    /// Loads every object the artifact's text generation executes onto the device.
    public static Qwen38Model load(Path path, Artifact artifact, GpuMemory gpu) throws IOException {
        return load(path, artifact, gpu, ArtifactProfile.Speculation.NONE, Set.of());
    }

    /// Loads the objects named in `hostBacked` into pinned host memory, and allocates a device staging
    /// ring of [WeightStaging#SLOTS] slots, each holding the largest of them. `speculation` also
    /// loads its drafter. The model owns both.
    public static Qwen38Model load(
            Path path,
            Artifact artifact,
            GpuMemory gpu,
            ArtifactProfile.Speculation speculation,
            Set<String> hostBacked)
            throws IOException {
        return load(path, artifact, gpu, speculation, hostBacked, Set.of());
    }

    /// As above, leaving the objects named in `diskBacked` in the artifact file; the staging ring holds the largest
    /// host-backed or disk-backed object.
    public static Qwen38Model load(
            Path path,
            Artifact artifact,
            GpuMemory gpu,
            ArtifactProfile.Speculation speculation,
            Set<String> hostBacked,
            Set<String> diskBacked)
            throws IOException {
        return load(path, artifact, gpu, speculation, hostBacked, diskBacked, ObjectLoader.OWNED);
    }

    /// As above, placing each device-resident object with `objects`: the model frees only what it allocated itself,
    /// so objects `objects` hands out as borrowed outlive it and stay their owner's to free.
    public static Qwen38Model load(
            Path path,
            Artifact artifact,
            GpuMemory gpu,
            ArtifactProfile.Speculation speculation,
            Set<String> hostBacked,
            Set<String> diskBacked,
            ObjectLoader objects)
            throws IOException {
        return load(
                gpu,
                memory -> WeightLoader.load(path, artifact, memory, speculation, hostBacked, diskBacked, objects),
                hostBacked.isEmpty() && diskBacked.isEmpty() ? 0 : WeightStaging.SLOTS,
                path);
    }

    /// The pinned window disk-backed weights are read into, or null when none is disk-backed. The model owns it.
    public DiskWindow window() {
        return this.window;
    }

    /// The artifact disk-backed weights are read from, or null when none is disk-backed.
    public Path artifactPath() {
        return this.artifactPath;
    }

    /// The device staging ring for host-backed weights, or null when every weight is resident.
    public WeightStaging staging() {
        return this.staging;
    }

    /// The host pages of the sequences' attention caches, or null when every page is on the device.
    public KvHostPages kvHostPages() {
        return this.kvHostPages;
    }

    /// Gives the model the host pages its sequences' caches keep past their device share; it closes them. Set once,
    /// before any sequence runs.
    public void useKvHostPages(KvHostPages pages) {
        if (this.kvHostPages != null) throw new IllegalStateException("the model already has host attention pages");
        this.kvHostPages = Objects.requireNonNull(pages, "pages");
    }

    public static Qwen38Model loadFirstLayer(Path path, Artifact artifact, GpuMemory gpu) throws IOException {
        return load(gpu, memory -> WeightLoader.loadFirstLayer(path, artifact, memory));
    }

    public Weights weights() {
        return this.weights;
    }

    @FunctionalInterface
    public interface Loader {
        Weights load(GpuMemory memory) throws IOException;
    }

    public static Qwen38Model load(GpuMemory memory, Loader loader) throws IOException {
        return load(memory, loader, 0, null);
    }

    private static Qwen38Model load(GpuMemory memory, Loader loader, int stagingSlots, Path path) throws IOException {
        var allocations = new Allocations(Objects.requireNonNull(memory));
        try {
            Weights weights = loader.load(allocations);
            if (memory instanceof ExecutionGpu gpu
                    && weights.runtimeObjects().values().stream()
                            .anyMatch(handle -> handle.format() == WeightFormat.TRELLIS))
                LOG.info("trellis weights: {}", gpu.trellisSettings());
            long largest = 0;
            for (var handle : weights.runtimeObjects().values())
                if (handle.hostBacked() || handle.diskBacked()) largest = Math.max(largest, handle.byteSize());
            WeightStaging staging = null;
            if (largest > 0) {
                long slotBytes = WeightStaging.slotBytesFor(largest);
                staging = new WeightStaging(
                        allocations.allocate(Math.multiplyExact(slotBytes, stagingSlots)), slotBytes, stagingSlots);
            }
            // The window holds the largest disk-backed weight in each slot, with the blocks around it.
            long largestDisk = 0;
            for (var handle : weights.diskBacked()) largestDisk = Math.max(largestDisk, handle.byteSize());
            DiskWindow window =
                    largestDisk == 0 ? null : new DiskWindow(allocations, DiskWindow.slotBytesFor(largestDisk));
            return new Qwen38Model(allocations, weights, staging, window, window == null ? null : path);
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
        try {
            if (this.kvHostPages != null) this.kvHostPages.close();
        } finally {
            this.allocations.close();
        }
    }

    private static final class Allocations implements GpuMemory, AutoCloseable {
        private final GpuMemory gpu;
        private final LinkedHashSet<Long> live = new LinkedHashSet<>();
        private final LinkedHashSet<Long> hostLive = new LinkedHashSet<>();

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

        public long allocateHostWeights(long bytes) {
            long address = this.gpu.allocateHostWeights(bytes);
            this.hostLive.add(address);
            return address;
        }

        public void freeHostWeights(long address) {
            this.gpu.freeHostWeights(address);
            this.hostLive.remove(address);
        }

        public long hostWeightsDeviceAddress(long hostAddress) {
            return this.gpu.hostWeightsDeviceAddress(hostAddress);
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
            for (long address : this.hostLive.reversed().stream().toList()) {
                try {
                    freeHostWeights(address);
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
