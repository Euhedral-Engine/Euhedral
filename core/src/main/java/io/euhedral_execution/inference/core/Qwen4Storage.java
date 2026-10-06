package io.euhedral_execution.inference.core;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.host.HostFrames;
import io.euhedral_execution.inference.core.model_loader.ModelArchitecture;
import io.euhedral_execution.inference.core.model_loader.qwen4.HostBudget;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Mode;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Model;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4ResidencyPlan;
import java.io.IOException;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The Flash-Next load the engine performs from the artifact and the maximum context alone: it reads and validates the
/// artifact, plans residency from the device's free memory and the context, loads the fixed objects and the host
/// stores, and builds the expert cache from what the plan leaves. It runs no model: execution for `qwen4_exp` builds on
/// the [Qwen4Model] this holds.
///
/// Nothing about placement is configured. The user chooses the artifact and the maximum context; the cache size, the
/// offload, the staging depth and the host stores are derived.
public final class Qwen4Storage implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(Qwen4Storage.class);

    /// The GPU and host the load runs on; a seam for tests.
    interface Environment {
        ExecutionGpu openGpu(InferenceConfig config);

        long freeDeviceBytes(ExecutionGpu gpu);

        void closeGpu(ExecutionGpu gpu);

        HostBudget hostBudget();
    }

    private static final Environment SYSTEM = new Environment() {
        @Override
        public ExecutionGpu openGpu(InferenceConfig config) {
            return new CudaGpuMemory(config.cudaLibraryPath());
        }

        @Override
        public long freeDeviceBytes(ExecutionGpu gpu) {
            return ((CudaGpuMemory) gpu).deviceMemoryInfo().freeBytes();
        }

        @Override
        public void closeGpu(ExecutionGpu gpu) {
            ((CudaGpuMemory) gpu).close();
        }

        @Override
        public HostBudget hostBudget() {
            return HostBudget.system();
        }
    };

    private final Environment environment;
    private final ExecutionGpu gpu;
    private final Qwen4Model model;
    private boolean closed;

    private Qwen4Storage(Environment environment, ExecutionGpu gpu, Qwen4Model model) {
        this.environment = environment;
        this.gpu = gpu;
        this.model = model;
    }

    /// Loads the `qwen4_exp` artifact of `config` for `config.maxContextTokens()`, selecting no MTP and no vision.
    public static Qwen4Storage load(InferenceConfig config, HostFrames frames) throws IOException {
        return load(config, SYSTEM, Qwen4Mode.TEXT, frames);
    }

    static Qwen4Storage load(InferenceConfig config, Environment environment, Qwen4Mode mode, HostFrames frames)
            throws IOException {
        Objects.requireNonNull(config, "config");
        ModelArchitecture architecture = ModelArchitecture.detect(config.artifactPath());
        if (architecture != ModelArchitecture.QWEN4_EXP)
            throw new IOException("the artifact is a " + architecture + " model, not " + ModelArchitecture.QWEN4_EXP);
        ExecutionGpu gpu = environment.openGpu(config);
        try {
            long free = environment.freeDeviceBytes(gpu);
            Qwen4Model model = Qwen4Model.open(
                    config.artifactPath(),
                    gpu,
                    free,
                    environment.hostBudget(),
                    mode,
                    config.maxContextTokens(),
                    frames);
            LOG.info(
                    "Flash-Next storage loaded: {} expert slots, {} MiB of fixed objects on the device",
                    model.plan().expertCache().slotCount(),
                    model.plan().device().fixedResidentBytes() >> 20);
            return new Qwen4Storage(environment, gpu, model);
        } catch (Throwable failure) {
            try {
                environment.closeGpu(gpu);
            } catch (RuntimeException cleanup) {
                failure.addSuppressed(cleanup);
            }
            if (failure instanceof IOException io) throw io;
            if (failure instanceof RuntimeException runtime) throw runtime;
            if (failure instanceof Error error) throw error;
            throw new IOException(failure);
        }
    }

    public Qwen4Model model() {
        return this.model;
    }

    public Qwen4ResidencyPlan plan() {
        return this.model.plan();
    }

    public ExecutionGpu gpu() {
        return this.gpu;
    }

    @Override
    public synchronized void close() {
        if (this.closed) return;
        this.closed = true;
        try {
            this.model.close();
        } finally {
            this.environment.closeGpu(this.gpu);
        }
    }
}
