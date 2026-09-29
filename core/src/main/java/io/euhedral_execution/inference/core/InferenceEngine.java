package io.euhedral_execution.inference.core;

import io.euhedral_execution.core.config.LatticeConfig;
import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.control_plane.ControlPlaneShard;
import io.euhedral_execution.core.generics.AbstractExecutor;
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.scheduling.InferenceGpuExecutor;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenGenerationSession;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;

/// High-level owner of model, CUDA backend, Euhedral lattice, and all sessions it creates.
/// Close sessions early when finished; engine close also closes every tracked session.
/// This standalone application's fabric is process-wide. The engine controls its start/stop;
/// independent sessions and their sources share it. Only one engine may control that lifecycle.
/// Initialize the process fabric through load, not a separate low-level lattice factory; advanced
/// sources may borrow the running fabric but must not initialize or stop it independently.
public final class InferenceEngine implements AutoCloseable {
    private static final AtomicBoolean LATTICE_OWNED = new AtomicBoolean();
    private static final AtomicReference<ControlPlaneLattice> LAST_CLOSED_LATTICE = new AtomicReference<>();
    private static final AtomicLong SEQUENCE_IDS = new AtomicLong();
    private final Bootstrap bootstrap;
    private final QwenTokenizer tokenizer;
    private final ExecutionGpu gpu;
    private final QwenModel model;
    private final ControlPlaneLattice lattice;
    private final QwenExecutionPlan plan;
    private final EuhedralInferenceRuntime runtime;
    private final InferenceTuning tuning;
    private final BitSet workerCoreIds;
    private final InferenceRunSnapshot.Model modelIdentity;
    private final InferenceRunSnapshot.RuntimeIdentity runtimeIdentity;
    private final List<QwenGenerationSession> sessions = new ArrayList<>();
    private final ReentrantLock shutdownLock = new ReentrantLock();
    private volatile boolean closing;
    private boolean resourcesClosed;

    private InferenceEngine(
            Bootstrap bootstrap,
            QwenTokenizer tokenizer,
            ExecutionGpu gpu,
            QwenModel model,
            ControlPlaneLattice lattice,
            QwenExecutionPlan plan,
            EuhedralInferenceRuntime runtime,
            InferenceTuning tuning,
            BitSet workerCoreIds,
            InferenceRunSnapshot.Model modelIdentity,
            InferenceRunSnapshot.RuntimeIdentity runtimeIdentity) {
        this.bootstrap = bootstrap;
        this.tokenizer = tokenizer;
        this.gpu = gpu;
        this.model = model;
        this.lattice = lattice;
        this.plan = plan;
        this.runtime = runtime;
        this.tuning = tuning;
        this.workerCoreIds = workerCoreIds;
        this.modelIdentity = modelIdentity;
        this.runtimeIdentity = runtimeIdentity;
    }

    /// Loads all model resources and starts the lattice before returning an engine.
    /// The config's [InferenceTuning] is applied as given; worker processor IDs are checked against
    /// the host [ProcessorTopology] before the tokenizer, model, or GPU is loaded.
    public static InferenceEngine load(InferenceConfig config) throws IOException {
        return load(config, new Bootstrap());
    }

    static InferenceEngine load(InferenceConfig config, Bootstrap bootstrap) throws IOException {
        Objects.requireNonNull(config, "config");
        InferenceTuning tuning = config.tuning();
        // Euhedral silently drops unavailable CPUs; fail before claiming the lattice or loading anything.
        ProcessorTopology topology = bootstrap.processorTopology();
        topology.requireAvailable(tuning.workerProcessorIds());
        BitSet workerCoreIds = topology.coreIds(tuning.workerProcessorIds());
        if (!LATTICE_OWNED.compareAndSet(false, true))
            throw new IllegalStateException("an inference engine already owns the process-wide Euhedral lattice");
        ExecutionGpu gpu = null;
        QwenModel model = null;
        ControlPlaneLattice lattice = null;
        try {
            QwenTokenizer tokenizer = QwenTokenizer.load(config.tokenizerDirectory());
            QwenArtifact artifact = bootstrap.readArtifact(config.artifactPath());
            gpu = bootstrap.openGpu(config.cudaLibraryPath(), tuning);
            model = bootstrap.loadModel(config.artifactPath(), artifact, gpu);
            QwenExecutionPlan plan = new QwenExecutionPlan(model.weights());
            lattice = bootstrap.createLattice(config, gpu);
            bootstrap.startLattice(lattice);
            EuhedralInferenceRuntime runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
            return new InferenceEngine(
                    bootstrap,
                    tokenizer,
                    gpu,
                    model,
                    lattice,
                    plan,
                    runtime,
                    tuning,
                    workerCoreIds,
                    modelIdentity(config.artifactPath(), artifact, model),
                    runtimeIdentity(config.cudaLibraryPath()));
        } catch (IOException | RuntimeException | Error failure) {
            var pending = new StartupFailure(
                    failure,
                    bootstrap,
                    gpu,
                    failure instanceof QwenModel.LoadFailure partial ? partial : model,
                    lattice);
            try {
                pending.close();
            } catch (RuntimeException | Error cleanup) {
                pending.addSuppressed(cleanup);
                throw pending;
            }
            throw failure;
        }
    }

    /// Startup and its rollback both failed. This exception retains ownership of unreleased resources;
    /// close it again to retry cleanup before loading another engine. No session has been admitted.
    public static final class StartupFailure extends IOException implements AutoCloseable {
        private final Bootstrap bootstrap;
        private ExecutionGpu gpu;
        private AutoCloseable model;
        private ControlPlaneLattice lattice;
        private boolean cleaned;

        private StartupFailure(
                Throwable failure,
                Bootstrap bootstrap,
                ExecutionGpu gpu,
                AutoCloseable model,
                ControlPlaneLattice lattice) {
            super("inference startup failed; retry close on this exception to finish resource cleanup", failure);
            this.bootstrap = bootstrap;
            this.gpu = gpu;
            this.model = model;
            this.lattice = lattice;
        }

        @Override
        public synchronized void close() {
            if (this.cleaned) return;
            if (this.gpu != null) this.gpu.abortWorkerStartup();
            if (this.lattice != null) {
                LAST_CLOSED_LATTICE.set(this.lattice);
                this.lattice.close();
                this.lattice = null;
            }
            if (this.model != null) {
                if (this.gpu != null) this.gpu.ensureWorkersClosed();
                try {
                    this.model.close();
                } catch (RuntimeException | Error cleanup) {
                    throw cleanup;
                } catch (Exception cleanup) {
                    throw new IllegalStateException("model cleanup failed", cleanup);
                }
                this.model = null;
            }
            if (this.gpu != null) {
                this.bootstrap.closeGpu(this.gpu);
                this.gpu = null;
            }
            this.cleaned = true;
            LATTICE_OWNED.set(false);
        }
    }

    /// Creates independent sequence/sampler/decoder state borrowing this engine's shared runtime.
    public synchronized QwenGenerationSession createSession(GenerationConfig config) {
        if (this.closing) throw new IllegalStateException("inference engine is closed");
        this.gpu.ensureHealthy();
        var session = new QwenGenerationSession(
                this.tokenizer,
                this.plan,
                this.runtime,
                this.gpu,
                SEQUENCE_IDS.getAndIncrement(),
                Objects.requireNonNull(config, "config"),
                this.tuning.prefillChunkTokens(),
                this::releaseSession);
        this.sessions.add(session);
        return session;
    }

    private synchronized void releaseSession(QwenGenerationSession session) {
        this.sessions.remove(session);
    }

    public QwenTokenizer tokenizer() {
        return this.tokenizer;
    }

    public QwenConfig modelConfig() {
        return this.model.weights().config();
    }

    /// Returns the immutable tuning this engine was loaded with; its worker IDs are the engine's workers.
    public InferenceTuning tuning() {
        return this.tuning;
    }

    /// Returns an experiment snapshot without generation settings. Identity was measured at load.
    public InferenceRunSnapshot snapshot() {
        return snapshot(null);
    }

    /// Returns an experiment snapshot recording the caller-supplied generation settings, if any.
    public InferenceRunSnapshot snapshot(GenerationConfig generation) {
        return new InferenceRunSnapshot(
                InferenceRunSnapshot.SCHEMA_VERSION,
                InferenceRunSnapshot.Tuning.of(this.tuning),
                InferenceRunSnapshot.ids(this.workerCoreIds),
                this.modelIdentity,
                generation,
                this.runtimeIdentity);
    }

    /// True as soon as shutdown begins; no further sessions can be admitted.
    public boolean isClosed() {
        return this.closing;
    }

    public synchronized CudaGpuMemory.DeviceMemoryInfo deviceMemoryInfo() {
        if (this.closing) throw new IllegalStateException("inference engine is closed");
        return this.bootstrap.memoryInfo(this.gpu);
    }

    /// Stops admission, closes sessions, detaches execution, closes the lattice, then frees model and CUDA.
    /// Do not call from a generation callback: it would wait for that same generation to finish.
    /// If cleanup throws, admission stays closed and a later close retries unreleased resources.
    @Override
    public void close() {
        List<QwenGenerationSession> owned;
        synchronized (this) {
            for (var session : this.sessions) {
                if (session.isGeneratingOnCurrentThread())
                    throw new IllegalStateException("cannot close the engine from its generation callback");
            }
            this.closing = true;
            owned = List.copyOf(this.sessions);
        }
        this.shutdownLock.lock();
        try {
            if (this.resourcesClosed) return;
            // A failed recovery cannot prove that model or sequence buffers are idle.
            this.gpu.ensureHealthy();
            // Cancel all before waiting for any one generation. No admission monitor is held while waiting.
            Throwable failure = null;
            for (var session : owned) {
                try {
                    session.cancel();
                } catch (RuntimeException | Error cleanup) {
                    if (failure == null) failure = cleanup;
                    else suppress(failure, cleanup);
                }
            }
            for (var session : owned) {
                try {
                    session.close();
                } catch (RuntimeException | Error cleanup) {
                    if (failure == null) failure = cleanup;
                    else suppress(failure, cleanup);
                }
            }
            // Fail closed: never unload resources after an unproven session shutdown.
            if (failure instanceof RuntimeException exception) throw exception;
            if (failure instanceof Error error) throw error;
            this.runtime.closeCompletionSink();
            // All inference sources have drained before fabric shutdown, even if fabric teardown is asynchronous.
            LAST_CLOSED_LATTICE.set(this.lattice);
            this.lattice.close();
            this.gpu.ensureWorkersClosed();
            this.model.close();
            this.bootstrap.closeGpu(this.gpu);
            this.resourcesClosed = true;
            synchronized (this) {
                this.sessions.clear();
            }
            LATTICE_OWNED.set(false);
        } finally {
            this.shutdownLock.unlock();
        }
    }

    private static InferenceRunSnapshot.Model modelIdentity(Path path, QwenArtifact artifact, QwenModel model) {
        Long bytes;
        try {
            bytes = Files.size(path);
        } catch (IOException | SecurityException unreadable) {
            bytes = null;
        }
        return new InferenceRunSnapshot.Model(
                path.toString(),
                bytes,
                artifact == null ? null : artifact.header().version(),
                InferenceRunSnapshot.Dimensions.of(model.weights().config()));
    }

    private static InferenceRunSnapshot.RuntimeIdentity runtimeIdentity(Path nativeLibrary) {
        Package euhedral = ControlPlaneLattice.class.getPackage();
        return new InferenceRunSnapshot.RuntimeIdentity(
                Runtime.version().toString(),
                System.getProperty("java.vendor"),
                System.getProperty("java.vm.name"),
                System.getProperty("os.name"),
                System.getProperty("os.arch"),
                euhedral == null ? null : euhedral.getImplementationVersion(),
                codeSourceName(ControlPlaneLattice.class),
                nativeLibrary.toString(),
                InferenceRunSnapshot.UNAVAILABLE);
    }

    /// Returns the file name of the JAR or directory that loaded a class, such as `euhedral-core-0.0.7.jar`.
    private static String codeSourceName(Class<?> type) {
        try {
            var source = type.getProtectionDomain().getCodeSource();
            if (source == null) return null;
            String location = source.getLocation().toString();
            while (location.endsWith("/") || location.endsWith("!"))
                location = location.substring(0, location.length() - 1);
            return location.substring(location.lastIndexOf('/') + 1);
        } catch (RuntimeException unavailable) {
            return null;
        }
    }

    private static void suppress(Throwable failure, Throwable cleanup) {
        if (failure != cleanup) failure.addSuppressed(cleanup);
    }

    // Package-private construction seam for failure injection, not an alternative public backend API.
    static class Bootstrap {
        ProcessorTopology processorTopology() {
            return ProcessorTopology.system();
        }

        QwenArtifact readArtifact(Path path) throws IOException {
            return QwenArtifactReader.read(path);
        }

        ExecutionGpu openGpu(Path path, InferenceTuning tuning) {
            return new CudaGpuMemory(
                    path,
                    tuning.gpuExecutionMode() == GpuExecutionMode.ASYNC_EXPERIMENTAL,
                    tuning.q3DispatchMode(),
                    tuning.q3SmallRowThreshold());
        }

        QwenModel loadModel(Path path, QwenArtifact artifact, ExecutionGpu gpu) throws IOException {
            return QwenModel.load(path, artifact, gpu);
        }

        ControlPlaneLattice createLattice(InferenceConfig config, ExecutionGpu gpu) {
            return gpu.asynchronous() ? createLattice(config, new InferenceGpuExecutor(gpu)) : createLattice(config);
        }

        ControlPlaneLattice createLattice(InferenceConfig config) {
            return createLattice(config, new DefaultExecutor());
        }

        private ControlPlaneLattice createLattice(InferenceConfig config, AbstractExecutor executor) {
            var shard = ControlPlaneShard.createBaseShard("InferenceShard", new BaseCloneableObject(executor));
            var latticeConfig =
                    new LatticeConfig("InferenceLattice", config.workerCpus(), config.shutdownTimeout(), shard);
            long started = System.nanoTime();
            while (true) {
                ControlPlaneLattice lattice = ControlPlaneLattice.getOrCreate(latticeConfig);
                if (lattice != LAST_CLOSED_LATTICE.get()) return lattice;
                // close may return while singleton removal finishes. Never return that stopped fabric on reload.
                if (Thread.currentThread().isInterrupted()
                        || System.nanoTime() - started
                                >= config.shutdownTimeout().toNanos())
                    throw new IllegalStateException("previous Euhedral fabric has not finished shutting down");
                LockSupport.parkNanos(1_000_000L);
            }
        }

        void startLattice(ControlPlaneLattice lattice) {
            lattice.start();
        }

        void closeGpu(ExecutionGpu gpu) {
            ((CudaGpuMemory) gpu).close();
        }

        CudaGpuMemory.DeviceMemoryInfo memoryInfo(ExecutionGpu gpu) {
            return ((CudaGpuMemory) gpu).deviceMemoryInfo();
        }
    }
}
