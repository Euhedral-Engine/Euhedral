package io.euhedral_execution.inference.core;

import io.euhedral_execution.core.config.FragmentConfig;
import io.euhedral_execution.core.config.IdlePolicy;
import io.euhedral_execution.core.config.LatticeConfig;
import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.control_plane.ControlPlaneShard;
import io.euhedral_execution.core.generics.AbstractExecutor;
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.SessionOptions;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.ModelArchitecture;
import io.euhedral_execution.inference.core.model.ModelRuntime;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Model;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.model.qwen38.Session;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.ResidencyPlanner;
import io.euhedral_execution.inference.core.model.qwen4.Qwen4Runtime;
import io.euhedral_execution.inference.core.prefix.PrefixCacheStats;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// High-level owner of model, CUDA backend, Euhedral lattice, and all sessions it creates.
/// Close sessions early when finished; engine close also closes every tracked session.
/// This standalone application's fabric is process-wide. The engine controls its start/stop;
/// independent sessions and their sources share it. Only one engine may control that lifecycle.
/// Initialize the process fabric through load, not a separate low-level lattice factory; advanced
/// sources may borrow the running fabric but must not initialize or stop it independently.
public final class InferenceEngine implements AutoCloseable {
    private static final Logger LOG = LoggerFactory.getLogger(InferenceEngine.class);
    private static final AtomicBoolean LATTICE_OWNED = new AtomicBoolean();
    private static final AtomicReference<ControlPlaneLattice> LAST_CLOSED_LATTICE = new AtomicReference<>();
    private static final AtomicLong SEQUENCE_IDS = new AtomicLong();
    private final Bootstrap bootstrap;
    private final QwenTokenizer tokenizer;
    private final ControlPlaneLattice lattice;
    /// The loaded model: the engine chooses it at load and delegates everything after that to it.
    private final ModelRuntime model;
    /// The GPU this engine opened through its bootstrap, closed after the lattice; null when the model's runtime
    /// owns its GPU (Flash-Next).
    private final ExecutionGpu ownedGpu;
    private final InferenceConfig config;
    private final BitSet workerCoreIds;
    private final InferenceRunSnapshot.RuntimeIdentity runtimeIdentity;
    private final List<GenerationSession> sessions = new ArrayList<>();
    private final ReentrantLock shutdownLock = new ReentrantLock();
    private volatile boolean closing;
    private boolean resourcesClosed;

    private InferenceEngine(
            Bootstrap bootstrap,
            QwenTokenizer tokenizer,
            ControlPlaneLattice lattice,
            ModelRuntime model,
            ExecutionGpu ownedGpu,
            InferenceConfig config,
            BitSet workerCoreIds,
            InferenceRunSnapshot.RuntimeIdentity runtimeIdentity) {
        this.bootstrap = bootstrap;
        this.tokenizer = tokenizer;
        this.lattice = lattice;
        this.model = model;
        this.ownedGpu = ownedGpu;
        this.config = config;
        this.workerCoreIds = workerCoreIds;
        this.runtimeIdentity = runtimeIdentity;
    }

    /// Loads all model resources and starts the lattice before returning an engine.
    /// Worker processor IDs are checked against the host [ProcessorTopology] before the tokenizer, model,
    /// or GPU is loaded. The execution policy and weight residency are derived from the artifact.
    public static InferenceEngine load(InferenceConfig config) throws IOException {
        return load(config, new Bootstrap());
    }

    static InferenceEngine load(InferenceConfig config, Bootstrap bootstrap) throws IOException {
        Objects.requireNonNull(config, "config");
        if (bootstrap.architecture(config.artifactPath()) == ModelArchitecture.QWEN4_EXP)
            return loadQwen4(config, bootstrap);
        // Euhedral silently drops unavailable CPUs; fail before claiming the lattice or loading anything.
        ProcessorTopology topology = bootstrap.processorTopology();
        topology.requireAvailable(config.workerCpus());
        BitSet workerCoreIds = topology.coreIds(config.workerCpus());
        if (!LATTICE_OWNED.compareAndSet(false, true))
            throw new IllegalStateException("an inference engine already owns the process-wide Euhedral lattice");
        ExecutionGpu gpu = null;
        Qwen38Model model = null;
        ControlPlaneLattice lattice = null;
        try {
            QwenTokenizer tokenizer = QwenTokenizer.load(config.tokenizerDirectory());
            Artifact artifact = bootstrap.readArtifact(config.artifactPath());
            gpu = bootstrap.openGpu(config.cudaLibraryPath());
            ArtifactProfile profile = artifact == null ? null : ArtifactProfile.of(artifact);
            model = bootstrap.loadModel(config.artifactPath(), artifact, profile, gpu, config.maxContextTokens());
            lattice = bootstrap.createLattice(config);
            bootstrap.startLattice(lattice);
            Qwen38Runtime dense = Qwen38Runtime.open(config, tokenizer, gpu, model, artifact, profile, lattice);
            return new InferenceEngine(
                    bootstrap,
                    tokenizer,
                    lattice,
                    dense,
                    gpu,
                    config,
                    workerCoreIds,
                    runtimeIdentity(config.cudaLibraryPath()));
        } catch (IOException | RuntimeException | Error failure) {
            var pending = new StartupFailure(
                    failure,
                    bootstrap,
                    gpu,
                    failure instanceof Qwen38Model.LoadFailure partial ? partial : model,
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

    /// Loads a Flash-Next (`qwen4_exp`) artifact as a text model. The residency is derived from the artifact and the
    /// maximum context alone. Generation, the expert hierarchy and the host work around a request all run on the
    /// engine's lattice, as the dense model's do: the runtime owns resources, not threads.
    private static InferenceEngine loadQwen4(InferenceConfig config, Bootstrap bootstrap) throws IOException {
        // Euhedral silently drops unavailable CPUs; fail before claiming the lattice or loading anything.
        ProcessorTopology topology = bootstrap.processorTopology();
        topology.requireAvailable(config.workerCpus());
        BitSet workerCoreIds = topology.coreIds(config.workerCpus());
        if (!LATTICE_OWNED.compareAndSet(false, true))
            throw new IllegalStateException("an inference engine already owns the process-wide Euhedral lattice");
        ControlPlaneLattice lattice = null;
        Qwen4Runtime runtime = null;
        try {
            QwenTokenizer tokenizer = QwenTokenizer.load(config.tokenizerDirectory());
            lattice = bootstrap.createLattice(config);
            bootstrap.startLattice(lattice);
            runtime = Qwen4Runtime.load(config, tokenizer, lattice);
            return new InferenceEngine(
                    bootstrap,
                    tokenizer,
                    lattice,
                    runtime,
                    null,
                    config,
                    workerCoreIds,
                    runtimeIdentity(config.cudaLibraryPath()));
        } catch (IOException | RuntimeException | Error failure) {
            try {
                if (runtime != null) runtime.stop();
            } catch (RuntimeException | Error cleanup) {
                failure.addSuppressed(cleanup);
            }
            try {
                if (lattice != null) {
                    LAST_CLOSED_LATTICE.set(lattice);
                    lattice.close();
                }
            } catch (RuntimeException | Error cleanup) {
                failure.addSuppressed(cleanup);
            } finally {
                LATTICE_OWNED.set(false);
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
            if (this.lattice != null) {
                LAST_CLOSED_LATTICE.set(this.lattice);
                this.lattice.close();
                this.lattice = null;
            }
            if (this.model != null) {
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

    /// Creates a session over the engine's model, with the default options.
    public GenerationSession createGenerationSession(GenerationConfig config) {
        return createGenerationSession(config, SessionOptions.DEFAULT);
    }

    /// Creates a session over the engine's model; the model honours `options` where it can.
    public synchronized GenerationSession createGenerationSession(GenerationConfig config, SessionOptions options) {
        if (this.closing) throw new IllegalStateException("inference engine is closed");
        this.model.gpu().ensureHealthy();
        var session = this.model.createSession(
                Objects.requireNonNull(config, "config"),
                Objects.requireNonNull(options, "options"),
                this::releaseGenerationSession);
        this.sessions.add(session);
        return session;
    }

    /// A dense session (tests): what [#createGenerationSession(GenerationConfig)] creates for a Qwen3.8 model.
    synchronized Session createSession(GenerationConfig config) {
        return createSession(config, Session.DEFAULT_PREFILL_CHUNK_TOKENS);
    }

    /// As [#createSession(GenerationConfig)] with a smaller prefill chunk, so tests can exercise several prefill
    /// quanta on short prompts.
    synchronized Session createSession(GenerationConfig config, int prefillChunkTokens) {
        if (!(this.model instanceof Qwen38Runtime dense))
            throw new UnsupportedOperationException("a Flash-Next engine opens sessions with createGenerationSession");
        if (this.closing) throw new IllegalStateException("inference engine is closed");
        this.model.gpu().ensureHealthy();
        var session = dense.createSession(
                Objects.requireNonNull(config, "config"),
                prefillChunkTokens,
                SessionOptions.DEFAULT,
                this::releaseGenerationSession);
        this.sessions.add(session);
        return session;
    }

    private synchronized void releaseGenerationSession(GenerationSession session) {
        this.sessions.remove(session);
    }

    /// The loaded model's runtime (tests).
    ModelRuntime modelRuntime() {
        return this.model;
    }

    public QwenTokenizer tokenizer() {
        return this.tokenizer;
    }

    /// Encodes `prompt` as a new session encodes its first prompt (with the model special tokens), on the
    /// lattice's workers. A fresh session generating from these IDs runs exactly that prompt.
    public CompletableFuture<int[]> tokenizePromptAsync(String prompt) {
        return this.model.tokenizePrompt(prompt);
    }

    /// Runs `work` as one frame on the lattice's workers: prompt-side host work (rendering, validation) of a
    /// client that must not do it on its own threads. The future completes on that worker.
    public <T> CompletableFuture<T> onWorker(Supplier<T> work) {
        return this.model.onWorker(work);
    }

    /// Entries of the model's output vocabulary.
    public int vocabularySize() {
        return this.model.vocabularySize();
    }

    /// The longest sequence the model was trained for.
    public int maxPositionEmbeddings() {
        return this.model.maxPositionEmbeddings();
    }

    /// The longest sequence the model in the artifact at `artifact` was trained for, read without loading it.
    public static int maxPositionEmbeddings(Path artifact) throws IOException {
        return ModelArchitecture.maxPositionEmbeddings(artifact);
    }

    /// Prompt tokens per prefill quantum.
    public int prefillChunkTokens() {
        return this.model.prefillChunkTokens();
    }

    /// Returns the configuration this engine was loaded with; its worker IDs are the engine's workers.
    public InferenceConfig config() {
        return this.config;
    }

    /// What the model is: the artifact's name and the speculation it selects.
    public ModelDescription description() {
        return this.model.description();
    }

    /// Returns an experiment snapshot without generation settings. Identity was measured at load.
    public InferenceRunSnapshot snapshot() {
        return snapshot(null);
    }

    /// Returns an experiment snapshot recording the caller-supplied generation settings, if any.
    public InferenceRunSnapshot snapshot(GenerationConfig generation) {
        return new InferenceRunSnapshot(
                InferenceRunSnapshot.SCHEMA_VERSION,
                InferenceRunSnapshot.Configuration.of(this.config, this.model.description()),
                InferenceRunSnapshot.ids(this.workerCoreIds),
                this.model.identity(),
                generation,
                this.runtimeIdentity);
    }

    /// Counters of the prefix cache, or null when the engine runs without one.
    public PrefixCacheStats prefixCacheStats() {
        return this.model.prefixCacheStats();
    }

    /// True as soon as shutdown begins; no further sessions can be admitted.
    public boolean isClosed() {
        return this.closing;
    }

    public synchronized CudaGpuMemory.DeviceMemoryInfo deviceMemoryInfo() {
        if (this.closing) throw new IllegalStateException("inference engine is closed");
        return this.bootstrap.memoryInfo(this.model.gpu());
    }

    /// Bytes of model weights this engine keeps in pinned host memory instead of on the device.
    public synchronized long hostBackedWeightBytes() {
        if (this.closing) throw new IllegalStateException("inference engine is closed");
        return this.model.hostBackedWeightBytes();
    }

    /// Device bytes this engine owns: model weights, open sessions' persistent state, and the reusable
    /// workspace storage of its execution graphs. Other processes and CUDA's own context and kernel
    /// modules are excluded.
    public synchronized long allocatedDeviceBytes() {
        if (this.closing) throw new IllegalStateException("inference engine is closed");
        return this.bootstrap.allocatedBytes(this.model.gpu());
    }

    /// The largest [#allocatedDeviceBytes] since the engine loaded or the last
    /// [#resetPeakAllocatedDeviceBytes], transient allocations included.
    public synchronized long peakAllocatedDeviceBytes() {
        if (this.closing) throw new IllegalStateException("inference engine is closed");
        return this.bootstrap.peakAllocatedBytes(this.model.gpu());
    }

    /// Restarts [#peakAllocatedDeviceBytes] at the current [#allocatedDeviceBytes].
    public synchronized void resetPeakAllocatedDeviceBytes() {
        if (this.closing) throw new IllegalStateException("inference engine is closed");
        this.bootstrap.resetPeakAllocatedBytes(this.model.gpu());
    }

    /// The part of [#allocatedDeviceBytes] that execution graphs retain between quanta, so that a quantum
    /// finds its workspace already allocated. It is bounded by the graphs' largest quanta, not by
    /// sessions or tokens, and is released when the engine closes.
    public synchronized long retainedWorkspaceBytes() {
        if (this.closing) throw new IllegalStateException("inference engine is closed");
        return this.model.retainedWorkspaceBytes();
    }

    /// Stops admission, closes sessions, stops the model, closes the lattice, then frees the model and the GPU.
    /// Do not call from a generation callback: it would wait for that same generation to finish.
    /// If cleanup throws, admission stays closed and a later close retries unreleased resources.
    @Override
    public void close() {
        List<GenerationSession> owned;
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
            this.model.gpu().ensureHealthy();
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
            // Every accepted quantum and host task ends and the model's sources detach before the fabric stops; the
            // model's memory and the GPU are freed after it.
            this.model.stop();
            LAST_CLOSED_LATTICE.set(this.lattice);
            this.lattice.close();
            this.model.close();
            if (this.ownedGpu != null) this.bootstrap.closeGpu(this.ownedGpu);
            this.resourcesClosed = true;
            synchronized (this) {
                this.sessions.clear();
            }
            LATTICE_OWNED.set(false);
        } finally {
            this.shutdownLock.unlock();
        }
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

        /// The artifact's architecture when it positively identifies as one other than the dense model; anything
        /// unreadable is left to [#readArtifact], which reports it.
        ModelArchitecture architecture(Path path) {
            try {
                return ModelArchitecture.detect(path);
            } catch (IOException | RuntimeException unreadable) {
                return ModelArchitecture.QWEN38_DENSE;
            }
        }

        Artifact readArtifact(Path path) throws IOException {
            return ArtifactReader.read(path);
        }

        ExecutionGpu openGpu(Path path) {
            return new CudaGpuMemory(path);
        }

        /// Loads the artifact's executed objects, keeping in host memory only as many weights as a context
        /// of `maxContextTokens` needs to fit in the device's free memory.
        Qwen38Model loadModel(
                Path path, Artifact artifact, ArtifactProfile profile, ExecutionGpu gpu, int maxContextTokens)
                throws IOException {
            int positions = artifact.config().maxPositionEmbeddings();
            if (maxContextTokens > positions)
                throw new IllegalArgumentException("a context of " + maxContextTokens + " tokens is longer than the "
                        + positions + " positions the model supports; set a smaller max context");
            // Tests pin residency with a device budget below the free memory, which other processes shift between
            // loads.
            long free = Math.min(
                    memoryInfo(gpu).freeBytes(),
                    Long.getLong("euhedral.residency.device-budget-bytes", Long.MAX_VALUE));
            var plan = ResidencyPlanner.plan(artifact, profile, free, maxContextTokens);
            if (!plan.fits())
                throw new IOException("a context of " + maxContextTokens + " tokens does not fit in the "
                        + (free >> 20) + " MiB free on this GPU, even with the weights in host memory; set a "
                        + "smaller max context");
            long hostBytes = 0;
            for (var tensor : artifact.tensors())
                if (plan.hostBacked().contains(tensor.name())) hostBytes += tensor.byteSize();
            LOG.info(
                    "Context {} tokens: {} MiB of the {} MiB free on the GPU, {} MiB of weights in host memory",
                    maxContextTokens,
                    plan.deviceBytes() >> 20,
                    free >> 20,
                    hostBytes >> 20);
            return Qwen38Model.load(path, artifact, gpu, profile.speculation(), plan.hostBacked());
        }

        /// Fragment defaults with fixed idle timing: an idle worker parks for the default 15 us. The adaptive
        /// timing derives a worker's park (up to 0.8 ms) and its choice to idle from that worker's own history,
        /// which only work refreshes; after long prefills it left fewer and fewer workers awake, and the
        /// frame-by-frame quanta of later requests waited on parked workers (docs/FRAME_MODEL.md).
        static FragmentConfig fragmentConfig() {
            FragmentConfig defaults = FragmentConfig.ofDefaults();
            return new FragmentConfig(
                    defaults.cloneConfig(),
                    defaults.cacheConfig(),
                    defaults.observer(),
                    defaults.maxBatchSize(),
                    defaults.smtEnabled(),
                    new IdlePolicy(IdlePolicy.DEFAULT_IDLE_PARK_NS, IdlePolicy.DEFAULT_CONTENTION_HALF_LIFE_NANOS),
                    defaults.benchmarkMode(),
                    defaults.metricPrefix(),
                    defaults.registry());
        }

        ControlPlaneLattice createLattice(InferenceConfig config) {
            return createLattice(config, new DefaultExecutor());
        }

        private ControlPlaneLattice createLattice(InferenceConfig config, AbstractExecutor executor) {
            var shard = ControlPlaneShard.createBaseShard(
                    "InferenceShard", new BaseCloneableObject(fragmentConfig(), executor));
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

        long allocatedBytes(ExecutionGpu gpu) {
            return ((CudaGpuMemory) gpu).allocatedBytes();
        }

        long peakAllocatedBytes(ExecutionGpu gpu) {
            return ((CudaGpuMemory) gpu).peakAllocatedBytes();
        }

        void resetPeakAllocatedBytes(ExecutionGpu gpu) {
            ((CudaGpuMemory) gpu).resetPeakAllocatedBytes();
        }
    }
}
