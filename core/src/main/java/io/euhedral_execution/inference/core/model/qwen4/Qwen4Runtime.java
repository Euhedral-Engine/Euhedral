package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.runtime.HostTasks;
import io.euhedral_execution.inference.core.runtime.graph.InferenceLake;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;
import java.util.function.Supplier;

/// What a Flash-Next engine runs on: the storage the residency plan produced, the execution plan
/// over it, the lattice runtime that instantiates and runs the plan's stage graphs, and the lattice
/// attachments its host work is published through. It owns resources, not execution threads and not
/// a scheduler: a generation is a chain of continuations ([Session]), a step is a
/// quantum of a static stage graph ([ExecutionPlan]) that the lattice runs, the expert
/// hierarchy's reads, copies and completions are frames too, and every one of them runs on the
/// lattice's workers. Nothing here waits for the device or the artifact with a thread of its own.
///
/// Two attachments, so that unrelated work stays independent: `execution` carries the expert
/// pipeline and the generation chain (everything that serves the GPU), `host` carries the work
/// around a request (tokenizing, rendering).
public final class Qwen4Runtime implements AutoCloseable {

    /// Device lanes the stages of a step spread over: the critical chain, the shared expert's side
    /// branch and the experts. `EUHEDRAL_QWEN4_LANES` overrides it for benchmarks.
    static final int LANES = Integer.parseInt(System.getenv().getOrDefault("EUHEDRAL_QWEN4_LANES", "2"));

    private final Storage storage;
    private final EuhedralInferenceRuntime runtime;
    private final ExecutionPlan plan;
    private final InferenceLake lake;
    private final HostTasks execution;
    private final HostTasks host;
    private boolean closed;

    private Qwen4Runtime(
            Storage storage,
            EuhedralInferenceRuntime runtime,
            ExecutionPlan plan,
            InferenceLake lake,
            HostTasks execution,
            HostTasks host) {
        this.lake = lake;
        this.storage = storage;
        this.runtime = runtime;
        this.plan = plan;
        this.execution = execution;
        this.host = host;
    }

    /// Loads the `qwen4_exp` artifact of `config` as a text model whose host work runs on
    /// `lattice`, which must be started and outlive the runtime.
    public static Qwen4Runtime load(InferenceConfig config, LatticeTerminal lattice) throws IOException {
        InferenceLake lake = EuhedralInferenceRuntime.newLake(lattice);
        HostTasks execution = new HostTasks(lake);
        HostTasks host = new HostTasks(lake);
        Storage storage = Storage.load(config);
        EuhedralInferenceRuntime runtime = null;
        try {
            runtime = new EuhedralInferenceRuntime(lake, storage.gpu(), EuhedralInferenceRuntime.Lanes.of(LANES));
            ExecutionPlan plan = new ExecutionPlan(storage.gpu(), storage.model(), config.maxContextTokens(), runtime);
            return new Qwen4Runtime(storage, runtime, plan, lake, execution, host);
        } catch (Throwable failure) {
            try {
                if (runtime != null) runtime.close();
                storage.close();
            } catch (RuntimeException cleanup) {
                failure.addSuppressed(cleanup);
            }
            try {
                host.close();
                execution.close();
                lake.completeGracefully();
                lake.awaitTermination();
            } catch (RuntimeException cleanup) {
                failure.addSuppressed(cleanup);
            }
            if (failure instanceof IOException io) throw io;
            if (failure instanceof RuntimeException runtimeFailure) throw runtimeFailure;
            if (failure instanceof Error error) throw error;
            throw new IOException(failure);
        }
    }

    public ExecutionGpu gpu() {
        return this.storage.gpu();
    }

    CudaGpuMemory cuda() {
        return (CudaGpuMemory) this.storage.gpu();
    }

    public Storage storage() {
        return this.storage;
    }

    public ExecutionPlan plan() {
        return this.plan;
    }

    EuhedralInferenceRuntime runtime() {
        return this.runtime;
    }

    public Qwen4Config config() {
        return this.storage.model().artifact().config();
    }

    public HostTasks executionTasks() {
        return this.execution;
    }

    public HostTasks hostTasks() {
        return this.host;
    }

    /// Prompt tokens per prefill step: the most the plan's workspace holds.
    public int prefillChunkTokens() {
        return this.plan.maxRows();
    }

    public Session createSession(
            QwenTokenizer tokenizer,
            GenerationConfig config,
            int prefillChunkTokens,
            Consumer<GenerationSession> release) {
        return new Session(tokenizer, this.plan, this.execution, gpu(), config, prefillChunkTokens, release::accept);
    }

    public <T> CompletableFuture<T> onWorker(Supplier<T> work) {
        return this.host.onWorker(work);
    }

    CompletableFuture<int[]> tokenize(QwenTokenizer tokenizer, String text, boolean modelSpecialTokens) {
        return this.host.tokenize(tokenizer, text, modelSpecialTokens);
    }

    /// Releases the executor, the storage (which settles every expert transfer: the lattice still
    /// runs their completions), then drains and detaches the host work. The owner has stopped
    /// admission and ended every generation before calling.
    @Override
    public synchronized void close() {
        if (this.closed) return;
        this.closed = true;
        RuntimeException failure = null;
        // Every accepted step retires (its expert sources still serve it) before the plan completes the sources.
        try {
            this.runtime.close();
        } catch (RuntimeException closeFailure) {
            failure = closeFailure;
        }
        this.plan.close();
        try {
            this.storage.close();
        } catch (RuntimeException closeFailure) {
            if (failure == null) failure = closeFailure;
            else failure.addSuppressed(closeFailure);
        }
        for (HostTasks tasks : new HostTasks[] {this.host, this.execution}) {
            try {
                tasks.close();
            } catch (RuntimeException closeFailure) {
                if (failure == null) failure = closeFailure;
                else failure.addSuppressed(closeFailure);
            }
        }
        try {
            this.lake.completeGracefully();
            this.lake.awaitTermination();
        } catch (RuntimeException closeFailure) {
            if (failure == null) failure = closeFailure;
            else failure.addSuppressed(closeFailure);
        }
        if (failure != null) throw failure;
    }
}
