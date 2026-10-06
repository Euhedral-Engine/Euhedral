package io.euhedral_execution.inference.core;

import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Config;
import io.euhedral_execution.inference.core.qwen4.Qwen4ExecutionPlan;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.scheduling.GenerationSession;
import io.euhedral_execution.inference.core.scheduling.HostTasks;
import io.euhedral_execution.inference.core.scheduling.Qwen4GenerationSession;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;
import java.util.function.Supplier;

/// What a Flash-Next engine runs on: the storage the residency plan produced, the execution plan over it, the
/// lattice runtime that instantiates and runs the plan's stage graphs, and the lattice attachments its host work
/// is published through. It owns resources, not execution threads and not a scheduler: a generation is a chain
/// of continuations ([Qwen4GenerationSession]), a step is a quantum of a static stage graph
/// ([Qwen4ExecutionPlan]) that the lattice runs, the expert hierarchy's reads, copies and completions are frames
/// too, and every one of them runs on the lattice's workers. Nothing here waits for the device or the artifact
/// with a thread of its own.
///
/// Two attachments, so that unrelated work stays independent: `execution` carries the expert pipeline and the
/// generation chain (everything that serves the GPU), `host` carries the work around a request (tokenizing,
/// rendering).
final class Qwen4Runtime implements AutoCloseable {

    /// Device lanes the stages of a step spread over: the critical chain and the shared expert's side branch.
    private static final int LANES = 2;

    private final Qwen4Storage storage;
    private final EuhedralInferenceRuntime runtime;
    private final Qwen4ExecutionPlan plan;
    private final HostTasks execution;
    private final HostTasks host;
    private boolean closed;

    private Qwen4Runtime(
            Qwen4Storage storage,
            EuhedralInferenceRuntime runtime,
            Qwen4ExecutionPlan plan,
            HostTasks execution,
            HostTasks host) {
        this.storage = storage;
        this.runtime = runtime;
        this.plan = plan;
        this.execution = execution;
        this.host = host;
    }

    /// Loads the `qwen4_exp` artifact of `config` as a text model whose host work runs on `lattice`, which must be
    /// started and outlive the runtime.
    static Qwen4Runtime load(InferenceConfig config, LatticeTerminal lattice) throws IOException {
        HostTasks execution = new HostTasks(lattice);
        HostTasks host = new HostTasks(lattice);
        Qwen4Storage storage = Qwen4Storage.load(config, execution);
        EuhedralInferenceRuntime runtime = null;
        try {
            runtime = new EuhedralInferenceRuntime(lattice, execution, storage.gpu(), LANES);
            Qwen4ExecutionPlan plan =
                    new Qwen4ExecutionPlan(storage.gpu(), storage.model(), config.maxContextTokens(), runtime);
            return new Qwen4Runtime(storage, runtime, plan, execution, host);
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
            } catch (RuntimeException cleanup) {
                failure.addSuppressed(cleanup);
            }
            if (failure instanceof IOException io) throw io;
            if (failure instanceof RuntimeException runtimeFailure) throw runtimeFailure;
            if (failure instanceof Error error) throw error;
            throw new IOException(failure);
        }
    }

    ExecutionGpu gpu() {
        return this.storage.gpu();
    }

    CudaGpuMemory cuda() {
        return (CudaGpuMemory) this.storage.gpu();
    }

    Qwen4Storage storage() {
        return this.storage;
    }

    Qwen4ExecutionPlan plan() {
        return this.plan;
    }

    EuhedralInferenceRuntime runtime() {
        return this.runtime;
    }

    Qwen4Config config() {
        return this.storage.model().artifact().config();
    }

    HostTasks executionTasks() {
        return this.execution;
    }

    HostTasks hostTasks() {
        return this.host;
    }

    Qwen4GenerationSession createSession(
            QwenTokenizer tokenizer,
            GenerationConfig config,
            int prefillChunkTokens,
            Consumer<GenerationSession> release) {
        return new Qwen4GenerationSession(
                tokenizer, this.plan, this.execution, gpu(), config, prefillChunkTokens, release::accept);
    }

    <T> CompletableFuture<T> onWorker(Supplier<T> work) {
        return this.host.onWorker(work);
    }

    CompletableFuture<int[]> tokenize(QwenTokenizer tokenizer, String text, boolean modelSpecialTokens) {
        return this.host.tokenize(tokenizer, text, modelSpecialTokens);
    }

    /// Releases the executor, the storage (which settles every expert transfer: the lattice still runs their
    /// completions), then drains and detaches the host work. The owner has stopped admission and ended every
    /// generation before calling.
    @Override
    public synchronized void close() {
        if (this.closed) return;
        this.closed = true;
        RuntimeException failure = null;
        this.plan.close();
        try {
            this.runtime.close();
        } catch (RuntimeException closeFailure) {
            failure = closeFailure;
        }
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
        if (failure != null) throw failure;
    }
}
