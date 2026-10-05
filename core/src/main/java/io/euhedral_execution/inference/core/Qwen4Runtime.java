package io.euhedral_execution.inference.core;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Config;
import io.euhedral_execution.inference.core.qwen4.Qwen4Executor;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.GenerationSession;
import io.euhedral_execution.inference.core.scheduling.Qwen4GenerationSession;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Supplier;

/// What a Flash-Next engine runs on: the storage the residency plan produced, the executor over it, and the one worker
/// thread that owns the executor's stream. Sessions submit their generations to that worker, which serializes them, and
/// host work the API asks the engine for (tokenizing, rendering) runs on a small separate pool so it never waits behind
/// a generation.
final class Qwen4Runtime implements AutoCloseable {

    private final Qwen4Storage storage;
    private final Qwen4Executor executor;
    private final ExecutorService generations;
    private final ExecutorService hostWork;
    private boolean closed;

    private Qwen4Runtime(Qwen4Storage storage, Qwen4Executor executor) {
        this.storage = storage;
        this.executor = executor;
        this.generations = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, "flash-next-generation");
            thread.setDaemon(true);
            return thread;
        });
        this.hostWork = Executors.newFixedThreadPool(2, runnable -> {
            Thread thread = new Thread(runnable, "flash-next-host");
            thread.setDaemon(true);
            return thread;
        });
    }

    /// Loads the `qwen4_exp` artifact of `config` as a text model.
    static Qwen4Runtime load(InferenceConfig config) throws IOException {
        Qwen4Storage storage = Qwen4Storage.load(config);
        try {
            Qwen4Executor executor = new Qwen4Executor(storage.gpu(), storage.model(), config.maxContextTokens());
            return new Qwen4Runtime(storage, executor);
        } catch (Throwable failure) {
            try {
                storage.close();
            } catch (RuntimeException cleanup) {
                failure.addSuppressed(cleanup);
            }
            if (failure instanceof IOException io) throw io;
            if (failure instanceof RuntimeException runtime) throw runtime;
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

    Qwen4Config config() {
        return this.storage.model().artifact().config();
    }

    Qwen4GenerationSession createSession(
            QwenTokenizer tokenizer,
            GenerationConfig config,
            int prefillChunkTokens,
            Consumer<GenerationSession> release) {
        return new Qwen4GenerationSession(
                tokenizer, this.executor, this.generations, gpu(), config, prefillChunkTokens, release::accept);
    }

    <T> CompletableFuture<T> onWorker(Supplier<T> work) {
        return CompletableFuture.supplyAsync(work, this.hostWork);
    }

    /// Stops accepting work, waits for the generation worker to drain, then releases the executor and the storage.
    @Override
    public synchronized void close() {
        if (this.closed) return;
        this.closed = true;
        this.generations.shutdown();
        this.hostWork.shutdown();
        try {
            if (!this.generations.awaitTermination(30, TimeUnit.SECONDS))
                throw new IllegalStateException("a Flash-Next generation did not stop within 30 seconds");
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
        try {
            this.executor.close();
        } finally {
            this.storage.close();
        }
    }
}
