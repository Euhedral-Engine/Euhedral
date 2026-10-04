package io.euhedral_execution.inference.api.web;

import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.api.engine.InferenceUnavailableException;
import io.euhedral_execution.inference.core.guidance.GrammarCompiler;
import io.euhedral_execution.inference.core.guidance.Llguidance;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/// Test backend that follows the engine's asynchronous contract without CUDA: a small pool stands in for the
/// lattice's workers, which encode prompts and run generations; text is emitted on a worker between
/// "quanta", cancellation stops the next quantum, and generations record their inputs and lifecycle so tests
/// can assert cancellation and cleanup. With `emptyReasoning`, output that opens in the think block begins with
/// an empty reasoning (`</think>` and a blank line, one token), so scripts of plain answers serve requests with
/// thinking on.
final class ScriptedInferenceBackend implements InferenceBackend {
    static final String MODEL_ID = "euhedral-test-model";

    final List<ScriptedGeneration> generations = new CopyOnWriteArrayList<>();
    private final AtomicInteger workerIds = new AtomicInteger();
    /// Stands in for the lattice's workers.
    final ExecutorService workers = Executors.newFixedThreadPool(4, task -> {
        Thread thread = new Thread(task, "scripted-worker-" + this.workerIds.incrementAndGet());
        thread.setDaemon(true);
        return thread;
    });
    volatile boolean available = true;
    volatile Script script = tokens(List.of("Hello", ", ", "world"), true);
    volatile int contextLength = 4096;
    volatile boolean emptyReasoning;

    /// Behavior of the next generations. Implementations must honor `generation.isCancelled()`.
    @FunctionalInterface
    interface Script {
        Result run(ScriptedGeneration generation, int maxNewTokens, Consumer<String> output)
                throws InterruptedException, ExecutionException;
    }

    /// Emits one chunk per token, then a stop token when `stopToken` and budget remain.
    static Script tokens(List<String> chunks, boolean stopToken) {
        return (generation, maxNewTokens, output) -> {
            int sampled = 0;
            for (String chunk : chunks) {
                if (generation.isCancelled() || sampled == maxNewTokens) return new Result(sampled, false);
                sampled++;
                generation.emit(output, chunk);
            }
            if (!stopToken || sampled == maxNewTokens || generation.isCancelled()) return new Result(sampled, false);
            return new Result(sampled + 1, true);
        };
    }

    /// Emits `tok0 tok1 ...` every `delayMillis` until cancelled or the budget is used.
    static Script endless(long delayMillis) {
        return (generation, maxNewTokens, output) -> {
            int sampled = 0;
            while (sampled < maxNewTokens && !generation.isCancelled()) {
                generation.emit(output, "tok" + sampled + " ");
                sampled++;
                Thread.sleep(delayMillis);
            }
            return new Result(sampled, false);
        };
    }

    /// The model closes its think block at once, then follows `answer`.
    static Script thinkingNothing(Script answer) {
        return (generation, maxNewTokens, output) -> {
            if (generation.isCancelled() || maxNewTokens == 0) return new Result(0, false);
            generation.emit(output, "</think>\n\n");
            Result rest = answer.run(generation, maxNewTokens - 1, output);
            return new Result(rest.completionTokens() + 1, rest.stopTokenReached(), 0);
        };
    }

    /// Reports `quanta` prefill quanta `delayMillis` apart, then emits `tok0 tok1 ...` until cancelled or the budget
    /// is used.
    static Script prefilling(int quanta, long delayMillis) {
        return (generation, maxNewTokens, output) -> {
            for (int quantum = 0; quantum < quanta; quantum++) {
                if (generation.isCancelled()) return new Result(0, false);
                generation.prefill();
                Thread.sleep(delayMillis);
            }
            return endless(delayMillis).run(generation, maxNewTokens, output);
        };
    }

    void reset() {
        this.generations.clear();
        this.available = true;
        this.contextLength = 4096;
        this.emptyReasoning = false;
        this.script = tokens(List.of("Hello", ", ", "world"), true);
    }

    ScriptedGeneration only() {
        if (this.generations.size() != 1)
            throw new AssertionError("expected one generation, found " + this.generations.size());
        return this.generations.getFirst();
    }

    /// The first generation, once a worker opened it: requests are planned and admitted asynchronously.
    ScriptedGeneration awaitFirst() throws InterruptedException {
        long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(10);
        while (this.generations.isEmpty()) {
            if (System.nanoTime() > deadline) throw new AssertionError("no generation was opened");
            Thread.sleep(1);
        }
        return this.generations.getFirst();
    }

    @Override
    public String modelId() {
        return MODEL_ID;
    }

    @Override
    public boolean isAvailable() {
        return this.available;
    }

    @Override
    public int contextLength() {
        return this.contextLength;
    }

    /// Deterministic stand-in for tokenization: one token per character.
    public int countPromptTokens(String prompt) {
        return prompt.length();
    }

    /// Prompts encoded so far; a request encodes its prompt once.
    final AtomicInteger encoded = new AtomicInteger();

    @Override
    public Executor workers() {
        return this.workers;
    }

    @Override
    public CompletableFuture<EncodedPrompt> encodePrompt(String prompt) {
        return CompletableFuture.supplyAsync(
                () -> {
                    this.encoded.incrementAndGet();
                    return new EncodedPrompt(prompt, new int[countPromptTokens(prompt)]);
                },
                this.workers);
    }

    /// Real llguidance over a byte vocabulary: whether a grammar or schema compiles does not depend on the
    /// vocabulary, so requests are refused as the engine refuses them.
    private static final GrammarCompiler GRAMMARS = new GrammarCompiler(
            Llguidance.load(Path.of(System.getProperty("euhedral.llguidance.library"))),
            257,
            id -> id < 256 ? new byte[] {(byte) id} : new byte[] {(byte) 0xff, '<', 'e', 'o', 's', '>'},
            new int[] {256});

    @Override
    public void checkGrammar(String grammar) {
        GRAMMARS.check(grammar);
    }

    @Override
    public void checkJsonSchema(String schema) {
        GRAMMARS.checkJsonSchema(schema);
    }

    @Override
    public Generation openGeneration(GenerationConfig config, OutputSpec output) {
        if (!this.available) throw new InferenceUnavailableException("inference engine is shutting down");
        Script script = this.script;
        if (output.reasoning() && this.emptyReasoning) script = thinkingNothing(script);
        var generation = new ScriptedGeneration(config, script, this.workers);
        generation.grammar = output.grammar();
        generation.output = output;
        this.generations.add(generation);
        return generation;
    }

    static final class ScriptedGeneration implements Generation {
        final GenerationConfig config;
        final CountDownLatch started = new CountDownLatch(1);
        final CountDownLatch closed = new CountDownLatch(1);
        final AtomicInteger emitted = new AtomicInteger();
        final AtomicInteger prefills = new AtomicInteger();
        private volatile Runnable prefilled = () -> {};
        /// Chunks emitted before the first one holding `</think>`; -1 until one does.
        volatile int reasoningChunks = -1;
        final AtomicInteger emittedAfterCancel = new AtomicInteger();
        final AtomicInteger closeCount = new AtomicInteger();
        private final Script script;
        private final AtomicBoolean cancelled = new AtomicBoolean();
        volatile String prompt;
        volatile String grammar;
        volatile OutputSpec output;
        volatile int maxNewTokens;
        volatile String generatingThread;

        private final Executor workers;

        private ScriptedGeneration(GenerationConfig config, Script script, Executor workers) {
            this.config = config;
            this.script = script;
            this.workers = workers;
        }

        @Override
        public CompletableFuture<Result> generate(
                EncodedPrompt prompt, int maxNewTokens, Consumer<String> text, Runnable prefilled) {
            if (this.closeCount.get() > 0) throw new IllegalStateException("Qwen generation session is closed");
            if (this.cancelled.get()) throw new IllegalStateException("Qwen generation session is cancelled");
            this.prompt = prompt.text();
            this.maxNewTokens = maxNewTokens;
            return CompletableFuture.supplyAsync(
                    () -> {
                        this.generatingThread = Thread.currentThread().getName();
                        this.prefilled = prefilled;
                        this.started.countDown();
                        try {
                            Result result = this.script.run(this, maxNewTokens, text);
                            if (this.output == null || !this.output.reasoning()) return result;
                            // One token per chunk: the reasoning is what came before `</think>`.
                            int reasoning = this.reasoningChunks >= 0
                                    ? this.reasoningChunks
                                    : result.completionTokens() - (result.stopTokenReached() ? 1 : 0);
                            return new Result(result.completionTokens(), result.stopTokenReached(), reasoning);
                        } catch (InterruptedException | ExecutionException failure) {
                            throw new java.util.concurrent.CompletionException(failure);
                        }
                    },
                    this.workers);
        }

        /// One retired prefill quantum, as the engine reports it.
        void prefill() {
            this.prefills.incrementAndGet();
            this.prefilled.run();
        }

        void emit(Consumer<String> output, String text) {
            if (this.cancelled.get()) this.emittedAfterCancel.incrementAndGet();
            if (this.reasoningChunks < 0 && text.contains("</think>")) this.reasoningChunks = this.emitted.get();
            this.emitted.incrementAndGet();
            output.accept(text);
        }

        @Override
        public void cancel() {
            this.cancelled.set(true);
        }

        @Override
        public boolean isCancelled() {
            return this.cancelled.get();
        }

        @Override
        public void close() {
            this.closeCount.incrementAndGet();
            this.closed.countDown();
        }

        boolean awaitClosed() throws InterruptedException {
            return this.closed.await(10, TimeUnit.SECONDS);
        }
    }
}
