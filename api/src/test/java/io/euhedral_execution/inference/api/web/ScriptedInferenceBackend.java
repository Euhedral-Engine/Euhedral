package io.euhedral_execution.inference.api.web;

import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.api.engine.InferenceUnavailableException;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/// Test backend that follows `QwenGenerationSession`'s callback contract without CUDA: text is emitted on the
/// generating thread between "quanta", cancellation stops the next quantum, and generations record their
/// inputs and lifecycle so tests can assert cancellation and cleanup.
final class ScriptedInferenceBackend implements InferenceBackend {
    static final String MODEL_ID = "euhedral-test-model";

    final List<ScriptedGeneration> generations = new CopyOnWriteArrayList<>();
    volatile boolean available = true;
    volatile Script script = tokens(List.of("Hello", ", ", "world"), true);
    volatile int contextLength = 4096;

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

    void reset() {
        this.generations.clear();
        this.available = true;
        this.contextLength = 4096;
        this.script = tokens(List.of("Hello", ", ", "world"), true);
    }

    ScriptedGeneration only() {
        if (this.generations.size() != 1)
            throw new AssertionError("expected one generation, found " + this.generations.size());
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
    public EncodedPrompt encodePrompt(String prompt) {
        this.encoded.incrementAndGet();
        return new EncodedPrompt(prompt, new int[countPromptTokens(prompt)]);
    }

    @Override
    public Generation openGeneration(GenerationConfig config, ToolConstraint constraint) {
        if (!this.available) throw new InferenceUnavailableException("inference engine is shutting down");
        var generation = new ScriptedGeneration(config, this.script);
        generation.constraint = constraint;
        this.generations.add(generation);
        return generation;
    }

    static final class ScriptedGeneration implements Generation {
        final GenerationConfig config;
        final CountDownLatch started = new CountDownLatch(1);
        final CountDownLatch closed = new CountDownLatch(1);
        final AtomicInteger emitted = new AtomicInteger();
        final AtomicInteger emittedAfterCancel = new AtomicInteger();
        final AtomicInteger closeCount = new AtomicInteger();
        private final Script script;
        private final AtomicBoolean cancelled = new AtomicBoolean();
        volatile String prompt;
        volatile ToolConstraint constraint;
        volatile int maxNewTokens;
        volatile String generatingThread;

        private ScriptedGeneration(GenerationConfig config, Script script) {
            this.config = config;
            this.script = script;
        }

        @Override
        public Result generate(EncodedPrompt prompt, int maxNewTokens, Consumer<String> output)
                throws InterruptedException, ExecutionException {
            if (this.closeCount.get() > 0) throw new IllegalStateException("Qwen generation session is closed");
            if (this.cancelled.get()) throw new IllegalStateException("Qwen generation session is cancelled");
            this.prompt = prompt.text();
            this.maxNewTokens = maxNewTokens;
            this.generatingThread = Thread.currentThread().getName();
            this.started.countDown();
            return this.script.run(this, maxNewTokens, output);
        }

        void emit(Consumer<String> output, String text) {
            if (this.cancelled.get()) this.emittedAfterCancel.incrementAndGet();
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
