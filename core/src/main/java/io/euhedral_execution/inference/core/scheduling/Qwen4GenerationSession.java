package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.qwen4.Qwen4Executor;
import io.euhedral_execution.inference.core.qwen4.Qwen4Sequence;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.tokenizer.IncrementalDecoder;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.core.tokenizer.TokenConstraint;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/// One persistent Flash-Next sequence being generated from: prefill chunks, then one decode step per token, with the
/// engine's sampler, constraint and text decoding around the model. The model is ordinary autoregressive text
/// generation: every token is the target model's own.
///
/// A generation runs as one task on the engine's Flash-Next worker, which owns the executor's single stream, so
/// generations of different sessions are serialized and the text callback runs on that worker, one call at a time and
/// in order. The session never knows the model is sparse: it hands token ids to the executor and takes a logits row
/// back.
public final class Qwen4GenerationSession implements GenerationSession {

    /// Prompt tokens per prefill step (the executor's chunk).
    public static final int DEFAULT_PREFILL_CHUNK_TOKENS = Qwen4Executor.MAX_ROWS;

    private final QwenTokenizer tokenizer;
    private final Qwen4Executor executor;
    private final Executor worker;
    private final int prefillChunkTokens;
    private final QwenLogitsSampler sampler;
    private final QwenHostLogits hostLogits;
    private final Qwen4Executor.LogitsSink sink;
    private final List<Integer> generatedTokenIds = new ArrayList<>();
    private final AtomicBoolean generationActive = new AtomicBoolean();
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicBoolean closed = new AtomicBoolean();
    private volatile CompletableFuture<List<Integer>> activeGeneration;
    private Consumer<? super Qwen4GenerationSession> closeListener;

    private Qwen4Sequence sequence;
    private IncrementalDecoder decoder;
    private boolean decoderFinished;
    private boolean promptPrefilled;

    public Qwen4GenerationSession(
            QwenTokenizer tokenizer,
            Qwen4Executor executor,
            Executor worker,
            ExecutionGpu gpu,
            GenerationConfig config,
            int prefillChunkTokens,
            Consumer<? super Qwen4GenerationSession> closeListener) {
        if (prefillChunkTokens <= 0 || prefillChunkTokens > Qwen4Executor.MAX_ROWS)
            throw new IllegalArgumentException("prefillChunkTokens must be in 1.." + Qwen4Executor.MAX_ROWS);
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.executor = Objects.requireNonNull(executor, "executor");
        this.worker = Objects.requireNonNull(worker, "worker");
        this.prefillChunkTokens = prefillChunkTokens;
        this.closeListener = Objects.requireNonNull(closeListener, "closeListener");
        Objects.requireNonNull(config, "config");
        this.sampler = new QwenLogitsSampler(config, executor.vocabularySize());
        this.hostLogits = new QwenHostLogits(Objects.requireNonNull(gpu, "gpu"), executor.vocabularySize());
        this.sink = address -> this.hostLogits.queueFinalRow(address, 1);
        this.decoder = tokenizer.newIncrementalDecoder();
    }

    @Override
    public boolean expectsFirstPrompt() {
        return !this.promptPrefilled;
    }

    @Override
    public CompletableFuture<List<Integer>> generateAsync(
            int[] promptTokenIds,
            int maxNewTokens,
            Consumer<String> text,
            TokenConstraint constraint,
            GenerationTimingListener timing) {
        Objects.requireNonNull(promptTokenIds, "promptTokenIds");
        Objects.requireNonNull(text, "text");
        if (maxNewTokens < 0) throw new IllegalArgumentException("maxNewTokens must not be negative");
        if (promptTokenIds.length == 0) throw new IllegalArgumentException("prompt must encode to at least one token");
        if (!this.generationActive.compareAndSet(false, true))
            throw new IllegalStateException("a generation is already active for this Flash-Next session");
        CompletableFuture<List<Integer>> finished = new CompletableFuture<>();
        this.activeGeneration = finished;
        try {
            ensureUsable();
            int[] prompt = promptTokenIds.clone();
            this.worker.execute(() -> {
                List<Integer> result = null;
                Throwable failure = null;
                try {
                    result = generate(prompt, maxNewTokens, text, constraint, timing);
                } catch (Throwable thrown) {
                    failure = thrown;
                    requestCancellation();
                } finally {
                    this.generationActive.set(false);
                    if (this.closed.get()) completeClose();
                }
                if (failure != null) finished.completeExceptionally(failure);
                else finished.complete(result);
            });
        } catch (RuntimeException | Error failure) {
            this.generationActive.set(false);
            finished.completeExceptionally(failure);
            throw failure;
        }
        return finished;
    }

    /// The whole generation, on the worker: prefill in chunks, sample, decode one token at a time.
    private List<Integer> generate(
            int[] prompt,
            int maxNewTokens,
            Consumer<String> text,
            TokenConstraint constraint,
            GenerationTimingListener timing)
            throws InterruptedException {
        if (isStopRequested()) return List.of();
        if (this.decoderFinished) {
            this.decoder = this.tokenizer.newIncrementalDecoder();
            this.decoderFinished = false;
        }
        if (this.sequence == null) this.sequence = this.executor.newSequence();
        if (timing != null) timing.promptEncoded(System.nanoTime(), prompt.length);
        // A greedy, unconstrained call selects each token on the device and reads back only its ID.
        this.hostLogits.selectOnDevice(this.sampler.greedy() && constraint == null);
        List<Integer> callTokens = new ArrayList<>();

        int next = -1;
        int offset = 0;
        while (offset < prompt.length) {
            if (isStopRequested()) return List.of();
            int rows = Math.min(this.prefillChunkTokens, prompt.length - offset);
            boolean samples = offset + rows == prompt.length && maxNewTokens > 0;
            long started = System.nanoTime();
            step(prompt, offset, rows, samples);
            long executed = System.nanoTime();
            if (timing != null) timing.prefillQuantum(started, executed, rows);
            if (samples) {
                next = select(constraint);
                if (timing != null) timing.firstTokenSelected(System.nanoTime(), next);
            }
            offset += rows;
        }
        this.promptPrefilled = true;
        if (maxNewTokens == 0) {
            finishDecoder(text);
            return List.of();
        }

        boolean endedNormally = false;
        for (int generated = 0; generated < maxNewTokens && !isStopRequested(); generated++) {
            int tokenId = next;
            if (this.tokenizer.isGenerationEosToken(tokenId)) {
                record(callTokens, tokenId);
                endedNormally = true;
                break;
            }
            boolean anotherTokenAllowed = generated + 1 < maxNewTokens;
            record(callTokens, tokenId);
            emit(text, this.decoder.append(tokenId));
            if (isStopRequested()) break;
            long started = System.nanoTime();
            step(new int[] {tokenId}, 0, 1, anotherTokenAllowed);
            long executed = System.nanoTime();
            if (anotherTokenAllowed) {
                next = select(constraint);
                if (timing != null) timing.decodeQuantum(started, executed, System.nanoTime(), true, next);
            } else {
                endedNormally = !isStopRequested();
                if (timing != null) timing.decodeQuantum(started, executed, executed, false, -1);
            }
        }
        if (endedNormally && !isStopRequested()) finishDecoder(text);
        return List.copyOf(callTokens);
    }

    private void step(int[] tokens, int offset, int rows, boolean wantsLogits) throws InterruptedException {
        boolean succeeded = false;
        try {
            this.executor.step(this.sequence, tokens, offset, rows, wantsLogits ? this.sink : null);
            succeeded = true;
        } finally {
            if (wantsLogits) this.hostLogits.retired(succeeded);
        }
    }

    private int select(TokenConstraint constraint) {
        int selected = this.sampler.selectToken(this.hostLogits, constraint);
        if (constraint != null) constraint.accept(selected);
        return selected;
    }

    private void record(List<Integer> callTokens, int tokenId) {
        callTokens.add(tokenId);
        synchronized (this.generatedTokenIds) {
            this.generatedTokenIds.add(tokenId);
        }
    }

    private void finishDecoder(Consumer<String> text) {
        if (this.decoderFinished) return;
        String remaining = this.decoder.finish();
        this.decoderFinished = true;
        emit(text, remaining);
    }

    private static void emit(Consumer<String> text, String decoded) {
        if (!decoded.isEmpty()) text.accept(decoded);
    }

    @Override
    public void cancel() {
        if (this.closed.get()) return;
        requestCancellation();
    }

    @Override
    public boolean isCancelled() {
        return this.cancelled.get();
    }

    @Override
    public boolean isClosed() {
        return this.closed.get();
    }

    @Override
    public long currentTokenPosition() {
        Qwen4Sequence current = this.sequence;
        return current == null ? 0 : current.position();
    }

    @Override
    public List<Integer> generatedTokenIds() {
        synchronized (this.generatedTokenIds) {
            return List.copyOf(this.generatedTokenIds);
        }
    }

    /// The model keeps no prefix cache: every prompt is prefilled whole.
    @Override
    public int restoredPromptTokens() {
        return 0;
    }

    @Override
    public void close() {
        boolean firstClose = this.closed.compareAndSet(false, true);
        if (this.generationActive.get()) {
            if (firstClose) requestCancellation();
            CompletableFuture<List<Integer>> active = this.activeGeneration;
            if (active != null) active.handle((tokens, failure) -> null).join();
        }
        completeClose();
    }

    private synchronized void completeClose() {
        if (this.generationActive.get()) return;
        Qwen4Sequence held = this.sequence;
        this.sequence = null;
        try {
            if (held != null) held.close();
        } finally {
            this.hostLogits.close();
            if (this.closeListener != null) {
                this.closeListener.accept(this);
                this.closeListener = null;
            }
        }
    }

    private void ensureUsable() {
        if (this.closed.get()) throw new IllegalStateException("Flash-Next generation session is closed");
        if (this.cancelled.get()) throw new IllegalStateException("Flash-Next generation session is cancelled");
    }

    private boolean isStopRequested() {
        return this.cancelled.get() || this.closed.get();
    }

    private void requestCancellation() {
        this.cancelled.set(true);
    }
}
