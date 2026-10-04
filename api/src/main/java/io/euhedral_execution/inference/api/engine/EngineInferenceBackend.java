package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.api.chat.QwenChatTemplate;
import io.euhedral_execution.inference.core.InferenceEngine;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.QwenGenerationSession;
import io.euhedral_execution.inference.core.tokenizer.JsonEnvelopeConstraint;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.function.Consumer;

/// Borrows the Spring-owned `InferenceEngine`; the engine bean remains the terminal owner of model,
/// GPU, lattice, and runtime. Each generation wraps one engine-tracked `QwenGenerationSession`.
public final class EngineInferenceBackend implements InferenceBackend {
    private final InferenceEngine engine;
    private final String modelId;
    /// Each task is one frame on the engine's lattice.
    private final Executor workers;

    public EngineInferenceBackend(InferenceEngine engine, String modelId, QwenChatTemplate chatTemplate) {
        this.engine = Objects.requireNonNull(engine, "engine");
        this.modelId = Objects.requireNonNull(modelId, "modelId");
        this.workers = task -> engine.onWorker(() -> {
            task.run();
            return null;
        });
        // The formatter emits these as control tokens; plain-text encoding would silently corrupt prompts.
        for (String token : chatTemplate.controlTokens()) {
            if (engine.tokenizer().controlTokenId(token).isEmpty())
                throw new IllegalStateException("tokenizer lacks chat-template control token " + token);
        }
    }

    @Override
    public String modelId() {
        return this.modelId;
    }

    @Override
    public boolean isAvailable() {
        return !this.engine.isClosed();
    }

    @Override
    public int contextLength() {
        return Math.min(
                this.engine.config().maxContextTokens(),
                this.engine.modelConfig().maxPositionEmbeddings());
    }

    @Override
    public Executor workers() {
        return this.workers;
    }

    @Override
    public CompletableFuture<EncodedPrompt> encodePrompt(String prompt) {
        // Every generation runs on a fresh session, whose first prompt carries the model special tokens.
        return this.engine.tokenizePromptAsync(prompt).thenApply(ids -> new EncodedPrompt(prompt, ids));
    }

    @Override
    public Generation openGeneration(GenerationConfig config, ToolConstraint constraint) {
        if (this.engine.isClosed()) throw new InferenceUnavailableException("inference engine is shutting down");
        try {
            JsonEnvelopeConstraint grammar = constraint == null
                    ? null
                    : new JsonEnvelopeConstraint(
                            this.engine.tokenizer(),
                            constraint.toolNames(),
                            constraint.requiresCall(),
                            constraint.parallel());
            return new SessionGeneration(this.engine.createSession(config), this.engine.tokenizer(), grammar);
        } catch (IllegalStateException closed) {
            // createSession's only state failure is closed admission; anything else is a real fault.
            if (this.engine.isClosed()) throw new InferenceUnavailableException("inference engine is shutting down");
            throw closed;
        }
    }

    private record SessionGeneration(
            QwenGenerationSession session, QwenTokenizer tokenizer, JsonEnvelopeConstraint constraint)
            implements Generation {

        @Override
        public CompletableFuture<Result> generate(EncodedPrompt prompt, int maxNewTokens, Consumer<String> text) {
            if (!this.session.expectsFirstPrompt())
                throw new IllegalStateException("an encoded prompt needs a fresh session");
            return this.session
                    .generateAsync(prompt.tokenIds(), maxNewTokens, text, this.constraint)
                    .thenApply(tokenIds -> new Result(
                            tokenIds.size(),
                            !tokenIds.isEmpty() && this.tokenizer.isGenerationEosToken(tokenIds.getLast())));
        }

        @Override
        public void cancel() {
            this.session.cancel();
        }

        @Override
        public boolean isCancelled() {
            return this.session.isCancelled();
        }

        @Override
        public void close() {
            this.session.close();
        }
    }
}
