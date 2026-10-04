package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.api.chat.QwenChatTemplate;
import io.euhedral_execution.inference.core.InferenceEngine;
import io.euhedral_execution.inference.core.guidance.GrammarCompiler;
import io.euhedral_execution.inference.core.guidance.Llguidance;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.QwenGenerationSession;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.core.tokenizer.ReasoningConstraint;
import io.euhedral_execution.inference.core.tokenizer.TokenConstraint;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.function.Consumer;

/// Borrows the Spring-owned `InferenceEngine`; the engine bean remains the terminal owner of model,
/// GPU, lattice, and runtime. Each generation wraps one engine-tracked `QwenGenerationSession`. Constrained
/// answers use llguidance, loaded from beside the CUDA library and compiled against the checkpoint vocabulary.
public final class EngineInferenceBackend implements InferenceBackend, AutoCloseable {
    private final InferenceEngine engine;
    private final String modelId;
    /// Each task is one frame on the engine's lattice.
    private final Executor workers;
    private final GrammarCompiler grammars;

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
        this.grammars = GrammarCompiler.forTokenizer(
                Llguidance.load(Llguidance.besideLibrary(engine.config().cudaLibraryPath())),
                engine.tokenizer(),
                engine.modelConfig().vocabSize());
    }

    @Override
    public void checkGrammar(String grammar) {
        this.grammars.check(grammar);
    }

    @Override
    public void checkJsonSchema(String schema) {
        this.grammars.checkJsonSchema(schema);
    }

    @Override
    public void close() {
        this.grammars.close();
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
    public Generation openGeneration(GenerationConfig config, OutputSpec output) {
        if (this.engine.isClosed()) throw new InferenceUnavailableException("inference engine is shutting down");
        try {
            QwenTokenizer tokenizer = this.engine.tokenizer();
            TokenConstraint answer = output.grammar() == null ? null : this.grammars.constraint(output.grammar());
            // Free, uncapped reasoning needs no constraint, so a greedy request keeps speculative decoding.
            boolean capped = output.reasoning() && output.reasoningBudget() < Integer.MAX_VALUE;
            TokenConstraint constraint = output.reasoning() && (answer != null || capped)
                    ? new ReasoningConstraint(tokenizer, output.reasoningBudget(), answer)
                    : answer;
            int thinkEnd = output.reasoning()
                    ? tokenizer.controlTokenId(QwenChatTemplate.THINK_END).orElseThrow()
                    : -1;
            try {
                return new SessionGeneration(this.engine.createSession(config), tokenizer, constraint, thinkEnd);
            } catch (RuntimeException | Error failure) {
                if (constraint != null) constraint.close();
                throw failure;
            }
        } catch (IllegalStateException closed) {
            // createSession's only state failure is closed admission; anything else is a real fault.
            if (this.engine.isClosed()) throw new InferenceUnavailableException("inference engine is shutting down");
            throw closed;
        }
    }

    /// `thinkEnd` is the `</think>` ID when the output opens as reasoning, else -1.
    private record SessionGeneration(
            QwenGenerationSession session, QwenTokenizer tokenizer, TokenConstraint constraint, int thinkEnd)
            implements Generation {

        @Override
        public CompletableFuture<Result> generate(EncodedPrompt prompt, int maxNewTokens, Consumer<String> text) {
            if (!this.session.expectsFirstPrompt())
                throw new IllegalStateException("an encoded prompt needs a fresh session");
            return this.session
                    .generateAsync(prompt.tokenIds(), maxNewTokens, text, this.constraint)
                    .thenApply(tokenIds -> {
                        boolean stopped =
                                !tokenIds.isEmpty() && this.tokenizer.isGenerationEosToken(tokenIds.getLast());
                        return new Result(
                                tokenIds.size(),
                                stopped,
                                reasoningTokens(tokenIds, stopped),
                                this.session.restoredPromptTokens());
                    });
        }

        /// The tokens before `</think>`; all but a terminator when the reasoning never ended.
        private int reasoningTokens(List<Integer> tokenIds, boolean stopped) {
            if (this.thinkEnd < 0) return 0;
            int end = tokenIds.indexOf(this.thinkEnd);
            return end >= 0 ? end : tokenIds.size() - (stopped ? 1 : 0);
        }

        @Override
        public void cancel() {
            this.session.cancel();
        }

        @Override
        public int cachedPromptTokens() {
            return this.session.restoredPromptTokens();
        }

        @Override
        public boolean isCancelled() {
            return this.session.isCancelled();
        }

        /// The session's close waits for its generation to end, after which nothing uses the constraint.
        @Override
        public void close() {
            try {
                this.session.close();
            } finally {
                if (this.constraint != null) this.constraint.close();
            }
        }
    }
}
