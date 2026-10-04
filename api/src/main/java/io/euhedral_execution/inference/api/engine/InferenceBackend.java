package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.function.Consumer;

/// The API layer's narrow view of the loaded model.
///
/// Production borrows the single `InferenceEngine`; tests substitute a scripted backend to exercise HTTP
/// behavior without CUDA. Implementations never expose engine internals to HTTP code.
public interface InferenceBackend {

    /// Public model ID accepted in requests and listed by `/v1/models`.
    String modelId();

    /// False once engine shutdown has begun; no further generation is admitted.
    boolean isAvailable();

    /// Maximum sequence length in tokens (prompt plus completion).
    int contextLength();

    /// A rendered prompt and its token IDs, encoded once, exactly as a new generation encodes a first prompt.
    record EncodedPrompt(String text, int[] tokenIds) {
        public EncodedPrompt {
            java.util.Objects.requireNonNull(text, "text");
            tokenIds = tokenIds.clone();
        }

        @Override
        public int[] tokenIds() {
            return this.tokenIds.clone();
        }

        public int tokenCount() {
            return this.tokenIds.length;
        }
    }

    /// Runs request work as tasks on the backend's workers (the lattice's frames). Every piece of a request's
    /// work runs there; a task may block, and the other workers take over the rest of the work meanwhile.
    Executor workers();

    /// Encodes a rendered prompt for [#openGeneration] on the workers: the plan counts its tokens and the
    /// generation runs them, so a prompt is tokenized once.
    CompletableFuture<EncodedPrompt> encodePrompt(String prompt);

    /// How a generation's output is shaped. With `reasoning` the prompt opened the model's think block, so the
    /// output is reasoning up to `</think>` and the answer after it. A non-null `grammar` (llguidance Lark)
    /// constrains the answer; a backend must enforce it while sampling.
    record OutputSpec(boolean reasoning, String grammar) {
        public static final OutputSpec TEXT = new OutputSpec(false, null);
    }

    /// Compiles an answer grammar (llguidance Lark) without generating; throws
    /// [io.euhedral_execution.inference.core.guidance.GrammarException] when it cannot be enforced. A compiled
    /// grammar is kept for the generation that uses it.
    void checkGrammar(String grammar);

    /// Compiles a JSON Schema on its own, so a refusal names the schema's problem.
    void checkJsonSchema(String schema);

    /// Opens one request-owned generation. The caller must close it on every path.
    ///
    /// @throws InferenceUnavailableException when the engine is closing or closed
    default Generation openGeneration(GenerationConfig config) {
        return openGeneration(config, OutputSpec.TEXT);
    }

    Generation openGeneration(GenerationConfig config, OutputSpec output);

    /// One request's sequence state. `cancel` may be called from any thread, including the output
    /// callback; `close` releases the sequence and waits for an in-flight quantum to detach.
    interface Generation extends AutoCloseable {

        /// Starts prefill and decode on the workers and returns at once. `text` receives newly decoded,
        /// non-empty text on a worker, one call at a time and in order, and must not block; the future
        /// completes on a worker after the last text.
        CompletableFuture<Result> generate(EncodedPrompt prompt, int maxNewTokens, Consumer<String> text);

        void cancel();

        /// True when cancellation was requested by any party, including engine shutdown.
        boolean isCancelled();

        @Override
        void close();
    }

    /// `completionTokens` counts every sampled token, including a terminating stop token; `reasoningTokens` counts
    /// those of the reasoning, before `</think>`.
    record Result(int completionTokens, boolean stopTokenReached, int reasoningTokens) {
        public Result(int completionTokens, boolean stopTokenReached) {
            this(completionTokens, stopTokenReached, 0);
        }
    }
}
