package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

/// The API layer's narrow view of the loaded model.
///
/// Production borrows the single `InferenceEngine`; tests substitute a scripted backend to exercise HTTP
/// behavior without CUDA. Implementations never expose engine internals to HTTP code.
public interface InferenceBackend {

    /// Validated offered names and whether calls are required or may be parallel this turn.
    record ToolConstraint(List<String> toolNames, boolean requiresCall, boolean parallel) {
        public ToolConstraint {
            toolNames = List.copyOf(toolNames);
            if (toolNames.isEmpty()) throw new IllegalArgumentException("constrained generation needs offered tools");
        }

        public ToolConstraint(List<String> toolNames, boolean requiresCall) {
            this(toolNames, requiresCall, true);
        }
    }

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

    /// Encodes a rendered prompt for [#openGeneration]: the plan counts its tokens and the generation runs them,
    /// so a prompt is tokenized once.
    EncodedPrompt encodePrompt(String prompt) throws InterruptedException;

    /// Opens one request-owned generation. The caller must close it on every path.
    ///
    /// @throws InferenceUnavailableException when the engine is closing or closed
    default Generation openGeneration(GenerationConfig config) {
        return openGeneration(config, null);
    }

    /// A non-null constraint is enforced while sampling; a backend must not ignore it.
    Generation openGeneration(GenerationConfig config, ToolConstraint constraint);

    /// One request's sequence state. `cancel` may be called from any thread, including the output
    /// callback; `close` releases the sequence and waits for an in-flight quantum to detach.
    interface Generation extends AutoCloseable {

        /// Runs prefill and decode on the calling thread. `output` receives newly decoded, non-empty text
        /// on that same thread between quanta.
        Result generate(EncodedPrompt prompt, int maxNewTokens, Consumer<String> output)
                throws InterruptedException, ExecutionException;

        void cancel();

        /// True when cancellation was requested by any party, including engine shutdown.
        boolean isCancelled();

        @Override
        void close();
    }

    /// `completionTokens` counts every sampled token, including a terminating stop token.
    record Result(int completionTokens, boolean stopTokenReached) {}
}
