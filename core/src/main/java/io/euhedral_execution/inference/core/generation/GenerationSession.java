package io.euhedral_execution.inference.core.generation;

import io.euhedral_execution.inference.core.tokenizer.TokenConstraint;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

/// One persistent sequence being generated from, whatever model runs it: what the engine's callers need of a session.
/// The dense Qwen3.8 session and the Flash-Next session implement it; neither is visible through it.
public interface GenerationSession extends AutoCloseable {

    /// Whether the session's next prompt is its first, which carries the model special tokens.
    boolean expectsFirstPrompt();

    /// Starts a generation without blocking: the future completes with the IDs this call sampled, a sampled
    /// terminator included. `text` receives newly decoded, non-empty text, one call at a time and in order, and
    /// must not block. A session runs one generation at a time; a failed or cancelled one leaves it cancelled.
    CompletableFuture<List<Integer>> generateAsync(
            int[] promptTokenIds,
            int maxNewTokens,
            Consumer<String> text,
            TokenConstraint constraint,
            GenerationTimingListener timing);

    /// Requests cancellation of the current quantum or prevents the next one from starting.
    void cancel();

    boolean isCancelled();

    boolean isClosed();

    /// The authoritative token position retained by the session's sequence state.
    long currentTokenPosition();

    /// An immutable snapshot of the tokens sampled by all prompts in this session.
    List<Integer> generatedTokenIds();

    /// Prompt tokens the session's last generation restored from a prefix cache instead of prefilling.
    int restoredPromptTokens();

    /// Releases the sequence; waits for an active generation to end after cancelling it.
    @Override
    void close();
}
