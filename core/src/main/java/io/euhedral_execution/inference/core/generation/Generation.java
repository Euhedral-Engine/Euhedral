package io.euhedral_execution.inference.core.generation;

import io.euhedral_execution.core.frames.AbstractFrame;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/// One call of a session's generation, as frames: [Admit] starts a step, the step's quantum throws [Select] when it
/// retires, and [Select] throws the next [Admit] or [Finish]. One frame of the chain is live at a time and each is
/// thrown by the one before, so the generation's state needs no lock: subclasses keep it in plain fields.
///
/// A generation is one unit of its lake from construction until [Finish], so the lake cannot complete under it.
public class Generation {

    private final GenerationFrames frames;
    private final CompletableFuture<List<Integer>> result = new CompletableFuture<>();
    private List<Integer> tokens = List.of();
    private Throwable failure;

    public Generation(GenerationFrames frames) {
        this.frames = Objects.requireNonNull(frames, "frames");
        frames.lake().admit();
    }

    /// The caller's view: completes after [#ended] ran, with the tokens or the failure.
    public final CompletableFuture<List<Integer>> result() {
        return this.result;
    }

    /// Starts the chain at `first`, from any thread: throws its [Admit].
    public final void start(StepPort first) {
        this.frames.publish(this.frames.admit(this, Objects.requireNonNull(first, "first")));
    }

    /// Ends the generation without another step: throws its [Finish].
    public final void finishNow() {
        this.frames.publish(this.frames.finish(this));
    }

    /// Throws `select` for a step that needed no quantum, from any thread.
    public final void skip(AbstractFrame select) {
        this.frames.publish(select);
    }

    /// The tokens the caller receives when the generation ends without failure.
    public final void complete(List<Integer> tokens) {
        this.tokens = Objects.requireNonNull(tokens, "tokens");
    }

    /// Ends the generation with `cause`; a later failure is suppressed into the first.
    public final void fail(Throwable cause) {
        Objects.requireNonNull(cause, "cause");
        if (this.failure == null) this.failure = cause;
        else if (this.failure != cause) this.failure.addSuppressed(cause);
    }

    public final Throwable failure() {
        return this.failure;
    }

    /// The session's bookkeeping once the generation ended, before its result completes (the session may then
    /// accept another generation). Runs in [Finish]; must not block.
    protected void ended(List<Integer> tokens, Throwable failure) {}

    final void conclude() {
        try {
            ended(this.tokens, this.failure);
        } catch (Throwable endFailure) {
            fail(endFailure);
        }
        try {
            this.frames.lake().terminated();
        } finally {
            if (this.failure != null) this.result.completeExceptionally(this.failure);
            else this.result.complete(this.tokens);
        }
    }
}
