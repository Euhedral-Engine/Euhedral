package io.euhedral_execution.inference.core.model.qwen38.prefix;

import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.function.Function;

/// A prefix-cache capture or restore on a sequence: admitted like a quantum, and concluded in the sequence's
/// admission order once its copies completed.
final class CopyWork<T> implements Sequence.Work {

    private final Sequence sequence;
    private final long at;
    private final CompletableFuture<T> concluded = new CompletableFuture<>();
    private volatile boolean ready;
    private Function<Throwable, T> conclusion;

    CopyWork(Sequence sequence, long at) {
        this.sequence = sequence;
        this.at = at;
    }

    /// Admits the work at its position, covering positions up to `end`.
    void admit(long end) {
        this.sequence.admit(this, this.at, end);
    }

    /// Completes the work. `conclusion` runs in admission order, given why the work can no longer commit (or null);
    /// it settles the work ([Sequence#commit] or [Sequence#abandon]) and returns its result, or throws its failure.
    CompletableFuture<T> complete(Function<Throwable, T> conclusion) {
        this.conclusion = conclusion;
        this.ready = true;
        this.sequence.drain();
        return this.concluded;
    }

    @Override
    public boolean ready() {
        return this.ready;
    }

    @Override
    public long admittedAt() {
        return this.at;
    }

    @Override
    public void concluded(Throwable blocked) {
        try {
            this.concluded.complete(this.conclusion.apply(blocked));
        } catch (Throwable failure) {
            this.concluded.completeExceptionally(
                    failure instanceof CompletionException ? failure : new CompletionException(failure));
        }
    }
}
