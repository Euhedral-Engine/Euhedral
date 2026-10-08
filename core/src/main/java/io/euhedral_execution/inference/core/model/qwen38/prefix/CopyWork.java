package io.euhedral_execution.inference.core.model.qwen38.prefix;

import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import java.util.function.Consumer;

/// A prefix-cache capture or restore on a sequence: admitted like a quantum, and concluded in the sequence's
/// admission order once its copies completed.
final class CopyWork implements Sequence.Work {

    private final Sequence sequence;
    private final long at;
    private volatile boolean ready;
    private Consumer<Throwable> conclusion;

    CopyWork(Sequence sequence, long at) {
        this.sequence = sequence;
        this.at = at;
    }

    /// Admits the work at its position, covering positions up to `end`.
    void admit(long end) {
        this.sequence.admit(this, this.at, end);
    }

    /// Completes the work. `conclusion` runs in admission order, given why the work can no longer commit (or null):
    /// it settles the work ([Sequence#commit] or [Sequence#abandon]) and throws the work's next frame. It must not
    /// throw.
    void complete(Consumer<Throwable> conclusion) {
        this.conclusion = conclusion;
        this.ready = true;
        this.sequence.drain();
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
        this.conclusion.accept(blocked);
    }
}
