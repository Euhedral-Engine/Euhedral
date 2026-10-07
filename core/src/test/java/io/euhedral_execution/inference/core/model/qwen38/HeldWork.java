package io.euhedral_execution.inference.core.model.qwen38;

/// Work on a sequence that a test holds in flight while it sets the sequence's state by hand, as a quantum would,
/// then commits.
public final class HeldWork implements Sequence.Work {

    private final Sequence sequence;
    private final long at;
    private volatile boolean ready;
    private long next;

    private HeldWork(Sequence sequence, long at) {
        this.sequence = sequence;
        this.at = at;
    }

    /// Admits held work at the sequence's submitted frontier.
    public static HeldWork admit(Sequence sequence) {
        long at = sequence.submittedFrontier();
        var work = new HeldWork(sequence, at);
        sequence.admit(work, at, at);
        return work;
    }

    /// Completes the work; it commits the sequence at `next` once it concludes.
    public void commit(long next) {
        this.next = next;
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
        if (blocked != null) throw new IllegalStateException("held work was blocked", blocked);
        this.sequence.commit(this.next);
    }
}
