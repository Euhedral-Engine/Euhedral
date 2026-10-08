package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.runtime.graph.SequenceOwner;
import io.euhedral_execution.inference.core.runtime.graph.Sequencer;
import java.lang.invoke.VarHandle;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/// Request-lifetime state shared by successive Qwen execution quanta.
///
/// Parallel execution, ordered completion. Work on the sequence (its quanta, and the prefix cache's captures and
/// restores) is admitted in order at the submitted frontier and may be in flight together. Each piece completes in
/// any order and concludes in admission order, through a [Sequencer]: whichever thread finds the oldest piece
/// complete concludes the ready ones, one thread at a time. A piece concludes by settling: it commits (the committed
/// frontier moves to where it left the sequence) or it is abandoned. A piece is told at its conclusion when it can no
/// longer commit: a piece before it failed the sequence, or committed short of where it starts.
///
/// The frontiers are written by one writer each: admission (the generation chain) moves the submitted frontier, and
/// the conclusion of a piece, which runs on one thread at a time, moves the committed one. Any thread reads them, so
/// they are volatile; there is no lock and no CAS state machine. Cancellation comes from any thread: [#cancel()]
/// publishes its flag and then reads whether work is in flight, and settlement publishes the settled count and then
/// reads the flag, so one of the two ends the sequence CANCELLED. The terminal state is first-writer-wins.
/// Persistent state (GDN buffers, KV pages) closes only in [#complete()], which the session's lifecycle runs after
/// its generation ended.
public final class Sequence {

    public enum TerminalState {
        ACTIVE,
        CANCELLED,
        FAILED,
        COMPLETED
    }

    /// A piece of work on the sequence, concluded in admission order.
    public interface Work extends Sequencer.Entry {
        /// The position it was admitted at.
        long admittedAt();

        /// Concludes the work once everything admitted before it concluded; it then calls [#commit(long)] or
        /// [#abandon()] once. `blocked` is null, or why it can no longer commit (it must abandon).
        void concluded(Throwable blocked);
    }

    private record Terminal(TerminalState state, Throwable failure) {}

    private static final Terminal ACTIVE = new Terminal(TerminalState.ACTIVE, null);
    private static final Terminal CANCELLED = new Terminal(TerminalState.CANCELLED, null);
    private static final Terminal COMPLETED = new Terminal(TerminalState.COMPLETED, null);

    private final long sequenceId;
    private final Order order = new Order();
    /// The last users of the sequence's carried state in its graphs; confined to the workspace's owner.
    private final SequenceOwner owner = new SequenceOwner();
    /// Written by admission only.
    private volatile long submitted;
    private volatile long admitted;
    /// Written by the conclusion of a piece only.
    private volatile long committed;
    private volatile long settled;
    private volatile boolean cancellationRequested;
    private final AtomicReference<Terminal> terminal = new AtomicReference<>(ACTIVE);
    private volatile Object kvCacheState;
    private volatile Object recurrentState;
    private final SequenceCleanup cleanup = new SequenceCleanup();

    public Sequence(long sequenceId) {
        this(sequenceId, 0L);
    }

    public Sequence(long sequenceId, long initialTokenPosition) {
        if (sequenceId < 0) {
            throw new IllegalArgumentException("sequenceId must be non-negative");
        }
        if (initialTokenPosition < 0) {
            throw new IllegalArgumentException("initialTokenPosition must be non-negative");
        }
        this.sequenceId = sequenceId;
        this.committed = initialTokenPosition;
        this.submitted = initialTokenPosition;
    }

    public long sequenceId() {
        return this.sequenceId;
    }

    /// The committed frontier: where the last concluded work left the sequence.
    public long currentTokenPosition() {
        return this.committed;
    }

    public long committedFrontier() {
        return this.committed;
    }

    /// Where the work in flight will leave the sequence, where the next work starts; the committed frontier when
    /// nothing is in flight.
    public long submittedFrontier() {
        return inFlight() ? this.submitted : this.committed;
    }

    /// Whether some admitted work has not settled yet.
    public boolean inFlight() {
        return this.admitted != this.settled;
    }

    public boolean cancellationRequested() {
        return this.cancellationRequested;
    }

    public TerminalState terminalState() {
        return this.terminal.get().state();
    }

    public Throwable terminalFailure() {
        return this.terminal.get().failure();
    }

    public Object kvCacheState() {
        return this.kvCacheState;
    }

    /// Sets the KV state; only work in flight does.
    public void setKvCacheState(Object kvCacheState) {
        requireInFlight();
        this.kvCacheState = kvCacheState;
    }

    public Object recurrentState() {
        return this.recurrentState;
    }

    /// Sets the recurrent state; only work in flight does.
    public void setRecurrentState(Object recurrentState) {
        requireInFlight();
        this.recurrentState = recurrentState;
    }

    /// Admits `work` covering positions `[start, end)`: it starts at the submitted frontier, on an active
    /// sequence whose cancellation was not requested, and moves the submitted frontier to `end`. Work that
    /// leaves the frontiers where they are (a draft, a capture) admits `[submitted, submitted)`. Admissions
    /// come from one chain of work.
    public void admit(Work work, long start, long end) {
        Objects.requireNonNull(work, "work");
        requireAdmissible();
        long frontier = submittedFrontier();
        if (start != frontier) {
            throw new IllegalArgumentException("Execution starts at " + start + " but the sequence is at " + frontier);
        }
        if (end < start) {
            throw new IllegalArgumentException("Execution ends at " + end + " before its start " + start);
        }
        enter(work, end);
    }

    /// Admits `work` that leaves the frontiers where they are (a draft) at the submitted frontier, and returns
    /// that position: the one it was admitted at.
    public long admitAtFrontier(Work work) {
        Objects.requireNonNull(work, "work");
        requireAdmissible();
        long frontier = submittedFrontier();
        enter(work, frontier);
        return frontier;
    }

    private void requireAdmissible() {
        TerminalState state = terminalState();
        if (state != TerminalState.ACTIVE) {
            throw new IllegalStateException("Sequence is terminal: " + state);
        }
        if (this.cancellationRequested) {
            throw new IllegalStateException("Sequence cancellation was requested");
        }
    }

    /// Counts `work` in flight, then reads the cancellation flag: [#cancel()] publishes the flag before it reads
    /// the count, so either the admission sees the cancellation and backs out, or the cancellation sees the work
    /// in flight and leaves its settlement to end the sequence. Work refused here, or by the queue, settles at
    /// once.
    private void enter(Work work, long end) {
        long previous = this.submitted;
        this.submitted = end;
        this.admitted = this.admitted + 1;
        VarHandle.fullFence();
        TerminalState state = terminalState();
        if (this.cancellationRequested || state != TerminalState.ACTIVE) {
            this.submitted = previous;
            settle();
            throw new IllegalStateException(
                    this.cancellationRequested
                            ? "Sequence cancellation was requested"
                            : "Sequence is terminal: " + state);
        }
        try {
            this.order.offer(work);
        } catch (RuntimeException | Error refused) {
            this.submitted = previous;
            settle();
            throw refused;
        }
    }

    /// Concludes the complete work at the head of the order, in order. Called by a piece of work after it became
    /// ready.
    /// The record of the last users of this sequence's carried state ([SequenceOwner]).
    public SequenceOwner owner() {
        return this.owner;
    }

    public void drain() {
        this.order.drain();
    }

    /// Settles the concluding work by committing it: the committed frontier moves to `next`. Called from
    /// [Work#concluded] only.
    public void commit(long next) {
        requireInFlight();
        if (next < this.committed) {
            throw new IllegalArgumentException("the committed frontier cannot move backwards");
        }
        this.committed = next;
        settle();
    }

    /// Settles the concluding work without its result: the committed frontier stays. The work failed (after
    /// [#fail(Throwable)]), was cancelled, or was blocked. Called from [Work#concluded] only.
    public void abandon() {
        requireInFlight();
        settle();
    }

    /// Ends the sequence FAILED, unless it already ended.
    public void fail(Throwable failure) {
        this.terminal.compareAndSet(ACTIVE, new Terminal(TerminalState.FAILED, Objects.requireNonNull(failure)));
    }

    /// Requests cancellation. With nothing in flight the sequence ends CANCELLED now; otherwise the settlement of the
    /// work in flight ends it. Closes nothing: [#complete()] does.
    public void cancel() {
        this.cancellationRequested = true;
        // The flag is published before the settled count is read; settlement publishes the count before it reads the
        // flag, so one of the two sees the other.
        VarHandle.fullFence();
        if (!inFlight()) this.terminal.compareAndSet(ACTIVE, CANCELLED);
    }

    /// Ends the sequence COMPLETED, unless it already ended, and closes its persistent state. Lifecycle: the
    /// session runs it after its generation ended. A failed close is reported and retried by the next call.
    public void complete() {
        if (inFlight()) {
            throw new IllegalStateException("Cannot complete a sequence during execution");
        }
        this.terminal.compareAndSet(ACTIVE, COMPLETED);
        this.cleanup.close(this.recurrentState, this.kvCacheState, terminalFailure());
    }

    private void settle() {
        this.settled = this.settled + 1;
        VarHandle.fullFence();
        if (this.cancellationRequested) this.terminal.compareAndSet(ACTIVE, CANCELLED);
    }

    /// Why work admitted at `start` can no longer commit, or null: a cancellation concludes it instead.
    private Throwable blocked(long start) {
        if (this.cancellationRequested) return null;
        Terminal ended = this.terminal.get();
        if (ended.state() == TerminalState.FAILED) {
            return new IllegalStateException("a preceding quantum failed the sequence", ended.failure());
        }
        if (start != this.committed) {
            return new IllegalStateException(
                    "admitted at " + start + " but the sequence committed only to " + this.committed);
        }
        return null;
    }

    private void requireInFlight() {
        if (!inFlight()) {
            throw new IllegalStateException("Sequence state changes only under its work in flight");
        }
    }

    /// The sequence's admission order.
    private final class Order extends Sequencer<Work> {
        @Override
        protected void drained(Work work) {
            long before = Sequence.this.settled;
            try {
                work.concluded(blocked(work.admittedAt()));
            } finally {
                // Work that concluded without settling would hold the sequence in flight for good.
                if (Sequence.this.settled == before) settle();
            }
        }
    }
}
