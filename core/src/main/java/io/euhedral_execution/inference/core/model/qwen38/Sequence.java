package io.euhedral_execution.inference.core.model.qwen38;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

/// Request-lifetime state shared by successive Qwen execution quanta.
///
/// The generation chain orders a sequence's work: it admits one quantum at a time and retires it before the next.
/// A sequence keeps two frontiers. The submitted frontier is where the quantum in flight will leave the sequence;
/// the committed frontier is where the last retired quantum left it. Admission and retirement are written only by
/// that chain, and read by any thread, so the fields are volatile and there is no execution lease.
///
/// Cancellation comes from any thread. Admission publishes the quantum in flight before it reads the cancellation
/// flag, and [#cancel()] publishes the flag before it reads the quantum in flight, so one of the two always sees
/// the other: either the admission backs out, or the quantum's retirement concludes the cancellation. The terminal
/// state is first-writer-wins. Persistent state (GDN buffers, KV pages) closes only in [#complete()], which the
/// session's lifecycle runs after its generation ended.
public final class Sequence {

    public enum TerminalState {
        ACTIVE,
        CANCELLED,
        FAILED,
        COMPLETED
    }

    private record Terminal(TerminalState state, Throwable failure) {}

    private static final Terminal ACTIVE = new Terminal(TerminalState.ACTIVE, null);
    private static final Terminal CANCELLED = new Terminal(TerminalState.CANCELLED, null);
    private static final Terminal COMPLETED = new Terminal(TerminalState.COMPLETED, null);

    private final long sequenceId;
    private volatile long committed;
    private volatile long submitted;
    private volatile boolean inFlight;
    private volatile boolean cancellationRequested;
    private final AtomicReference<Terminal> terminal = new AtomicReference<>(ACTIVE);
    private volatile Object kvCacheState;
    private volatile Object recurrentState;
    // Only terminal resource cleanup, which is lifecycle, is serialized.
    private final ReentrantLock cleanupLock = new ReentrantLock();
    private boolean recurrentReleased;
    private boolean kvReleased;

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

    /// The committed frontier: where the last retired quantum left the sequence.
    public long currentTokenPosition() {
        return this.committed;
    }

    public long committedFrontier() {
        return this.committed;
    }

    /// Where the quantum in flight will leave the sequence; the committed frontier when none is.
    public long submittedFrontier() {
        return this.submitted;
    }

    /// Whether a quantum has been admitted and not yet retired.
    public boolean inFlight() {
        return this.inFlight;
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

    /// Sets the KV state; only the quantum in flight does.
    public void setKvCacheState(Object kvCacheState) {
        requireInFlight();
        this.kvCacheState = kvCacheState;
    }

    public Object recurrentState() {
        return this.recurrentState;
    }

    /// Sets the recurrent state; only the quantum in flight does.
    public void setRecurrentState(Object recurrentState) {
        requireInFlight();
        this.recurrentState = recurrentState;
    }

    /// Admits the quantum covering positions `[start, end)`. It must start at the committed frontier, with none in
    /// flight, on an active sequence whose cancellation was not requested. A draft quantum admits `[committed,
    /// committed)`: it is still the one quantum in flight, and leaves the frontiers where they are.
    public void admit(long start, long end) {
        if (this.inFlight) {
            throw new IllegalStateException("Sequence already has a quantum in flight");
        }
        requireActive();
        if (start != this.committed) {
            throw new IllegalArgumentException(
                    "Execution starts at " + start + " but sequence is at " + this.committed);
        }
        if (end < start) {
            throw new IllegalArgumentException("Execution ends at " + end + " before its start " + start);
        }
        this.inFlight = true;
        // Published before the flag is read; cancel() writes the flag before it reads this.
        if (this.cancellationRequested) {
            this.inFlight = false;
            this.terminal.compareAndSet(ACTIVE, CANCELLED);
            throw new IllegalStateException("Sequence cancellation was requested");
        }
        this.submitted = end;
    }

    /// Retires the quantum in flight successfully: both frontiers move to `next`, unless a cancellation was
    /// requested first. Returns true when the sequence ends cancelled.
    public boolean commit(long next) {
        requireInFlight();
        if (next < this.committed) {
            throw new IllegalArgumentException("the committed frontier cannot move backwards");
        }
        if (!this.cancellationRequested) {
            this.committed = next;
            this.submitted = next;
        } else {
            this.submitted = this.committed;
        }
        return retireInFlight();
    }

    /// Retires the quantum in flight without its work: the submitted frontier returns to the committed one. The
    /// quantum failed (after [#fail(Throwable)]) or was cancelled.
    public void abandon() {
        requireInFlight();
        this.submitted = this.committed;
        retireInFlight();
    }

    /// Ends the sequence FAILED, unless it already ended.
    public void fail(Throwable failure) {
        this.terminal.compareAndSet(ACTIVE, new Terminal(TerminalState.FAILED, Objects.requireNonNull(failure)));
    }

    /// Requests cancellation. With no quantum in flight the sequence ends CANCELLED now; otherwise that quantum's
    /// retirement ends it. Closes nothing: [#complete()] does.
    public void cancel() {
        this.cancellationRequested = true;
        // Published before the quantum in flight is read; admission and retirement write that before the flag.
        if (!this.inFlight) this.terminal.compareAndSet(ACTIVE, CANCELLED);
    }

    /// Ends the sequence COMPLETED, unless it already ended, and closes its persistent state. Lifecycle: the
    /// session runs it after its generation ended. A failed close is reported and retried by the next call.
    public void complete() {
        if (this.inFlight) {
            throw new IllegalStateException("Cannot complete a sequence during execution");
        }
        this.terminal.compareAndSet(ACTIVE, COMPLETED);
        closePersistentState();
    }

    private boolean retireInFlight() {
        this.inFlight = false;
        // Published before the flag is read; cancel() writes the flag before it reads this.
        if (!this.cancellationRequested) return false;
        this.terminal.compareAndSet(ACTIVE, CANCELLED);
        return this.terminal.get().state() == TerminalState.CANCELLED;
    }

    private void requireActive() {
        TerminalState state = terminalState();
        if (state != TerminalState.ACTIVE) {
            throw new IllegalStateException("Sequence is terminal: " + state);
        }
        if (this.cancellationRequested) {
            throw new IllegalStateException("Sequence cancellation was requested");
        }
    }

    private void requireInFlight() {
        if (!this.inFlight) {
            throw new IllegalStateException("Sequence state changes only under its quantum in flight");
        }
    }

    private void closePersistentState() {
        this.cleanupLock.lock();
        try {
            Object recurrent = this.recurrentState;
            Object kv = this.kvCacheState;
            Throwable failure = null;
            if (!this.recurrentReleased) {
                try {
                    closeResource(recurrent);
                    this.recurrentReleased = true;
                } catch (Throwable cleanup) {
                    failure = cleanup;
                }
            }
            if (kv == recurrent) {
                this.kvReleased = this.recurrentReleased;
            } else if (!this.kvReleased) {
                try {
                    closeResource(kv);
                    this.kvReleased = true;
                } catch (Throwable cleanup) {
                    if (failure == null) failure = cleanup;
                    else if (failure != cleanup) failure.addSuppressed(cleanup);
                }
            }
            if (failure == null) return;
            Throwable terminalFailure = terminalFailure();
            if (terminalFailure != null && terminalFailure != failure) terminalFailure.addSuppressed(failure);
            throw new IllegalStateException("Unable to release persistent Qwen sequence state", failure);
        } finally {
            this.cleanupLock.unlock();
        }
    }

    private static void closeResource(Object resource) throws Exception {
        if (resource instanceof AutoCloseable closeable) closeable.close();
    }
}
