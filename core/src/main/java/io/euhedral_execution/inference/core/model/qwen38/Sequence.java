package io.euhedral_execution.inference.core.model.qwen38;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

/// Request-lifetime state shared by successive Qwen execution quanta.
///
/// The execution lease is intentionally exclusive: one submitted chain may mutate this state at a
/// time. Sequence-owned recurrent resources are closed when the sequence reaches a terminal state.
public final class Sequence {

    public enum TerminalState {
        ACTIVE,
        CANCELLED,
        FAILED,
        COMPLETED
    }

    /// Capability held by the one chain currently allowed to mutate sequence state.
    public static final class ExecutionLease {

        private final Sequence owner;

        private ExecutionLease(Sequence owner) {
            this.owner = owner;
        }
    }

    private final long sequenceId;
    private final AtomicReference<State> state;
    // Only terminal resource cleanup is serialized; lease and position transitions remain CAS-based.
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
        this.state = new AtomicReference<>(
                new State(initialTokenPosition, null, false, TerminalState.ACTIVE, null, null, null));
    }

    public long sequenceId() {
        return this.sequenceId;
    }

    public long currentTokenPosition() {
        return this.state.get().currentTokenPosition();
    }

    public boolean isExecutionClaimed() {
        return this.state.get().activeLease() != null;
    }

    public boolean cancellationRequested() {
        return this.state.get().cancellationRequested();
    }

    public TerminalState terminalState() {
        return this.state.get().terminalState();
    }

    public Throwable terminalFailure() {
        return this.state.get().terminalFailure();
    }

    public Object kvCacheState() {
        return this.state.get().kvCacheState();
    }

    public void setKvCacheState(ExecutionLease lease, Object kvCacheState) {
        while (true) {
            State current = this.state.get();
            requireLease(current, lease);
            State updated = new State(
                    current.currentTokenPosition(),
                    current.activeLease(),
                    current.cancellationRequested(),
                    current.terminalState(),
                    current.terminalFailure(),
                    kvCacheState,
                    current.recurrentState());
            if (this.state.compareAndSet(current, updated)) {
                return;
            }
        }
    }

    public Object recurrentState() {
        return this.state.get().recurrentState();
    }

    public void setRecurrentState(ExecutionLease lease, Object recurrentState) {
        while (true) {
            State current = this.state.get();
            requireLease(current, lease);
            State updated = new State(
                    current.currentTokenPosition(),
                    current.activeLease(),
                    current.cancellationRequested(),
                    current.terminalState(),
                    current.terminalFailure(),
                    current.kvCacheState(),
                    recurrentState);
            if (this.state.compareAndSet(current, updated)) {
                return;
            }
        }
    }

    /// Requests cancellation of the active or next execution quantum.
    public void cancel() {
        while (true) {
            State current = this.state.get();
            if (current.terminalState() != TerminalState.ACTIVE) {
                return;
            }
            TerminalState terminal = current.activeLease() == null ? TerminalState.CANCELLED : TerminalState.ACTIVE;
            State updated = new State(
                    current.currentTokenPosition(),
                    current.activeLease(),
                    true,
                    terminal,
                    current.terminalFailure(),
                    current.kvCacheState(),
                    current.recurrentState());
            if (this.state.compareAndSet(current, updated)) {
                if (terminal == TerminalState.CANCELLED) {
                    closePersistentRecurrentState(updated);
                }
                return;
            }
        }
    }

    /// Marks the sequence terminal after its owner has stopped admitting work.
    public void complete() {
        while (true) {
            State current = this.state.get();
            if (current.activeLease() != null) {
                throw new IllegalStateException("Cannot complete a sequence during execution");
            }
            if (current.terminalState() != TerminalState.ACTIVE) {
                closePersistentState(current, true);
                return;
            }
            State updated = current.withTerminal(TerminalState.COMPLETED, current.terminalFailure());
            if (this.state.compareAndSet(current, updated)) {
                closePersistentState(updated, true);
                return;
            }
        }
    }

    /// Claims exclusive mutation for one execution quantum.
    public ExecutionLease claimExecution(long expectedStartPosition) {
        while (true) {
            State current = this.state.get();
            if (current.activeLease() != null) {
                throw new IllegalStateException("Sequence already has an executing Qwen chain");
            }
            if (current.terminalState() != TerminalState.ACTIVE) {
                throw new IllegalStateException("Sequence is terminal: " + current.terminalState());
            }
            if (current.cancellationRequested()) {
                throw new IllegalStateException("Sequence cancellation was requested");
            }
            if (expectedStartPosition != current.currentTokenPosition()) {
                throw new IllegalArgumentException("Execution starts at " + expectedStartPosition
                        + " but sequence is at " + current.currentTokenPosition());
            }
            ExecutionLease lease = new ExecutionLease(this);
            State updated = new State(
                    current.currentTokenPosition(),
                    lease,
                    false,
                    TerminalState.ACTIVE,
                    null,
                    current.kvCacheState(),
                    current.recurrentState());
            if (this.state.compareAndSet(current, updated)) {
                return lease;
            }
        }
    }

    /// Releases the lease, atomically resolving cancellation against successful completion.
    public void releaseExecution(ExecutionLease lease, long nextTokenPosition) {
        releaseExecutionAndCheckCancellation(lease, nextTokenPosition);
    }

    /// Releases the lease and reports whether cancellation won before token-position publication.
    ///
    /// Returns true when a cancellation request won before release.
    public boolean releaseExecutionAndCheckCancellation(ExecutionLease lease, long nextTokenPosition) {
        while (true) {
            State current = this.state.get();
            requireLease(current, lease);
            if (nextTokenPosition < current.currentTokenPosition()) {
                throw new IllegalArgumentException("nextTokenPosition cannot move backwards");
            }
            boolean cancelled = current.cancellationRequested();
            State updated = new State(
                    cancelled ? current.currentTokenPosition() : nextTokenPosition,
                    null,
                    current.cancellationRequested(),
                    cancelled ? TerminalState.CANCELLED : current.terminalState(),
                    current.terminalFailure(),
                    current.kvCacheState(),
                    current.recurrentState());
            if (this.state.compareAndSet(current, updated)) {
                if (cancelled) {
                    closePersistentRecurrentState(updated);
                }
                return cancelled;
            }
        }
    }

    void markCancelled(ExecutionLease lease) {
        while (true) {
            State current = this.state.get();
            requireLease(current, lease);
            State updated = new State(
                    current.currentTokenPosition(),
                    null,
                    true,
                    TerminalState.CANCELLED,
                    current.terminalFailure(),
                    current.kvCacheState(),
                    current.recurrentState());
            if (this.state.compareAndSet(current, updated)) {
                closePersistentRecurrentState(updated);
                return;
            }
        }
    }

    void markCancelledBeforeClaim() {
        while (true) {
            State current = this.state.get();
            if (current.terminalState() != TerminalState.ACTIVE) {
                return;
            }
            if (current.activeLease() != null) {
                throw new IllegalStateException("Sequence execution is already claimed");
            }
            State updated = new State(
                    current.currentTokenPosition(),
                    null,
                    true,
                    TerminalState.CANCELLED,
                    current.terminalFailure(),
                    current.kvCacheState(),
                    current.recurrentState());
            if (this.state.compareAndSet(current, updated)) {
                closePersistentRecurrentState(updated);
                return;
            }
        }
    }

    public void markFailed(ExecutionLease lease, Throwable failure) {
        Objects.requireNonNull(failure, "failure");
        while (true) {
            State current = this.state.get();
            requireLease(current, lease);
            State updated = new State(
                    current.currentTokenPosition(),
                    null,
                    current.cancellationRequested(),
                    TerminalState.FAILED,
                    failure,
                    current.kvCacheState(),
                    current.recurrentState());
            if (this.state.compareAndSet(current, updated)) {
                closePersistentRecurrentState(updated);
                return;
            }
        }
    }

    void markFailedBeforeClaim(Throwable failure) {
        Objects.requireNonNull(failure, "failure");
        while (true) {
            State current = this.state.get();
            if (current.activeLease() != null) {
                throw new IllegalStateException("Sequence execution is already claimed");
            }
            boolean wasActive = current.terminalState() == TerminalState.ACTIVE;
            State updated = new State(
                    current.currentTokenPosition(),
                    null,
                    current.cancellationRequested(),
                    TerminalState.FAILED,
                    failure,
                    current.kvCacheState(),
                    current.recurrentState());
            if (this.state.compareAndSet(current, updated)) {
                if (wasActive) {
                    closePersistentRecurrentState(updated);
                }
                return;
            }
        }
    }

    void markFailedAfterRelease(Throwable failure) {
        Objects.requireNonNull(failure, "failure");
        while (true) {
            State current = this.state.get();
            if (current.activeLease() != null) {
                throw new IllegalStateException("Sequence execution is still claimed");
            }
            boolean wasActive = current.terminalState() == TerminalState.ACTIVE;
            State updated = current.withTerminal(TerminalState.FAILED, failure);
            if (this.state.compareAndSet(current, updated)) {
                if (wasActive) {
                    closePersistentRecurrentState(updated);
                }
                return;
            }
        }
    }

    private void closePersistentRecurrentState(State terminalState) {
        closePersistentState(terminalState, false);
    }

    private void closePersistentState(State terminalState, boolean reportCleanupFailure) {
        this.cleanupLock.lock();
        try {
            Throwable failure = null;
            if (!this.recurrentReleased) {
                try {
                    closeResource(terminalState.recurrentState());
                    this.recurrentReleased = true;
                } catch (Throwable cleanup) {
                    failure = cleanup;
                }
            }
            if (terminalState.kvCacheState() == terminalState.recurrentState()) {
                this.kvReleased = this.recurrentReleased;
            } else if (!this.kvReleased) {
                try {
                    closeResource(terminalState.kvCacheState());
                    this.kvReleased = true;
                } catch (Throwable cleanup) {
                    if (failure == null) failure = cleanup;
                    else if (failure != cleanup) failure.addSuppressed(cleanup);
                }
            }
            if (failure == null) return;
            Throwable terminalFailure = terminalState.terminalFailure();
            if (terminalFailure != null && terminalFailure != failure) terminalFailure.addSuppressed(failure);
            if (terminalFailure == null || reportCleanupFailure)
                throw new IllegalStateException("Unable to release persistent Qwen sequence state", failure);
        } finally {
            this.cleanupLock.unlock();
        }
    }

    private static void closeResource(Object resource) throws Exception {
        if (resource instanceof AutoCloseable closeable) closeable.close();
    }

    private void requireLease(State current, ExecutionLease lease) {
        if (lease == null || lease.owner != this || current.activeLease() != lease) {
            throw new IllegalStateException("Sequence mutation requires its active execution lease");
        }
    }

    private record State(
            long currentTokenPosition,
            ExecutionLease activeLease,
            boolean cancellationRequested,
            TerminalState terminalState,
            Throwable terminalFailure,
            Object kvCacheState,
            Object recurrentState) {

        private State withTerminal(TerminalState terminal, Throwable failure) {
            return new State(
                    this.currentTokenPosition,
                    this.activeLease,
                    this.cancellationRequested,
                    terminal,
                    failure,
                    this.kvCacheState,
                    this.recurrentState);
        }
    }
}
