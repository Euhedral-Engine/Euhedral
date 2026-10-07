package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// What every model's quantum shares: the first failure (later ones suppressed into it), a cancel flag, the terminal
/// template, and the continuation. When the graph publishes the outcome, the quantum throws its continuation frame
/// into the lake: whatever reads the outcome runs there, on whichever worker takes it, never inside the graph's
/// retirement.
public abstract class AbstractQuantum implements StageQuantum {

    private static final Logger LOG = LoggerFactory.getLogger(AbstractQuantum.class);

    /// Marks a quantum that concluded without failure: later failures are ignored, not suppressed into it.
    private static final Throwable SEALED = new Throwable("the quantum concluded", null, false, false) {};

    /// A continuation that is told which quantum concluded before it is thrown.
    public interface Continuation {
        void concluded(AbstractQuantum quantum);
    }

    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private volatile boolean cancelled;
    private FrameLake lake;
    private AbstractFrame continuation;
    private Throwable outcome;

    /// Binds the frame thrown into `lake` once the outcome is published. Called once, before admission.
    public final void continueWith(FrameLake lake, AbstractFrame continuation) {
        Objects.requireNonNull(lake, "lake");
        Objects.requireNonNull(continuation, "continuation");
        if (this.continuation != null) throw new IllegalStateException("the quantum already has a continuation");
        this.lake = lake;
        this.continuation = continuation;
    }

    /// Asks the quantum to stop at its next stage.
    public void cancel() {
        this.cancelled = true;
    }

    protected final boolean cancelRequested() {
        return this.cancelled;
    }

    @Override
    public boolean stopRequested() {
        return this.cancelled || failure() != null;
    }

    @Override
    public final void fail(Throwable cause) {
        Objects.requireNonNull(cause, "cause");
        if (this.failure.compareAndSet(null, cause)) return;
        Throwable first = this.failure.get();
        if (first != SEALED && first != cause) first.addSuppressed(cause);
    }

    /// The first failure so far, or null.
    public final Throwable failure() {
        Throwable first = this.failure.get();
        return first == SEALED ? null : first;
    }

    /// Seals a quantum that concluded without failure; false when a failure came first.
    protected final boolean seal() {
        return this.failure.compareAndSet(null, SEALED);
    }

    /// Runs on a worker after the device work retired: a device failure or a cancellation becomes the quantum's
    /// failure, [#release] always runs, and [#commit] runs only when nothing failed. [#terminalFailure] then
    /// holds the failure that ended the quantum, or null when it committed.
    @Override
    public void retire(Throwable deviceFailure) {
        if (deviceFailure != null) fail(deviceFailure);
        if (this.cancelled && failure() == null) fail(new CancellationException("the step was cancelled"));
        try {
            release();
        } catch (RuntimeException | Error cleanup) {
            fail(cleanup);
        }
        if (failure() == null) {
            try {
                commit();
            } catch (RuntimeException | Error commitFailure) {
                fail(commitFailure);
            }
        }
        concluded(failure());
    }

    /// Publishes externally visible state of a quantum that did not fail.
    protected void commit() {}

    /// Releases what the quantum holds, whatever its outcome.
    protected void release() {}

    /// Records the outcome of a quantum that retires by its own rules.
    protected final void concluded(Throwable outcome) {
        this.outcome = outcome;
    }

    /// The failure that ended the quantum, or null when it committed; read by the continuation.
    public final Throwable terminalFailure() {
        return this.outcome;
    }

    /// Runs [#published], then throws the continuation into the lake. A lake that no longer takes frames (it is
    /// closing) does not strand the chain: the continuation runs here, on this worker.
    @Override
    public final void publishOutcome() {
        published();
        AbstractFrame next = this.continuation;
        if (next == null) return;
        this.continuation = null;
        if (next instanceof Continuation told) told.concluded(this);
        try {
            this.lake.publish(next);
        } catch (RuntimeException refused) {
            LOG.debug("the lake refused a continuation; running it here", refused);
            try {
                next.execute();
                next.doFinally();
            } catch (Throwable failed) {
                next.doFinallyWithError(failed);
            }
        }
    }

    /// The model's own publication, before the continuation is thrown.
    protected void published() {}
}
