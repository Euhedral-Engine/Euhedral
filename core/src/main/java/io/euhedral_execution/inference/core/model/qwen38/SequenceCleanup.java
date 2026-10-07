package io.euhedral_execution.inference.core.model.qwen38;

import java.util.concurrent.locks.ReentrantLock;

/// Closes a sequence's persistent state (its recurrent and KV state) once. Lifecycle only: [Sequence#complete()]
/// runs it after the session's generation ended. A failed close is reported, and the next call retries only what
/// is still open.
final class SequenceCleanup {

    private final ReentrantLock lock = new ReentrantLock();
    private boolean recurrentReleased;
    private boolean kvReleased;

    /// Closes `recurrent` and `kv` (the same object counts once), attaching a failure to `terminalFailure` when
    /// there is one, and throws if anything could not be closed.
    void close(Object recurrent, Object kv, Throwable terminalFailure) {
        this.lock.lock();
        try {
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
            if (terminalFailure != null && terminalFailure != failure) terminalFailure.addSuppressed(failure);
            throw new IllegalStateException("Unable to release persistent Qwen sequence state", failure);
        } finally {
            this.lock.unlock();
        }
    }

    private static void closeResource(Object resource) throws Exception {
        if (resource instanceof AutoCloseable closeable) closeable.close();
    }
}
