package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.core.frames.AbstractFrame;

/// Where frames that became ready are put for the lattice to pull: the pool of work that every
/// producer (a stage that made its successors ready, a driver callback, a request thread) throws
/// frames into, and that workers take from. Publishing never runs a frame and never waits.
///
/// The lake also counts what it was asked to carry: a quantum (or a host task) is admitted when it
/// begins and terminated when it ends, and a lake that is closing refuses new admissions and
/// completes once every admitted unit terminated.
public interface FrameLake {

    /// Makes `frame` available from a worker or an ordinary thread.
    void publish(AbstractFrame frame);

    /// Makes `frame` available from a CUDA driver callback thread, where nothing but enqueueing is
    /// allowed.
    void publishFromCallback(AbstractFrame frame);

    /// Accepts one unit of work. The lake cannot complete until it terminates.
    void admit();

    /// Accepts one more unit while the lake drains: allowed only while an accepted unit is still
    /// running, whose continuation this is.
    void admitDuringDrain();

    /// An admitted unit reached its terminal state.
    void terminated();

    /// Publishes `frame`; a lake that refuses it (closing, or a full partition) does not strand it: it runs here, on
    /// this thread, as a worker would run it (`execute`, then `doFinally`, or `doFinallyWithError` when `execute`
    /// threw).
    default void publishOrRun(AbstractFrame frame) {
        try {
            publish(frame);
        } catch (RuntimeException refused) {
            try {
                frame.execute();
            } catch (Throwable failed) {
                frame.doFinallyWithError(failed);
                return;
            }
            try {
                frame.doFinally();
            } catch (RuntimeException | Error ended) {
                // As a worker does: a frame's end that fails is logged, never rethrown into its producer.
                org.slf4j.LoggerFactory.getLogger(FrameLake.class)
                        .error("a frame run in place failed at its end", ended);
            }
        }
    }
}
