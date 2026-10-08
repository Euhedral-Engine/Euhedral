package io.euhedral_execution.inference.core.gpu;

/// Quantum-owned device ordering.
///
/// Every GPU stage of a quantum submits to its quantum's stream, whichever CPU worker runs the stage,
/// so stream order carries ordinary device dependencies. A device-completion boundary is requested
/// explicitly, announced on a driver callback thread, and then confirmed on an ordinary thread.
public interface GpuStream extends AutoCloseable {

    /// Receives a notified device-completion boundary. On a CUDA driver callback thread
    /// (`driverThread`) it may only enqueue work: it must not call CUDA, block, run frames, or release
    /// memory. Otherwise it runs on an ordinary thread, such as the registering one.
    @FunctionalInterface
    interface RetirementListener {
        void retired(long ticket, boolean driverThread);
    }

    /// Runs `launches` on the calling thread with this stream selected for every native launch. With
    /// `overlapPredecessor`, registered kernels use programmatic dependent launch.
    void submit(Runnable launches, boolean overlapPredecessor);

    /// Registers a device-completion boundary after all work submitted so far. The listener runs
    /// exactly once, possibly before this call returns. A thrown exception means nothing was armed and
    /// the listener never runs.
    long notifyRetired(RetirementListener listener);

    /// Confirms a notified boundary from an ordinary thread after its listener ran. Returns the device
    /// failure, or null when every operation submitted before the boundary completed; never throws.
    Throwable confirmRetired(long ticket);

    /// Opens a reusable cross-stream marker. Synchronous streams need none and return 0.
    default long openMarker() {
        return 0L;
    }

    /// Records `marker` after all work submitted to this stream so far.
    default void mark(long marker) {}

    /// Orders work submitted to this stream from now on behind the last recording of `marker`, on the
    /// device; the host does not wait.
    default void await(long marker) {}

    /// Releases a marker from [#openMarker()] once no recorded or awaited use remains outstanding.
    default void closeMarker(long marker) {}

    /// Whether this stream records and replays captured quanta (CUDA graphs): the methods below.
    default boolean capturesGraphs() {
        return false;
    }

    /// As [#submit], also submitting every launch, copy and memset to `shadow`, a stream under capture
    /// (with programmatic dependent launch where `shadowOverlap` allows it). Returns the hash of the
    /// submissions, never 0, or 0 when the shadow submission failed; the stream's own work is unaffected.
    default long submitRecording(
            Runnable launches, boolean overlapPredecessor, GpuStream shadow, boolean shadowOverlap) {
        throw new UnsupportedOperationException("graph capture is not supported");
    }

    /// Starts checking on the calling thread and returns its sink: a stream under capture that receives
    /// whatever bypasses the submission hooks, so that it never runs.
    default long beginChecking() {
        throw new UnsupportedOperationException("graph capture is not supported");
    }

    /// Runs `launches` on the calling thread with `sink` selected, submitting nothing: returns the hash
    /// that [#submitRecording] would have returned for the same submissions.
    default long submitChecking(Runnable launches, long sink) {
        throw new UnsupportedOperationException("graph capture is not supported");
    }

    /// Ends checking on the calling thread. Returns whether the sink captured nothing.
    default boolean endChecking() {
        throw new UnsupportedOperationException("graph capture is not supported");
    }

    /// Starts capturing this stream's submissions into a graph instead of running them.
    default void beginCapture() {
        throw new UnsupportedOperationException("graph capture is not supported");
    }

    /// Ends this stream's capture and instantiates the graph. Returns its handle, or 0 when the capture
    /// failed; the capture has ended either way.
    default long endCapture() {
        throw new UnsupportedOperationException("graph capture is not supported");
    }

    /// Submits one run of an instantiated graph. Returns false when the graph cannot run (the quantum then runs
    /// stage by stage).
    default boolean launchGraph(long graph) {
        throw new UnsupportedOperationException("graph capture is not supported");
    }

    /// Releases an instantiated graph none of whose runs is outstanding.
    default void destroyGraph(long graph) {
        throw new UnsupportedOperationException("graph capture is not supported");
    }

    /// Blocks until all submitted work has finished. Only recovery and teardown use it.
    void synchronize();

    /// After a failed submission or boundary registration, proves that this stream stopped reading
    /// quantum storage. When that cannot be proven, all device ownership is retained permanently.
    void recover(Throwable failure);

    @Override
    void close();
}
