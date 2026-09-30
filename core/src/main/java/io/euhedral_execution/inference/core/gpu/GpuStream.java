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

    /// Blocks until all submitted work has finished. Only recovery and teardown use it.
    void synchronize();

    /// After a failed submission or boundary registration, proves that this stream stopped reading
    /// quantum storage. When that cannot be proven, all device ownership is retained permanently.
    void recover(Throwable failure);

    @Override
    void close();
}
