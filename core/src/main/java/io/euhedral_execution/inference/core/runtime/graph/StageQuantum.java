package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.inference.core.gpu.GpuStream;

/// The quantum bound to a [StageGraph] while it runs.
///
/// It supplies stop and failure state and owns the quantum's terminal work. The graph never inspects
/// what its stages compute.
public interface StageQuantum {

    /// Failure or cancellation: no further stage may submit work or make a successor ready.
    boolean stopRequested();

    void fail(Throwable failure);

    /// Whether registered kernels of this quantum may overlap their predecessor's tail.
    default boolean overlapLaunches() {
        return false;
    }

    /// Identifies what this quantum submits apart from values it reads from device memory: two quanta with
    /// equal keys submit the same launches, copies and memsets, so one's captured graph can run the other.
    /// Null for a quantum that is never captured.
    default Object captureKey() {
        return null;
    }

    /// Every lane the quantum used has joined `home`, and no stage of it can submit again; null when the
    /// lanes were proven idle instead. Runs once, before the retirement boundary is armed.
    default void lanesJoined(GpuStream home) {}

    /// Runs on an ordinary worker after the quantum's device-completion boundary and every stage's
    /// retirement hook. It releases quantum-owned storage and publishes externally visible state.
    void retire(Throwable deviceFailure);

    /// Publishes the terminal outcome. The graph is already recycled when this runs.
    void publishOutcome();
}
