package io.euhedral_execution.inference.core.scheduling.graph;

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

    /// Runs on an ordinary worker after the quantum's device-completion boundary and every stage's
    /// retirement hook. It releases quantum-owned storage and publishes externally visible state.
    void retire(Throwable deviceFailure);

    /// Publishes the terminal outcome. The graph is already recycled when this runs.
    void publishOutcome();
}
