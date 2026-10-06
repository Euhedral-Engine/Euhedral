package io.euhedral_execution.inference.core.scheduling.graph;

/// Device storage that one reusable [StageGraph] owns for the quanta that run on it, whatever the
/// model: the graph runs one quantum at a time and its pool recycles it only after that quantum's
/// device work retired, so the graph is the natural owner of storage whose shape repeats from
/// quantum to quantum.
public interface GraphStorage extends AutoCloseable {

    /// Device bytes this storage currently holds.
    long retainedBytes();

    boolean isClosed();

    /// Frees what the storage holds. Only an owner whose device work provably stopped may call it.
    @Override
    void close();
}
