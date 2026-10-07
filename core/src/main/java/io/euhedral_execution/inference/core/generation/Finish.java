package io.euhedral_execution.inference.core.generation;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.runtime.graph.FrameSeeds;

/// Ends a generation: the session's bookkeeping, then the caller's result.
final class Finish extends AbstractFrame {

    Generation generation;

    Finish(long idHash, FrameManager<Generation, Finish> recycler, Generation generation) {
        super(idHash, recycler, null);
        // Unordered, so the factory draws a fresh seed on create and on every reuse.
        randomizeHash(FrameSeeds.forHostWork().next());
        this.generation = generation;
    }

    @Override
    public void execute() {
        this.generation.conclude();
    }

    @Override
    public void doFinally() {
        this.generation = null;
        recycle();
    }

    /// `execute` never throws, so this is a frame the lattice rejected without running it: it runs here.
    @Override
    public void doFinallyWithError(Throwable rejection) {
        execute();
        doFinally();
    }
}
