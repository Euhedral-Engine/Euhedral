package io.euhedral_execution.inference.core.generation;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.impl.FrameManager;

/// Ends a generation: the session's bookkeeping, then the caller's result.
final class Finish extends AbstractFrame {

    Generation generation;

    Finish(long idHash, FrameManager<Generation, Finish> recycler, Generation generation) {
        super(idHash, recycler, null);
        randomizeHash(0);
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

    /// The lattice rejected the frame without running it: the generation still ends, here.
    @Override
    public void doFinallyWithError(Throwable rejection) {
        execute();
        doFinally();
    }
}
