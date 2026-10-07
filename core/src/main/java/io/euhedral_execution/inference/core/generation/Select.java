package io.euhedral_execution.inference.core.generation;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;

/// Reads a concluded step through its port (sampling, acceptance, text, timing, the stop and budget checks) and
/// throws the next [Admit], or [Finish] once the generation is done or failed.
final class Select extends AbstractFrame implements AbstractQuantum.Continuation {

    private final GenerationFrames frames;
    Generation generation;
    StepPort port;
    private AbstractQuantum step;
    private StepPort next;

    Select(long idHash, FrameManager<Generation, Select> recycler, GenerationFrames frames, Generation generation) {
        super(idHash, recycler, null);
        randomizeHash(0);
        this.frames = frames;
        this.generation = generation;
    }

    @Override
    public void concluded(AbstractQuantum quantum) {
        this.step = quantum;
    }

    @Override
    public void execute() {
        try {
            this.next = this.port.retired(this.step);
        } catch (Throwable failure) {
            this.generation.fail(failure);
            this.next = null;
        }
        if (this.generation.failure() != null) this.next = null;
    }

    @Override
    public void doFinally() {
        Generation generation = this.generation;
        StepPort next = this.next;
        release();
        recycle();
        this.frames.publish(next == null ? this.frames.finish(generation) : this.frames.admit(generation, next));
    }

    /// The lattice rejected the frame without running it: the step's result is still read, here.
    @Override
    public void doFinallyWithError(Throwable rejection) {
        execute();
        doFinally();
    }

    /// Drops what the frame held, before it is recycled.
    void release() {
        this.generation = null;
        this.port = null;
        this.step = null;
        this.next = null;
    }
}
