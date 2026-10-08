package io.euhedral_execution.inference.core.generation;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.runtime.graph.WorkspaceOwner;

/// Starts a generation's next step: checks out the step's [Select] and hands it to the port, whose quantum throws it
/// when it retires. A port that could not hand its step over fails the generation, which then finishes.
///
/// Runs on the workspace's owner: its `idHash` is [WorkspaceOwner#HASH] and it stays ordered (its routing hash is
/// its `idHash`, which recycling keeps), so a port admits its quantum where admissions bind, one at a time.
final class Admit extends AbstractFrame {

    private final GenerationFrames frames;
    Generation generation;
    StepPort port;
    private Select refused;

    Admit(FrameManager<Generation, Admit> recycler, GenerationFrames frames, Generation generation) {
        super(WorkspaceOwner.HASH, recycler, null);
        this.frames = frames;
        this.generation = generation;
    }

    @Override
    public void execute() {
        Select select = this.frames.select(this.generation, this.port);
        try {
            this.port.admit(select);
        } catch (Throwable failure) {
            this.generation.fail(failure);
            this.refused = select;
        }
    }

    @Override
    public void doFinally() {
        Generation finished = this.generation;
        Select refused = this.refused;
        this.generation = null;
        this.port = null;
        this.refused = null;
        if (refused != null) {
            refused.release();
            refused.recycle();
        }
        recycle();
        if (refused != null) this.frames.publish(this.frames.finish(finished));
    }

    /// `execute` never throws, so this is a frame the lattice rejected without running it: it runs here.
    @Override
    public void doFinallyWithError(Throwable rejection) {
        execute();
        doFinally();
    }
}
