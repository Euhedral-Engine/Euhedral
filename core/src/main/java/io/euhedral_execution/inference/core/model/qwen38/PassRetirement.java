package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import io.euhedral_execution.inference.core.runtime.graph.FrameLake;
import io.euhedral_execution.inference.core.runtime.graph.FrameSeeds;
import java.util.List;

/// The continuation of a pass's quantum: reports the retirement to the owner, then throws each member's Select,
/// told which quantum concluded. The report is published first, so the owner applies it before any member's next
/// offer (the owner's sink keeps publication order). Every Select is thrown even when the report or an earlier
/// Select fails.
final class PassRetirement extends AbstractFrame implements AbstractQuantum.Continuation {

    private final Execution execution;
    private final FrameLake lake;
    private final List<Sequence> members;
    private final List<AbstractFrame> selects;
    private final boolean counted;
    private AbstractQuantum quantum;
    private boolean ran;

    /// @param members the sequences whose steps the quantum ran
    /// @param selects the continuations of those steps
    /// @param counted whether the quantum is one the pass assembler owes a retirement for
    PassRetirement(
            Execution execution, FrameLake lake, List<Sequence> members, List<AbstractFrame> selects, boolean counted) {
        super(FrameSeeds.ID_HASH);
        randomizeHash(FrameSeeds.forHostWork().next());
        this.execution = execution;
        this.lake = lake;
        this.members = List.copyOf(members);
        this.selects = List.copyOf(selects);
        this.counted = counted;
    }

    @Override
    public void concluded(AbstractQuantum quantum) {
        this.quantum = quantum;
    }

    @Override
    public void execute() {
        if (this.ran) return;
        this.ran = true;
        RuntimeException failure = null;
        try {
            if (this.counted) this.execution.passRetired(this.members);
        } catch (RuntimeException reportFailure) {
            failure = reportFailure;
        }
        for (AbstractFrame select : this.selects) {
            try {
                if (select instanceof AbstractQuantum.Continuation told) told.concluded(this.quantum);
            } catch (RuntimeException toldFailure) {
                failure = combine(failure, toldFailure);
            } finally {
                try {
                    this.lake.publishOrRun(select);
                } catch (RuntimeException publishFailure) {
                    failure = combine(failure, publishFailure);
                }
            }
        }
        if (failure != null) throw failure;
    }

    private static RuntimeException combine(RuntimeException first, RuntimeException next) {
        if (first == null) return next;
        first.addSuppressed(next);
        return first;
    }

    /// A frame the lattice rejected without running it: it runs here, once.
    @Override
    public void doFinallyWithError(Throwable rejection) {
        execute();
    }
}
