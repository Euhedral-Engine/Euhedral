package io.euhedral_execution.inference.core.model.qwen38;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.ToIntFunction;

/// Forms passes from the ready steps of concurrent generations. Confined to the workspace's owner: every event is
/// applied by an owner frame, and the record never waits. Rule: no pass forms while one is in flight, nor while a
/// member of the last retired pass has not reported (offered its next step or left); a pass then takes the pending
/// offers in arrival order, up to `maxRows` rows and `maxParts` offers. `Passes#run` returns how many retirements
/// the pass will report (one per quantum it admitted).
///
/// @param <S> the sequence a step belongs to
/// @param <O> a ready step offered for a pass
public final class PassAssembler<S, O> {
    /// Runs one formed pass.
    @FunctionalInterface
    public interface Passes<O> {
        /// Admits the pass and returns how many retirements it will report.
        int run(List<O> pass);
    }

    private final Function<O, S> sequenceOf;
    private final ToIntFunction<O> rowsOf;
    private final int maxRows;
    private final int maxParts;
    private final Passes<O> passes;
    private final ArrayDeque<O> pending = new ArrayDeque<>();
    private final Set<S> awaiting = new HashSet<>();
    private int inFlight;
    private boolean running;
    private boolean again;

    public PassAssembler(
            Function<O, S> sequenceOf, ToIntFunction<O> rowsOf, int maxRows, int maxParts, Passes<O> passes) {
        if (maxRows < 1 || maxParts < 1) {
            throw new IllegalArgumentException("a pass needs at least one row and one part");
        }
        this.sequenceOf = sequenceOf;
        this.rowsOf = rowsOf;
        this.maxRows = maxRows;
        this.maxParts = maxParts;
        this.passes = passes;
    }

    /// Queues a ready step, which also counts as its sequence's report.
    public void offer(O offer) {
        int rows = rowsOf.applyAsInt(offer);
        if (rows < 1 || rows > maxRows) {
            throw new IllegalArgumentException("a step of " + rows + " rows does not fit a pass of " + maxRows);
        }
        awaiting.remove(sequenceOf.apply(offer));
        pending.add(offer);
        tryRun();
    }

    /// Releases the wait for a sequence that will not offer again.
    public void leave(S sequence) {
        if (awaiting.remove(sequence)) {
            tryRun();
        }
    }

    /// Retires the pass in flight; its members must report before the next pass forms.
    public void retired(Collection<S> members) {
        if (inFlight == 0) {
            throw new IllegalStateException("no pass is in flight");
        }
        inFlight--;
        awaiting.addAll(members);
        tryRun();
    }

    /// Removes the pending offers matching `selected`, in arrival order.
    public List<O> withdraw(Predicate<O> selected) {
        List<O> out = new ArrayList<>();
        for (var it = pending.iterator(); it.hasNext(); ) {
            O o = it.next();
            if (selected.test(o)) {
                it.remove();
                out.add(o);
            }
        }
        return out;
    }

    public int pendingCount() {
        return pending.size();
    }

    public int awaitingCount() {
        return awaiting.size();
    }

    public int passesInFlight() {
        return inFlight;
    }

    private void tryRun() {
        if (running) {
            again = true;
            return;
        }
        running = true;
        try {
            do {
                again = false;
                if (inFlight > 0 || pending.isEmpty() || !awaiting.isEmpty()) {
                    break;
                }
                List<O> pass = new ArrayList<>();
                int rows = 0;
                while (!pending.isEmpty()
                        && pass.size() < maxParts
                        && rows + rowsOf.applyAsInt(pending.peek()) <= maxRows) {
                    O next = pending.poll();
                    rows += rowsOf.applyAsInt(next);
                    pass.add(next);
                }
                inFlight += passes.run(pass);
            } while (again);
        } finally {
            running = false;
        }
    }
}
