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
        /// Admits the pass and returns how many retirements it will report. Must not throw once anything was
        /// admitted: admission failures are reported through the admitted quantum's continuation. A throw is
        /// counted as no retirements and surfaces as an `IllegalStateException`.
        int run(List<O> pass);
    }

    private final Function<O, S> sequenceOf;
    private final ToIntFunction<O> rowsOf;
    private final int maxRows;
    private final int maxParts;
    private final Passes<O> passes;
    private final ArrayDeque<O> pending = new ArrayDeque<>();
    private final Set<S> awaiting = new HashSet<>();
    private final Set<S> pendingSequences = new HashSet<>();
    private final Set<S> flight = new HashSet<>();
    private final Set<S> departed = new HashSet<>();
    private int inFlight;
    private boolean running;
    private boolean again;

    /// @param sequenceOf the sequence an offer belongs to
    /// @param rowsOf the rows an offer occupies in a pass
    /// @param maxRows the row cap of a pass
    /// @param maxParts the offer cap of a pass
    /// @param passes runs a formed pass
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
        S sequence = sequenceOf.apply(offer);
        if (!pendingSequences.add(sequence)) {
            throw new IllegalStateException("a sequence already has a pending step");
        }
        awaiting.remove(sequence);
        pending.add(offer);
        tryRun();
    }

    /// Releases the wait for a sequence that will not offer again. A member of the pass in flight is recorded as
    /// departed so that its retirement does not await it.
    public void leave(S sequence) {
        if (awaiting.remove(sequence)) {
            tryRun();
        } else if (flight.contains(sequence)) {
            departed.add(sequence);
        }
    }

    /// Retires one quantum of the pass in flight; its members must report before the next pass forms, unless they
    /// already offered or left. Must be applied by an owner frame later than the one whose `Passes#run` returned
    /// the count.
    public void retired(Collection<S> members) {
        if (inFlight == 0) {
            throw new IllegalStateException("no pass is in flight");
        }
        inFlight--;
        for (S member : members) {
            flight.remove(member);
            if (!departed.remove(member) && !pendingSequences.contains(member)) {
                awaiting.add(member);
            }
        }
        if (inFlight == 0) {
            flight.clear();
            departed.clear();
        }
        tryRun();
    }

    /// Removes the pending offers matching `selected`, in arrival order.
    public List<O> withdraw(Predicate<O> selected) {
        List<O> out = new ArrayList<>();
        for (var it = pending.iterator(); it.hasNext(); ) {
            O o = it.next();
            if (selected.test(o)) {
                it.remove();
                pendingSequences.remove(sequenceOf.apply(o));
                out.add(o);
            }
        }
        return out;
    }

    /// Returns the number of offers waiting for a pass.
    public int pendingCount() {
        return pending.size();
    }

    /// Returns the number of members of the last retired pass yet to report.
    public int awaitingCount() {
        return awaiting.size();
    }

    /// Returns the number of retirements the passes in flight still owe.
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
                    pendingSequences.remove(sequenceOf.apply(next));
                    pass.add(next);
                }
                int retirements;
                try {
                    retirements = passes.run(pass);
                } catch (RuntimeException e) {
                    throw new IllegalStateException("a pass of " + pass.size() + " steps failed to run: " + pass, e);
                }
                if (retirements < 0) {
                    throw new IllegalStateException("a pass reported " + retirements + " retirements");
                }
                if (retirements > 0) {
                    inFlight += retirements;
                    for (O o : pass) {
                        flight.add(sequenceOf.apply(o));
                    }
                }
            } while (again);
        } finally {
            running = false;
        }
    }
}
