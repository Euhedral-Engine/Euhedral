package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class PassAssemblerTest {
    record Step(String sequence, int rows, int serial) {}

    private final List<List<Step>> passes = new ArrayList<>();
    private int serial;

    private PassAssembler<String, Step> assembler(int maxRows, int maxParts) {
        return new PassAssembler<>(Step::sequence, Step::rows, maxRows, maxParts, pass -> {
            passes.add(List.copyOf(pass));
            return 1;
        });
    }

    private Step step(String sequence, int rows) {
        return new Step(sequence, rows, serial++);
    }

    private static List<String> names(List<Step> pass) {
        return pass.stream().map(s -> s.sequence()).toList();
    }

    @Test
    void aFirstOfferRunsAtOnceAlone() {
        var a = assembler(8, 8);
        Step s = step("A", 1);
        a.offer(s);
        assertEquals(List.of(List.of(s)), passes);
        assertEquals(1, a.passesInFlight());
        assertEquals(0, a.pendingCount());
    }

    @Test
    void offersDuringAPassWaitForEveryMemberToReport() {
        var a = assembler(8, 8);
        a.offer(step("A", 1));
        a.offer(step("B", 1));
        assertEquals(1, passes.size());
        a.retired(List.of("A"));
        assertEquals(1, passes.size());
        assertEquals(1, a.awaitingCount());
        a.offer(step("A", 1));
        assertEquals(2, passes.size());
        assertEquals(List.of("B", "A"), names(passes.get(1)));
    }

    @Test
    void aMemberThatLeavesReleasesTheWait() {
        var a = assembler(8, 8);
        a.offer(step("A", 1));
        a.offer(step("B", 1));
        a.retired(List.of("A"));
        assertEquals(1, passes.size());
        a.leave("A");
        assertEquals(2, passes.size());
        assertEquals(List.of("B"), names(passes.get(1)));
        assertEquals(0, a.awaitingCount());
    }

    @Test
    void leavingWithoutBeingAwaitedChangesNothing() {
        var a = assembler(8, 8);
        a.offer(step("A", 1));
        a.offer(step("B", 1));
        a.leave("Z");
        a.leave("B");
        assertEquals(1, passes.size());
        assertEquals(1, a.pendingCount());
    }

    @Test
    void theRowCapSplitsInArrivalOrderAndStopsAtTheFirstMisfit() {
        var a = assembler(4, 8);
        a.offer(step("X", 1));
        a.offer(step("A", 3));
        a.offer(step("B", 3));
        a.offer(step("C", 1));
        a.retired(List.of());
        assertEquals(2, passes.size());
        assertEquals(List.of("A"), names(passes.get(1)));
        a.retired(List.of("A"));
        assertEquals(2, passes.size());
        a.offer(step("A", 3));
        assertEquals(3, passes.size());
        assertEquals(List.of("B", "C"), names(passes.get(2)));
        assertEquals(1, a.pendingCount());
    }

    @Test
    void maxPartsOneRunsSingles() {
        var a = assembler(8, 1);
        a.offer(step("A", 1));
        a.offer(step("B", 1));
        a.offer(step("C", 1));
        a.retired(List.of());
        assertEquals(List.of("B"), names(passes.get(1)));
        a.retired(List.of());
        assertEquals(List.of("C"), names(passes.get(2)));
    }

    @Test
    void anOversizedStepRunsAloneAndAnEmptyOneIsRefused() {
        var a = assembler(4, 8);
        Step big = step("A", 5);
        a.offer(big);
        assertEquals(List.of(List.of(big)), passes);
        assertThrows(IllegalArgumentException.class, () -> a.offer(step("B", 0)));
        assertEquals(0, a.pendingCount());
        assertEquals(1, passes.size());
    }

    @Test
    void anEmptyStepStillReportsItsSequence() {
        var a = assembler(4, 8);
        a.offer(step("A", 1));
        a.offer(step("B", 1));
        a.retired(List.of("A"));
        assertEquals(1, passes.size());
        assertThrows(IllegalArgumentException.class, () -> a.offer(step("A", 0)));
        assertEquals(2, passes.size());
    }

    @Test
    void aPassThatAdmitsNothingDoesNotStrandTheOffersBehindIt() {
        List<List<Step>> seen = new ArrayList<>();
        var a = new PassAssembler<String, Step>(Step::sequence, Step::rows, 8, 1, pass -> {
            seen.add(List.copyOf(pass));
            return pass.get(0).sequence().equals("BAD") ? 0 : 1;
        });
        a.offer(step("X", 1));
        a.offer(step("BAD", 1));
        a.offer(step("B", 1));
        a.retired(List.of());
        assertEquals(
                List.of("X", "BAD", "B"),
                seen.stream().map(p -> p.get(0).sequence()).toList());
        assertEquals(0, a.pendingCount());
        assertEquals(1, a.passesInFlight());
    }

    @Test
    void aStepExactlyTheRowCapRunsAlone() {
        var a = assembler(4, 8);
        a.offer(step("X", 1));
        a.offer(step("A", 1));
        a.offer(step("B", 4));
        a.retired(List.of());
        assertEquals(List.of("A"), names(passes.get(1)));
        a.retired(List.of());
        assertEquals(List.of("B"), names(passes.get(2)));
    }

    @Test
    void withdrawRemovesOnlyMatchingPendingOffers() {
        var a = assembler(8, 8);
        a.offer(step("X", 1));
        Step b = step("B", 1);
        Step c = step("C", 1);
        Step b2 = step("D", 1);
        a.offer(b);
        a.offer(c);
        a.offer(b2);
        assertEquals(
                List.of(b, b2),
                a.withdraw(s -> s.sequence().equals("B") || s.sequence().equals("D")));
        assertEquals(1, a.pendingCount());
        a.retired(List.of());
        assertEquals(List.of("C"), names(passes.get(1)));
        assertEquals(List.of(), a.withdraw(s -> true));
    }

    @Test
    void withdrawingEverythingPendingFormsNoEmptyPass() {
        var a = assembler(8, 8);
        a.offer(step("X", 1));
        a.offer(step("B", 1));
        a.withdraw(s -> true);
        a.retired(List.of());
        assertEquals(1, passes.size());
        assertEquals(0, a.passesInFlight());
    }

    @Test
    void anEmptyRetirementFormsTheNextPassAtOnce() {
        var a = assembler(8, 8);
        a.offer(step("A", 1));
        a.offer(step("B", 1));
        a.retired(List.of());
        assertEquals(2, passes.size());
        assertEquals(List.of("B"), names(passes.get(1)));
    }

    @Test
    void retiringWithNothingInFlightIsRefused() {
        var a = assembler(8, 8);
        assertThrows(IllegalStateException.class, () -> a.retired(List.of()));
        a.offer(step("A", 1));
        a.retired(List.of());
        assertThrows(IllegalStateException.class, () -> a.retired(List.of()));
    }

    @Test
    void aLastMemberLeavingWithNothingPendingFormsNothing() {
        var a = assembler(8, 8);
        a.offer(step("A", 1));
        a.retired(List.of("A"));
        a.leave("A");
        assertEquals(1, passes.size());
        a.offer(step("B", 1));
        assertEquals(2, passes.size());
    }

    @Test
    void offersAreHeldWhileAPassIsInFlightEvenWhenNobodyIsAwaited() {
        var a = assembler(8, 8);
        a.offer(step("A", 1));
        a.offer(step("B", 1));
        a.offer(step("C", 1));
        assertEquals(1, passes.size());
        assertEquals(2, a.pendingCount());
        a.retired(List.of("A"));
        a.offer(step("A", 1));
        assertEquals(List.of("B", "C", "A"), names(passes.get(1)));
    }

    @Test
    @SuppressWarnings("unchecked")
    void anOfferMadeWhileAPassIsBeingRunWaitsForItsRetirement() {
        List<List<Step>> seen = new ArrayList<>();
        PassAssembler<String, Step>[] holder = new PassAssembler[1];
        Step late = step("B", 1);
        holder[0] = new PassAssembler<>(Step::sequence, Step::rows, 4, 8, pass -> {
            seen.add(List.copyOf(pass));
            if (seen.size() == 1) {
                holder[0].offer(late);
            }
            return 1;
        });
        holder[0].offer(step("A", 1));
        assertEquals(1, seen.size());
        assertEquals(1, holder[0].pendingCount());
        holder[0].retired(List.of());
        assertEquals(List.of(List.of(seen.get(0).get(0)), List.of(late)), seen);
    }

    @Test
    @SuppressWarnings("unchecked")
    void aRetirementBeforeTheRunReturnsItsCountIsRefused() {
        PassAssembler<String, Step>[] holder = new PassAssembler[1];
        holder[0] = new PassAssembler<>(Step::sequence, Step::rows, 1, 8, pass -> {
            assertThrows(IllegalStateException.class, () -> holder[0].retired(List.of()));
            return 1;
        });
        holder[0].offer(step("A", 1));
        assertEquals(1, holder[0].passesInFlight());
    }

    @Test
    void aPassThatAdmitsNothingLeavesNothingInFlight() {
        List<List<Step>> seen = new ArrayList<>();
        var a = new PassAssembler<String, Step>(Step::sequence, Step::rows, 4, 8, pass -> {
            seen.add(List.copyOf(pass));
            return 0;
        });
        a.offer(step("A", 1));
        a.offer(step("B", 1));
        assertEquals(2, seen.size());
        assertEquals(0, a.passesInFlight());
    }

    @Test
    void aMemberThatLeavesBeforeItsPassRetiresIsNotAwaited() {
        var a = assembler(8, 8);
        a.offer(step("A", 1));
        a.offer(step("B", 1));
        a.leave("A");
        a.retired(List.of("A"));
        assertEquals(0, a.awaitingCount());
        assertEquals(2, passes.size());
        assertEquals(List.of("B"), names(passes.get(1)));
    }

    @Test
    void aMemberThatOfferedBeforeItsPassRetiresIsNotAwaited() {
        var a = assembler(8, 8);
        a.offer(step("A", 1));
        a.offer(step("A", 1));
        a.retired(List.of("A"));
        assertEquals(0, a.awaitingCount());
        assertEquals(2, passes.size());
    }

    @Test
    void aFailedRunLeavesTheGateConsistent() {
        boolean[] fail = {true};
        var a = new PassAssembler<String, Step>(Step::sequence, Step::rows, 4, 8, pass -> {
            if (fail[0]) {
                throw new RuntimeException("boom");
            }
            passes.add(List.copyOf(pass));
            return 1;
        });
        assertThrows(IllegalStateException.class, () -> a.offer(step("A", 1)));
        assertEquals(0, a.passesInFlight());
        assertEquals(0, a.pendingCount());
        fail[0] = false;
        a.offer(step("A", 1));
        assertEquals(1, passes.size());
        assertThrows(
                IllegalStateException.class,
                () -> new PassAssembler<String, Step>(Step::sequence, Step::rows, 4, 8, pass -> -1)
                        .offer(step("B", 1)));
    }

    @Test
    void aSecondPendingOfferFromOneSequenceIsRefused() {
        var a = assembler(8, 8);
        a.offer(step("A", 1));
        a.offer(step("B", 1));
        assertThrows(IllegalStateException.class, () -> a.offer(step("B", 1)));
        assertEquals(1, a.pendingCount());
    }
}
