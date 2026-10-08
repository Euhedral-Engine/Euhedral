package io.euhedral_execution.inference.core.generation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import io.euhedral_execution.inference.core.runtime.graph.FrameLake;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import org.junit.jupiter.api.Test;

class GenerationFramesTest {

    /// Runs frames one at a time, as workers would, on the test thread.
    private static final class Lake implements FrameLake {
        final ArrayDeque<AbstractFrame> ready = new ArrayDeque<>();
        int admitted;
        int terminated;
        boolean refuse;

        @Override
        public void publish(AbstractFrame frame) {
            if (this.refuse) throw new IllegalStateException("the lake refuses frames");
            this.ready.add(frame);
        }

        @Override
        public void publishFromCallback(AbstractFrame frame) {
            publish(frame);
        }

        @Override
        public void admit() {
            this.admitted++;
        }

        @Override
        public void admitDuringDrain() {
            this.admitted++;
        }

        @Override
        public void terminated() {
            this.terminated++;
        }

        void run() {
            AbstractFrame frame;
            while ((frame = this.ready.poll()) != null) {
                frame.execute();
                frame.doFinally();
            }
        }
    }

    /// A port whose step is a quantum that concludes immediately; `steps` counts admissions.
    private static class Counting implements StepPort {
        final Lake lake;
        final Generation generation;
        final int total;
        final List<String> log;
        int steps;
        boolean stop;

        Counting(Lake lake, Generation generation, int total, List<String> log) {
            this.lake = lake;
            this.generation = generation;
            this.total = total;
            this.log = log;
        }

        @Override
        public void admit(AbstractFrame select) {
            this.steps++;
            this.log.add("admit " + this.steps);
            var quantum = new AbstractQuantum() {};
            quantum.continueWith(this.lake, select);
            quantum.retire(null);
            quantum.publishOutcome();
        }

        @Override
        public StepPort retired(AbstractQuantum step) {
            this.log.add("retired " + this.steps);
            if (this.stop || this.steps == this.total) {
                this.generation.complete(List.of(this.steps));
                return null;
            }
            return this;
        }
    }

    @Test
    void admitIsOrderedOnTheWorkspaceOwnerAndTheOthersDrawANewSeedOnReuse() {
        var lake = new Lake();
        var frames = new GenerationFrames(lake);
        var generation = new Generation(frames);
        StepPort port = new Counting(lake, generation, 1, new ArrayList<>());
        List<AbstractFrame> firsts =
                List.of(frames.admit(generation, port), frames.select(generation, port), frames.finish(generation));
        List<Long> hashes = new ArrayList<>();
        for (AbstractFrame frame : firsts) {
            hashes.add(frame.getRoutingHash());
            frame.recycle();
        }
        List<AbstractFrame> reused =
                List.of(frames.admit(generation, port), frames.select(generation, port), frames.finish(generation));
        for (int i = 0; i < 3; i++) assertSame(firsts.get(i), reused.get(i), "recycled");
        AbstractFrame admit = reused.getFirst();
        org.junit.jupiter.api.Assertions.assertTrue(admit.isOrdered(), "Admit runs ordered on the workspace owner");
        org.junit.jupiter.api.Assertions.assertEquals(
                io.euhedral_execution.inference.core.runtime.graph.WorkspaceOwner.HASH, admit.getRoutingHash());
        for (int i = 1; i < 3; i++) {
            org.junit.jupiter.api.Assertions.assertFalse(reused.get(i).isOrdered());
            org.junit.jupiter.api.Assertions.assertNotEquals(
                    hashes.get(i), reused.get(i).getRoutingHash(), "a reused parallel frame draws a new seed");
        }
    }

    @Test
    void aLakeThatRefusesFramesStillConcludesTheGeneration() throws Exception {
        var lake = new Lake();
        lake.refuse = true;
        var generation = new Generation(new GenerationFrames(lake));
        generation.start(new Counting(lake, generation, 3, new ArrayList<>()));
        assertTrue(generation.result().isDone(), "every frame ran where the lake refused it");
        assertEquals(List.of(3), generation.result().get());
        assertEquals(lake.admitted, lake.terminated);
    }

    @Test
    void aChainRunsItsStepsThenFinishes() throws Exception {
        var lake = new Lake();
        var frames = new GenerationFrames(lake);
        var generation = new Generation(frames);
        var log = new ArrayList<String>();
        var port = new Counting(lake, generation, 3, log);
        generation.start(port);
        lake.run();
        assertEquals(List.of(3), generation.result().get());
        assertEquals(List.of("admit 1", "retired 1", "admit 2", "retired 2", "admit 3", "retired 3"), log);
    }

    @Test
    void framesAreRecycledWithNewPorts() throws Exception {
        var lake = new Lake();
        var frames = new GenerationFrames(lake);
        for (int call = 0; call < 4; call++) {
            var generation = new Generation(frames);
            generation.start(new Counting(lake, generation, 5, new ArrayList<>()));
            lake.run();
            assertEquals(List.of(5), generation.result().get());
        }
        assertTrue(
                frames.createdFrames() <= 3,
                "one Admit, Select and Finish serve every step: " + frames.createdFrames());
    }

    @Test
    void aPortThatSeesTheStopEndsBeforeAnotherAdmit() throws Exception {
        var lake = new Lake();
        var generation = new Generation(new GenerationFrames(lake));
        var log = new ArrayList<String>();
        var port = new Counting(lake, generation, 10, log) {
            @Override
            public StepPort retired(AbstractQuantum step) {
                if (this.steps == 2) this.stop = true;
                return super.retired(step);
            }
        };
        generation.start(port);
        lake.run();
        assertEquals(List.of(2), generation.result().get());
        assertEquals(2, port.steps, "no Admit after the stop");
    }

    @Test
    void aRefusedAdmissionFinishesOnceWithItsFailure() {
        var lake = new Lake();
        var refused = new IllegalStateException("refused");
        var ended = new ArrayList<Throwable>();
        var counted = new Generation(new GenerationFrames(lake)) {
            @Override
            protected void ended(List<Integer> tokens, Throwable failure) {
                ended.add(failure);
            }
        };
        counted.start(new StepPort() {
            @Override
            public void admit(AbstractFrame select) {
                throw refused;
            }

            @Override
            public StepPort retired(AbstractQuantum step) {
                throw new AssertionError("a refused step never retires");
            }
        });
        lake.run();
        var failure = org.junit.jupiter.api.Assertions.assertThrows(
                ExecutionException.class, () -> counted.result().get());
        assertSame(refused, failure.getCause());
        assertEquals(1, ended.size(), "Finish runs exactly once");
    }

    @Test
    void aThrowingRetirementFailsTheGeneration() {
        var lake = new Lake();
        var generation = new Generation(new GenerationFrames(lake));
        var broken = new IllegalStateException("select");
        generation.start(new Counting(lake, generation, 3, new ArrayList<>()) {
            @Override
            public StepPort retired(AbstractQuantum step) {
                throw broken;
            }
        });
        lake.run();
        var failure = org.junit.jupiter.api.Assertions.assertThrows(
                ExecutionException.class, () -> generation.result().get());
        assertSame(broken, failure.getCause());
    }

    @Test
    void everyGenerationBalancesTheLakesUnits() {
        var lake = new Lake();
        var frames = new GenerationFrames(lake);
        var ran = new Generation(frames);
        ran.start(new Counting(lake, ran, 2, new ArrayList<>()));
        var skipped = new Generation(frames);
        skipped.finishNow();
        lake.run();
        assertEquals(2, lake.admitted);
        assertEquals(2, lake.terminated);
    }

    @Test
    void endedRunsBeforeTheResultCompletes() throws Exception {
        var lake = new Lake();
        var order = new ArrayList<String>();
        var generation = new Generation(new GenerationFrames(lake)) {
            @Override
            protected void ended(List<Integer> tokens, Throwable failure) {
                order.add("ended " + tokens + " done=" + result().isDone());
            }
        };
        generation.complete(List.of(7));
        generation.finishNow();
        lake.run();
        assertEquals(List.of("ended [7] done=false"), order);
        assertEquals(List.of(7), generation.result().get());
    }
}
