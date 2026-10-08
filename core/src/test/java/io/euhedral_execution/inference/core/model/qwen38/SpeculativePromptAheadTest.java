package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.generation.StepPort;
import io.euhedral_execution.inference.core.model.qwen38.speculative.MtpDecoder;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import io.euhedral_execution.inference.core.runtime.graph.FrameSeeds;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// A speculative prompt's chunks run ahead of their drafter steps: every chunk but the last is admitted with its
/// MTP catch-up at once, ordered by the sequence's carried state (the seed rows, the MTP cache) instead of by
/// retirement, and the prompt still leaves the sequence where the serial prompt does.
class SpeculativePromptAheadTest {

    private static final int VOCABULARY = 8;
    private static final int CHUNK = 4;

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void anMtpPromptAdmitsTheNextChunkBeforeTheCatchUpRetires() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.mtpCompactWeights(VOCABULARY));
        var gpu = new SelectingGpu();
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, gpu);
        var sequence = new Sequence(1201);
        try (var decoder = new MtpDecoder(runtime, plan, gpu, sequence, token -> false, 1, CHUNK)) {
            StepPort first = decoder.start(new int[10], 1, token -> {}, null, null, 0, tokens -> {});
            var select = new Captured();
            first.admit(select);
            // Nothing ran yet: the lattice was not driven.
            assertEquals(8, sequence.submittedFrontier(), "the chunks before the last were admitted together");
            drive(first, select, lattice);
        } finally {
            close(runtime, lattice, sequence);
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void anMtpPromptThatRunsAheadEndsWhereTheSerialPromptDoes() throws Exception {
        var plan = new ExecutionPlan(ExecutionFixtures.mtpCompactWeights(VOCABULARY));
        var gpu = new SelectingGpu();
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, gpu);
        var sequence = new Sequence(1202);
        List<List<Integer>> ended = new ArrayList<>();
        try (var decoder = new MtpDecoder(runtime, plan, gpu, sequence, token -> false, 1, CHUNK)) {
            StepPort first = decoder.start(new int[10], 1, token -> {}, null, null, 0, ended::add);
            var select = new Captured();
            first.admit(select);
            drive(first, select, lattice);
            assertEquals(1, ended.size(), "the generation ended once");
            assertEquals(1, ended.getFirst().size());
            // The prompt, then the one token fed without sampling.
            assertEquals(11, sequence.committedFrontier());
            var attention = (AttentionStates) sequence.kvCacheState();
            int mtpLayer = plan.weights().config().numHiddenLayers();
            assertEquals(10, attention.forLayer(mtpLayer).length(), "the catch-ups cover every prompt position");
        } finally {
            close(runtime, lattice, sequence);
        }
    }

    /// Runs the generation's ports to the end, as the generation frames do: `port` was admitted with `select`.
    private static void drive(StepPort port, Captured select, ExecutionFixtures.ManualLattice lattice)
            throws Exception {
        while (true) {
            lattice.drive();
            assertTrue(select.ran, "the step concluded");
            port = port.retired(select.quantum);
            if (port == null) return;
            select = new Captured();
            port.admit(select);
        }
    }

    private static void close(Execution runtime, ExecutionFixtures.ManualLattice lattice, Sequence sequence) {
        lattice.drive();
        runtime.close();
        lattice.drive();
        if (!sequence.inFlight()) sequence.complete();
    }

    /// Selects token 1 on the device for every row, as a speculative decoder's host rows read it.
    private static final class SelectingGpu extends EngineExecutionFixture.SamplingGpu {
        SelectingGpu() {
            super(VOCABULARY);
        }

        @Override
        public boolean argmaxBf16(long logitsAddress, int count, long resultAddress) {
            return true;
        }

        @Override
        public void copyDeviceToReadback(
                io.euhedral_execution.inference.core.gpu.ExecutionGpu.ReadbackBuffer destination,
                long source,
                long bytes) {
            for (long at = 0; at + Long.BYTES <= bytes; at += Long.BYTES)
                destination.segment().set(java.lang.foreign.ValueLayout.JAVA_LONG, at, (1L << 32) | (0xFFFF_FFFFL - 1));
        }
    }

    /// A step's `select`: told which quantum concluded, then run.
    private static final class Captured extends AbstractFrame implements AbstractQuantum.Continuation {
        AbstractQuantum quantum;
        boolean ran;

        Captured() {
            super(FrameSeeds.forHostWork().next());
            randomizeHash(FrameSeeds.forHostWork().next());
        }

        @Override
        public void concluded(AbstractQuantum concluded) {
            this.quantum = concluded;
        }

        @Override
        public void execute() {
            this.ran = true;
        }

        @Override
        public void doFinallyWithError(Throwable rejection) {
            execute();
        }
    }
}
