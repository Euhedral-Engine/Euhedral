package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/// Borrowed streamed-FFN storage stays owned until the quantum's device work has provably retired.
class QwenStreamedFfnLifecycleTest {
    enum Scenario {
        SUCCESS,
        PARTIAL_LAUNCH,
        COMPLETION_REGISTRATION,
        COMPLETION_FAILURE,
        CANCEL,
        UNPROVEN_DRAIN,
        UNPROVEN_DRAIN_AFTER_SUCCESSFUL_LAUNCH
    }

    @ParameterizedTest
    @EnumSource(Scenario.class)
    void borrowedSlotsAndContinuationStayOwnedUntilRetirement(Scenario scenario) throws Exception {
        runCase(scenario, 1024);
    }

    private static void runCase(Scenario scenario, int rows) throws Exception {
        var weights = QwenExecutionFixtures.statefulCompactWeights(8, 5120, 17408);
        var plan = new QwenExecutionPlan(weights).forExecution(QwenExecutionContext.ExecutionKind.PREFILL, rows);
        var sequence = new QwenSequenceState(2048);
        var gpu = new HoldingGpu(scenario);
        var context = new QwenExecutionContext(
                plan,
                sequence,
                QwenExecutionContext.ExecutionKind.PREFILL,
                0,
                new int[rows],
                QwenLogitsRequirement.NONE);
        var lattice = new QwenExecutionFixtures.ManualLattice();
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        try {
            var outcome = runtime.submit(context);
            lattice.drive();
            assertTrue(gpu.calls > 0, "must reach the real streamed FFN stage");
            if (scenario != Scenario.COMPLETION_REGISTRATION) {
                assertEquals(1, gpu.stream.armed, "one device-completion boundary for the quantum");
                assertFalse(outcome.isDone());
                assertTrue(gpu.borrowed.stream().noneMatch(gpu.freed()::contains));
                if (scenario == Scenario.CANCEL) sequence.cancel();
                gpu.stream.announce();
                lattice.drive();
            }
            var expected = scenario == Scenario.SUCCESS
                    ? QwenExecutionContext.Status.SUCCESS
                    : scenario == Scenario.CANCEL
                            ? QwenExecutionContext.Status.CANCELLED
                            : QwenExecutionContext.Status.FAILED;
            assertEquals(expected, outcome.get(2, TimeUnit.SECONDS).status());
            if (scenario == Scenario.UNPROVEN_DRAIN || scenario == Scenario.UNPROVEN_DRAIN_AFTER_SUCCESSFUL_LAUNCH) {
                assertTrue(gpu.poisoned);
            } else {
                assertFalse(gpu.pending);
            }
            // The graph retains the borrowed slots for its next quantum; only closing the runtime frees them.
            assertTrue(gpu.borrowed.stream().noneMatch(gpu.freed()::contains));
            if (scenario == Scenario.COMPLETION_REGISTRATION) assertTrue(gpu.stream.recoveries > 0);
        } finally {
            context.cancel();
            runtime.close();
            // A poisoned device retains the sequence's persistent state and the graph's storage as well.
            if (gpu.poisoned) {
                assertTrue(gpu.borrowed.stream().noneMatch(gpu.freed()::contains));
                assertThrows(IllegalStateException.class, sequence::complete);
            } else {
                assertTrue(gpu.freed().containsAll(gpu.borrowed));
                sequence.complete();
            }
        }
    }

    private static final class HoldingGpu extends EngineExecutionFixture.SamplingGpu {
        final Scenario scenario;
        final Stream stream = new Stream();
        Set<Long> borrowed = Set.of();
        int calls;
        boolean pending, poisoned;

        HoldingGpu(Scenario scenario) {
            super(8);
            this.scenario = scenario;
        }

        @Override
        public GpuStream openStream() {
            return this.stream;
        }

        @Override
        public boolean completionProven() {
            return !poisoned;
        }

        @Override
        public void poison(Throwable failure) {
            poisoned = true;
        }

        @Override
        public synchronized void free(long address) {
            // A poisoned device retains every allocation, as the CUDA binding does.
            if (poisoned) throw new IllegalStateException("GPU is poisoned; ownership is retained");
            assertFalse(pending && borrowed.contains(address), "borrowed FFN storage released before retirement");
            super.free(address);
        }

        @Override
        public void q3FfnStreamedBf16(
                long input,
                long gateWeights,
                long downWeights,
                long output,
                long slots,
                long accumulators,
                int rows,
                int hidden,
                int intermediate,
                long gateBytes,
                long downBytes) {
            assertTrue(rows == 64 || rows == 1024);
            assertEquals(5120, hidden);
            assertEquals(17408, intermediate);
            borrowed = Set.of(input, output, slots, accumulators);
            pending = true;
            calls++;
            if (calls == 1 && (scenario == Scenario.PARTIAL_LAUNCH || scenario == Scenario.UNPROVEN_DRAIN))
                throw new IllegalStateException("partial native FFN");
        }

        @Override
        public void residualRmsNormBf16(
                long residual,
                long delta,
                long weight,
                long hidden,
                long normalized,
                int rows,
                int width,
                float epsilon) {}

        @Override
        public void gdnProjectControlFp32(
                long input,
                long aWeight,
                long bWeight,
                long aLog,
                long dtBias,
                long g,
                long beta,
                int rows,
                int width,
                int heads) {}

        /// The quantum's device ordering: its boundary retires only when the test announces it.
        final class Stream implements GpuStream {
            int armed;
            int recoveries;
            private RetirementListener listener;
            private long ticket;

            @Override
            public void submit(Runnable launches, boolean overlapPredecessor) {
                launches.run();
            }

            @Override
            public long notifyRetired(RetirementListener retired) {
                if (scenario == Scenario.COMPLETION_REGISTRATION) throw new IllegalStateException("register");
                this.armed++;
                this.listener = retired;
                return ++this.ticket;
            }

            void announce() {
                this.listener.retired(this.ticket, true);
            }

            @Override
            public Throwable confirmRetired(long retired) {
                return switch (scenario) {
                    case COMPLETION_FAILURE -> {
                        pending = false;
                        yield new IllegalStateException("completion");
                    }
                    case UNPROVEN_DRAIN, UNPROVEN_DRAIN_AFTER_SUCCESSFUL_LAUNCH -> {
                        IllegalStateException failure = new IllegalStateException("drain");
                        poison(failure);
                        yield failure;
                    }
                    default -> {
                        pending = false;
                        yield null;
                    }
                };
            }

            @Override
            public void synchronize() {
                pending = false;
            }

            @Override
            public void recover(Throwable failure) {
                this.recoveries++;
                pending = false;
            }

            @Override
            public void close() {}
        }
    }
}
