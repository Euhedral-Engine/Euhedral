package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.core.impl.DefaultExecutor;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class QwenStreamedFfnLifecycleTest {
    enum Scenario {
        SUCCESS,
        PARTIAL_LAUNCH,
        COMPLETION_REGISTRATION,
        COMPLETION_FAILURE,
        CANCEL,
        UNPROVEN_DRAIN
    }

    @ParameterizedTest
    @EnumSource(Scenario.class)
    void borrowedSlotsAndContinuationStayOwnedUntilCompletion(Scenario scenario) throws Exception {
        runCase(scenario, true);
    }

    @ParameterizedTest
    @EnumSource(
            value = Scenario.class,
            names = {"PARTIAL_LAUNCH", "UNPROVEN_DRAIN"})
    void synchronousOuterModeStillRecoversInternalStreamFailures(Scenario scenario) throws Exception {
        runCase(scenario, false);
    }

    @org.junit.jupiter.api.Test
    void synchronousCompletionFailureAlsoQuarantinesBorrowedStorage() throws Exception {
        runCase(Scenario.UNPROVEN_DRAIN, false, true);
    }

    private static void runCase(Scenario scenario, boolean asynchronous) throws Exception {
        runCase(scenario, asynchronous, false);
    }

    private static void runCase(Scenario scenario, boolean asynchronous, boolean nativeSucceeded) throws Exception {
        var weights = QwenExecutionFixtures.statefulCompactWeights(8, 5120, 17408);
        var plan = new QwenExecutionPlan(weights, QwenExecutionPlan.PrefillRegions.STREAMED_FFN)
                .forExecution(QwenExecutionContext.ExecutionKind.PREFILL, 256);
        var sequence = new QwenSequenceState(901);
        var gpu = new HoldingGpu(scenario, asynchronous);
        gpu.nativeSucceeded = nativeSucceeded;
        var context = new QwenExecutionContext(
                plan,
                sequence,
                QwenExecutionContext.ExecutionKind.PREFILL,
                0,
                new int[256],
                QwenLogitsRequirement.NONE);
        var runner = new QwenExecutionRunner(plan, gpu);
        new DefaultExecutor().input(runner);
        try {
            var outcome = runner.submit(context);
            runner.request(plan.instructions().size());
            assertTrue(gpu.calls > 0, "must reach the real streamed FFN frame");
            if (gpu.completion != null) {
                assertFalse(outcome.isDone());
                assertTrue(gpu.borrowed.stream().noneMatch(gpu.freed()::contains));
                if (scenario == Scenario.CANCEL) sequence.cancel();
                if (scenario == Scenario.COMPLETION_FAILURE)
                    gpu.failure.accept(new IllegalStateException("completion"));
                else gpu.completion.run();
                gpu.completion = null;
                runner.request(plan.instructions().size());
            }
            var expected = scenario == Scenario.SUCCESS
                    ? QwenExecutionContext.Status.SUCCESS
                    : scenario == Scenario.CANCEL
                            ? QwenExecutionContext.Status.CANCELLED
                            : QwenExecutionContext.Status.FAILED;
            assertEquals(expected, outcome.get(2, TimeUnit.SECONDS).status());
            if (scenario == Scenario.UNPROVEN_DRAIN) {
                assertTrue(gpu.poisoned);
                assertTrue(gpu.borrowed.stream().noneMatch(gpu.freed()::contains));
            } else {
                assertFalse(gpu.pending);
                assertTrue(gpu.freed().containsAll(gpu.borrowed));
            }
            if (scenario == Scenario.PARTIAL_LAUNCH || scenario == Scenario.COMPLETION_REGISTRATION)
                assertTrue(gpu.synchronizations > 0);
        } finally {
            context.cancel();
            if (gpu.completion != null) gpu.completion.run();
            runner.completeGracefully();
            sequence.complete();
        }
    }

    private static final class HoldingGpu extends EngineExecutionFixture.SamplingGpu {
        final Scenario scenario;
        final boolean asynchronous;
        Set<Long> borrowed = Set.of();
        int calls;
        boolean pending, holdNext, poisoned, nativeSucceeded;
        Runnable completion;
        Consumer<Throwable> failure;

        HoldingGpu(Scenario scenario, boolean asynchronous) {
            super(8);
            this.scenario = scenario;
            this.asynchronous = asynchronous;
        }

        @Override
        public boolean asynchronous() {
            return this.asynchronous;
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
        public void synchronize() {
            if (pending && scenario == Scenario.UNPROVEN_DRAIN) throw new IllegalStateException("drain");
            pending = false;
            super.synchronize();
        }

        @Override
        public synchronized void free(long address) {
            assertFalse(pending && borrowed.contains(address), "borrowed FFN storage released before drain");
            super.free(address);
        }

        @Override
        public void deferCompletion(Runnable completed, Consumer<Throwable> failed) {
            if (!holdNext) {
                pending = false;
                completed.run();
                return;
            }
            holdNext = false;
            if (scenario == Scenario.COMPLETION_REGISTRATION) throw new IllegalStateException("register");
            completion = () -> {
                pending = false;
                completed.run();
            };
            failure = error -> {
                pending = false;
                failed.accept(error);
            };
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
            assertEquals(256, rows);
            assertEquals(5120, hidden);
            assertEquals(17408, intermediate);
            borrowed = Set.of(input, output, slots, accumulators);
            pending = true;
            calls++;
            if (!nativeSucceeded
                    && calls == 1
                    && (scenario == Scenario.PARTIAL_LAUNCH || scenario == Scenario.UNPROVEN_DRAIN))
                throw new IllegalStateException("partial native FFN");
            holdNext = calls == 1;
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
    }
}
