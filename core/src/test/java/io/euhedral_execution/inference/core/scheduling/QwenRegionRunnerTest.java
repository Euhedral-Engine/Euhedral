package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.core.impl.DefaultExecutor;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class QwenRegionRunnerTest {
    @Test
    void oneAttachedSourceAcceptsItsPrefillAndDecodeTopologiesAndRejectsAnotherOwner() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var owner = new QwenExecutionPlan(weights, QwenExecutionPlan.PrefillRegions.RESIDUAL_NORM);
        var gpu = new RegionGpu(weights.config().vocabSize());
        var runner = new QwenExecutionRunner(owner, gpu);
        var sequences = new ArrayList<QwenSequenceState>();
        new DefaultExecutor().input(runner);
        try {
            var completions = new ArrayList<java.util.concurrent.CompletableFuture<QwenExecutionContext.Outcome>>();
            for (var kind : QwenExecutionContext.ExecutionKind.values()) {
                var sequence = new QwenSequenceState(sequences.size() + 1);
                sequences.add(sequence);
                completions.add(runner.submit(new QwenExecutionContext(
                        owner,
                        sequence,
                        kind,
                        0,
                        new int[kind == QwenExecutionContext.ExecutionKind.PREFILL ? 64 : 1],
                        QwenLogitsRequirement.NONE)));
            }
            assertEquals(0, runner.pull(frame -> fail("stop predicate must prevent transfer"), frame -> true, 1000));
            assertTrue(completions.stream().noneMatch(java.util.concurrent.CompletableFuture::isDone));
            runner.request(1000);
            for (var completion : completions)
                assertEquals(
                        QwenExecutionContext.Status.SUCCESS,
                        completion.get(2, TimeUnit.SECONDS).status());
            var foreign = new QwenExecutionPlan(weights, QwenExecutionPlan.PrefillRegions.RESIDUAL_NORM);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> runner.submit(new QwenExecutionContext(
                            foreign,
                            new QwenSequenceState(900),
                            QwenExecutionContext.ExecutionKind.PREFILL,
                            0,
                            new int[64])));
        } finally {
            runner.completeGracefully();
            runner.awaitTermination();
            sequences.forEach(QwenSequenceState::complete);
        }
        assertTrue(runner.isComplete());
        assertFalse(runner.isAttached());
    }

    @Test
    void runnerBuiltFromDerivedViewAdmitsRequalifiedFallbackAndDecode() throws Exception {
        for (var regions : new QwenExecutionPlan.PrefillRegions[] {
            QwenExecutionPlan.PrefillRegions.ATTENTION_KV, QwenExecutionPlan.PrefillRegions.STREAMED_KV
        }) {
            var weights = EngineExecutionFixture.weights();
            var owner = new QwenExecutionPlan(weights, regions);
            var view = owner.forExecution(QwenExecutionContext.ExecutionKind.PREFILL);
            var gpu = new RegionGpu(weights.config().vocabSize());
            var runner = new QwenExecutionRunner(view, gpu);
            new DefaultExecutor().input(runner);
            try {
                for (var kind : QwenExecutionContext.ExecutionKind.values()) {
                    var sequence = new QwenSequenceState(700 + kind.ordinal());
                    try {
                        var context = new QwenExecutionContext(
                                view, sequence, kind, 0, new int[1], QwenLogitsRequirement.NONE);
                        assertSame(owner.forExecution(kind, 1), context.plan());
                        var completion = runner.submit(context);
                        runner.request(1000);
                        assertEquals(
                                QwenExecutionContext.Status.SUCCESS,
                                completion.get(2, TimeUnit.SECONDS).status());
                    } finally {
                        sequence.complete();
                    }
                }
            } finally {
                runner.completeGracefully();
                runner.awaitTermination();
            }
        }
    }

    @Test
    void gracefulCloseWaitsForDerivedGpuCompletionAndItsSuccessors() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var owner = new QwenExecutionPlan(weights, QwenExecutionPlan.PrefillRegions.RESIDUAL_NORM);
        var gpu = new HoldingRegionGpu(weights.config().vocabSize());
        var sequence = new QwenSequenceState(601);
        var runner = new QwenExecutionRunner(owner, gpu);
        new DefaultExecutor().input(runner);
        try {
            var completion = runner.submit(new QwenExecutionContext(
                    owner,
                    sequence,
                    QwenExecutionContext.ExecutionKind.PREFILL,
                    0,
                    new int[64],
                    QwenLogitsRequirement.NONE));
            runner.request(1);
            assertNotNull(gpu.completion);
            runner.completeGracefully();
            assertFalse(runner.isComplete());
            assertFalse(completion.isDone());
            gpu.completion.run();
            runner.request(1000);
            assertEquals(
                    QwenExecutionContext.Status.SUCCESS,
                    completion.get(2, TimeUnit.SECONDS).status());
            runner.awaitTermination();
            assertTrue(runner.isComplete());
        } finally {
            runner.completeGracefully();
            sequence.complete();
        }
    }

    @Test
    void cancellingTheCallersFutureDoesNotRetireDerivedGpuWork() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var owner = new QwenExecutionPlan(weights, QwenExecutionPlan.PrefillRegions.RESIDUAL_NORM);
        var gpu = new HoldingRegionGpu(weights.config().vocabSize());
        var sequence = new QwenSequenceState(602);
        var runner = new QwenExecutionRunner(owner, gpu);
        var context = new QwenExecutionContext(
                owner,
                sequence,
                QwenExecutionContext.ExecutionKind.PREFILL,
                0,
                new int[64],
                QwenLogitsRequirement.NONE);
        new DefaultExecutor().input(runner);
        try {
            var caller = runner.submit(context);
            runner.request(1);
            assertNotNull(gpu.completion);
            assertTrue(caller.cancel(false));
            runner.completeGracefully();
            assertFalse(runner.isComplete(), "caller cancellation must not retire internal GPU work");
            assertFalse(context.outcome().isDone());
            Runnable held = gpu.completion;
            gpu.completion = null;
            held.run();
            runner.request(1000);
            assertEquals(
                    QwenExecutionContext.Status.SUCCESS,
                    context.outcome().get(2, TimeUnit.SECONDS).status());
            runner.awaitTermination();
            assertTrue(runner.isComplete());
            assertTrue(caller.isCancelled());
        } finally {
            context.cancel();
            if (gpu.completion != null) gpu.completion.run();
            runner.completeGracefully();
            sequence.complete();
        }
    }

    private static final class HoldingRegionGpu extends RegionGpu {
        private boolean hold = true;
        private Runnable completion;

        HoldingRegionGpu(int vocabularySize) {
            super(vocabularySize);
        }

        @Override
        public boolean asynchronous() {
            return true;
        }

        @Override
        public void deferCompletion(Runnable completed, java.util.function.Consumer<Throwable> failed) {
            if (hold) {
                hold = false;
                completion = completed;
            } else completed.run();
        }
    }

    private static class RegionGpu extends EngineExecutionFixture.SamplingGpu {
        @Override
        public void gdnProjectControlFp32(
                long input, long a, long b, long log, long bias, long g, long beta, int rows, int width, int heads) {}

        RegionGpu(int vocabularySize) {
            super(vocabularySize);
        }

        @Override
        public void residualRmsNormBf16(
                long residual,
                long delta,
                long weights,
                long hidden,
                long normalized,
                int rows,
                int width,
                float epsilon) {}
    }
}
