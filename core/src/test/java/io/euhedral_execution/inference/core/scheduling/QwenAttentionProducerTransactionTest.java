package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.core.impl.DefaultExecutor;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

class QwenAttentionProducerTransactionTest {
    @Test
    void cacheCommitWaitsForBothProducers() throws Exception {
        runCase(false, false, false);
    }

    @Test
    void failureBetweenPhysicalProducersDoesNotCommit() throws Exception {
        runCase(true, false, false);
    }

    @Test
    void deviceCompletionFailureDoesNotCommit() throws Exception {
        runCase(false, true, false);
    }

    @Test
    void cancellationKeepsCacheUncommittedUntilBothWritesFinish() throws Exception {
        runCase(false, false, true);
    }

    private static void runCase(boolean launchFailure, boolean completionFailure, boolean cancel) throws Exception {
        var weights = EngineExecutionFixture.weights();
        var plan = new QwenExecutionPlan(weights).forExecution(QwenExecutionContext.ExecutionKind.PREFILL, 64);
        var sequence = new QwenSequenceState(807);
        var gpu = new HoldingGpu(weights.config().vocabSize(), sequence, launchFailure);
        var context = new QwenExecutionContext(
                plan, sequence, QwenExecutionContext.ExecutionKind.PREFILL, 0, new int[64], QwenLogitsRequirement.NONE);
        var runner = new QwenExecutionRunner(plan, gpu);
        new DefaultExecutor().input(runner);
        try {
            var outcome = runner.submit(context);
            runner.request(plan.instructions().size());
            assertTrue(gpu.called, "the real frame must submit the producer region");
            if (!launchFailure) {
                assertFalse(outcome.isDone());
                assertEquals(0, gpu.cache.length(), "physical writes cannot publish the logical append");
                if (cancel) sequence.cancel();
                assertEquals(0, gpu.cache.length());
                if (completionFailure) gpu.failure.accept(new IllegalStateException("injected completion failure"));
                else gpu.completion.run();
                gpu.completion = null;
                runner.request(plan.instructions().size());
            }
            var result = outcome.get(2, java.util.concurrent.TimeUnit.SECONDS);
            var expected = launchFailure || completionFailure
                    ? QwenExecutionContext.Status.FAILED
                    : cancel ? QwenExecutionContext.Status.CANCELLED : QwenExecutionContext.Status.SUCCESS;
            assertEquals(expected, result.status());
            if (expected == QwenExecutionContext.Status.SUCCESS) assertEquals(64, gpu.cache.length());
            else assertEquals(launchFailure || completionFailure ? 0 : 64, gpu.lengthAtFree);
            context.logitsOutput().ifPresent(QwenDeviceLogits::close);
        } finally {
            if (gpu.completion != null) gpu.completion.run();
            runner.completeGracefully();
            sequence.complete();
        }
    }

    private static final class HoldingGpu extends EngineExecutionFixture.SamplingGpu {
        private final QwenSequenceState sequence;
        private final boolean launchFailure;
        private AttentionKvState cache;
        private long cacheAddress;
        private boolean called, hold;
        private int lengthAtFree = -1;
        private Runnable completion;
        private Consumer<Throwable> failure;

        HoldingGpu(int vocabulary, QwenSequenceState sequence, boolean launchFailure) {
            super(vocabulary);
            this.sequence = sequence;
            this.launchFailure = launchFailure;
        }

        @Override
        public boolean asynchronous() {
            return true;
        }

        @Override
        public void deferCompletion(Runnable completed, Consumer<Throwable> failed) {
            if (hold) {
                completion = completed;
                failure = failed;
                hold = false;
            } else completed.run();
        }

        @Override
        public synchronized void free(long address) {
            if (address == cacheAddress) lengthAtFree = cache.length();
            super.free(address);
        }

        @Override
        public void attentionProducersNvfp4(
                long input,
                long q4,
                long q5,
                long queryNorm,
                long keyNorm,
                long queryKey,
                long gate,
                long keys,
                long values,
                int rows,
                int hidden,
                int queryHeads,
                int keyHeads,
                int headDim,
                int rotaryDim,
                long start,
                float epsilon,
                double theta,
                long q4Bytes,
                long q5Bytes) {
            called = true;
            cache = ((AttentionSequenceStates) sequence.kvCacheState()).forLayer(1);
            cacheAddress = keys;
            assertTrue(cache.capacity() >= rows, "reservation must precede either physical write");
            assertEquals(0, cache.length());
            assertEquals(keys, cache.keyCacheAddress());
            assertEquals(values, cache.valueCacheAddress());
            if (launchFailure) throw new IllegalStateException("injected failure after key producer");
            hold = true;
        }
    }
}
