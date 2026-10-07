package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import org.junit.jupiter.api.Test;

class AttentionAppendTransactionTest {
    @Test
    void cacheCommitWaitsForTheAppendsRetirement() throws Exception {
        runCase(false, false, false);
    }

    @Test
    void failedAppendLaunchDoesNotCommit() throws Exception {
        runCase(true, false, false);
    }

    @Test
    void deviceCompletionFailureDoesNotCommit() throws Exception {
        runCase(false, true, false);
    }

    @Test
    void cancellationNeverCommitsTheSubmittedAppend() throws Exception {
        runCase(false, false, true);
    }

    private static void runCase(boolean launchFailure, boolean completionFailure, boolean cancel) throws Exception {
        var weights = EngineExecutionFixture.weights();
        var plan = new ExecutionPlan(weights);
        var sequence = new Sequence(807);
        var gpu = new HoldingGpu(weights.config().vocabSize(), sequence, launchFailure);
        var context =
                new Quantum(plan, sequence, Quantum.ExecutionKind.PREFILL, 0, new int[64], LogitsRequirement.NONE);
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, gpu);
        try {
            var outcome = runtime.submit(context);
            lattice.drive();
            assertTrue(gpu.called, "the real append stage must submit");
            assertFalse(outcome.isDone(), "the quantum waits for its device-completion boundary");
            assertEquals(1, gpu.stream.held());
            assertEquals(0, gpu.cache.length(), "physical writes cannot publish the logical append");
            assertEquals(
                    launchFailure ? 0 : 64,
                    gpu.cache.submittedLength(),
                    "later stages of the quantum read the submitted frontier");
            if (cancel) sequence.cancel();
            assertEquals(0, gpu.cache.length());
            gpu.stream.release(completionFailure ? new IllegalStateException("injected completion failure") : null);
            lattice.drive();
            var result = outcome.get(2, java.util.concurrent.TimeUnit.SECONDS);
            var expected = launchFailure || completionFailure
                    ? Quantum.Status.FAILED
                    : cancel ? Quantum.Status.CANCELLED : Quantum.Status.SUCCESS;
            assertEquals(expected, result.status());
            if (expected == Quantum.Status.SUCCESS) assertEquals(64, gpu.cache.length());
            else assertEquals(0, gpu.lengthAtFree, "an unsuccessful quantum never publishes its append");
            context.logitsOutput().ifPresent(DeviceLogits::close);
        } finally {
            runtime.close();
            sequence.complete();
        }
    }

    private static final class HoldingGpu extends EngineExecutionFixture.SamplingGpu {
        private final Sequence sequence;
        private final boolean launchFailure;
        private AttentionKvState cache;
        private long cacheAddress;
        private final ExecutionFixtures.HoldingStream stream = new ExecutionFixtures.HoldingStream();
        private boolean called;
        private int lengthAtFree = -1;

        HoldingGpu(int vocabulary, Sequence sequence, boolean launchFailure) {
            super(vocabulary);
            this.sequence = sequence;
            this.launchFailure = launchFailure;
        }

        @Override
        public GpuStream openStream() {
            return this.stream;
        }

        @Override
        public synchronized void free(long address) {
            if (address == cacheAddress) lengthAtFree = cache.length();
            super.free(address);
        }

        @Override
        public void attentionKvAppendNvfp4(
                long queryKey,
                long gateValue,
                long keys,
                long values,
                int rows,
                int queryWidth,
                int keyValueWidth,
                long start,
                long position) {
            called = true;
            cache = ((AttentionStates) sequence.kvCacheState()).forLayer(1);
            cacheAddress = keys;
            assertTrue(cache.capacity() >= rows, "reservation must precede the physical write");
            assertEquals(0, cache.length());
            assertEquals(keys, cache.keyCacheAddress());
            assertEquals(values, cache.valueCacheAddress());
            if (launchFailure) throw new IllegalStateException("injected append launch failure");
        }
    }
}
