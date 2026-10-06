package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/// One runtime admits every plan view of its owner; each reusable graph publishes through its own
/// source.
class QwenRegionRuntimeTest {

    private final QwenExecutionFixtures.ManualLattice lattice = new QwenExecutionFixtures.ManualLattice();

    private EuhedralInferenceRuntime runtime(QwenExecutionPlan plan, RegionGpu gpu) {
        return new EuhedralInferenceRuntime(this.lattice, plan, gpu);
    }

    @Test
    void oneRuntimeAcceptsItsPrefillAndDecodeTopologiesAndRejectsAnotherOwner() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var owner = new QwenExecutionPlan(weights);
        var gpu = new RegionGpu(weights.config().vocabSize());
        var runtime = runtime(owner, gpu);
        var sequences = new ArrayList<QwenSequenceState>();
        try {
            var completions = new ArrayList<CompletableFuture<QwenExecutionContext.Outcome>>();
            for (var kind :
                    List.of(QwenExecutionContext.ExecutionKind.PREFILL, QwenExecutionContext.ExecutionKind.DECODE)) {
                var sequence = new QwenSequenceState(sequences.size() + 1);
                sequences.add(sequence);
                completions.add(runtime.submit(new QwenExecutionContext(
                        owner,
                        sequence,
                        kind,
                        0,
                        new int[kind == QwenExecutionContext.ExecutionKind.PREFILL ? 64 : 1],
                        QwenLogitsRequirement.NONE)));
            }
            assertEquals(
                    0, this.lattice.pull(frame -> fail("stop predicate must prevent transfer"), frame -> true, 1000));
            assertTrue(completions.stream().noneMatch(CompletableFuture::isDone));
            assertEquals(2, runtime.activeQuanta());
            this.lattice.drive();
            for (var completion : completions)
                assertEquals(
                        QwenExecutionContext.Status.SUCCESS,
                        completion.get(2, TimeUnit.SECONDS).status());
            var foreign = new QwenExecutionPlan(weights);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> runtime.submit(new QwenExecutionContext(
                            foreign,
                            new QwenSequenceState(900),
                            QwenExecutionContext.ExecutionKind.PREFILL,
                            0,
                            new int[64])));
        } finally {
            runtime.close();
            sequences.forEach(QwenSequenceState::complete);
        }
        assertTrue(this.lattice.sources.stream().allMatch(LatticeSource::isComplete));
        assertEquals(
                EuhedralInferenceRuntime.LAKE_SINKS, this.lattice.sources.size(), "the lake's sinks, attached once");
        assertFalse(runtime.isAttached());
    }

    @Test
    void runtimeBuiltFromDerivedViewAdmitsRequalifiedFallbackAndDecode() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var owner = new QwenExecutionPlan(weights);
        var view = owner.forExecution(QwenExecutionContext.ExecutionKind.PREFILL, 256);
        var gpu = new RegionGpu(weights.config().vocabSize());
        var runtime = runtime(view, gpu);
        try {
            for (var kind :
                    List.of(QwenExecutionContext.ExecutionKind.PREFILL, QwenExecutionContext.ExecutionKind.DECODE)) {
                for (int rows : new int[] {1, 64}) {
                    if (kind == QwenExecutionContext.ExecutionKind.DECODE && rows != 1) continue;
                    var sequence = new QwenSequenceState(700 + kind.ordinal() * 100 + rows);
                    try {
                        var context = new QwenExecutionContext(
                                view, sequence, kind, 0, new int[rows], QwenLogitsRequirement.NONE);
                        assertSame(owner.forExecution(kind, rows), context.plan());
                        var completion = runtime.submit(context);
                        this.lattice.drive();
                        assertEquals(
                                QwenExecutionContext.Status.SUCCESS,
                                completion.get(2, TimeUnit.SECONDS).status());
                    } finally {
                        sequence.complete();
                    }
                }
            }
        } finally {
            runtime.close();
        }
    }

    @Test
    void gracefulCloseWaitsForTheQuantumsDeviceRetirement() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var owner = new QwenExecutionPlan(weights);
        var gpu = new HoldingRegionGpu(weights.config().vocabSize());
        var sequence = new QwenSequenceState(601);
        var runtime = runtime(owner, gpu);
        try {
            var completion = runtime.submit(new QwenExecutionContext(
                    owner,
                    sequence,
                    QwenExecutionContext.ExecutionKind.PREFILL,
                    0,
                    new int[64],
                    QwenLogitsRequirement.NONE));
            this.lattice.drive();
            assertEquals(1, gpu.stream.held(), "every stage submitted; only the device boundary is pending");
            var closing = CompletableFuture.runAsync(runtime::close);
            assertThrows(java.util.concurrent.TimeoutException.class, () -> closing.get(200, TimeUnit.MILLISECONDS));
            assertFalse(this.lattice.source.isComplete());
            assertFalse(completion.isDone());
            gpu.stream.release(null);
            assertFalse(completion.isDone(), "the driver callback only enqueues the retirement frame");
            this.lattice.drive();
            assertEquals(
                    QwenExecutionContext.Status.SUCCESS,
                    completion.get(2, TimeUnit.SECONDS).status());
            closing.get(2, TimeUnit.SECONDS);
            assertTrue(this.lattice.source.isComplete());
        } finally {
            sequence.complete();
        }
    }

    @Test
    void cancellingTheCallersFutureDoesNotRetireSubmittedGpuWork() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var owner = new QwenExecutionPlan(weights);
        var gpu = new HoldingRegionGpu(weights.config().vocabSize());
        var sequence = new QwenSequenceState(602);
        var runtime = runtime(owner, gpu);
        var context = new QwenExecutionContext(
                owner,
                sequence,
                QwenExecutionContext.ExecutionKind.PREFILL,
                0,
                new int[64],
                QwenLogitsRequirement.NONE);
        try {
            var caller = runtime.submit(context);
            this.lattice.drive();
            assertEquals(1, gpu.stream.held());
            assertTrue(caller.cancel(false));
            assertFalse(context.outcome().isDone(), "caller cancellation must not retire internal GPU work");
            assertEquals(1, runtime.activeQuanta());
            gpu.stream.release(null);
            this.lattice.drive();
            assertEquals(
                    QwenExecutionContext.Status.SUCCESS,
                    context.outcome().get(2, TimeUnit.SECONDS).status());
            assertTrue(caller.isCancelled());
            assertEquals(0, runtime.activeQuanta());
        } finally {
            runtime.close();
            sequence.complete();
        }
    }

    private static final class HoldingRegionGpu extends RegionGpu {
        final QwenExecutionFixtures.HoldingStream stream = new QwenExecutionFixtures.HoldingStream();

        HoldingRegionGpu(int vocabularySize) {
            super(vocabularySize);
        }

        @Override
        public GpuStream openStream() {
            return this.stream;
        }
    }

    private static class RegionGpu extends EngineExecutionFixture.SamplingGpu {
        RegionGpu(int vocabularySize) {
            super(vocabularySize);
        }
    }
}
