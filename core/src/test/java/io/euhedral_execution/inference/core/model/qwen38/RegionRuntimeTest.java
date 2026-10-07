package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/// One runtime admits every plan view of its owner; each reusable graph publishes through its own
/// source.
class RegionRuntimeTest {

    private final ExecutionFixtures.ManualLattice lattice = new ExecutionFixtures.ManualLattice();

    private EuhedralInferenceRuntime runtime(ExecutionPlan plan, RegionGpu gpu) {
        return new EuhedralInferenceRuntime(this.lattice, plan, gpu);
    }

    @Test
    void oneRuntimeAcceptsItsPrefillAndDecodeTopologiesAndRejectsAnotherOwner() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var owner = new ExecutionPlan(weights);
        var gpu = new RegionGpu(weights.config().vocabSize());
        var runtime = runtime(owner, gpu);
        var sequences = new ArrayList<Sequence>();
        try {
            var completions = new ArrayList<CompletableFuture<Quantum.Outcome>>();
            for (var kind : List.of(Quantum.ExecutionKind.PREFILL, Quantum.ExecutionKind.DECODE)) {
                var sequence = new Sequence(sequences.size() + 1);
                sequences.add(sequence);
                completions.add(runtime.submit(new Quantum(
                        owner,
                        sequence,
                        kind,
                        0,
                        new int[kind == Quantum.ExecutionKind.PREFILL ? 64 : 1],
                        LogitsRequirement.NONE)));
            }
            assertEquals(
                    0, this.lattice.pull(frame -> fail("stop predicate must prevent transfer"), frame -> true, 1000));
            assertTrue(completions.stream().noneMatch(CompletableFuture::isDone));
            assertEquals(2, runtime.activeQuanta());
            this.lattice.drive();
            for (var completion : completions)
                assertEquals(
                        Quantum.Status.SUCCESS,
                        completion.get(2, TimeUnit.SECONDS).status());
            var foreign = new ExecutionPlan(weights);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> runtime.submit(
                            new Quantum(foreign, new Sequence(900), Quantum.ExecutionKind.PREFILL, 0, new int[64])));
        } finally {
            runtime.close();
            sequences.forEach(Sequence::complete);
        }
        assertTrue(this.lattice.sources.stream().allMatch(LatticeSource::isComplete));
        assertEquals(
                EuhedralInferenceRuntime.LAKE_SINKS, this.lattice.sources.size(), "the lake's sinks, attached once");
        assertFalse(runtime.isAttached());
    }

    @Test
    void runtimeBuiltFromDerivedViewAdmitsRequalifiedFallbackAndDecode() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var owner = new ExecutionPlan(weights);
        var view = owner.forExecution(Quantum.ExecutionKind.PREFILL, 256);
        var gpu = new RegionGpu(weights.config().vocabSize());
        var runtime = runtime(view, gpu);
        try {
            for (var kind : List.of(Quantum.ExecutionKind.PREFILL, Quantum.ExecutionKind.DECODE)) {
                for (int rows : new int[] {1, 64}) {
                    if (kind == Quantum.ExecutionKind.DECODE && rows != 1) continue;
                    var sequence = new Sequence(700 + kind.ordinal() * 100 + rows);
                    try {
                        var context = new Quantum(view, sequence, kind, 0, new int[rows], LogitsRequirement.NONE);
                        assertSame(owner.forExecution(kind, rows), context.plan());
                        var completion = runtime.submit(context);
                        this.lattice.drive();
                        assertEquals(
                                Quantum.Status.SUCCESS,
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
        var owner = new ExecutionPlan(weights);
        var gpu = new HoldingRegionGpu(weights.config().vocabSize());
        var sequence = new Sequence(601);
        var runtime = runtime(owner, gpu);
        try {
            var completion = runtime.submit(new Quantum(
                    owner, sequence, Quantum.ExecutionKind.PREFILL, 0, new int[64], LogitsRequirement.NONE));
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
                    Quantum.Status.SUCCESS, completion.get(2, TimeUnit.SECONDS).status());
            closing.get(2, TimeUnit.SECONDS);
            assertTrue(this.lattice.source.isComplete());
        } finally {
            sequence.complete();
        }
    }

    @Test
    void cancellingTheCallersFutureDoesNotRetireSubmittedGpuWork() throws Exception {
        var weights = EngineExecutionFixture.weights();
        var owner = new ExecutionPlan(weights);
        var gpu = new HoldingRegionGpu(weights.config().vocabSize());
        var sequence = new Sequence(602);
        var runtime = runtime(owner, gpu);
        var context =
                new Quantum(owner, sequence, Quantum.ExecutionKind.PREFILL, 0, new int[64], LogitsRequirement.NONE);
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
                    Quantum.Status.SUCCESS,
                    context.outcome().get(2, TimeUnit.SECONDS).status());
            assertTrue(caller.isCancelled());
            assertEquals(0, runtime.activeQuanta());
        } finally {
            runtime.close();
            sequence.complete();
        }
    }

    private static final class HoldingRegionGpu extends RegionGpu {
        final ExecutionFixtures.HoldingStream stream = new ExecutionFixtures.HoldingStream();

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
