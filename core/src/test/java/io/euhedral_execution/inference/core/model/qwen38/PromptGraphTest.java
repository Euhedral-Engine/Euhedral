package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// A prompt of several chunks runs as one quantum on one graph of its chunks: one admission, one retirement, the
/// rows of each chunk where the serial path puts them, and nothing left behind when it stops early.
@Timeout(20)
class PromptGraphTest {

    private static final int VOCABULARY = 8;

    @Test
    void aPromptIsOneQuantumWithOneRetirement() throws Exception {
        var plan = new ExecutionPlan(EngineExecutionFixture.weights());
        var gpu = new EngineExecutionFixture.SamplingGpu(VOCABULARY);
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        try {
            var sequence = new Sequence(1);
            int[] tokens = {1, 2, 3, 4, 5};
            var prompt = Quantum.prompt(plan, sequence, 0, tokens, 2, LogitsRequirement.NONE, null);
            assertEquals(
                    Quantum.Status.SUCCESS,
                    runtime.submit(prompt).get(5, TimeUnit.SECONDS).status());
            assertEquals(5, sequence.currentTokenPosition(), "the whole prompt committed");
            assertFalse(sequence.inFlight());
            List<int[]> embedded = gpu.embeddingInputs;
            assertEquals(3, embedded.size(), "one embedding per chunk");
            assertArrayEquals(new int[] {1, 2}, embedded.get(0));
            assertArrayEquals(new int[] {3, 4}, embedded.get(1));
            assertArrayEquals(new int[] {5}, embedded.get(2));
        } finally {
            runtime.close();
        }
    }

    @Test
    void theLastChunkSamplesAndTheOthersDoNot() throws Exception {
        var plan = new ExecutionPlan(EngineExecutionFixture.weights());
        var gpu = new EngineExecutionFixture.SamplingGpu(VOCABULARY);
        gpu.selectTokens(6);
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        var logits = new HostLogits(gpu, VOCABULARY);
        try {
            var prompt = Quantum.prompt(
                    plan, new Sequence(2), 0, new int[] {1, 2, 3, 4, 5}, 2, LogitsRequirement.LAST_TOKEN, logits);
            assertEquals(
                    Quantum.Status.SUCCESS,
                    runtime.submit(prompt).get(5, TimeUnit.SECONDS).status());
            int best = 0;
            for (int token = 1; token < VOCABULARY; token++)
                if (logits.row().get(ValueLayout.JAVA_SHORT, token * 2L)
                        > logits.row().get(ValueLayout.JAVA_SHORT, best * 2L)) best = token;
            assertEquals(6, best, "the last chunk's row was sampled");
        } finally {
            logits.close();
            runtime.close();
        }
    }

    @Test
    void aCancelledPromptReleasesEverything() throws Exception {
        var plan = new ExecutionPlan(EngineExecutionFixture.weights());
        var stream = new ExecutionFixtures.HoldingStream();
        var gpu = new EngineExecutionFixture.SamplingGpu(VOCABULARY) {
            @Override
            public io.euhedral_execution.inference.core.gpu.GpuStream openStream() {
                return stream;
            }
        };
        var lattice = new ExecutionFixtures.ManualLattice();
        var runtime = new Execution(lattice, plan, gpu);
        try {
            var sequence = new Sequence(3);
            var prompt = Quantum.prompt(plan, sequence, 0, new int[] {1, 2, 3, 4, 5}, 2, LogitsRequirement.NONE, null);
            var outcome = runtime.submit(prompt);
            lattice.drive();
            sequence.cancel();
            while (stream.held() > 0) stream.release(null);
            lattice.drive();
            assertEquals(
                    Quantum.Status.CANCELLED, outcome.get(5, TimeUnit.SECONDS).status());
            assertEquals(Sequence.TerminalState.CANCELLED, sequence.terminalState());
            assertFalse(sequence.inFlight());
            assertEquals(0, runtime.runtime().activeQuanta(), "the graph retired");
        } finally {
            while (stream.held() > 0) stream.release(null);
            lattice.drive();
            runtime.close();
        }
    }

    @Test
    void attentionSeesOnlyItsChunksRows() throws Exception {
        var plan = new ExecutionPlan(EngineExecutionFixture.weights());
        List<Integer> cacheLengths = new java.util.concurrent.CopyOnWriteArrayList<>();
        var gpu = new EngineExecutionFixture.SamplingGpu(VOCABULARY) {
            @Override
            public void attentionCausalNvfp4(
                    long queryKeyAddress,
                    long gateValueAddress,
                    long keyCacheAddress,
                    long valueCacheAddress,
                    long outputAddress,
                    int rows,
                    int queryHeads,
                    int keyValueHeads,
                    int headDim,
                    int cacheLength,
                    long startPosition,
                    long positionAddress,
                    long scratchAddress) {
                cacheLengths.add(cacheLength);
            }
        };
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        try {
            // Every row's KV is reserved and the later chunks' appends continue the submitted frontier, but a chunk's
            // attention reads the rows up to its own end only.
            var prompt = Quantum.prompt(
                    plan, new Sequence(5), 0, new int[] {1, 2, 3, 4, 5}, 2, LogitsRequirement.NONE, null);
            assertEquals(
                    Quantum.Status.SUCCESS,
                    runtime.submit(prompt).get(5, TimeUnit.SECONDS).status());
            assertEquals(List.of(2, 4, 5), cacheLengths);
        } finally {
            runtime.close();
        }
    }

    @Test
    void aFailedChunkStopsItsDependentsAndRetiresOnce() throws Exception {
        var plan = new ExecutionPlan(EngineExecutionFixture.weights());
        var gpu = new EngineExecutionFixture.SamplingGpu(VOCABULARY) {
            private int embeddings;

            @Override
            public void embedQ3(
                    long tokenIds, long embedding, long bytes, long hidden, int count, int vocab, int size) {
                if (++this.embeddings == 2) throw new IllegalStateException("chunk 1 failed");
                super.embedQ3(tokenIds, embedding, bytes, hidden, count, vocab, size);
            }
        };
        var runtime = ExecutionFixtures.runtime(plan, gpu);
        try {
            var sequence = new Sequence(4);
            var prompt = Quantum.prompt(plan, sequence, 0, new int[] {1, 2, 3, 4, 5}, 2, LogitsRequirement.NONE, null);
            var outcome = runtime.submit(prompt).get(5, TimeUnit.SECONDS);
            assertEquals(Quantum.Status.FAILED, outcome.status());
            assertEquals(2, gpu.embeddingInputs.size() + 1, "chunk 2 never embedded");
            assertEquals(Sequence.TerminalState.FAILED, sequence.terminalState());
            assertFalse(sequence.inFlight());
        } finally {
            runtime.close();
        }
    }

    @Test
    void promptShapesAreReusedAndEvicted() {
        var plan = new ExecutionPlan(EngineExecutionFixture.weights());
        var runtime = ExecutionFixtures.runtime(plan, new EngineExecutionFixture.SamplingGpu(VOCABULARY));
        try {
            var shapes = new PromptShapes(runtime.runtime());
            var first =
                    shapes.shape(Quantum.prompt(plan, new Sequence(5), 0, new int[4], 2, LogitsRequirement.NONE, null));
            assertTrue(first
                    == shapes.shape(
                            Quantum.prompt(plan, new Sequence(6), 0, new int[4], 2, LogitsRequirement.NONE, null)));
            for (int chunks = 3; chunks < 3 + PromptShapes.CAPACITY; chunks++)
                shapes.shape(Quantum.prompt(
                        plan, new Sequence(chunks), 0, new int[2 * chunks], 2, LogitsRequirement.NONE, null));
            assertEquals(PromptShapes.CAPACITY, shapes.size(), "the least recently used shape was evicted");
        } finally {
            runtime.close();
        }
    }

    @Test
    void aCheckpointCopyFollowsItsChunkAndPrecedesTheNextChunksStateUpdates() {
        var plan = new ExecutionPlan(EngineExecutionFixture.weights());
        Shape full = plan.forExecution(Quantum.ExecutionKind.PREFILL, 64);
        var checkpointed = new CheckpointView(full);
        var shape = new io.euhedral_execution.inference.core.runtime.graph.ChunkedShape(
                new io.euhedral_execution.inference.core.runtime.graph.GraphShape[] {checkpointed, full},
                new io.euhedral_execution.inference.core.runtime.graph.ChunkedShape.Chunks() {
                    @Override
                    public io.euhedral_execution.inference.core.runtime.graph.StageFrame create(
                            io.euhedral_execution.inference.core.runtime.graph.StageGraph graph,
                            int stage,
                            int chunk,
                            int templateStage,
                            io.euhedral_execution.inference.core.gpu.ExecutionGpu gpu) {
                        throw new UnsupportedOperationException();
                    }

                    @Override
                    public io.euhedral_execution.inference.core.runtime.graph.GraphStorage newStorage(
                            io.euhedral_execution.inference.core.gpu.ExecutionGpu gpu) {
                        throw new UnsupportedOperationException();
                    }
                });
        int copy = shape.stage(0, full.topology().size());
        var topology = shape.topology();
        assertTrue(topology.inDegree(copy) > 0, "the copy follows its chunk");
        int[] successors = topology.submittedSuccessors(copy);
        boolean beforeGdn = false;
        for (int next : successors) {
            assertEquals(1, shape.chunkOf(next), "the copy is followed only by the next chunk");
            var kind = full.instructions().get(shape.templateStageOf(next)).kind();
            beforeGdn |= kind == ExecutionPlan.Kind.GDN_CONVOLUTION || kind == ExecutionPlan.Kind.GDN_RECURRENCE;
        }
        assertTrue(beforeGdn, "the next chunk's GDN updates wait for the copy");
        assertArrayEquals(
                io.euhedral_execution.inference.core.runtime.graph.GraphShape.NO_BUFFERS,
                checkpointed.workspaceBuffers(full.topology().size()),
                "the copy touches no workspace buffer");
    }
}
