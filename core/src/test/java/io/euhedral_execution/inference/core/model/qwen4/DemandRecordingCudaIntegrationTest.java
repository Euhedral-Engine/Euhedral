package io.euhedral_execution.inference.core.model.qwen4;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertBank;
import io.euhedral_execution.inference.core.model.qwen4.loader.HostBudget;
import io.euhedral_execution.inference.core.model.qwen4.loader.Mode;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/// Records what the expert cache is asked for on a realistic run, for a simulator to replay through other cache
/// policies (`tools/expert_cache_sim.py`). Requests are slices of the repository's documents, each prefilled in the
/// plan's chunks and continued greedily, one after another as a server would take them. Runs only when
/// `EUHEDRAL_QWEN4_TRACE` names the output file; `EUHEDRAL_QWEN4_TRACE_DECODE` sets the generated tokens per request
/// (256).
///
/// The file is little-endian 32-bit integers: the header `EXD1`, version 1, device slots, tier slots (0 without a
/// tier), banks, then each bank's experts and record bytes; then records. A block is `1, layer, bank, rows, count`
/// followed by `count` pairs of expert and routed rows; a mark is `2, kind, request, tokens` with kind 0 for a
/// request's prefill and 1 for its decode; a prediction is `3, layer, bank, count` followed by `count` experts of the
/// next layer ranked by its router applied to a decode step's input at the layer before (written before the block of
/// the layer before is reported).
/// Own JVM: records an expert trace of a whole run with a lattice of its own: opt-in.
@ModelGroup.OwnJvm
class DemandRecordingCudaIntegrationTest {

    /// Prompt lengths of the requests, in the order they arrive.
    private static final int[] REQUESTS = {512, 2048, 300, 4096, 1200, 700, 3000, 150};

    private static float bf(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }

    private static int argmax(short[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) if (bf(logits[i]) > bf(logits[best])) best = i;
        return best;
    }

    /// Writes blocks and marks; blocks come from the plan stages of one sequence at a time, marks from the test.
    private static final class Recorder implements ExecutionPlan.ExpertDemand, AutoCloseable {
        private final DataOutputStream out;

        Recorder(OutputStream stream) {
            this.out = new DataOutputStream(new BufferedOutputStream(stream, 1 << 20));
        }

        synchronized void header(int deviceSlots, int tierSlots, ExpertBank[] banks) throws IOException {
            write(0x31445845); // "EXD1"
            write(1);
            write(deviceSlots);
            write(tierSlots);
            write(banks.length);
            for (ExpertBank bank : banks) {
                write(bank.expertCount());
                write((int) bank.recordBytes(0));
            }
        }

        synchronized void mark(int kind, int request, int tokens) throws IOException {
            write(2);
            write(kind);
            write(request);
            write(tokens);
        }

        @Override
        public boolean predicts() {
            return true;
        }

        @Override
        public synchronized void prediction(int layer, int bank, int[] ranked) {
            try {
                write(3);
                write(layer);
                write(bank);
                write(ranked.length);
                for (int expert : ranked) write(expert);
            } catch (IOException failure) {
                throw new java.io.UncheckedIOException(failure);
            }
        }

        @Override
        public synchronized void block(int layer, int bank, int rows, int[] experts, int[] pairs, int count) {
            try {
                write(1);
                write(layer);
                write(bank);
                write(rows);
                write(count);
                for (int i = 0; i < count; i++) {
                    write(experts[i]);
                    write(pairs[i]);
                }
            } catch (IOException failure) {
                throw new java.io.UncheckedIOException(failure);
            }
        }

        private void write(int value) throws IOException {
            this.out.writeInt(Integer.reverseBytes(value));
        }

        @Override
        public synchronized void close() throws IOException {
            this.out.close();
        }
    }

    @Test
    void recordTheExpertCachesDemand() throws Exception {
        String target = System.getenv("EUHEDRAL_QWEN4_TRACE");
        assumeTrue(target != null, "set EUHEDRAL_QWEN4_TRACE to the output file");
        assumeTrue(TestSupport.hasArtifact(), "no artifact");
        int decode = Integer.parseInt(System.getenv().getOrDefault("EUHEDRAL_QWEN4_TRACE_DECODE", "256"));
        // `EUHEDRAL_QWEN4_TRACE_REQUESTS` (comma-separated prompt lengths) and `EUHEDRAL_QWEN4_TRACE_OFFSET` (where in
        // the
        // documents the first request starts) give other workloads.
        String lengths = System.getenv("EUHEDRAL_QWEN4_TRACE_REQUESTS");
        int[] requests = lengths == null
                ? REQUESTS
                : java.util.Arrays.stream(lengths.split(","))
                        .mapToInt(Integer::parseInt)
                        .toArray();
        int start = Integer.parseInt(System.getenv().getOrDefault("EUHEDRAL_QWEN4_TRACE_OFFSET", "0"));
        int context = 0;
        for (int length : requests) context = Math.max(context, length + decode);
        int[] corpus = PerformanceCudaIntegrationTest.corpus(1 << 16);
        try (TestLattice lattice = TestLattice.start(4);
                CudaGpuMemory gpu = TestSupport.openGpu();
                Qwen4Model model = Qwen4Model.open(
                        TestSupport.artifactPath(),
                        gpu,
                        gpu.deviceMemoryInfo().freeBytes(),
                        HostBudget.system(),
                        Mode.TEXT,
                        context);
                TestLattice.Run run = lattice.run(gpu, model, context);
                ExecutionPlan executor = run.plan();
                Recorder recorder = new Recorder(Files.newOutputStream(Path.of(target)))) {
            var tier = model.hierarchyStats().ram();
            recorder.header(
                    model.expertCache().slotCount(), tier == null ? 0 : model.ramTierSlots(), model.expertBanks());
            int vocabulary = executor.vocabularySize();
            var readback = gpu.allocateReadbackBuffer((long) vocabulary * 2);
            ExecutionPlan.LogitsSink sink = address -> gpu.copyDeviceToReadback(readback, address, vocabulary * 2L);
            executor.demand(recorder);
            try {
                int offset = start;
                for (int request = 0; request < requests.length; request++) {
                    int length = requests[request];
                    int[] prompt = new int[length];
                    for (int i = 0; i < length; i++) prompt[i] = corpus[(offset + i) % corpus.length];
                    offset += 7919 + length;
                    try (Sequence sequence = executor.newSequence()) {
                        recorder.mark(0, request, length);
                        int at = 0;
                        while (at < length) {
                            int rows = Math.min(executor.maxRows(), length - at);
                            Blocking.step(executor, sequence, prompt, at, rows, at + rows == length ? sink : null);
                            at += rows;
                        }
                        recorder.mark(1, request, decode);
                        short[] logits = new short[vocabulary];
                        int[] token = new int[1];
                        for (int step = 0; step < decode; step++) {
                            MemorySegment.copy(readback.segment(), ValueLayout.JAVA_SHORT, 0, logits, 0, vocabulary);
                            token[0] = argmax(logits);
                            Blocking.step(executor, sequence, token, 0, 1, sink);
                        }
                    }
                }
            } finally {
                executor.demand(null);
                readback.close();
            }
        }
    }
}
