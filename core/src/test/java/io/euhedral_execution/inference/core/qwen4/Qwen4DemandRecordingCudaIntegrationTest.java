package io.euhedral_execution.inference.core.qwen4;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.qwen4.HostBudget;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Mode;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Model;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;
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
/// request's prefill and 1 for its decode.
class Qwen4DemandRecordingCudaIntegrationTest {

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
    private static final class Recorder implements Qwen4ExecutionPlan.ExpertDemand, AutoCloseable {
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
        assumeTrue(Qwen4TestSupport.hasArtifact(), "no artifact");
        int decode = Integer.parseInt(System.getenv().getOrDefault("EUHEDRAL_QWEN4_TRACE_DECODE", "256"));
        int context = 0;
        for (int length : REQUESTS) context = Math.max(context, length + decode);
        int[] corpus = Qwen4PerformanceCudaIntegrationTest.corpus(1 << 16);
        try (Qwen4TestLattice lattice = Qwen4TestLattice.start(4);
                CudaGpuMemory gpu = Qwen4TestSupport.openGpu();
                Qwen4Model model = Qwen4Model.open(
                        Qwen4TestSupport.artifactPath(),
                        gpu,
                        gpu.deviceMemoryInfo().freeBytes(),
                        HostBudget.system(),
                        Qwen4Mode.TEXT,
                        context);
                Qwen4TestLattice.Run run = lattice.run(gpu, model, context);
                Qwen4ExecutionPlan executor = run.plan();
                Recorder recorder = new Recorder(Files.newOutputStream(Path.of(target)))) {
            var tier = model.hierarchyStats().ram();
            recorder.header(
                    model.expertCache().slotCount(), tier == null ? 0 : model.ramTierSlots(), model.expertBanks());
            int vocabulary = executor.vocabularySize();
            var readback = gpu.allocateReadbackBuffer((long) vocabulary * 2);
            Qwen4ExecutionPlan.LogitsSink sink =
                    address -> gpu.copyDeviceToReadback(readback, address, vocabulary * 2L);
            executor.demand(recorder);
            try {
                int offset = 0;
                for (int request = 0; request < REQUESTS.length; request++) {
                    int length = REQUESTS[request];
                    int[] prompt = new int[length];
                    for (int i = 0; i < length; i++) prompt[i] = corpus[(offset + i) % corpus.length];
                    offset += 7919 + length;
                    try (Qwen4Sequence sequence = executor.newSequence()) {
                        recorder.mark(0, request, length);
                        int at = 0;
                        while (at < length) {
                            int rows = Math.min(executor.maxRows(), length - at);
                            Qwen4Blocking.step(executor, sequence, prompt, at, rows, at + rows == length ? sink : null);
                            at += rows;
                        }
                        recorder.mark(1, request, decode);
                        short[] logits = new short[vocabulary];
                        int[] token = new int[1];
                        for (int step = 0; step < decode; step++) {
                            MemorySegment.copy(readback.segment(), ValueLayout.JAVA_SHORT, 0, logits, 0, vocabulary);
                            token[0] = argmax(logits);
                            Qwen4Blocking.step(executor, sequence, token, 0, 1, sink);
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
