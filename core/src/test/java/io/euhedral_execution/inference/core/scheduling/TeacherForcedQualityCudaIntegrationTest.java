package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.HostWeightSelection;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.WeightResidency;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Measurement tool, not a regression test (skipped unless `euhedral.quality.report` names an output
/// file): teacher-forced decode of the benchmark corpus document, recording every step's BF16 logits
/// row. Artifacts compare by their per-token negative log-likelihood of the document (paired, on the
/// same tokens) and by the KL divergence between their logits rows (tools in docs/NVFP4_COMPRESSED.md).
///
/// Report format, little endian: int32 steps, int32 vocabulary, then per step the int32 forced next
/// token and the logits row as `vocabulary` BF16 values.
class TeacherForcedQualityCudaIntegrationTest {

    @Test
    @Timeout(value = 7200, unit = TimeUnit.SECONDS)
    void recordTeacherForcedLogits() throws Throwable {
        String report = System.getProperty("euhedral.quality.report", "");
        assumeTrue(!report.isBlank(), "measurement tool");
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        Path artifact = Path.of(System.getProperty("euhedral.quality.artifact"));
        int prefix = Integer.getInteger("euhedral.quality.prefix", 512);
        int steps = Integer.getInteger("euhedral.quality.steps", 2048);
        long hostBytes = Long.getLong("euhedral.quality.host-mib", 0L) << 20;
        QwenTokenizer tokenizer = QwenTokenizer.load(
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen")));
        int[] document = tokenizer.encodeText(Files.readString(SpeculativeVerifyCudaIntegrationTest.repositoryRoot()
                .resolve("benchmark/src/main/resources/io/euhedral_execution/inference/benchmark/prompt/"
                        + "chat-corpus-v1-document.md")));
        if (document.length < prefix + steps + 1) throw new IllegalStateException("document too short");
        var data = QwenArtifactReader.read(artifact);
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                QwenModel model = QwenModel.load(
                        artifact,
                        data,
                        gpu,
                        WeightResidency.EXECUTED,
                        HostWeightSelection.select(data, hostBytes),
                        QwenModel.DEFAULT_STAGING_SLOTS);
                var lattice = new PullingLattice();
                OutputStream file = Files.newOutputStream(Path.of(report));
                DataOutputStream out = new DataOutputStream(new BufferedOutputStream(file, 1 << 22))) {
            var plan = new QwenExecutionPlan(model.weights(), model.staging());
            var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
            var sequence = new QwenSequenceState(1);
            int vocabulary = model.weights().config().vocabSize();
            out.writeInt(Integer.reverseBytes(steps));
            out.writeInt(Integer.reverseBytes(vocabulary));
            byte[] row = new byte[vocabulary * Short.BYTES];
            double nll = 0;
            try {
                short[] logits = run(
                        runtime,
                        gpu,
                        plan,
                        sequence,
                        QwenExecutionContext.ExecutionKind.PREFILL,
                        Arrays.copyOf(document, prefix));
                for (int step = 0; step < steps; step++) {
                    int next = document[prefix + step];
                    out.writeInt(Integer.reverseBytes(next));
                    for (int i = 0; i < vocabulary; i++) {
                        row[2 * i] = (byte) logits[i];
                        row[2 * i + 1] = (byte) (logits[i] >> 8);
                    }
                    out.write(row);
                    nll += negativeLogLikelihood(logits, next);
                    logits = run(
                            runtime, gpu, plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, new int[] {next});
                }
            } finally {
                sequence.complete();
                runtime.close();
            }
            System.out.printf(
                    Locale.ROOT,
                    "quality %s: %d forced tokens after %d, mean NLL %.5f nats%n",
                    artifact.getFileName(),
                    steps,
                    prefix,
                    nll / steps);
        }
    }

    private static double negativeLogLikelihood(short[] logits, int token) {
        double max = Double.NEGATIVE_INFINITY;
        for (short value : logits) max = Math.max(max, Float.intBitsToFloat(value << 16));
        double sum = 0;
        for (short value : logits) sum += Math.exp(Float.intBitsToFloat(value << 16) - max);
        return max + Math.log(sum) - Float.intBitsToFloat(logits[token] << 16);
    }

    private static short[] run(
            EuhedralInferenceRuntime runtime,
            CudaGpuMemory gpu,
            QwenExecutionPlan plan,
            QwenSequenceState sequence,
            QwenExecutionContext.ExecutionKind kind,
            int[] tokens)
            throws Exception {
        AtomicReference<short[]> captured = new AtomicReference<>();
        var context = new QwenExecutionContext(
                plan, sequence, kind, sequence.currentTokenPosition(), tokens, QwenLogitsRequirement.LAST_TOKEN);
        var outcome = runtime.submit(context, done -> {
                    try (Arena arena = Arena.ofConfined();
                            QwenDeviceLogits logits = done.logitsOutput().orElseThrow()) {
                        int vocabulary = logits.vocabularySize();
                        MemorySegment host = arena.allocate((long) vocabulary * Short.BYTES, Short.BYTES);
                        gpu.copyDeviceToHost(
                                host,
                                logits.deviceAddress() + (long) (logits.tokenCount() - 1) * vocabulary * Short.BYTES,
                                host.byteSize());
                        short[] values = new short[vocabulary];
                        MemorySegment.copy(
                                host,
                                ValueLayout.JAVA_SHORT.withOrder(java.nio.ByteOrder.LITTLE_ENDIAN),
                                0,
                                values,
                                0,
                                vocabulary);
                        captured.set(values);
                    }
                })
                .get(600, TimeUnit.SECONDS);
        if (outcome.status() != QwenExecutionContext.Status.SUCCESS) throw new AssertionError(outcome.failure());
        return captured.get();
    }
}
