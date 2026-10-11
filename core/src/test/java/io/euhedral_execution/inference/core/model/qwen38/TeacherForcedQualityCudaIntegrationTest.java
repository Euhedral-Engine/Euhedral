package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.HostWeightSelection;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.testing.ModelGroup;
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
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Measurement tool, not a regression test (skipped unless `euhedral.quality.report` names an output
/// file): teacher-forced decode of the benchmark corpus document, recording every step's BF16 logits
/// row. With `euhedral.quality.tokens` naming a llama.cpp reference logits file, every chunk of that file
/// is forced instead, each on a fresh sequence, into `REPORT-chunkNNN.bin` (docs/QUALITY.md). Artifacts compare by
/// their per-token negative log-likelihood of the document (paired, on the
/// same tokens) and by the KL divergence between their logits rows (tools in docs/NVFP4_COMPRESSED.md).
///
/// Report format, little endian: int32 steps, int32 vocabulary, then per step the int32 forced next
/// token and the logits row as `vocabulary` BF16 values.
@ModelGroup.OwnJvm // loads the artifact named by euhedral.quality.artifact, not one of the shared ones
class TeacherForcedQualityCudaIntegrationTest {

    @Test
    @Timeout(value = 86400, unit = TimeUnit.SECONDS)
    void recordTeacherForcedLogits() throws Throwable {
        String report = System.getProperty("euhedral.quality.report", "");
        assumeTrue(!report.isBlank(), "measurement tool");
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        Path artifact = Path.of(System.getProperty("euhedral.quality.artifact"));
        long hostBytes = Long.getLong("euhedral.quality.host-mib", 0L) << 20;
        String tokenFile = System.getProperty("euhedral.quality.tokens", "");
        List<int[]> chunks;
        int prefix;
        int steps;
        if (tokenFile.isBlank()) {
            prefix = Integer.getInteger("euhedral.quality.prefix", 512);
            steps = Integer.getInteger("euhedral.quality.steps", 2048);
            QwenTokenizer tokenizer = QwenTokenizer.load(
                    Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen")));
            int[] document = tokenizer.encodeText(Files.readString(SpeculativeVerifyCudaIntegrationTest.repositoryRoot()
                    .resolve("benchmark/src/main/resources/io/euhedral_execution/inference/benchmark/prompt/"
                            + "chat-corpus-v1-document.md")));
            if (document.length < prefix + steps + 1) throw new IllegalStateException("document too short");
            chunks = List.of(document);
        } else {
            // A llama.cpp reference (`llama-perplexity --kl-divergence-base`) scores the second half of each
            // chunk; each chunk here forces the same tokens: prefix N / 2 + 1, then N / 2 - 1 steps.
            ReferenceChunks reference = ReferenceChunks.read(Path.of(tokenFile));
            prefix = reference.context() / 2 + 1;
            steps = reference.context() - prefix;
            chunks = reference.chunks();
        }
        var data = ArtifactReader.read(artifact);
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Qwen38Model model = Qwen38Model.load(
                        artifact,
                        data,
                        gpu,
                        io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile.Speculation.NONE,
                        HostWeightSelection.select(data, hostBytes));
                var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new Execution(lattice, plan, gpu);
            int vocabulary = model.weights().config().vocabSize();
            try {
                for (int chunk = 0; chunk < chunks.size(); chunk++) {
                    Path out = tokenFile.isBlank()
                            ? Path.of(report)
                            : Path.of(String.format(Locale.ROOT, "%s-chunk%03d.bin", report, chunk));
                    double nll = record(
                            runtime,
                            gpu,
                            plan,
                            new Sequence(chunk + 1),
                            chunks.get(chunk),
                            prefix,
                            steps,
                            vocabulary,
                            out);
                    System.out.printf(
                            Locale.ROOT,
                            "quality %s chunk %d/%d: %d forced tokens after %d, mean NLL %.5f nats%n",
                            artifact.getFileName(),
                            chunk + 1,
                            chunks.size(),
                            steps,
                            prefix,
                            nll);
                }
            } finally {
                runtime.close();
            }
        }
    }

    /// Teacher-forces `document` on a fresh sequence: prefills `prefix` tokens, then decodes `steps` forced
    /// tokens, writing each step's logits row to `report`; returns the mean negative log-likelihood.
    private static double record(
            Execution runtime,
            CudaGpuMemory gpu,
            ExecutionPlan plan,
            Sequence sequence,
            int[] document,
            int prefix,
            int steps,
            int vocabulary,
            Path report)
            throws Exception {
        try (OutputStream file = Files.newOutputStream(report);
                DataOutputStream out = new DataOutputStream(new BufferedOutputStream(file, 1 << 22))) {
            out.writeInt(Integer.reverseBytes(steps));
            out.writeInt(Integer.reverseBytes(vocabulary));
            byte[] row = new byte[vocabulary * Short.BYTES];
            double nll = 0;
            // The prefix is prefilled in production-sized chunks: the workspace is sized for one chunk.
            short[] logits = null;
            for (int begin = 0; begin < prefix; begin += InferenceConfig.PREFILL_CHUNK_TOKENS) {
                int end = Math.min(prefix, begin + InferenceConfig.PREFILL_CHUNK_TOKENS);
                logits = run(
                        runtime,
                        gpu,
                        plan,
                        sequence,
                        Quantum.ExecutionKind.PREFILL,
                        Arrays.copyOfRange(document, begin, end));
            }
            for (int step = 0; step < steps; step++) {
                int next = document[prefix + step];
                out.writeInt(Integer.reverseBytes(next));
                for (int i = 0; i < vocabulary; i++) {
                    row[2 * i] = (byte) logits[i];
                    row[2 * i + 1] = (byte) (logits[i] >> 8);
                }
                out.write(row);
                nll += negativeLogLikelihood(logits, next);
                logits = run(runtime, gpu, plan, sequence, Quantum.ExecutionKind.DECODE, new int[] {next});
            }
            return nll / steps;
        } finally {
            sequence.complete();
        }
    }

    /// The chunks of a llama.cpp logits file: "_logits_", int32 context, vocabulary and chunk count, then
    /// context * chunks int32 tokens (little endian).
    private record ReferenceChunks(int context, List<int[]> chunks) {
        static ReferenceChunks read(Path path) throws java.io.IOException {
            try (var channel = java.nio.channels.FileChannel.open(path)) {
                var header = java.nio.ByteBuffer.allocate(20).order(java.nio.ByteOrder.LITTLE_ENDIAN);
                channel.read(header, 0);
                if (!new String(header.array(), 0, 8, java.nio.charset.StandardCharsets.US_ASCII).equals("_logits_"))
                    throw new IllegalArgumentException(path + " is not a llama.cpp logits file");
                int context = header.getInt(8);
                int count = header.getInt(16);
                var tokens = java.nio.ByteBuffer.allocate(4 * context * count).order(java.nio.ByteOrder.LITTLE_ENDIAN);
                channel.read(tokens, 20);
                tokens.flip();
                List<int[]> chunks = new java.util.ArrayList<>();
                for (int chunk = 0; chunk < count; chunk++) {
                    int[] values = new int[context];
                    tokens.asIntBuffer().position(chunk * context).get(values);
                    chunks.add(values);
                }
                return new ReferenceChunks(context, chunks);
            }
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
            Execution runtime,
            CudaGpuMemory gpu,
            ExecutionPlan plan,
            Sequence sequence,
            Quantum.ExecutionKind kind,
            int[] tokens)
            throws Exception {
        AtomicReference<short[]> captured = new AtomicReference<>();
        var context = new Quantum(
                plan, sequence, kind, sequence.currentTokenPosition(), tokens, LogitsRequirement.LAST_TOKEN);
        var outcome = runtime.submit(context, done -> {
                    try (Arena arena = Arena.ofConfined();
                            DeviceLogits logits = done.logitsOutput().orElseThrow()) {
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
        if (outcome.status() != Quantum.Status.SUCCESS) throw new AssertionError(outcome.failure());
        return captured.get();
    }
}
