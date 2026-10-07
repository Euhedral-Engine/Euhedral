package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Model;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.model.qwen38.TestExecution;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The P2E2 artifact (tools/euhedral_artifacts/q3_p2e2.py) holds the compact artifact's Q3 values in
/// less memory. Run through every route that reads them, it must reproduce the compact artifact's
/// logits bit for bit: prefill regions and an all-token LM head (expanded, the head in output-row
/// chunks), short prompts on the small-row kernels, single-row decode on the P2E2 kernel, and exact
/// numerics. Skipped unless -Peuhedral.qwen.q3-compressed-artifact names an existing artifact.
class P2e2CudaIntegrationTest {

    private static final int PROMPT = 80;
    private static final int SHORT_PROMPT = 5;
    private static final int DECODE_STEPS = 6;

    @Test
    @Timeout(value = 1800, unit = TimeUnit.SECONDS)
    void p2e2ArtifactReproducesTheCompactArtifactBitwise() throws Exception {
        Path compact = Path.of(System.getProperty("euhedral.qwen.artifact"));
        Path p2e2 = Path.of(System.getProperty("euhedral.qwen.q3-compressed-artifact", ""));
        assumeTrue(Files.isRegularFile(p2e2), "no P2E2 artifact: " + p2e2);
        Trace expected = trace(compact, false);
        Trace actual = trace(p2e2, true);
        assertEquals(expected.logits().size(), actual.logits().size());
        for (int i = 0; i < expected.logits().size(); i++) {
            assertArrayEquals(expected.logits().get(i), actual.logits().get(i), "logits of step " + i);
        }
        assertTrue(actual.weightBytes() < expected.weightBytes() - (3L << 29), "P2E2 must save at least 1.5 GiB");
        System.out.println("P2E2_BITWISE PASS steps=" + expected.logits().size()
                + " compact_weight_bytes=" + expected.weightBytes()
                + " p2e2_weight_bytes=" + actual.weightBytes()
                + " p2e2_resident_after_run_bytes=" + actual.residentBytes());
    }

    private record Trace(List<short[]> logits, long weightBytes, long residentBytes) {}

    private static Trace trace(Path artifact, boolean expectP2e2) throws Exception {
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Qwen38Model model = Qwen38Model.load(artifact, ArtifactReader.read(artifact), gpu)) {
            long weightBytes = gpu.allocatedBytes();
            WeightLayout q3 = expectP2e2 ? WeightLayout.ROW_SPLIT_P2E2_V1 : WeightLayout.ROW_SPLIT_K128_V1;
            assertEquals(q3, model.weights().tokenEmbedding().layout());
            assertEquals(q3, model.weights().lmHead().layout());
            var plan = new ExecutionPlan(model.weights());
            List<short[]> logits = new ArrayList<>();
            int[] prompt = new int[PROMPT];
            for (int i = 0; i < PROMPT; i++) prompt[i] = 1814 + (i * 7919) % 90001;
            var sequence = new Sequence(71);
            try {
                logits.add(run(
                        gpu, plan, sequence, Quantum.ExecutionKind.PREFILL, 0, prompt, LogitsRequirement.ALL_TOKENS));
                int token = 1814;
                for (int step = 0; step < DECODE_STEPS; step++) {
                    short[] row = run(
                            gpu,
                            plan,
                            sequence,
                            Quantum.ExecutionKind.DECODE,
                            PROMPT + step,
                            new int[] {token},
                            LogitsRequirement.LAST_TOKEN);
                    logits.add(row);
                    token = argmax(row);
                }
                boolean previous = gpu.selectExactNumerics(true);
                try {
                    logits.add(run(
                            gpu,
                            plan,
                            sequence,
                            Quantum.ExecutionKind.DECODE,
                            PROMPT + DECODE_STEPS,
                            new int[] {token},
                            LogitsRequirement.LAST_TOKEN));
                } finally {
                    gpu.selectExactNumerics(previous);
                }
            } finally {
                sequence.complete();
            }
            var shortSequence = new Sequence(72);
            try {
                logits.add(run(
                        gpu,
                        plan,
                        shortSequence,
                        Quantum.ExecutionKind.PREFILL,
                        0,
                        java.util.Arrays.copyOf(prompt, SHORT_PROMPT),
                        LogitsRequirement.ALL_TOKENS));
            } finally {
                shortSequence.complete();
            }
            return new Trace(logits, weightBytes, gpu.allocatedBytes());
        }
    }

    private static short[] run(
            CudaGpuMemory gpu,
            ExecutionPlan plan,
            Sequence sequence,
            Quantum.ExecutionKind kind,
            long start,
            int[] tokens,
            LogitsRequirement requirement)
            throws Exception {
        AtomicReference<short[]> logits = new AtomicReference<>();
        var context = new Quantum(plan, sequence, kind, start, tokens, requirement);
        var outcome = TestExecution.run(
                plan,
                gpu,
                context,
                completed -> completed.logitsOutput().ifPresent(device -> {
                    try (Arena arena = Arena.ofConfined()) {
                        int count = Math.multiplyExact(device.tokenCount(), device.vocabularySize());
                        var host = arena.allocate((long) count * Short.BYTES, Short.BYTES);
                        gpu.copyDeviceToHost(host, device.deviceAddress(), host.byteSize());
                        logits.set(host.toArray(ValueLayout.JAVA_SHORT));
                    } finally {
                        device.close();
                    }
                }),
                900);
        if (outcome.status() != Quantum.Status.SUCCESS) {
            throw new AssertionError("Euhedral graph failed: " + outcome.failure());
        }
        return logits.get();
    }

    private static int argmax(short[] row) {
        int best = 0;
        for (int i = 1; i < row.length; i++) if (bf16(row[i]) > bf16(row[best])) best = i;
        return best;
    }

    private static float bf16(short value) {
        return Float.intBitsToFloat((value & 0xffff) << 16);
    }
}
