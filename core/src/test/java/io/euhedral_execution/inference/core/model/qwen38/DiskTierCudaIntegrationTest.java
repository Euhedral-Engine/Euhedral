package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.HostWeightSelection;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.settings.Tuning;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Weights read from the artifact on use compute exactly what the same weights staged from host memory compute: the
/// Q3 artifact runs twice with the same offloaded selection, once all in host memory and once with every staged
/// projection of it left on disk. A 512-token prompt, 64 greedy decode tokens (captured graphs replay from the third
/// on) and a verification of 8 forced rows give every logits row bit for bit, and the same tokens. Closing with a
/// quantum in flight frees every device and pinned byte the load took.
@ModelGroup.OwnJvm
class DiskTierCudaIntegrationTest {

    private static final int PROMPT = 512;
    private static final int DECODE = 64;
    /// Bytes of weights taken off the device (`euhedral.disk.offload-mib`), so two loads fit the shared card.
    private static final long OFFLOADED = Long.getLong("euhedral.disk.offload-mib", 4096L) << 20;

    private record Run(List<short[]> rows, int[] tokens) {}

    private static Path library() {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        return Path.of(library);
    }

    private static Path artifact() {
        Path artifact = Path.of(System.getProperty("euhedral.qwen.artifact", "unset-euhedral.qwen.artifact"));
        assumeTrue(Files.isRegularFile(artifact), "no artifact " + artifact);
        return artifact;
    }

    private static int[] prompt() throws Exception {
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "unset-euhedral.qwen.tokenizer-dir"));
        int[] text = QwenTokenizer.load(tokenizer)
                .encodeText(Files.readString(
                        SpeculativeVerifyCudaIntegrationTest.repositoryRoot().resolve("docs/FRAME_MODEL.md")));
        assumeTrue(text.length >= PROMPT + 8, "the prompt text is too short");
        return text;
    }

    @Test
    @Timeout(value = 1800, unit = TimeUnit.SECONDS)
    void diskBackedWeightsComputeWhatHostBackedOnesDo() throws Throwable {
        Path artifact = artifact();
        int[] text = prompt();
        Artifact data = ArtifactReader.read(artifact);
        Set<String> offloaded = HostWeightSelection.select(data, OFFLOADED);
        Set<String> disk = new LinkedHashSet<>();
        for (TensorDescriptor tensor : data.tensors())
            if (offloaded.contains(tensor.name()) && HostWeightSelection.stageable(tensor)) disk.add(tensor.name());
        Set<String> host = new LinkedHashSet<>(offloaded);
        host.removeAll(disk);
        assertFalse(disk.isEmpty());
        try (CudaGpuMemory gpu = new CudaGpuMemory(library())) {
            Run staged = run(gpu, artifact, data, offloaded, Set.of(), text, false);
            Run read = run(gpu, artifact, data, host, disk, text, false);
            assertArrayEquals(staged.tokens(), read.tokens(), "the greedy tokens");
            assertEquals(staged.rows().size(), read.rows().size());
            for (int row = 0; row < staged.rows().size(); row++)
                assertArrayEquals(staged.rows().get(row), read.rows().get(row), "logits row " + row);
        }
    }

    @Test
    @Timeout(value = 900, unit = TimeUnit.SECONDS)
    void closingWithReadsInFlightFreesAfterThemOnly() throws Throwable {
        Path artifact = artifact();
        int[] text = prompt();
        Artifact data = ArtifactReader.read(artifact);
        Set<String> offloaded = HostWeightSelection.select(data, OFFLOADED);
        Set<String> disk = new LinkedHashSet<>();
        for (TensorDescriptor tensor : data.tensors())
            if (offloaded.contains(tensor.name()) && HostWeightSelection.stageable(tensor)) disk.add(tensor.name());
        Set<String> host = new LinkedHashSet<>(offloaded);
        host.removeAll(disk);
        try (CudaGpuMemory gpu = new CudaGpuMemory(library())) {
            long device = gpu.allocatedBytes();
            long pinned = gpu.hostWeightBytes();
            run(gpu, artifact, data, host, disk, text, true);
            assertEquals(device, gpu.allocatedBytes(), "device bytes after close");
            assertEquals(pinned, gpu.hostWeightBytes(), "pinned bytes after close");
        }
    }

    /// Loads with `host` in host memory and `disk` in the file, prefills, decodes greedily and verifies; with
    /// `abandon`, submits one more decode and closes while it runs instead.
    private static Run run(
            CudaGpuMemory gpu,
            Path artifact,
            Artifact data,
            Set<String> host,
            Set<String> disk,
            int[] text,
            boolean abandon)
            throws Exception {
        try (Qwen38Model model = Qwen38Model.load(artifact, data, gpu, ArtifactProfile.Speculation.NONE, host, disk);
                var lattice = new PullingLattice()) {
            assertEquals(!disk.isEmpty(), model.window() != null);
            var plan = new ExecutionPlan(model.weights(), model.staging(), model.window());
            var runtime = new Execution(
                    lattice,
                    plan,
                    gpu,
                    io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime.laneCount(),
                    Tuning.defaults(),
                    model.artifactPath());
            Sequence sequence = new Sequence(disk.isEmpty() ? 1 : 2);
            try {
                List<short[]> rows = new ArrayList<>(SpeculativeVerifyCudaIntegrationTest.run(
                        runtime,
                        plan,
                        sequence,
                        Quantum.ExecutionKind.PREFILL,
                        Arrays.copyOf(text, PROMPT),
                        LogitsRequirement.LAST_TOKEN,
                        gpu));
                int[] tokens = new int[DECODE];
                for (int step = 0; step < DECODE; step++) {
                    tokens[step] = argmax(rows.getLast());
                    if (abandon && step == 2) {
                        // A decode in flight while the execution closes: its reads complete before the window goes.
                        var pending = new Quantum(
                                plan,
                                sequence,
                                Quantum.ExecutionKind.DECODE,
                                sequence.currentTokenPosition(),
                                new int[] {tokens[step]},
                                LogitsRequirement.NONE);
                        CompletableFuture<Quantum.Outcome> outcome = runtime.submit(pending);
                        runtime.close();
                        assertTrue(outcome.isDone());
                        return new Run(rows, tokens);
                    }
                    rows.addAll(SpeculativeVerifyCudaIntegrationTest.run(
                            runtime,
                            plan,
                            sequence,
                            Quantum.ExecutionKind.DECODE,
                            new int[] {tokens[step]},
                            LogitsRequirement.LAST_TOKEN,
                            gpu));
                }
                rows.addAll(SpeculativeVerifyCudaIntegrationTest.run(
                        runtime,
                        plan,
                        sequence,
                        Quantum.ExecutionKind.VERIFY,
                        Arrays.copyOfRange(text, PROMPT, PROMPT + 8),
                        LogitsRequirement.ALL_TOKENS,
                        gpu));
                return new Run(rows, tokens);
            } finally {
                sequence.complete();
                runtime.close();
            }
        }
    }

    private static int argmax(short[] row) {
        int best = 0;
        for (int index = 1; index < row.length; index++) if (bf16(row[index]) > bf16(row[best])) best = index;
        return best;
    }

    private static float bf16(short bits) {
        return Float.intBitsToFloat((bits & 0xffff) << 16);
    }
}
