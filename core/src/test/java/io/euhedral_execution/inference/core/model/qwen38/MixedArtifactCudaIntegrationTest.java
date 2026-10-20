package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.HostWeightSelection;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.settings.Tuning;
import io.euhedral_execution.inference.core.testing.MixedArtifacts;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.core.tokenizer.TokenizerFiles;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// An artifact whose objects come from several files runs bit for bit as the engine runs any artifact.
///
/// The files are a trellis artifact and an NVFP4 one ([MixedArtifacts]); a test skips when either is not set. The
/// first test names every object's source in the trellis artifact (the trellis file itself) and expects the logits of
/// loading the artifact plainly. The others run a composite whose FFN projections (`mlp/gate_up`, `mlp/down`) are the
/// NVFP4 artifact's, with the rest trellis: one-row decode against a verification's rows, two sequences in one step
/// against each alone, and the same weights resident, staged from host memory and read from their own files on
/// disk. MTP speculation and the prefix cache of such an artifact are tested through the engine
/// (`MixedArtifactEngineCudaIntegrationTest`).
@ModelGroup.OwnJvm
class MixedArtifactCudaIntegrationTest {

    private static final int PROMPT = 512;
    private static final int DECODE = 16;
    private static final int ROWS = 8;
    /// Bytes of weights kept off the device (`euhedral.mixed.host-mib`), so a load fits the shared card.
    private static final long OFFLOADED = Long.getLong("euhedral.mixed.host-mib", 4096L) << 20;
    /// Bytes of weights the reference run keeps off the device (`euhedral.mixed.reference-host-mib`; 0 is fully
    /// resident, which a 16 GB card does not hold).
    private static final long REFERENCE_OFFLOADED =
            Long.getLong("euhedral.mixed.reference-host-mib", OFFLOADED >> 20) << 20;
    /// Layers whose staged objects the disk run leaves in their files.
    private static final int DISK_LAYERS = 4;

    private record Run(List<short[]> rows, int[] tokens) {}

    private static Path library() {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        return Path.of(library);
    }

    private static int[] text() throws Exception {
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "unset-euhedral.qwen.tokenizer-dir"));
        assumeTrue(Files.isRegularFile(tokenizer.resolve("tokenizer.json")), "no tokenizer");
        int[] text = QwenTokenizer.load(TokenizerFiles.fromCheckpoint(tokenizer))
                .encodeText(Files.readString(SpeculativeVerifyCudaIntegrationTest.repositoryRoot()
                        .resolve("docs/architecture/FRAME_MODEL.md")));
        assumeTrue(text.length >= PROMPT + 2 * ROWS + 500, "the prompt text is too short");
        return text;
    }

    @Test
    @Timeout(value = 1800, unit = TimeUnit.SECONDS)
    void anArtifactWithEverySourceNamedRunsAsTheArtifactDoes() throws Throwable {
        Path trellis = MixedArtifacts.trellis();
        int[] text = text();
        Artifact plain = ArtifactReader.read(trellis);
        Artifact named = MixedArtifacts.identity(trellis);
        assertEquals(plain.tensors().length, named.sources().size());
        try (CudaGpuMemory gpu = new CudaGpuMemory(library())) {
            Run expected = run(gpu, trellis, plain, HostWeightSelection.select(plain, OFFLOADED), Set.of(), text);
            Run actual = run(gpu, trellis, named, HostWeightSelection.select(named, OFFLOADED), Set.of(), text);
            assertSame(expected, actual);
        }
    }

    @Test
    @Timeout(value = 1800, unit = TimeUnit.SECONDS)
    void aMixedArtifactVerifiesRowsAsDecodesAndBatchesSequencesAsEachAlone() throws Throwable {
        Path trellis = MixedArtifacts.trellis();
        Path nvfp4 = MixedArtifacts.nvfp4();
        int[] text = text();
        Artifact mixed = MixedArtifacts.mixed(trellis, nvfp4, 4);
        assertEquals(
                ArtifactProfile.Quantization.MIXED, ArtifactProfile.of(mixed).quantization());
        assertEquals(4, ArtifactProfile.of(mixed).draftLength().most());
        try (CudaGpuMemory gpu = new CudaGpuMemory(library());
                Qwen38Model model = Qwen38Model.load(
                        trellis,
                        mixed,
                        gpu,
                        ArtifactProfile.Speculation.NONE,
                        HostWeightSelection.select(mixed, OFFLOADED));
                var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging(), model.window());
            var runtime = execution(lattice, plan, gpu, model, mixed);
            List<Sequence> sequences = new ArrayList<>();
            try {
                // (a) The rows of a verification are the rows of one-row decodes, bit for bit.
                Sequence decoded = new Sequence(1);
                Sequence verified = new Sequence(2);
                sequences.addAll(List.of(decoded, verified));
                int[] prompt = Arrays.copyOf(text, PROMPT);
                int[] forced = Arrays.copyOfRange(text, PROMPT, PROMPT + ROWS);
                SpeculativeVerifyCudaIntegrationTest.prefill(runtime, plan, decoded, prompt, gpu);
                SpeculativeVerifyCudaIntegrationTest.prefill(runtime, plan, verified, prompt, gpu);
                List<short[]> single = new ArrayList<>();
                for (int token : forced)
                    single.addAll(SpeculativeVerifyCudaIntegrationTest.run(
                            runtime,
                            plan,
                            decoded,
                            Quantum.ExecutionKind.DECODE,
                            new int[] {token},
                            LogitsRequirement.LAST_TOKEN,
                            gpu));
                List<short[]> rows = SpeculativeVerifyCudaIntegrationTest.run(
                        runtime,
                        plan,
                        verified,
                        Quantum.ExecutionKind.VERIFY,
                        forced,
                        LogitsRequirement.ALL_TOKENS,
                        gpu);
                assertEquals(ROWS, rows.size());
                for (int row = 0; row < ROWS; row++)
                    assertArrayEquals(single.get(row), rows.get(row), "verify row " + row + " against decode");

                // (b) Two sequences in one step equal each of them alone.
                int[] lengths = {300, 411};
                List<Sequence> together = new ArrayList<>();
                List<Quantum.Part> parts = new ArrayList<>();
                List<List<short[]>> expected = new ArrayList<>();
                for (int s = 0; s < lengths.length; s++) {
                    int[] document = Arrays.copyOfRange(text, 97 * s, 97 * s + lengths[s] + ROWS);
                    int[] start = Arrays.copyOf(document, lengths[s]);
                    int[] rowsToVerify = Arrays.copyOfRange(document, lengths[s], lengths[s] + ROWS);
                    Sequence batched = new Sequence(10 + s);
                    Sequence alone = new Sequence(20 + s);
                    sequences.addAll(List.of(batched, alone));
                    SpeculativeVerifyCudaIntegrationTest.prefill(runtime, plan, batched, start, gpu);
                    SpeculativeVerifyCudaIntegrationTest.prefill(runtime, plan, alone, start, gpu);
                    expected.add(SpeculativeVerifyCudaIntegrationTest.run(
                            runtime,
                            plan,
                            alone,
                            Quantum.ExecutionKind.VERIFY,
                            rowsToVerify,
                            LogitsRequirement.ALL_TOKENS,
                            gpu));
                    parts.add(new Quantum.Part(batched, batched.currentTokenPosition(), rowsToVerify));
                    together.add(batched);
                }
                List<short[]> batchRows = batch(runtime, plan, parts, gpu);
                int next = 0;
                for (int s = 0; s < lengths.length; s++)
                    for (int row = 0; row < ROWS; row++)
                        assertArrayEquals(
                                expected.get(s).get(row), batchRows.get(next++), "sequence " + s + " row " + row);
            } finally {
                for (Sequence sequence : sequences) sequence.complete();
                runtime.close();
            }
        }
    }

    @Test
    @Timeout(value = 2400, unit = TimeUnit.SECONDS)
    void aMixedArtifactComputesTheSameResidentStagedAndReadFromItsFiles() throws Throwable {
        Path trellis = MixedArtifacts.trellis();
        Path nvfp4 = MixedArtifacts.nvfp4();
        int[] text = text();
        Artifact mixed = MixedArtifacts.mixed(trellis, nvfp4, 4);
        // The first layers' staged objects stay in their files; the rest of the larger selection goes to host memory.
        Set<String> disk = new LinkedHashSet<>();
        Pattern layer = Pattern.compile("text/layers/(\\d+)/.*");
        for (TensorDescriptor tensor : mixed.tensors()) {
            Matcher matcher = layer.matcher(tensor.name());
            if (matcher.matches()
                    && Integer.parseInt(matcher.group(1)) < DISK_LAYERS
                    && HostWeightSelection.stageable(tensor)) disk.add(tensor.name());
        }
        assertTrue(
                disk.stream().anyMatch(mixed.sources()::containsKey),
                "no disk-backed object comes from the NVFP4 file");
        assertTrue(
                disk.stream().anyMatch(name -> !mixed.sources().containsKey(name)),
                "no disk-backed object comes from the trellis file");
        Set<String> host = new LinkedHashSet<>(HostWeightSelection.select(mixed, REFERENCE_OFFLOADED + OFFLOADED));
        host.removeAll(disk);
        assertFalse(host.isEmpty());
        try (CudaGpuMemory gpu = new CudaGpuMemory(library())) {
            Run staged =
                    run(gpu, trellis, mixed, HostWeightSelection.select(mixed, REFERENCE_OFFLOADED), Set.of(), text);
            Run read = run(gpu, trellis, mixed, host, disk, text);
            assertSame(staged, read);
        }
    }

    private static void assertSame(Run expected, Run actual) {
        assertArrayEquals(expected.tokens(), actual.tokens(), "the greedy tokens");
        assertEquals(expected.rows().size(), actual.rows().size());
        for (int row = 0; row < expected.rows().size(); row++)
            assertArrayEquals(expected.rows().get(row), actual.rows().get(row), "logits row " + row);
    }

    private static Execution execution(
            PullingLattice lattice, ExecutionPlan plan, CudaGpuMemory gpu, Qwen38Model model, Artifact artifact) {
        return new Execution(
                lattice,
                plan,
                gpu,
                EuhedralInferenceRuntime.laneCount(),
                Tuning.defaults(),
                model.artifactPath(),
                artifact.sources());
    }

    /// Loads with `host` in host memory and `disk` in the files, prefills, decodes greedily and verifies 8 forced
    /// rows; returns every logits row and the greedy tokens.
    private static Run run(
            CudaGpuMemory gpu, Path path, Artifact artifact, Set<String> host, Set<String> disk, int[] text)
            throws Exception {
        try (Qwen38Model model = Qwen38Model.load(path, artifact, gpu, ArtifactProfile.Speculation.NONE, host, disk);
                var lattice = new PullingLattice()) {
            assertEquals(!disk.isEmpty(), model.window() != null);
            var plan = new ExecutionPlan(model.weights(), model.staging(), model.window());
            var runtime = execution(lattice, plan, gpu, model, artifact);
            Sequence sequence = new Sequence(1);
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
                        Arrays.copyOfRange(text, PROMPT, PROMPT + ROWS),
                        LogitsRequirement.ALL_TOKENS,
                        gpu));
                return new Run(rows, tokens);
            } finally {
                sequence.complete();
                runtime.close();
            }
        }
    }

    /// Runs one batched verification; returns its logits rows in part order.
    private static List<short[]> batch(
            Execution runtime, ExecutionPlan plan, List<Quantum.Part> parts, CudaGpuMemory gpu) throws Exception {
        AtomicReference<List<short[]>> captured = new AtomicReference<>(List.of());
        var context = Quantum.batch(plan, Quantum.ExecutionKind.VERIFY, parts, LogitsRequirement.ALL_TOKENS);
        var outcome = runtime.submit(context, done -> {
                    try (Arena arena = Arena.ofConfined();
                            DeviceLogits output = done.logitsOutput().orElseThrow()) {
                        List<short[]> rows = new ArrayList<>();
                        int vocabulary = output.vocabularySize();
                        for (int row = 0; row < output.tokenCount(); row++)
                            rows.add(SpeculativeVerifyCudaIntegrationTest.shorts(
                                    gpu,
                                    arena,
                                    output.deviceAddress() + (long) row * vocabulary * Short.BYTES,
                                    vocabulary));
                        captured.set(rows);
                    }
                })
                .get(600, TimeUnit.SECONDS);
        if (outcome.status() != Quantum.Status.SUCCESS) throw new AssertionError(outcome.failure());
        return captured.get();
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
