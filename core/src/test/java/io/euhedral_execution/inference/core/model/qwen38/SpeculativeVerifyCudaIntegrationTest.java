package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// A speculative verification quantum must compute every row bit for bit as one-row decode at that
/// position, and leave the same sequence state: two sequences prefill the same prompt, then one decodes
/// four forced tokens one quantum at a time while the other verifies them in one four-row quantum. Every
/// logits row, every GDN recurrent and convolution state and the committed attention KV bytes must match.
class SpeculativeVerifyCudaIntegrationTest {

    @Test
    @Timeout(value = 1800, unit = TimeUnit.SECONDS)
    void verificationRowsAndCommittedStateEqualSequentialDecode() throws Throwable {
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        Path artifact = Path.of(
                System.getProperty("euhedral.speculative.artifact", System.getProperty("euhedral.qwen.artifact", "")));
        assumeTrue(Files.isRegularFile(artifact), "no artifact: " + artifact);
        Path tokenizerDirectory =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        int prefix = Integer.getInteger("euhedral.speculative.prefix", 300);
        int rows = Integer.getInteger("euhedral.speculative.rows", 4);
        int[] text = QwenTokenizer.load(tokenizerDirectory)
                .encodeText(Files.readString(repositoryRoot().resolve("docs/FRAME_MODEL.md")));
        int[] prompt = java.util.Arrays.copyOf(text, prefix);
        int[] forced = java.util.Arrays.copyOfRange(text, prefix, prefix + rows);
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Qwen38Model model = load(artifact, gpu);
                var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
            var sequential = new Sequence(1);
            var verified = new Sequence(2);
            try {
                prefill(runtime, plan, sequential, prompt, gpu);
                prefill(runtime, plan, verified, prompt, gpu);
                List<short[]> decodeRows = new ArrayList<>();
                for (int token : forced) {
                    decodeRows.addAll(run(
                            runtime,
                            plan,
                            sequential,
                            Quantum.ExecutionKind.DECODE,
                            new int[] {token},
                            LogitsRequirement.ALL_TOKENS,
                            gpu));
                }
                List<short[]> verifyRows = run(
                        runtime,
                        plan,
                        verified,
                        Quantum.ExecutionKind.VERIFY,
                        forced,
                        LogitsRequirement.ALL_TOKENS,
                        gpu);
                assertEquals(rows, verifyRows.size());
                for (int row = 0; row < rows; row++)
                    assertArrayEquals(decodeRows.get(row), verifyRows.get(row), "logits row " + row);
                assertEquals(sequential.currentTokenPosition(), verified.currentTokenPosition());
                assertStatesEqual(gpu, model.weights().config().layerTypes(), sequential, verified);
            } finally {
                sequential.complete();
                verified.complete();
                runtime.close();
            }
        }
    }

    /// Partial acceptance: a verification commits only its first k rows. KV publishes those rows and the
    /// GDN state is replayed from the checkpoint by the next quantum, so after a further partial
    /// verification and a one-row decode, everything equals sequential decode of the committed tokens.
    @Test
    @Timeout(value = 1800, unit = TimeUnit.SECONDS)
    void partiallyCommittedVerificationsLeaveSequentialState() throws Throwable {
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        Path artifact = Path.of(
                System.getProperty("euhedral.speculative.artifact", System.getProperty("euhedral.qwen.artifact", "")));
        assumeTrue(Files.isRegularFile(artifact), "no artifact: " + artifact);
        Path tokenizerDirectory =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        int prefix = Integer.getInteger("euhedral.speculative.prefix", 300);
        int[] text = QwenTokenizer.load(tokenizerDirectory)
                .encodeText(Files.readString(repositoryRoot().resolve("docs/FRAME_MODEL.md")));
        int[] prompt = java.util.Arrays.copyOf(text, prefix);
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Qwen38Model model = load(artifact, gpu);
                var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
            try {
                for (int first = 1; first <= 3; first++) {
                    var sequential = new Sequence(10 + first);
                    var verified = new Sequence(20 + first);
                    try {
                        prefill(runtime, plan, sequential, prompt, gpu);
                        prefill(runtime, plan, verified, prompt, gpu);
                        int cursor = prefix;
                        // Verification 1 commits `first` rows, verification 2 commits 2, then one decode.
                        for (int committed : new int[] {first, 2}) {
                            int[] rows = java.util.Arrays.copyOfRange(text, cursor, cursor + 4);
                            List<short[]> verifiedRows = verify(runtime, plan, verified, rows, committed, gpu);
                            for (int row = 0; row < committed; row++) {
                                List<short[]> decoded = run(
                                        runtime,
                                        plan,
                                        sequential,
                                        Quantum.ExecutionKind.DECODE,
                                        new int[] {rows[row]},
                                        LogitsRequirement.ALL_TOKENS,
                                        gpu);
                                assertArrayEquals(decoded.getFirst(), verifiedRows.get(row), "committed row " + row);
                            }
                            cursor += committed;
                            assertEquals(sequential.currentTokenPosition(), verified.currentTokenPosition());
                        }
                        int[] next = {text[cursor]};
                        assertArrayEquals(
                                run(
                                                runtime,
                                                plan,
                                                sequential,
                                                Quantum.ExecutionKind.DECODE,
                                                next,
                                                LogitsRequirement.ALL_TOKENS,
                                                gpu)
                                        .getFirst(),
                                run(
                                                runtime,
                                                plan,
                                                verified,
                                                Quantum.ExecutionKind.DECODE,
                                                next,
                                                LogitsRequirement.ALL_TOKENS,
                                                gpu)
                                        .getFirst(),
                                "decode after replay, first commit " + first);
                        assertStatesEqual(gpu, model.weights().config().layerTypes(), sequential, verified);
                    } finally {
                        sequential.complete();
                        verified.complete();
                    }
                }
            } finally {
                runtime.close();
            }
        }
    }

    /// A VERIFY quantum that commits only its first `committed` rows; returns all its logits rows.
    static List<short[]> verify(
            EuhedralInferenceRuntime runtime,
            ExecutionPlan plan,
            Sequence sequence,
            int[] rows,
            int committed,
            CudaGpuMemory gpu)
            throws Exception {
        AtomicReference<List<short[]>> captured = new AtomicReference<>(List.of());
        var context = new Quantum(
                        plan,
                        sequence,
                        Quantum.ExecutionKind.VERIFY,
                        sequence.currentTokenPosition(),
                        rows,
                        LogitsRequirement.ALL_TOKENS)
                .withCommittedRows(committed);
        var outcome = runtime.submit(context, done -> {
                    try (Arena arena = Arena.ofConfined();
                            DeviceLogits output = done.logitsOutput().orElseThrow()) {
                        List<short[]> out = new ArrayList<>();
                        int vocabulary = output.vocabularySize();
                        for (int row = 0; row < output.tokenCount(); row++)
                            out.add(shorts(
                                    gpu,
                                    arena,
                                    output.deviceAddress() + (long) row * vocabulary * Short.BYTES,
                                    vocabulary));
                        captured.set(out);
                    }
                })
                .get(600, TimeUnit.SECONDS);
        if (outcome.status() != Quantum.Status.SUCCESS) throw new AssertionError(outcome.failure());
        return captured.get();
    }

    /// Executed weights; `euhedral.speculative.host-mib` host-backs that many MiB of base weights (the
    /// NVFP4 artifact needs it for two sequences).
    static Qwen38Model load(Path artifact, CudaGpuMemory gpu) throws Exception {
        var data = ArtifactReader.read(artifact);
        long hostBytes = Long.getLong("euhedral.speculative.host-mib", 0L) << 20;
        return Qwen38Model.load(
                artifact,
                data,
                gpu,
                io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile.Speculation.NONE,
                io.euhedral_execution.inference.core.model.qwen38.loader.HostWeightSelection.select(data, hostBytes));
    }

    static Path repositoryRoot() {
        Path root = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        while (!Files.isDirectory(root.resolve("docs"))) root = root.getParent();
        return root;
    }

    /// Prefills in the session's 512-row chunks.
    static void prefill(
            EuhedralInferenceRuntime runtime, ExecutionPlan plan, Sequence sequence, int[] prompt, CudaGpuMemory gpu)
            throws Exception {
        for (int offset = 0; offset < prompt.length; offset += 512) {
            run(
                    runtime,
                    plan,
                    sequence,
                    Quantum.ExecutionKind.PREFILL,
                    java.util.Arrays.copyOfRange(prompt, offset, Math.min(prompt.length, offset + 512)),
                    null,
                    gpu);
        }
    }

    /// Runs one quantum; returns its logits rows when `logits` asks for them.
    static List<short[]> run(
            EuhedralInferenceRuntime runtime,
            ExecutionPlan plan,
            Sequence sequence,
            Quantum.ExecutionKind kind,
            int[] tokens,
            LogitsRequirement logits,
            CudaGpuMemory gpu)
            throws Exception {
        AtomicReference<List<short[]>> captured = new AtomicReference<>(List.of());
        var context = new Quantum(
                plan,
                sequence,
                kind,
                sequence.currentTokenPosition(),
                tokens,
                logits == null ? LogitsRequirement.NONE : logits);
        var outcome = runtime.submit(context, done -> {
                    if (logits == null) return;
                    try (Arena arena = Arena.ofConfined();
                            DeviceLogits output = done.logitsOutput().orElseThrow()) {
                        List<short[]> rows = new ArrayList<>();
                        int vocabulary = output.vocabularySize();
                        for (int row = 0; row < output.tokenCount(); row++) {
                            rows.add(shorts(
                                    gpu,
                                    arena,
                                    output.deviceAddress() + (long) row * vocabulary * Short.BYTES,
                                    vocabulary));
                        }
                        captured.set(rows);
                    }
                })
                .get(600, TimeUnit.SECONDS);
        if (outcome.status() != Quantum.Status.SUCCESS) throw new AssertionError(outcome.failure());
        return captured.get();
    }

    /// Every GDN layer's recurrent and convolution state, and every attention layer's committed K and V
    /// rows, bit for bit.
    static void assertStatesEqual(CudaGpuMemory gpu, LayerType[] layerTypes, Sequence expected, Sequence actual) {
        int length = Math.toIntExact(expected.currentTokenPosition());
        var expectedGdn = (GdnStates) expected.recurrentState();
        var actualGdn = (GdnStates) actual.recurrentState();
        var expectedKv = (AttentionStates) expected.kvCacheState();
        var actualKv = (AttentionStates) actual.kvCacheState();
        try (Arena arena = Arena.ofConfined()) {
            for (int layer = 0; layer < layerTypes.length; layer++) {
                if (layerTypes[layer] == LayerType.FULL_ATTENTION) {
                    AttentionKvState a = expectedKv.forLayer(layer), b = actualKv.forLayer(layer);
                    assertEquals(length, a.length(), "expected KV length, layer " + layer);
                    assertEquals(length, b.length(), "KV length, layer " + layer);
                    long rowBytes = a.planePageBytes() / AttentionKvState.PAGE_TOKENS;
                    for (int page = 0; page * AttentionKvState.PAGE_TOKENS < length; page++) {
                        long valid = Math.min(
                                        AttentionKvState.PAGE_TOKENS,
                                        length - (long) page * AttentionKvState.PAGE_TOKENS)
                                * rowBytes;
                        for (int plane = 0; plane < 2; plane++) {
                            long offset = plane * a.planePageBytes();
                            assertArrayEquals(
                                    bytes(gpu, arena, a.pageAddresses().get(page) + offset, valid),
                                    bytes(gpu, arena, b.pageAddresses().get(page) + offset, valid),
                                    "KV " + (plane == 0 ? "K" : "V") + " page " + page + ", layer " + layer);
                        }
                    }
                } else {
                    GdnState a = expectedGdn.forLayer(layer), b = actualGdn.forLayer(layer);
                    long recurrent = 48L * 128 * 128 * Float.BYTES, convolution = 10240L * 3 * Short.BYTES;
                    assertArrayEquals(
                            bytes(gpu, arena, a.recurrentStateAddress(), recurrent),
                            bytes(gpu, arena, b.recurrentStateAddress(), recurrent),
                            "GDN recurrent state, layer " + layer);
                    assertArrayEquals(
                            bytes(gpu, arena, a.convolutionStateAddress(), convolution),
                            bytes(gpu, arena, b.convolutionStateAddress(), convolution),
                            "GDN convolution state, layer " + layer);
                }
            }
        }
    }

    static byte[] bytes(CudaGpuMemory gpu, Arena arena, long device, long count) {
        MemorySegment host = arena.allocate(count, 16);
        gpu.copyDeviceToHost(host, device, count);
        return host.toArray(ValueLayout.JAVA_BYTE);
    }

    static short[] shorts(CudaGpuMemory gpu, Arena arena, long device, int count) {
        MemorySegment host = arena.allocate((long) count * Short.BYTES, Short.BYTES);
        gpu.copyDeviceToHost(host, device, host.byteSize());
        return host.toArray(ValueLayout.JAVA_SHORT);
    }
}
