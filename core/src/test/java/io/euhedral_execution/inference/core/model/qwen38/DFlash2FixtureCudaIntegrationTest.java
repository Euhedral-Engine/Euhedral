package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Config;
import io.euhedral_execution.inference.core.model.qwen38.speculative.DFlash2Proposal;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The drafter's intermediates against the upstream reference (tools/dflash2_reference.py).
///
/// The test prefills a prompt that taps its rows, runs the context and one draft block, and writes every
/// intermediate under the reference's names to `-Peuhedral.dflash2.dump` (default build/dflash2-fixture), with the
/// taps and the anchor as the reference's inputs and the embedded block rows. Given reference fixtures computed from
/// that dump (`-Peuhedral.dflash2.fixtures`, `dflash2_reference.py --inputs DUMP --embedding
/// DUMP/noise_embedding.bin --lm-head-artifact ARTIFACT`), every tensor must stay within the BF16 reference's own
/// distance from the FP32 reference, and the proposal must be the reference's.
@ModelGroup.OwnJvm // loads the DFlash2 artifact, which no shared group holds
class DFlash2FixtureCudaIntegrationTest {

    /// Relative RMS bounds per tensor family: the BF16 reference measured 2-9e-3 per layer-0 tensor, 6-9e-2 at the
    /// last layer and 4.7e-2 on the logits against the FP32 reference (synthetic taps); the engine's BF16 drafter
    /// differs from the BF16 reference by summation order alone, so it must stay inside that spread.
    private static final double LAYER_BOUND = 0.1;

    @Test
    @Timeout(value = 1800, unit = TimeUnit.SECONDS)
    void drafterIntermediatesFollowTheReference() throws Throwable {
        Path artifact = Path.of(System.getProperty("euhedral.qwen.dflash2-artifact", ""));
        assumeTrue(Files.isRegularFile(artifact), "no DFlash2 artifact: " + artifact);
        Path dump = Path.of(System.getProperty("euhedral.dflash2.dump", "build/dflash2-fixture"))
                .toAbsolutePath();
        String reference = System.getProperty("euhedral.dflash2.fixtures", "");
        QwenTokenizer tokenizer = QwenTokenizer.load(
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen")));
        // -Peuhedral.dflash2.prompt-tokens takes that many tokens of docs/FRAME_MODEL.md instead of a chat prompt: past
        // 2048 the drafter's window and its ring's wraparound are in play.
        int promptTokens = Integer.getInteger("euhedral.dflash2.prompt-tokens", 0);
        int[] prompt = promptTokens > 0
                ? Arrays.copyOf(
                        tokenizer.encodeText(Files.readString(SpeculativeVerifyCudaIntegrationTest.repositoryRoot()
                                .resolve("docs/FRAME_MODEL.md"))),
                        promptTokens)
                : tokenizer.encodeWithModelSpecialTokens("<|im_start|>user\nWrite a Java method that reverses a"
                        + " singly linked list, with a short explanation.<|im_end|>\n<|im_start|>assistant\n"
                        + "<think>\n\n</think>\n\n");
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Qwen38Model model = DFlash2SpeculativeDecodeCudaIntegrationTest.load(artifact, gpu, 4096);
                var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new Execution(lattice, plan, gpu);
            DFlash2Config config = model.weights().dflash2().config();
            var sequence = new Sequence(900);
            var recorder = new Recorder(gpu, config, prompt.length);
            try (var logits = new HostLogits(gpu, model.weights().config().vocabSize());
                    var proposal = new DFlash2Proposal(gpu, config.blockSize() - 1, config.selectorTopK())) {
                logits.selectOnDevice(true);
                // Prefill in the engine's 512-row chunks, each followed by its context quantum, as DFlash2Decoder
                // runs a prompt; the taps of every chunk are kept for the reference.
                byte[] taps = new byte[0];
                for (int offset = 0; offset < prompt.length; offset += 512) {
                    int end = Math.min(prompt.length, offset + 512);
                    boolean last = end == prompt.length;
                    int[] chunk = Arrays.copyOfRange(prompt, offset, end);
                    SpeculativeDecodeCudaIntegrationTest.execute(
                            runtime,
                            new Quantum(
                                            plan,
                                            sequence,
                                            Quantum.ExecutionKind.PREFILL,
                                            offset,
                                            chunk,
                                            last ? LogitsRequirement.LAST_TOKEN : LogitsRequirement.NONE,
                                            last ? logits : null)
                                    .seedingDraft());
                    var drafter = ((AttentionStates) sequence.kvCacheState()).dflash2(config);
                    recorder.copy("taps", drafter.taps(chunk.length), chunk.length, config.tapWidth(), "bf16", 0, 0);
                    taps = Recorder.concat(taps, recorder.data.remove("taps"));
                    // Only the last chunk's context stages are recorded (the reference's names cover one call).
                    if (last) Stages.DFlash2.observe(recorder);
                    SpeculativeDecodeCudaIntegrationTest.execute(
                            runtime,
                            new Quantum(
                                    plan,
                                    sequence,
                                    Quantum.ExecutionKind.DRAFT_CONTEXT,
                                    offset,
                                    chunk,
                                    LogitsRequirement.NONE));
                }
                recorder.put("taps", taps, prompt.length, config.tapWidth(), "bf16");
                int anchor = logits.selectedToken();
                try {
                    int[] block = new int[config.blockSize()];
                    Arrays.fill(block, config.maskToken());
                    block[0] = anchor;
                    SpeculativeDecodeCudaIntegrationTest.execute(
                            runtime,
                            new Quantum(
                                            plan,
                                            sequence,
                                            Quantum.ExecutionKind.DRAFT,
                                            prompt.length,
                                            block,
                                            LogitsRequirement.ALL_TOKENS)
                                    .withProposal(proposal));
                } finally {
                    Stages.DFlash2.observe(null);
                }
                recorder.write(dump, prompt.length, anchor);
                System.out.println("DFLASH2_FIXTURE dump " + dump + " anchor " + anchor + " proposal "
                        + Arrays.toString(proposal.tokens()));
                if (!reference.isEmpty()) compare(Path.of(reference), recorder, proposal.tokens());
            } finally {
                sequence.complete();
                runtime.close();
            }
        }
    }

    /// Records the drafter's stages under the reference's names.
    private static final class Recorder implements Stages.DFlash2.Observer {
        private final CudaGpuMemory gpu;
        private final DFlash2Config config;
        private final int contextRows;
        final Map<String, byte[]> data = new LinkedHashMap<>();
        final Map<String, long[]> shapes = new HashMap<>();
        final Map<String, String> types = new HashMap<>();
        private final Map<String, Integer> seen = new HashMap<>();

        Recorder(CudaGpuMemory gpu, DFlash2Config config, int contextRows) {
            this.gpu = gpu;
            this.config = config;
            this.contextRows = contextRows;
        }

        /// `rows` rows of `width` values from `rowOffset`, columns `[column, column + width)` of rows `stride` wide.
        void copy(String name, long address, int rows, int width, String type, int rowOffset, int column) {
            copy(name, address, rows, width, width, type, rowOffset, column);
        }

        void copy(String name, long address, int rows, int width, int stride, String type, int rowOffset, int column) {
            int element = type.equals("bf16") ? 2 : 4;
            byte[] all = new byte[rows * width * element];
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment host = arena.allocate((long) (rowOffset + rows) * stride * element);
                this.gpu.copyDeviceToHost(host, address, host.byteSize());
                for (int row = 0; row < rows; row++)
                    MemorySegment.copy(
                            host,
                            ValueLayout.JAVA_BYTE,
                            ((long) (rowOffset + row) * stride + column) * element,
                            all,
                            row * width * element,
                            width * element);
            }
            this.data.put(name, all);
            this.shapes.put(name, new long[] {rows, width});
            this.types.put(name, type);
        }

        private int occurrence(String key) {
            return this.seen.merge(key, 1, Integer::sum);
        }

        @Override
        public void submitted(Quantum context, ExecutionPlan.Instruction instruction) {
            this.gpu.synchronize();
            Workspace workspace = context.workspace();
            int rows = context.inputTokenCount();
            int layer = instruction.layerIndex();
            String prefix = "layer" + layer + "/";
            boolean block = context.kind() == Quantum.ExecutionKind.DRAFT;
            long out = instruction.outputBuffers().isEmpty()
                    ? 0
                    : workspace.address(instruction.outputBuffers().getFirst());
            int width = instruction.outputWidth();
            int hidden = this.config.hiddenSize();
            int kv = this.config.keyValueWidth();
            switch (instruction.kind()) {
                case DFLASH_LINEAR -> {
                    ExecutionPlan.Buffer buffer = instruction.outputBuffers().getFirst();
                    switch (buffer) {
                        case DRAFT_FUSED -> {
                            if (rows == this.contextRows) copy("fc", out, rows, width, "bf16", 0, 0);
                        }
                        case DRAFT_KV -> {
                            String part = block ? "block" : "context";
                            if (!block && rows != this.contextRows) return;
                            copy(prefix + "k_" + part, out, rows, kv, 2 * kv, "bf16", 0, 0);
                            copy(prefix + "v_" + part, out, rows, kv, 2 * kv, "bf16", 0, kv);
                        }
                        case DRAFT_DYNAMIC ->
                            copy(
                                    prefix + (occurrence(prefix + "dynamic") == 1 ? "attention_conv" : "mlp_conv")
                                            + "/dynamic",
                                    out,
                                    rows,
                                    width,
                                    "bf16",
                                    0,
                                    0);
                        case DRAFT_QUERY -> copy(prefix + "q", out, rows, width, "bf16", 0, 0);
                        case MIXER_DELTA ->
                            copy(
                                    prefix + (occurrence(prefix + "delta") == 1 ? "o" : "mlp"),
                                    out,
                                    rows,
                                    width,
                                    "bf16",
                                    0,
                                    0);
                        case DRAFT_SELECTOR -> copy("selector_hidden", out, rows - 1, width, "bf16", 1, 0);
                        default -> {}
                    }
                }
                case DFLASH_RMS_NORM -> {
                    long in = workspace.address(instruction.inputBuffers().getFirst());
                    if (!block) {
                        if (rows == this.contextRows) copy("context", out, rows, width, "bf16", 0, 0);
                    } else if (layer < 0) {
                        copy("layer" + (this.config.layers() - 1) + "/out", in, rows, width, "bf16", 0, 0);
                        copy("final_hidden", out, rows, width, "bf16", 0, 0);
                    } else if (occurrence(prefix + "norm") == 1) {
                        copy(
                                layer == 0 ? "noise_embedding" : "layer" + (layer - 1) + "/out",
                                in,
                                rows,
                                width,
                                "bf16",
                                0,
                                0);
                        copy(prefix + "input_norm", out, rows, width, "bf16", 0, 0);
                    } else copy(prefix + "post_attention_norm", out, rows, width, "bf16", 0, 0);
                }
                case DFLASH_CONV -> {
                    boolean prepare = instruction.outputBufferIndex() == 0;
                    String conv =
                            occurrence(prefix + (prepare ? "prepare" : "finish")) == 1 ? "attention_conv" : "mlp_conv";
                    copy(prefix + conv + (prepare ? "/prepared" : "/finished"), out, rows, hidden, "bf16", 0, 0);
                }
                case DFLASH_ATTENTION -> {
                    copy(prefix + "attention", out, rows, width, "bf16", 0, 0);
                    var state = ((AttentionStates) context.sequenceState().kvCacheState()).dflash2();
                    // The ring holds the last window of positions only; earlier ones the reference keeps are gone.
                    if (this.contextRows > this.config.slidingWindow()) return;
                    long keys = workspace.address(ExecutionPlan.Buffer.DRAFT_KEY_ROPE);
                    long values = workspace.address(ExecutionPlan.Buffer.DRAFT_KV);
                    copy("ring", state.ringKeys(layer), this.contextRows, kv, "bf16", 0, 0);
                    byte[] ringKeys = this.data.remove("ring");
                    copy("ring", state.ringValues(layer), this.contextRows, kv, "bf16", 0, 0);
                    byte[] ringValues = this.data.remove("ring");
                    copy("blockKeys", keys, rows, kv, "bf16", 0, 0);
                    byte[] blockKeys = this.data.remove("blockKeys");
                    copy("blockValues", values, rows, kv, 2 * kv, "bf16", 0, kv);
                    byte[] blockValues = this.data.remove("blockValues");
                    put(prefix + "cache_keys", concat(ringKeys, blockKeys), this.contextRows + rows, kv, "bf16");
                    put(prefix + "cache_values", concat(ringValues, blockValues), this.contextRows + rows, kv, "bf16");
                }
                case DFLASH_LM_HEAD -> copy("logits", out, rows - 1, width, "bf16", 0, 0);
                case DFLASH_TOPK -> {
                    copy("topk_values", out, rows - 1, width, "bf16", 0, 0);
                    copy(
                            "topk_indices",
                            workspace.address(instruction.outputBuffers().get(1)),
                            rows - 1,
                            width,
                            "i32",
                            0,
                            0);
                }
                case DFLASH_SELECT -> {
                    copy("proposal", out, rows - 1, 1, "i32", 0, 0);
                    this.shapes.put("proposal", new long[] {rows - 1});
                    copy(
                            "selector_scores",
                            workspace.address(instruction.outputBuffers().get(1)),
                            rows - 1,
                            this.config.selectorTopK(),
                            "f32",
                            0,
                            0);
                }
                default -> {}
            }
        }

        void put(String name, byte[] bytes, int rows, int width, String type) {
            this.data.put(name, bytes);
            this.shapes.put(name, new long[] {rows, width});
            this.types.put(name, type);
        }

        static byte[] concat(byte[] a, byte[] b) {
            byte[] all = Arrays.copyOf(a, a.length + b.length);
            System.arraycopy(b, 0, all, a.length, b.length);
            return all;
        }

        void write(Path directory, int start, int anchor) throws Exception {
            Files.createDirectories(directory);
            ObjectMapper json = new ObjectMapper();
            ObjectNode manifest = json.createObjectNode();
            manifest.putObject("metadata")
                    .put("start", start)
                    .put("anchor", anchor)
                    .put("source", "engine");
            ObjectNode tensors = manifest.putObject("tensors");
            for (var entry : this.data.entrySet()) {
                String file = entry.getKey().replace('/', '.') + ".bin";
                Files.write(directory.resolve(file), entry.getValue());
                ObjectNode tensor = tensors.putObject(entry.getKey());
                tensor.put("dtype", this.types.get(entry.getKey())).put("file", file);
                var shape = tensor.putArray("shape");
                for (long extent : this.shapes.get(entry.getKey())) shape.add(extent);
            }
            Files.writeString(
                    directory.resolve("manifest.json"),
                    json.writerWithDefaultPrettyPrinter().writeValueAsString(manifest));
            Files.writeString(
                    directory.resolve("inputs.json"),
                    "{\"start\": " + start + ", \"anchor\": " + anchor + ", \"context_rows\": " + start + "}\n");
        }
    }

    /// Every tensor the reference holds is within its bound, and the proposal is the reference's.
    private static void compare(Path reference, Recorder engine, int[] proposal) throws Exception {
        JsonNode manifest =
                new ObjectMapper().readTree(reference.resolve("manifest.json").toFile());
        StringBuilder report = new StringBuilder();
        boolean failed = false;
        for (var names = manifest.get("tensors").fieldNames(); names.hasNext(); ) {
            String name = names.next();
            if (!engine.data.containsKey(name) || name.equals("proposal") || name.equals("topk_indices")) continue;
            JsonNode entry = manifest.get("tensors").get(name);
            float[] expected = floats(
                    Files.readAllBytes(reference.resolve(entry.get("file").asText())),
                    entry.get("dtype").asText());
            float[] actual = floats(engine.data.get(name), engine.types.get(name));
            if (name.equals("topk_values") || name.equals("selector_scores")) {
                sortRows(expected, 16);
                sortRows(actual, 16);
            }
            if (expected.length != actual.length) {
                report.append(String.format("%-36s length %d != %d%n", name, actual.length, expected.length));
                failed = true;
                continue;
            }
            double difference = 0, norm = 0;
            for (int i = 0; i < expected.length; i++) {
                difference += Math.pow(actual[i] - expected[i], 2);
                norm += Math.pow(expected[i], 2);
            }
            double relative = norm == 0 ? Math.sqrt(difference) : Math.sqrt(difference / norm);
            boolean over = relative > LAYER_BOUND;
            failed |= over;
            report.append(String.format("%-36s %10.3e%s%n", name, relative, over ? "  OVER" : ""));
        }
        JsonNode proposalEntry = manifest.get("tensors").get("proposal");
        ByteBuffer proposalBytes = ByteBuffer.wrap(Files.readAllBytes(
                        reference.resolve(proposalEntry.get("file").asText())))
                .order(ByteOrder.LITTLE_ENDIAN);
        int[] expectedProposal = new int[proposalBytes.capacity() / Integer.BYTES];
        for (int i = 0; i < expectedProposal.length; i++) expectedProposal[i] = proposalBytes.getInt(4 * i);
        System.out.println("DFLASH2_FIXTURE comparison\n" + report);
        assertArrayEquals(expectedProposal, proposal, "the proposal");
        assertTrue(!failed, "a tensor exceeds its bound:\n" + report);
    }

    private static float[] floats(byte[] bytes, String type) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int element = type.equals("bf16") ? 2 : 4;
        float[] values = new float[bytes.length / element];
        for (int i = 0; i < values.length; i++)
            values[i] = switch (type) {
                case "bf16" -> Float.intBitsToFloat((buffer.getShort(2 * i) & 0xFFFF) << 16);
                case "f32" -> buffer.getFloat(4 * i);
                default -> (float) buffer.getInt(4 * i);
            };
        return values;
    }

    private static void sortRows(float[] values, int width) {
        for (int row = 0; row < values.length / width; row++) Arrays.sort(values, row * width, (row + 1) * width);
    }
}
