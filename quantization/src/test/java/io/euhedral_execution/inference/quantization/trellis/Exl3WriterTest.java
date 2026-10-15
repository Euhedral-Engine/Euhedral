package io.euhedral_execution.inference.quantization.trellis;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import io.euhedral_execution.inference.quantization.checkpoint.Safetensors;
import io.euhedral_execution.inference.quantization.checkpoint.SafetensorsWriter;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The writer's files carry exactly the tensors, dtypes, shapes and order of a real exllamav3 output (the fixture in
/// `exl3-layout`), and a miniature checkpoint round-trips through the whole write, index included. The layout tests
/// need the Swift-1.5 checkpoint for shapes (they skip without it).
class Exl3WriterTest {

    private static final ObjectMapper JSON = new ObjectMapper();

    private static List<String> fixtureLines(String name) throws IOException {
        try (InputStream in = Exl3WriterTest.class.getResourceAsStream("/trellis-fixture/exl3-layout/" + name)) {
            return Arrays.stream(new String(in.readAllBytes(), StandardCharsets.UTF_8).split("\n"))
                    .filter(l -> !l.isBlank())
                    .toList();
        }
    }

    private static JsonNode fixtureJson(String name) throws IOException {
        try (InputStream in = Exl3WriterTest.class.getResourceAsStream("/trellis-fixture/exl3-layout/" + name)) {
            return JSON.readTree(in);
        }
    }

    /// `name<TAB>dtype<TAB>[shape]` of every tensor of a safetensors file, in file order.
    private static List<String> described(Path file) throws IOException {
        byte[] bytes = Files.readAllBytes(file);
        long headerBytes = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getLong(0);
        JsonNode header = JSON.readTree(new String(bytes, 8, (int) headerBytes, StandardCharsets.UTF_8));
        List<JsonNode> nodes = new ArrayList<>();
        Map<JsonNode, String> names = new HashMap<>();
        for (Iterator<Map.Entry<String, JsonNode>> it = header.fields(); it.hasNext(); ) {
            var e = it.next();
            if (e.getKey().equals("__metadata__")) continue;
            nodes.add(e.getValue());
            names.put(e.getValue(), e.getKey());
        }
        nodes.sort(Comparator.comparingLong(n -> n.get("data_offsets").get(0).asLong()));
        List<String> lines = new ArrayList<>();
        for (JsonNode n : nodes) {
            long[] shape = new long[n.get("shape").size()];
            for (int i = 0; i < shape.length; i++)
                shape[i] = n.get("shape").get(i).asLong();
            lines.add(names.get(n) + "\t" + n.get("dtype").asText() + "\t" + Arrays.toString(shape));
        }
        return lines;
    }

    private static TrellisQuantizer.Result zeros(long n, long k, double bits) {
        return new TrellisQuantizer.Result(
                new short[(int) (k / 16 * (n / 16) * Math.round(bits * 16))],
                new short[(int) k],
                new short[(int) n],
                0,
                0,
                1f,
                0);
    }

    // ---- against the real layout ----

    @Test
    void layersMatchExllamav3sFiles(@TempDir Path out) throws Exception {
        Path checkpoint = LinearsTest.checkpoint();
        try (Safetensors store = Safetensors.open(checkpoint);
                Exl3Writer writer = new Exl3Writer(checkpoint, out, 2.5, 218, 2048)) {
            var objects = new Qwen38Objects(store, checkpoint);
            JsonNode expectedStorage = fixtureJson("quantization_config.json").get("tensor_storage");
            for (int index : new int[] {0, 3}) {
                Map<String, TrellisQuantizer.Result> results = new HashMap<>();
                Map<String, Double> bits = new HashMap<>();
                for (Linears.Linear linear : Linears.layer(objects, index)) {
                    if (!linear.quantized()) continue;
                    results.put(linear.tensor(), zeros(linear.rows(), linear.columns(), 2.5));
                    bits.put(linear.tensor(), 2.5);
                }
                writer.layer(index, results, bits);
                assertEquals(
                        fixtureLines("layers." + index + ".tensors.txt"),
                        described(out.resolve("model.language_model.layers." + index + ".safetensors")));
            }
            JsonNode storage = writer.tensorStorage();
            int compared = 0;
            for (Iterator<String> it = storage.fieldNames(); it.hasNext(); ) {
                String key = it.next();
                assertEquals(
                        JSON.writeValueAsString(expectedStorage.get(key)),
                        JSON.writeValueAsString(storage.get(key)),
                        key);
                compared++;
            }
            assertEquals(22, compared); // 11 modules per layer (GDN has a/b, attention has q/k norms)
            String[] order = {"input_layernorm", "in_proj_qkv", "in_proj_z", "in_proj_b"};
            Iterator<String> keys = storage.fieldNames();
            for (String suffix : order) assertTrue(keys.next().endsWith(suffix));
        }
    }

    @Test
    void mtpMatchesExllamav3sTensorsAndQuantizationConfig(@TempDir Path out) throws Exception {
        Path checkpoint = LinearsTest.checkpoint();
        Map<String, TrellisQuantizer.Result> results = new HashMap<>();
        try (Safetensors store = Safetensors.open(checkpoint);
                Exl3Writer writer = new Exl3Writer(checkpoint, out, 2.5, 218, 2048)) {
            for (String line : fixtureLines("mtp.tensors.txt")) {
                String name = line.split("\t")[0];
                if (!name.endsWith(".trellis")) continue;
                String linear = name.substring(0, name.length() - ".trellis".length());
                long[] shape = store.tensor(linear + ".weight").shape();
                results.put(linear, zeros(shape[0], shape[1], Exl3Writer.MTP_BITS));
            }
            writer.mtp(results);
            assertEquals(fixtureLines("mtp.tensors.txt"), described(out.resolve("mtp.safetensors")));
            var expected = (com.fasterxml.jackson.databind.node.ObjectNode) fixtureJson("quantization_config.json");
            expected.remove("tensor_storage");
            expected.remove("vision_bits"); // the vision tower is copied unquantized
            assertEquals(JSON.writeValueAsString(expected), JSON.writeValueAsString(writer.quantizationConfig()));
        }
    }

    @Test
    void auxiliaryFilesAreTheOnesExllamav3Copies() throws Exception {
        Path checkpoint = LinearsTest.checkpoint();
        Set<String> expected = new HashSet<>();
        for (String line : fixtureLines("files.txt")) {
            if (!line.startsWith("-")) continue;
            String name = line.substring(line.lastIndexOf(' ') + 1);
            if (!name.endsWith(".safetensors")
                    && !Set.of("config.json", "quantization_config.json").contains(name)) expected.add(name);
        }
        expected.remove("model.safetensors.index.json");
        assertEquals(expected, new HashSet<>(Exl3Writer.auxiliaryFiles(checkpoint)));
    }

    // ---- a miniature checkpoint, end to end ----

    private static final short BF16_ONE = 0x3F80;
    private static final short BF16_HUGE = 0x7F00; // about 1.7e38, beyond F16

    private static void tiny(Path dir, boolean overflow) throws Exception {
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("config.json"), """
                {"model_type":"qwen3_5","tie_word_embeddings":true,"text_config":{"num_hidden_layers":1,
                 "layer_types":["linear_attention"],"hidden_size":128,"intermediate_size":128,"num_attention_heads":2,
                 "num_key_value_heads":1,"head_dim":8,"partial_rotary_factor":0.25,"linear_num_key_heads":1,
                 "linear_num_value_heads":8,"linear_key_head_dim":16,"linear_value_head_dim":16,
                 "linear_conv_kernel_dim":4,"rms_norm_eps":1e-06,"rope_parameters":{"rope_theta":10000000},
                 "vocab_size":128}}
                """);
        Files.writeString(dir.resolve("tokenizer.json"), "{}");
        Files.writeString(dir.resolve("LICENSE"), "x");
        Files.writeString(dir.resolve("old.bin"), "x");
        Map<String, long[]> shapes = new java.util.TreeMap<>();
        String l = "model.language_model.layers.0.";
        for (String n : new String[] {"in_proj_qkv", "in_proj_z", "out_proj"})
            shapes.put(l + "linear_attn." + n + ".weight", new long[] {128, 128});
        shapes.put(l + "linear_attn.in_proj_a.weight", new long[] {8, 128});
        shapes.put(l + "linear_attn.in_proj_b.weight", new long[] {8, 128});
        shapes.put(l + "linear_attn.A_log", new long[] {8});
        shapes.put(l + "linear_attn.dt_bias", new long[] {8});
        shapes.put(l + "linear_attn.conv1d.weight", new long[] {128, 1, 4});
        shapes.put(l + "linear_attn.norm.weight", new long[] {16});
        shapes.put(l + "input_layernorm.weight", new long[] {128});
        shapes.put(l + "post_attention_layernorm.weight", new long[] {128});
        for (String n : new String[] {"gate_proj", "up_proj", "down_proj"})
            shapes.put(l + "mlp." + n + ".weight", new long[] {128, 128});
        shapes.put("model.language_model.embed_tokens.weight", new long[] {128, 128});
        shapes.put("model.language_model.norm.weight", new long[] {128});
        shapes.put("lm_head.weight", new long[] {128, 128});
        shapes.put("model.visual.blocks.0.attn.qkv.weight", new long[] {48, 16});
        shapes.put("model.visual.blocks.0.attn.qkv.bias", new long[] {48});
        shapes.put("mtp.fc.weight", new long[] {128, 256});
        shapes.put("mtp.norm.weight", new long[] {128});
        shapes.put("mtp.layers.0.input_layernorm.weight", new long[] {128});
        for (String n : new String[] {"q_proj", "k_proj", "v_proj", "o_proj"})
            shapes.put("mtp.layers.0.self_attn." + n + ".weight", new long[] {128, 128});
        for (String n : new String[] {"gate_proj", "up_proj", "down_proj"})
            shapes.put("mtp.layers.0.mlp." + n + ".weight", new long[] {128, 128});
        try (var writer = new SafetensorsWriter(dir.resolve("model.safetensors"))) {
            for (var e : shapes.entrySet()) {
                long count = 1;
                for (long d : e.getValue()) count *= d;
                short[] data = new short[(int) count];
                Arrays.fill(data, BF16_ONE);
                if (overflow && e.getKey().equals("lm_head.weight")) data[5] = BF16_HUGE;
                writer.add(e.getKey(), "BF16", e.getValue(), MemorySegment.ofArray(data));
            }
        }
    }

    private static Map<String, TrellisQuantizer.Result> tinyResults(String[] linears, long n, long k, double bits) {
        Map<String, TrellisQuantizer.Result> results = new HashMap<>();
        for (String linear : linears) results.put(linear, zeros(n, k, bits));
        return results;
    }

    @Test
    void miniatureCheckpointRoundTrips(@TempDir Path dir) throws Exception {
        Path source = dir.resolve("source"), out = dir.resolve("out");
        tiny(source, false);
        String l = "model.language_model.layers.0.";
        String[] gdn = {
            l + "linear_attn.in_proj_qkv",
            l + "linear_attn.in_proj_z",
            l + "linear_attn.out_proj",
            l + "mlp.gate_proj",
            l + "mlp.up_proj",
            l + "mlp.down_proj"
        };
        Map<String, Double> bits = new HashMap<>();
        for (String linear : gdn) bits.put(linear, 2.0);
        Map<String, TrellisQuantizer.Result> results = tinyResults(gdn, 128, 128, 2.0);
        results.put(
                gdn[0],
                new TrellisQuantizer.Result(
                        new short[8 * 8 * 32], shorts(128, 0x3C00), shorts(128, 0x4000), 0, 0, 1f, 0));
        Map<String, TrellisQuantizer.Result> mtp = new HashMap<>();
        for (String linear : new String[] {
            "mtp.layers.0.self_attn.q_proj", "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.v_proj",
            "mtp.layers.0.self_attn.o_proj", "mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj",
            "mtp.layers.0.mlp.down_proj"
        }) mtp.put(linear, zeros(128, 128, 4));
        mtp.put("mtp.fc", zeros(128, 256, 4));

        try (Exl3Writer writer = new Exl3Writer(source, out, 2.0, 4, 8)) {
            assertThrows(IllegalStateException.class, () -> writer.finish(Map.of())); // layer 0 is missing
        }
        out = dir.resolve("out2");
        try (Exl3Writer writer = new Exl3Writer(source, out, 2.0, 4, 8)) {
            writer.layer(0, results, bits);
            writer.mtp(mtp);
            writer.finish(bits);
        }

        try (Safetensors written = Safetensors.open(out);
                Safetensors original = Safetensors.open(source);
                Arena arena = Arena.ofConfined()) {
            // every source tensor is there, quantized linears as trellis/suh/svh/mul1
            for (String name : original.names()) {
                String linear = name.substring(0, name.lastIndexOf('.'));
                assertTrue(written.contains(name) || written.contains(linear + ".trellis"), name);
            }
            assertEquals("I16", written.tensor(gdn[0] + ".trellis").dtype());
            assertArrayEquals(
                    new long[] {8, 8, 32}, written.tensor(gdn[0] + ".trellis").shape());
            assertEquals("F16", written.tensor(gdn[0] + ".suh").dtype());
            assertArrayEquals(new long[] {}, written.tensor(gdn[0] + ".mul1").shape());
            assertArrayEquals(
                    new long[] {8, 8, 64},
                    written.tensor("mtp.layers.0.self_attn.q_proj.trellis").shape());
            MemorySegment scale = arena.allocate(256);
            written.read(gdn[0] + ".suh", scale);
            assertEquals((short) 0x3C00, scale.getAtIndex(ValueLayout.JAVA_SHORT, 17));
            MemorySegment mul1 = arena.allocate(4);
            written.read(gdn[0] + ".mul1", mul1);
            assertEquals(0x83DCD12D, mul1.get(ValueLayout.JAVA_INT, 0));
            // BF16 1.0 becomes F16 1.0 for the unquantized a/b projections and lm_head; other tensors keep BF16
            for (String name : new String[] {l + "linear_attn.in_proj_a.weight", "lm_head.weight"}) {
                assertEquals("F16", written.tensor(name).dtype());
                MemorySegment half = arena.allocate(2);
                written.read(name, 0, half);
                assertEquals((short) 0x3C00, half.get(ValueLayout.JAVA_SHORT, 0));
            }
            assertEquals(
                    "BF16",
                    written.tensor("model.language_model.embed_tokens.weight").dtype());
            assertEquals(
                    "BF16",
                    written.tensor("model.visual.blocks.0.attn.qkv.weight").dtype());
            assertEquals("BF16", written.tensor(l + "linear_attn.A_log").dtype());
        }

        JsonNode index =
                JSON.readTree(out.resolve("model.safetensors.index.json").toFile());
        long total = 0;
        try (Safetensors written = Safetensors.open(out)) {
            Set<String> mapped = new HashSet<>();
            index.get("weight_map").fieldNames().forEachRemaining(mapped::add);
            assertEquals(written.names(), mapped);
            for (String name : mapped) total += written.tensor(name).bytes();
        }
        assertEquals(total, index.get("metadata").get("total_size").asLong());
        assertEquals(
                "lm_head.safetensors",
                index.get("weight_map").get("lm_head.weight").asText());
        assertEquals(
                "model.language_model.layers.0.safetensors",
                index.get("weight_map").get(gdn[0] + ".suh").asText());

        JsonNode config = JSON.readTree(out.resolve("config.json").toFile());
        assertFalse(config.get("tie_word_embeddings").asBoolean());
        assertEquals(1, config.get("text_config").get("num_hidden_layers").asInt());
        JsonNode q = config.get("quantization_config");
        assertEquals("exl3", q.get("quant_method").asText());
        assertEquals("1.6.1", q.get("version").asText());
        assertEquals(2.0, q.get("bits").asDouble());
        assertEquals(16, q.get("head_bits").asInt());
        assertEquals(4, q.get("calibration").get("rows").asInt());
        assertEquals(8, q.get("calibration").get("cols").asInt());
        assertEquals("auto", q.get("out_scales").asText());
        assertEquals("mul1", q.get("codebook").asText());
        assertEquals(4, q.get("mtp_bits").asInt());
        assertFalse(q.has("vision_bits"));

        JsonNode storage =
                JSON.readTree(out.resolve("quantization_config.json").toFile()).get("tensor_storage");
        JsonNode qkv = storage.get(gdn[0]);
        assertEquals("exl3", qkv.get("quant_format").asText());
        assertEquals(2, qkv.get("bits_per_weight").asInt());
        assertEquals(2212286765L, qkv.get("mul1_multiplier").asLong());
        assertEquals(
                "torch.int16",
                qkv.get("stored_tensors").get(gdn[0] + ".trellis").get("dtype").asText());
        assertEquals(
                "torch.float16",
                storage.get(l + "linear_attn.in_proj_a")
                        .get("stored_tensors")
                        .get(l + "linear_attn.in_proj_a.weight")
                        .get("dtype")
                        .asText());
        assertEquals(
                "torch.float16",
                storage.get("lm_head")
                        .get("stored_tensors")
                        .get("lm_head.weight")
                        .get("dtype")
                        .asText());
        assertEquals(1 + 11 + 2, storage.size()); // embedding, layer 0's modules (GDN), norm and lm_head
        assertFalse(storage.has(l + "linear_attn.conv1d"));

        assertTrue(Files.exists(out.resolve("tokenizer.json")));
        assertTrue(Files.exists(out.resolve("LICENSE")));
        assertFalse(Files.exists(out.resolve("old.bin")));
    }

    private static short[] shorts(int count, int value) {
        short[] all = new short[count];
        Arrays.fill(all, (short) value);
        return all;
    }

    @Test
    void failsLoudlyWhenAnUnquantizedLinearOverflowsHalfPrecision(@TempDir Path dir) throws Exception {
        tiny(dir.resolve("source"), true);
        try (Exl3Writer writer = new Exl3Writer(dir.resolve("source"), dir.resolve("out"), 2.0, 4, 8)) {
            ArithmeticException e = assertThrows(ArithmeticException.class, () -> writer.finish(Map.of()));
            assertTrue(e.getMessage().contains("lm_head.weight"));
        }
    }
}
