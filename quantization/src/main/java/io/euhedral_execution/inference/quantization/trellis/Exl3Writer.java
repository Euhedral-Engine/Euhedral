package io.euhedral_execution.inference.quantization.trellis;

import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.core.util.Separators;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import io.euhedral_execution.inference.quantization.checkpoint.Safetensors;
import io.euhedral_execution.inference.quantization.checkpoint.SafetensorsWriter;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Stream;

/// Writes a Qwen3.8 checkpoint in exllamav3's EXL3 layout from quantized results and the BF16 source.
///
/// - One safetensors file per decoder layer (`model.language_model.layers.<i>.safetensors`), `mtp.safetensors`,
///   `model.language_model.embed_tokens.safetensors`, `lm_head.safetensors` (with the final norm) and
///   `model.visual.safetensors`, plus a standard `model.safetensors.index.json` over all of them. Tensors inside a
///   file are sorted by name, as exllamav3 writes them.
/// - A quantized linear is `.trellis` (I16 `[k/16, n/16, 16 * bits]`), `.suh` (F16 `[k]`), `.svh` (F16 `[n]`) and
///   `.mul1` (I32 scalar, the codebook multiplier). Linears that are not quantized are stored as F16 (the GDN
///   `in_proj_a`/`in_proj_b`, `lm_head`); everything else, vision tower included, keeps its source dtype.
/// - `config.json` is the source config with the `quantization_config` block; `quantization_config.json` adds
///   `tensor_storage` for the text modules, as exllamav3's `create_quantization_config_json` does.
///
/// Only one layer's results are held at a time and source tensors are streamed through a fixed chunk buffer.
///
/// Portions ported from exllamav3 (MIT License, Copyright (c) 2025 Turboderp).
public final class Exl3Writer implements AutoCloseable {

    /// The multiplier of the `mul1` codebook, as the signed I32 exllamav3 stores.
    public static final int MUL1 = 0x83DCD12D;

    /// Bits of the MTP linears.
    public static final int MTP_BITS = 4;

    private static final String VERSION = "1.6.1";
    private static final String TEXT = "model.language_model.";
    private static final ObjectMapper JSON = new ObjectMapper();

    /// Module order inside a decoder layer, as exllamav3 lists the leaf modules of `tensor_storage`.
    private static final List<String> LAYER_MODULES = List.of(
            "input_layernorm",
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_a",
            "linear_attn.out_proj",
            "linear_attn.norm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "post_attention_layernorm",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj");
    private static final List<String> STORED_ORDER = List.of("weight", "suh", "svh", "mul1", "trellis");
    private static final Map<String, String> TORCH_DTYPES = Map.of(
            "I16", "torch.int16",
            "F16", "torch.float16",
            "BF16", "torch.bfloat16",
            "F32", "torch.float32",
            "I32", "torch.int32");

    private record Stored(String dtype, long[] shape, long bytes) {}

    private final Path source;
    private final Path out;
    private final double bits;
    private final int rows;
    private final int columns;
    private final Safetensors store;
    private final Qwen38Objects objects;

    private final Map<String, String> weightMap = new LinkedHashMap<>();
    private final Map<String, Map<String, Stored>> modules = new HashMap<>();
    private final Map<String, Long> fileBytes = new HashMap<>();
    private final int writeBufferBytes;
    private boolean mtpWritten;

    /// As [#Exl3Writer(Path, Path, double, int, int, int)] with a 16 MiB write buffer.
    public Exl3Writer(Path source, Path out, double bits, int rows, int columns) throws IOException {
        this(source, out, bits, rows, columns, SafetensorsWriter.DEFAULT_BUFFER_BYTES);
    }

    /// `bits` is the average bits per weight of the text linears, `rows` and `columns` the calibration shape and
    /// `writeBufferBytes` the size of the buffer each file is copied through.
    public Exl3Writer(Path source, Path out, double bits, int rows, int columns, int writeBufferBytes)
            throws IOException {
        this.writeBufferBytes = writeBufferBytes;
        this.source = source;
        this.out = out;
        this.bits = bits;
        this.rows = rows;
        this.columns = columns;
        Files.createDirectories(out);
        this.store = Safetensors.open(source);
        this.objects = new Qwen38Objects(store, source);
    }

    // ---- files ----

    /// The tensors of one output file, written sorted by name.
    private final class Shard {
        private final String file;
        private final TreeMap<String, Pending> tensors = new TreeMap<>();

        private Shard(String file) {
            this.file = file;
        }

        private record Pending(String dtype, long[] shape, long bytes, SafetensorsWriter.Source source) {}

        void add(String name, String dtype, long[] shape, long bytes, SafetensorsWriter.Source source) {
            if (tensors.put(name, new Pending(dtype, shape, bytes, source)) != null)
                throw new IllegalStateException("tensor " + name + " written twice");
        }

        void add(String name, String dtype, long[] shape, MemorySegment data) {
            add(
                    name,
                    dtype,
                    shape,
                    data.byteSize(),
                    (offset, destination) -> MemorySegment.copy(data, offset, destination, 0, destination.byteSize()));
        }

        /// Copies source tensor `name` unchanged.
        void copy(String name) {
            Safetensors.Tensor t = store.tensor(name);
            add(name, t.dtype(), t.shape(), t.bytes(), (offset, destination) -> store.read(name, offset, destination));
        }

        /// Copies source tensor `name` as F16 (round to nearest even), failing on any value that overflows F16.
        void copyAsHalf(String name) {
            Safetensors.Tensor t = store.tensor(name);
            if (t.dtype().equals("F16")) {
                copy(name);
                return;
            }
            if (!t.dtype().equals("BF16"))
                throw new IllegalArgumentException(name + " is " + t.dtype() + ", not BF16 or F16");
            add(name, "F16", t.shape(), t.bytes(), (offset, destination) -> {
                store.read(name, offset, destination);
                bf16ToHalfInPlace(name, destination);
            });
        }

        void write() throws IOException {
            try (SafetensorsWriter writer = new SafetensorsWriter(out.resolve(file), writeBufferBytes)) {
                for (var e : tensors.entrySet()) {
                    Pending p = e.getValue();
                    writer.add(e.getKey(), p.dtype(), p.shape(), p.bytes(), p.source());
                }
            }
            weightMap.values().removeIf(file::equals); // a rewritten file replaces its earlier tensors
            long bytes = 0;
            for (var e : tensors.entrySet()) {
                Pending p = e.getValue();
                weightMap.put(e.getKey(), file);
                bytes += p.bytes();
                record(e.getKey(), new Stored(p.dtype(), p.shape(), p.bytes()));
            }
            fileBytes.put(file, bytes);
        }
    }

    private static void bf16ToHalfInPlace(String name, MemorySegment words) {
        long count = words.byteSize() / 2;
        for (long i = 0; i < count; i++) {
            float value = Float.intBitsToFloat((words.getAtIndex(ValueLayout.JAVA_SHORT, i) & 0xFFFF) << 16);
            short half = Float.floatToFloat16(value);
            if (Float.isFinite(value) && (half & 0x7C00) == 0x7C00)
                throw new ArithmeticException(name + ": " + value + " overflows F16");
            words.setAtIndex(ValueLayout.JAVA_SHORT, i, half);
        }
    }

    /// Notes a written tensor under its `tensor_storage` module, if it belongs to one.
    private void record(String tensor, Stored stored) {
        int dot = tensor.lastIndexOf('.');
        String module = tensor.substring(0, dot);
        String part = tensor.substring(dot + 1);
        if (!STORED_ORDER.contains(part)) return;
        boolean listed =
                module.equals(TEXT + "embed_tokens") || module.equals(TEXT + "norm") || module.equals("lm_head");
        if (module.startsWith(TEXT + "layers.")) {
            String rest = module.substring((TEXT + "layers.").length());
            listed = LAYER_MODULES.contains(rest.substring(rest.indexOf('.') + 1));
        }
        if (listed) modules.computeIfAbsent(module, m -> new HashMap<>()).put(part, stored);
    }

    // ---- quantized linears ----

    private void quantizedLinear(Shard shard, String linear, TrellisQuantizer.Result result, double expectedBits) {
        long[] weight = store.tensor(linear + ".weight").shape();
        long n = weight[0], k = weight[1];
        long tiles = k / 16 * (n / 16);
        double words = expectedBits * 16;
        if (words != Math.rint(words)) throw new IllegalArgumentException(linear + ": " + expectedBits + " bits");
        if (result.trellis().length != tiles * (long) words)
            throw new IllegalArgumentException(linear + ": trellis has " + result.trellis().length + " words, "
                    + expectedBits + " bits over " + n + "x" + k + " need " + tiles * (long) words);
        if (result.suh().length != k || result.svh().length != n)
            throw new IllegalArgumentException(linear + ": scales " + result.suh().length + "/" + result.svh().length
                    + " do not fit " + n + "x" + k);
        shard.add(
                linear + ".trellis",
                "I16",
                new long[] {k / 16, n / 16, (long) words},
                MemorySegment.ofArray(result.trellis()));
        shard.add(linear + ".suh", "F16", new long[] {k}, MemorySegment.ofArray(result.suh()));
        shard.add(linear + ".svh", "F16", new long[] {n}, MemorySegment.ofArray(result.svh()));
        shard.add(linear + ".mul1", "I32", new long[] {}, MemorySegment.ofArray(new int[] {MUL1}));
    }

    /// Writes decoder layer `index`: the quantized linears (keyed by tensor name without `.weight`, as
    /// [Linears#layer] names them, with their bits in `bitsByLinear`), the unquantized linears as F16 and every
    /// other tensor of the layer as in the source.
    public void layer(int index, Map<String, TrellisQuantizer.Result> quantized, Map<String, Double> bitsByLinear)
            throws IOException {
        String prefix = TEXT + "layers." + index + ".";
        Shard shard = new Shard(TEXT + "layers." + index + ".safetensors");
        Set<String> linears = new TreeSet<>();
        for (Linears.Linear linear : Linears.layer(objects, index)) {
            linears.add(linear.tensor() + ".weight");
            if (linear.quantized()) {
                TrellisQuantizer.Result result = quantized.get(linear.tensor());
                Double linearBits = bitsByLinear.get(linear.tensor());
                if (result == null || linearBits == null)
                    throw new IllegalArgumentException("no result or bits for " + linear.tensor());
                quantizedLinear(shard, linear.tensor(), result, linearBits);
            } else {
                shard.copyAsHalf(linear.tensor() + ".weight");
            }
        }
        for (String key : quantized.keySet()) {
            if (!linears.contains(key + ".weight") || !key.startsWith(prefix))
                throw new IllegalArgumentException(key + " is not a quantized linear of layer " + index);
        }
        for (String name : sorted(store.names())) {
            if (name.startsWith(prefix) && !linears.contains(name)) shard.copy(name);
        }
        shard.write();
    }

    /// Writes `mtp.safetensors`: the eight MTP linears (keyed by tensor name without `.weight`, 4 bits each) and the
    /// remaining `mtp.*` tensors as in the source.
    public void mtp(Map<String, TrellisQuantizer.Result> quantized) throws IOException {
        if (!quantized.keySet().equals(Set.copyOf(Linears.MTP_LINEARS)))
            throw new IllegalArgumentException(
                    "mtp linears are " + Linears.MTP_LINEARS + ", got " + quantized.keySet());
        Shard shard = new Shard("mtp.safetensors");
        for (String linear : Linears.MTP_LINEARS) quantizedLinear(shard, linear, quantized.get(linear), MTP_BITS);
        for (String name : sorted(store.names())) {
            if (!name.startsWith("mtp.")) continue;
            boolean linear = name.endsWith(".weight")
                    && Linears.MTP_LINEARS.contains(name.substring(0, name.length() - ".weight".length()));
            if (!linear) shard.copy(name);
        }
        shard.write();
        mtpWritten = true;
    }

    /// Writes the embedding, the final norm, `lm_head` (F16), the vision tower, the index, both configs and the
    /// auxiliary files, once every source tensor is accounted for.
    public void finish(Map<String, Double> bitsByLinear) throws IOException {
        adoptFilesFromEarlierRun();
        Set<String> headNames = new TreeSet<>();
        headNames.add(TEXT + "embed_tokens.weight");
        headNames.add("lm_head.weight");
        headNames.add(TEXT + "norm.weight");
        List<String> visionNames = sorted(store.names()).stream()
                .filter(n -> n.startsWith("model.visual."))
                .toList();

        // everything is checked before the large files are written
        List<String> missing = new ArrayList<>();
        for (String name : sorted(store.names())) {
            String linear = name.endsWith(".weight") ? name.substring(0, name.length() - ".weight".length()) : name;
            if (!weightMap.containsKey(name)
                    && !weightMap.containsKey(linear + ".trellis")
                    && !headNames.contains(name)
                    && !visionNames.contains(name)) missing.add(name);
        }
        if (!missing.isEmpty())
            throw new IllegalStateException(missing.size() + " source tensors were not written, e.g. "
                    + missing.subList(0, Math.min(5, missing.size())));
        for (var e : bitsByLinear.entrySet()) {
            Map<String, Stored> module = modules.get(e.getKey());
            if (module != null
                    && module.containsKey("trellis")
                    && module.get("trellis").shape()[2] != Math.round(e.getValue() * 16))
                throw new IllegalStateException(e.getKey() + " is stored at a different rate than " + e.getValue());
        }
        Set<String> expectedFiles = new HashSet<>(weightMap.values());
        expectedFiles.add(TEXT + "embed_tokens.safetensors");
        expectedFiles.add("lm_head.safetensors");
        if (!visionNames.isEmpty()) expectedFiles.add("model.visual.safetensors");
        try (Stream<Path> files = Files.list(out)) {
            List<String> stray = files.map(p -> p.getFileName().toString())
                    .filter(f -> f.endsWith(".safetensors") && !expectedFiles.contains(f))
                    .sorted()
                    .toList();
            if (!stray.isEmpty())
                throw new IllegalStateException(
                        "unexpected safetensors files in " + out + " (a loader reads them all): " + stray);
        }

        Shard embedding = new Shard(TEXT + "embed_tokens.safetensors");
        embedding.copy(TEXT + "embed_tokens.weight");
        embedding.write();
        Shard head = new Shard("lm_head.safetensors");
        head.copyAsHalf("lm_head.weight");
        head.copy(TEXT + "norm.weight");
        head.write();
        if (!visionNames.isEmpty()) {
            Shard vision = new Shard("model.visual.safetensors");
            for (String name : visionNames) vision.copy(name);
            vision.write();
        }

        for (String file : auxiliaryFiles(source))
            Files.copy(source.resolve(file), out.resolve(file), StandardCopyOption.REPLACE_EXISTING);

        ObjectNode index = JSON.createObjectNode();
        index.putObject("metadata")
                .put(
                        "total_size",
                        fileBytes.values().stream().mapToLong(Long::longValue).sum());
        ObjectNode map = index.putObject("weight_map");
        weightMap.forEach(map::put);
        writeJson(out.resolve("model.safetensors.index.json"), index);

        ObjectNode config =
                (ObjectNode) JSON.readTree(source.resolve("config.json").toFile());
        if (config.has("quantization_config"))
            throw new IllegalArgumentException("the source checkpoint is already quantized");
        ObjectNode quantization = quantizationConfig();
        config.put("tie_word_embeddings", false);
        config.set("quantization_config", quantization);
        writeJson(out.resolve("config.json"), config);

        ObjectNode full = quantization.deepCopy();
        full.set("tensor_storage", tensorStorage());
        writeJson(out.resolve("quantization_config.json"), full);
    }

    /// Records the layer and MTP files an earlier run left in `out` (the tensor names, shapes and sizes come from
    /// their headers), after checking each holds exactly the tensors this writer would write.
    private void adoptFilesFromEarlierRun() throws IOException {
        for (int index = 0; index < objects.config.layers(); index++) {
            String file = TEXT + "layers." + index + ".safetensors";
            if (fileBytes.containsKey(file) || !Files.isRegularFile(out.resolve(file))) continue;
            adopt(file, expectedLayerTensors(index));
        }
        String file = "mtp.safetensors";
        if (!fileBytes.containsKey(file) && Files.isRegularFile(out.resolve(file))) {
            adopt(file, expectedMtpTensors());
            mtpWritten = true;
        }
    }

    private Set<String> expectedLayerTensors(int index) {
        String prefix = TEXT + "layers." + index + ".";
        Set<String> expected = new TreeSet<>();
        Set<String> linears = new HashSet<>();
        for (Linears.Linear linear : Linears.layer(objects, index)) {
            linears.add(linear.tensor() + ".weight");
            if (linear.quantized())
                for (String part : List.of("trellis", "suh", "svh", "mul1")) expected.add(linear.tensor() + "." + part);
            else expected.add(linear.tensor() + ".weight");
        }
        for (String name : store.names()) if (name.startsWith(prefix) && !linears.contains(name)) expected.add(name);
        return expected;
    }

    private Set<String> expectedMtpTensors() {
        Set<String> expected = new TreeSet<>();
        for (String linear : Linears.MTP_LINEARS)
            for (String part : List.of("trellis", "suh", "svh", "mul1")) expected.add(linear + "." + part);
        for (String name : store.names()) {
            boolean linear = name.endsWith(".weight")
                    && Linears.MTP_LINEARS.contains(name.substring(0, name.length() - ".weight".length()));
            if (name.startsWith("mtp.") && !linear) expected.add(name);
        }
        return expected;
    }

    private void adopt(String file, Set<String> expected) throws IOException {
        Map<String, Stored> found = readHeader(out.resolve(file));
        if (!found.keySet().equals(expected)) {
            Set<String> absent = new TreeSet<>(expected);
            absent.removeAll(found.keySet());
            Set<String> extra = new TreeSet<>(found.keySet());
            extra.removeAll(expected);
            throw new IllegalStateException(file + " from an earlier run does not hold the expected tensors; missing "
                    + absent + ", unexpected " + extra);
        }
        long bytes = 0;
        for (var e : new TreeMap<>(found).entrySet()) {
            weightMap.put(e.getKey(), file);
            bytes += e.getValue().bytes();
            record(e.getKey(), e.getValue());
        }
        fileBytes.put(file, bytes);
    }

    /// The tensors of a safetensors file, from its header only.
    private static Map<String, Stored> readHeader(Path file) throws IOException {
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            ByteBuffer size = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, size, 0);
            long headerBytes = size.getLong(0);
            if (headerBytes <= 0 || headerBytes > (1L << 30))
                throw new IOException(file + ": bad header size " + headerBytes);
            ByteBuffer header = ByteBuffer.allocate((int) headerBytes);
            readFully(channel, header, 8);
            JsonNode root = JSON.readTree(new String(header.array(), StandardCharsets.UTF_8));
            Map<String, Stored> found = new HashMap<>();
            for (Iterator<Map.Entry<String, JsonNode>> it = root.fields(); it.hasNext(); ) {
                var entry = it.next();
                if (entry.getKey().equals("__metadata__")) continue;
                JsonNode node = entry.getValue();
                long[] shape = new long[node.get("shape").size()];
                for (int i = 0; i < shape.length; i++)
                    shape[i] = node.get("shape").get(i).asLong();
                long bytes = node.get("data_offsets").get(1).asLong()
                        - node.get("data_offsets").get(0).asLong();
                found.put(entry.getKey(), new Stored(node.get("dtype").asText(), shape, bytes));
            }
            return found;
        }
    }

    private static void readFully(FileChannel channel, ByteBuffer buffer, long position) throws IOException {
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, position);
            if (read < 0) throw new IOException("unexpected end of file");
            position += read;
        }
    }

    ObjectNode quantizationConfig() {
        ObjectNode q = JSON.createObjectNode();
        q.put("quant_method", "exl3");
        q.put("version", VERSION);
        q.put("bits", bits);
        q.put("head_bits", 16);
        ObjectNode calibration = q.putObject("calibration");
        calibration.put("rows", rows);
        calibration.put("cols", columns);
        q.put("out_scales", "always");
        q.put("codebook", "mul1");
        if (mtpWritten) q.put("mtp_bits", MTP_BITS);
        return q;
    }

    /// `tensor_storage` for the text modules written so far, in exllamav3's module order.
    ObjectNode tensorStorage() {
        ObjectNode storage = JSON.createObjectNode();
        List<String> keys = new ArrayList<>();
        keys.add(TEXT + "embed_tokens");
        for (int i = 0; i < objects.config.layers(); i++)
            for (String module : LAYER_MODULES) keys.add(TEXT + "layers." + i + "." + module);
        keys.add(TEXT + "norm");
        keys.add("lm_head");
        for (String key : keys) {
            Map<String, Stored> parts = modules.get(key);
            if (parts == null) continue;
            ObjectNode module = storage.putObject(key);
            ObjectNode stored = module.putObject("stored_tensors");
            for (String part : STORED_ORDER) {
                Stored s = parts.get(part);
                if (s == null) continue;
                ObjectNode node = stored.putObject(key + "." + part);
                ArrayNode shape = node.putArray("shape");
                for (long dimension : s.shape()) shape.add(dimension);
                node.put("n_bytes", s.bytes());
                node.put("dtype", TORCH_DTYPES.get(s.dtype()));
            }
            Stored trellis = parts.get("trellis");
            if (trellis != null) {
                double bpw = trellis.shape()[2] / 16.0;
                module.put("quant_format", "exl3");
                if (bpw == Math.rint(bpw)) module.put("bits_per_weight", (int) bpw);
                else module.put("bits_per_weight", bpw);
                module.put("mul1_multiplier", Integer.toUnsignedLong(MUL1));
            }
        }
        return storage;
    }

    /// The files exllamav3 copies next to the weights: every regular file of `source` except `.safetensors` files,
    /// `config.json`, the weight index and pickled checkpoints (`.bin`, `.ckpt`, `.pth`, `.pt`).
    public static List<String> auxiliaryFiles(Path source) throws IOException {
        try (Stream<Path> files = Files.list(source)) {
            return files.filter(Files::isRegularFile)
                    .map(p -> p.getFileName().toString())
                    .filter(f -> !f.endsWith(".safetensors")
                            && !f.equals("config.json")
                            && !f.equals("model.safetensors.index.json")
                            && Stream.of(".bin", ".ckpt", ".pth", ".pt").noneMatch(f::endsWith))
                    .sorted()
                    .toList();
        }
    }

    private static List<String> sorted(Set<String> names) {
        return names.stream().sorted().toList();
    }

    private static void writeJson(Path file, JsonNode node) throws IOException {
        DefaultIndenter indent = new DefaultIndenter("    ", "\n");
        DefaultPrettyPrinter printer = new DefaultPrettyPrinter()
                .withSeparators(
                        Separators.createDefaultInstance().withObjectFieldValueSpacing(Separators.Spacing.AFTER));
        printer.indentObjectsWith(indent);
        printer.indentArraysWith(indent);
        JSON.writer(printer).writeValue(file.toFile(), node);
    }

    @Override
    public void close() throws IOException {
        store.close();
    }
}
