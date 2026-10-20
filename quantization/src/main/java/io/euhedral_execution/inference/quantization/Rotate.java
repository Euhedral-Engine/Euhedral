package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.quantization.checkpoint.Safetensors;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.CudaLibraries;
import io.euhedral_execution.inference.quantization.gpu.Device;
import io.euhedral_execution.inference.quantization.gpu.Upload;
import io.euhedral_execution.inference.quantization.training.Ops;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/// A copy of a Qwen3.8 checkpoint with its residual stream rotated by an orthogonal Q (a row-vector residual h
/// becomes h Q), which leaves the model's function unchanged: Q = diag(s) (H_20 ⊗ H_256) / sqrt(5120), a
/// Hadamard product with random signs s. Each unit-offset RMSNorm weight w that feeds a linear is folded into
/// it and stored as 0 (RMS is invariant under Q, so norm(h Q) = norm(h) Q):
///
/// - the token embedding E becomes E Q;
/// - every linear reading a normed residual (the mixers' input projections, the MLP gate and up projections, the LM
/// head) becomes W diag(1 + w) Q;
/// - every linear writing the residual (the mixers' output projections, the MLP down projection) becomes Q^T W.
///
/// The MTP layer is left as it is: it reads the embedding and the post-final-norm hidden, which are now
/// rotated, so an artifact built from the copy does not draft (score it without speculation). Each shard keeps
/// its header and every tensor its byte range. The products run in FP32 on the device, one lattice frame per
/// tensor; the excess kurtosis of each rotated tensor's groups of 64 input columns, before and after, goes to
/// `kurtosis.tsv`.
public final class Rotate {

    /// Command-line settings.
    public record Settings(Path model, Path out, long seed, CudaLibraries library) {}

    static final int HIDDEN = 5120;
    private static final int GROUP = 64;
    /// Rows of a tensor transformed at once: as many as the widest residual-writing tensor (the MLP down
    /// projection, 5120 x 17408) has values, so that one buffer holds either.
    private static final int CHUNK_ROWS = 17408;
    private static final String TEXT = "model.language_model.";
    private static final Pattern INPUT = Pattern.compile(
            "model\\.language_model\\.layers\\.(\\d+)\\.(linear_attn\\.in_proj_(qkv|z|a|b)|self_attn\\.[qkv]_proj"
                    + "|mlp\\.(gate|up)_proj)\\.weight");
    private static final Pattern OUTPUT = Pattern.compile(
            "model\\.language_model\\.layers\\.\\d+\\.(linear_attn\\.out_proj|self_attn\\.o_proj|mlp\\.down_proj)"
                    + "\\.weight");
    private static final Pattern NORM = Pattern.compile(
            "model\\.language_model\\.(layers\\.\\d+\\.(input|post_attention)_layernorm|norm)\\.weight");

    /// How a tensor changes.
    enum Role {
        EMBEDDING,
        INPUT,
        OUTPUT,
        NORM,
        COPY
    }

    static Role role(String name) {
        if (name.equals(TEXT + "embed_tokens.weight")) return Role.EMBEDDING;
        if (name.equals("lm_head.weight") || INPUT.matcher(name).matches()) return Role.INPUT;
        if (OUTPUT.matcher(name).matches()) return Role.OUTPUT;
        if (NORM.matcher(name).matches()) return Role.NORM;
        return Role.COPY;
    }

    /// The norm whose weight an input-side linear folds.
    static String normOf(String name) {
        if (name.equals("lm_head.weight")) return TEXT + "norm.weight";
        Matcher m = INPUT.matcher(name);
        if (!m.matches()) throw new IllegalArgumentException(name + " reads no norm");
        String norm = m.group(2).startsWith("mlp") ? "post_attention_layernorm" : "input_layernorm";
        return TEXT + "layers." + m.group(1) + "." + norm + ".weight";
    }

    /// A Hadamard matrix of order 20 (Paley's first construction with q = 19): H = I + S with S skew, its first
    /// row and column ±1 and the rest the Jacobsthal matrix of the quadratic residues; entries ±1, H H^T = 20
    /// I.
    static int[][] hadamard20() {
        int q = 19;
        boolean[] square = new boolean[q];
        for (int x = 1; x < q; x++) square[x * x % q] = true;
        int[][] h = new int[q + 1][q + 1];
        for (int i = 0; i <= q; i++) {
            for (int j = 0; j <= q; j++) {
                int s;
                if (i == j) s = 0;
                else if (i == 0) s = 1;
                else if (j == 0) s = -1;
                else s = square[((j - i) % q + q) % q] ? 1 : -1;
                h[i][j] = s + (i == j ? 1 : 0);
            }
        }
        return h;
    }

    /// Entry (i, j) of Sylvester's Hadamard matrix of order 2^k: (-1)^popcount(i & j).
    static int sylvester(int i, int j) {
        return Integer.bitCount(i & j) % 2 == 0 ? 1 : -1;
    }

    /// Q, row-major: Q[i][j] = s_i H_20[i / 256][j / 256] H_256[i % 256][j % 256] / sqrt(5120).
    static float[] rotation(long seed) {
        int[][] h20 = hadamard20();
        Random random = new Random(seed);
        float[] q = new float[HIDDEN * HIDDEN];
        double scale = 1.0 / Math.sqrt(HIDDEN);
        for (int i = 0; i < HIDDEN; i++) {
            int sign = random.nextBoolean() ? 1 : -1;
            for (int j = 0; j < HIDDEN; j++)
                q[i * HIDDEN + j] = (float) (sign * h20[i / 256][j / 256] * sylvester(i % 256, j % 256) * scale);
        }
        return q;
    }

    private final Settings settings;
    private final Device device;
    private final Safetensors source;
    private final float[] rotation;
    private Ops ops;
    private Cuda.Function kurtosis;
    private long q, scaledQ, input, output, bf16, sums;
    private String scaledFor;
    private final List<String> rows = new ArrayList<>();
    private final long start = System.nanoTime();

    private Rotate(Settings settings, Device device, Safetensors source) {
        this.settings = settings;
        this.device = device;
        this.source = source;
        this.rotation = rotation(settings.seed());
    }

    /// Starts the rotation on the lattice and waits for it: the setup, then one frame per tensor, then the
    /// table.
    public static void run(Settings settings) throws Exception {
        try (Safetensors source = Safetensors.open(settings.model());
                Device device = new Device(settings.library());
                CalibrationLattice lattice = CalibrationLattice.start(2)) {
            var rotate = new Rotate(settings, device, source);
            try {
                lattice.submit(() -> rotate.begin())
                        .thenCompose(tensors -> {
                            CompletableFuture<Void> chain = CompletableFuture.completedFuture(null);
                            for (Safetensors.Tensor tensor : tensors)
                                chain = chain.thenCompose(ignored -> lattice.submit(() -> {
                                    rotate.tensor(tensor);
                                    return null;
                                }));
                            return chain.thenCompose(ignored -> lattice.submit(() -> {
                                rotate.finish();
                                return null;
                            }));
                        })
                        .get();
            } finally {
                lattice.submit(() -> {
                            rotate.close();
                            return null;
                        })
                        .get();
            }
        }
    }

    private void log(String message) {
        System.out.printf(Locale.ROOT, "[%7.0f s] %s%n", (System.nanoTime() - start) / 1e9, message);
    }

    /// Copies the checkpoint's other files and each shard's header; returns every tensor in shard and offset
    /// order.
    private List<Safetensors.Tensor> begin() {
        try {
            device.select(device.stream);
            ops = new Ops(device.cuda, device.stream, device.cublas);
            kurtosis = device.cuda
                    .compileResource("rotate.cu", List.of("group_kurtosis"))
                    .getFirst();
            Files.createDirectories(settings.out());
            List<Safetensors.Tensor> tensors = new ArrayList<>();
            try (var files = Files.list(settings.model())) {
                for (Path file : (Iterable<Path>) files::iterator) {
                    if (file.getFileName().toString().endsWith(".safetensors")) copyHeader(file);
                    else
                        Files.copy(
                                file,
                                settings.out().resolve(file.getFileName()),
                                java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                }
            }
            for (String name : names()) tensors.add(source.tensor(name));
            tensors.sort(
                    Comparator.comparing((Safetensors.Tensor t) -> t.shard().toString())
                            .thenComparingLong(Safetensors.Tensor::offset));
            long elements = (long) CHUNK_ROWS * HIDDEN;
            q = device.gpu.allocate(4L * HIDDEN * HIDDEN);
            scaledQ = device.gpu.allocate(4L * HIDDEN * HIDDEN);
            input = device.gpu.allocate(4 * elements);
            output = device.gpu.allocate(4 * elements);
            bf16 = device.gpu.allocate(2 * elements);
            sums = device.gpu.allocate(16);
            Upload.floats(device.gpu, q, rotation);
            log(tensors.size() + " tensors, rotation seed " + settings.seed());
            return tensors;
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }

    private List<String> names() throws IOException {
        var index = new com.fasterxml.jackson.databind.ObjectMapper()
                .readTree(
                        settings.model().resolve("model.safetensors.index.json").toFile())
                .get("weight_map");
        List<String> names = new ArrayList<>();
        index.fieldNames().forEachRemaining(names::add);
        return names;
    }

    private void copyHeader(Path shard) throws IOException {
        try (FileChannel in = FileChannel.open(shard, StandardOpenOption.READ);
                FileChannel out = FileChannel.open(
                        settings.out().resolve(shard.getFileName()),
                        StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE,
                        StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer size = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            in.read(size, 0);
            long header = 8 + size.getLong(0);
            in.transferTo(0, header, out);
        }
    }

    /// Writes `tensor` into its shard of the copy: rotated, folded, zeroed or as it is.
    private void tensor(Safetensors.Tensor tensor) {
        try {
            device.select(device.stream);
            Role role = role(tensor.name());
            if (role != Role.COPY && !tensor.dtype().equals("BF16"))
                throw new IllegalStateException(tensor.name() + " is " + tensor.dtype() + ", not BF16");
            Path target = settings.out().resolve(tensor.shard().getFileName());
            try (FileChannel out = FileChannel.open(target, StandardOpenOption.WRITE);
                    Arena arena = Arena.ofConfined()) {
                switch (role) {
                    case COPY -> {
                        try (FileChannel in = FileChannel.open(tensor.shard(), StandardOpenOption.READ)) {
                            long done = 0;
                            while (done < tensor.bytes())
                                done += in.transferTo(
                                        tensor.offset() + done,
                                        tensor.bytes() - done,
                                        out.position(tensor.offset() + done));
                        }
                    }
                    case NORM -> write(out, tensor.offset(), arena.allocate(tensor.bytes()));
                    case EMBEDDING, INPUT -> inputSide(tensor, role, out, arena);
                    case OUTPUT -> outputSide(tensor, out, arena);
                }
            }
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }

    /// Rows of `tensor` (columns: the hidden size) times Q, or diag(1 + w) Q for its norm, a chunk of rows at a
    /// time.
    private void inputSide(Safetensors.Tensor tensor, Role role, FileChannel out, Arena arena) throws IOException {
        if (tensor.shape().length != 2 || tensor.shape()[1] != HIDDEN)
            throw new IllegalStateException(tensor.name() + " does not read the hidden state");
        long factor = q;
        if (role == Role.INPUT) {
            scale(normOf(tensor.name()));
            factor = scaledQ;
        }
        long rowsTotal = tensor.shape()[0];
        double[] before = new double[2], after = new double[2];
        MemorySegment host = arena.allocate(2L * CHUNK_ROWS * HIDDEN, 16);
        for (long first = 0; first < rowsTotal; first += CHUNK_ROWS) {
            int rows = (int) Math.min(CHUNK_ROWS, rowsTotal - first);
            long bytes = 2L * rows * HIDDEN, count = (long) rows * HIDDEN;
            MemorySegment chunk = host.asSlice(0, bytes);
            source.read(tensor.name(), 2 * first * HIDDEN, chunk);
            device.gpu.copyHostToDevice(bf16, chunk, bytes);
            ops.bf16ToFloat(bf16, count, input);
            add(before, kurtosis(input, count));
            ops.product(input, HIDDEN, factor, HIDDEN, output, HIDDEN, rows, HIDDEN, HIDDEN, 1, 0);
            add(after, kurtosis(output, count));
            ops.floatToBf16(output, count, bf16);
            device.synchronize();
            device.gpu.copyDeviceToHost(chunk, bf16, bytes);
            write(out, tensor.offset() + 2 * first * HIDDEN, chunk);
        }
        record(tensor, role, before, after);
    }

    /// Q^T times `tensor` (rows: the hidden size), whole.
    private void outputSide(Safetensors.Tensor tensor, FileChannel out, Arena arena) throws IOException {
        if (tensor.shape().length != 2 || tensor.shape()[0] != HIDDEN)
            throw new IllegalStateException(tensor.name() + " does not write the hidden state");
        int columns = (int) tensor.shape()[1];
        long count = (long) HIDDEN * columns;
        if (count > (long) CHUNK_ROWS * HIDDEN) throw new IllegalStateException(tensor.name() + " is too wide");
        MemorySegment host = arena.allocate(2 * count, 16);
        source.read(tensor.name(), host);
        device.gpu.copyHostToDevice(bf16, host, 2 * count);
        ops.bf16ToFloat(bf16, count, input);
        double[] before = kurtosis(input, count);
        ops.productLeftTransposed(q, HIDDEN, input, columns, output, columns, HIDDEN, columns, HIDDEN, 1, 0);
        double[] after = kurtosis(output, count);
        ops.floatToBf16(output, count, bf16);
        device.synchronize();
        device.gpu.copyDeviceToHost(host, bf16, 2 * count);
        write(out, tensor.offset(), host);
        record(tensor, Role.OUTPUT, before, after);
    }

    /// diag(1 + w) Q into `scaledQ` for the unit-offset norm `norm` (kept while consecutive tensors share it).
    private void scale(String norm) throws IOException {
        if (norm.equals(scaledFor)) return;
        float[] weight;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment bf16 = arena.allocate(source.tensor(norm).bytes(), 16);
            source.read(norm, bf16);
            weight = new float[HIDDEN];
            for (int i = 0; i < HIDDEN; i++)
                weight[i] = Float.intBitsToFloat(bf16.getAtIndex(ValueLayout.JAVA_SHORT_UNALIGNED, i) << 16);
        }
        float[] scaled = new float[HIDDEN * HIDDEN];
        for (int i = 0; i < HIDDEN; i++) {
            float factor = 1 + weight[i];
            for (int j = 0; j < HIDDEN; j++) scaled[i * HIDDEN + j] = factor * rotation[i * HIDDEN + j];
        }
        Upload.floats(device.gpu, scaledQ, scaled);
        scaledFor = norm;
    }

    /// Sum of the groups' excess kurtosis and the number of groups.
    private double[] kurtosis(long values, long count) {
        ops.zero(sums, 4);
        long groups = count / GROUP;
        device.cuda.launch(
                kurtosis,
                (int) ((groups + 255) / 256),
                1,
                256,
                0,
                device.stream,
                new Cuda.Arguments()
                        .pointer(values)
                        .pointer(groups)
                        .integer(GROUP)
                        .pointer(sums));
        device.synchronize();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(16, 8);
            device.gpu.copyDeviceToHost(host, sums, 16);
            return new double[] {host.get(ValueLayout.JAVA_DOUBLE, 0), host.get(ValueLayout.JAVA_DOUBLE, 8)};
        }
    }

    private static void add(double[] total, double[] part) {
        total[0] += part[0];
        total[1] += part[1];
    }

    private void record(Safetensors.Tensor tensor, Role role, double[] before, double[] after) {
        rows.add(String.format(
                Locale.ROOT,
                "%s\t%s\t%.4f\t%.4f",
                tensor.name(),
                role.name().toLowerCase(Locale.ROOT),
                before[0] / before[1],
                after[0] / after[1]));
        if (tensor.name().endsWith("layers.63.mlp.down_proj.weight") || role == Role.EMBEDDING)
            log(tensor.name() + " rotated");
    }

    private static void write(FileChannel out, long position, MemorySegment data) throws IOException {
        long slice = 1L << 30;
        for (long at = 0; at < data.byteSize(); at += slice) {
            ByteBuffer buffer =
                    data.asSlice(at, Math.min(slice, data.byteSize() - at)).asByteBuffer();
            long written = position + at;
            while (buffer.hasRemaining()) written += out.write(buffer, written);
        }
    }

    private void finish() {
        try (Writer writer = Files.newBufferedWriter(settings.out().resolve("kurtosis.tsv"))) {
            writer.write("tensor\trole\tkurtosis_before\tkurtosis_after\n");
            for (String row : rows) writer.write(row + "\n");
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
        log("rotated checkpoint in " + settings.out());
    }

    private void close() {
        device.select(device.stream);
        for (long address : new long[] {q, scaledQ, input, output, bf16, sums})
            if (address != 0) device.gpu.free(address);
    }
}
