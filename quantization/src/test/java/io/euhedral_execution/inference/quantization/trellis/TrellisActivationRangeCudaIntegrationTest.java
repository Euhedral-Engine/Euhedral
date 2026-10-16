package io.euhedral_execution.inference.quantization.trellis;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import io.euhedral_execution.inference.quantization.checkpoint.Safetensors;
import io.euhedral_execution.inference.quantization.gpu.Device;
import io.euhedral_execution.inference.quantization.model.LayerForward;
import io.euhedral_execution.inference.quantization.model.LayerWeights;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/// The activations a trellis artifact's prepass meets: the BF16 forward of the checkpoint over calibration
/// sequences, every quantized linear's input rows multiplied by that linear's `suh` (from a trellis checkpoint) and
/// rotated by the 128-point Hadamard, as the engine's prepass does before it rounds to FP16. Records per linear
/// `max|x|`, `max|suh|` and the largest prepass output to `euhedral.trellis.extremes.report` (the fixture
/// `native/tests/test_trellis.py` reads), and fails when any output would not fit FP16.
///
/// Needs `euhedral.quantization.checkpoint` (BF16), `euhedral.trellis.checkpoint` (a finished trellis checkpoint)
/// and `euhedral.quantization.calibration-tokens`; `euhedral.trellis.extremes.sequences` (default 2) sequences run.
class TrellisActivationRangeCudaIntegrationTest {

    private static final float FP16_MAX = 65504f;
    private static final int BLOCK = 128;

    /// The extremes of one linear.
    record Extremes(String linear, float maxX, float maxSuh, float maxPrepass) {}

    @Test
    void prepassOutputsFitFp16() throws Exception {
        Path source = property("euhedral.quantization.checkpoint");
        Path trellis = property("euhedral.trellis.checkpoint");
        Path tokensFile = property("euhedral.quantization.calibration-tokens");
        assumeTrue(
                Files.isDirectory(source) && Files.isDirectory(trellis) && Files.isRegularFile(tokensFile),
                "needs the BF16 checkpoint, a trellis checkpoint and the calibration tokens");
        int sequences = Integer.getInteger("euhedral.trellis.extremes.sequences", 2);
        int[][] tokens = readTokens(tokensFile, sequences);
        int length = tokens[0].length, rows = sequences * length;
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        List<Extremes> found = new ArrayList<>();
        try (Safetensors store = Safetensors.open(source);
                Safetensors quantized = Safetensors.open(trellis);
                Device device = new Device(library);
                Arena arena = Arena.ofConfined()) {
            var objects = new Qwen38Objects(store, source);
            int hidden = objects.config.hidden();
            long bytes = 2L * rows * hidden;
            MemorySegment host = arena.allocate(bytes, 16);
            embed(store, tokens, hidden, host);
            long states = device.gpu.allocate(bytes);
            try (LayerForward forward = new LayerForward(device, objects.config, rows)) {
                device.gpu.copyHostToDevice(states, host, bytes);
                for (int index = 0; index < objects.config.layers(); index++) {
                    Map<String, List<Linears.Linear>> byObject = new HashMap<>();
                    Map<String, float[]> suhs = new HashMap<>();
                    for (Linears.Linear linear : Linears.layer(objects, index))
                        if (linear.quantized()) {
                            byObject.computeIfAbsent(linear.hessianObject(), o -> new ArrayList<>())
                                    .add(linear);
                            suhs.put(linear.tensor(), suh(quantized, linear));
                        }
                    try (LayerWeights layer = LayerWeights.layer(device.gpu, objects, index)) {
                        forward.forward(layer, states, sequences, length, (object, input, count, columns) -> {
                            List<Linears.Linear> readers = byObject.get(object);
                            if (readers == null) return;
                            device.synchronize();
                            float[] x = readBf16(device, input, (long) count * columns);
                            for (Linears.Linear linear : readers)
                                found.add(extremes(linear.tensor(), x, count, columns, suhs.get(linear.tensor())));
                        });
                    }
                    device.synchronize();
                }
            } finally {
                device.gpu.free(states);
            }
        }
        write(found);
        for (Extremes e : found)
            assertTrue(
                    Float.isFinite(e.maxPrepass()) && e.maxPrepass() < FP16_MAX,
                    e.linear() + ": prepass reaches " + e.maxPrepass());
    }

    /// The extremes of `x` (`count` rows of `k`) under `suh`: the largest `|x|`, `|suh|` and `|H128(x ∘ suh)|`.
    static Extremes extremes(String linear, float[] x, int count, int k, float[] suh) {
        float maxX = 0, maxSuh = 0, maxOut = 0;
        for (float s : suh) maxSuh = Math.max(maxSuh, Math.abs(s));
        float[] block = new float[BLOCK];
        float scale = (float) (1 / Math.sqrt(BLOCK));
        for (int r = 0; r < count; r++) {
            for (int b = 0; b < k; b += BLOCK) {
                for (int i = 0; i < BLOCK; i++) {
                    float v = x[r * k + b + i];
                    maxX = Math.max(maxX, Math.abs(v));
                    block[i] = v * suh[b + i];
                }
                hadamard(block);
                for (float v : block) maxOut = Math.max(maxOut, Math.abs(v * scale));
            }
        }
        return new Extremes(linear, maxX, maxSuh, maxOut);
    }

    /// The unnormalised fast Walsh-Hadamard transform, in place.
    static void hadamard(float[] v) {
        for (int h = 1; h < v.length; h <<= 1)
            for (int i = 0; i < v.length; i += h << 1)
                for (int j = i; j < i + h; j++) {
                    float a = v[j], b = v[j + h];
                    v[j] = a + b;
                    v[j + h] = a - b;
                }
    }

    private static float[] suh(Safetensors trellis, Linears.Linear linear) throws IOException {
        String name = linear.tensor() + ".suh";
        long count = trellis.tensor(name).bytes() / 2;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment bits = arena.allocate(2 * count, 2);
            trellis.read(name, bits);
            float[] values = new float[(int) count];
            for (int i = 0; i < count; i++)
                values[i] = Float.float16ToFloat(bits.getAtIndex(ValueLayout.JAVA_SHORT_UNALIGNED, i));
            return values;
        }
    }

    private static float[] readBf16(Device device, long address, long count) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(2 * count, 2);
            device.gpu.copyDeviceToHost(host, address, 2 * count);
            float[] values = new float[(int) count];
            for (int i = 0; i < count; i++)
                values[i] = Float.intBitsToFloat(host.getAtIndex(ValueLayout.JAVA_SHORT, i) << 16);
            return values;
        }
    }

    /// The token embeddings of every sequence, read row by row from the checkpoint.
    private static void embed(Safetensors store, int[][] tokens, int hidden, MemorySegment host) throws IOException {
        String table = "model.language_model.embed_tokens.weight";
        long row = 2L * hidden, at = 0;
        for (int[] sequence : tokens)
            for (int token : sequence) {
                store.read(table, token * row, host.asSlice(at, row));
                at += row;
            }
    }

    private static int[][] readTokens(Path path, int limit) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int headerLength = Short.toUnsignedInt(buffer.getShort(8));
        String header = new String(bytes, 10, headerLength, StandardCharsets.US_ASCII);
        var shape = Pattern.compile("\\((\\d+), (\\d+)\\)").matcher(header);
        if (!shape.find()) throw new IOException(path + ": no 2-D shape");
        int sequences = Math.min(limit, Integer.parseInt(shape.group(1))), length = Integer.parseInt(shape.group(2));
        var ints = ByteBuffer.wrap(bytes, 10 + headerLength, bytes.length - 10 - headerLength)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asIntBuffer();
        int[][] tokens = new int[sequences][length];
        for (int[] sequence : tokens) ints.get(sequence);
        return tokens;
    }

    private static void write(List<Extremes> found) throws IOException {
        String report = System.getProperty("euhedral.trellis.extremes.report");
        if (report == null) return;
        StringBuilder json = new StringBuilder("{\"cases\": [\n");
        for (int i = 0; i < found.size(); i++) {
            Extremes e = found.get(i);
            json.append(String.format(
                    Locale.ROOT,
                    "  {\"linear\": \"%s\", \"max_abs_x\": %.6g, \"max_abs_suh\": %.6g, \"max_abs_prepass\": %.6g}%s%n",
                    e.linear(),
                    e.maxX(),
                    e.maxSuh(),
                    e.maxPrepass(),
                    i + 1 < found.size() ? "," : ""));
        }
        json.append("]}\n");
        Path path = Path.of(report);
        Files.createDirectories(path.toAbsolutePath().getParent());
        Files.writeString(path, json);
    }

    private static Path property(String name) {
        return Path.of(System.getProperty(name, "unset-" + name));
    }
}
