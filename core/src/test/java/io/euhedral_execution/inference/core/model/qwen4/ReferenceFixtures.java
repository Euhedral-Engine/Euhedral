package io.euhedral_execution.inference.core.model.qwen4;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/// The fixtures of tools/flash_next_reference.py: a directory with `manifest.json` (`tensors`: name to dtype,
/// shape and file) and one raw little-endian file per tensor.
final class ReferenceFixtures {

    private final Path directory;
    private final JsonNode tensors;
    private final JsonNode metadata;

    ReferenceFixtures(Path directory) throws IOException {
        this.directory = directory;
        JsonNode manifest = new ObjectMapper().readTree(Files.readAllBytes(directory.resolve("manifest.json")));
        this.tensors = manifest.get("tensors");
        this.metadata = manifest.get("metadata");
    }

    static boolean exists(Path directory) {
        return Files.isRegularFile(directory.resolve("manifest.json"));
    }

    JsonNode metadata() {
        return this.metadata;
    }

    boolean has(String name) {
        return this.tensors.has(name);
    }

    /// Names that start with `prefix`.
    List<String> names(String prefix) {
        List<String> names = new ArrayList<>();
        Iterator<String> all = this.tensors.fieldNames();
        while (all.hasNext()) {
            String name = all.next();
            if (name.startsWith(prefix)) names.add(name);
        }
        return names;
    }

    long[] shape(String name) {
        JsonNode shape = entry(name).get("shape");
        long[] result = new long[shape.size()];
        for (int i = 0; i < result.length; i++) result[i] = shape.get(i).asLong();
        return result;
    }

    int count(String name) {
        long count = 1;
        for (long extent : shape(name)) count *= extent;
        return Math.toIntExact(count);
    }

    private JsonNode entry(String name) {
        JsonNode entry = this.tensors.get(name);
        if (entry == null) throw new IllegalArgumentException("fixture " + this.directory + " has no tensor " + name);
        return entry;
    }

    private ByteBuffer read(String name, String dtype) throws IOException {
        JsonNode entry = entry(name);
        if (!entry.get("dtype").asText().equals(dtype))
            throw new IllegalArgumentException(
                    name + " is " + entry.get("dtype").asText() + ", not " + dtype);
        byte[] bytes =
                Files.readAllBytes(this.directory.resolve(entry.get("file").asText()));
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    short[] bf16(String name) throws IOException {
        ByteBuffer buffer = read(name, "bf16");
        short[] values = new short[buffer.remaining() / 2];
        buffer.asShortBuffer().get(values);
        return values;
    }

    float[] f32(String name) throws IOException {
        ByteBuffer buffer = read(name, "f32");
        float[] values = new float[buffer.remaining() / 4];
        buffer.asFloatBuffer().get(values);
        return values;
    }

    int[] i32(String name) throws IOException {
        ByteBuffer buffer = read(name, "i32");
        int[] values = new int[buffer.remaining() / 4];
        buffer.asIntBuffer().get(values);
        return values;
    }

    long[] i64(String name) throws IOException {
        ByteBuffer buffer = read(name, "i64");
        long[] values = new long[buffer.remaining() / 8];
        buffer.asLongBuffer().get(values);
        return values;
    }

    byte[] u8(String name) throws IOException {
        ByteBuffer buffer = read(name, "u8");
        byte[] values = new byte[buffer.remaining()];
        buffer.get(values);
        return values;
    }
}
