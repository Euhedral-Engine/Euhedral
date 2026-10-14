package io.euhedral_execution.inference.quantization.checkpoint;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/// A checkpoint directory that reads as `source` with some 1-D tensors replaced: every file of `source` linked, the
/// replacements (BF16) in `trained.safetensors`, and the index's weight map sending their names there. Readers that
/// follow the index (the converter's SourceStore, [Safetensors]) see the replacements; the source's own copies are
/// never read.
public final class Overlay {

    static final String SHARD = "trained.safetensors";
    private static final String INDEX = "model.safetensors.index.json";

    private Overlay() {}

    public static void write(Path source, Path target, Map<String, float[]> tensors) throws IOException {
        Files.createDirectories(target);
        try (var files = Files.list(source)) {
            for (Path file : (Iterable<Path>) files::iterator) {
                String name = file.getFileName().toString();
                Path link = target.resolve(name);
                if (name.equals(INDEX)
                        || name.equals(SHARD)
                        || Files.exists(link, java.nio.file.LinkOption.NOFOLLOW_LINKS)) continue;
                Files.createSymbolicLink(link, file.toAbsolutePath());
            }
        }
        writeShard(target.resolve(SHARD), tensors);
        ObjectMapper json = new ObjectMapper();
        ObjectNode index = (ObjectNode) json.readTree(source.resolve(INDEX).toFile());
        ObjectNode weightMap = (ObjectNode) index.get("weight_map");
        for (String name : tensors.keySet()) {
            if (!weightMap.has(name)) throw new IllegalArgumentException(name + " is not a tensor of " + source);
            weightMap.put(name, SHARD);
        }
        Files.deleteIfExists(target.resolve(INDEX));
        json.writerWithDefaultPrettyPrinter().writeValue(target.resolve(INDEX).toFile(), index);
    }

    /// A safetensors file of 1-D BF16 tensors (rounded to nearest even).
    static void writeShard(Path path, Map<String, float[]> tensors) throws IOException {
        StringBuilder header = new StringBuilder("{");
        long offset = 0;
        for (var entry : tensors.entrySet()) {
            long bytes = 2L * entry.getValue().length;
            if (header.length() > 1) header.append(',');
            header.append('"')
                    .append(entry.getKey())
                    .append("\":{\"dtype\":\"BF16\",\"shape\":[")
                    .append(entry.getValue().length)
                    .append("],\"data_offsets\":[")
                    .append(offset)
                    .append(',')
                    .append(offset + bytes)
                    .append("]}");
            offset += bytes;
        }
        header.append('}');
        while ((header.length() + 8) % 8 != 0) header.append(' ');
        byte[] headerBytes = header.toString().getBytes(StandardCharsets.UTF_8);
        ByteBuffer out =
                ByteBuffer.allocate(8 + headerBytes.length + (int) offset).order(ByteOrder.LITTLE_ENDIAN);
        out.putLong(headerBytes.length).put(headerBytes);
        for (float[] values : tensors.values()) {
            for (float value : values) {
                int bits = Float.floatToRawIntBits(value);
                out.putShort((short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16));
            }
        }
        Files.deleteIfExists(path);
        Files.write(path, out.array());
    }
}
