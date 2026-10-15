package io.euhedral_execution.inference.quantization.checkpoint;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/// Writes one safetensors file: tensors are registered with [#add] and written, in insertion order, by [#close]. The
/// header (`{"__metadata__":{"format":"pt"}, name: {dtype, shape, data_offsets}}`, padded with spaces to a multiple
/// of 8 bytes) needs every tensor's size, so the data is pulled from each tensor's [Source] in chunks only when the
/// file is closed; a tensor never has to be resident whole. The file is written as `<name>.part` and renamed once
/// complete, so an interrupted run never leaves a truncated file under the final name.
public final class SafetensorsWriter implements AutoCloseable {

    /// Supplies a tensor's bytes: fills `destination` with the bytes `[offset, offset + destination.byteSize())`.
    @FunctionalInterface
    public interface Source {

        void read(long offset, MemorySegment destination) throws IOException;
    }

    private static final Map<String, Integer> ELEMENT_BYTES =
            Map.of("I16", 2, "F16", 2, "BF16", 2, "F32", 4, "I32", 4, "I64", 8, "U8", 1, "I8", 1);
    private static final int CHUNK = 16 << 20;
    private static final ObjectMapper JSON = new ObjectMapper();

    private record Entry(String name, String dtype, long[] shape, long bytes, Source source) {}

    private final Path file;
    private final List<Entry> entries = new ArrayList<>();
    private final Set<String> names = new HashSet<>();
    private long dataBytes;
    private boolean closed;

    public SafetensorsWriter(Path file) {
        this.file = file;
    }

    /// Adds a tensor held in `data`, which must stay valid until [#close].
    public void add(String name, String dtype, long[] shape, MemorySegment data) {
        add(
                name,
                dtype,
                shape,
                data.byteSize(),
                (offset, destination) -> MemorySegment.copy(data, offset, destination, 0, destination.byteSize()));
    }

    /// Adds a tensor of `bytes` bytes that `source` produces on demand.
    public void add(String name, String dtype, long[] shape, long bytes, Source source) {
        Integer width = ELEMENT_BYTES.get(dtype);
        if (width == null) throw new IllegalArgumentException(name + ": unsupported dtype " + dtype);
        long elements = 1;
        for (long dimension : shape) elements *= dimension;
        if (elements * width != bytes)
            throw new IllegalArgumentException(name + ": shape implies " + elements * width + " bytes, got " + bytes);
        if (!names.add(name)) throw new IllegalArgumentException("duplicate tensor " + name);
        entries.add(new Entry(name, dtype, shape.clone(), bytes, source));
        dataBytes += bytes;
    }

    /// The total size of the tensor data added so far.
    public long dataBytes() {
        return dataBytes;
    }

    @Override
    public void close() throws IOException {
        if (closed) return;
        closed = true;
        ObjectNode header = JSON.createObjectNode();
        header.putObject("__metadata__").put("format", "pt");
        long offset = 0;
        for (Entry entry : entries) {
            ObjectNode node = header.putObject(entry.name());
            node.put("dtype", entry.dtype());
            ArrayNode shape = node.putArray("shape");
            for (long dimension : entry.shape()) shape.add(dimension);
            node.putArray("data_offsets").add(offset).add(offset + entry.bytes());
            offset += entry.bytes();
        }
        byte[] encoded = JSON.writeValueAsString(header).getBytes(StandardCharsets.UTF_8);
        byte[] json = java.util.Arrays.copyOf(encoded, (encoded.length + 7) / 8 * 8);
        java.util.Arrays.fill(json, encoded.length, json.length, (byte) ' ');
        Path part = file.resolveSibling(file.getFileName() + ".part");
        try {
            try (FileChannel out = FileChannel.open(
                            part,
                            StandardOpenOption.CREATE,
                            StandardOpenOption.TRUNCATE_EXISTING,
                            StandardOpenOption.WRITE);
                    Arena arena = Arena.ofConfined()) {
                ByteBuffer size = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
                size.putLong(json.length).flip();
                writeFully(out, size);
                writeFully(out, ByteBuffer.wrap(json));
                MemorySegment chunk = arena.allocate(CHUNK, 8);
                for (Entry entry : entries) {
                    for (long at = 0; at < entry.bytes(); at += CHUNK) {
                        MemorySegment piece = chunk.asSlice(0, Math.min(CHUNK, entry.bytes() - at));
                        entry.source().read(at, piece);
                        writeFully(out, piece.asByteBuffer());
                    }
                }
                out.force(true);
            }
            Files.move(part, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException | RuntimeException | Error e) {
            Files.deleteIfExists(part);
            throw e;
        }
    }

    private static void writeFully(FileChannel out, ByteBuffer buffer) throws IOException {
        while (buffer.hasRemaining()) out.write(buffer);
    }
}
