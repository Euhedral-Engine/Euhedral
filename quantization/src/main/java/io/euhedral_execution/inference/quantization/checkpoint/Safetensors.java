package io.euhedral_execution.inference.quantization.checkpoint;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/// The tensors of a Hugging Face checkpoint directory: `model.safetensors.index.json` and its shards, or a single
/// `model.safetensors`. Tensors are read with positioned reads, never memory-mapped: the checkpoints live on a shared
/// exFAT drive, where page faults on large mappings hung the kernel.
public final class Safetensors implements AutoCloseable {

    /// One tensor: its data type (`BF16`, `F32`, ...), shape, and byte range in its shard.
    public record Tensor(String name, String dtype, long[] shape, Path shard, long offset, long bytes) {

        public long elements() {
            long count = 1;
            for (long dimension : shape) count *= dimension;
            return count;
        }
    }

    private static final ObjectMapper JSON = new ObjectMapper();

    private final Map<String, Tensor> tensors = new HashMap<>();
    private final Map<Path, FileChannel> channels = new HashMap<>();

    private Safetensors() {}

    public static Safetensors open(Path directory) throws IOException {
        Safetensors store = new Safetensors();
        Path index = directory.resolve("model.safetensors.index.json");
        List<Path> shards;
        if (Files.isRegularFile(index)) {
            JsonNode weightMap = JSON.readTree(index.toFile()).get("weight_map");
            var names = new java.util.TreeSet<String>();
            for (Iterator<JsonNode> it = weightMap.elements(); it.hasNext(); )
                names.add(it.next().asText());
            shards = names.stream().map(directory::resolve).toList();
        } else {
            shards = List.of(directory.resolve("model.safetensors"));
        }
        for (Path shard : shards) store.readHeader(shard);
        return store;
    }

    private void readHeader(Path shard) throws IOException {
        FileChannel channel = FileChannel.open(shard, StandardOpenOption.READ);
        channels.put(shard, channel);
        ByteBuffer size = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        readFully(channel, size, 0);
        long headerBytes = size.getLong(0);
        if (headerBytes <= 0 || headerBytes > (1L << 30))
            throw new IOException(shard + ": bad header size " + headerBytes);
        ByteBuffer header = ByteBuffer.allocate((int) headerBytes);
        readFully(channel, header, 8);
        JsonNode root = JSON.readTree(new String(header.array(), StandardCharsets.UTF_8));
        long data = 8 + headerBytes;
        for (Iterator<Map.Entry<String, JsonNode>> it = root.fields(); it.hasNext(); ) {
            var entry = it.next();
            if (entry.getKey().equals("__metadata__")) continue;
            JsonNode node = entry.getValue();
            JsonNode shapeNode = node.get("shape");
            long[] shape = new long[shapeNode.size()];
            for (int i = 0; i < shape.length; i++) shape[i] = shapeNode.get(i).asLong();
            long begin = node.get("data_offsets").get(0).asLong();
            long end = node.get("data_offsets").get(1).asLong();
            tensors.put(
                    entry.getKey(),
                    new Tensor(entry.getKey(), node.get("dtype").asText(), shape, shard, data + begin, end - begin));
        }
    }

    public Tensor tensor(String name) {
        Tensor tensor = tensors.get(name);
        if (tensor == null) throw new IllegalArgumentException("checkpoint has no tensor " + name);
        return tensor;
    }

    /// The names of all tensors.
    public java.util.Set<String> names() {
        return java.util.Collections.unmodifiableSet(tensors.keySet());
    }

    public boolean contains(String name) {
        return tensors.containsKey(name);
    }

    /// Reads `bytes` bytes of tensor `name` starting `skip` bytes into it, into `destination` (a host segment).
    public void read(String name, long skip, MemorySegment destination) throws IOException {
        Tensor tensor = tensor(name);
        if (skip < 0 || skip + destination.byteSize() > tensor.bytes()) {
            throw new IllegalArgumentException(
                    name + ": range " + skip + "+" + destination.byteSize() + " outside " + tensor.bytes() + " bytes");
        }
        long slice = 1L << 30; // a ByteBuffer covers at most 2 GiB
        for (long at = 0; at < destination.byteSize(); at += slice) {
            long bytes = Math.min(slice, destination.byteSize() - at);
            readFully(
                    channels.get(tensor.shard()),
                    destination.asSlice(at, bytes).asByteBuffer(),
                    tensor.offset() + skip + at);
        }
    }

    /// Reads all of tensor `name` into `destination`, which must be exactly its size.
    public void read(String name, MemorySegment destination) throws IOException {
        Tensor tensor = tensor(name);
        if (destination.byteSize() != tensor.bytes()) {
            throw new IllegalArgumentException(
                    name + " holds " + tensor.bytes() + " bytes, not " + destination.byteSize());
        }
        read(name, 0, destination);
    }

    private static void readFully(FileChannel channel, ByteBuffer buffer, long position) throws IOException {
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, position);
            if (read < 0) throw new IOException("unexpected end of file at " + position);
            position += read;
        }
    }

    @Override
    public void close() throws IOException {
        for (FileChannel channel : channels.values()) channel.close();
        channels.clear();
    }
}
