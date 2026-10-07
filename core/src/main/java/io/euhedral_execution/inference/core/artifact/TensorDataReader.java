package io.euhedral_execution.inference.core.artifact;

import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactCodec;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/// Reads an unmodified tensor payload from a Qwen artifact file.
public final class TensorDataReader {

    private TensorDataReader() {}

    public static MemorySegment read(Path path, TensorDescriptor tensor, Arena arena) throws IOException {
        if (path == null) {
            throw new ArtifactFormatException("input path is null");
        }
        if (tensor == null) {
            throw new ArtifactFormatException("tensor descriptor is null");
        }
        if (arena == null) {
            throw new ArtifactFormatException("arena is null");
        }
        long payloadEnd = ArtifactCodec.checkedEnd(tensor.dataOffset(), tensor.byteSize(), "tensor data");

        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            if (payloadEnd > channel.size()) {
                throw new ArtifactFormatException("tensor payload extends beyond the file: " + tensor.name());
            }
            MemorySegment payload = arena.allocate(tensor.byteSize());
            ArtifactFileAccess.readFully(channel, tensor.dataOffset(), payload, "tensor payload");
            return payload;
        }
    }
}
