package io.euhedral_execution.inference.core.model.qwen38.artifact;

import io.euhedral_execution.inference.core.artifact.ArtifactFormatException;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/// Writes compact version 2 EDRL metadata, descriptors, and small test payloads.
///
/// The production converter writes large payloads directly in deterministic descriptor order; this
/// writer intentionally accepts byte arrays so compact format round-trip tests do not allocate a
/// model-sized Java heap object.
public final class CompactArtifactWriter {

    private CompactArtifactWriter() {}

    /// The encoded size of the metadata section for `config`.
    public static long metadataSize(Qwen38Config config) {
        try {
            return ArtifactCodec.encodeMetadata(config).length;
        } catch (ArtifactFormatException exception) {
            throw new IllegalArgumentException(exception.getMessage(), exception);
        }
    }

    /// The encoded size of the tensor table for `tensors`.
    public static long tensorTableSize(TensorDescriptor[] tensors) {
        try {
            return ArtifactCodec.encodeTensorTable(tensors).length;
        } catch (ArtifactFormatException exception) {
            throw new IllegalArgumentException(exception.getMessage(), exception);
        }
    }

    public static void write(Path path, Artifact artifact, byte[][] tensorData) throws IOException {
        if (path == null || artifact == null || artifact.header() == null || tensorData == null) {
            throw new ArtifactFormatException("compact writer arguments are incomplete");
        }
        if (artifact.header().version() != ArtifactHeader.COMPACT_VERSION) {
            throw new ArtifactFormatException("compact writer requires version 2 header");
        }
        TensorDescriptor[] tensors = artifact.tensors();
        if (tensors == null || tensorData.length != tensors.length) {
            throw new ArtifactFormatException("compact payload count does not match descriptors");
        }
        byte[] metadata = ArtifactCodec.encodeMetadata(artifact.config());
        byte[] table = ArtifactCodec.encodeTensorTable(tensors);
        long expectedMetadataEnd = ArtifactCodec.checkedEnd(ArtifactHeader.BYTE_SIZE, metadata.length, "metadata");
        if (artifact.header().metadataOffset() != ArtifactHeader.BYTE_SIZE
                || artifact.header().metadataSize() != metadata.length
                || artifact.header().tensorTableOffset() != expectedMetadataEnd
                || artifact.header().tensorDataOffset()
                        != ArtifactCodec.checkedEnd(artifact.header().tensorTableOffset(), table.length, "table")
                || artifact.header().tensorCount() != tensors.length) {
            throw new ArtifactFormatException("compact header does not match encoded metadata and table");
        }

        long fileSize = artifact.header().tensorDataOffset();
        for (int i = 0; i < tensors.length; i++) {
            TensorDescriptor descriptor = tensors[i];
            byte[] payload = tensorData[i];
            if (payload == null || payload.length != descriptor.byteSize()) {
                throw new ArtifactFormatException("compact payload size mismatch: " + descriptor.name());
            }
            if (descriptor.dataOffset() < artifact.header().tensorDataOffset()) {
                throw new ArtifactFormatException("compact payload precedes data section");
            }
            fileSize = Math.max(
                    fileSize, ArtifactCodec.checkedEnd(descriptor.dataOffset(), descriptor.byteSize(), "tensor data"));
        }

        ByteBuffer header = ByteBuffer.allocate(ArtifactHeader.BYTE_SIZE).order(ArtifactCodec.BYTE_ORDER);
        header.putInt(artifact.header().magic());
        header.putInt(ArtifactHeader.COMPACT_VERSION);
        header.putLong(artifact.header().metadataOffset());
        header.putLong(artifact.header().metadataSize());
        header.putLong(artifact.header().tensorTableOffset());
        header.putInt(artifact.header().tensorCount());
        header.putInt(0);
        header.putLong(artifact.header().tensorDataOffset());
        header.flip();

        try (FileChannel channel = FileChannel.open(
                path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            if (fileSize > 0) {
                channel.position(fileSize - 1);
                channel.write(ByteBuffer.wrap(new byte[] {0}));
            }
            writeAt(channel, 0, header);
            writeAt(channel, artifact.header().metadataOffset(), ByteBuffer.wrap(metadata));
            writeAt(channel, artifact.header().tensorTableOffset(), ByteBuffer.wrap(table));
            for (int i = 0; i < tensors.length; i++) {
                if (tensorData[i].length != 0) {
                    writeAt(channel, tensors[i].dataOffset(), ByteBuffer.wrap(tensorData[i]));
                }
            }
        }
    }

    private static void writeAt(FileChannel channel, long offset, ByteBuffer bytes) throws IOException {
        channel.position(offset);
        while (bytes.hasRemaining()) {
            channel.write(bytes);
        }
    }
}
