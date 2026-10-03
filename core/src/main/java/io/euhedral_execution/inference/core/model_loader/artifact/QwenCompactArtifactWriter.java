package io.euhedral_execution.inference.core.model_loader.artifact;

import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
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
public final class QwenCompactArtifactWriter {

    private QwenCompactArtifactWriter() {}

    /// The encoded size of the metadata section for `config`.
    public static long metadataSize(QwenConfig config) {
        try {
            return QwenArtifactCodec.encodeMetadata(config).length;
        } catch (QwenArtifactFormatException exception) {
            throw new IllegalArgumentException(exception.getMessage(), exception);
        }
    }

    /// The encoded size of the tensor table for `tensors`.
    public static long tensorTableSize(TensorDescriptor[] tensors) {
        try {
            return QwenArtifactCodec.encodeTensorTable(tensors).length;
        } catch (QwenArtifactFormatException exception) {
            throw new IllegalArgumentException(exception.getMessage(), exception);
        }
    }

    public static void write(Path path, QwenArtifact artifact, byte[][] tensorData) throws IOException {
        if (path == null || artifact == null || artifact.header() == null || tensorData == null) {
            throw new QwenArtifactFormatException("compact writer arguments are incomplete");
        }
        if (artifact.header().version() != QwenArtifactHeader.COMPACT_VERSION) {
            throw new QwenArtifactFormatException("compact writer requires version 2 header");
        }
        TensorDescriptor[] tensors = artifact.tensors();
        if (tensors == null || tensorData.length != tensors.length) {
            throw new QwenArtifactFormatException("compact payload count does not match descriptors");
        }
        byte[] metadata = QwenArtifactCodec.encodeMetadata(artifact.config());
        byte[] table = QwenArtifactCodec.encodeTensorTable(tensors);
        long expectedMetadataEnd =
                QwenArtifactCodec.checkedEnd(QwenArtifactHeader.BYTE_SIZE, metadata.length, "metadata");
        if (artifact.header().metadataOffset() != QwenArtifactHeader.BYTE_SIZE
                || artifact.header().metadataSize() != metadata.length
                || artifact.header().tensorTableOffset() != expectedMetadataEnd
                || artifact.header().tensorDataOffset()
                        != QwenArtifactCodec.checkedEnd(artifact.header().tensorTableOffset(), table.length, "table")
                || artifact.header().tensorCount() != tensors.length) {
            throw new QwenArtifactFormatException("compact header does not match encoded metadata and table");
        }

        long fileSize = artifact.header().tensorDataOffset();
        for (int i = 0; i < tensors.length; i++) {
            TensorDescriptor descriptor = tensors[i];
            byte[] payload = tensorData[i];
            if (payload == null || payload.length != descriptor.byteSize()) {
                throw new QwenArtifactFormatException("compact payload size mismatch: " + descriptor.name());
            }
            if (descriptor.dataOffset() < artifact.header().tensorDataOffset()) {
                throw new QwenArtifactFormatException("compact payload precedes data section");
            }
            fileSize = Math.max(
                    fileSize,
                    QwenArtifactCodec.checkedEnd(descriptor.dataOffset(), descriptor.byteSize(), "tensor data"));
        }

        ByteBuffer header = ByteBuffer.allocate(QwenArtifactHeader.BYTE_SIZE).order(QwenArtifactCodec.BYTE_ORDER);
        header.putInt(artifact.header().magic());
        header.putInt(QwenArtifactHeader.COMPACT_VERSION);
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
