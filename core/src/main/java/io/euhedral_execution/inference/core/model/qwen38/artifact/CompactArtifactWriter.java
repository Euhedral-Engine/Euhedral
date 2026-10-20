package io.euhedral_execution.inference.core.model.qwen38.artifact;

import io.euhedral_execution.inference.core.artifact.ArtifactFileAccess;
import io.euhedral_execution.inference.core.artifact.ArtifactFormatException;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TokenizerSection;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.tokenizer.TokenizerFiles;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/// Writes compact version 2 EDRL metadata, descriptors, the tokenizer section, and small test payloads.
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

    /// Writes `artifact` with `tensorData` as its payloads and `tokenizer` as the tokenizer it carries.
    public static void write(Path path, Artifact artifact, byte[][] tensorData, TokenizerFiles tokenizer)
            throws IOException {
        if (tensorData == null) {
            throw new ArtifactFormatException("compact writer arguments are incomplete");
        }
        TensorDescriptor[] tensors = artifact == null ? null : artifact.tensors();
        if (tensors == null || tensorData.length != tensors.length) {
            throw new ArtifactFormatException("compact payload count does not match descriptors");
        }
        for (int i = 0; i < tensors.length; i++) {
            if (tensorData[i] == null || tensorData[i].length != tensors[i].byteSize()) {
                throw new ArtifactFormatException("compact payload size mismatch: " + tensors[i].name());
            }
        }
        try (FileChannel channel = writeLayout(path, artifact, tokenizer)) {
            for (int i = 0; i < tensors.length; i++) {
                if (tensorData[i].length != 0) {
                    writeAt(channel, tensors[i].dataOffset(), ByteBuffer.wrap(tensorData[i]));
                }
            }
        }
    }

    /// Creates `path` with `artifact`'s header, metadata and tensor table, sized to hold every payload, which the
    /// caller writes at the descriptors' offsets (a large payload in chunks: nothing here holds one), and with
    /// `tokenizer` after the payloads ([TokenizerSection]). The file is returned open for writing.
    public static FileChannel writeLayout(Path path, Artifact artifact, TokenizerFiles tokenizer) throws IOException {
        if (path == null || artifact == null || artifact.header() == null || tokenizer == null) {
            throw new ArtifactFormatException("compact writer arguments are incomplete");
        }
        if (artifact.header().version() != ArtifactHeader.COMPACT_VERSION) {
            throw new ArtifactFormatException("compact writer requires version 2 header");
        }
        TensorDescriptor[] tensors = artifact.tensors();
        if (tensors == null) {
            throw new ArtifactFormatException("compact descriptors are missing");
        }
        byte[] metadata = ArtifactCodec.encodeMetadata(artifact.config());
        byte[] table = ArtifactCodec.encodeTensorTable(tensors);
        long expectedMetadataEnd = ArtifactFileAccess.checkedEnd(ArtifactHeader.BYTE_SIZE, metadata.length, "metadata");
        if (artifact.header().metadataOffset() != ArtifactHeader.BYTE_SIZE
                || artifact.header().metadataSize() != metadata.length
                || artifact.header().tensorTableOffset() != expectedMetadataEnd
                || artifact.header().tensorDataOffset()
                        != ArtifactFileAccess.checkedEnd(artifact.header().tensorTableOffset(), table.length, "table")
                || artifact.header().tensorCount() != tensors.length) {
            throw new ArtifactFormatException("compact header does not match encoded metadata and table");
        }

        long fileSize = artifact.header().tensorDataOffset();
        for (TensorDescriptor descriptor : tensors) {
            if (descriptor.dataOffset() < artifact.header().tensorDataOffset()) {
                throw new ArtifactFormatException("compact payload precedes data section");
            }
            fileSize = Math.max(
                    fileSize,
                    ArtifactFileAccess.checkedEnd(descriptor.dataOffset(), descriptor.byteSize(), "tensor data"));
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

        FileChannel channel = FileChannel.open(
                path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING);
        try {
            if (fileSize > 0) {
                channel.position(fileSize - 1);
                channel.write(ByteBuffer.wrap(new byte[] {0}));
            }
            writeAt(channel, 0, header);
            writeAt(channel, artifact.header().metadataOffset(), ByteBuffer.wrap(metadata));
            writeAt(channel, artifact.header().tensorTableOffset(), ByteBuffer.wrap(table));
            TokenizerSection.write(channel, fileSize, tokenizer);
            return channel;
        } catch (IOException | RuntimeException failure) {
            channel.close();
            throw failure;
        }
    }

    private static void writeAt(FileChannel channel, long offset, ByteBuffer bytes) throws IOException {
        channel.position(offset);
        while (bytes.hasRemaining()) {
            channel.write(bytes);
        }
    }
}
