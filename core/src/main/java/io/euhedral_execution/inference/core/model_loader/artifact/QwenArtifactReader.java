package io.euhedral_execution.inference.core.model_loader.artifact;

import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/// Reads and validates compact version 2 EDRL artifacts.
public final class QwenArtifactReader {

    private QwenArtifactReader() {}

    public static QwenArtifact read(Path path) throws IOException {
        if (path == null) {
            throw new QwenArtifactFormatException("input path is null");
        }
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            if (fileSize < QwenArtifactHeader.BYTE_SIZE) {
                throw new QwenArtifactFormatException("artifact is truncated before the header");
            }

            QwenArtifactHeader header = readHeader(channel);
            validateHeader(header, fileSize);
            byte[] metadata = readRegion(channel, header.metadataOffset(), header.metadataSize(), "metadata");
            var config = QwenArtifactCodec.decodeMetadata(metadata);
            TensorDescriptor[] tensors = readTensorTable(channel, header, fileSize);
            return new QwenArtifact(header, config, tensors);
        }
    }

    private static QwenArtifactHeader readHeader(FileChannel channel) throws IOException {
        ByteBuffer bytes = ByteBuffer.allocate(QwenArtifactHeader.BYTE_SIZE).order(QwenArtifactCodec.BYTE_ORDER);
        ArtifactFileAccess.readFully(channel, 0, bytes, "header");
        bytes.flip();
        int magic = bytes.getInt();
        int version = bytes.getInt();
        long metadataOffset = bytes.getLong();
        long metadataSize = bytes.getLong();
        long tensorTableOffset = bytes.getLong();
        int tensorCount = bytes.getInt();
        int reserved = bytes.getInt();
        long tensorDataOffset = bytes.getLong();
        if (reserved != 0) {
            throw new QwenArtifactFormatException("header reserved field must be zero");
        }
        return new QwenArtifactHeader(
                magic, version, metadataOffset, metadataSize, tensorTableOffset, tensorCount, tensorDataOffset);
    }

    private static void validateHeader(QwenArtifactHeader header, long fileSize) throws QwenArtifactFormatException {
        if (header.magic() != QwenArtifactHeader.MAGIC) {
            throw new QwenArtifactFormatException("unsupported artifact magic");
        }
        if (header.version() != QwenArtifactHeader.COMPACT_VERSION) {
            throw new QwenArtifactFormatException("unsupported artifact version " + header.version()
                    + ": only compact version " + QwenArtifactHeader.COMPACT_VERSION + " artifacts are supported");
        }
        if (header.tensorCount() < 0 || header.tensorCount() > QwenArtifactCodec.MAX_COUNT) {
            throw new QwenArtifactFormatException("tensor count is outside the supported range");
        }
        if (header.metadataOffset() != QwenArtifactHeader.BYTE_SIZE) {
            throw new QwenArtifactFormatException("metadata must immediately follow the header");
        }
        long metadataEnd = QwenArtifactCodec.checkedEnd(header.metadataOffset(), header.metadataSize(), "metadata");
        if (metadataEnd > fileSize) {
            throw new QwenArtifactFormatException("metadata extends beyond the file");
        }
        if (header.tensorTableOffset() != metadataEnd) {
            throw new QwenArtifactFormatException("tensor table must immediately follow metadata");
        }
        if (header.tensorTableOffset() > fileSize) {
            throw new QwenArtifactFormatException("tensor table starts beyond the file");
        }
        if (header.tensorDataOffset() < header.tensorTableOffset() || header.tensorDataOffset() > fileSize) {
            throw new QwenArtifactFormatException("tensor data offset is outside the file");
        }
        if (header.metadataSize() > Integer.MAX_VALUE) {
            throw new QwenArtifactFormatException("metadata is too large");
        }
        if (header.metadataSize() > QwenArtifactCodec.MAX_METADATA_BYTES) {
            throw new QwenArtifactFormatException("metadata is too large");
        }
    }

    private static TensorDescriptor[] readTensorTable(FileChannel channel, QwenArtifactHeader header, long fileSize)
            throws IOException {
        TableCursor cursor = new TableCursor(channel, header.tensorTableOffset(), header.tensorDataOffset());
        TensorDescriptor[] tensors = new TensorDescriptor[header.tensorCount()];
        for (int i = 0; i < tensors.length; i++) {
            int nameLength = cursor.readInt("tensor name length");
            if (nameLength <= 0 || nameLength > QwenArtifactCodec.MAX_STRING_BYTES) {
                throw new QwenArtifactFormatException(
                        "tensor name length is outside the supported range: " + nameLength);
            }
            String name = QwenArtifactCodec.decodeUtf8(cursor.readBytes(nameLength, "tensor name"), "tensor name");
            int rank = cursor.readInt("tensor rank");
            if (rank < 0 || rank > QwenArtifactCodec.MAX_RANK) {
                throw new QwenArtifactFormatException("tensor rank is outside the supported range: " + rank);
            }
            long[] shape = new long[rank];
            for (int dimension = 0; dimension < rank; dimension++) {
                shape[dimension] = cursor.readLong("tensor shape");
                if (shape[dimension] < 0) {
                    throw new QwenArtifactFormatException("tensor shape contains a negative dimension");
                }
            }
            TensorDataType dataType = QwenArtifactCodec.enumValue(
                    cursor.readInt("tensor data type"), TensorDataType.values(), "tensor data type");
            WeightFormat format = QwenArtifactCodec.enumValue(
                    cursor.readInt("tensor weight format"), WeightFormat.values(), "tensor weight format");
            WeightLayout layout = QwenArtifactCodec.enumValue(
                    cursor.readInt("tensor weight layout"), WeightLayout.values(), "tensor weight layout");
            long dataOffset = cursor.readLong("tensor data offset");
            long byteSize = cursor.readLong("tensor byte size");
            if (dataOffset < header.tensorDataOffset()) {
                throw new QwenArtifactFormatException("tensor data offset precedes tensor data section");
            }
            long dataEnd = QwenArtifactCodec.checkedEnd(dataOffset, byteSize, "tensor data");
            if (dataEnd > fileSize) {
                throw new QwenArtifactFormatException("tensor data extends beyond the file: " + name);
            }
            tensors[i] = new TensorDescriptor(name, shape, dataType, format, layout, dataOffset, byteSize);
            try {
                if (!CompactTensorLayout.acceptsByteSize(shape, dataType, format, layout, byteSize)) {
                    throw new QwenArtifactFormatException("compact tensor byte size does not match metadata: " + name);
                }
            } catch (IllegalArgumentException exception) {
                throw new QwenArtifactFormatException(
                        "unsupported compact tensor metadata for '" + name + "': " + exception.getMessage());
            }
        }
        if (cursor.position() != header.tensorDataOffset()) {
            throw new QwenArtifactFormatException("tensor table has trailing or missing bytes");
        }
        QwenArtifactCodec.validateNonOverlapping(tensors);
        return tensors;
    }

    private static byte[] readRegion(FileChannel channel, long offset, long size, String field) throws IOException {
        if (size < 0 || size > Integer.MAX_VALUE || size > QwenArtifactCodec.MAX_METADATA_BYTES) {
            throw new QwenArtifactFormatException(field + " size is outside the supported range");
        }
        ByteBuffer bytes = ByteBuffer.allocate((int) size);
        ArtifactFileAccess.readFully(channel, offset, bytes, field);
        return bytes.array();
    }

    private static final class TableCursor {

        private final FileChannel channel;
        private final long limit;
        private long position;

        private TableCursor(FileChannel channel, long position, long limit) {
            this.channel = channel;
            this.position = position;
            this.limit = limit;
        }

        private int readInt(String field) throws IOException {
            ByteBuffer bytes = ByteBuffer.wrap(readBytes(Integer.BYTES, field)).order(QwenArtifactCodec.BYTE_ORDER);
            return bytes.getInt();
        }

        private long readLong(String field) throws IOException {
            ByteBuffer bytes = ByteBuffer.wrap(readBytes(Long.BYTES, field)).order(QwenArtifactCodec.BYTE_ORDER);
            return bytes.getLong();
        }

        private byte[] readBytes(int size, String field) throws IOException {
            if (size < 0 || position > limit - size) {
                throw new QwenArtifactFormatException("tensor table is truncated while reading " + field);
            }
            ByteBuffer bytes = ByteBuffer.allocate(size).order(QwenArtifactCodec.BYTE_ORDER);
            ArtifactFileAccess.readFully(channel, position, bytes, field);
            position += size;
            return bytes.array();
        }

        private long position() {
            return position;
        }
    }
}
