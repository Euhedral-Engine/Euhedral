package io.euhedral_execution.inference.core.model_loader.qwen4;

import io.euhedral_execution.inference.core.model_loader.artifact.ArtifactFileAccess;
import io.euhedral_execution.inference.core.model_loader.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.model_loader.artifact.Nvfp4Layout;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactFormatException;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertProjection;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/// Reads and structurally validates version 3 artifacts: header geometry, metadata, the tensor and expert-bank
/// tables, every object's shape against its layout's size, and that no two object ranges overlap and all lie in the
/// data section. Payloads are not read; [Qwen4Validator] checks the inventory against the configuration.
public final class Qwen4ArtifactReader {

    static final int MAX_OBJECTS = 1 << 20;
    static final int MAX_EXPERTS = 65536;
    static final int MAX_PROJECTIONS = 64;
    static final int MAX_NAME_BYTES = 4096;
    static final int MAX_RANK = 16;
    private static final ByteOrder ORDER = ByteOrder.BIG_ENDIAN;

    private Qwen4ArtifactReader() {}

    public static Qwen4Artifact read(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            if (fileSize < Qwen4Header.BYTE_SIZE) throw invalid("artifact is truncated before the header");
            Qwen4Header header = readHeader(channel, fileSize);
            Qwen4Config config = Qwen4Config.fromMetadata(Qwen4Metadata.decode(
                    readRegion(channel, header.metadataOffset(), header.metadataSize(), "metadata")));
            byte[] tables = readRegion(channel, header.tablesOffset(), header.tablesSize(), "tables");
            Tables parsed = parseTables(ByteBuffer.wrap(tables).order(ORDER), header);
            checkRanges(header, parsed.tensors, parsed.banks);
            return new Qwen4Artifact(header, config, parsed.tensors, parsed.banks);
        }
    }

    private record Tables(Qwen4Tensor[] tensors, ExpertBank[] banks) {}

    private static Qwen4Header readHeader(FileChannel channel, long fileSize) throws IOException {
        ByteBuffer in = ByteBuffer.allocate(Qwen4Header.BYTE_SIZE).order(ORDER);
        ArtifactFileAccess.readFully(channel, 0, in, "header");
        in.flip();
        int magic = in.getInt();
        int version = in.getInt();
        int architecture = in.getInt();
        int reserved = in.getInt();
        long metadataOffset = in.getLong();
        long metadataSize = in.getLong();
        long tablesOffset = in.getLong();
        long tablesSize = in.getLong();
        long dataOffset = in.getLong();
        long declaredSize = in.getLong();
        if (magic != Qwen4Header.MAGIC) throw invalid("unsupported artifact magic");
        if (version != Qwen4Header.VERSION)
            throw invalid(
                    "unsupported artifact version " + version + ": this reader handles version " + Qwen4Header.VERSION);
        if (architecture != Qwen4Header.ARCHITECTURE_QWEN4_EXP)
            throw invalid("unsupported architecture " + architecture);
        if (reserved != 0) throw invalid("header reserved field must be zero");
        if (declaredSize != fileSize)
            throw invalid("header declares " + declaredSize + " bytes but the file holds " + fileSize);
        if (metadataOffset != Qwen4Header.BYTE_SIZE) throw invalid("metadata must immediately follow the header");
        if (metadataSize < 0 || metadataSize > Qwen4Metadata.MAX_BYTES) throw invalid("metadata size is out of range");
        if (tablesOffset != metadataOffset + metadataSize) throw invalid("tables must immediately follow the metadata");
        if (tablesSize < 0 || tablesSize > (1L << 30)) throw invalid("table size is out of range");
        long tablesEnd = tablesOffset + tablesSize;
        if (tablesEnd > fileSize) throw invalid("tables extend beyond the file");
        if (dataOffset < tablesEnd || dataOffset > fileSize || dataOffset % Qwen4Header.DATA_ALIGNMENT != 0)
            throw invalid("data section offset is invalid: " + dataOffset);
        return new Qwen4Header(
                magic,
                version,
                architecture,
                metadataOffset,
                metadataSize,
                tablesOffset,
                tablesSize,
                dataOffset,
                fileSize);
    }

    private static byte[] readRegion(FileChannel channel, long offset, long size, String field) throws IOException {
        if (size < 0 || size > Integer.MAX_VALUE) throw invalid(field + " size is out of range");
        ByteBuffer bytes = ByteBuffer.allocate((int) size);
        ArtifactFileAccess.readFully(channel, offset, bytes, field);
        return bytes.array();
    }

    private static Tables parseTables(ByteBuffer in, Qwen4Header header) throws IOException {
        int tensorCount = count(in, "tensor count", MAX_OBJECTS);
        Qwen4Tensor[] tensors = new Qwen4Tensor[tensorCount];
        Set<String> names = new HashSet<>();
        for (int i = 0; i < tensorCount; i++) {
            String name = name(in, "tensor name");
            if (!names.add(name)) throw invalid("duplicate object name " + name);
            long[] shape = shape(in, name);
            TensorDataType dataType = enumValue(in, TensorDataType.values(), "data type of " + name);
            WeightFormat format = enumValue(in, WeightFormat.values(), "format of " + name);
            WeightLayout layout = enumValue(in, WeightLayout.values(), "layout of " + name);
            ComponentGroup group = enumValue(in, ComponentGroup.values(), "group of " + name);
            long offset = longValue(in, "offset of " + name);
            long size = longValue(in, "size of " + name);
            int crc = intValue(in, "CRC-32 of " + name);
            checkFixedObject(name, shape, dataType, format, layout, size);
            tensors[i] = new Qwen4Tensor(name, shape, dataType, format, layout, group, offset, size, crc);
        }
        int bankCount = count(in, "expert bank count", MAX_OBJECTS);
        ExpertBank[] banks = new ExpertBank[bankCount];
        for (int i = 0; i < bankCount; i++) {
            String name = name(in, "expert bank name");
            if (!names.add(name)) throw invalid("duplicate object name " + name);
            ComponentGroup group = enumValue(in, ComponentGroup.values(), "group of " + name);
            int layer = intValue(in, "layer of " + name);
            int experts = count(in, "expert count of " + name, MAX_EXPERTS);
            int projectionCount = count(in, "projection count of " + name, MAX_PROJECTIONS);
            if (experts == 0 || projectionCount == 0) throw invalid("expert bank " + name + " is empty");
            List<ExpertProjection> projections = new ArrayList<>();
            Set<String> projectionNames = new HashSet<>();
            for (int p = 0; p < projectionCount; p++) {
                String projectionName = name(in, "projection name of " + name);
                if (!projectionNames.add(projectionName)) throw invalid(name + " repeats projection " + projectionName);
                long[] shape = shape(in, name + "/" + projectionName);
                TensorDataType dataType = enumValue(in, TensorDataType.values(), "data type of " + projectionName);
                WeightFormat format = enumValue(in, WeightFormat.values(), "format of " + projectionName);
                WeightLayout layout = enumValue(in, WeightLayout.values(), "layout of " + projectionName);
                long recordOffset = longValue(in, "offset of " + projectionName);
                long size = longValue(in, "size of " + projectionName);
                checkFixedObject(name + "/" + projectionName, shape, dataType, format, layout, size);
                if (layout == WeightLayout.ROW_INTERLEAVED_NVFP4_V1)
                    throw invalid("expert projections use the row-split layout: " + name + "/" + projectionName);
                if (recordOffset < 0 || recordOffset % Qwen4Header.OBJECT_ALIGNMENT != 0)
                    throw invalid("projection " + projectionName + " of " + name + " is not 256-byte aligned");
                projections.add(
                        new ExpertProjection(projectionName, shape, dataType, format, layout, recordOffset, size));
            }
            long[] offsets = new long[experts];
            long[] sizes = new long[experts];
            int[] crcs = new int[experts];
            for (int e = 0; e < experts; e++) {
                offsets[e] = longValue(in, "expert offset");
                sizes[e] = longValue(in, "expert size");
                crcs[e] = intValue(in, "expert CRC-32");
            }
            try {
                banks[i] = new ExpertBank(name, group, layer, projections, offsets, sizes, crcs);
            } catch (IllegalArgumentException | ArithmeticException exception) {
                throw invalid("expert bank " + name + ": " + exception.getMessage());
            }
        }
        if (in.hasRemaining()) throw invalid("tables contain trailing bytes");
        return new Tables(tensors, banks);
    }

    /// Whether the object's recorded size is the size its layout gives for its shape.
    private static void checkFixedObject(
            String name, long[] shape, TensorDataType dataType, WeightFormat format, WeightLayout layout, long size)
            throws QwenArtifactFormatException {
        if (size <= 0) throw invalid(name + " has no payload");
        for (long dimension : shape) if (dimension <= 0) throw invalid(name + " has an empty dimension");
        try {
            if (layout == WeightLayout.ROW_INTERLEAVED_NVFP4_V1) {
                if (format != WeightFormat.NVFP4 || shape.length != 2 || !NgramLayout.supports(shape[1]))
                    throw invalid(name + " is not a valid interleaved NVFP4 table");
                if (NgramLayout.byteSize(shape[0], shape[1]) != size)
                    throw invalid(name + ": size " + size + " does not match its shape " + Arrays.toString(shape));
                return;
            }
            if (layout == WeightLayout.ROW_SPLIT_P2E2_V1 || layout == WeightLayout.ROW_SPLIT_K128_SD4_V1)
                throw invalid(name + " uses a compressed layout that version 3 artifacts do not carry");
            if (format == WeightFormat.NVFP4 && layout != WeightLayout.ROW_SPLIT_K128_V1)
                throw invalid(name + ": NVFP4 needs the row-split layout");
            if (format != WeightFormat.NVFP4 && layout != WeightLayout.CONTIGUOUS_LE_V1)
                throw invalid(name + ": " + format + " needs the contiguous layout");
            if (!CompactTensorLayout.acceptsByteSize(shape, dataType, format, layout, size))
                throw invalid(name + ": size " + size + " does not match shape " + Arrays.toString(shape) + " in "
                        + format + "/" + layout);
        } catch (IllegalArgumentException | ArithmeticException exception) {
            throw invalid("unsupported object " + name + ": " + exception.getMessage());
        }
    }

    /// Every object (a fixed tensor, or an expert record) lies in the data section, starts on a 256-byte boundary and
    /// overlaps no other.
    private static void checkRanges(Qwen4Header header, Qwen4Tensor[] tensors, ExpertBank[] banks)
            throws QwenArtifactFormatException {
        int total = tensors.length;
        for (ExpertBank bank : banks) total = Math.addExact(total, bank.expertCount());
        long[] offsets = new long[total];
        long[] sizes = new long[total];
        int n = 0;
        for (Qwen4Tensor tensor : tensors) {
            offsets[n] = tensor.dataOffset();
            sizes[n++] = tensor.byteSize();
        }
        for (ExpertBank bank : banks) {
            for (int e = 0; e < bank.expertCount(); e++) {
                offsets[n] = bank.fileOffset(e);
                sizes[n++] = bank.recordBytes(e);
            }
        }
        Integer[] order = new Integer[total];
        for (int i = 0; i < total; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Long.compare(offsets[a], offsets[b]));
        long previousEnd = header.dataOffset();
        for (int index : order) {
            long offset = offsets[index];
            long size = sizes[index];
            if (offset < previousEnd)
                throw invalid("object range at " + offset + " overlaps another object or precedes the data section");
            if (offset % Qwen4Header.OBJECT_ALIGNMENT != 0)
                throw invalid("object at " + offset + " is not 256-byte aligned");
            long end;
            try {
                end = Math.addExact(offset, size);
            } catch (ArithmeticException overflow) {
                throw invalid("object range at " + offset + " overflows");
            }
            if (end > header.fileSize()) throw invalid("object range at " + offset + " extends beyond the file");
            previousEnd = end;
        }
    }

    private static String name(ByteBuffer in, String field) throws QwenArtifactFormatException {
        int length = intValue(in, field + " length");
        if (length <= 0 || length > MAX_NAME_BYTES) throw invalid(field + " length is out of range: " + length);
        need(in, length, field);
        byte[] bytes = new byte[length];
        in.get(bytes);
        try {
            return StandardCharsets.UTF_8
                    .newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(bytes))
                    .toString();
        } catch (CharacterCodingException exception) {
            throw invalid(field + " is not valid UTF-8");
        }
    }

    private static long[] shape(ByteBuffer in, String name) throws QwenArtifactFormatException {
        int rank = intValue(in, "rank of " + name);
        if (rank < 0 || rank > MAX_RANK) throw invalid("rank of " + name + " is out of range: " + rank);
        long[] shape = new long[rank];
        for (int i = 0; i < rank; i++) shape[i] = longValue(in, "shape of " + name);
        return shape;
    }

    private static int count(ByteBuffer in, String field, int max) throws QwenArtifactFormatException {
        int value = intValue(in, field);
        if (value < 0 || value > max) throw invalid(field + " is out of range: " + value);
        return value;
    }

    private static <E extends Enum<E>> E enumValue(ByteBuffer in, E[] values, String field)
            throws QwenArtifactFormatException {
        int ordinal = intValue(in, field);
        if (ordinal < 0 || ordinal >= values.length) throw invalid(field + " is out of range: " + ordinal);
        return values[ordinal];
    }

    private static int intValue(ByteBuffer in, String field) throws QwenArtifactFormatException {
        need(in, Integer.BYTES, field);
        return in.getInt();
    }

    private static long longValue(ByteBuffer in, String field) throws QwenArtifactFormatException {
        need(in, Long.BYTES, field);
        return in.getLong();
    }

    private static void need(ByteBuffer in, int bytes, String field) throws QwenArtifactFormatException {
        if (bytes < 0 || in.remaining() < bytes) throw invalid("tables are truncated while reading " + field);
    }

    static QwenArtifactFormatException invalid(String message) {
        return new QwenArtifactFormatException(message);
    }

    /// The NVFP4 size of an expert projection, for validators.
    static long projectionBytes(ExpertProjection projection) {
        return Nvfp4Layout.byteSize(projection.shape()[0], projection.shape()[1]);
    }
}
