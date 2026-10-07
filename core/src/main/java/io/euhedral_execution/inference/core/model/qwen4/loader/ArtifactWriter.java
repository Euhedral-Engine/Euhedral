package io.euhedral_execution.inference.core.model.qwen4.loader;

import io.euhedral_execution.inference.core.artifact.ArtifactFormatException;
import io.euhedral_execution.inference.core.model.qwen4.Qwen4Config;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertBank;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertProjection;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.zip.CRC32;

/// Writes version 3 artifacts. The converter (tools/convert_flash_next.py) is how artifacts are made; this writer
/// produces the same bytes from a [Artifact] and the payloads, so tests can build artifacts of any size.
public final class ArtifactWriter {

    private ArtifactWriter() {}

    /// The payload of a fixed object.
    @FunctionalInterface
    public interface TensorPayload extends Function<Tensor, byte[]> {}

    /// The record of expert `expert` of a bank.
    @FunctionalInterface
    public interface RecordPayload extends BiFunction<ExpertBank, Integer, byte[]> {}

    /// Encodes the tables (the tensors, then the banks) of `artifact`, with the CRC-32s it carries.
    public static byte[] encodeTables(Tensor[] tensors, ExpertBank[] banks) throws ArtifactFormatException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(bytes)) {
            out.writeInt(tensors.length);
            for (Tensor tensor : tensors) {
                writeName(out, tensor.name());
                out.writeInt(tensor.shape().length);
                for (long dimension : tensor.shape()) out.writeLong(dimension);
                out.writeInt(tensor.dataType().ordinal());
                out.writeInt(tensor.format().ordinal());
                out.writeInt(tensor.layout().ordinal());
                out.writeInt(tensor.group().ordinal());
                out.writeLong(tensor.dataOffset());
                out.writeLong(tensor.byteSize());
                out.writeInt(tensor.crc32());
            }
            out.writeInt(banks.length);
            for (ExpertBank bank : banks) {
                writeName(out, bank.name());
                out.writeInt(bank.group().ordinal());
                out.writeInt(bank.layer());
                out.writeInt(bank.expertCount());
                out.writeInt(bank.projections().size());
                for (ExpertProjection projection : bank.projections()) {
                    writeName(out, projection.name());
                    out.writeInt(projection.shape().length);
                    for (long dimension : projection.shape()) out.writeLong(dimension);
                    out.writeInt(projection.dataType().ordinal());
                    out.writeInt(projection.format().ordinal());
                    out.writeInt(projection.layout().ordinal());
                    out.writeLong(projection.recordOffset());
                    out.writeLong(projection.byteSize());
                }
                for (int expert = 0; expert < bank.expertCount(); expert++) {
                    out.writeLong(bank.fileOffset(expert));
                    out.writeLong(bank.recordBytes(expert));
                    out.writeInt(bank.crc32(expert));
                }
            }
        } catch (IOException exception) {
            throw new AssertionError(exception);
        }
        return bytes.toByteArray();
    }

    public static byte[] encodeHeader(long metadataSize, long tablesSize, long dataOffset, long fileSize) {
        ByteBuffer header = ByteBuffer.allocate(Header.BYTE_SIZE).order(ByteOrder.BIG_ENDIAN);
        header.putInt(Header.MAGIC);
        header.putInt(Header.VERSION);
        header.putInt(Header.ARCHITECTURE_QWEN4_EXP);
        header.putInt(0);
        header.putLong(Header.BYTE_SIZE);
        header.putLong(metadataSize);
        header.putLong(Header.BYTE_SIZE + metadataSize);
        header.putLong(tablesSize);
        header.putLong(dataOffset);
        header.putLong(fileSize);
        return header.array();
    }

    /// Writes an artifact whose tensors and banks already have their offsets and sizes; `tensorPayload` and
    /// `recordPayload` supply the bytes, whose CRC-32s replace the ones in the inputs. Returns the artifact as
    /// written.
    public static Artifact write(
            Path path,
            Qwen4Config config,
            Tensor[] tensors,
            ExpertBank[] banks,
            long dataOffset,
            long fileSize,
            TensorPayload tensorPayload,
            RecordPayload recordPayload)
            throws IOException {
        byte[] metadata = Metadata.encode(config.toMetadata());
        Tensor[] written = new Tensor[tensors.length];
        ExpertBank[] writtenBanks = new ExpertBank[banks.length];
        try (FileChannel channel = FileChannel.open(
                path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            channel.truncate(fileSize);
            for (int i = 0; i < tensors.length; i++) {
                Tensor tensor = tensors[i];
                byte[] payload = tensorPayload.apply(tensor);
                if (payload.length != tensor.byteSize())
                    throw new ArtifactFormatException(
                            "payload of " + tensor.name() + " has " + payload.length + " bytes");
                channel.write(ByteBuffer.wrap(payload), tensor.dataOffset());
                written[i] = new Tensor(
                        tensor.name(),
                        tensor.shape(),
                        tensor.dataType(),
                        tensor.format(),
                        tensor.layout(),
                        tensor.group(),
                        tensor.dataOffset(),
                        tensor.byteSize(),
                        crc(payload));
            }
            for (int b = 0; b < banks.length; b++) {
                ExpertBank bank = banks[b];
                int[] crcs = new int[bank.expertCount()];
                long[] offsets = new long[bank.expertCount()];
                long[] sizes = new long[bank.expertCount()];
                for (int expert = 0; expert < bank.expertCount(); expert++) {
                    byte[] record = recordPayload.apply(bank, expert);
                    if (record.length != bank.recordBytes(expert))
                        throw new ArtifactFormatException(
                                "record " + expert + " of " + bank.name() + " has the wrong size");
                    channel.write(ByteBuffer.wrap(record), bank.fileOffset(expert));
                    crcs[expert] = crc(record);
                    offsets[expert] = bank.fileOffset(expert);
                    sizes[expert] = bank.recordBytes(expert);
                }
                writtenBanks[b] = new ExpertBank(
                        bank.name(), bank.group(), bank.layer(), bank.projections(), offsets, sizes, crcs);
            }
            byte[] tables = encodeTables(written, writtenBanks);
            if (Header.BYTE_SIZE + metadata.length + tables.length > dataOffset)
                throw new ArtifactFormatException("metadata and tables do not fit before the data section");
            channel.write(ByteBuffer.wrap(metadata), Header.BYTE_SIZE);
            channel.write(ByteBuffer.wrap(tables), Header.BYTE_SIZE + metadata.length);
            channel.write(ByteBuffer.wrap(encodeHeader(metadata.length, tables.length, dataOffset, fileSize)), 0);
        }
        return ArtifactReader.read(path);
    }

    /// The size of the metadata and tables for `tensors` and `banks`, for placing the data section.
    public static long preambleBytes(Qwen4Config config, Tensor[] tensors, ExpertBank[] banks)
            throws ArtifactFormatException {
        return Header.BYTE_SIZE + Metadata.encode(config.toMetadata()).length + encodeTables(tensors, banks).length;
    }

    public static int crc(byte[] payload) {
        CRC32 crc = new CRC32();
        crc.update(payload);
        return (int) crc.getValue();
    }

    private static void writeName(DataOutputStream out, String name) throws IOException {
        byte[] bytes = name.getBytes(StandardCharsets.UTF_8);
        out.writeInt(bytes.length);
        out.write(bytes);
    }

    /// Kept for callers that hold metadata maps.
    public static byte[] encodeMetadata(Map<String, Object> metadata) throws ArtifactFormatException {
        return Metadata.encode(metadata);
    }
}
