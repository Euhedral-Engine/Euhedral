package io.euhedral_execution.inference.quantization.gptq;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/// One object's stored rounding, as tools/euhedral_artifacts/codes.py's QuantState: grouped formats hold int8 codes
/// [N, K] and float16 scales [N, K/64]; NVFP4 holds uint8 codes (magnitude | sign << 3) [N, K], uint8 scale codes (or
/// SD4 table indices) [N, K/16], the FP32 tensor scale and, for SD4, the 16-entry table. `save` writes the .npz that
/// `QuantState.load` and `convert_checkpoint.py --codes-from` read.
public record QuantState(
        String formatName,
        String layout,
        long rows,
        long columns,
        MemorySegment codes,
        MemorySegment scales,
        float globalScale,
        byte[] table) {

    private static final int CHUNK = 1 << 22;

    /// The state file of object `name` in `directory` (codes.state_file: '/' becomes '.').
    public static Path stateFile(Path directory, String name) {
        return directory.resolve(name.replace('/', '.') + ".npz");
    }

    public void save(Path path) throws IOException {
        boolean nvfp4 = formatName.equals("NVFP4");
        long groups = columns / (nvfp4 ? 16 : 64);
        Path partial = path.resolveSibling(path.getFileName() + ".partial");
        try (var zip = new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(partial), 1 << 20))) {
            zip.setMethod(ZipOutputStream.STORED);
            entry(zip, "format_name", text(formatName));
            entry(zip, "layout", text(layout));
            entry(
                    zip,
                    "shape",
                    array(
                            "<i8",
                            new long[] {2},
                            MemorySegment.ofArray(ByteBuffer.allocate(16)
                                    .order(ByteOrder.LITTLE_ENDIAN)
                                    .putLong(rows)
                                    .putLong(columns)
                                    .array())));
            entry(zip, "codes", array(nvfp4 ? "|u1" : "|i1", new long[] {rows, columns}, codes));
            entry(zip, "scales", array(nvfp4 ? "|u1" : "<f2", new long[] {rows, groups}, scales));
            entry(
                    zip,
                    "global_scale",
                    array(
                            "<f4",
                            new long[] {},
                            MemorySegment.ofArray(ByteBuffer.allocate(4)
                                    .order(ByteOrder.LITTLE_ENDIAN)
                                    .putFloat(globalScale)
                                    .array())));
            if (table != null)
                entry(zip, "table", array("|u1", new long[] {table.length}, MemorySegment.ofArray(table)));
        }
        Files.move(partial, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    /// A .npy file: its header and its data.
    private record Npy(byte[] header, MemorySegment data) {}

    private static Npy text(String value) {
        byte[] utf32 = value.getBytes(java.nio.charset.Charset.forName("UTF-32LE"));
        return array("<U" + value.length(), new long[] {}, MemorySegment.ofArray(utf32));
    }

    private static Npy array(String descr, long[] shape, MemorySegment data) {
        StringBuilder dims = new StringBuilder("(");
        for (int i = 0; i < shape.length; i++)
            dims.append(shape[i]).append(shape.length == 1 ? "," : i + 1 < shape.length ? ", " : "");
        dims.append(")");
        String dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + dims + ", }";
        int unpadded = 10 + dict.length() + 1;
        int padding = (64 - unpadded % 64) % 64;
        String headerText = dict + " ".repeat(padding) + "\n";
        ByteBuffer header = ByteBuffer.allocate(10 + headerText.length()).order(ByteOrder.LITTLE_ENDIAN);
        header.put(new byte[] {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0});
        header.putShort((short) headerText.length());
        header.put(headerText.getBytes(StandardCharsets.US_ASCII));
        return new Npy(header.array(), data);
    }

    private static void entry(ZipOutputStream zip, String name, Npy npy) throws IOException {
        CRC32 crc = new CRC32();
        crc.update(npy.header());
        for (long at = 0; at < npy.data().byteSize(); at += CHUNK) {
            crc.update(npy.data()
                    .asSlice(at, Math.min(CHUNK, npy.data().byteSize() - at))
                    .asByteBuffer());
        }
        ZipEntry entry = new ZipEntry(name + ".npy");
        long size = npy.header().length + npy.data().byteSize();
        entry.setMethod(ZipEntry.STORED);
        entry.setSize(size);
        entry.setCompressedSize(size);
        entry.setCrc(crc.getValue());
        zip.putNextEntry(entry);
        zip.write(npy.header());
        write(zip, npy.data());
        zip.closeEntry();
    }

    private static void write(OutputStream out, MemorySegment data) throws IOException {
        byte[] buffer = new byte[(int) Math.min(CHUNK, Math.max(1, data.byteSize()))];
        for (long at = 0; at < data.byteSize(); at += buffer.length) {
            int length = (int) Math.min(buffer.length, data.byteSize() - at);
            MemorySegment.copy(data, java.lang.foreign.ValueLayout.JAVA_BYTE, at, buffer, 0, length);
            out.write(buffer, 0, length);
        }
    }
}
