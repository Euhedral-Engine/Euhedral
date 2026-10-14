package io.euhedral_execution.inference.quantization.gptq;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.regex.Pattern;

/// A minimal .npy reader for test fixtures (little-endian, C order).
public record Npy(String descr, long[] shape, ByteBuffer data) {

    private static final Pattern DESCR = Pattern.compile("'descr': '([^']+)'");
    private static final Pattern SHAPE = Pattern.compile("'shape': \\(([^)]*)\\)");

    public static Npy read(InputStream input) throws IOException {
        byte[] bytes = input.readAllBytes();
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int major = bytes[6];
        int headerLength = major == 1 ? Short.toUnsignedInt(buffer.getShort(8)) : buffer.getInt(8);
        int start = major == 1 ? 10 : 12;
        String header = new String(bytes, start, headerLength, StandardCharsets.US_ASCII);
        var descr = DESCR.matcher(header);
        var shape = SHAPE.matcher(header);
        if (!descr.find() || !shape.find()) throw new IOException("bad .npy header " + header);
        String[] parts = shape.group(1).replace(" ", "").split(",");
        long[] dims = java.util.Arrays.stream(parts)
                .filter(p -> !p.isEmpty())
                .mapToLong(Long::parseLong)
                .toArray();
        ByteBuffer data = ByteBuffer.wrap(bytes, start + headerLength, bytes.length - start - headerLength)
                .slice()
                .order(ByteOrder.LITTLE_ENDIAN);
        return new Npy(descr.group(1), dims, data);
    }
}
