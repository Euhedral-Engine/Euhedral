package io.euhedral_execution.inference.core.model_loader.qwen4;

import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactFormatException;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/// The typed key/value metadata of a version 3 artifact: `int32 count`, then for each entry (in key order) a
/// length-prefixed UTF-8 key, a type byte and the value, all big-endian:
///
/// | type | value |
/// |---|---|
/// | 0 | int64 |
/// | 1 | IEEE 754 binary64 |
/// | 2 | boolean byte (0 or 1) |
/// | 3 | string: int32 length, UTF-8 |
/// | 4 | int64 array: int32 count, values |
/// | 5 | string array: int32 count, strings |
///
/// Values decode to `Long`, `Double`, `Boolean`, `String`, `long[]` and `String[]`.
public final class Qwen4Metadata {

    static final int MAX_ENTRIES = 4096;
    static final int MAX_STRING_BYTES = 1 << 20;
    static final int MAX_ARRAY = 1 << 20;
    public static final long MAX_BYTES = 8L << 20;
    private static final ByteOrder ORDER = ByteOrder.BIG_ENDIAN;

    private Qwen4Metadata() {}

    public static byte[] encode(Map<String, Object> values) throws QwenArtifactFormatException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(bytes)) {
            out.writeInt(values.size());
            for (Map.Entry<String, Object> entry : new TreeMap<>(values).entrySet()) {
                writeString(out, entry.getKey());
                Object value = entry.getValue();
                switch (value) {
                    case Boolean flag -> {
                        out.writeByte(2);
                        out.writeByte(flag ? 1 : 0);
                    }
                    case Long number -> {
                        out.writeByte(0);
                        out.writeLong(number);
                    }
                    case Integer number -> {
                        out.writeByte(0);
                        out.writeLong(number);
                    }
                    case Double number -> {
                        out.writeByte(1);
                        out.writeDouble(number);
                    }
                    case String text -> {
                        out.writeByte(3);
                        writeString(out, text);
                    }
                    case long[] numbers -> {
                        out.writeByte(4);
                        out.writeInt(numbers.length);
                        for (long number : numbers) out.writeLong(number);
                    }
                    case String[] texts -> {
                        out.writeByte(5);
                        out.writeInt(texts.length);
                        for (String text : texts) writeString(out, text);
                    }
                    default ->
                        throw new QwenArtifactFormatException(
                                "metadata " + entry.getKey() + " has an unsupported value");
                }
            }
        } catch (IOException exception) {
            if (exception instanceof QwenArtifactFormatException format) throw format;
            throw new AssertionError(exception);
        }
        if (bytes.size() > MAX_BYTES) throw new QwenArtifactFormatException("metadata is too large");
        return bytes.toByteArray();
    }

    public static Map<String, Object> decode(byte[] metadata) throws QwenArtifactFormatException {
        ByteBuffer in = ByteBuffer.wrap(metadata).order(ORDER);
        int count = readInt(in, "entry count");
        if (count < 0 || count > MAX_ENTRIES) throw invalid("metadata entry count is out of range: " + count);
        Map<String, Object> values = new LinkedHashMap<>();
        String previous = null;
        for (int i = 0; i < count; i++) {
            String key = readString(in, "key");
            if (previous != null && key.compareTo(previous) <= 0)
                throw invalid("metadata keys are not in strictly increasing order at '" + key + "'");
            previous = key;
            need(in, 1, key);
            int type = in.get();
            Object value =
                    switch (type) {
                        case 0 -> {
                            need(in, 8, key);
                            yield in.getLong();
                        }
                        case 1 -> {
                            need(in, 8, key);
                            double number = in.getDouble();
                            if (!Double.isFinite(number)) throw invalid("metadata " + key + " is not finite");
                            yield number;
                        }
                        case 2 -> {
                            need(in, 1, key);
                            int flag = in.get() & 0xff;
                            if (flag > 1) throw invalid("metadata " + key + " is not a boolean");
                            yield flag == 1;
                        }
                        case 3 -> readString(in, key);
                        case 4 -> {
                            int length = readCount(in, key);
                            need(in, Math.multiplyExact(length, 8), key);
                            long[] numbers = new long[length];
                            for (int index = 0; index < length; index++) numbers[index] = in.getLong();
                            yield numbers;
                        }
                        case 5 -> {
                            int length = readCount(in, key);
                            String[] texts = new String[length];
                            for (int index = 0; index < length; index++) texts[index] = readString(in, key);
                            yield texts;
                        }
                        default -> throw invalid("metadata " + key + " has unknown type " + type);
                    };
            values.put(key, value);
        }
        if (in.hasRemaining()) throw invalid("metadata contains trailing bytes");
        return values;
    }

    /// Typed access to decoded metadata that tracks what was read, so [#finish()] can reject keys the reader
    /// does not know.
    public static final class Reader {
        private final Map<String, Object> values;
        private final Set<String> used = new HashSet<>();

        public Reader(Map<String, Object> values) {
            this.values = values;
        }

        private Object take(String key, Class<?> type) throws QwenArtifactFormatException {
            Object value = this.values.get(key);
            if (value == null) throw invalid("metadata is missing '" + key + "'");
            if (!type.isInstance(value))
                throw invalid("metadata '" + key + "' has type "
                        + value.getClass().getSimpleName() + ", expected " + type.getSimpleName());
            this.used.add(key);
            return value;
        }

        public long integer(String key) throws QwenArtifactFormatException {
            return (Long) take(key, Long.class);
        }

        /// A non-negative value that fits a Java `int`.
        public int count(String key) throws QwenArtifactFormatException {
            long value = integer(key);
            if (value < 0 || value > Integer.MAX_VALUE)
                throw invalid("metadata '" + key + "' is out of range: " + value);
            return (int) value;
        }

        /// A signed value that fits a Java `int`.
        public int signed(String key) throws QwenArtifactFormatException {
            long value = integer(key);
            if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE)
                throw invalid("metadata '" + key + "' is out of range: " + value);
            return (int) value;
        }

        public double real(String key) throws QwenArtifactFormatException {
            return (Double) take(key, Double.class);
        }

        public boolean bool(String key) throws QwenArtifactFormatException {
            return (Boolean) take(key, Boolean.class);
        }

        public String string(String key) throws QwenArtifactFormatException {
            return (String) take(key, String.class);
        }

        public long[] longs(String key) throws QwenArtifactFormatException {
            return ((long[]) take(key, long[].class)).clone();
        }

        public int[] ints(String key) throws QwenArtifactFormatException {
            long[] values = (long[]) take(key, long[].class);
            int[] result = new int[values.length];
            for (int i = 0; i < values.length; i++) {
                if (values[i] < Integer.MIN_VALUE || values[i] > Integer.MAX_VALUE)
                    throw invalid("metadata '" + key + "' has an out-of-range element " + values[i]);
                result[i] = (int) values[i];
            }
            return result;
        }

        public String[] strings(String key) throws QwenArtifactFormatException {
            return ((String[]) take(key, String[].class)).clone();
        }

        /// Fails when the metadata holds a key that was never read.
        public void finish() throws QwenArtifactFormatException {
            List<String> unknown = new ArrayList<>(this.values.keySet());
            unknown.removeAll(this.used);
            if (!unknown.isEmpty()) throw invalid("metadata has unknown keys: " + unknown);
        }
    }

    private static void writeString(DataOutputStream out, String value) throws IOException {
        byte[] bytes = encodeUtf8(value);
        out.writeInt(bytes.length);
        out.write(bytes);
    }

    private static byte[] encodeUtf8(String value) throws QwenArtifactFormatException {
        if (value == null) throw invalid("metadata string is null");
        try {
            var encoded = StandardCharsets.UTF_8
                    .newEncoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .encode(java.nio.CharBuffer.wrap(value));
            byte[] bytes = new byte[encoded.remaining()];
            encoded.get(bytes);
            if (bytes.length > MAX_STRING_BYTES) throw invalid("metadata string is too long");
            return bytes;
        } catch (CharacterCodingException exception) {
            throw invalid("metadata string is not valid UTF-8");
        }
    }

    private static int readInt(ByteBuffer in, String field) throws QwenArtifactFormatException {
        need(in, 4, field);
        return in.getInt();
    }

    private static int readCount(ByteBuffer in, String field) throws QwenArtifactFormatException {
        int count = readInt(in, field + " length");
        if (count < 0 || count > MAX_ARRAY) throw invalid("metadata '" + field + "' length is out of range: " + count);
        return count;
    }

    private static String readString(ByteBuffer in, String field) throws QwenArtifactFormatException {
        int length = readInt(in, field + " length");
        if (length < 0 || length > MAX_STRING_BYTES)
            throw invalid("metadata '" + field + "' string length is out of range: " + length);
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
            throw invalid("metadata '" + field + "' is not valid UTF-8");
        }
    }

    private static void need(ByteBuffer in, int bytes, String field) throws QwenArtifactFormatException {
        if (bytes < 0 || in.remaining() < bytes) throw invalid("metadata is truncated while reading '" + field + "'");
    }

    private static QwenArtifactFormatException invalid(String message) {
        return new QwenArtifactFormatException(message);
    }
}
