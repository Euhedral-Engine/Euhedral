package io.euhedral_execution.inference.quantization.checkpoint;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SafetensorsTest {

    @Test
    void readsTensorsAndRowRangesOfASingleFileCheckpoint(@TempDir Path directory) throws Exception {
        String header = "{\"__metadata__\":{\"format\":\"pt\"},"
                + "\"a\":{\"dtype\":\"BF16\",\"shape\":[3,2],\"data_offsets\":[0,12]},"
                + "\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[12,20]}}";
        byte[] json = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer file = ByteBuffer.allocate(8 + json.length + 20).order(ByteOrder.LITTLE_ENDIAN);
        file.putLong(json.length).put(json);
        for (short v = 1; v <= 6; v++) file.putShort(v);
        file.putFloat(1.5f).putFloat(-2f);
        Files.write(directory.resolve("model.safetensors"), file.array());
        try (Safetensors store = Safetensors.open(directory);
                Arena arena = Arena.ofConfined()) {
            assertArrayEquals(new long[] {3, 2}, store.tensor("a").shape());
            var rows = arena.allocate(8);
            store.read("a", 4, rows); // rows 1..2
            assertArrayEquals(new short[] {3, 4, 5, 6}, rows.toArray(ValueLayout.JAVA_SHORT_UNALIGNED));
            var b = arena.allocate(8);
            store.read("b", b);
            assertEquals(-2f, b.getAtIndex(ValueLayout.JAVA_FLOAT_UNALIGNED, 1));
        }
    }
}
