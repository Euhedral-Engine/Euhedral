package io.euhedral_execution.inference.quantization.checkpoint;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Map;
import org.junit.jupiter.api.Test;

/// The fused objects read from the BF16 checkpoint equal tools/gptq/qwen38.py's byte for byte (SHA-256 fixture written
/// from Swift-1.5: layers 0 and 3 and the MTP layer).
class Qwen38ObjectsTest {

    @Test
    void fusedObjectsMatchThePythonCalibrationModel() throws Exception {
        String checkpoint = System.getProperty("euhedral.quantization.checkpoint", "");
        assumeTrue(!checkpoint.isBlank(), "needs -Peuhedral.quantization.checkpoint");
        @SuppressWarnings("unchecked")
        Map<String, String> expected = new ObjectMapper()
                .readValue(getClass().getResourceAsStream("/swift-1.5-object-sha256.json"), Map.class);
        try (Safetensors store = Safetensors.open(Path.of(checkpoint))) {
            var objects = new Qwen38Objects(store, Path.of(checkpoint));
            for (var entry : expected.entrySet()) {
                var source = objects.object(entry.getKey());
                try (Arena arena = Arena.ofConfined()) {
                    var bytes = arena.allocate(source.bytes(), 16);
                    objects.read(source, bytes);
                    var digest = MessageDigest.getInstance("SHA-256");
                    digest.update(bytes.asByteBuffer());
                    assertEquals(entry.getValue(), HexFormat.of().formatHex(digest.digest()), entry.getKey());
                }
            }
        }
    }
}
