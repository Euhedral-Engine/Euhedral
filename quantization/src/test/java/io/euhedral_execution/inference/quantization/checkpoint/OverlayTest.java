package io.euhedral_execution.inference.quantization.checkpoint;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// An overlay checkpoint reads its replaced tensors from its own shard and every other tensor from the source.
class OverlayTest {

    @Test
    void replacedTensorsReadFromTheOverlay(@TempDir Path directory) throws Exception {
        Path source = directory.resolve("source"), target = directory.resolve("overlay");
        Files.createDirectories(source);
        Map<String, float[]> original = new LinkedHashMap<>();
        original.put("a.weight", new float[] {1, 2, 3});
        original.put("b.weight", new float[] {4, 5});
        Overlay.writeShard(source.resolve("model-00001.safetensors"), original);
        Files.writeString(
                source.resolve("model.safetensors.index.json"),
                "{\"metadata\":{},\"weight_map\":{\"a.weight\":\"model-00001.safetensors\","
                        + "\"b.weight\":\"model-00001.safetensors\"}}");
        Files.writeString(source.resolve("config.json"), "{}");
        Overlay.write(source, target, Map.of("b.weight", new float[] {-0.5f, 6.5f}));
        try (Safetensors store = Safetensors.open(target);
                Arena arena = Arena.ofConfined()) {
            assertEquals(java.util.List.of(1f, 2f, 3f), values(store, "a.weight", arena));
            assertEquals(java.util.List.of(-0.5f, 6.5f), values(store, "b.weight", arena));
        }
        assertEquals(true, Files.isSymbolicLink(target.resolve("config.json")));
    }

    private static java.util.List<Float> values(Safetensors store, String name, Arena arena) throws Exception {
        long bytes = store.tensor(name).bytes();
        MemorySegment host = arena.allocate(bytes, 2);
        store.read(name, 0, host);
        java.util.List<Float> out = new java.util.ArrayList<>();
        for (long i = 0; i < bytes / 2; i++)
            out.add(Float.intBitsToFloat((host.getAtIndex(ValueLayout.JAVA_SHORT, i) & 0xffff) << 16));
        return out;
    }
}
