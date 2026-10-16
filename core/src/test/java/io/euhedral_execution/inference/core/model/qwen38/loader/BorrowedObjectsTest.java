package io.euhedral_execution.inference.core.model.qwen38.loader;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.MemoryGpu;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// A loader whose handles are borrowed: the load takes them as given, and a failed load frees only what it
/// placed itself.
class BorrowedObjectsTest {

    @TempDir
    Path directory;

    private static TensorDescriptor bf16(String name) {
        return new TensorDescriptor(
                name, new long[] {8}, TensorDataType.BF16, WeightFormat.BF16, WeightLayout.CONTIGUOUS_LE_V1, 0, 16);
    }

    private static byte[] filled(int value) {
        byte[] bytes = new byte[16];
        Arrays.fill(bytes, (byte) value);
        return bytes;
    }

    /// A loader that lends the handles it was given by name, loads the other objects itself and can be made to
    /// fail on one.
    private static final class Lending implements ObjectLoader {
        final Map<String, TensorHandle> lent;
        final String failOn;
        final Set<Long> ownLoads = new HashSet<>();

        Lending(Map<String, TensorHandle> lent, String failOn) {
            this.lent = lent;
            this.failOn = failOn;
        }

        @Override
        public TensorHandle load(Path path, TensorDescriptor descriptor, GpuMemory memory) throws IOException {
            if (descriptor.name().equals(failOn)) throw new IOException("unreadable");
            TensorHandle given = lent.get(descriptor.name());
            if (given != null) return given;
            TensorHandle own = OWNED.load(path, descriptor, memory);
            ownLoads.add(own.deviceAddress());
            return own;
        }

        @Override
        public boolean borrowed(TensorHandle handle) {
            return lent.containsValue(handle);
        }
    }

    @Test
    void borrowedHandlesAreUsedAsGivenAndSurviveAFailedLoad() throws Exception {
        Path file = directory.resolve("a.edrl");
        Files.write(file, filled(3));
        TensorDescriptor a = bf16("text/final_norm");
        TensorDescriptor b = bf16("text/layers/0/input_norm");
        TensorDescriptor c = bf16("text/layers/0/post_attention_norm");
        MemoryGpu gpu = new MemoryGpu();
        TensorHandle shared = ObjectLoader.OWNED.load(file, a, gpu);

        Lending lending = new Lending(Map.of(a.name(), shared), null);
        Map<String, TensorHandle> handles = CompactWeightLoader.loadObjects(
                d -> file, List.of(a, b), gpu, ArtifactProfile.Speculation.NONE, Set.of(), Set.of(), lending);
        assertEquals(shared, handles.get(a.name()));
        assertEquals(1, lending.ownLoads.size());
        assertArrayEquals(filled(3), gpu.bytes(handles.get(b.name()).deviceAddress(), 16));

        // The third object fails after the second was placed: the second is freed, the borrowed first is not.
        Lending failing = new Lending(Map.of(a.name(), shared), c.name());
        assertThrows(
                IOException.class,
                () -> CompactWeightLoader.loadObjects(
                        d -> file,
                        List.of(a, b, c),
                        gpu,
                        ArtifactProfile.Speculation.NONE,
                        Set.of(),
                        Set.of(),
                        failing));
        assertEquals(1, failing.ownLoads.size());
        long freed = failing.ownLoads.iterator().next();
        assertThrows(IllegalStateException.class, () -> gpu.free(freed), "already freed by the failed load");
        assertArrayEquals(filled(3), gpu.bytes(shared.deviceAddress(), 16));
        gpu.free(shared.deviceAddress());
    }

    @Test
    void theEnginesOwnLoaderBorrowsNothing() throws Exception {
        TensorHandle handle = new TensorHandle("x", new long[] {1}, TensorDataType.BF16, WeightFormat.BF16, 1L, 2);
        assertEquals(false, ObjectLoader.OWNED.borrowed(handle));
    }
}
