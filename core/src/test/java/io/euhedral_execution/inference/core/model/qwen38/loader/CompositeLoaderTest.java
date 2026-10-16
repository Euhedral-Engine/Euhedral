package io.euhedral_execution.inference.core.model.qwen38.loader;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.artifact.WeightLoadException;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.MemoryGpu;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// An artifact with sources loads each object from its own file: the payloads and the trellis segment lists.
class CompositeLoaderTest {

    private static final String DOWN = "text/layers/0/mlp/down";

    @TempDir
    Path directory;

    private static TensorDescriptor bf16(String name, long offset) {
        return new TensorDescriptor(
                name,
                new long[] {8},
                TensorDataType.BF16,
                WeightFormat.BF16,
                WeightLayout.CONTIGUOUS_LE_V1,
                offset,
                16);
    }

    private static byte[] filled(int value) {
        byte[] bytes = new byte[16];
        Arrays.fill(bytes, (byte) value);
        return bytes;
    }

    @Test
    void everyObjectIsReadFromItsOwnSource() throws Exception {
        Path main = this.directory.resolve("main.edrl");
        Path other = this.directory.resolve("other.edrl");
        Files.write(main, filled(1));
        Files.write(other, filled(2));
        TensorDescriptor a = bf16("text/final_norm", 0);
        TensorDescriptor b = bf16("text/layers/0/input_norm", 0);
        Artifact artifact =
                new Artifact(null, null, new TensorDescriptor[] {a, b}).withSources(Map.of(b.name(), other));
        MemoryGpu gpu = new MemoryGpu();
        Map<String, TensorHandle> handles = CompactWeightLoader.loadObjects(
                d -> artifact.pathOf(main, d),
                List.of(a, b),
                gpu,
                ArtifactProfile.Speculation.NONE,
                Set.of(),
                Set.of());
        assertArrayEquals(filled(1), gpu.bytes(handles.get(a.name()).deviceAddress(), 16));
        assertArrayEquals(filled(2), gpu.bytes(handles.get(b.name()).deviceAddress(), 16));
    }

    @Test
    void aSegmentListIsReadFromTheSourceOfItsObject() throws Exception {
        Path main = this.directory.resolve("main.edrl");
        Path other = this.directory.resolve("other.edrl");
        // The main file holds no valid segment list at the offset; the other one holds [5120].
        Files.write(main, new byte[4]);
        Files.write(
                other,
                ByteBuffer.allocate(4)
                        .order(ByteOrder.LITTLE_ENDIAN)
                        .putInt(5120)
                        .array());
        Map<String, TensorDescriptor> descriptors = new LinkedHashMap<>();
        descriptors.put(
                DOWN,
                new TensorDescriptor(
                        DOWN,
                        new long[] {320, 320, 56},
                        TensorDataType.BF16,
                        WeightFormat.TRELLIS,
                        WeightLayout.TRELLIS_TILE16_V1,
                        0,
                        2));
        descriptors.put(DOWN + ".suh", companion(DOWN + ".suh", new long[] {1, 5120}));
        descriptors.put(DOWN + ".segments", companion(DOWN + ".segments", new long[] {1}));
        Artifact artifact = new Artifact(null, null, descriptors.values().toArray(new TensorDescriptor[0]));
        Function<TensorDescriptor, Path> own = d -> artifact.pathOf(main, d);
        assertThrows(WeightLoadException.class, () -> CompactWeightLoader.validateTrellisSegments(own, descriptors));

        Artifact composite = artifact.withSources(Map.of(DOWN + ".segments", other));
        Map<String, int[]> ends =
                CompactWeightLoader.validateTrellisSegments(d -> composite.pathOf(main, d), descriptors);
        assertArrayEquals(new int[] {5120}, ends.get(DOWN));
    }

    private static TensorDescriptor companion(String name, long[] shape) {
        boolean segments = name.endsWith(".segments");
        return new TensorDescriptor(
                name,
                shape,
                segments ? TensorDataType.INT32 : TensorDataType.FP16,
                segments ? WeightFormat.I32 : WeightFormat.FP16,
                WeightLayout.CONTIGUOUS_LE_V1,
                0,
                4);
    }
}
