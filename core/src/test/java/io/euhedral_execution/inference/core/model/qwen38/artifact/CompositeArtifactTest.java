package io.euhedral_execution.inference.core.model.qwen38.artifact;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.ArtifactFormatException;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;

/// An artifact can name a source file per object, and its metadata can carry the MTP draft depth.
class CompositeArtifactTest {

    private static final Path MAIN = Path.of("main.edrl");
    private static final Path OTHER = Path.of("other.edrl");

    private static TensorDescriptor descriptor(String name) {
        return new TensorDescriptor(
                name, new long[] {1}, TensorDataType.BF16, WeightFormat.BF16, WeightLayout.CONTIGUOUS_LE_V1, 0, 2);
    }

    private static Qwen38Config config() {
        return new Qwen38Config(
                32_000,
                4_096,
                1,
                32,
                8,
                128,
                11_008,
                16,
                8,
                64,
                128,
                4,
                1.0e-6,
                1_000_000.0,
                1.0,
                32_768,
                "silu",
                new LayerType[] {LayerType.FULL_ATTENTION},
                0,
                0,
                0,
                0,
                true,
                false,
                0);
    }

    @Test
    void anArtifactWithoutSourcesReadsEveryObjectFromItsOwnFile() {
        TensorDescriptor a = descriptor("text/layers/0/mlp/down");
        Artifact artifact = new Artifact(null, null, new TensorDescriptor[] {a});
        assertTrue(artifact.sources().isEmpty());
        assertEquals(MAIN, artifact.pathOf(MAIN, a));
    }

    @Test
    void aSourceOverridesTheFileOfOneObjectOnly() {
        TensorDescriptor a = descriptor("text/layers/0/mlp/down");
        TensorDescriptor b = descriptor("text/layers/0/mlp/gate_up");
        Artifact base = new Artifact(null, null, new TensorDescriptor[] {a, b});
        Artifact composite = base.withSources(Map.of(a.name(), OTHER));
        assertEquals(OTHER, composite.pathOf(MAIN, a));
        assertEquals(MAIN, composite.pathOf(MAIN, b));
        assertTrue(base.sources().isEmpty(), "the original artifact is unchanged");
        assertSame(base.tensors(), composite.tensors());
    }

    @Test
    void theSourcesAreACopy() {
        TensorDescriptor a = descriptor("text/layers/0/mlp/down");
        HashMap<String, Path> sources = new HashMap<>();
        sources.put(a.name(), OTHER);
        Artifact composite = new Artifact(null, null, new TensorDescriptor[] {a}).withSources(sources);
        sources.clear();
        assertEquals(OTHER, composite.pathOf(MAIN, a));
    }

    @Test
    void theDraftDepthRoundTripsThroughTheMetadata() throws Exception {
        Qwen38Config with = config().withDraftDepth(4);
        assertEquals(OptionalInt.of(4), with.draftDepth());
        Qwen38Config decoded = ArtifactCodec.decodeMetadata(ArtifactCodec.encodeMetadata(with));
        assertEquals(OptionalInt.of(4), decoded.draftDepth());
        assertEquals(with.mtpLayerCount(), decoded.mtpLayerCount());
        assertArrayEquals(with.layerTypes(), decoded.layerTypes());
    }

    @Test
    void anArtifactWithoutADraftDepthEncodesAsBeforeAndDecodesWithoutOne() throws Exception {
        Qwen38Config without = config();
        assertEquals(OptionalInt.empty(), without.draftDepth());
        byte[] plain = ArtifactCodec.encodeMetadata(without);
        // The depth is one trailing int32, written only when the artifact has one.
        assertEquals(plain.length + Integer.BYTES, ArtifactCodec.encodeMetadata(without.withDraftDepth(2)).length);
        assertEquals(OptionalInt.empty(), ArtifactCodec.decodeMetadata(plain).draftDepth());
    }

    @Test
    void aDraftDepthOutsideTheSupportedRangeIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> config().withDraftDepth(0));
        assertThrows(IllegalArgumentException.class, () -> config().withDraftDepth(17));
        assertThrows(
                ArtifactFormatException.class,
                () -> ArtifactCodec.decodeMetadata(trailing(ArtifactCodec.encodeMetadata(config()), 0)));
        assertThrows(
                ArtifactFormatException.class,
                () -> ArtifactCodec.decodeMetadata(trailing(ArtifactCodec.encodeMetadata(config()), 99)));
    }

    private static byte[] trailing(byte[] metadata, int depth) {
        ByteBuffer buffer = ByteBuffer.allocate(metadata.length + Integer.BYTES);
        buffer.put(metadata).putInt(depth);
        return buffer.array();
    }
}
