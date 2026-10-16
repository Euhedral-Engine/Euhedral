package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SourcesTest {

    @TempDir
    Path directory;

    private Map<Option, Path> pair() throws Exception {
        return Map.of(
                Option.TRELLIS_3_0, TinyArtifacts.write(directory.resolve("trellis.edrl"), 1, true),
                Option.NVFP4, TinyArtifacts.write(directory.resolve("nvfp4.edrl"), 2, false));
    }

    @Test
    void sourcesEqualOutsideTheProjectionsOpen() throws Exception {
        Sources sources = Sources.open(pair(), Option.TRELLIS_3_0);
        assertEquals(TinyArtifacts.PROJECTIONS, sources.projections());
        assertEquals(List.of(TinyArtifacts.DRAFTER_PROJECTION), sources.drafterProjections());
        assertEquals(Option.TRELLIS_3_0, sources.base());
    }

    @Test
    void headsAndTheDrafterMayDifferBetweenSources() throws Exception {
        // the fixtures fill the token embedding, output head and drafter from each source's own seed
        Sources sources = Sources.open(pair(), Option.NVFP4);
        assertFalse(sources.projections().contains("text/output_head"));
        assertFalse(sources.projections().contains("text/token_embedding"));
    }

    @Test
    void companionsTravelWithTheirObject() throws Exception {
        Sources sources = Sources.open(pair(), Option.NVFP4);
        String name = "text/layers/0/mlp/down";
        assertEquals(
                List.of(name, name + ".suh", name + ".svh", name + ".segments"),
                sources.objectWithCompanions(Option.TRELLIS_3_0, name).stream()
                        .map(TensorDescriptor::name)
                        .toList());
        assertEquals(
                List.of(name),
                sources.objectWithCompanions(Option.NVFP4, name).stream()
                        .map(TensorDescriptor::name)
                        .toList());
        assertEquals(
                sources.artifact(Option.TRELLIS_3_0).tensors().length - 15,
                sources.artifact(Option.NVFP4).tensors().length);
        assertTrue(sources.projections().stream().noneMatch(n -> n.endsWith(".suh")));
    }

    @Test
    void aDifferingNormIsRefusedNamingIt() throws Exception {
        Path other = TinyArtifacts.write(directory.resolve("b.edrl"), 2, false);
        TensorDescriptor norm = Arrays.stream(ArtifactReader.read(other).tensors())
                .filter(tensor -> tensor.name().equals("text/layers/0/input_norm"))
                .findFirst()
                .orElseThrow();
        try (FileChannel channel = FileChannel.open(other, StandardOpenOption.WRITE)) {
            channel.write(ByteBuffer.wrap(new byte[] {(byte) 0x5a}), norm.dataOffset() + 3);
            channel.force(false);
        }
        Map<Option, Path> artifacts = Map.of(
                Option.TRELLIS_3_0, TinyArtifacts.write(directory.resolve("a.edrl"), 1, true), Option.NVFP4, other);
        var failure = assertThrows(IllegalArgumentException.class, () -> Sources.open(artifacts, Option.TRELLIS_3_0));
        assertTrue(failure.getMessage().contains("text/layers/0/input_norm"), failure.getMessage());
    }

    @Test
    void differingProjectionSetsAreRefused() throws Exception {
        // a source whose projections lack the companions-free 1/mlp/down cannot be built from the fixture, so the
        // sets differ through a second base: a source against itself with a layer removed is built by name instead
        Map<Option, Path> artifacts = Map.of(
                Option.TRELLIS_3_0, TinyArtifacts.write(directory.resolve("a.edrl"), 1, true),
                Option.NVFP4, TinyArtifacts.writeWithout(directory.resolve("b.edrl"), 2, "text/layers/1/mlp/down"));
        var failure = assertThrows(IllegalArgumentException.class, () -> Sources.open(artifacts, Option.TRELLIS_3_0));
        assertTrue(failure.getMessage().contains("text/layers/1/mlp/down"), failure.getMessage());
    }

    @Test
    void theBaseMustHaveASource() throws Exception {
        Map<Option, Path> artifacts = Map.of(Option.NVFP4, TinyArtifacts.write(directory.resolve("b.edrl"), 2, false));
        assertThrows(IllegalArgumentException.class, () -> Sources.open(artifacts, Option.TRELLIS_3_0));
    }
}
