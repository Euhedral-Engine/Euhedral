package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class AssembleTest {

    @TempDir
    Path directory;

    Path trellis, nvfp4;
    Sources sources;

    @BeforeEach
    void open() throws Exception {
        trellis = TinyArtifacts.write(directory.resolve("trellis.edrl"), 1, true);
        nvfp4 = TinyArtifacts.write(directory.resolve("nvfp4.edrl"), 2, false);
        sources = Sources.open(Map.of(Option.TRELLIS_3_0, trellis, Option.NVFP4, nvfp4), Option.TRELLIS_3_0);
    }

    private static Allocation allocation(String name, Option option) {
        return new Allocation(Map.of(name, option));
    }

    private static TensorDescriptor find(Artifact artifact, String name) {
        return Arrays.stream(artifact.tensors())
                .filter(tensor -> tensor.name().equals(name))
                .findFirst()
                .orElse(null);
    }

    @Test
    void viewRoutesChosenObjectsAndTheirCompanionsToTheirSource() throws Exception {
        String chosen = "text/layers/0/mlp/down", kept = "text/layers/0/mlp/gate_up";
        Artifact view = Assemble.view(sources, allocation(chosen, Option.NVFP4), OptionalInt.of(3));
        TensorDescriptor taken = find(view, chosen);
        assertEquals(sources.descriptor(Option.NVFP4, chosen), taken);
        assertEquals(nvfp4, view.pathOf(trellis, taken));
        // an NVFP4 object has no companions: the trellis ones for it are gone
        for (String suffix : List.of(".suh", ".svh", ".segments")) assertEquals(null, find(view, chosen + suffix));
        // an unlisted object, and the companions of one, stay with the base
        assertEquals(sources.descriptor(Option.TRELLIS_3_0, kept), find(view, kept));
        assertEquals(trellis, view.pathOf(nvfp4, find(view, kept + ".suh")));
        assertEquals(trellis, view.pathOf(nvfp4, find(view, "text/layers/0/input_norm")));
        assertEquals(3, view.config().draftDepth().getAsInt());
        assertEquals(sources.artifact(Option.TRELLIS_3_0).tensors().length - 3, view.tensors().length);
    }

    @Test
    void anObjectChosenFromATrellisSourceBringsItsCompanions() throws Exception {
        sources = Sources.open(Map.of(Option.TRELLIS_3_0, trellis, Option.NVFP4, nvfp4), Option.NVFP4);
        String chosen = "text/layers/1/attention/output";
        Artifact view = Assemble.view(sources, allocation(chosen, Option.TRELLIS_3_0), OptionalInt.empty());
        for (String suffix : List.of("", ".suh", ".svh", ".segments")) {
            TensorDescriptor tensor = find(view, chosen + suffix);
            assertEquals(sources.descriptor(Option.TRELLIS_3_0, chosen + suffix), tensor);
            assertEquals(trellis, view.pathOf(nvfp4, tensor));
        }
        assertTrue(view.config().draftDepth().isEmpty());
    }

    @Test
    void viewRefusesUnknownNonProjectionAndDrafterNamesAndUnsourcedOptions() {
        var unknown = assertThrows(
                IllegalArgumentException.class,
                () -> Assemble.view(sources, allocation("text/layers/9/mlp/down", Option.NVFP4), OptionalInt.empty()));
        assertTrue(unknown.getMessage().contains("text/layers/9/mlp/down"), unknown.getMessage());
        for (String name : List.of("text/layers/0/input_norm", "text/output_head", "text/layers/0/mlp/down.suh")) {
            var failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> Assemble.view(sources, allocation(name, Option.NVFP4), OptionalInt.empty()));
            assertTrue(failure.getMessage().contains(name), failure.getMessage());
        }
        var drafter = assertThrows(
                IllegalArgumentException.class,
                () -> Assemble.view(
                        sources, allocation(TinyArtifacts.DRAFTER_PROJECTION, Option.NVFP4), OptionalInt.empty()));
        assertTrue(drafter.getMessage().contains(TinyArtifacts.DRAFTER_PROJECTION), drafter.getMessage());
        var option = assertThrows(
                IllegalArgumentException.class,
                () -> Assemble.view(
                        sources, allocation("text/layers/0/mlp/down", Option.TRELLIS_4_0), OptionalInt.empty()));
        assertTrue(option.getMessage().contains("trellis-4.0"), option.getMessage());
    }

    @Test
    void writtenArtifactHasTheViewsDescriptorsAndBytes() throws Exception {
        var allocation = new Allocation(
                Map.of("text/layers/0/mlp/down", Option.NVFP4, "text/layers/1/attention/output", Option.NVFP4));
        Artifact view = Assemble.view(sources, allocation, OptionalInt.of(2));
        Path out = directory.resolve("mixed.edrl");
        // a copy chunk smaller than every payload: each is copied in several pieces
        Assemble.write(sources, allocation, OptionalInt.of(2), out, 100);
        assertFalse(java.nio.file.Files.exists(directory.resolve("mixed.edrl.partial")));
        Artifact written = ArtifactReader.read(out);
        assertEquals(2, written.config().draftDepth().getAsInt());
        assertEquals(view.tensors().length, written.tensors().length);
        for (int i = 0; i < view.tensors().length; i++) {
            TensorDescriptor expected = view.tensors()[i], actual = written.tensors()[i];
            assertEquals(expected.name(), actual.name());
            assertArrayEquals(expected.shape(), actual.shape());
            assertEquals(expected.dataType(), actual.dataType());
            assertEquals(expected.format(), actual.format());
            assertEquals(expected.layout(), actual.layout());
            assertEquals(expected.byteSize(), actual.byteSize());
            assertArrayEquals(read(view.pathOf(trellis, expected), expected), read(out, actual), expected.name());
        }
    }

    @Test
    void residentBytesIsTheChosenPayloadsPlusTheBaseRemainder() {
        String chosen = "text/layers/0/mlp/down";
        long base = sources.artifact(Option.TRELLIS_3_0).tensors().length;
        long all = Arrays.stream(sources.artifact(Option.TRELLIS_3_0).tensors())
                .mapToLong(TensorDescriptor::byteSize)
                .sum();
        assertEquals(all, Assemble.residentBytes(sources, new Allocation(Map.of())));
        long companions = Arrays.stream(sources.artifact(Option.TRELLIS_3_0).tensors())
                .filter(tensor -> tensor.name().startsWith(chosen + "."))
                .mapToLong(TensorDescriptor::byteSize)
                .sum();
        long baseObject = sources.descriptor(Option.TRELLIS_3_0, chosen).byteSize();
        long chosenObject = sources.descriptor(Option.NVFP4, chosen).byteSize();
        assertEquals(
                all - baseObject - companions + chosenObject,
                Assemble.residentBytes(sources, allocation(chosen, Option.NVFP4)));
        assertTrue(base > 0);
    }

    private static byte[] read(Path file, TensorDescriptor tensor) throws IOException {
        ByteBuffer bytes = ByteBuffer.allocate((int) tensor.byteSize());
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            channel.read(bytes, tensor.dataOffset());
        }
        return bytes.array();
    }
}
