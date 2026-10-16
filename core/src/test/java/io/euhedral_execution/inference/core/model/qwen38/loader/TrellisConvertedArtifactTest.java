package io.euhedral_execution.inference.core.model.qwen38.loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The converter's output, read by the engine: tools/test_trellis_convert.py writes the header, metadata and tensor
/// table of a Swift-1.5 trellis artifact and the bytes of every `.segments` region (the payloads are left out).
/// The file is rebuilt here with the regions at their offsets, extended sparsely to the table's data end, opened by
/// the artifact reader, and checked by the loader's descriptor, companion and segment rules. Nothing is uploaded.
class TrellisConvertedArtifactTest {

    @TempDir
    Path directory;

    private Path rebuild() throws Exception {
        ByteBuffer fixture;
        try (var stream = getClass().getResourceAsStream("/trellis-artifact-fixture.bin")) {
            fixture = ByteBuffer.wrap(stream.readAllBytes());
        }
        int prefix = (int) fixture.getLong(40); // the header's data offset ends the table
        Path file = this.directory.resolve("converted.edrl");
        Files.write(file, java.util.Arrays.copyOf(fixture.array(), prefix));
        try (RandomAccessFile out = new RandomAccessFile(file.toFile(), "rw")) {
            fixture.position(prefix);
            out.setLength(fixture.getLong());
            while (fixture.hasRemaining()) {
                long offset = fixture.getLong();
                byte[] region = new byte[fixture.getInt()];
                fixture.get(region);
                out.seek(offset);
                out.write(region);
            }
        }
        return file;
    }

    @Test
    void aConvertedArtifactPassesTheLoadersChecksBeforeAnyUpload() throws Exception {
        Path file = rebuild();
        Artifact artifact = ArtifactReader.read(file);
        Map<String, TensorDescriptor> descriptors = new LinkedHashMap<>();
        for (TensorDescriptor tensor : artifact.tensors()) descriptors.put(tensor.name(), tensor);

        long trellis = descriptors.values().stream()
                .filter(d -> d.format() == WeightFormat.TRELLIS)
                .count();
        assertEquals(325, trellis);
        assertEquals(785 + 3 * trellis, descriptors.size());
        CompactWeightLoader.validateInventory(artifact.config(), descriptors);
        Map<String, int[]> ends = CompactWeightLoader.validateTrellisSegments(file, descriptors);
        assertEquals(trellis, ends.size());
        assertEquals(4, ends.get("mtp/layer/attention/query_key_gate_value").length);
        assertTrue(ends.get("text/layers/3/attention/query_key").length == 2);
        assertEquals(
                ArtifactProfile.Quantization.TRELLIS,
                ArtifactProfile.of(artifact).quantization());
    }
}
