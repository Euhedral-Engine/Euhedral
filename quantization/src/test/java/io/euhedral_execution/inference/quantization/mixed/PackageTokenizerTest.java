package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.artifact.TokenizerSection;
import io.euhedral_execution.inference.core.model.ArtifactTokenizer;
import io.euhedral_execution.inference.core.tokenizer.TokenizerFiles;
import io.euhedral_execution.inference.quantization.PackageTokenizer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.Isolated;

/// `package-tokenizer` on an artifact from before artifacts carried a tokenizer, with the checkpoint's real files
/// (`-Peuhedral.qwen.tokenizer-dir`; skipped without them). Alone: the real tokenizer takes much of a test JVM's heap.
@Isolated
class PackageTokenizerTest {

    @TempDir
    Path directory;

    @Test
    void anArtifactGainsTheCheckpointsTokenizerAsACopyOrInPlace() throws Exception {
        Path checkpoint =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "unset-euhedral.qwen.tokenizer-dir"));
        assumeTrue(Files.isRegularFile(checkpoint.resolve("tokenizer.json")), "no tokenizer at " + checkpoint);
        Path artifact = TinyArtifacts.write(this.directory.resolve("old.edrl"), 3, true);
        try (FileChannel channel = FileChannel.open(artifact, StandardOpenOption.WRITE)) {
            channel.truncate(ArtifactTokenizer.contentEnd(artifact));
        }
        byte[] content = Files.readAllBytes(artifact);
        TokenizerFiles expected = TokenizerFiles.fromCheckpoint(checkpoint);

        Path copy = this.directory.resolve("new.edrl");
        PackageTokenizer.run(artifact, checkpoint, copy);
        assertArrayEquals(content, Files.readAllBytes(artifact), "the source is left alone");
        assertFalse(Files.exists(this.directory.resolve("new.edrl.partial")));
        assertEquals(expected, TokenizerSection.require(copy));
        byte[] copied = Files.readAllBytes(copy);
        assertArrayEquals(content, Arrays.copyOf(copied, content.length));

        PackageTokenizer.run(artifact, checkpoint, null);
        assertArrayEquals(copied, Files.readAllBytes(artifact));
        assertEquals(Optional.of(expected), TokenizerSection.read(artifact));
    }
}
