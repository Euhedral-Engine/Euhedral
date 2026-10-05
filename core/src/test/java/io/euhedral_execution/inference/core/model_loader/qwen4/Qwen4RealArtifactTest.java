package io.euhedral_execution.inference.core.model_loader.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/// Load-only check of the real converted Flash-Next artifact (`-Peuhedral.qwen4.artifact=PATH`; skipped when the file
/// is
/// absent): the tables parse, the inventory is exactly the configuration's, and with `-Peuhedral.qwen4.verify=true`
/// every
/// range's CRC-32 matches.
class Qwen4RealArtifactTest {

    static Path artifactPath() {
        return Path.of(System.getProperty(
                "euhedral.qwen4.artifact", "/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl"));
    }

    @Test
    void realArtifactParsesAndMatchesItsInventory() throws IOException {
        Path path = artifactPath();
        assumeTrue(Files.isRegularFile(path), "no Flash-Next artifact at " + path);
        Qwen4Artifact artifact = Qwen4ArtifactReader.read(path);
        Qwen4Config config = artifact.config();
        assertEquals(48, config.text().numLayers());
        assertEquals(512, config.moe().numExperts());
        assertEquals(10, config.moe().expertsPerToken());
        assertEquals(36, config.gdnLayers());
        assertEquals(12, config.sparseAttentionLayers());
        assertEquals(262144, config.text().maxPositionEmbeddings());
        assertEquals(49, artifact.banks().length);
        long experts = 0;
        for (ExpertBank bank : artifact.banks()) {
            assertEquals(512, bank.expertCount());
            assertEquals(bank.recordBytes(0), bank.maxRecordBytes());
            experts += bank.expertCount();
        }
        assertEquals(49L * 512, experts);
        var report = Qwen4Validator.validateInventory(artifact);
        assertEquals(artifact.tensors().length, report.tensors());
        if (Boolean.getBoolean("euhedral.qwen4.verify")) {
            var verified = Qwen4Validator.verifyChecksums(path, artifact, 8);
            assertTrue(verified.checkedBytes() > 100_000_000_000L, "checked " + verified.checkedBytes());
        }
    }
}
