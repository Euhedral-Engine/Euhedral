package io.euhedral_execution.inference.core.model_loader.qwen4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactFormatException;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class Qwen4ArtifactReaderTest {

    @TempDir
    Path directory;

    Path path;
    Qwen4Artifact artifact;

    @BeforeEach
    void write() throws IOException {
        this.path = this.directory.resolve("mini.edrl");
        this.artifact = Qwen4TestArtifact.write(this.path, Qwen4TestArtifact.miniConfig(), 7);
    }

    @Test
    void roundTripsConfigurationInventoryAndBanks() throws IOException {
        Qwen4Artifact read = Qwen4ArtifactReader.read(this.path);
        assertEquals(Qwen4Header.VERSION, read.header().version());
        Qwen4Config expected = Qwen4TestArtifact.miniConfig();
        assertEquals(expected.toMetadata().keySet(), read.config().toMetadata().keySet());
        assertArrayEquals(
                Qwen4Metadata.encode(expected.toMetadata()),
                Qwen4Metadata.encode(read.config().toMetadata()));
        assertEquals(4 + 1, read.banks().length);
        assertEquals(8, read.banks()[0].expertCount());
        assertEquals(Qwen4Inventory.expected(expected).tensors().size(), read.tensors().length);
        Qwen4Validator.validateInventory(read);
        var report = Qwen4Validator.verifyChecksums(this.path, read, 2);
        assertEquals(5 * 8, report.expertRecords());
        assertTrue(report.checkedBytes() > 0);
    }

    @Test
    void expertRecordsAreUniformContiguousAndAligned() throws IOException {
        for (ExpertBank bank : this.artifact.banks()) {
            long stride = bank.recordBytes(0);
            assertEquals(0, stride % 4096);
            for (int e = 1; e < bank.expertCount(); e++) {
                assertEquals(bank.fileOffset(e - 1) + stride, bank.fileOffset(e));
                assertEquals(stride, bank.recordBytes(e));
            }
            assertEquals(stride, bank.maxRecordBytes());
        }
    }

    @Test
    void groupsFollowTheInventory() {
        assertEquals(
                ComponentGroup.ROUTED_EXPERT,
                this.artifact.bank("text/layers/0/moe/experts").orElseThrow().group());
        assertEquals(
                ComponentGroup.MTP,
                this.artifact.bank("mtp/layers/0/moe/experts").orElseThrow().group());
        assertEquals(
                ComponentGroup.NGRAM,
                this.artifact
                        .tensor("text/layers/1/ple/ngram/shard_003")
                        .orElseThrow()
                        .group());
        assertEquals(
                WeightLayout.ROW_INTERLEAVED_NVFP4_V1,
                this.artifact
                        .tensor("text/layers/1/ple/ngram/shard_000")
                        .orElseThrow()
                        .layout());
        assertEquals(
                ComponentGroup.GDN,
                this.artifact
                        .tensor("text/layers/0/gdn/in_proj_qkv")
                        .orElseThrow()
                        .group());
        assertEquals(
                ComponentGroup.QSA,
                this.artifact
                        .tensor("text/layers/2/attention/q_proj")
                        .orElseThrow()
                        .group());
        assertEquals(
                ComponentGroup.ROUTER,
                this.artifact.tensor("text/layers/3/moe/router").orElseThrow().group());
        assertEquals(
                ComponentGroup.VISION,
                this.artifact.tensor("vision/pos_embed/weight").orElseThrow().group());
    }

    @Test
    void rejectsBadMagicVersionAndArchitecture() throws IOException {
        patchInt(0, 0x12345678);
        assertRejected("magic");
        patchInt(0, Qwen4Header.MAGIC);
        patchInt(4, 2);
        assertRejected("version 2");
        patchInt(4, Qwen4Header.VERSION);
        patchInt(8, 7);
        assertRejected("architecture");
        patchInt(8, 1);
        patchInt(12, 1);
        assertRejected("reserved");
    }

    @Test
    void rejectsTruncation() throws IOException {
        try (RandomAccessFile file = new RandomAccessFile(this.path.toFile(), "rw")) {
            file.setLength(file.length() - 4096);
        }
        assertRejected("file holds");
        Files.write(this.path, new byte[10]);
        assertRejected("truncated");
    }

    @Test
    void rejectsAnObjectBeyondTheFileOrOverlappingAnother() throws IOException {
        Qwen4Tensor first = this.artifact.tensors()[0];
        Qwen4Tensor second = this.artifact.tensors()[1];
        Path overlapped = this.directory.resolve("overlap.edrl");
        Qwen4Tensor[] tensors = this.artifact.tensors().clone();
        tensors[1] = new Qwen4Tensor(
                second.name(),
                second.shape(),
                second.dataType(),
                second.format(),
                second.layout(),
                second.group(),
                first.dataOffset(),
                second.byteSize(),
                second.crc32());
        rewrite(overlapped, tensors, this.artifact.banks());
        assertThrows(QwenArtifactFormatException.class, () -> Qwen4ArtifactReader.read(overlapped));

        Path misaligned = this.directory.resolve("misaligned.edrl");
        tensors = this.artifact.tensors().clone();
        tensors[1] = new Qwen4Tensor(
                second.name(),
                second.shape(),
                second.dataType(),
                second.format(),
                second.layout(),
                second.group(),
                second.dataOffset() + 8,
                second.byteSize(),
                second.crc32());
        rewrite(misaligned, tensors, this.artifact.banks());
        var failure = assertThrows(QwenArtifactFormatException.class, () -> Qwen4ArtifactReader.read(misaligned));
        assertTrue(
                failure.getMessage().contains("aligned") || failure.getMessage().contains("overlap"),
                failure.getMessage());

        Path beyond = this.directory.resolve("beyond.edrl");
        tensors = this.artifact.tensors().clone();
        tensors[1] = new Qwen4Tensor(
                second.name(),
                second.shape(),
                second.dataType(),
                second.format(),
                second.layout(),
                second.group(),
                this.artifact.header().fileSize() - 256,
                second.byteSize(),
                second.crc32());
        rewrite(beyond, tensors, this.artifact.banks());
        assertThrows(QwenArtifactFormatException.class, () -> Qwen4ArtifactReader.read(beyond));
    }

    @Test
    void rejectsASizeThatDoesNotMatchTheShape() throws IOException {
        Qwen4Tensor victim = this.artifact.tensor("text/layers/0/gdn/in_proj_z").orElseThrow();
        Path wrong = this.directory.resolve("size.edrl");
        Qwen4Tensor[] tensors = this.artifact.tensors().clone();
        for (int i = 0; i < tensors.length; i++)
            if (tensors[i].name().equals(victim.name()))
                tensors[i] = new Qwen4Tensor(
                        victim.name(),
                        victim.shape(),
                        victim.dataType(),
                        victim.format(),
                        victim.layout(),
                        victim.group(),
                        victim.dataOffset(),
                        victim.byteSize() - 256,
                        victim.crc32());
        rewrite(wrong, tensors, this.artifact.banks());
        var failure = assertThrows(QwenArtifactFormatException.class, () -> Qwen4ArtifactReader.read(wrong));
        assertTrue(failure.getMessage().contains("does not match"), failure.getMessage());
    }

    @Test
    void validatorNamesMissingUnexpectedAndMisshapenObjects() throws IOException {
        Qwen4Tensor[] tensors = this.artifact.tensors();
        Qwen4Tensor[] fewer = java.util.Arrays.stream(tensors)
                .filter(t -> !t.name().equals("text/layers/3/moe/router"))
                .toArray(Qwen4Tensor[]::new);
        Qwen4Artifact missing =
                new Qwen4Artifact(this.artifact.header(), this.artifact.config(), fewer, this.artifact.banks());
        var failure = assertThrows(QwenArtifactFormatException.class, () -> Qwen4Validator.validateInventory(missing));
        assertTrue(failure.getMessage().contains("missing object text/layers/3/moe/router"), failure.getMessage());

        Qwen4Tensor router = this.artifact.tensor("text/layers/0/moe/router").orElseThrow();
        Qwen4Tensor[] renamed = tensors.clone();
        for (int i = 0; i < renamed.length; i++)
            if (renamed[i].name().equals(router.name()))
                renamed[i] = new Qwen4Tensor(
                        "text/layers/0/moe/rooter",
                        router.shape(),
                        router.dataType(),
                        router.format(),
                        router.layout(),
                        router.group(),
                        router.dataOffset(),
                        router.byteSize(),
                        router.crc32());
        Qwen4Artifact unexpected =
                new Qwen4Artifact(this.artifact.header(), this.artifact.config(), renamed, this.artifact.banks());
        failure = assertThrows(QwenArtifactFormatException.class, () -> Qwen4Validator.validateInventory(unexpected));
        assertTrue(failure.getMessage().contains("unexpected object text/layers/0/moe/rooter"), failure.getMessage());
    }

    @Test
    void validatorRejectsAWrongExpertCount() {
        ExpertBank[] banks = this.artifact.banks().clone();
        ExpertBank bank = banks[0];
        long[] offsets = java.util.Arrays.copyOf(new long[] {bank.fileOffset(0), bank.fileOffset(1)}, 2);
        banks[0] = new ExpertBank(
                bank.name(),
                bank.group(),
                bank.layer(),
                bank.projections(),
                offsets,
                new long[] {bank.recordBytes(0), bank.recordBytes(1)},
                new int[2]);
        Qwen4Artifact fewer =
                new Qwen4Artifact(this.artifact.header(), this.artifact.config(), this.artifact.tensors(), banks);
        var failure = assertThrows(QwenArtifactFormatException.class, () -> Qwen4Validator.validateInventory(fewer));
        assertTrue(failure.getMessage().contains("has 2 experts, expected 8"), failure.getMessage());
    }

    @Test
    void checksumVerificationDetectsACorruptedExpertRecord() throws IOException {
        ExpertBank bank = this.artifact.banks()[2];
        try (RandomAccessFile file = new RandomAccessFile(this.path.toFile(), "rw")) {
            file.seek(bank.fileOffset(5) + 100);
            int value = file.read();
            file.seek(bank.fileOffset(5) + 100);
            file.write(value ^ 0x10);
        }
        Qwen4Artifact read = Qwen4ArtifactReader.read(this.path);
        var failure = assertThrows(IOException.class, () -> Qwen4Validator.verifyChecksums(this.path, read, 2));
        assertTrue(failure.getMessage().contains(bank.name() + "#5"), failure.getMessage());
    }

    @Test
    void checksumVerificationRejectsANanBlockScaleEvenWithAMatchingChecksum() throws IOException {
        // The writer records the CRC-32 of whatever it is given, so a bad scale arrives with a good checksum.
        Qwen4Config config = Qwen4TestArtifact.miniConfig();
        Path poisoned = this.directory.resolve("nan.edrl");
        Qwen4Artifact fresh = Qwen4TestArtifact.write(poisoned, config, 7);
        ExpertBank bank = fresh.banks()[1];
        long scaleOffset = bank.fileOffset(3)
                + bank.projection("down").recordOffset()
                + io.euhedral_execution.inference.core.model_loader.artifact.Nvfp4Layout.scaleOffset(
                        bank.projection("down").shape()[0],
                        bank.projection("down").shape()[1]);
        byte[] record = new byte[(int) bank.recordBytes(3)];
        try (RandomAccessFile file = new RandomAccessFile(poisoned.toFile(), "rw")) {
            file.seek(scaleOffset);
            file.write(0xff);
            file.seek(bank.fileOffset(3));
            file.readFully(record);
        }
        int[] crcs = new int[bank.expertCount()];
        long[] offsets = new long[bank.expertCount()];
        long[] sizes = new long[bank.expertCount()];
        for (int e = 0; e < crcs.length; e++) {
            crcs[e] = e == 3 ? Qwen4ArtifactWriter.crc(record) : bank.crc32(e);
            offsets[e] = bank.fileOffset(e);
            sizes[e] = bank.recordBytes(e);
        }
        ExpertBank[] banks = fresh.banks().clone();
        banks[1] = new ExpertBank(bank.name(), bank.group(), bank.layer(), bank.projections(), offsets, sizes, crcs);
        byte[] tables = Qwen4ArtifactWriter.encodeTables(fresh.tensors(), banks);
        try (RandomAccessFile file = new RandomAccessFile(poisoned.toFile(), "rw")) {
            file.seek(fresh.header().tablesOffset());
            file.write(tables);
        }
        Qwen4Artifact read = Qwen4ArtifactReader.read(poisoned);
        var failure = assertThrows(IOException.class, () -> Qwen4Validator.verifyChecksums(poisoned, read, 2));
        assertTrue(failure.getMessage().contains("NaN"), failure.getMessage());
        assertTrue(failure.getMessage().contains(bank.name() + "#3"), failure.getMessage());
    }

    @Test
    void metadataRejectsUnknownMissingAndMistypedKeys() throws IOException {
        Map<String, Object> metadata =
                new LinkedHashMap<>(Qwen4TestArtifact.miniConfig().toMetadata());
        Qwen4Config.fromMetadata(Qwen4Metadata.decode(Qwen4Metadata.encode(metadata)));

        Map<String, Object> extra = new LinkedHashMap<>(metadata);
        extra.put("moe.router_bias", 1L);
        var failure = assertThrows(
                QwenArtifactFormatException.class,
                () -> Qwen4Config.fromMetadata(Qwen4Metadata.decode(Qwen4Metadata.encode(extra))));
        assertTrue(failure.getMessage().contains("unknown keys"), failure.getMessage());

        Map<String, Object> missing = new LinkedHashMap<>(metadata);
        missing.remove("moe.num_experts");
        failure = assertThrows(
                QwenArtifactFormatException.class,
                () -> Qwen4Config.fromMetadata(Qwen4Metadata.decode(Qwen4Metadata.encode(missing))));
        assertTrue(failure.getMessage().contains("moe.num_experts"), failure.getMessage());

        Map<String, Object> mistyped = new LinkedHashMap<>(metadata);
        mistyped.put("moe.num_experts", "eight");
        failure = assertThrows(
                QwenArtifactFormatException.class,
                () -> Qwen4Config.fromMetadata(Qwen4Metadata.decode(Qwen4Metadata.encode(mistyped))));
        assertTrue(failure.getMessage().contains("expected Long"), failure.getMessage());

        Map<String, Object> inconsistent = new LinkedHashMap<>(metadata);
        inconsistent.put("ngram.heads_offsets", new long[] {0, 40});
        failure = assertThrows(
                QwenArtifactFormatException.class,
                () -> Qwen4Config.fromMetadata(Qwen4Metadata.decode(Qwen4Metadata.encode(inconsistent))));
        assertTrue(failure.getMessage().contains("cumulative"), failure.getMessage());
        assertFalse(failure.getMessage().isEmpty());
    }

    private void assertRejected(String fragment) {
        var failure = assertThrows(IOException.class, () -> Qwen4ArtifactReader.read(this.path));
        assertTrue(failure.getMessage().contains(fragment), failure.getMessage());
    }

    private void patchInt(int offset, int value) throws IOException {
        try (RandomAccessFile file = new RandomAccessFile(this.path.toFile(), "rw")) {
            file.seek(offset);
            file.writeInt(value);
        }
    }

    /// Writes the tables of `tensors`/`banks` over a copy of the artifact's file.
    private void rewrite(Path target, Qwen4Tensor[] tensors, ExpertBank[] banks) throws IOException {
        Files.copy(this.path, target);
        byte[] tables = Qwen4ArtifactWriter.encodeTables(tensors, banks);
        try (RandomAccessFile file = new RandomAccessFile(target.toFile(), "rw")) {
            file.seek(this.artifact.header().tablesOffset());
            file.write(tables);
        }
    }
}
