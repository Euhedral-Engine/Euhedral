package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import io.euhedral_execution.inference.core.model_loader.qwen4.ComponentGroup;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.zip.CRC32;

/// A plain file of deterministic pseudo-random expert records at the offsets an [ExpertBank] array describes,
/// built without the artifact reader: a junk header, then each bank's records, with random-length gaps between
/// unaligned records so offsets are arbitrary. The file is kept in memory as well, for byte comparisons.
final class ExpertFixture {
    /// One bank to write: record sizes and the alignment of each record's file offset.
    record Spec(String name, int[] sizes, int alignment) {
        static Spec uniform(String name, int experts, int size, int alignment) {
            int[] sizes = new int[experts];
            Arrays.fill(sizes, size);
            return new Spec(name, sizes, alignment);
        }
    }

    private static final long HEADER_BYTES = 123;

    final Path file;
    final ExpertBank[] banks;
    final byte[] bytes;

    private ExpertFixture(Path file, ExpertBank[] banks, byte[] bytes) {
        this.file = file;
        this.banks = banks;
        this.bytes = bytes;
    }

    /// Five banks: 8 experts of 9000 bytes; 3 experts of unequal sizes; 16 experts of 8192 bytes at 4096-aligned
    /// offsets; a single expert; 12 experts of 6144 bytes at 64-byte-aligned offsets.
    static ExpertFixture standard(Path directory, long seed) throws IOException {
        return create(
                directory,
                seed,
                List.of(
                        Spec.uniform("layer0", 8, 9000, 1),
                        new Spec("layer1", new int[] {6000, 9001, 7777}, 1),
                        Spec.uniform("layer2", 16, 8192, 4096),
                        Spec.uniform("layer3", 1, 5200, 1),
                        Spec.uniform("layer4", 12, 6144, 64)));
    }

    static ExpertFixture create(Path directory, long seed, List<Spec> specs) throws IOException {
        Random random = new Random(seed);
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        byte[] header = new byte[(int) HEADER_BYTES];
        random.nextBytes(header);
        out.write(header);
        List<ExpertBank> banks = new ArrayList<>();
        for (int ordinal = 0; ordinal < specs.size(); ordinal++) {
            Spec spec = specs.get(ordinal);
            long[] offsets = new long[spec.sizes().length];
            long[] sizes = new long[spec.sizes().length];
            int[] crcs = new int[spec.sizes().length];
            for (int expert = 0; expert < offsets.length; expert++) {
                int gap = spec.alignment() == 1 ? random.nextInt(14) : 0;
                gap += (int) ((spec.alignment() - (out.size() + gap) % spec.alignment()) % spec.alignment());
                byte[] filler = new byte[gap];
                random.nextBytes(filler);
                out.write(filler);
                byte[] record = new byte[spec.sizes()[expert]];
                random.nextBytes(record);
                offsets[expert] = out.size();
                sizes[expert] = record.length;
                CRC32 crc = new CRC32();
                crc.update(record);
                crcs[expert] = (int) crc.getValue();
                out.write(record);
            }
            banks.add(new ExpertBank(
                    spec.name(),
                    ComponentGroup.ROUTED_EXPERT,
                    ordinal,
                    List.of(projection("gate_up", 0, 4096), projection("down", 4096, 1024)),
                    offsets,
                    sizes,
                    crcs));
        }
        byte[] trailer = new byte[77];
        random.nextBytes(trailer);
        out.write(trailer);
        byte[] bytes = out.toByteArray();
        Path file = Files.createTempFile(directory, "experts", ".bin");
        Files.write(file, bytes);
        return new ExpertFixture(file, banks.toArray(ExpertBank[]::new), bytes);
    }

    private static ExpertProjection projection(String name, long offset, long size) {
        return new ExpertProjection(
                name,
                new long[] {size, 1},
                TensorDataType.UINT8,
                WeightFormat.NVFP4,
                WeightLayout.CONTIGUOUS_LE_V1,
                offset,
                size);
    }

    int expertCount(int bank) {
        return this.banks[bank].expertCount();
    }

    int totalExperts() {
        int total = 0;
        for (ExpertBank bank : this.banks) total += bank.expertCount();
        return total;
    }

    /// The record's bytes as written to the file.
    byte[] record(int bank, int expert) {
        long offset = this.banks[bank].fileOffset(expert);
        return Arrays.copyOfRange(this.bytes, (int) offset, (int) (offset + this.banks[bank].recordBytes(expert)));
    }

    static int crc32(byte[] data) {
        CRC32 crc = new CRC32();
        crc.update(data);
        return (int) crc.getValue();
    }

    long slotBytes() {
        return ExpertCache.slotBytesFor(this.banks);
    }
}
