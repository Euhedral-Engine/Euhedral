package io.euhedral_execution.inference.quantization;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The reference file the engine writes: llama.cpp's record encoding against its decoding (the one `score.cu`
/// applies), the header and tokens as [Scorer.Reference] reads them, and a write resumed after a stop.
class ReferenceWriterTest {

    @TempDir
    Path directory;

    private static float[] logits(long seed, int vocabulary, double spread) {
        var random = new Random(seed);
        float[] row = new float[vocabulary];
        for (int i = 0; i < vocabulary; i++) row[i] = (float) (random.nextGaussian() * spread);
        return row;
    }

    @Test
    void recordsRoundTripWithinTheQuantizationStep() {
        int vocabulary = 1001;
        float[] row = logits(3, vocabulary, 3);
        byte[] record = ReferenceWriter.encode(row);
        assertEquals(2 * (2 * ((vocabulary + 1) / 2) + 4), record.length);
        double[] decoded = ReferenceWriter.decode(record, vocabulary);
        double max = Double.NEGATIVE_INFINITY;
        for (float value : row) max = Math.max(max, value);
        double sum = 0;
        for (float value : row) sum += Math.exp(value - max);
        double logZ = max + Math.log(sum);
        float scale = java.nio.ByteBuffer.wrap(record)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                .getFloat(0);
        for (int i = 0; i < vocabulary; i++) {
            double expected = row[i] - logZ;
            if (expected > -16) assertEquals(expected, decoded[i], scale * 0.51 + 1e-5, "entry " + i);
            else assertEquals(-16, decoded[i], scale * 0.51 + 1e-5, "floored entry " + i);
        }
        // The distribution sums to one within the step.
        double mass = 0;
        for (double value : decoded) mass += Math.exp(value);
        assertEquals(1.0, mass, 1e-3);
    }

    @Test
    void theLowestLogProbabilityIsFlooredAtMinusSixteen() {
        float[] row = new float[8];
        row[0] = 30;
        byte[] record = ReferenceWriter.encode(row);
        double[] decoded = ReferenceWriter.decode(record, 8);
        for (int i = 1; i < 8; i++) assertEquals(-16, decoded[i], 1e-5);
        assertEquals(0, decoded[0], 1e-3);
    }

    @Test
    void aSingleValuedRowDecodesToTheUniformLogProbability() {
        double[] decoded = ReferenceWriter.decode(ReferenceWriter.encode(new float[5]), 5);
        for (double value : decoded) assertEquals(-Math.log(5), value, 1e-5);
    }

    @Test
    void theHeaderAndTokensReadBackAsALlamaCppFile() throws IOException {
        int context = 6, vocabulary = 11, chunks = 3;
        int[][] tokens = tokens(chunks, context, vocabulary);
        Path out = directory.resolve("ref.kld");
        try (ReferenceWriter writer = ReferenceWriter.open(out, context, vocabulary, tokens)) {
            assertEquals(0, writer.completed());
            for (int chunk = 0; chunk < chunks; chunk++) {
                for (int record = 0; record < writer.records(); record++)
                    writer.write(chunk, record, ReferenceWriter.encode(logits(chunk * 10 + record, vocabulary, 2)));
                writer.complete(chunk + 1);
            }
            writer.finish();
        }
        assertTrue(Files.exists(out));
        assertFalse(Files.exists(Path.of(out + ".partial")));
        try (Scorer.Reference reference = Scorer.Reference.open(out)) {
            assertEquals(context, reference.context);
            assertEquals(vocabulary, reference.vocabulary);
            assertEquals(chunks, reference.chunks);
            assertEquals(2, reference.records);
            for (int chunk = 0; chunk < chunks; chunk++) assertArrayEquals(tokens[chunk], reference.tokens[chunk]);
            assertEquals(reference.dataOffset + 2L * reference.width * reference.records * chunks, Files.size(out));
        }
        // Chunk 1, record 1 is where Scorer.Reference.read finds it.
        byte[] expected = ReferenceWriter.encode(logits(11, vocabulary, 2));
        byte[] actual = new byte[expected.length];
        try (Scorer.Reference reference = Scorer.Reference.open(out);
                Arena arena = Arena.ofConfined()) {
            MemorySegment memory = arena.allocate(expected.length);
            reference.read(1, 1, 1, memory.address());
            MemorySegment.copy(memory, ValueLayout.JAVA_BYTE, 0, actual, 0, actual.length);
        }
        assertArrayEquals(expected, actual);
    }

    @Test
    void aWriteStoppedAfterOneOfThreeChunksContinuesAtTheSecond() throws IOException {
        int context = 6, vocabulary = 11, chunks = 3;
        int[][] tokens = tokens(chunks, context, vocabulary);
        Path out = directory.resolve("ref.kld");
        try (ReferenceWriter writer = ReferenceWriter.open(out, context, vocabulary, tokens)) {
            for (int record = 0; record < writer.records(); record++)
                writer.write(0, record, ReferenceWriter.encode(logits(record, vocabulary, 2)));
            writer.complete(1);
            // Chunk 1 half written when the process stops: not counted.
            writer.write(1, 0, ReferenceWriter.encode(logits(99, vocabulary, 2)));
        }
        assertFalse(Files.exists(out));
        try (ReferenceWriter writer = ReferenceWriter.open(out, context, vocabulary, tokens)) {
            assertEquals(1, writer.completed());
            for (int chunk = 1; chunk < chunks; chunk++) {
                for (int record = 0; record < writer.records(); record++)
                    writer.write(chunk, record, ReferenceWriter.encode(logits(chunk * 10 + record, vocabulary, 2)));
                writer.complete(chunk + 1);
            }
            writer.finish();
        }
        Path whole = directory.resolve("whole.kld");
        try (ReferenceWriter writer = ReferenceWriter.open(whole, context, vocabulary, tokens)) {
            for (int chunk = 0; chunk < chunks; chunk++) {
                for (int record = 0; record < writer.records(); record++)
                    writer.write(
                            chunk,
                            record,
                            ReferenceWriter.encode(logits(chunk == 0 ? record : chunk * 10 + record, vocabulary, 2)));
                writer.complete(chunk + 1);
            }
            writer.finish();
        }
        assertArrayEquals(Files.readAllBytes(whole), Files.readAllBytes(out));
    }

    @Test
    void aPartialFileOfOtherTokensIsRefused() throws IOException {
        int[][] tokens = tokens(2, 6, 11);
        Path out = directory.resolve("ref.kld");
        try (ReferenceWriter writer = ReferenceWriter.open(out, 6, 11, tokens)) {
            writer.complete(0);
        }
        int[][] other = tokens(2, 6, 11);
        other[1][3] += 1;
        var failure = assertThrows(IllegalStateException.class, () -> ReferenceWriter.open(out, 6, 11, other));
        assertTrue(failure.getMessage().contains("ref.kld.partial"), failure.getMessage());
    }

    @Test
    void anExistingReferenceIsNotOverwritten() throws IOException {
        Path out = directory.resolve("ref.kld");
        Files.writeString(out, "x");
        assertThrows(IllegalStateException.class, () -> ReferenceWriter.open(out, 6, 11, tokens(1, 6, 11)));
    }

    private static int[][] tokens(int chunks, int context, int vocabulary) {
        int[][] tokens = new int[chunks][context];
        for (int c = 0; c < chunks; c++) for (int i = 0; i < context; i++) tokens[c][i] = (c * 7 + i * 3) % vocabulary;
        return tokens;
    }
}
