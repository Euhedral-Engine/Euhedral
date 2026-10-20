package io.euhedral_execution.inference.quantization.mtp;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CorpusTest {
    private static int[] ints(Path file) throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN);
        int[] values = new int[buffer.remaining() / 4];
        for (int i = 0; i < values.length; i++) values[i] = buffer.getInt();
        return values;
    }

    @Test
    void labelsAreKeptPerTokenAndAnOlderGreedyCorpusTakesItsTokensAsLabels(@TempDir Path directory) throws IOException {
        int width = 2;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment rows = arena.allocate(2L * width * 8);
            try (var writer = Corpus.Writer.open(directory, width)) {
                writer.write("greedy", new int[] {1, 2, 3, 4}, new int[] {1, 2, 3, 4}, 4, 2, false, rows, 3);
            }
            // A corpus from before labels were kept: no labels file.
            Files.delete(directory.resolve("labels.bin"));
            try (var writer = Corpus.Writer.open(directory, width)) {
                writer.write("sampled", new int[] {5, 6, 7}, new int[] {5, 9, 8}, 3, 1, true, rows, 2);
            }
        }
        assertArrayEquals(new int[] {1, 2, 3, 4, 5, 9, 8}, ints(directory.resolve("labels.bin")));
        assertArrayEquals(new int[] {1, 2, 3, 4, 5, 6, 7}, ints(directory.resolve("tokens.bin")));
        List<Corpus.Entry> entries = Corpus.entries(directory);
        assertEquals(2, entries.size());
        assertFalse(entries.get(0).sampled());
        assertTrue(entries.get(1).sampled());
        assertEquals(4, entries.get(1).token());
        assertEquals(3, entries.get(1).row());
    }
}
