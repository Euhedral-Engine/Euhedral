package io.euhedral_execution.inference.quantization.checkpoint;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// A written file reads back through Safetensors: names, dtypes, shapes and bytes; the header is padded to 8 bytes;
/// large tensors are pulled from their source in chunks.
class SafetensorsWriterTest {

    @Test
    void roundTrips(@TempDir Path dir) throws Exception {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment words =
                    arena.allocateFrom(ValueLayout.JAVA_SHORT, (short) 1, (short) -2, (short) 3, (short) 4);
            MemorySegment scalar = arena.allocateFrom(ValueLayout.JAVA_INT, 0x83DCD12D);
            try (var writer = new SafetensorsWriter(dir.resolve("model.safetensors"))) {
                writer.add("x.trellis", "I16", new long[] {1, 1, 4}, words);
                writer.add("x.mul1", "I32", new long[] {}, scalar);
            }
            assertFalse(Files.exists(dir.resolve("model.safetensors.part")));
            try (Safetensors store = Safetensors.open(dir)) {
                var t = store.tensor("x.trellis");
                assertEquals("I16", t.dtype());
                assertArrayEquals(new long[] {1, 1, 4}, t.shape());
                MemorySegment back = arena.allocate(8);
                store.read("x.trellis", back);
                assertArrayEquals(words.toArray(ValueLayout.JAVA_SHORT), back.toArray(ValueLayout.JAVA_SHORT));
                assertEquals(0, store.tensor("x.mul1").shape().length);
                MemorySegment one = arena.allocate(4);
                store.read("x.mul1", one);
                assertEquals(0x83DCD12D, one.get(ValueLayout.JAVA_INT, 0));
            }
            ByteBuffer head = ByteBuffer.wrap(Files.readAllBytes(dir.resolve("model.safetensors")))
                    .order(ByteOrder.LITTLE_ENDIAN);
            assertEquals(0, head.getLong(0) % 8);
        }
    }

    @Test
    void streamsTensorsLargerThanOneChunk(@TempDir Path dir) throws Exception {
        long elements = 20L << 20; // 40 MB of F16, more than two chunks
        try (var writer = new SafetensorsWriter(dir.resolve("model.safetensors"))) {
            writer.add("big", "F16", new long[] {elements}, elements * 2, (offset, destination) -> {
                for (long i = 0; i < destination.byteSize() / 2; i++)
                    destination.setAtIndex(ValueLayout.JAVA_SHORT, i, (short) ((offset / 2 + i) * 31));
            });
            writer.add(
                    "tail",
                    "F32",
                    new long[] {1},
                    4,
                    (offset, destination) -> destination.set(ValueLayout.JAVA_INT, 0, 7));
        }
        try (Safetensors store = Safetensors.open(dir);
                Arena arena = Arena.ofConfined()) {
            for (long at : new long[] {0, 16L << 20, 33_554_000L, elements * 2 - 16}) {
                MemorySegment piece = arena.allocate(16);
                store.read("big", at, piece);
                for (int i = 0; i < 8; i++)
                    assertEquals((short) ((at / 2 + i) * 31), piece.getAtIndex(ValueLayout.JAVA_SHORT, i), "at " + at);
            }
            MemorySegment tail = arena.allocate(4);
            store.read("tail", tail);
            assertEquals(7, tail.get(ValueLayout.JAVA_INT, 0));
        }
    }

    @Test
    void rejectsInconsistentTensors(@TempDir Path dir) {
        var writer = new SafetensorsWriter(dir.resolve("model.safetensors"));
        writer.add("a", "F16", new long[] {2}, MemorySegment.ofArray(new short[2]));
        assertThrows(
                IllegalArgumentException.class,
                () -> writer.add("a", "F16", new long[] {2}, MemorySegment.ofArray(new short[2])));
        assertThrows(
                IllegalArgumentException.class,
                () -> writer.add("b", "F16", new long[] {3}, MemorySegment.ofArray(new short[2])));
        assertThrows(
                IllegalArgumentException.class,
                () -> writer.add("c", "F8", new long[] {2}, MemorySegment.ofArray(new short[2])));
    }

    @Test
    void padsTheHeaderOnItsEncodedLength(@TempDir Path dir) throws Exception {
        try (var writer = new SafetensorsWriter(dir.resolve("model.safetensors"))) {
            writer.add("t\u00e9nsor\u00e9", "F16", new long[] {1}, MemorySegment.ofArray(new short[1]));
        }
        ByteBuffer file = ByteBuffer.wrap(Files.readAllBytes(dir.resolve("model.safetensors")))
                .order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(0, file.getLong(0) % 8);
        try (Safetensors store = Safetensors.open(dir)) {
            assertEquals("F16", store.tensor("t\u00e9nsor\u00e9").dtype());
        }
    }

    @Test
    void aFailedWriteLeavesNoPartFile(@TempDir Path dir) {
        var writer = new SafetensorsWriter(dir.resolve("model.safetensors"));
        writer.add("a", "F16", new long[] {1}, 2, (offset, destination) -> {
            throw new java.io.IOException("boom");
        });
        assertThrows(java.io.IOException.class, writer::close);
        assertFalse(Files.exists(dir.resolve("model.safetensors.part")));
        assertFalse(Files.exists(dir.resolve("model.safetensors")));
    }
}
