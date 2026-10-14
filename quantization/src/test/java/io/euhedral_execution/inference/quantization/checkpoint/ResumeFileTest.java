package io.euhedral_execution.inference.quantization.checkpoint;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The resume file keeps its header and sections across reopening, and overwriting it keeps one file.
class ResumeFileTest {

    @Test
    void overwritesInPlace(@TempDir Path directory) throws Exception {
        Path path = directory.resolve("resume.bin");
        long[] first = {1, 2, 3}, second = {7, 8, 9};
        for (long[] values : new long[][] {first, second}) {
            try (Arena arena = Arena.ofConfined();
                    ResumeFile file = new ResumeFile(path)) {
                file.header(new Properties());
                long end = file.write(ResumeFile.HEADER, arena.allocateFrom(ValueLayout.JAVA_LONG, values));
                assertEquals(ResumeFile.HEADER + 24, end);
                Properties header = new Properties();
                header.setProperty("step", Long.toString(values[0]));
                file.header(header);
            }
        }
        try (Arena arena = Arena.ofConfined();
                ResumeFile file = new ResumeFile(path)) {
            assertEquals("7", file.header().getProperty("step"));
            MemorySegment read = arena.allocate(ValueLayout.JAVA_LONG, 3);
            file.read(ResumeFile.HEADER, read);
            assertArrayEquals(second, read.toArray(ValueLayout.JAVA_LONG));
        }
        assertEquals(ResumeFile.HEADER + 24, Files.size(path));
        try (var files = Files.list(directory)) {
            assertEquals(1, files.count());
        }
    }

    @Test
    void newFileHasEmptyHeader(@TempDir Path directory) throws Exception {
        try (ResumeFile file = new ResumeFile(directory.resolve("resume.bin"))) {
            assertTrue(file.header().isEmpty());
        }
    }
}
