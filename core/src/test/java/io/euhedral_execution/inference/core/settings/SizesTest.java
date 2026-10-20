package io.euhedral_execution.inference.core.settings;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.time.Duration;
import org.junit.jupiter.api.Test;

class SizesTest {

    @Test
    void sizes() {
        assertEquals(4294967296L, Sizes.parseSize("4GiB"));
        assertEquals(734003200L, Sizes.parseSize("700MiB"));
        assertEquals(1048576L, Sizes.parseSize("1048576"));
        assertEquals(1024L, Sizes.parseSize("1KiB"));
    }

    @Test
    void badSizes() {
        assertThrows(IllegalArgumentException.class, () -> Sizes.parseSize("1.5GiB"));
        assertThrows(IllegalArgumentException.class, () -> Sizes.parseSize("4GB"));
        assertThrows(IllegalArgumentException.class, () -> Sizes.parseSize("99999999999TiB"));
    }

    @Test
    void durations() {
        assertEquals(Duration.ofSeconds(10), Sizes.parseDuration("10s"));
        assertEquals(Duration.ofMinutes(30), Sizes.parseDuration("30m"));
        assertEquals(Duration.ofSeconds(1), Sizes.parseDuration("1000ms"));
        assertEquals(Duration.ZERO, Sizes.parseDuration("0"));
    }

    @Test
    void badDurations() {
        assertThrows(IllegalArgumentException.class, () -> Sizes.parseDuration("10"));
        assertThrows(IllegalArgumentException.class, () -> Sizes.parseDuration("1.5s"));
    }
}
