package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import org.junit.jupiter.api.Test;

class InferenceConfigTest {
    private static BitSet cpus() {
        BitSet cpus = new BitSet();
        cpus.set(0);
        return cpus;
    }

    private static InferenceConfig withCache(long bytes, int interval) {
        return new InferenceConfig(
                Path.of("a"), Path.of("t"), Path.of("l"), cpus(), 32768, Duration.ofSeconds(1), bytes, interval);
    }

    @Test
    void programmaticConstructorsLeaveThePrefixCacheOff() {
        var config = new InferenceConfig(Path.of("a"), Path.of("t"), Path.of("l"), cpus(), Duration.ofSeconds(1));
        assertEquals(0, config.prefixCacheBytes());
        assertEquals(InferenceConfig.DEFAULT_PREFIX_CACHE_CHECKPOINT_TOKENS, config.prefixCacheCheckpointTokens());
        var sixArguments =
                new InferenceConfig(Path.of("a"), Path.of("t"), Path.of("l"), cpus(), 4096, Duration.ofSeconds(1));
        assertEquals(0, sixArguments.prefixCacheBytes());
    }

    @Test
    void carriesTheCacheSettings() {
        var config = withCache(4L << 30, 4096);
        assertEquals(4L << 30, config.prefixCacheBytes());
        assertEquals(4096, config.prefixCacheCheckpointTokens());
        assertEquals(0, withCache(0, 2048).prefixCacheBytes());
    }

    @Test
    void rejectsANegativeBudgetAndAnIntervalOffTheChunkGrid() {
        assertThrows(IllegalArgumentException.class, () -> withCache(-1, 2048));
        assertThrows(IllegalArgumentException.class, () -> withCache(1 << 30, 0));
        assertThrows(IllegalArgumentException.class, () -> withCache(1 << 30, 1000));
        assertThrows(IllegalArgumentException.class, () -> withCache(1 << 30, -512));
    }
}
