package io.euhedral_execution.inference.core.settings;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.ProcessorTopology;
import java.time.Duration;
import java.util.BitSet;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class HostSettingsTest {
    private static HostSettings with(Optional<String> cpus) {
        return new HostSettings(cpus, 8, Duration.ofSeconds(1), Optional.empty());
    }

    @Test
    void parsesProcessorIdsAndInclusiveRanges() {
        BitSet expected = new BitSet();
        expected.set(2, 6);
        expected.set(8);
        assertEquals(expected, with(Optional.of(" 2-5, 8 ,")).selectedCpus());
        assertEquals(
                BitSet.valueOf(new long[] {1L << 3}), with(Optional.of("3")).selectedCpus());
    }

    @Test
    void rejectsInvalidProcessorLists() {
        for (String invalid : new String[] {"5-2", "-1", "a", " , "})
            assertThrows(
                    IllegalArgumentException.class,
                    () -> with(Optional.of(invalid)).selectedCpus(),
                    invalid);
    }

    @Test
    void unsetCpusAreTheOnesTheQuotaPaysFor() {
        assertEquals(
                ProcessorTopology.system().withinQuota().processorIds(),
                with(Optional.empty()).selectedCpus());
    }
}
