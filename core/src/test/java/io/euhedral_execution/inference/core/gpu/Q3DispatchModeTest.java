package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.inference.core.InferenceRunSnapshot;
import io.euhedral_execution.inference.core.InferenceTuning;
import java.util.BitSet;
import org.junit.jupiter.api.Test;

class Q3DispatchModeTest {
    @Test
    void thresholdAndForcedPathsAreExplicit() {
        int threshold = Q3DispatchMode.DEFAULT_SMALL_ROW_THRESHOLD;
        assertEquals(Q3DispatchMode.DECODE, Q3DispatchMode.AUTO.select(threshold - 1, threshold));
        assertEquals(Q3DispatchMode.DECODE, Q3DispatchMode.AUTO.select(threshold, threshold));
        assertEquals(Q3DispatchMode.PREFILL, Q3DispatchMode.AUTO.select(threshold + 1, threshold));
        assertEquals(Q3DispatchMode.DECODE, Q3DispatchMode.AUTO.select(3, 4));
        assertEquals(Q3DispatchMode.DECODE, Q3DispatchMode.AUTO.select(4, 4));
        assertEquals(Q3DispatchMode.PREFILL, Q3DispatchMode.AUTO.select(5, 4));
        assertEquals(Q3DispatchMode.PREFILL, Q3DispatchMode.AUTO.select(1, 0));
        for (var mode : new Q3DispatchMode[] {Q3DispatchMode.SCALAR, Q3DispatchMode.DECODE, Q3DispatchMode.PREFILL}) {
            assertEquals(mode, mode.select(1, 4));
            assertEquals(mode, mode.select(512, 4));
        }
        assertThrows(IllegalArgumentException.class, () -> Q3DispatchMode.AUTO.select(0, 4));
        assertThrows(IllegalArgumentException.class, () -> Q3DispatchMode.AUTO.select(1, -1));
    }

    @Test
    void immutableTuningAndSnapshotsRetainKernelPolicies() {
        BitSet cpus = new BitSet();
        cpus.set(0);
        var tuning = InferenceTuning.defaults(cpus)
                .withQ3Dispatch(Q3DispatchMode.AUTO, 8)
                .withPrefillChunkTokens(256)
                .withWorkerProcessorIds(cpus);
        var snapshot = InferenceRunSnapshot.Tuning.of(tuning);
        assertEquals(Q3DispatchMode.AUTO, snapshot.q3DispatchMode());
        assertEquals(8, snapshot.q3SmallRowThreshold());
        assertThrows(IllegalArgumentException.class, () -> tuning.withQ3Dispatch(Q3DispatchMode.AUTO, -1));
        assertThrows(NullPointerException.class, () -> tuning.withQ3Dispatch(null, 4));
    }
}
