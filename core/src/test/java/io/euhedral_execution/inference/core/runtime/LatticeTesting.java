package io.euhedral_execution.inference.core.runtime;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.hardware_utils.affinity.PinnedThreadExecutor;
import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import java.util.BitSet;
import java.util.concurrent.TimeUnit;

/// Helpers for tests that start a real Euhedral lattice.
public final class LatticeTesting {
    private LatticeTesting() {}

    /// Closes `lattice` and waits until the executors pinned to its CPUs terminated. A lattice's close returns
    /// while its worker threads still finish; a lattice started on the same CPUs before then is refused ("a closed
    /// executor still owns active tasks"), so a test that starts another must wait for them.
    public static void closeAndAwait(ControlPlaneLattice lattice) throws InterruptedException {
        lattice.close();
        BitSet cpus = SystemInfo.getPCpuSet();
        for (int cpu = cpus.nextSetBit(0); cpu >= 0; cpu = cpus.nextSetBit(cpu + 1)) {
            PinnedThreadExecutor executor = PinnedThreadExecutor.get(cpu);
            if (executor != null && executor.isShutdown())
                assertTrue(
                        executor.awaitTermination(30, TimeUnit.SECONDS),
                        "the closed lattice's worker on CPU " + cpu + " did not end");
        }
    }
}
