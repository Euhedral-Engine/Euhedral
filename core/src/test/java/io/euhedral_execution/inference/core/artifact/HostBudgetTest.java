package io.euhedral_execution.inference.core.artifact;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.OptionalLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HostBudgetTest {

    private static final long GIB = 1L << 30;

    @TempDir
    Path directory;

    @Test
    void leavesRoomForTheOperatingSystem() {
        HostBudget budget = HostBudget.ofAvailable(64 * GIB);
        assertTrue(budget.residentBytes() < 64 * GIB);
        assertTrue(64 * GIB - budget.residentBytes() >= 4 * GIB);
        assertEquals(budget.pinnableBytes(), budget.residentBytes());
        assertEquals(0, HostBudget.ofAvailable(3 * GIB).residentBytes(), "never closer than 4 GiB to exhaustion");
    }

    @Test
    void theRuntimesReserveBoundsTheHeapsRoomToGrow() {
        long reserve = HostBudget.runtimeReserve();
        long least = HostBudget.NATIVE_RESERVE + HostBudget.NGRAM_WORKING_SET;
        assertTrue(reserve >= least, "native memory and the n-gram rows' page cache are always kept");
        assertTrue(
                reserve <= least + HostBudget.HEAP_GROWTH_RESERVE,
                "a server started without -Xmx may grow its heap to a quarter of the machine; the reserve is bounded");
    }

    @Test
    void withoutAStatementTheBudgetIsWhatTheMachineCanSpareLessTheRuntimesReserve() {
        HostBudget budget = HostBudget.of(OptionalLong.empty(), 20 * GIB, 2 * GIB);
        assertEquals(HostBudget.Source.AUTOMATIC, budget.source());
        assertEquals(HostBudget.ofAvailable(20 * GIB).residentBytes() - 2 * GIB, budget.residentBytes());
        assertEquals(budget.residentBytes(), budget.pinnableBytes());
        assertEquals(0, HostBudget.of(OptionalLong.empty(), 5 * GIB, 2 * GIB).residentBytes(), "never below zero");
        assertTrue(HostBudget.of(OptionalLong.empty(), -1).pinnableBytes() > 1L << 50, "unreadable means unlimited");
    }

    @Test
    void aStatedBudgetIsRespectedAsStated() {
        HostBudget stated = HostBudget.of(OptionalLong.of(32 * GIB), 8 * GIB);
        assertEquals(HostBudget.Source.EXPLICIT, stated.source());
        assertEquals(32 * GIB, stated.residentBytes(), "more than the machine spares now is still the user's call");
        assertEquals(
                HostBudget.ofAvailable(8 * GIB).pinnableBytes(), stated.pinnableBytes(), "pinning follows the machine");
        assertEquals(1 * GIB, HostBudget.of(OptionalLong.of(GIB), 48 * GIB).residentBytes(), "less is respected too");
        assertEquals(0, HostBudget.of(OptionalLong.of(0), 48 * GIB).residentBytes(), "zero means no RAM tier");
    }

    /// `memory.host-limit` states the budget in bytes; it is the only statement, whatever the process environment.
    @Test
    void theHostLimitSettingStatesTheBudget() {
        HostBudget stated = HostBudget.system(OptionalLong.of(1536L << 20));
        assertEquals(HostBudget.Source.EXPLICIT, stated.source());
        assertEquals(1536L << 20, stated.residentBytes());
        assertEquals(1536L << 20, stated.pinnedBytes());
        assertTrue(stated.describe().contains("memory.host-limit"), stated.describe());
        assertEquals(
                HostBudget.Source.AUTOMATIC,
                HostBudget.system(OptionalLong.empty()).source());
    }

    @Test
    void theCgroupLimitIsTheMachineWhenItIsLess() throws Exception {
        Path root = this.directory.resolve("cgroup");
        Path scope = root.resolve("user.slice/test.scope");
        Files.createDirectories(scope);
        Files.writeString(root.resolve("user.slice/memory.max"), "max\n");
        Files.writeString(root.resolve("user.slice/memory.current"), "1\n");
        Files.writeString(scope.resolve("memory.max"), String.valueOf(14 * GIB) + "\n");
        Files.writeString(scope.resolve("memory.current"), String.valueOf(3 * GIB) + "\n");
        Path self = this.directory.resolve("self-cgroup");
        Files.writeString(self, "0::/user.slice/test.scope\n");
        assertEquals(11 * GIB, HostBudget.cgroupHeadroom(self, root));
        Files.writeString(scope.resolve("memory.max"), "max\n");
        assertEquals(-1, HostBudget.cgroupHeadroom(self, root), "no limit anywhere");
        assertEquals(-1, HostBudget.cgroupHeadroom(this.directory.resolve("missing"), root));
    }

    @Test
    void anInvalidStatementIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> HostBudget.of(OptionalLong.of(-5), GIB));
        assertThrows(IllegalArgumentException.class, () -> new HostBudget(-1, 0));
    }

    @Test
    void aStatedBudgetPinsAsStatedAndAnAutomaticOneWhatTheMachineSpares() {
        HostBudget stated = HostBudget.of(OptionalLong.of(40960L << 20), 8L << 30);
        assertEquals(40960L << 20, stated.pinnedBytes());
        HostBudget automatic = HostBudget.of(OptionalLong.empty(), 64L << 30);
        assertEquals(Math.min(automatic.pinnableBytes(), automatic.residentBytes()), automatic.pinnedBytes());
    }
}
