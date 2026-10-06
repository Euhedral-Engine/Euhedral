package io.euhedral_execution.inference.core.model_loader.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
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
    void withoutAStatementTheBudgetIsWhatTheMachineCanSpareLessTheRuntimesReserve() {
        HostBudget budget = HostBudget.of(null, null, 20 * GIB, 2 * GIB);
        assertEquals(HostBudget.Source.AUTOMATIC, budget.source());
        assertEquals(HostBudget.ofAvailable(20 * GIB).residentBytes() - 2 * GIB, budget.residentBytes());
        assertEquals(budget.residentBytes(), budget.pinnableBytes());
        assertEquals(
                budget.residentBytes(),
                HostBudget.of(" ", "", 20 * GIB, 2 * GIB).residentBytes());
        assertEquals(0, HostBudget.of(null, null, 5 * GIB, 2 * GIB).residentBytes(), "never below zero");
        assertTrue(HostBudget.of(null, null, -1).pinnableBytes() > 1L << 50, "unreadable means unlimited");
    }

    @Test
    void aStatedBudgetIsRespectedAsStated() {
        HostBudget stated = HostBudget.of("32768", null, 8 * GIB);
        assertEquals(HostBudget.Source.EXPLICIT, stated.source());
        assertEquals(32 * GIB, stated.residentBytes(), "more than the machine spares now is still the user's call");
        assertEquals(
                HostBudget.ofAvailable(8 * GIB).pinnableBytes(), stated.pinnableBytes(), "pinning follows the machine");
        assertEquals(1 * GIB, HostBudget.of("1024", null, 48 * GIB).residentBytes(), "less is respected too");
        assertEquals(0, HostBudget.of("0", null, 48 * GIB).residentBytes(), "zero means no RAM tier");
        assertEquals(16 * GIB, HostBudget.of(null, "16384", 8 * GIB).residentBytes(), "the property is the fallback");
        assertEquals(32 * GIB, HostBudget.of("32768", "1", 8 * GIB).residentBytes(), "the variable wins");
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
        assertThrows(NumberFormatException.class, () -> HostBudget.of("lots", null, GIB));
        assertThrows(IllegalArgumentException.class, () -> HostBudget.of("-5", null, GIB));
        assertThrows(IllegalArgumentException.class, () -> new HostBudget(-1, 0));
    }
}
