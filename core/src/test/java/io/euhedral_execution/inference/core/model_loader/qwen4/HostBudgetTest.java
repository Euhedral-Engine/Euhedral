package io.euhedral_execution.inference.core.model_loader.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class HostBudgetTest {

    private static final long GIB = 1L << 30;

    @Test
    void leavesRoomForTheOperatingSystem() {
        HostBudget budget = HostBudget.ofAvailable(64 * GIB);
        assertTrue(budget.residentBytes() < 64 * GIB);
        assertTrue(64 * GIB - budget.residentBytes() >= 4 * GIB);
        assertEquals(budget.pinnableBytes(), budget.residentBytes());
        assertEquals(0, HostBudget.ofAvailable(3 * GIB).residentBytes(), "never closer than 4 GiB to exhaustion");
    }

    @Test
    void theEnvironmentStatesTheMemoryAndTheMarginIsStillTaken() {
        HostBudget stated = HostBudget.of("32768", null, 8 * GIB);
        assertEquals(HostBudget.ofAvailable(32 * GIB).residentBytes(), stated.residentBytes());
        assertEquals(
                HostBudget.ofAvailable(8 * GIB).pinnableBytes(), stated.pinnableBytes(), "pinning follows the machine");
        assertEquals(
                HostBudget.ofAvailable(16 * GIB).residentBytes(),
                HostBudget.of(null, "16384", 8 * GIB).residentBytes(),
                "the property is the fallback");
        assertEquals(
                HostBudget.ofAvailable(32 * GIB).residentBytes(),
                HostBudget.of("32768", "1", 8 * GIB).residentBytes(),
                "the variable wins");
    }

    @Test
    void withoutAStatementNoOrdinaryMemoryIsTaken() {
        HostBudget budget = HostBudget.of(null, null, 20 * GIB);
        assertEquals(0, budget.residentBytes());
        assertEquals(HostBudget.ofAvailable(20 * GIB).pinnableBytes(), budget.pinnableBytes());
        assertEquals(0, HostBudget.of(" ", "", 20 * GIB).residentBytes());
        assertTrue(HostBudget.of(null, null, -1).pinnableBytes() > 1L << 50, "unreadable means unlimited pinning");
    }

    @Test
    void anInvalidStatementIsRefused() {
        assertThrows(NumberFormatException.class, () -> HostBudget.of("lots", null, GIB));
        assertThrows(IllegalArgumentException.class, () -> HostBudget.of("-5", null, GIB));
        assertThrows(IllegalArgumentException.class, () -> new HostBudget(-1, 0));
    }
}
