package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class WorkerProcessorSelectionTest {
    /// Sparse IDs with Linux-style SMT siblings (N, N+8), one efficiency core, and processor 13 known
    /// to the topology but outside the effective cpuset.
    static final ProcessorTopology HYBRID = ProcessorTopology.of(
            Map.of(0, 0, 8, 0, 1, 1, 9, 1, 4, 2, 5, 2, 12, 3, 13, 4),
            ProcessorTopology.bits(0, 1, 4, 5, 8, 9, 12),
            ProcessorTopology.bits(3));

    static final ProcessorTopology HOMOGENEOUS =
            ProcessorTopology.of(Map.of(0, 0, 1, 0, 2, 1, 3, 1), ProcessorTopology.bits(0, 1, 2, 3), new BitSet());

    @Test
    void allProcessorsIncludesSmtSiblingsButNotUnavailableIds() {
        var all = HYBRID.allProcessors();
        assertEquals(ProcessorTopology.bits(0, 1, 4, 5, 8, 9, 12), all.processorIds());
        assertEquals(ProcessorTopology.bits(0, 1, 2, 3), all.coreIds());
    }

    @Test
    void onePerPhysicalCoreKeepsTheLowestProcessorOfEachCore() {
        var perCore = HYBRID.onePerPhysicalCore();
        assertEquals(ProcessorTopology.bits(0, 1, 4, 12), perCore.processorIds());
        assertEquals(ProcessorTopology.bits(0, 1, 2, 3), perCore.coreIds());
        assertEquals(
                ProcessorTopology.bits(8, 9, 5),
                HYBRID.processors(8, 9, 5).onePerCore().processorIds(),
                "a reduced selection keeps the lowest selected sibling, not the lowest known one");
    }

    @Test
    void performanceCoresExcludeEfficiencyCoresWhenTheyAreClassified() {
        assertTrue(HYBRID.distinguishesPerformanceCores());
        assertEquals(
                ProcessorTopology.bits(0, 1, 4, 5, 8, 9),
                HYBRID.performanceCoreProcessors().processorIds());
        assertEquals(
                ProcessorTopology.bits(0, 1, 4),
                HYBRID.performanceCoreProcessors().onePerCore().processorIds());
    }

    @Test
    void performanceCoresSelectEverythingWithoutDistinctClassification() {
        assertFalse(HOMOGENEOUS.distinguishesPerformanceCores());
        assertEquals(
                HOMOGENEOUS.allProcessors().processorIds(),
                HOMOGENEOUS.performanceCoreProcessors().processorIds());
    }

    @Test
    void exclusionsDistinguishProcessorIdsFromCoreIds() {
        assertEquals(
                ProcessorTopology.bits(0, 1, 4, 5, 9, 12),
                HYBRID.allProcessors().excludingProcessors(8).processorIds());
        // Core ID 0 holds processors 0 and 8; processor ID 1 is a sibling on core 1.
        assertEquals(
                ProcessorTopology.bits(4, 5, 9, 12),
                HYBRID.allProcessors().excludingCores(0).excludingProcessors(1).processorIds());
        assertEquals(
                ProcessorTopology.bits(0, 4, 5, 8, 12),
                HYBRID.allProcessors().excludingCores(1).processorIds());
    }

    @Test
    void rejectsUnknownUnavailableEmptyAndNegativeSelections() {
        assertThrows(IllegalArgumentException.class, () -> HYBRID.processors(2), "unknown sparse ID");
        var unavailable = assertThrows(IllegalArgumentException.class, () -> HYBRID.processors(0, 13));
        assertTrue(unavailable.getMessage().contains("{13}"), unavailable.getMessage());
        assertThrows(IllegalArgumentException.class, () -> HYBRID.processors());
        assertThrows(IllegalArgumentException.class, () -> HYBRID.processors(-1));
        assertThrows(IllegalArgumentException.class, () -> HYBRID.processors(0).excludingProcessors(0));
        assertThrows(
                IllegalArgumentException.class, () -> HYBRID.allProcessors().excludingProcessors(2));
        assertThrows(
                IllegalArgumentException.class, () -> HYBRID.allProcessors().excludingCores(9));
        assertThrows(IllegalArgumentException.class, () -> HYBRID.processors(12).excludingCores(3));
    }

    @Test
    void selectionCopiesItsInputAndOutput() {
        BitSet input = ProcessorTopology.bits(0, 8);
        var selection = HYBRID.processors(input);
        input.set(1);
        BitSet output = selection.processorIds();
        output.clear();
        selection.coreIds().clear();
        assertEquals(ProcessorTopology.bits(0, 8), selection.processorIds());
        assertEquals(ProcessorTopology.bits(0), selection.coreIds());
        BitSet available = HYBRID.availableProcessorIds();
        available.clear();
        assertFalse(HYBRID.availableProcessorIds().isEmpty());
    }

    @Test
    void systemTopologyMatchesEuhedralSystemInfo() {
        var system = ProcessorTopology.system();
        BitSet all = system.allProcessors().processorIds();
        BitSet known = SystemInfo.getCpuSet();
        BitSet outside = (BitSet) all.clone();
        outside.andNot(known);
        assertTrue(outside.isEmpty(), "available processors must be known to SystemInfo");
        var perCore = system.onePerPhysicalCore();
        assertEquals(perCore.processorIds().cardinality(), perCore.coreIds().cardinality());
        assertEquals(system.allProcessors().coreIds(), perCore.coreIds());
        for (int cpu = all.nextSetBit(0); cpu >= 0; cpu = all.nextSetBit(cpu + 1))
            assertEquals(SystemInfo.getCpuInfo(cpu).core(), system.coreOf(cpu));
        assertEquals(!SystemInfo.getECoreSet().isEmpty(), system.distinguishesPerformanceCores());
    }

    /// Two sockets of four cores (0-3 and 4-7) with SMT siblings (N, N+8): the host a container sees.
    static ProcessorTopology twoSockets(double quotaCpus) {
        Map<Integer, Integer> cores = new HashMap<>();
        Map<Integer, Integer> sockets = new HashMap<>();
        for (int core = 0; core < 8; core++) {
            cores.put(core, core);
            cores.put(core + 8, core);
            sockets.put(core, core / 4);
        }
        BitSet all = new BitSet();
        all.set(0, 16);
        return ProcessorTopology.of(cores, sockets, all, new BitSet(), quotaCpus);
    }

    @Test
    void withoutAQuotaOrWithOneCoveringEveryCoreEveryProcessorWorks() {
        assertEquals(
                twoSockets(0).allProcessors().processorIds(),
                twoSockets(0).withinQuota().processorIds());
        assertEquals(
                twoSockets(8).allProcessors().processorIds(),
                twoSockets(8).withinQuota().processorIds());
        assertEquals(
                twoSockets(12.5).allProcessors().processorIds(),
                twoSockets(12.5).withinQuota().processorIds());
    }

    @Test
    void aQuotaBelowTheCoreCountSelectsThatManyWholeCores() {
        var selection = twoSockets(3.7).withinQuota();
        assertEquals(ProcessorTopology.bits(0, 1, 4), selection.coreIds(), "floor(3.7) cores, the sockets in turn");
        assertEquals(ProcessorTopology.bits(0, 1, 4, 8, 9, 12), selection.processorIds(), "with their SMT siblings");
        assertEquals(ProcessorTopology.bits(0, 4), twoSockets(0.4).withinQuota().coreIds(), "a core on every socket");
    }

    @Test
    void aQuotaPrefersPerformanceCores() {
        // HYBRID: cores 0, 1, 2 performance, 3 efficiency; with a quota of two cores, two performance cores.
        var hybrid = ProcessorTopology.of(
                Map.of(0, 0, 8, 0, 1, 1, 9, 1, 4, 2, 5, 2, 12, 3, 13, 4),
                Map.of(),
                ProcessorTopology.bits(0, 1, 4, 5, 8, 9, 12),
                ProcessorTopology.bits(3),
                2);
        assertEquals(ProcessorTopology.bits(0, 1), hybrid.withinQuota().coreIds());
    }
}
