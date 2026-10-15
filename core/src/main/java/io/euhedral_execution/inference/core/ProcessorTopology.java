package io.euhedral_execution.inference.core;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import java.util.BitSet;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/// Immutable view of Euhedral's processor topology used to resolve worker processor IDs.
///
/// Processor IDs are logical CPU IDs (Linux `cpuN`, one per SMT thread) and may be sparse. Core IDs
/// are Euhedral's normalized physical-core indices from `SystemInfo`, not sysfs `core_id` values.
/// Euhedral runs one lattice worker per selected core, pinned to that core's selected processors.
///
/// Available processors are those Euhedral knows and, when its resource snapshot can be read,
/// those in the current effective cpuset. Euhedral silently drops any other requested ID, so
/// resolution rejects such IDs instead of reporting workers that would never run.
public final class ProcessorTopology {
    private final Map<Integer, Integer> coreByProcessor;
    private final BitSet available;
    private final BitSet performance;
    private final boolean distinguishesPerformanceCores;

    private ProcessorTopology(
            Map<Integer, Integer> coreByProcessor,
            BitSet available,
            BitSet performance,
            boolean distinguishesPerformanceCores) {
        this.coreByProcessor = Map.copyOf(coreByProcessor);
        this.available = (BitSet) available.clone();
        this.performance = (BitSet) performance.clone();
        this.distinguishesPerformanceCores = distinguishesPerformanceCores;
    }

    /// Reads the host topology and effective cpuset from Euhedral's hardware utilities.
    ///
    /// On platforms where Euhedral cannot classify distinct performance and efficiency cores, it
    /// reports every core as a performance core. There, [#performanceCoreProcessors()] selects all
    /// available processors and [#distinguishesPerformanceCores()] is false.
    public static ProcessorTopology system() {
        Map<Integer, Integer> cores = new TreeMap<>();
        BitSet known = SystemInfo.getCpuSet();
        for (int cpu = known.nextSetBit(0); cpu >= 0; cpu = known.nextSetBit(cpu + 1)) {
            cores.put(cpu, SystemInfo.getCpuInfo(cpu).core());
        }
        BitSet available = (BitSet) known.clone();
        if (SystemInfo.SNAPSHOTTER != null)
            available.and(SystemInfo.getSystemSnapshot().effectiveCpus());
        return new ProcessorTopology(
                cores,
                available,
                SystemInfo.getPCpuSet(),
                !SystemInfo.getECoreSet().isEmpty());
    }

    /// Test topology. `coreByProcessor` maps every known processor ID to its core ID; `available`
    /// and `efficiencyCores` are subsets. An empty efficiency set models a platform without P/E data.
    static ProcessorTopology of(Map<Integer, Integer> coreByProcessor, BitSet available, BitSet efficiencyCores) {
        BitSet unknown = (BitSet) available.clone();
        coreByProcessor.keySet().forEach(unknown::clear);
        if (!unknown.isEmpty()) throw new IllegalArgumentException("available processors must be known");
        BitSet performance = new BitSet();
        coreByProcessor.forEach((cpu, core) -> {
            if (cpu < 0 || core < 0) throw new IllegalArgumentException("negative processor or core ID");
            if (!efficiencyCores.get(core)) performance.set(cpu);
        });
        return new ProcessorTopology(coreByProcessor, available, performance, !efficiencyCores.isEmpty());
    }

    /// Returns a copy of the processor IDs that may be selected as workers.
    public BitSet availableProcessorIds() {
        return (BitSet) this.available.clone();
    }

    /// Reports whether Euhedral identified at least one efficiency core.
    public boolean distinguishesPerformanceCores() {
        return this.distinguishesPerformanceCores;
    }

    /// Returns the core ID of a known processor.
    public int coreOf(int processorId) {
        Integer core = this.coreByProcessor.get(processorId);
        if (core == null) throw new IllegalArgumentException("unknown processor ID " + processorId);
        return core;
    }

    /// Returns the distinct core IDs of the given known processors.
    public BitSet coreIds(BitSet processorIds) {
        BitSet cores = new BitSet();
        for (int cpu = processorIds.nextSetBit(0); cpu >= 0; cpu = processorIds.nextSetBit(cpu + 1)) {
            cores.set(coreOf(cpu));
        }
        return cores;
    }

    /// Throws unless every requested processor is available and at least one is requested.
    public void requireAvailable(BitSet processorIds) {
        Objects.requireNonNull(processorIds, "processorIds");
        if (processorIds.isEmpty()) throw new IllegalArgumentException("worker processor set is empty");
        BitSet unavailable = (BitSet) processorIds.clone();
        unavailable.andNot(this.available);
        if (!unavailable.isEmpty())
            throw new IllegalArgumentException("worker processor IDs " + unavailable
                    + " are not available; Euhedral would not run workers on them (available: "
                    + this.available + ")");
    }

    /// Selects every available processor, including all SMT siblings.
    public WorkerProcessorSelection allProcessors() {
        return new WorkerProcessorSelection(this, this.available);
    }

    /// Selects the lowest available processor ID of each available core.
    public WorkerProcessorSelection onePerPhysicalCore() {
        return allProcessors().onePerCore();
    }

    /// Selects available processors of cores Euhedral classifies as performance cores.
    /// Without distinct classification this equals [#allProcessors()].
    public WorkerProcessorSelection performanceCoreProcessors() {
        BitSet selected = (BitSet) this.available.clone();
        selected.and(this.performance);
        return new WorkerProcessorSelection(this, selected);
    }

    /// Selects explicit processor IDs, each of which must be available.
    public WorkerProcessorSelection processors(int... processorIds) {
        return processors(bits(processorIds));
    }

    /// Selects explicit processor IDs, each of which must be available. The input is copied.
    public WorkerProcessorSelection processors(BitSet processorIds) {
        return new WorkerProcessorSelection(this, Objects.requireNonNull(processorIds, "processorIds"));
    }

    static BitSet bits(int... ids) {
        BitSet set = new BitSet();
        for (int id : Objects.requireNonNull(ids, "ids")) {
            if (id < 0) throw new IllegalArgumentException("negative ID " + id);
            set.set(id);
        }
        return set;
    }

    boolean isKnown(int processorId) {
        return this.coreByProcessor.containsKey(processorId);
    }

    BitSet knownCoreIds() {
        BitSet cores = new BitSet();
        this.coreByProcessor.values().forEach(cores::set);
        return cores;
    }
}
