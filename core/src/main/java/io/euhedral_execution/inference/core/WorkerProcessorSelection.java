package io.euhedral_execution.inference.core;

import java.util.BitSet;
import java.util.Objects;

/// Immutable, non-empty set of worker processor IDs resolved against a [ProcessorTopology].
///
/// Obtain one from the topology (all processors, one per physical core, performance cores, or
/// explicit IDs), refine it, then pass [#processorIds()] to [InferenceConfig]. Every refinement
/// returns a new selection and rejects results that are empty or reference unavailable IDs.
public final class WorkerProcessorSelection {
    private final ProcessorTopology topology;
    private final BitSet processorIds;

    WorkerProcessorSelection(ProcessorTopology topology, BitSet processorIds) {
        this.topology = Objects.requireNonNull(topology, "topology");
        this.processorIds = (BitSet) processorIds.clone();
        topology.requireAvailable(this.processorIds);
    }

    /// Keeps the lowest selected processor ID of each selected core, dropping SMT siblings.
    public WorkerProcessorSelection onePerCore() {
        BitSet seenCores = new BitSet();
        BitSet selected = new BitSet();
        for (int cpu = this.processorIds.nextSetBit(0); cpu >= 0; cpu = this.processorIds.nextSetBit(cpu + 1)) {
            int core = this.topology.coreOf(cpu);
            if (!seenCores.get(core)) {
                seenCores.set(core);
                selected.set(cpu);
            }
        }
        return new WorkerProcessorSelection(this.topology, selected);
    }

    /// Removes processor IDs. Each ID must be known to the topology.
    public WorkerProcessorSelection excludingProcessors(int... processorIds) {
        BitSet excluded = ProcessorTopology.bits(processorIds);
        for (int cpu = excluded.nextSetBit(0); cpu >= 0; cpu = excluded.nextSetBit(cpu + 1)) {
            if (!this.topology.isKnown(cpu)) throw new IllegalArgumentException("unknown processor ID " + cpu);
        }
        BitSet selected = processorIds();
        selected.andNot(excluded);
        return new WorkerProcessorSelection(this.topology, selected);
    }

    /// Removes every selected processor of the given core IDs. Each core must be known.
    public WorkerProcessorSelection excludingCores(int... coreIds) {
        BitSet excluded = ProcessorTopology.bits(coreIds);
        BitSet unknown = (BitSet) excluded.clone();
        unknown.andNot(this.topology.knownCoreIds());
        if (!unknown.isEmpty()) throw new IllegalArgumentException("unknown core IDs " + unknown);
        BitSet selected = new BitSet();
        for (int cpu = this.processorIds.nextSetBit(0); cpu >= 0; cpu = this.processorIds.nextSetBit(cpu + 1)) {
            if (!excluded.get(this.topology.coreOf(cpu))) selected.set(cpu);
        }
        return new WorkerProcessorSelection(this.topology, selected);
    }

    /// Returns a copy of the selected processor IDs.
    public BitSet processorIds() {
        return (BitSet) this.processorIds.clone();
    }

    /// Returns the distinct core IDs of the selected processors; one Euhedral worker runs per core.
    public BitSet coreIds() {
        return this.topology.coreIds(this.processorIds);
    }

    @Override
    public String toString() {
        return "WorkerProcessorSelection[processorIds=" + this.processorIds + ", coreIds=" + coreIds() + "]";
    }
}
