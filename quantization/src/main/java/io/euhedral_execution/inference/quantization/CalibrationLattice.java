package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.core.config.FragmentConfig;
import io.euhedral_execution.core.config.IdlePolicy;
import io.euhedral_execution.core.config.LatticeConfig;
import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.control_plane.ControlPlaneShard;
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.runtime.HostTasks;
import io.euhedral_execution.inference.core.runtime.graph.InferenceLake;
import java.time.Duration;
import java.util.BitSet;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;

/// The Euhedral lattice the calibration runs its concurrent work on: reading the next layer from the checkpoint,
/// rounding a layer's objects on their own streams, and writing their states. Workers take one CPU per performance
/// core first, then efficiency cores, never the CPUs in `excluded`.
public final class CalibrationLattice implements AutoCloseable {

    private final ControlPlaneLattice lattice;
    private final InferenceLake lake;
    private final HostTasks tasks;

    private CalibrationLattice(ControlPlaneLattice lattice, InferenceLake lake, HostTasks tasks) {
        this.lattice = lattice;
        this.lake = lake;
        this.tasks = tasks;
    }

    public static CalibrationLattice start(int workers, Set<Integer> excluded) {
        BitSet cpus = new BitSet();
        BitSet cores = new BitSet();
        var performance = SystemInfo.getPCpuSet();
        for (int cpu = performance.nextSetBit(0);
                cpu >= 0 && cpus.cardinality() < workers;
                cpu = performance.nextSetBit(cpu + 1)) {
            var info = SystemInfo.getCpuInfo(cpu);
            if (excluded.contains(cpu) || info == null || cores.get(info.core())) continue;
            cores.set(info.core());
            cpus.set(cpu);
        }
        var efficiency = SystemInfo.getECpuSet();
        for (int cpu = efficiency.nextSetBit(0);
                cpu >= 0 && cpus.cardinality() < workers;
                cpu = efficiency.nextSetBit(cpu + 1)) {
            if (!excluded.contains(cpu)) cpus.set(cpu);
        }
        FragmentConfig defaults = FragmentConfig.ofDefaults();
        FragmentConfig fixed = new FragmentConfig(
                defaults.cloneConfig(),
                defaults.cacheConfig(),
                defaults.observer(),
                defaults.maxBatchSize(),
                defaults.smtEnabled(),
                new IdlePolicy(IdlePolicy.DEFAULT_IDLE_PARK_NS, IdlePolicy.DEFAULT_CONTENTION_HALF_LIFE_NANOS),
                defaults.benchmarkMode(),
                defaults.metricPrefix(),
                defaults.registry());
        var shard = ControlPlaneShard.createBaseShard(
                "CalibrationShard", new BaseCloneableObject(fixed, new DefaultExecutor()));
        ControlPlaneLattice lattice = ControlPlaneLattice.getOrCreate(
                new LatticeConfig("CalibrationLattice", cpus, Duration.ofSeconds(10), shard));
        lattice.start();
        InferenceLake lake = EuhedralInferenceRuntime.newLake(lattice);
        return new CalibrationLattice(lattice, lake, new HostTasks(lake));
    }

    /// The lattice itself, for an engine [io.euhedral_execution.inference.core.model.qwen38.Execution] of its own.
    public ControlPlaneLattice terminal() {
        return lattice;
    }

    /// The lake frames are published into.
    public InferenceLake lake() {
        return lake;
    }

    /// Attaches a source the lattice polls along with the lake (completed with it).
    public void attach(io.euhedral_execution.core.ingest.AbstractIngestSink source) {
        lake.attach(source.getDelegate());
    }

    /// Runs `work` as a frame on a worker.
    public <T> CompletableFuture<T> submit(Supplier<T> work) {
        return tasks.onWorker(work);
    }

    @Override
    public void close() {
        tasks.close();
        lake.completeGracefully();
        lake.awaitTermination();
        lattice.close();
    }
}
