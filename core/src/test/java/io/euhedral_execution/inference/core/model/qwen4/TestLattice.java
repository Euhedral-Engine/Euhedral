package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.core.config.FragmentConfig;
import io.euhedral_execution.core.config.IdlePolicy;
import io.euhedral_execution.core.config.LatticeConfig;
import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.control_plane.ControlPlaneShard;
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.runtime.HostTasks;
import io.euhedral_execution.inference.core.runtime.graph.InferenceLake;
import java.time.Duration;
import java.util.BitSet;

/// A started lattice and the host work attached to it, for CUDA tests that measure or exercise
/// Flash-Next as the engine runs it: the executor's steps and the expert pipeline run as frames on
/// these workers. One per JVM.
public final class TestLattice implements AutoCloseable {

    private final ControlPlaneLattice lattice;
    private final InferenceLake lake;
    private final HostTasks tasks;
    private final BitSet cpus;

    private TestLattice(ControlPlaneLattice lattice, InferenceLake lake, HostTasks tasks, BitSet cpus) {
        this.lattice = lattice;
        this.lake = lake;
        this.tasks = tasks;
        this.cpus = cpus;
    }

    /// The logical processors the workers run on, one worker each.
    public BitSet cpus() {
        return (BitSet) this.cpus.clone();
    }

    /// Starts a lattice with `workers` workers (`EUHEDRAL_QWEN4_WORKERS` overrides), one per logical processor: one
    /// on each performance core first, then the efficiency cores, then the performance cores' other hardware
    /// threads, so a count of every processor (as the engine is deployed) takes them all.
    public static TestLattice start(int workers) {
        String override = System.getenv("EUHEDRAL_QWEN4_WORKERS");
        if (override != null) workers = Integer.parseInt(override);
        BitSet cpus = new BitSet();
        BitSet cores = new BitSet();
        var performance = SystemInfo.getPCpuSet();
        for (int cpu = performance.nextSetBit(0);
                cpu >= 0 && cpus.cardinality() < workers;
                cpu = performance.nextSetBit(cpu + 1)) {
            var info = SystemInfo.getCpuInfo(cpu);
            if (info == null || cores.get(info.core())) continue;
            cores.set(info.core());
            cpus.set(cpu);
        }
        var efficiency = SystemInfo.getECpuSet();
        for (int cpu = efficiency.nextSetBit(0);
                cpu >= 0 && cpus.cardinality() < workers;
                cpu = efficiency.nextSetBit(cpu + 1)) cpus.set(cpu);
        for (int cpu = performance.nextSetBit(0);
                cpu >= 0 && cpus.cardinality() < workers;
                cpu = performance.nextSetBit(cpu + 1)) cpus.set(cpu);
        // Fixed idle timing, as the engine runs the lattice: adaptive parking leaves workers asleep for the
        // frame-by-frame quanta of generation.
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
                "Qwen4TestShard", new BaseCloneableObject(fixed, new DefaultExecutor()));
        ControlPlaneLattice lattice =
                ControlPlaneLattice.getOrCreate(new LatticeConfig("TestLattice", cpus, Duration.ofSeconds(10), shard));
        lattice.start();
        InferenceLake lake = EuhedralInferenceRuntime.newLake(lattice);
        return new TestLattice(lattice, lake, new HostTasks(lake), cpus);
    }

    public HostTasks tasks() {
        return this.tasks;
    }

    private static TestLattice shared;

    /// A lattice of two workers that lives as long as the JVM, for tests that only need the plan to
    /// run somewhere.
    public static synchronized TestLattice shared() {
        if (shared == null) {
            shared = start(2);
            TestLattice lattice = shared;
            Runtime.getRuntime().addShutdownHook(new Thread(lattice::close));
        }
        return shared;
    }

    /// The execution plan of `model` on this lattice, with the runtime that runs its graphs.
    public Run run(ExecutionGpu gpu, Qwen4Model model, int maxContextTokens) {
        EuhedralInferenceRuntime runtime = new EuhedralInferenceRuntime(
                this.lake,
                this.tasks,
                gpu,
                Integer.parseInt(System.getenv().getOrDefault("EUHEDRAL_QWEN4_LANES", "2")));
        try {
            return new Run(runtime, new ExecutionPlan(gpu, model, maxContextTokens, runtime));
        } catch (RuntimeException | Error failure) {
            runtime.close();
            throw failure;
        }
    }

    /// A plan and the runtime that runs it; closing it closes the runtime, which retires every
    /// graph.
    public static final class Run implements AutoCloseable {
        private final EuhedralInferenceRuntime runtime;
        private final ExecutionPlan plan;

        private Run(EuhedralInferenceRuntime runtime, ExecutionPlan plan) {
            this.runtime = runtime;
            this.plan = plan;
        }

        public ExecutionPlan plan() {
            return this.plan;
        }

        public EuhedralInferenceRuntime runtime() {
            return this.runtime;
        }

        @Override
        public void close() {
            // Every accepted step retires (its expert sources still serve it) before the plan completes the sources.
            this.runtime.close();
            this.plan.close();
        }
    }

    @Override
    public void close() {
        this.tasks.close();
        this.lake.completeGracefully();
        this.lake.awaitTermination();
        this.lattice.close();
    }
}
