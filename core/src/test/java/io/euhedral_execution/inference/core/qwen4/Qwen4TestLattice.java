package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.core.config.FragmentConfig;
import io.euhedral_execution.core.config.IdlePolicy;
import io.euhedral_execution.core.config.LatticeConfig;
import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.control_plane.ControlPlaneShard;
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Model;
import io.euhedral_execution.inference.core.scheduling.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.scheduling.HostTasks;
import io.euhedral_execution.inference.core.scheduling.graph.InferenceLake;
import java.time.Duration;
import java.util.BitSet;

/// A started lattice and the host work attached to it, for CUDA tests that measure or exercise
/// Flash-Next as the engine runs it: the executor's steps and the expert pipeline run as frames on
/// these workers. One per JVM.
public final class Qwen4TestLattice implements AutoCloseable {

    private final ControlPlaneLattice lattice;
    private final InferenceLake lake;
    private final HostTasks tasks;

    private Qwen4TestLattice(ControlPlaneLattice lattice, InferenceLake lake, HostTasks tasks) {
        this.lattice = lattice;
        this.lake = lake;
        this.tasks = tasks;
    }

    /// Starts a lattice with workers on `workers` distinct physical cores (`EUHEDRAL_QWEN4_WORKERS`
    /// overrides).
    public static Qwen4TestLattice start(int workers) {
        String override = System.getenv("EUHEDRAL_QWEN4_WORKERS");
        if (override != null) workers = Integer.parseInt(override);
        BitSet cpus = new BitSet();
        BitSet cores = new BitSet();
        var physical = SystemInfo.getPCpuSet();
        int selected = 0;
        for (int cpu = physical.nextSetBit(0); cpu >= 0 && selected < workers; cpu = physical.nextSetBit(cpu + 1)) {
            var info = SystemInfo.getCpuInfo(cpu);
            if (info == null || cores.get(info.core())) continue;
            cores.set(info.core());
            cpus.set(cpu);
            selected++;
        }
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
        ControlPlaneLattice lattice = ControlPlaneLattice.getOrCreate(
                new LatticeConfig("Qwen4TestLattice", cpus, Duration.ofSeconds(10), shard));
        lattice.start();
        InferenceLake lake = EuhedralInferenceRuntime.newLake(lattice);
        return new Qwen4TestLattice(lattice, lake, new HostTasks(lake));
    }

    public HostTasks tasks() {
        return this.tasks;
    }

    private static Qwen4TestLattice shared;

    /// A lattice of two workers that lives as long as the JVM, for tests that only need the plan to
    /// run somewhere.
    public static synchronized Qwen4TestLattice shared() {
        if (shared == null) {
            shared = start(2);
            Qwen4TestLattice lattice = shared;
            Runtime.getRuntime().addShutdownHook(new Thread(lattice::close));
        }
        return shared;
    }

    /// The execution plan of `model` on this lattice, with the runtime that runs its graphs.
    public Run run(ExecutionGpu gpu, Qwen4Model model, int maxContextTokens) {
        EuhedralInferenceRuntime runtime = new EuhedralInferenceRuntime(this.lake, this.tasks, gpu, 2);
        try {
            return new Run(runtime, new Qwen4ExecutionPlan(gpu, model, maxContextTokens, runtime));
        } catch (RuntimeException | Error failure) {
            runtime.close();
            throw failure;
        }
    }

    /// A plan and the runtime that runs it; closing it closes the runtime, which retires every
    /// graph.
    public static final class Run implements AutoCloseable {
        private final EuhedralInferenceRuntime runtime;
        private final Qwen4ExecutionPlan plan;

        private Run(EuhedralInferenceRuntime runtime, Qwen4ExecutionPlan plan) {
            this.runtime = runtime;
            this.plan = plan;
        }

        public Qwen4ExecutionPlan plan() {
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
