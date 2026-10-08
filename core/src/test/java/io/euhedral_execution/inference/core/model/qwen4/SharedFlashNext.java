package io.euhedral_execution.inference.core.model.qwen4;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen4.loader.HostBudget;
import io.euhedral_execution.inference.core.model.qwen4.loader.Mode;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.testing.Shared;
import io.euhedral_execution.inference.core.testing.SharedQwen38;
import java.util.function.Supplier;

/// The Flash-Next model that the CUDA tests of one JVM share. One model is on the device at a time: asking for another
/// [Spec] closes the model that is open first (two do not fit the device, and the host pinning is sized from what the
/// first leaves free), and asking for the open one returns it. The holder is a [Shared] value of the
/// [ModelGroup#FLASH_NEXT] scope, so a class of another group closes it too.
///
/// The model is immutable to the tests: they free what they allocate, read cache counters as deltas and do not assume
/// a cold expert cache.
final class SharedFlashNext {
    private SharedFlashNext() {}

    /// How a model is opened: the device memory it may plan with (`deviceCap`, 0 for all that is free), the mode and
    /// the
    /// context the residency plan is sized for, and the longest sequence of the execution plan over it.
    record Spec(String name, long deviceCap, Mode mode, int context, int planContext) {}

    /// All the free device memory, text only, room for a context of 8,192 tokens: what the tests that run the model
    /// plan against.
    static final Spec ROOMY = new Spec("roomy", 0, Mode.TEXT, 8192, 8192);

    /// The smallest expert cache the planner allows: 5 GiB of device and a 262,144-token context leave about 20 slots.
    static final Spec SMALL_CACHE = new Spec("small-cache", 5L << 30, Mode.TEXT, 262144, 4096);

    /// The MTP layer and its expert bank loaded as well.
    static final Spec MTP = new Spec("mtp", 0, new Mode(true, false), 8192, 8192);

    /// An open model and the device it is on, and the one execution plan over it. A model supports one plan at a time:
    /// the expert cache keeps fences of the plan's streams, so a second plan after the first closed waits on
    /// destroyed events. The plan is therefore made once, with the model, and shared; a test makes sequences of its
    /// own and undoes whatever it set on the plan (observers, trace).
    static final class Loaded {
        private final CudaGpuMemory gpu;
        private final Qwen4Model model;
        private final Spec spec;
        private TestLattice.Run run;

        Loaded(CudaGpuMemory gpu, Qwen4Model model, Spec spec) {
            this.gpu = gpu;
            this.model = model;
            this.spec = spec;
        }

        CudaGpuMemory gpu() {
            return this.gpu;
        }

        Qwen4Model model() {
            return this.model;
        }

        Spec spec() {
            return this.spec;
        }

        /// The plan over the model, made on first use.
        synchronized ExecutionPlan plan() {
            if (this.run == null) this.run = TestLattice.shared().run(this.gpu, this.model, this.spec.planContext());
            return this.run.plan();
        }

        synchronized void close() {
            try {
                if (this.run != null) this.run.close();
            } finally {
                this.run = null;
                this.model.close();
            }
        }
    }

    private static final class Holder implements AutoCloseable {
        private Loaded current;

        synchronized Loaded get(Spec spec) throws Exception {
            if (this.current != null && this.current.spec().equals(spec)) return this.current;
            closeCurrent();
            CudaGpuMemory gpu = SharedQwen38.gpu();
            long free = gpu.deviceMemoryInfo().freeBytes();
            long budget = spec.deviceCap() == 0 ? free : Math.min(spec.deviceCap(), free);
            Qwen4Model model = Qwen4Model.open(
                    TestSupport.artifactPath(), gpu, budget, HostBudget.system(), spec.mode(), spec.context());
            this.current = new Loaded(gpu, model, spec);
            return this.current;
        }

        synchronized void closeCurrent() {
            if (this.current == null) return;
            Loaded closing = this.current;
            this.current = null;
            closing.close();
        }

        @Override
        public void close() {
            closeCurrent();
        }
    }

    private static Holder holder() {
        // The device handle first: the registry closes newest first, and the model must go before the device.
        SharedQwen38.gpu();
        return Shared.get(ModelGroup.FLASH_NEXT, "flash-next-model", Holder::new);
    }

    /// The model opened as `spec`, loaded on first use; the artifact must exist.
    static Loaded model(Spec spec) throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(TestSupport.hasArtifact(), "no artifact");
        return holder().get(spec);
    }

    /// Closes the open model, for a test that loads models of its own.
    static void release() {
        holder().closeCurrent();
    }

    /// Results of work on the [#ROOMY] model that a later test of another model compares against, computed once per
    /// JVM by whichever test needs them first.
    private static final java.util.Map<String, Object> BASELINES = new java.util.concurrent.ConcurrentHashMap<>();

    @SuppressWarnings("unchecked")
    static synchronized <T> T baseline(String key, Supplier<T> compute) {
        Object held = BASELINES.get(key);
        if (held == null) {
            held = compute.get();
            assertTrue(held != null);
            BASELINES.put(key, held);
        }
        return (T) held;
    }
}
