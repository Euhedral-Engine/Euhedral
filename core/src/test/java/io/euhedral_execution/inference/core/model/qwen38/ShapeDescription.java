package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightStaging;
import io.euhedral_execution.inference.core.runtime.graph.StageTopology;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/// The canonical text of a 3.8 view: one line per stage (kind, layer, buffers, widths, its dependencies, its submitted
/// and retired successors, and its weights by name, format, layout, size and residency), then its buffer specs and
/// flags. Device addresses never appear: a staged weight is named by its staging slot, any other by where it lives.
/// Golden descriptions recorded before a refactor pin every view's topology and specs.
public final class ShapeDescription {

    private ShapeDescription() {}

    /// Every view `plan` offers, in a fixed order: its own topology, then decode, preloaded decode, small and region
    /// prefill, the draft view and the DFlash2 context view, where the plan has them. A view the plan answers with
    /// one already described (a plan without views answers every kind with its own topology) is not repeated.
    public static String views(ExecutionPlan plan) {
        List<Object> seen = new ArrayList<>();
        StringBuilder text = new StringBuilder();
        add(text, seen, "own", plan, plan.shape());
        var decode = plan.forExecution(Quantum.ExecutionKind.DECODE, 1);
        add(text, seen, "decode", plan, decode);
        add(text, seen, "decode-preloaded", plan, decode.preloadedVariant());
        var verify = plan.forExecution(Quantum.ExecutionKind.VERIFY, 16);
        add(text, seen, "verify", plan, verify);
        add(text, seen, "verify-preloaded", plan, verify.preloadedVariant());
        add(text, seen, "small-prefill", plan, plan.forExecution(Quantum.ExecutionKind.PREFILL, 1));
        add(text, seen, "region-prefill", plan, plan.forExecution(Quantum.ExecutionKind.PREFILL, 64));
        if (plan.drafts() || plan.draftsWithDFlash2())
            add(text, seen, "draft", plan, plan.forExecution(Quantum.ExecutionKind.DRAFT, 1));
        if (plan.draftsWithDFlash2())
            add(text, seen, "draft-context", plan, plan.forExecution(Quantum.ExecutionKind.DRAFT_CONTEXT, 1));
        return text.toString();
    }

    private static void add(StringBuilder text, List<Object> seen, String name, ExecutionPlan owner, Shape view) {
        for (Object earlier : seen) if (earlier == view) return;
        seen.add(view);
        text.append(describe(
                name,
                view.instructions(),
                view.topology(),
                view.bufferSpecs(),
                view.reusePrefillStorage(),
                view.prefetchesRing(),
                owner.staging()));
    }

    public static String describe(
            String name,
            List<ExecutionPlan.Instruction> specs,
            StageTopology topology,
            List<ExecutionPlan.BufferSpec> buffers,
            boolean reuseStorage,
            boolean prefetchesRing,
            WeightStaging staging) {
        StringBuilder text = new StringBuilder("view ")
                .append(name)
                .append(" stages=")
                .append(specs.size())
                .append(" reuseStorage=")
                .append(reuseStorage)
                .append(" prefetchesRing=")
                .append(prefetchesRing)
                .append('\n');
        for (ExecutionPlan.Instruction spec : specs) {
            text.append(spec.id())
                    .append(' ')
                    .append(spec.kind())
                    .append(" L")
                    .append(spec.layerIndex())
                    .append(" in=")
                    .append(spec.inputBuffers())
                    .append(" out=")
                    .append(spec.outputBuffers())
                    .append(" w=")
                    .append(spec.inputWidth())
                    .append('/')
                    .append(spec.outputWidth())
                    .append(" idx=")
                    .append(spec.outputBufferIndex())
                    .append(" deps=")
                    .append(spec.dependencies())
                    .append(" in-degree=")
                    .append(topology.inDegree(spec.id()))
                    .append(" sub=")
                    .append(Arrays.toString(topology.submittedSuccessors(spec.id())))
                    .append(" ret=")
                    .append(Arrays.toString(topology.retiredSuccessors(spec.id())))
                    .append(" weights=[");
            List<TensorHandle> weights = spec.weights();
            for (int i = 0; i < weights.size(); i++) {
                TensorHandle weight = weights.get(i);
                if (i > 0) text.append(", ");
                text.append(weight.name())
                        .append(':')
                        .append(weight.format())
                        .append(':')
                        .append(weight.layout())
                        .append(':')
                        .append(weight.byteSize())
                        .append(':')
                        .append(residency(weight, staging));
            }
            text.append("]\n");
        }
        for (ExecutionPlan.BufferSpec buffer : buffers)
            text.append("buffer ").append(buffer).append('\n');
        return text.toString();
    }

    private static String residency(TensorHandle weight, WeightStaging staging) {
        if (staging != null)
            for (int slot = 0; slot < staging.slots(); slot++)
                if (weight.deviceAddress() == staging.slotAddress(slot))
                    return "slot" + slot + (weight.hostBacked() ? "+host" : "");
        return weight.hostBacked() ? "host" : "device";
    }

    /// Compares `actual` with the golden `name` under `directory`, or records it with `-Deuhedral.shapes.record=true`.
    public static void check(Path directory, String name, String actual) throws IOException {
        Path golden = directory.resolve(name + ".txt.gz");
        if (Boolean.getBoolean("euhedral.shapes.record")) {
            Files.createDirectories(directory);
            try (var out = new GZIPOutputStream(Files.newOutputStream(golden))) {
                out.write(actual.getBytes(StandardCharsets.UTF_8));
            }
            return;
        }
        if (!Files.isRegularFile(golden)) throw new AssertionError("no recorded shape for " + name);
        String expected;
        try (var in = new GZIPInputStream(Files.newInputStream(golden))) {
            expected = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
        if (!expected.equals(actual)) {
            // The whole description, for a diff against the golden.
            Path written = Path.of("build/shapes", name + ".actual.txt");
            Files.createDirectories(written.getParent());
            Files.writeString(written, actual);
            String[] want = expected.split("\n"), got = actual.split("\n");
            int line = 0;
            while (line < Math.min(want.length, got.length) && want[line].equals(got[line])) line++;
            throw new AssertionError(name + " changed shape at line " + (line + 1) + ":\n  expected "
                    + (line < want.length ? want[line] : "<end>") + "\n  actual   "
                    + (line < got.length ? got[line] : "<end>"));
        }
    }
}
