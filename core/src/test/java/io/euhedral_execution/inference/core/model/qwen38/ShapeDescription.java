package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightStaging;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.ChunkedShape;
import io.euhedral_execution.inference.core.runtime.graph.GraphShape;
import io.euhedral_execution.inference.core.runtime.graph.GraphStorage;
import io.euhedral_execution.inference.core.runtime.graph.StageFrame;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;
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
        if (plan.drafts()) add(text, seen, "draft-catch-up", plan, plan.forExecution(Quantum.ExecutionKind.DRAFT, 16));
        return text.toString();
    }

    /// The prompt graph of `chunks - 1` chunks of `chunkRows` rows and a last one of `lastRows`: its two templates'
    /// names, then one line per stage (its chunk, its template stage, and its submitted and retired successors as
    /// chunk.stage). Before describing it, checks that each chunk keeps its template's edges, and that every other
    /// edge runs from a chunk to the next between two stages that share a workspace buffer or a carried key.
    public static String prompt(ExecutionPlan plan, int chunkRows, int lastRows, int chunks) {
        Shape full = plan.forExecution(Quantum.ExecutionKind.PREFILL, chunkRows);
        Shape last = plan.forExecution(Quantum.ExecutionKind.PREFILL, lastRows);
        var shape = new ChunkedShape(full, last, chunks, NO_FRAMES);
        StringBuilder text = new StringBuilder("prompt chunks=")
                .append(chunks)
                .append(" rows=")
                .append(chunkRows)
                .append('/')
                .append(lastRows)
                .append(" stages=")
                .append(shape.topology().size())
                .append('\n');
        for (int stage = 0; stage < shape.topology().size(); stage++) {
            int[] submitted = shape.topology().submittedSuccessors(stage);
            int[] retired = shape.topology().retiredSuccessors(stage);
            checkEdges(shape, stage, submitted, true);
            checkEdges(shape, stage, retired, false);
            text.append(shape.chunkOf(stage))
                    .append('.')
                    .append(shape.templateStageOf(stage))
                    .append(" sub=")
                    .append(named(shape, submitted))
                    .append(" ret=")
                    .append(named(shape, retired))
                    .append('\n');
        }
        for (int chunk = 0; chunk < chunks; chunk++) {
            GraphShape template = shape.template(chunk);
            for (int stage = 0; stage < template.topology().size(); stage++) {
                requireAll(shape, chunk, stage, template.topology().submittedSuccessors(stage), true);
                requireAll(shape, chunk, stage, template.topology().retiredSuccessors(stage), false);
            }
        }
        return text.toString();
    }

    /// Each successor of `stage` is its template's, in its chunk, or a stage of the next chunk it shares state with.
    private static void checkEdges(ChunkedShape shape, int stage, int[] successors, boolean submitted) {
        int chunk = shape.chunkOf(stage);
        int from = shape.templateStageOf(stage);
        GraphShape template = shape.template(chunk);
        for (int successor : successors) {
            int to = shape.templateStageOf(successor);
            if (shape.chunkOf(successor) == chunk) {
                int[] own = submitted
                        ? template.topology().submittedSuccessors(from)
                        : template.topology().retiredSuccessors(from);
                if (Arrays.stream(own).noneMatch(s -> s == to))
                    throw new AssertionError("chunk " + chunk + " adds an edge " + from + " -> " + to);
            } else if (shape.chunkOf(successor) != chunk + 1
                    || !shares(
                                    template.workspaceBuffers(from),
                                    shape.template(chunk + 1).workspaceBuffers(to))
                            && !shares(
                                    template.carriedState(from),
                                    shape.template(chunk + 1).carriedState(to)))
                throw new AssertionError("an edge " + chunk + "." + from + " -> " + shape.chunkOf(successor) + "." + to
                        + " is not between neighbouring chunks' users of one buffer or key");
        }
    }

    private static void requireAll(ChunkedShape shape, int chunk, int stage, int[] successors, boolean submitted) {
        int at = shape.stage(chunk, stage);
        int[] got = submitted
                ? shape.topology().submittedSuccessors(at)
                : shape.topology().retiredSuccessors(at);
        for (int successor : successors) {
            int expected = shape.stage(chunk, successor);
            if (Arrays.stream(got).noneMatch(s -> s == expected))
                throw new AssertionError("chunk " + chunk + " lost its template's edge " + stage + " -> " + successor);
        }
    }

    private static boolean shares(int[] a, int[] b) {
        for (int x : a) for (int y : b) if (x == y) return true;
        return false;
    }

    private static String named(ChunkedShape shape, int[] stages) {
        StringBuilder text = new StringBuilder("[");
        for (int i = 0; i < stages.length; i++) {
            if (i > 0) text.append(", ");
            text.append(shape.chunkOf(stages[i])).append('.').append(shape.templateStageOf(stages[i]));
        }
        return text.append(']').toString();
    }

    /// A description builds no frames.
    private static final ChunkedShape.Chunks NO_FRAMES = new ChunkedShape.Chunks() {
        @Override
        public StageFrame create(StageGraph graph, int stage, int chunk, int templateStage, ExecutionGpu gpu) {
            throw new UnsupportedOperationException();
        }

        @Override
        public GraphStorage newStorage(ExecutionGpu gpu) {
            throw new UnsupportedOperationException();
        }
    };

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
