package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/// A guard on the layout of the core and on the shape of host execution, in addition to the behavioural tests: the
/// model's runtime and its execution path create no scheduling infrastructure of their own. Every
/// piece of host work is a frame on the lattice; the only waiting a path may do is a parked
/// continuation.
class ArchitectureTest {

    private static final Path MAIN = Path.of("src/main/java/io/euhedral_execution/inference/core");

    /// What would be a private scheduler: pools, executors, virtual threads, threads,
    /// CompletableFuture's async forms (which run on the common pool), and blocking waits on
    /// asynchronous operations.
    private static final List<Pattern> FORBIDDEN = List.of(
            Pattern.compile("\\bExecutorService\\b"),
            Pattern.compile("\\bExecutors\\b"),
            Pattern.compile("\\bForkJoinPool\\b"),
            Pattern.compile("\\bThread\\.ofVirtual\\b"),
            Pattern.compile("\\bnewVirtualThreadPerTaskExecutor\\b"),
            Pattern.compile("\\bnew Thread\\("),
            Pattern.compile("\\bThread\\.ofPlatform\\b"),
            Pattern.compile("\\.(supplyAsync|runAsync|thenApplyAsync|thenAcceptAsync|thenRunAsync)\\("),
            Pattern.compile("\\bCountDownLatch\\b"),
            Pattern.compile("\\.awaitCompletion\\("),
            Pattern.compile("\\bparallelStream\\b"));

    /// Startup loaders that read the artifact before any request exists. They are not execution
    /// paths: the threads end with the load.
    private static final Set<String> STARTUP = Set.of(
            "model/qwen4/loader/FixedLoader.java",
            "model/qwen4/loader/Validator.java",
            "model/qwen4/loader/NgramStore.java",
            "model/qwen4/expert/RamTierPreload.java");

    /// Files that hold a lock only to close what they own; their execution paths hold none.
    private static final Set<String> LIFECYCLE = Set.of(
            "model/qwen4/Qwen4Runtime.java",
            "model/qwen4/Qwen4Model.java",
            "model/qwen4/Storage.java",
            "model/qwen4/Session.java");

    private static List<Path> javaFiles(String directory, boolean recursive) throws IOException {
        try (Stream<Path> walk =
                recursive ? Files.walk(MAIN.resolve(directory)) : Files.list(MAIN.resolve(directory))) {
            return walk.filter(p -> p.toString().endsWith(".java")).toList();
        }
    }

    private static List<Path> executionSources() throws IOException {
        List<Path> files = new ArrayList<>();
        files.add(MAIN.resolve("runtime/HostTasks.java"));
        files.add(MAIN.resolve("runtime/HostFrames.java"));
        files.addAll(javaFiles("model/qwen4", true));
        return files.stream().filter(p -> !STARTUP.contains(relative(p))).toList();
    }

    @Test
    void flashNextExecutionCreatesNoSchedulingInfrastructureOfItsOwn() throws IOException {
        List<String> violations = new ArrayList<>();
        for (Path file : executionSources()) {
            List<String> lines = Files.readAllLines(file);
            for (int i = 0; i < lines.size(); i++) {
                String line = lines.get(i);
                String code = line.strip();
                if (code.startsWith("///") || code.startsWith("//") || code.startsWith("*")) continue;
                for (Pattern forbidden : FORBIDDEN)
                    if (forbidden.matcher(line).find()) violations.add(relative(file) + ":" + (i + 1) + ": " + code);
            }
        }
        assertTrue(violations.isEmpty(), "private scheduling infrastructure:\n" + String.join("\n", violations));
    }

    @Test
    void theRuntimeOwnsResourcesNotThreads() throws IOException {
        String runtime = Files.readString(MAIN.resolve("model/qwen4/Qwen4Runtime.java"));
        assertTrue(!runtime.contains("Executor"));
        assertTrue(!runtime.contains("flash-next-generation") && !runtime.contains("flash-next-host"));
    }

    /// The work of a step is the shape of a graph that the lattice's runtime runs, not a runner of
    /// our own: no executor or step machine exists, no stage runs another or waits for one, and
    /// every piece of the model's execution is a
    /// [io.euhedral_execution.inference.core.runtime.graph.StageFrame] of the shape.
    @Test
    void aStepIsTheShapeOfAStageGraphNotARunner() throws IOException {
        Path qwen4 = MAIN.resolve("model/qwen4");
        assertTrue(!Files.exists(qwen4.resolve("Executor.java")), "a private executor");
        assertTrue(!Files.exists(qwen4.resolve("Step.java")), "a private step machine");
        String stages = Files.readString(qwen4.resolve("Stages.java"));
        assertTrue(stages.contains("extends StageFrame"), "stages are frames of the lattice's graph");
        List<String> offenders = new ArrayList<>();
        for (Path p : javaFiles("model/qwen4", false)) {
            // The session runs the generation chain as host tasks until PR 3 makes it frames.
            if (relative(p).equals("model/qwen4/Session.java")) continue;
            String text = Files.readString(p);
            if (text.contains("HostFrames.Task") || text.contains("AtomicReference<Step>")) offenders.add(relative(p));
        }
        assertTrue(offenders.isEmpty(), "host work outside the graph: " + offenders);
    }

    /// The path through the cache and back holds no lock: its bookkeeping is changed only by frames
    /// routed to its owner, and everything else reaches it as such a frame.
    @Test
    void theExpertHotPathHoldsNoLockAndStartsNoThread() throws IOException {
        Pattern lock = Pattern.compile(
                "\\bsynchronized\\b|\\bReentrantLock\\b|\\bReadWriteLock\\b|\\bCondition\\b|\\bSemaphore\\b|\\.wait\\(|\\bnew Thread\\(|\\bExecutors\\b");
        List<Path> hot = new ArrayList<>();
        hot.addAll(javaFiles("model/qwen4", false));
        hot.addAll(javaFiles("model/qwen4/expert", true));
        hot.add(MAIN.resolve("runtime/graph/Join.java"));
        List<String> violations = new ArrayList<>();
        for (Path file : hot) {
            // The resident tier is read by loader threads that end with the load, before any request exists.
            if (file.getFileName().toString().equals("RamTierPreload.java")) continue;
            if (LIFECYCLE.contains(relative(file))) continue;
            List<String> lines = Files.readAllLines(file);
            for (int i = 0; i < lines.size(); i++) {
                String code = lines.get(i).strip();
                if (code.startsWith("///") || code.startsWith("//") || code.startsWith("*")) continue;
                if (lock.matcher(code).find()) violations.add(relative(file) + ":" + (i + 1) + ": " + code);
            }
        }
        assertTrue(violations.isEmpty(), "locks or threads on the hot path:\n" + String.join("\n", violations));
    }

    /// Loading experts is ordinary lattice work, not a subsystem attached to the lattice: frames, dependencies,
    /// and state confined by routing every frame that touches it to its owner. Nothing in the expert path is a
    /// source or a sink of its own, keeps a queue of work, admits work, groups experts into waves, binds a staging
    /// buffer to a scheduling identity, or locks the owner's state.
    @Test
    void theExpertPathIsFramesAndDependenciesNotAScheduler() throws IOException {
        for (String gone : List.of(
                "runtime/graph/SerialSource.java",
                "runtime/graph/Confined.java",
                "runtime/graph/GatedSink.java",
                "model/qwen4/ExpertSource.java",
                "model/qwen4/ExpertOwner.java",
                "model/qwen4/StagingPool.java",
                "model/qwen4/ExpertWave.java"))
            assertTrue(!Files.exists(MAIN.resolve(gone)), "a scheduler of the expert path: " + gone);
        Pattern scheduler = Pattern.compile("implements\\s+LatticeSource|extends\\s+AbstractIngestSink"
                + "|(Deque|Queue|List)<(AbstractFrame|ExpertLoad|Claim)>|\\bclass\\s+(Lane|Claim)\\b"
                + "|\\btryExclusive\\(|\\bfreeLane\\(|\\bwaveStart\\(|\\bWAVE\\b");
        List<Path> scanned = new ArrayList<>();
        scanned.addAll(javaFiles("model/qwen4", false));
        scanned.addAll(javaFiles("model/qwen4/expert", true));
        scanned.addAll(javaFiles("runtime/graph", true));
        List<String> violations = new ArrayList<>();
        for (Path file : scanned) {
            // The lattice's ingest sinks (the lake's) are its sources; the lake counts the units it admits for its
            // own completion. The asynchronous reads' sink is how the disk's completions reach the lattice, as driver
            // callbacks publish the device's.
            String name = file.getFileName().toString();
            if (name.equals("InferenceLake.java") || name.equals("FrameLake.java") || name.equals("AsyncReads.java"))
                continue;
            List<String> lines = Files.readAllLines(file);
            for (int i = 0; i < lines.size(); i++) {
                String code = lines.get(i).strip();
                if (code.startsWith("///") || code.startsWith("//") || code.startsWith("*")) continue;
                if (scheduler.matcher(code).find()) violations.add(relative(file) + ":" + (i + 1) + ": " + code);
            }
        }
        assertTrue(violations.isEmpty(), "a scheduler in the expert path:\n" + String.join("\n", violations));
    }

    /// The packages of the core: shared layers and the model folders. The old split packages are gone.
    private static final Set<String> LAYERS = Set.of(
            "artifact",
            "generation",
            "gpu",
            "guidance",
            "model",
            "prefix",
            "runtime",
            "sampling",
            "state",
            "tokenizer");

    /// Directories count only when they hold sources: git does not track an empty directory a move leaves behind.
    @Test
    void everyClassLivesInALayerOrAModelFolder() throws IOException {
        try (Stream<Path> top = Files.list(MAIN)) {
            List<String> unknown = new ArrayList<>();
            for (Path directory : top.filter(Files::isDirectory).toList()) {
                String name = directory.getFileName().toString();
                if (!LAYERS.contains(name) && !javaFiles(name, true).isEmpty()) unknown.add(name);
            }
            assertTrue(unknown.isEmpty(), "packages outside the layout (old packages included): " + unknown);
        }
    }

    /// Only the types other packages see carry a model prefix; inside a model folder, classes are named by role.
    private static final Set<String> PREFIXED = Set.of(
            "QwenTokenizer",
            "Qwen38Runtime",
            "Qwen38Model",
            "Qwen38Config",
            "Qwen4Runtime",
            "Qwen4Model",
            "Qwen4Config");

    @Test
    void onlyEdgeTypesCarryAModelPrefix() throws IOException {
        try (Stream<Path> walk = Files.walk(MAIN)) {
            List<String> prefixed = walk.filter(p -> p.toString().endsWith(".java"))
                    .map(p -> p.getFileName().toString().replace(".java", ""))
                    .filter(name -> name.startsWith("Qwen") && !PREFIXED.contains(name))
                    .toList();
            assertTrue(prefixed.isEmpty(), "model-prefixed classes: " + prefixed);
        }
    }

    /// The shared artifact layer reads containers and tensors for every model; it knows no model.
    @Test
    void theSharedArtifactLayerKnowsNoModel() throws IOException {
        List<String> violations = new ArrayList<>();
        for (Path file : javaFiles("artifact", true))
            for (String line : Files.readAllLines(file))
                if (line.startsWith("import io.euhedral_execution.inference.core.model."))
                    violations.add(relative(file) + ": " + line);
        assertTrue(violations.isEmpty(), "shared artifact code imports a model:\n" + String.join("\n", violations));
    }

    /// Shared code knows no model: only the engine, which chooses a model's runtime, imports one. The other entries
    /// are what PR 2 is removing; each task deletes the ones it fixes.
    private static final Set<String> MODEL_IMPORTS_OUTSIDE_MODEL =
            Set.of("InferenceEngine.java", "benchmark:run/BenchmarkRunner.java", "benchmark:run/Prerequisites.java");

    private static final Path API = Path.of("../api/src/main/java/io/euhedral_execution/inference/api");
    private static final Path BENCHMARK =
            Path.of("../benchmark/src/main/java/io/euhedral_execution/inference/benchmark");

    private static boolean importsAModel(Path file) throws IOException {
        return Files.readAllLines(file).stream()
                .anyMatch(line -> line.startsWith("import io.euhedral_execution.inference.core.model."));
    }

    @Test
    void sharedCodeImportsNoModel() throws IOException {
        List<String> violations = new ArrayList<>();
        for (var tree : List.of(Map.entry(MAIN, ""), Map.entry(API, "api:"), Map.entry(BENCHMARK, "benchmark:"))) {
            try (Stream<Path> walk = Files.walk(tree.getKey())) {
                for (Path file :
                        walk.filter(p -> p.toString().endsWith(".java")).toList()) {
                    String path = tree.getKey().relativize(file).toString().replace('\\', '/');
                    if (tree.getValue().isEmpty() && path.startsWith("model/")) continue;
                    String name = tree.getValue() + path;
                    if (importsAModel(file) && !MODEL_IMPORTS_OUTSIDE_MODEL.contains(name)) violations.add(name);
                }
            }
        }
        assertTrue(violations.isEmpty(), "shared code imports a model: " + violations);
    }

    /// Entries in the exception list that no longer import a model must leave it, so it only shrinks.
    @Test
    void theExceptionListOnlyNamesRealImports() throws IOException {
        List<String> stale = new ArrayList<>();
        for (String name : MODEL_IMPORTS_OUTSIDE_MODEL) {
            Path file = name.startsWith("api:")
                    ? API.resolve(name.substring(4))
                    : name.startsWith("benchmark:") ? BENCHMARK.resolve(name.substring(10)) : MAIN.resolve(name);
            if (!Files.exists(file) || !importsAModel(file)) stale.add(name);
        }
        assertTrue(stale.isEmpty(), "no longer imports a model, remove from the list: " + stale);
    }

    /// A source path relative to the main tree, with forward slashes on every platform.
    private static String relative(Path file) {
        return MAIN.relativize(file).toString().replace('\\', '/');
    }
}
