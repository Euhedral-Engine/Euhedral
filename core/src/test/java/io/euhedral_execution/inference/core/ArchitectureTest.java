package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
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

    /// Nothing on either model's hot path holds a lock, waits blocking or starts a thread: state is confined by
    /// routing every frame that touches it to its owner, and completions arrive as frames. Locks, waits and threads
    /// are allowed only in the lifecycle and tool methods below (opening, closing, the blocking caller-thread
    /// forms), and in the startup loaders.
    @Test
    void theHotPathHoldsNoLockBlocksNorStartsThreads() throws IOException {
        List<String> violations = new ArrayList<>();
        for (Path file : hotPathSources()) {
            String path = relative(file);
            if (path.startsWith("model/qwen4/loader/") || path.equals("model/qwen4/expert/RamTierPreload.java"))
                continue;
            List<String> lines = Files.readAllLines(file);
            for (int i = 0; i < lines.size(); i++) {
                String code = lines.get(i).strip();
                if (code.startsWith("///") || code.startsWith("//") || code.startsWith("*")) continue;
                // Imports and fields name what a method may use; the methods that use them are what is checked.
                if (code.startsWith("import ") || !BLOCKING.matcher(code).find()) continue;
                String method = enclosingMethod(lines, i);
                if (method.equals("?") && FIELD.matcher(code).find()) continue;
                if (LIFECYCLE_METHODS.contains(path + "#" + method)) continue;
                violations.add(path + ":" + (i + 1) + " (" + method + "): " + code);
            }
        }
        assertTrue(
                violations.isEmpty(),
                "locks, blocking waits or threads on the hot path:\n" + String.join("\n", violations));
    }

    /// A lock, a blocking wait, or a thread. A future's blocking `get` is found by its usual receivers and by its
    /// timed form.
    static final Pattern BLOCKING = Pattern.compile("\\bsynchronized\\b"
            + "|\\b(ReentrantLock|ReadWriteLock|StampedLock|Semaphore|Condition)\\b|\\.wait\\(|LockSupport\\.park"
            + "|\\.join\\(\\)|\\bnew Thread\\b|Thread\\.of(Platform|Virtual)|\\bExecutors\\b|\\bExecutorService\\b"
            + "|\\.(lock|lockInterruptibly)\\(\\)|\\.tryLock\\(|\\.(acquire|acquireUninterruptibly)\\(\\d*\\)"
            + "|\\.await\\(\\)|\\.await\\(\\d+\\s*,"
            + "|(result|future|outcome|done|settled|completion)\\w*\\(\\)\\.get\\(|\\.get\\(\\d+\\s*,\\s*TimeUnit");

    /// The lifecycle and tool methods that may lock, wait or start threads, as `path#method`.
    static final Set<String> LIFECYCLE_METHODS = Set.of(
            "runtime/EuhedralInferenceRuntime.java#close",
            "runtime/EuhedralInferenceRuntime.java#awaitIdle",
            "runtime/EuhedralInferenceRuntime.java#openLanes",
            "runtime/EuhedralInferenceRuntime.java#openPool",
            "runtime/EuhedralInferenceRuntime.java#release",
            "runtime/graph/InferenceLake.java#attach",
            "runtime/graph/InferenceLake.java#detach",
            "runtime/graph/InferenceLake.java#signalComplete",
            "runtime/graph/InferenceLake.java#awaitTermination",
            "runtime/HostTasks.java#close",
            "gpu/CudaGpuMemory.java#close",
            "model/qwen38/Execution.java#execute",
            "model/qwen38/Execution.java#close",
            "model/qwen38/Session.java#close",
            "model/qwen38/Session.java#completeClose",
            "model/qwen38/Emission.java#text",
            "model/qwen38/Emission.java#end",
            "model/qwen38/Emission.java#drain",
            "model/qwen38/SequenceCleanup.java#close",
            "model/qwen38/Qwen38Model.java#close",
            "model/qwen38/prefix/PrefixCache.java#close",
            "model/qwen38/speculative/MtpDecoder.java#generate",
            "model/qwen38/speculative/DFlash2Decoder.java#generate",
            "model/qwen4/Session.java#close",
            "model/qwen4/Session.java#completeClose",
            "model/qwen4/Qwen4Runtime.java#stop",
            "model/qwen4/Qwen4Model.java#close",
            "model/qwen4/Storage.java#close");

    private static List<Path> hotPathSources() throws IOException {
        List<Path> files = new ArrayList<>();
        for (String directory :
                List.of("model/qwen38", "model/qwen4", "runtime", "generation", "prefix", "state", "gpu"))
            files.addAll(javaFiles(directory, true));
        return files;
    }

    private static final Pattern FIELD = Pattern.compile(
            "^(private|protected|public)?\\s*(static\\s+)?(final\\s+)?[\\w<>\\[\\]?, .]+\\s+\\w+\\s*(=.*)?;$");

    private static final Pattern SIGNATURE = Pattern.compile(
            "^\\s*(public|protected|private|static|final|synchronized|abstract|default|\\s)*[\\w<>\\[\\]?, .]+\\s+(\\w+)\\s*\\(");
    private static final Set<String> NOT_METHODS =
            Set.of("if", "for", "while", "switch", "catch", "synchronized", "return", "new", "throw", "else");

    /// The method whose body holds line `at`: the nearest enclosing block opened by a method signature (blocks of
    /// statements, lambdas and anonymous classes are looked through); the signature's own line names itself.
    static String enclosingMethod(List<String> lines, int at) {
        String own = methodName(lines.get(at));
        if (own != null) return own;
        int depth = 0;
        for (int j = at - 1; j >= 0; j--) {
            String code = lines.get(j).strip();
            if (code.startsWith("//") || code.startsWith("*")) continue;
            for (int c = code.length() - 1; c >= 0; c--) {
                char ch = code.charAt(c);
                if (ch == '}') depth++;
                else if (ch == '{' && --depth < 0) {
                    // A block opens here: a method's, or one to look through.
                    for (int k = j; k >= Math.max(0, j - 8); k--) {
                        String name = methodName(lines.get(k));
                        if (name != null && !code.contains("->")) return name;
                        if (lines.get(k).strip().endsWith(";")
                                || lines.get(k).strip().endsWith("}")) break;
                    }
                    depth = 0;
                    break;
                }
            }
        }
        return "?";
    }

    private static String methodName(String line) {
        String code = line.strip();
        if (code.startsWith("//")
                || code.startsWith("*")
                || code.contains("->")
                || code.contains(" new ")
                || code.contains("=")
                || code.startsWith("return ")) return null;
        var matcher = SIGNATURE.matcher(line);
        if (!matcher.find()) return null;
        String name = matcher.group(2);
        return NOT_METHODS.contains(name) ? null : name;
    }

    /// Loading experts is ordinary lattice work, not a subsystem attached to the lattice: frames, dependencies,
    /// and one source for the state they share. The cache's bookkeeping belongs to `ExpertCacheOwner`, a source
    /// the lattice polls one worker at a time (so the state needs no lock): it applies the records posted to it
    /// and keeps the requests it cannot serve yet. Nothing else in the expert path is a source or a sink of its
    /// own, keeps a queue of work, admits work, groups experts into waves, binds a staging buffer to a scheduling
    /// identity, or locks the owner's state.
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
        Pattern source = Pattern.compile("implements\\s+LatticeSource|extends\\s+AbstractIngestSink");
        Pattern scheduler = Pattern.compile("(Deque|Queue|List)<(AbstractFrame|ExpertLoad|Claim)>"
                + "|\\bclass\\s+(Lane|Claim)\\b|\\btryExclusive\\(|\\bfreeLane\\(|\\bwaveStart\\(|\\bWAVE\\b");
        List<Path> scanned = new ArrayList<>();
        scanned.addAll(javaFiles("model/qwen4", false));
        scanned.addAll(javaFiles("model/qwen4/expert", true));
        scanned.addAll(javaFiles("runtime/graph", true));
        List<String> violations = new ArrayList<>();
        for (Path file : scanned) {
            // The lattice's ingest sinks (the lake's) are its sources; the lake counts the units it admits for its
            // own completion. The asynchronous reads' sink is how the disk's completions reach the lattice, as driver
            // callbacks publish the device's. The cache's owner is the one source of the expert path.
            String name = file.getFileName().toString();
            if (name.equals("InferenceLake.java") || name.equals("FrameLake.java") || name.equals("AsyncReads.java"))
                continue;
            boolean owner = name.equals("ExpertCacheOwner.java");
            List<String> lines = Files.readAllLines(file);
            for (int i = 0; i < lines.size(); i++) {
                String code = lines.get(i).strip();
                if (code.startsWith("///") || code.startsWith("//") || code.startsWith("*")) continue;
                if (scheduler.matcher(code).find()
                        || (!owner && source.matcher(code).find()))
                    violations.add(relative(file) + ":" + (i + 1) + ": " + code);
            }
        }
        assertTrue(violations.isEmpty(), "a scheduler in the expert path:\n" + String.join("\n", violations));
    }

    /// The packages of the core: shared layers and the model folders. The old split packages are gone.
    /// The generation path runs as frames: no future chains and no locks. A session's `close` may wait (lifecycle),
    /// `completeClose` is `synchronized` as a method, not a block, and the prefix cache's futures are met by one
    /// `whenComplete` each until the cache becomes frames.
    @Test
    void generationRunsAsFramesWithoutFutureChainsOrLocks() throws IOException {
        Pattern forbidden = Pattern.compile(
                "\\.then(Compose|Apply|Accept|Run|Combine)(Async)?\\(|\\.whenComplete(Async)?\\(|\\bsynchronized\\s*\\("
                        + "|\\bReentrantLock\\b|\\bSemaphore\\b|\\bBlockingQueue\\b|\\.wait\\(");
        // The prefix cache is owner frames and device-completion frames: no futures, and no lock but its close.
        Pattern prefixForbidden = Pattern.compile("\\bCompletableFuture\\b|\\bsynchronized\\b(?!.*\\bclose\\(\\))");
        List<Path> prefix = new ArrayList<>(javaFiles("prefix", false));
        prefix.addAll(javaFiles("model/qwen38/prefix", false));
        List<Path> files = new ArrayList<>(javaFiles("generation", false));
        files.add(MAIN.resolve("model/qwen38/Session.java"));
        files.add(MAIN.resolve("model/qwen38/Emission.java"));
        files.add(MAIN.resolve("model/qwen4/Session.java"));
        files.add(MAIN.resolve("model/qwen38/speculative/MtpDecoder.java"));
        files.add(MAIN.resolve("model/qwen38/speculative/DFlash2Decoder.java"));
        List<String> violations = new ArrayList<>();
        for (Path file : files) {
            List<String> lines = Files.readAllLines(file);
            for (int i = 0; i < lines.size(); i++) {
                String code = lines.get(i).strip();
                if (code.startsWith("///") || code.startsWith("//") || code.startsWith("*")) continue;
                if (forbidden.matcher(code).find()) violations.add(relative(file) + ":" + (i + 1) + ": " + code);
            }
        }
        for (Path file : prefix) {
            List<String> lines = Files.readAllLines(file);
            for (int i = 0; i < lines.size(); i++) {
                String code = lines.get(i).strip();
                if (code.startsWith("///") || code.startsWith("//") || code.startsWith("*")) continue;
                if (forbidden.matcher(code).find()
                        || prefixForbidden.matcher(code).find())
                    violations.add(relative(file) + ":" + (i + 1) + ": " + code);
            }
        }
        assertTrue(
                violations.isEmpty(),
                "future chains or locks on the generation path:\n" + String.join("\n", violations));
    }

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
    private static final Set<String> MODEL_IMPORTS_OUTSIDE_MODEL = Set.of("InferenceEngine.java");

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

    @Test
    void onlyTheEngineNamesAModel() {
        assertEquals(Set.of("InferenceEngine.java"), MODEL_IMPORTS_OUTSIDE_MODEL);
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
