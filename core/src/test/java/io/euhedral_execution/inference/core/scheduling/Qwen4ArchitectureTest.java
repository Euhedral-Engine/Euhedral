package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/// A guard on the shape of Flash-Next host execution, in addition to the behavioural tests: the
/// model's runtime and its execution path create no scheduling infrastructure of their own. Every
/// piece of host work is a frame on the lattice; the only waiting a path may do is a parked
/// continuation.
class Qwen4ArchitectureTest {

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
            "model_loader/qwen4/Qwen4FixedLoader.java",
            "model_loader/qwen4/Qwen4Validator.java",
            "model_loader/qwen4/NgramStore.java",
            "model_loader/qwen4/expert/RamTierPreload.java");

    private static List<Path> executionSources() throws IOException {
        List<Path> files = new ArrayList<>();
        files.add(MAIN.resolve("Qwen4Runtime.java"));
        files.add(MAIN.resolve("Qwen4Storage.java"));
        files.add(MAIN.resolve("scheduling/Qwen4GenerationSession.java"));
        files.add(MAIN.resolve("scheduling/HostTasks.java"));
        for (String directory : List.of("host", "qwen4", "model_loader/qwen4")) {
            try (Stream<Path> walk = Files.walk(MAIN.resolve(directory))) {
                walk.filter(p -> p.toString().endsWith(".java")).forEach(files::add);
            }
        }
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
        String runtime = Files.readString(MAIN.resolve("Qwen4Runtime.java"));
        assertTrue(!runtime.contains("Executor"));
        assertTrue(!runtime.contains("flash-next-generation") && !runtime.contains("flash-next-host"));
    }

    /// The work of a step is the shape of a graph that the lattice's runtime runs, not a runner of
    /// our own: no executor or step machine exists, no stage runs another or waits for one, and
    /// every piece of the model's execution is a [StageFrame](../scheduling/graph/StageFrame.java)
    /// of the shape.
    @Test
    void aStepIsTheShapeOfAStageGraphNotARunner() throws IOException {
        Path qwen4 = MAIN.resolve("qwen4");
        assertTrue(!Files.exists(qwen4.resolve("Qwen4Executor.java")), "a private executor");
        assertTrue(!Files.exists(qwen4.resolve("Qwen4Step.java")), "a private step machine");
        String stages = Files.readString(qwen4.resolve("Qwen4Stages.java"));
        assertTrue(stages.contains("extends StageFrame"), "stages are frames of the lattice's graph");
        try (Stream<Path> walk = Files.walk(qwen4)) {
            List<String> offenders = walk.filter(p -> p.toString().endsWith(".java"))
                    .filter(p -> {
                        try {
                            String text = Files.readString(p);
                            return text.contains("HostFrames.Task") || text.contains("AtomicReference<Qwen4Step>");
                        } catch (IOException failure) {
                            throw new java.io.UncheckedIOException(failure);
                        }
                    })
                    .map(p -> relative(p))
                    .toList();
            assertTrue(offenders.isEmpty(), "host work outside the graph: " + offenders);
        }
    }

    /// The path through the cache and back holds no lock: its bookkeeping is changed only by frames
    /// routed to its owner, and everything else reaches it as such a frame.
    @Test
    void theExpertHotPathHoldsNoLockAndStartsNoThread() throws IOException {
        Pattern lock = Pattern.compile(
                "\\bsynchronized\\b|\\bReentrantLock\\b|\\bReadWriteLock\\b|\\bCondition\\b|\\bSemaphore\\b|\\.wait\\(|\\bnew Thread\\(|\\bExecutors\\b");
        List<Path> hot = new ArrayList<>();
        for (String directory : List.of("qwen4", "model_loader/qwen4/expert")) {
            try (Stream<Path> walk = Files.walk(MAIN.resolve(directory))) {
                walk.filter(p -> p.toString().endsWith(".java")).forEach(hot::add);
            }
        }
        hot.add(MAIN.resolve("scheduling/graph/Join.java"));
        List<String> violations = new ArrayList<>();
        for (Path file : hot) {
            // The resident tier is read by loader threads that end with the load, before any request exists.
            if (file.getFileName().toString().equals("RamTierPreload.java")) continue;
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
                "scheduling/graph/SerialSource.java",
                "scheduling/graph/Confined.java",
                "scheduling/graph/GatedSink.java",
                "qwen4/ExpertSource.java",
                "qwen4/ExpertOwner.java",
                "qwen4/StagingPool.java",
                "qwen4/Qwen4ExpertWave.java"))
            assertTrue(!Files.exists(MAIN.resolve(gone)), "a scheduler of the expert path: " + gone);
        Pattern scheduler = Pattern.compile("implements\\s+LatticeSource|extends\\s+AbstractIngestSink"
                + "|(Deque|Queue|List)<(AbstractFrame|ExpertLoad|Claim)>|\\bclass\\s+(Lane|Claim)\\b"
                + "|\\btryExclusive\\(|\\bfreeLane\\(|\\bwaveStart\\(|\\bWAVE\\b");
        List<String> violations = new ArrayList<>();
        for (String directory : List.of("qwen4", "model_loader/qwen4/expert", "scheduling/graph")) {
            try (Stream<Path> walk = Files.walk(MAIN.resolve(directory))) {
                for (Path file :
                        walk.filter(p -> p.toString().endsWith(".java")).toList()) {
                    // The lattice's ingest sinks (the lake's, the dense runtime's) are its sources; the lake counts the
                    // units it admits for its own completion.
                    String name = file.getFileName().toString();
                    // The asynchronous reads' sink is how the disk's completions reach the lattice, as driver
                    // callbacks publish the device's.
                    if (name.equals("InferenceLake.java")
                            || name.equals("FrameLake.java")
                            || name.equals("AsyncReads.java")) continue;
                    List<String> lines = Files.readAllLines(file);
                    for (int i = 0; i < lines.size(); i++) {
                        String code = lines.get(i).strip();
                        if (code.startsWith("///") || code.startsWith("//") || code.startsWith("*")) continue;
                        if (scheduler.matcher(code).find())
                            violations.add(relative(file) + ":" + (i + 1) + ": " + code);
                    }
                }
            }
        }
        assertTrue(violations.isEmpty(), "a scheduler in the expert path:\n" + String.join("\n", violations));
    }

    /// A source path relative to the main tree, with forward slashes on every platform.
    private static String relative(Path file) {
        return MAIN.relativize(file).toString().replace('\\', '/');
    }
}
