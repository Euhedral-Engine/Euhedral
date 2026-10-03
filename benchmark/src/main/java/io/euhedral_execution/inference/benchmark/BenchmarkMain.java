package io.euhedral_execution.inference.benchmark;

import io.euhedral_execution.inference.benchmark.config.BenchmarkOptions;
import io.euhedral_execution.inference.benchmark.result.BenchmarkResult;
import io.euhedral_execution.inference.benchmark.result.ResultStore;
import io.euhedral_execution.inference.benchmark.result.Summary;
import io.euhedral_execution.inference.benchmark.run.BenchmarkRunner;
import io.euhedral_execution.inference.benchmark.run.EngineTarget;
import io.euhedral_execution.inference.benchmark.run.Prerequisites;
import io.euhedral_execution.inference.core.ProcessorTopology;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.time.Clock;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/// Command line: `run CONFIG.json` measures this engine end to end; `import` stores external rows.
/// See `docs/BENCHMARKING.md`.
public final class BenchmarkMain {
    static final String USAGE = """
            usage:
              run CONFIG.json [--fork-id ID] [--validate-only]
              import --input FILE --implementation LABEL --output FILE [--note TEXT] [--overwrite|--append]

            CONFIG.json requires "artifact", "tokenizer", and "cudaLibrary"; see docs/BENCHMARKING.md.
            --fork-id labels this JVM as one externally launched fork; iterations are never forks.
            --validate-only checks configuration and prerequisites, prints the plan, and exits without
            touching the GPU. Exit codes: 0 success, 1 a measured row failed or was ineligible, 2 usage,
            3 the engine did not load.
            """;

    private BenchmarkMain() {}

    public static void main(String[] args) throws Exception {
        System.exit(execute(Arrays.asList(args)));
    }

    static int execute(List<String> args) throws Exception {
        if (args.isEmpty() || args.contains("--help")) {
            System.out.print(USAGE);
            return args.isEmpty() ? 2 : 0;
        }
        List<String> rest = args.subList(1, args.size());
        try {
            return switch (args.getFirst()) {
                case "run" -> run(rest, String.join(" ", args));
                case "import" -> importResults(rest);
                default -> throw new IllegalArgumentException("unknown command " + args.getFirst());
            };
        } catch (IllegalArgumentException invalid) {
            System.err.println("error: " + invalid.getMessage());
            System.err.print(USAGE);
            return 2;
        }
    }

    private static int run(List<String> args, String commandLine) throws IOException, InterruptedException {
        if (args.isEmpty() || args.getFirst().startsWith("--"))
            throw new IllegalArgumentException("run needs a configuration file");
        Path configFile = Path.of(args.getFirst());
        String forkId = null;
        boolean validateOnly = false;
        for (int index = 1; index < args.size(); index++) {
            switch (args.get(index)) {
                case "--validate-only" -> validateOnly = true;
                case "--fork-id" -> {
                    if (index + 1 >= args.size() || args.get(index + 1).isBlank())
                        throw new IllegalArgumentException("--fork-id needs a value");
                    forkId = args.get(++index);
                }
                default -> throw new IllegalArgumentException("unknown run option " + args.get(index));
            }
        }
        BenchmarkOptions options;
        try {
            options = BenchmarkOptions.load(configFile, Instant.now());
        } catch (com.fasterxml.jackson.databind.JsonMappingException invalid) {
            Throwable cause = invalid.getCause() instanceof IllegalArgumentException argument ? argument : invalid;
            throw new IllegalArgumentException(configFile + ": " + cause.getMessage(), invalid);
        }
        var prepared = Prerequisites.check(options, ProcessorTopology.system());
        System.out.println("workers: processors " + prepared.workers().processorIds() + ", cores "
                + prepared.workers().coreIds());
        System.out.println("max context: " + options.maxContextTokens() + " tokens; warmup " + options.warmup()
                + ", measured " + options.iterations() + " per scenario; generation " + options.generation()
                + "; fork " + (forkId == null ? "unspecified" : forkId));
        prepared.prompts()
                .forEach((scenario, prompts) -> prompts.forEach(prompt -> System.out.println("scenario "
                        + scenario.name() + ": " + prompt.generator() + " prompt " + prompt.actualTokens() + "/"
                        + prompt.targetTokens() + " tokens, sha256 " + prompt.sha256())));
        System.out.println("output: " + options.output().toAbsolutePath());
        if (validateOnly) {
            System.out.println("validation passed; no model was loaded");
            return 0;
        }
        var runtime = ManagementFactory.getRuntimeMXBean();
        var context = new BenchmarkRunner.Context(
                UUID.randomUUID().toString(),
                new BenchmarkResult.Fork(
                        forkId,
                        forkId == null ? "unspecified" : "external",
                        runtime.getPid(),
                        Instant.ofEpochMilli(runtime.getStartTime()).toString()),
                new BenchmarkResult.Provenance(
                        "measured", "euhedral-inference-benchmark", commandLine, null, null, null),
                Clock.systemUTC());
        var factory = EngineTarget.factory(
                options.artifact(),
                options.tokenizer(),
                options.cudaLibrary(),
                prepared.workers().processorIds(),
                options.maxContextTokens(),
                options.shutdownTimeout());
        // Load before creating the output, so a refused load leaves no empty result file.
        BenchmarkRunner.Target target;
        try {
            target = factory.open();
        } catch (IOException failure) {
            System.err.println("not run: the engine did not load: " + failure.getMessage());
            return 3;
        }
        List<BenchmarkResult> rows;
        ResultStore.Writer opened;
        try {
            opened = new ResultStore.Writer(options.output(), options.json(), options.overwrite(), options.append());
        } catch (IOException | RuntimeException failure) {
            target.close();
            throw failure;
        }
        try (var writer = opened) {
            BenchmarkRunner.run(options, prepared.prompts(), () -> target, context, row -> {
                writer.write(row);
                System.out.println((row.warmup() ? "warmup " : "measured ")
                        + row.scenario().name() + " #"
                        + row.iteration() + ": " + row.status()
                        + (row.statusReason() == null ? "" : " (" + row.statusReason() + ")"));
            });
            rows = writer.rows();
        }
        System.out.println();
        System.out.print(Summary.format(rows));
        System.out.println("results: " + options.output().toAbsolutePath());
        return measuredExitCode(rows);
    }

    static int measuredExitCode(List<BenchmarkResult> rows) {
        return rows.stream().filter(row -> !row.warmup()).anyMatch(row -> !BenchmarkResult.SUCCESS.equals(row.status()))
                ? 1
                : 0;
    }

    private static int importResults(List<String> args) throws IOException {
        Map<String, String> values = new java.util.LinkedHashMap<>();
        for (int index = 0; index < args.size(); index++) {
            String key = args.get(index);
            switch (key) {
                case "--overwrite", "--append" -> values.put(key.substring(2), "true");
                case "--input", "--implementation", "--output", "--note" -> {
                    if (index + 1 >= args.size()) throw new IllegalArgumentException(key + " needs a value");
                    values.put(key.substring(2), args.get(++index));
                }
                default -> throw new IllegalArgumentException("unknown import option " + key);
            }
        }
        for (String required : List.of("input", "implementation", "output"))
            if (!values.containsKey(required)) throw new IllegalArgumentException("--" + required + " is required");
        Path output = Path.of(values.get("output"));
        boolean overwrite = values.containsKey("overwrite");
        boolean append = values.containsKey("append");
        if (overwrite && append) throw new IllegalArgumentException("choose either overwrite or append");
        boolean json = output.toString().endsWith(".json");
        if (append && json) throw new IllegalArgumentException("append requires JSONL output");
        var rows =
                ResultStore.importRows(Path.of(values.get("input")), values.get("implementation"), values.get("note"));
        try (var writer = new ResultStore.Writer(output, json, overwrite, append)) {
            rows.forEach(writer::write);
        }
        System.out.print(Summary.format(rows));
        System.out.println("imported " + rows.size() + " rows into " + output.toAbsolutePath());
        return 0;
    }
}
