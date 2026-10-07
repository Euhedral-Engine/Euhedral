package io.euhedral_execution.inference.benchmark.run;

import io.euhedral_execution.inference.benchmark.config.BenchmarkOptions;
import io.euhedral_execution.inference.benchmark.config.Scenario;
import io.euhedral_execution.inference.benchmark.prompt.ChatPromptCorpus;
import io.euhedral_execution.inference.benchmark.prompt.PromptMaterial;
import io.euhedral_execution.inference.benchmark.result.ResultStore;
import io.euhedral_execution.inference.core.ProcessorTopology;
import io.euhedral_execution.inference.core.WorkerProcessorSelection;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/// Checks everything that can be checked before the GPU or model is touched: input files, output
/// path, worker CPUs against host topology, artifact metadata, prompt sizes, and model context.
public final class Prerequisites {
    private Prerequisites() {}

    public record Prepared(
            WorkerProcessorSelection workers, Map<Scenario, List<PromptMaterial>> prompts, int maxPositionEmbeddings) {}

    public static Prepared check(BenchmarkOptions options, ProcessorTopology topology) throws IOException {
        requireFile(options.artifact(), "artifact");
        requireFile(options.cudaLibrary(), "cuda-library");
        if (!Files.isRegularFile(options.tokenizer().resolve("tokenizer.json")))
            throw new IllegalArgumentException("tokenizer directory has no tokenizer.json: " + options.tokenizer());
        ResultStore.checkOutput(options.output(), options.overwrite(), options.append());
        WorkerProcessorSelection workers =
                resolveWorkers(options.cpus(), options.excludeCpus(), options.excludeCores(), topology);

        int modelContext = ArtifactReader.read(options.artifact()).config().maxPositionEmbeddings();
        int context = Math.min(modelContext, options.maxContextTokens());
        QwenTokenizer tokenizer = QwenTokenizer.load(options.tokenizer());
        // One prompt per scenario (words), or one per corpus task (chat): iteration i runs prompt i mod size.
        Map<Integer, List<PromptMaterial>> byTarget = new HashMap<>();
        Map<Scenario, List<PromptMaterial>> prompts = new LinkedHashMap<>();
        java.util.function.ToIntFunction<String> count = text -> tokenizer.encodeWithModelSpecialTokens(text).length;
        for (Scenario scenario : options.scenarios()) {
            List<PromptMaterial> scenarioPrompts = byTarget.computeIfAbsent(
                    scenario.targetPromptTokens(),
                    target -> options.promptCorpus().equals("chat")
                            ? ChatPromptCorpus.build(target, count)
                            : List.of(PromptMaterial.build(target, options.promptSeed(), count)));
            for (PromptMaterial prompt : scenarioPrompts) {
                long required = (long) prompt.actualTokens() + scenario.requestedNewTokens();
                if (required > context)
                    throw new IllegalArgumentException(
                            scenario.name() + " needs " + required + " positions; context is " + context + " (model "
                                    + modelContext + ", maxContextTokens " + options.maxContextTokens() + ")");
            }
            prompts.put(scenario, scenarioPrompts);
        }
        return new Prepared(workers, prompts, modelContext);
    }

    /// `all`, `one-per-core`, `performance`, `performance-one-per-core`, or explicit processor IDs
    /// and ranges; then processor and core exclusions. Every step rejects unavailable or empty sets.
    public static WorkerProcessorSelection resolveWorkers(
            String spec, List<Integer> excludeCpus, List<Integer> excludeCores, ProcessorTopology topology) {
        WorkerProcessorSelection selection =
                switch (spec) {
                    case "all" -> topology.allProcessors();
                    case "one-per-core" -> topology.onePerPhysicalCore();
                    case "performance" -> topology.performanceCoreProcessors();
                    case "performance-one-per-core" ->
                        topology.performanceCoreProcessors().onePerCore();
                    default -> {
                        List<Integer> ids = ids(spec, "cpus");
                        if (ids.isEmpty()) throw new IllegalArgumentException("cpus selects no processors");
                        yield topology.processors(
                                ids.stream().mapToInt(Integer::intValue).toArray());
                    }
                };
        if (!excludeCpus.isEmpty())
            selection = selection.excludingProcessors(
                    excludeCpus.stream().mapToInt(Integer::intValue).toArray());
        if (!excludeCores.isEmpty())
            selection = selection.excludingCores(
                    excludeCores.stream().mapToInt(Integer::intValue).toArray());
        return selection;
    }

    private static void requireFile(Path path, String option) {
        if (!Files.isRegularFile(path) || !Files.isReadable(path))
            throw new IllegalArgumentException("--" + option + " is not a readable file: " + path);
    }

    /// Parses comma-separated non-negative IDs and inclusive ranges such as `2,3,8-11`, in order.
    static List<Integer> ids(String text, String option) {
        List<Integer> ids = new ArrayList<>();
        for (String part : text.split(",")) {
            String item = part.strip();
            if (item.isEmpty()) continue;
            try {
                int dash = item.indexOf('-', 1);
                int first = Integer.parseInt((dash < 0 ? item : item.substring(0, dash)).strip());
                int last = dash < 0
                        ? first
                        : Integer.parseInt(item.substring(dash + 1).strip());
                if (first < 0 || last < first) throw new NumberFormatException();
                for (int id = first; id <= last; id++) ids.add(id);
            } catch (NumberFormatException invalid) {
                throw new IllegalArgumentException(option + " has an invalid value or range: " + item);
            }
        }
        return ids;
    }
}
