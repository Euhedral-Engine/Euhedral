package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.quantization.gptq.Gptq;
import io.euhedral_execution.inference.quantization.gpu.CudaLibraries;
import io.euhedral_execution.inference.quantization.mixed.Allocation;
import io.euhedral_execution.inference.quantization.mixed.Assemble;
import io.euhedral_execution.inference.quantization.mixed.Cost;
import io.euhedral_execution.inference.quantization.mixed.CostTable;
import io.euhedral_execution.inference.quantization.mixed.Feasibility;
import io.euhedral_execution.inference.quantization.mixed.FidelityModel;
import io.euhedral_execution.inference.quantization.mixed.Option;
import io.euhedral_execution.inference.quantization.mixed.Priors;
import io.euhedral_execution.inference.quantization.mixed.Rco;
import io.euhedral_execution.inference.quantization.mixed.Search;
import io.euhedral_execution.inference.quantization.mixed.SearchState;
import io.euhedral_execution.inference.quantization.mixed.Seeds;
import io.euhedral_execution.inference.quantization.mixed.Sources;
import io.euhedral_execution.inference.quantization.mtp.DrafterExport;
import io.euhedral_execution.inference.quantization.mtp.DrafterTraining;
import io.euhedral_execution.inference.quantization.trellis.TrellisQuantizer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;

/// Command line of the quantization tools:
///
/// ```
/// calibrate --model CHECKPOINT --quantization nvfp4|q3 [--compressed] [--output-head FORMAT] --tokens TOKENS.npy
/// --draft-ids-from ARTIFACT
/// --out CODES_DIR [--act-order] [--damp 0.01] [--batch 8] [--sequences N] [--layers N] [--skip-mtp]
/// [--nvfp4-above 2] [--nvfp4-scale-max 448] [--library LIBEUHEDRAL_CUDA]
/// calibrate --model CHECKPOINT --format trellis --bits 2.5|3.0|3.5 --tokens TOKENS.npy --draft-ids-from ARTIFACT
/// --out CHECKPOINT_DIR [--output-head FORMAT]
/// [--batch 8] [--sequences N] [--layers N] [--rounding-lanes N] [--viterbi-scratch-mib 256]
/// [--memory-reserve-mib 1024] [--write-buffer-mib 16] [--library LIBEUHEDRAL_CUDA]
/// train-blocks --model CHECKPOINT --codes CODES_DIR --tokens TOKENS.npy --out CODES_DIR --train-layers 10,11|all
/// [--overlay CHECKPOINT_DIR] [--sequences 96]
/// [--holdout 16] [--epochs 2] [--scale-lr 1e-5] [--norm-lr 1e-4] [--batch 4] [--library LIBEUHEDRAL_CUDA]
/// train-model --model CHECKPOINT --codes CODES_DIR [--norms NORMS_DIR] --tokens TOKENS.npy --out CODES_DIR
/// [--overlay CHECKPOINT_DIR] [--sequences 218] [--holdout 18] [--epochs 1] [--scale-lr 2e-6]
/// [--norm-lr 2e-5] [--top-k 64] [--evaluate-every 50] [--probe 0] [--accumulate 8] [--fp32] [--batch 4]
/// [--library
/// LIBEUHEDRAL_CUDA]
/// score --artifact ARTIFACT --results RESULTS.txt (--reference REFERENCE.kld --name NAME)... [--values DIR]
/// [--rows 8] [--host-mib N] [--batch 4] [--workers N] [--library LIBEUHEDRAL_CUDA]
/// generate --artifact ARTIFACT --tokenizer CHECKPOINT --out CORPUS_DIR (--prompts PROMPTS.jsonl)... [--max-tokens
/// 4096]
/// [--batch 12] [--greedy-fraction 0.25] [--host-mib N] [--limit N] [--workers N] [--library LIBEUHEDRAL_CUDA]
/// assemble --allocation FILE --source OPTION=ARTIFACT... --base OPTION --out ARTIFACT [--draft-depth D]
/// [--copy-buffer-mib 64]
/// allocate --source OPTION=ARTIFACT... --base OPTION --costs COSTS.json --priors PRIORS.json
/// --reference R.kld... --context 32768 --state SEARCH.resume --out DIR [--seed-file F] [--import-state OLD.resume]
/// [--batch 8]
/// [--explore 0.3] [--patience 3] [--random 16] [--bands 8] [--verify-rows 5] [--workers N] [--host-mib N]
/// [--free-mib N] [--score-rows 8] [--score-batch 0 (the chunks the device holds)] [--members 8] [--ridge 1e-3]
/// [--lambdas 0,0.002,0.005,0.01,0.02]
/// [--budget-fractions 1.0,0.8] [--rco-steps 200] [--rco-lr 0.1] [--rco-samples 8] [--rco-restarts 4]
/// [--knapsack-steps 512] [--max-rounds N] [--seed 1] [--host-sinks N] [--max-concurrent 0 (as many as memory
/// holds)] [--concurrent-margin-mib 512] [--library LIBEUHEDRAL_CUDA]
/// cost --source OPTION=ARTIFACT... --base OPTION --out COSTS.json [--rows 1,2,3,4,5,6,512] [--repeats 50]
/// [--rotate-mib 256] [--library LIBEUHEDRAL_CUDA]
/// reference --artifact ARTIFACT --tokenizer CHECKPOINT --text FILE --out OUT.kld [--context 2560] [--chunks N]
/// [--parse-special] [--rows 8] [--batch 4] [--host-mib N] [--workers N] [--library LIBEUHEDRAL_CUDA]
/// rotate --model CHECKPOINT --out CHECKPOINT_DIR [--seed 1] [--library LIBEUHEDRAL_CUDA]
/// train-drafter --corpus CORPUS_DIR --model CHECKPOINT --artifact ARTIFACT --out DIR [--depths 4] [--rows 1024]
/// [--keys 4096] [--tokens 16384] [--epochs 1] [--projection-rate 2e-5] [--norm-rate 2e-5] [--warmup 10]
/// [--holdout 50] [--evaluate-every 50] [--limit N] [--library LIBEUHEDRAL_CUDA]
/// export-drafter --model CHECKPOINT --artifact BASE_ARTIFACT --out ARTIFACT [--parameters PARAMETERS.bin]
/// package-tokenizer --artifact ARTIFACT --tokenizer CHECKPOINT [--out ARTIFACT]
/// ```
///
/// Every command that runs on the GPU takes `--library` (the engine's CUDA library, by default the build's
/// `build/native/linux-x64/lib/libeuhedral_cuda.so`) and `--cuda-libraries DIR`, the directory of cuBLAS and
/// cuSOLVER (by default the library search path); the Gradle `run` task passes the module's download.
public final class QuantizationMain {

    private QuantizationMain() {}

    /// Decoder layers of Swift-1.5, the default of `--layers`.
    private static final int TEXT_LAYERS = 64;

    /// Calibrate flags that only the GPTQ path reads; a trellis run rejects them.
    private static final List<String> GPTQ_ONLY_FLAGS =
            List.of("skip-mtp", "act-order", "damp", "quantization", "compressed", "nvfp4-above", "nvfp4-scale-max");

    public static void main(String[] args) throws Exception {
        if (args.length == 0
                || !List.of(
                                "calibrate",
                                "score",
                                "generate",
                                "train-drafter",
                                "export-drafter",
                                "train-blocks",
                                "train-model",
                                "rotate",
                                "assemble",
                                "cost",
                                "allocate",
                                "reference",
                                "package-tokenizer")
                        .contains(args[0])) {
            System.err.println(
                    "usage: calibrate --model DIR --quantization nvfp4|q3 [--compressed] [--output-head FORMAT] --tokens T.npy "
                            + "--draft-ids-from ARTIFACT --out DIR [--act-order] [--damp D] [--batch B] [--sequences N] "
                            + "[--layers N] [--skip-mtp] [--library PATH]");
            System.err.println("       calibrate --model DIR --format trellis --bits 2.5|3.0|3.5 --tokens T.npy "
                    + "--draft-ids-from ARTIFACT --out DIR [--output-head FORMAT] [--batch B] [--sequences N] [--layers N] "
                    + "[--rounding-lanes N] [--viterbi-scratch-mib 256] "
                    + "[--memory-reserve-mib 1024] [--write-buffer-mib 16] [--library PATH]");
            System.err.println("       score --artifact ARTIFACT --results FILE (--reference R.kld --name NAME)... "
                    + "[--values DIR] [--rows 8] [--host-mib N] [--batch 4] [--workers N] [--library PATH]");
            System.err.println(
                    "       generate --artifact ARTIFACT --tokenizer CHECKPOINT --out DIR (--prompts P.jsonl)... "
                            + "[--max-tokens 4096] [--batch 12] [--greedy-fraction 0.25] [--host-mib N] [--limit N] [--workers N] [--library PATH]");
            System.err.println("       reference --artifact ARTIFACT --tokenizer CHECKPOINT --text FILE --out OUT.kld "
                    + "[--context 2560] [--chunks N] [--parse-special] [--rows 8] [--batch 4] "
                    + "[--host-mib N] [--workers N] [--library PATH]");
            System.err.println("       rotate --model CHECKPOINT --out DIR [--seed 1] [--library PATH]");
            System.err.println(
                    "       export-drafter --model CHECKPOINT --artifact BASE --out ARTIFACT [--parameters FILE]");
            System.err.println("       train-drafter --corpus DIR --model CHECKPOINT --artifact ARTIFACT --out DIR "
                    + "[--depths 4] [--rows 1024] [--keys 4096] [--tokens 16384] [--epochs 1] [--projection-rate 2e-5] [--norm-rate 2e-5] "
                    + "[--warmup 10] [--holdout 50] [--evaluate-every 50] [--limit N] [--library PATH]");
            System.err.println(
                    "       assemble --allocation FILE --source OPTION=ARTIFACT... --base OPTION --out ARTIFACT "
                            + "[--draft-depth D] [--copy-buffer-mib 64]");
            System.err.println("       allocate --source OPTION=ARTIFACT... --base OPTION --costs COSTS.json "
                    + "--priors PRIORS.json --reference R.kld... --context 32768 --state SEARCH.resume --out DIR "
                    + "[--seed-file F] [--import-state OLD.resume] [--batch 8] [--explore 0.3] [--patience 3] [--random 16] [--bands 8] "
                    + "[--verify-rows 5] [--workers N] [--library PATH]");
            System.err.println("       cost --source OPTION=ARTIFACT... --base OPTION --out COSTS.json "
                    + "[--rows 1,2,3,4,5,6,512] [--repeats 50] [--rotate-mib 256] [--library PATH]");
            System.err.println("       package-tokenizer --artifact ARTIFACT --tokenizer CHECKPOINT [--out ARTIFACT]");
            System.exit(2);
        }
        Map<String, String> options = new HashMap<>();
        List<String> references = new ArrayList<>(),
                names = new ArrayList<>(),
                prompts = new ArrayList<>(),
                artifacts = new ArrayList<>();
        for (int i = 1; i < args.length; i++) {
            String key = args[i];
            if (!key.startsWith("--")) throw new IllegalArgumentException("unexpected " + key);
            boolean flag = key.equals("--compressed")
                    || key.equals("--act-order")
                    || key.equals("--skip-mtp")
                    || key.equals("--fp32")
                    || key.equals("--parse-special");
            String value = flag ? "true" : args[++i];
            if (key.equals("--reference")) references.add(value);
            else if (key.equals("--name")) names.add(value);
            else if (key.equals("--prompts")) prompts.add(value);
            else if (key.equals("--source")) artifacts.add(value);
            else options.put(key.substring(2), value);
        }
        if (args[0].equals("package-tokenizer")) {
            PackageTokenizer.run(
                    Path.of(required(options, "artifact")),
                    Path.of(required(options, "tokenizer")),
                    options.containsKey("out") ? Path.of(options.get("out")) : null);
            return;
        }
        CudaLibraries library = new CudaLibraries(
                Path.of(options.getOrDefault("library", "build/native/linux-x64/lib/libeuhedral_cuda.so")),
                Optional.ofNullable(options.get("cuda-libraries")).map(Path::of));
        if (args[0].equals("allocate")) {
            allocate(options, references, artifacts, library);
            return;
        }
        if (args[0].equals("assemble") || args[0].equals("cost")) {
            Map<Option, Path> sources = sourcesOf(artifacts);
            if (args[0].equals("cost")) {
                List<Integer> rows = new ArrayList<>();
                for (String count :
                        options.getOrDefault("rows", "1,2,3,4,5,6,512").split(","))
                    rows.add(Integer.parseInt(count.trim()));
                Cost.run(
                        library,
                        Sources.open(sources, Option.of(required(options, "base"))),
                        rows,
                        Path.of(required(options, "out")),
                        Integer.parseInt(options.getOrDefault("repeats", "50")),
                        Long.parseLong(options.getOrDefault("rotate-mib", "256")) << 20);
                return;
            }
            Assemble.write(
                    Sources.open(sources, Option.of(required(options, "base"))),
                    Allocation.read(Path.of(required(options, "allocation"))),
                    options.containsKey("draft-depth")
                            ? OptionalInt.of(Integer.parseInt(options.get("draft-depth")))
                            : OptionalInt.empty(),
                    Path.of(required(options, "out")),
                    Long.parseLong(options.getOrDefault(
                                    "copy-buffer-mib", Long.toString(Assemble.DEFAULT_COPY_BUFFER_MIB)))
                            << 20);
            return;
        }
        if (args[0].equals("train-model")) {
            ModelTraining.run(
                    new ModelTraining.Settings(
                            Path.of(required(options, "model")),
                            Path.of(required(options, "codes")),
                            options.containsKey("norms") ? Path.of(options.get("norms")) : null,
                            Path.of(required(options, "tokens")),
                            Path.of(required(options, "out")),
                            options.containsKey("overlay") ? Path.of(options.get("overlay")) : null,
                            Integer.parseInt(options.getOrDefault("sequences", "218")),
                            Integer.parseInt(options.getOrDefault("holdout", "18")),
                            Integer.parseInt(options.getOrDefault("epochs", "1")),
                            Float.parseFloat(options.getOrDefault("scale-lr", "2e-6")),
                            Float.parseFloat(options.getOrDefault("norm-lr", "2e-5")),
                            Integer.parseInt(options.getOrDefault("top-k", "64")),
                            Integer.parseInt(options.getOrDefault("evaluate-every", "50")),
                            Integer.parseInt(options.getOrDefault("probe", "0")),
                            Integer.parseInt(options.getOrDefault("accumulate", "8")),
                            !options.containsKey("fp32"),
                            Integer.parseInt(options.getOrDefault("batch", "4"))),
                    library);
            return;
        }
        if (args[0].equals("train-blocks")) {
            java.util.Set<Integer> layers = new java.util.TreeSet<>();
            String chosen = required(options, "train-layers");
            if (chosen.equals("all")) for (int layer = 0; layer < 64; layer++) layers.add(layer);
            else for (String layer : chosen.split(",")) layers.add(Integer.parseInt(layer.trim()));
            BlockTraining.run(
                    new BlockTraining.Settings(
                            Path.of(required(options, "model")),
                            Path.of(required(options, "codes")),
                            Path.of(required(options, "tokens")),
                            layers,
                            Integer.parseInt(options.getOrDefault("sequences", "96")),
                            Integer.parseInt(options.getOrDefault("holdout", "16")),
                            Integer.parseInt(options.getOrDefault("epochs", "2")),
                            Float.parseFloat(options.getOrDefault("scale-lr", "1e-5")),
                            Float.parseFloat(options.getOrDefault("norm-lr", "1e-4")),
                            Integer.parseInt(options.getOrDefault("batch", "4")),
                            Path.of(required(options, "out")),
                            options.containsKey("overlay") ? Path.of(options.get("overlay")) : null),
                    library);
            return;
        }
        if (args[0].equals("generate")) {
            if (prompts.isEmpty()) throw new IllegalArgumentException("--prompts is required");
            io.euhedral_execution.inference.quantization.mtp.Generate.run(
                    new io.euhedral_execution.inference.quantization.mtp.Generate.Settings(
                            Path.of(required(options, "artifact")),
                            prompts.stream().map(Path::of).toList(),
                            Path.of(required(options, "tokenizer")),
                            Path.of(required(options, "out")),
                            Integer.parseInt(options.getOrDefault("max-tokens", "4096")),
                            Integer.parseInt(options.getOrDefault("batch", "12")),
                            hostBytes(options),
                            workers(options),
                            Integer.parseInt(options.getOrDefault("limit", "0")),
                            Double.parseDouble(options.getOrDefault("greedy-fraction", "0.25")),
                            library));
            return;
        }
        if (args[0].equals("export-drafter")) {
            DrafterExport.run(new DrafterExport.Settings(
                    options.containsKey("parameters") ? Path.of(options.get("parameters")) : null,
                    Path.of(required(options, "model")),
                    Path.of(required(options, "artifact")),
                    Path.of(required(options, "out"))));
            return;
        }
        if (args[0].equals("train-drafter")) {
            DrafterTraining.run(
                    new DrafterTraining.Settings(
                            Path.of(required(options, "corpus")),
                            Path.of(required(options, "model")),
                            Path.of(required(options, "artifact")),
                            Path.of(required(options, "out")),
                            Integer.parseInt(options.getOrDefault("depths", "4")),
                            Integer.parseInt(options.getOrDefault("rows", "1024")),
                            Integer.parseInt(options.getOrDefault("keys", "4096")),
                            Integer.parseInt(options.getOrDefault("tokens", "16384")),
                            Integer.parseInt(options.getOrDefault("epochs", "1")),
                            Float.parseFloat(options.getOrDefault("projection-rate", "2e-5")),
                            Float.parseFloat(options.getOrDefault("norm-rate", "2e-5")),
                            Integer.parseInt(options.getOrDefault("warmup", "10")),
                            Integer.parseInt(options.getOrDefault("holdout", "50")),
                            Integer.parseInt(options.getOrDefault("evaluate-every", "50")),
                            Integer.parseInt(options.getOrDefault("limit", "0"))),
                    library);
            return;
        }
        if (args[0].equals("reference")) {
            ReferenceLogits.run(new ReferenceLogits.Settings(
                    Path.of(required(options, "artifact")),
                    Path.of(required(options, "tokenizer")),
                    Path.of(required(options, "text")),
                    Path.of(required(options, "out")),
                    Integer.parseInt(options.getOrDefault("context", "2560")),
                    Integer.parseInt(options.getOrDefault("chunks", "0")),
                    options.containsKey("parse-special"),
                    Integer.parseInt(options.getOrDefault("rows", "8")),
                    Integer.parseInt(options.getOrDefault("batch", "4")),
                    hostBytes(options),
                    workers(options),
                    library));
            return;
        }
        if (args[0].equals("rotate")) {
            Rotate.run(new Rotate.Settings(
                    Path.of(required(options, "model")),
                    Path.of(required(options, "out")),
                    Long.parseLong(options.getOrDefault("seed", "1")),
                    library));
            return;
        }
        if (args[0].equals("score")) {
            if (references.isEmpty() || references.size() != names.size()) {
                throw new IllegalArgumentException("give each --reference a --name");
            }
            Score.run(new Score.Settings(
                    Path.of(required(options, "artifact")),
                    references.stream().map(Path::of).toList(),
                    names,
                    Path.of(required(options, "results")),
                    options.containsKey("values") ? Path.of(options.get("values")) : null,
                    Integer.parseInt(options.getOrDefault("rows", "8")),
                    hostBytes(options),
                    Integer.parseInt(options.getOrDefault("batch", "4")),
                    workers(options),
                    library));
            return;
        }
        String format = options.getOrDefault("format", "gptq");
        boolean trellis = format.equals("trellis");
        if (!trellis && !format.equals("gptq")) throw new IllegalArgumentException("--format is gptq or trellis");
        if (trellis) {
            for (String flag : GPTQ_ONLY_FLAGS)
                if (options.containsKey(flag))
                    throw new IllegalArgumentException("--" + flag + " applies to --format gptq, not trellis");
        }
        if (!trellis && options.getOrDefault("quantization", "").equals("trellis"))
            throw new IllegalArgumentException("--quantization trellis needs --format trellis");
        // a trellis run over the whole model rounds the heads; a partial one (--layers N) writes layers only
        boolean heads = !trellis || Integer.parseInt(options.getOrDefault("layers", "64")) >= TEXT_LAYERS;
        if (trellis && heads) required(options, "draft-ids-from");
        Double bits = trellis ? Double.valueOf(required(options, "bits")) : null;
        var settings = new Calibrate.Settings(
                Path.of(required(options, "model")),
                new Recipe(
                        trellis ? "trellis" : required(options, "quantization"),
                        options.containsKey("compressed"),
                        options.get("output-head")),
                Path.of(required(options, "tokens")),
                heads ? Path.of(required(options, "draft-ids-from")) : null,
                Path.of(required(options, "out")),
                options.containsKey("act-order"),
                Float.parseFloat(options.getOrDefault("damp", "0.01")),
                Integer.parseInt(options.getOrDefault("batch", "8")),
                Integer.parseInt(options.getOrDefault("sequences", "0")),
                Integer.parseInt(options.getOrDefault("layers", "64")),
                options.containsKey("skip-mtp"),
                new Gptq.Nvfp4Search(
                        Integer.parseInt(options.getOrDefault("nvfp4-above", "2")),
                        Float.parseFloat(options.getOrDefault("nvfp4-scale-max", "448"))),
                bits,
                options.containsKey("rounding-lanes") ? Integer.valueOf(options.get("rounding-lanes")) : null,
                Long.parseLong(options.getOrDefault(
                        "viterbi-scratch-mib", Long.toString(TrellisQuantizer.DEFAULT_VITERBI_SCRATCH_BYTES >> 20))),
                Long.parseLong(options.getOrDefault("memory-reserve-mib", "1024")),
                Long.parseLong(options.getOrDefault("write-buffer-mib", "16")));
        Calibrate.run(settings, library);
    }

    private static Map<Option, Path> sourcesOf(List<String> artifacts) {
        Map<Option, Path> sources = new EnumMap<>(Option.class);
        for (String source : artifacts) {
            int equals = source.indexOf('=');
            if (equals < 1) throw new IllegalArgumentException("--source is OPTION=ARTIFACT, not " + source);
            if (sources.put(Option.of(source.substring(0, equals)), Path.of(source.substring(equals + 1))) != null)
                throw new IllegalArgumentException("two --source for " + source.substring(0, equals));
        }
        return sources;
    }

    private static List<Double> numbers(String list) {
        List<Double> numbers = new ArrayList<>();
        for (String number : list.split(",")) numbers.add(Double.parseDouble(number.trim()));
        return numbers;
    }

    /// The search over allocations: the free device memory is measured once, here, after the device is open.
    private static void allocate(
            Map<String, String> options, List<String> references, List<String> artifacts, CudaLibraries library)
            throws Exception {
        Sources sources = Sources.open(sourcesOf(artifacts), Option.of(required(options, "base")));
        List<Path> referencePaths = references.stream().map(Path::of).toList();
        if (referencePaths.isEmpty()) throw new IllegalArgumentException("--reference is required");
        Path costsFile = Path.of(required(options, "costs")), priorsFile = Path.of(required(options, "priors"));
        int context = Integer.parseInt(options.getOrDefault("context", "32768"));
        int workers = workers(options);
        int scoreRows = Integer.parseInt(options.getOrDefault("score-rows", "8"));
        int scoreBatch = Integer.parseInt(options.getOrDefault("score-batch", "0"));
        long hostBytes = hostBytes(options);
        try (Scorer scorer = Scorer.open(library, workers)) {
            long free = options.containsKey("free-mib")
                    ? Long.parseLong(options.get("free-mib")) << 20
                    : scorer.freeDeviceBytes();
            List<SearchState.Measured> imported = options.containsKey("import-state")
                    ? SearchState.read(Path.of(options.get("import-state")))
                    : List.of();
            var settings = new Search.Settings(
                    sources,
                    Priors.read(priorsFile),
                    CostTable.read(costsFile),
                    context,
                    Path.of(required(options, "state")),
                    Path.of(required(options, "out")),
                    options.containsKey("seed-file") ? Seeds.read(Path.of(options.get("seed-file"))) : List.of(),
                    Integer.parseInt(options.getOrDefault("batch", "8")),
                    Double.parseDouble(options.getOrDefault("explore", "0.3")),
                    Integer.parseInt(options.getOrDefault("patience", "3")),
                    Integer.parseInt(options.getOrDefault("random", "16")),
                    Integer.parseInt(options.getOrDefault("bands", Integer.toString(FidelityModel.DEFAULT_BANDS))),
                    Integer.parseInt(options.getOrDefault("verify-rows", "5")),
                    free,
                    freeBytes -> Feasibility.planner(
                            sources,
                            freeBytes,
                            context,
                            Integer.parseInt(options.getOrDefault("verify-rows", "5")) - 1),
                    Integer.parseInt(options.getOrDefault("host-sinks", Integer.toString(workers))),
                    Integer.parseInt(options.getOrDefault("members", "8")),
                    Double.parseDouble(options.getOrDefault("ridge", "1e-3")),
                    numbers(options.getOrDefault("lambdas", "0,0.002,0.005,0.01,0.02")),
                    numbers(options.getOrDefault("budget-fractions", "1.0,0.8")),
                    new Rco.Settings(
                            Integer.parseInt(options.getOrDefault("rco-steps", Integer.toString(Rco.DEFAULT_STEPS))),
                            Double.parseDouble(options.getOrDefault("rco-lr", "0.1")),
                            Integer.parseInt(options.getOrDefault("rco-samples", "8")),
                            Integer.parseInt(options.getOrDefault("rco-restarts", "4"))),
                    Integer.parseInt(options.getOrDefault("knapsack-steps", "512")),
                    Integer.parseInt(options.getOrDefault("max-rounds", Integer.toString(Integer.MAX_VALUE))),
                    Long.parseLong(options.getOrDefault("seed", "1")),
                    // Priors only steer proposals and the scoring batch only how many chunks run together (row-exact
                    // verification gives the same bits at any batch), so a search may change either and resume.
                    String.join("\n", references) + "\n" + costsFile);
            int maxConcurrent = Integer.parseInt(options.getOrDefault("max-concurrent", "0"));
            long marginBytes = Long.parseLong(options.getOrDefault("concurrent-margin-mib", "512")) << 20;
            // Candidates share one cache of device-resident objects and never host-back one beside another, so an
            // explicit --host-mib, or a limit of one, scores them one at a time as before.
            if (hostBytes != Residency.PLANNED || maxConcurrent == 1) {
                Search.run(
                        settings,
                        (path, artifact) ->
                                scorer.score(path, artifact, referencePaths, scoreRows, scoreBatch, hostBytes),
                        imported);
                return;
            }
            try (ConcurrentScoring scoring = new ConcurrentScoring(
                    scorer,
                    referencePaths,
                    scoreRows,
                    scoreBatch,
                    maxConcurrent,
                    marginBytes,
                    Integer.parseInt(options.getOrDefault("host-sinks", Integer.toString(workers))))) {
                Search.run(settings, scoring, imported);
            }
        }
    }

    /// `--host-mib` in bytes, or [Residency#PLANNED] when it is not given.
    private static long hostBytes(Map<String, String> options) {
        return options.containsKey("host-mib") ? Long.parseLong(options.get("host-mib")) << 20 : Residency.PLANNED;
    }

    /// `--workers`, by default every CPU the process may run on.
    private static int workers(Map<String, String> options) {
        return options.containsKey("workers")
                ? Integer.parseInt(options.get("workers"))
                : CalibrationLattice.allowedCpuCount();
    }

    private static String required(Map<String, String> options, String key) {
        String value = options.get(key);
        if (value == null) throw new IllegalArgumentException("--" + key + " is required");
        return value;
    }
}
