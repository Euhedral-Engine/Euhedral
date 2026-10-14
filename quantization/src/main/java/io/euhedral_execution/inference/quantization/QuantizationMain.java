package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.quantization.gptq.Gptq;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// Command line of the quantization tools:
///
/// ```
/// calibrate --model CHECKPOINT --quantization nvfp4|q3 [--compressed] [--output-head FORMAT] --tokens TOKENS.npy
/// --draft-ids-from ARTIFACT
/// --out CODES_DIR [--act-order] [--damp 0.01] [--batch 8] [--sequences N] [--layers N] [--skip-mtp]
/// [--nvfp4-above 2] [--nvfp4-scale-max 448] [--library LIBEUHEDRAL_CUDA]
/// train-blocks --model CHECKPOINT --codes CODES_DIR --tokens TOKENS.npy --out CODES_DIR --train-layers 10,11|all
/// [--overlay CHECKPOINT_DIR] [--sequences 96]
/// [--holdout 16] [--epochs 2] [--scale-lr 1e-5] [--norm-lr 1e-4] [--batch 4] [--library LIBEUHEDRAL_CUDA]
/// train-model --model CHECKPOINT --codes CODES_DIR [--norms NORMS_DIR] --tokens TOKENS.npy --out CODES_DIR
/// [--overlay CHECKPOINT_DIR] [--sequences 218] [--holdout 18] [--epochs 1] [--scale-lr 2e-6]
/// [--norm-lr 2e-5] [--top-k 64] [--evaluate-every 50] [--probe 0] [--accumulate 8] [--fp32] [--batch 4]
/// [--library
/// LIBEUHEDRAL_CUDA]
/// score --artifact ARTIFACT --results RESULTS.txt (--reference REFERENCE.kld --name NAME)... [--rows 8]
/// [--host-mib N] [--batch 4] [--workers 6] [--library LIBEUHEDRAL_CUDA]
/// generate --artifact ARTIFACT --tokenizer CHECKPOINT --out CORPUS_DIR (--prompts PROMPTS.jsonl)... [--max-tokens
/// 4096]
/// [--batch 12] [--host-mib N] [--limit N] [--workers 6] [--library LIBEUHEDRAL_CUDA]
/// rotate --model CHECKPOINT --out CHECKPOINT_DIR [--seed 1] [--library LIBEUHEDRAL_CUDA]
/// ```
public final class QuantizationMain {

    private QuantizationMain() {}

    public static void main(String[] args) throws Exception {
        if (args.length == 0
                || !List.of("calibrate", "score", "generate", "train-blocks", "train-model", "rotate")
                        .contains(args[0])) {
            System.err.println(
                    "usage: calibrate --model DIR --quantization nvfp4|q3 [--compressed] [--output-head FORMAT] --tokens T.npy "
                            + "--draft-ids-from ARTIFACT --out DIR [--act-order] [--damp D] [--batch B] [--sequences N] "
                            + "[--layers N] [--skip-mtp] [--library PATH]");
            System.err.println("       score --artifact ARTIFACT --results FILE (--reference R.kld --name NAME)... "
                    + "[--rows 8] [--host-mib N] [--batch 4] [--workers 6] [--library PATH]");
            System.err.println(
                    "       generate --artifact ARTIFACT --tokenizer CHECKPOINT --out DIR (--prompts P.jsonl)... "
                            + "[--max-tokens 4096] [--batch 12] [--host-mib N] [--limit N] [--workers 6] [--library PATH]");
            System.err.println("       rotate --model CHECKPOINT --out DIR [--seed 1] [--library PATH]");
            System.exit(2);
        }
        Map<String, String> options = new HashMap<>();
        List<String> references = new ArrayList<>(), names = new ArrayList<>(), prompts = new ArrayList<>();
        for (int i = 1; i < args.length; i++) {
            String key = args[i];
            if (!key.startsWith("--")) throw new IllegalArgumentException("unexpected " + key);
            boolean flag = key.equals("--compressed")
                    || key.equals("--act-order")
                    || key.equals("--skip-mtp")
                    || key.equals("--fp32");
            String value = flag ? "true" : args[++i];
            if (key.equals("--reference")) references.add(value);
            else if (key.equals("--name")) names.add(value);
            else if (key.equals("--prompts")) prompts.add(value);
            else options.put(key.substring(2), value);
        }
        String library = options.getOrDefault(
                "library",
                System.getProperty("euhedral.cuda.library", "build/native/linux-x64/lib/libeuhedral_cuda.so"));
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
                    Path.of(library));
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
                    Path.of(library));
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
                            Integer.parseInt(options.getOrDefault("workers", "6")),
                            Integer.parseInt(options.getOrDefault("limit", "0")),
                            Path.of(library)));
            return;
        }
        if (args[0].equals("rotate")) {
            Rotate.run(new Rotate.Settings(
                    Path.of(required(options, "model")),
                    Path.of(required(options, "out")),
                    Long.parseLong(options.getOrDefault("seed", "1")),
                    Path.of(library)));
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
                    Integer.parseInt(options.getOrDefault("rows", "8")),
                    hostBytes(options),
                    Integer.parseInt(options.getOrDefault("batch", "4")),
                    Integer.parseInt(options.getOrDefault("workers", "6")),
                    Path.of(library)));
            return;
        }
        var settings = new Calibrate.Settings(
                Path.of(required(options, "model")),
                new Recipe(
                        required(options, "quantization"),
                        options.containsKey("compressed"),
                        options.get("output-head")),
                Path.of(required(options, "tokens")),
                Path.of(required(options, "draft-ids-from")),
                Path.of(required(options, "out")),
                options.containsKey("act-order"),
                Float.parseFloat(options.getOrDefault("damp", "0.01")),
                Integer.parseInt(options.getOrDefault("batch", "8")),
                Integer.parseInt(options.getOrDefault("sequences", "0")),
                Integer.parseInt(options.getOrDefault("layers", "64")),
                options.containsKey("skip-mtp"),
                new Gptq.Nvfp4Search(
                        Integer.parseInt(options.getOrDefault("nvfp4-above", "2")),
                        Float.parseFloat(options.getOrDefault("nvfp4-scale-max", "448"))));
        Calibrate.run(settings, Path.of(library));
    }

    /// `--host-mib` in bytes, or [Residency#PLANNED] when it is not given.
    private static long hostBytes(Map<String, String> options) {
        return options.containsKey("host-mib") ? Long.parseLong(options.get("host-mib")) << 20 : Residency.PLANNED;
    }

    private static String required(Map<String, String> options, String key) {
        String value = options.get(key);
        if (value == null) throw new IllegalArgumentException("--" + key + " is required");
        return value;
    }
}
