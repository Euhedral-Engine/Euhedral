package io.euhedral_execution.inference.quantization;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// Command line of the quantization tools:
///
/// ```
/// calibrate --model CHECKPOINT --quantization nvfp4|q3 [--compressed] --tokens TOKENS.npy --draft-ids-from ARTIFACT
///           --out CODES_DIR [--act-order] [--damp 0.01] [--batch 8] [--sequences N] [--layers N] [--skip-mtp]
///           [--library LIBEUHEDRAL_CUDA]
/// score --artifact ARTIFACT --results RESULTS.txt (--reference REFERENCE.kld --name NAME)... [--rows 8]
///       [--host-mib 2304] [--concurrency 3] [--workers 6] [--library LIBEUHEDRAL_CUDA]
/// ```
public final class QuantizationMain {

    private QuantizationMain() {}

    public static void main(String[] args) throws Exception {
        if (args.length == 0 || !(args[0].equals("calibrate") || args[0].equals("score"))) {
            System.err.println("usage: calibrate --model DIR --quantization nvfp4|q3 [--compressed] --tokens T.npy "
                    + "--draft-ids-from ARTIFACT --out DIR [--act-order] [--damp D] [--batch B] [--sequences N] "
                    + "[--layers N] [--skip-mtp] [--library PATH]");
            System.err.println("       score --artifact ARTIFACT --results FILE (--reference R.kld --name NAME)... "
                    + "[--rows 8] [--host-mib 2304] [--concurrency 3] [--workers 6] [--library PATH]");
            System.exit(2);
        }
        Map<String, String> options = new HashMap<>();
        List<String> references = new ArrayList<>(), names = new ArrayList<>();
        for (int i = 1; i < args.length; i++) {
            String key = args[i];
            if (!key.startsWith("--")) throw new IllegalArgumentException("unexpected " + key);
            boolean flag = key.equals("--compressed") || key.equals("--act-order") || key.equals("--skip-mtp");
            String value = flag ? "true" : args[++i];
            if (key.equals("--reference")) references.add(value);
            else if (key.equals("--name")) names.add(value);
            else options.put(key.substring(2), value);
        }
        String library = options.getOrDefault(
                "library",
                System.getProperty("euhedral.cuda.library", "build/native/linux-x64/lib/libeuhedral_cuda.so"));
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
                    Long.parseLong(options.getOrDefault("host-mib", "2304")) << 20,
                    Integer.parseInt(options.getOrDefault("concurrency", "3")),
                    Integer.parseInt(options.getOrDefault("workers", "6")),
                    Path.of(library)));
            return;
        }
        var settings = new Calibrate.Settings(
                Path.of(required(options, "model")),
                new Recipe(required(options, "quantization"), options.containsKey("compressed")),
                Path.of(required(options, "tokens")),
                Path.of(required(options, "draft-ids-from")),
                Path.of(required(options, "out")),
                options.containsKey("act-order"),
                Float.parseFloat(options.getOrDefault("damp", "0.01")),
                Integer.parseInt(options.getOrDefault("batch", "8")),
                Integer.parseInt(options.getOrDefault("sequences", "0")),
                Integer.parseInt(options.getOrDefault("layers", "64")),
                options.containsKey("skip-mtp"));
        Calibrate.run(settings, Path.of(library));
    }

    private static String required(Map<String, String> options, String key) {
        String value = options.get(key);
        if (value == null) throw new IllegalArgumentException("--" + key + " is required");
        return value;
    }
}
