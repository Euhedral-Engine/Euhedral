package io.euhedral_execution.inference.quantization;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/// Command line of the quantization tools:
///
/// ```
/// calibrate --model CHECKPOINT --quantization nvfp4|q3 [--compressed] --tokens TOKENS.npy --draft-ids-from ARTIFACT
///           --out CODES_DIR [--act-order] [--damp 0.01] [--batch 8] [--sequences N] [--layers N] [--skip-mtp]
///           [--library LIBEUHEDRAL_CUDA]
/// ```
public final class QuantizationMain {

    private QuantizationMain() {}

    public static void main(String[] args) throws Exception {
        if (args.length == 0 || !args[0].equals("calibrate")) {
            System.err.println("usage: calibrate --model DIR --quantization nvfp4|q3 [--compressed] --tokens T.npy "
                    + "--draft-ids-from ARTIFACT --out DIR [--act-order] [--damp D] [--batch B] [--sequences N] "
                    + "[--layers N] [--skip-mtp] [--library PATH]");
            System.exit(2);
        }
        Map<String, String> options = new HashMap<>();
        for (int i = 1; i < args.length; i++) {
            String key = args[i];
            if (!key.startsWith("--")) throw new IllegalArgumentException("unexpected " + key);
            boolean flag = key.equals("--compressed") || key.equals("--act-order") || key.equals("--skip-mtp");
            options.put(key.substring(2), flag ? "true" : args[++i]);
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
        String library = options.getOrDefault(
                "library",
                System.getProperty("euhedral.cuda.library", "build/native/linux-x64/lib/libeuhedral_cuda.so"));
        Calibrate.run(settings, Path.of(library));
    }

    private static String required(Map<String, String> options, String key) {
        String value = options.get(key);
        if (value == null) throw new IllegalArgumentException("--" + key + " is required");
        return value;
    }
}
