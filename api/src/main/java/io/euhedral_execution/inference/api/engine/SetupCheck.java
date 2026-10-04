package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.core.guidance.Llguidance;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/// Checks the files the settings point at before anything loads, and reports every problem at once, each with the
/// setting that names it. Without it a wrong path surfaces as the first exception of the engine's load, after the
/// weights were read, or as a symptom: kernel sources that cannot be found read as a GPU without the kernels'
/// instructions.
final class SetupCheck {
    /// The newest header the kernels include; present from CUDA 12.8.
    private static final String NEWEST_HEADER = "cuda_fp4.h";

    private SetupCheck() {}

    static void require(InferenceProperties properties) {
        List<String> problems = problems(properties, System.getenv());
        if (!problems.isEmpty()) throw new SetupException(problems);
    }

    static List<String> problems(InferenceProperties properties, Map<String, String> environment) {
        List<String> problems = new ArrayList<>();
        Path artifact = properties.artifactPath();
        if (!Files.isRegularFile(artifact)) problems.add(setting("artifact-path") + ": no file at " + artifact);
        else if (!Files.isReadable(artifact))
            problems.add(setting("artifact-path") + ": " + artifact + " is not readable");

        Path tokenizer = properties.tokenizerDirectory();
        if (!Files.isDirectory(tokenizer))
            problems.add(setting("tokenizer-directory") + ": no directory at " + tokenizer);
        else {
            List<String> missing = new ArrayList<>();
            for (String file : List.of("tokenizer.json", "tokenizer_config.json", "generation_config.json"))
                if (!Files.isRegularFile(tokenizer.resolve(file))) missing.add(file);
            if (!missing.isEmpty())
                problems.add(setting("tokenizer-directory") + ": " + tokenizer + " has no " + String.join(", ", missing)
                        + "; it is the checkpoint's directory, with its tokenizer files");
        }

        Path library = properties.cudaLibraryPath();
        if (!Files.isRegularFile(library)) {
            problems.add(setting("cuda-library-path") + ": no file at " + library);
        } else {
            Path kernels =
                    library.toAbsolutePath().getParent().resolveSibling("share").resolve("euhedral_cuda");
            if (!Files.isDirectory(kernels))
                problems.add(setting("cuda-library-path") + ": the kernel sources belong in " + kernels
                        + ", beside the library's directory, and are missing");
            Path guidance = Llguidance.besideLibrary(library);
            if (!Files.isRegularFile(guidance))
                problems.add(setting("cuda-library-path") + ": the constrained-decoding library belongs at " + guidance
                        + ", beside the CUDA library, and is missing");
        }

        Path headers = cudaHeaders(environment);
        if (!Files.isRegularFile(headers.resolve(NEWEST_HEADER)))
            problems.add("EUHEDRAL_CUDA_INCLUDE_DIR: the kernels compile at startup against the CUDA headers, and "
                    + headers.resolve(NEWEST_HEADER) + " is missing; point it at a CUDA 13 include directory");
        return problems;
    }

    /// Where the native library looks for the CUDA headers: `EUHEDRAL_CUDA_INCLUDE_DIR`, else the `include`
    /// directory of `CUDA_HOME` or `CUDA_PATH`, else `/usr/local/cuda/include`.
    private static Path cudaHeaders(Map<String, String> environment) {
        String explicit = environment.get("EUHEDRAL_CUDA_INCLUDE_DIR");
        if (explicit != null && !explicit.isEmpty()) return Path.of(explicit);
        for (String home : List.of("CUDA_HOME", "CUDA_PATH")) {
            String value = environment.get(home);
            if (value != null && !value.isEmpty()) return Path.of(value, "include");
        }
        return Path.of("/usr/local/cuda/include");
    }

    private static String setting(String name) {
        return "euhedral.inference." + name + " (EUHEDRAL_INFERENCE_"
                + name.toUpperCase(java.util.Locale.ROOT).replace('-', '_') + ")";
    }

    /// The settings point at files that are missing or unusable.
    static final class SetupException extends RuntimeException {
        private final List<String> problems;

        SetupException(List<String> problems) {
            super(String.join("; ", problems));
            this.problems = List.copyOf(problems);
        }

        List<String> problems() {
            return this.problems;
        }
    }
}
