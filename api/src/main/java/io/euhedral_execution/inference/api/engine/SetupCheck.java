package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.core.nativefiles.NativeFiles;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/// Checks what the server needs before anything loads, and reports every problem at once, each with the setting that
/// names it: the model files the settings point at, the NVIDIA driver and a GPU, and a native cache the jar's native
/// files unpack into. Without it a wrong path surfaces as the first exception of the engine's load, after the weights
/// were read.
final class SetupCheck {
    private SetupCheck() {}

    /// Checks the installation and unpacks the jar's native files into the native cache (`native-cache`, else the
    /// user's cache directory).
    static NativeFiles require(InferenceProperties properties) {
        List<String> problems = problems(properties, DriverProbe.system());
        NativeFiles files = null;
        try {
            files = NativeFiles.unpack(
                    properties.nativeCache() != null ? properties.nativeCache() : NativeFiles.defaultCache());
        } catch (NativeFiles.CacheException | IllegalStateException | UncheckedIOException failure) {
            problems.add(failure.getMessage());
        }
        if (!problems.isEmpty()) throw new SetupException(problems);
        return files;
    }

    static List<String> problems(InferenceProperties properties, DriverProbe driver) {
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

        driver.problem()
                .ifPresent(problem -> problems.add("the NVIDIA driver is required, with an NVIDIA Blackwell GPU: "
                        + problem + "; install the driver (nvidia-smi lists the GPU when it works)"));
        return problems;
    }

    private static String setting(String name) {
        return "euhedral.inference." + name + " (EUHEDRAL_INFERENCE_"
                + name.toUpperCase(Locale.ROOT).replace('-', '_') + ")";
    }

    /// The settings point at files that are missing or unusable, the driver is missing, or the native cache is
    /// unwritable.
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
