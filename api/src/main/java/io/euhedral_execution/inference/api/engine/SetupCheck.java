package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.core.settings.Settings;
import io.euhedral_execution.inference.core.settings.SettingsLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/// Checks what the server needs before anything loads, and reports every problem at once, each with the setting that
/// names it: the model files the settings point at, and the NVIDIA driver with a GPU. Without it a wrong path surfaces
/// as the first exception of the engine's load, after the weights
/// were read.
final class SetupCheck {
    private SetupCheck() {}

    /// Checks the installation.
    ///
    /// @throws SetupException when anything is missing or unusable
    static Checked require(Settings settings) {
        List<String> problems = problems(settings, DriverProbe.system());
        if (!problems.isEmpty()) throw new SetupException(problems);
        return new Checked();
    }

    /// Proof that [#require] passed; the beans that read the model's files depend on it.
    record Checked() {}

    static List<String> problems(Settings settings, DriverProbe driver) {
        List<String> problems = new ArrayList<>();
        Path artifact = settings.model().artifact();
        if (!Files.isRegularFile(artifact)) problems.add(setting("model.artifact") + ": no file at " + artifact);
        else if (!Files.isReadable(artifact))
            problems.add(setting("model.artifact") + ": " + artifact + " is not readable");

        Path tokenizer = settings.model().tokenizer();
        if (!Files.isDirectory(tokenizer)) problems.add(setting("model.tokenizer") + ": no directory at " + tokenizer);
        else {
            List<String> missing = new ArrayList<>();
            for (String file : List.of("tokenizer.json", "tokenizer_config.json", "generation_config.json"))
                if (!Files.isRegularFile(tokenizer.resolve(file))) missing.add(file);
            if (!missing.isEmpty())
                problems.add(setting("model.tokenizer") + ": " + tokenizer + " has no " + String.join(", ", missing)
                        + "; it is the checkpoint's directory, with its tokenizer files");
        }

        driver.problem()
                .ifPresent(problem -> problems.add("the NVIDIA driver is required, with an NVIDIA Blackwell GPU: "
                        + problem + "; install the driver (nvidia-smi lists the GPU when it works)"));
        return problems;
    }

    private static String setting(String name) {
        return name + " (" + SettingsLoader.variableName(name) + ")";
    }

    /// The settings point at files that are missing or unusable, or the driver is missing.
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
