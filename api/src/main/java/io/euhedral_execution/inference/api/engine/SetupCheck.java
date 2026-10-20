package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.core.artifact.TokenizerSection;
import io.euhedral_execution.inference.core.settings.Settings;
import io.euhedral_execution.inference.core.settings.SettingsLoader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/// Checks what the server needs before anything loads, and reports every problem at once, each with the setting that
/// names it: the artifact the settings point at (with the tokenizer it carries), and the NVIDIA driver with a GPU.
/// Without it a wrong path surfaces
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

    /// Proof that [#require] passed; the beans that read the model's files take it so the check runs before them.
    record Checked() {}

    static List<String> problems(Settings settings, DriverProbe driver) {
        List<String> problems = new ArrayList<>();
        Path artifact = settings.model().artifact();
        if (!Files.isRegularFile(artifact)) problems.add(setting("model.artifact") + ": no file at " + artifact);
        else if (!Files.isReadable(artifact))
            problems.add(setting("model.artifact") + ": " + artifact + " is not readable");
        else {
            try {
                TokenizerSection.require(artifact);
            } catch (IOException missing) {
                problems.add(setting("model.artifact") + ": " + missing.getMessage());
            }
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
