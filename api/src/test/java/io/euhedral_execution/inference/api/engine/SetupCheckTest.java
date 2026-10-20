package io.euhedral_execution.inference.api.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TokenizerSection;
import io.euhedral_execution.inference.core.gpu.GpuMemoryException;
import io.euhedral_execution.inference.core.settings.Settings;
import io.euhedral_execution.inference.core.settings.SettingsLoader;
import io.euhedral_execution.inference.core.tokenizer.TokenizerFiles;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.beans.factory.BeanCreationException;
import org.springframework.boot.diagnostics.FailureAnalysis;

class SetupCheckTest {
    @TempDir
    Path directory;

    private static final DriverProbe DRIVER = () -> Optional.empty();

    /// The artifact the settings name, carrying a tokenizer (the check reads the section, not the model).
    private Settings installation() throws IOException {
        Path artifact = Files.write(this.directory.resolve("model.edrl"), new byte[64]);
        try (FileChannel channel = FileChannel.open(artifact, StandardOpenOption.WRITE)) {
            TokenizerSection.write(channel, 64, new TokenizerFiles("{}", "{}", "template", "{}"));
        }
        return properties(artifact);
    }

    private static Settings properties(Path artifact) {
        return SettingsLoader.load(
                Optional.empty(),
                Map.of("EUHEDRAL_MODEL_ARTIFACT", artifact.toString(), "EUHEDRAL_MODEL_NAME", "qwen"));
    }

    @Test
    void aCompleteInstallationHasNoProblems() throws IOException {
        assertEquals(List.of(), SetupCheck.problems(installation(), DRIVER));
    }

    @Test
    void reportsEveryProblemWithTheSettingThatNamesIt() throws IOException {
        installation();
        DriverProbe missingDriver = () -> Optional.of("libcuda.so.1 cannot be loaded");
        var problems = SetupCheck.problems(properties(this.directory.resolve("missing.edrl")), missingDriver);
        assertEquals(2, problems.size(), problems.toString());
        assertTrue(
                problems.get(0).startsWith("model.artifact (EUHEDRAL_MODEL_ARTIFACT): no file at "), problems.get(0));
        assertTrue(problems.get(1).contains("NVIDIA driver is required"), problems.get(1));
    }

    @Test
    void anArtifactWithoutATokenizerNamesTheCommandThatAddsIt() throws IOException {
        Path artifact = Files.write(this.directory.resolve("old.edrl"), new byte[64]);
        var problems = SetupCheck.problems(properties(artifact), DRIVER);
        assertEquals(1, problems.size(), problems.toString());
        assertTrue(
                problems.get(0)
                        .startsWith("model.artifact (EUHEDRAL_MODEL_ARTIFACT): " + artifact + " carries no tokenizer"),
                problems.get(0));
        assertTrue(problems.get(0).contains("package-tokenizer --artifact " + artifact), problems.get(0));
    }

    @Test
    void aMissingDriverIsOneProblemThatNamesTheDriver() throws IOException {
        DriverProbe missing = () -> Optional.of("libcuda.so.1 cannot be loaded");
        var problems = SetupCheck.problems(installation(), missing);
        assertEquals(1, problems.size(), problems.toString());
        assertTrue(problems.get(0).contains("NVIDIA driver is required"), problems.get(0));
        assertTrue(problems.get(0).contains("libcuda.so.1 cannot be loaded"), problems.get(0));
    }

    @Test
    void theAnalyzerReportsSetupProblemsAndEngineFailuresWithoutTheBeanChain() {
        var analyzer = new StartupFailureAnalyzer();
        var setup = new SetupCheck.SetupException(List.of("first problem", "second problem"));
        FailureAnalysis analysis = analyzer.analyze(new BeanCreationException("qwenChatTemplate", "wrapped", setup));
        assertNotNull(analysis);
        assertEquals("The server cannot start:\n\n    first problem\n    second problem", analysis.getDescription());

        var gpu = new EngineStartupException(
                "Loading the inference engine",
                new GpuMemoryException("an NVIDIA Blackwell GPU (compute capability 12.x) is required"));
        analysis = analyzer.analyze(new BeanCreationException("inferenceEngine", "wrapped", gpu));
        assertEquals(
                "Loading the inference engine failed:\n\n    an NVIDIA Blackwell GPU (compute capability 12.x) is required",
                analysis.getDescription());
        assertTrue(analysis.getAction().contains("nvidia-smi"), analysis.getAction());

        var context = new EngineStartupException(
                "Loading the inference engine", new IllegalArgumentException("set a smaller max context"));
        analysis = analyzer.analyze(context);
        assertTrue(analysis.getAction().startsWith("Correct what the message names."), analysis.getAction());

        assertNull(analyzer.analyze(new IllegalStateException("not ours")));
    }
}
