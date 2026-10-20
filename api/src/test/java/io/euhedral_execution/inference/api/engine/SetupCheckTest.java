package io.euhedral_execution.inference.api.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.GpuMemoryException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.beans.factory.BeanCreationException;
import org.springframework.boot.diagnostics.FailureAnalysis;

class SetupCheckTest {
    @TempDir
    Path directory;

    private static final DriverProbe DRIVER = () -> Optional.empty();

    /// The model files the settings name: an artifact and a checkpoint directory.
    private InferenceProperties installation() throws IOException {
        Files.createFile(this.directory.resolve("model.edrl"));
        Path checkpoint = Files.createDirectories(this.directory.resolve("checkpoint"));
        for (String file : List.of("tokenizer.json", "tokenizer_config.json", "generation_config.json"))
            Files.createFile(checkpoint.resolve(file));
        return properties(this.directory.resolve("model.edrl"), checkpoint);
    }

    private static InferenceProperties properties(Path artifact, Path tokenizer) {
        return new InferenceProperties(
                artifact, tokenizer, null, "0", 32768, Duration.ofSeconds(10), "qwen", 0, 2048, null, 1, 65536);
    }

    @Test
    void aCompleteInstallationHasNoProblems() throws IOException {
        assertEquals(List.of(), SetupCheck.problems(installation(), DRIVER));
    }

    @Test
    void reportsEveryProblemWithTheSettingThatNamesIt() throws IOException {
        InferenceProperties complete = installation();
        Files.delete(complete.tokenizerDirectory().resolve("tokenizer.json"));
        Files.delete(complete.tokenizerDirectory().resolve("generation_config.json"));
        var problems = SetupCheck.problems(
                properties(this.directory.resolve("missing.edrl"), complete.tokenizerDirectory()), DRIVER);
        assertEquals(2, problems.size(), problems.toString());
        assertTrue(
                problems.get(0)
                        .startsWith("euhedral.inference.artifact-path (EUHEDRAL_INFERENCE_ARTIFACT_PATH): "
                                + "no file at "),
                problems.get(0));
        assertTrue(problems.get(1).contains("has no tokenizer.json, generation_config.json"), problems.get(1));
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
