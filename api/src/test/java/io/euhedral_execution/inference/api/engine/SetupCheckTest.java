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
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.beans.factory.BeanCreationException;
import org.springframework.boot.diagnostics.FailureAnalysis;

class SetupCheckTest {
    @TempDir
    Path directory;

    /// An installation laid out as the image lays it out: `lib/` with both libraries, `share/euhedral_cuda/`,
    /// CUDA headers, an artifact and a checkpoint directory.
    private InferenceProperties installation() throws IOException {
        Path lib = Files.createDirectories(this.directory.resolve("lib"));
        Files.createFile(lib.resolve(System.mapLibraryName("euhedral_cuda")));
        Files.createFile(lib.resolve(System.mapLibraryName("llguidance")));
        Files.createDirectories(this.directory.resolve("share/euhedral_cuda"));
        Files.createFile(
                Files.createDirectories(this.directory.resolve("include")).resolve("cuda_fp4.h"));
        Files.createFile(this.directory.resolve("model.edrl"));
        Path checkpoint = Files.createDirectories(this.directory.resolve("checkpoint"));
        for (String file : List.of("tokenizer.json", "tokenizer_config.json", "generation_config.json"))
            Files.createFile(checkpoint.resolve(file));
        return properties(
                this.directory.resolve("model.edrl"), checkpoint, lib.resolve(System.mapLibraryName("euhedral_cuda")));
    }

    private static InferenceProperties properties(Path artifact, Path tokenizer, Path library) {
        return new InferenceProperties(
                artifact, tokenizer, library, "0", 32768, Duration.ofSeconds(10), "qwen", 0, 2048);
    }

    private Map<String, String> headers() {
        return Map.of(
                "EUHEDRAL_CUDA_INCLUDE_DIR", this.directory.resolve("include").toString());
    }

    @Test
    void aCompleteInstallationHasNoProblems() throws IOException {
        assertEquals(List.of(), SetupCheck.problems(installation(), headers()));
    }

    @Test
    void reportsEveryProblemWithTheSettingThatNamesIt() throws IOException {
        InferenceProperties complete = installation();
        Files.delete(complete.tokenizerDirectory().resolve("tokenizer.json"));
        Files.delete(complete.tokenizerDirectory().resolve("generation_config.json"));
        Files.delete(llguidance(complete));
        Files.delete(this.directory.resolve("share/euhedral_cuda"));
        var problems = SetupCheck.problems(
                properties(
                        this.directory.resolve("missing.edrl"),
                        complete.tokenizerDirectory(),
                        complete.cudaLibraryPath()),
                Map.of("CUDA_HOME", this.directory.resolve("nowhere").toString()));
        assertEquals(5, problems.size(), problems.toString());
        assertTrue(
                problems.get(0)
                        .startsWith("euhedral.inference.artifact-path (EUHEDRAL_INFERENCE_ARTIFACT_PATH): "
                                + "no file at "),
                problems.get(0));
        assertTrue(problems.get(1).contains("has no tokenizer.json, generation_config.json"), problems.get(1));
        assertTrue(problems.get(2).contains(Path.of("share", "euhedral_cuda").toString()), problems.get(2));
        assertTrue(problems.get(3).startsWith("euhedral.inference.cuda-library-path"), problems.get(3));
        assertTrue(problems.get(3).contains(System.mapLibraryName("llguidance")), problems.get(3));
        assertTrue(problems.get(4).startsWith("EUHEDRAL_CUDA_INCLUDE_DIR: "), problems.get(4));
        assertTrue(problems.get(4)
                .contains(this.directory.resolve("nowhere/include/cuda_fp4.h").toString()));
    }

    @Test
    void aMissingLibraryHidesTheChecksOfWhatSitsBesideIt() throws IOException {
        InferenceProperties complete = installation();
        var problems = SetupCheck.problems(
                properties(complete.artifactPath(), this.directory.resolve("none"), this.directory.resolve("none.so")),
                headers());
        assertEquals(
                List.of(
                        "euhedral.inference.tokenizer-directory (EUHEDRAL_INFERENCE_TOKENIZER_DIRECTORY): no directory at "
                                + this.directory.resolve("none"),
                        "euhedral.inference.cuda-library-path (EUHEDRAL_INFERENCE_CUDA_LIBRARY_PATH): no file at "
                                + this.directory.resolve("none.so")),
                problems);
    }

    @Test
    void theAnalyzerReportsSetupProblemsAndEngineFailuresWithoutTheBeanChain() {
        var analyzer = new StartupFailureAnalyzer();
        var setup = new SetupCheck.SetupException(List.of("first problem", "second problem"));
        FailureAnalysis analysis = analyzer.analyze(new BeanCreationException("qwenChatTemplate", "wrapped", setup));
        assertNotNull(analysis);
        assertEquals(
                "The server's settings point at files that are missing:\n\n    first problem\n    second problem",
                analysis.getDescription());

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

    private static Path llguidance(InferenceProperties properties) {
        return properties.cudaLibraryPath().resolveSibling(System.mapLibraryName("llguidance"));
    }
}
