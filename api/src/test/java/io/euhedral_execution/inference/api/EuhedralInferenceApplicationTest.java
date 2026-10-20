package io.euhedral_execution.inference.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.nativefiles.NativeFiles;
import io.euhedral_execution.inference.core.settings.RequestSettings;
import io.euhedral_execution.inference.core.settings.Settings;
import io.euhedral_execution.inference.core.settings.SettingsLoader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.net.ServerSocket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.context.ConfigurableApplicationContext;

@ResourceLock("scripted-api-context")
class EuhedralInferenceApplicationTest {
    @TempDir
    Path directory;

    private static String lines(ByteArrayOutputStream bytes) {
        return bytes.toString(StandardCharsets.UTF_8).replace("\r\n", "\n");
    }

    private Path file(String content) throws IOException {
        Path file = this.directory.resolve("config.yaml");
        Files.writeString(file, content);
        return file;
    }

    private static int freePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        }
    }

    @Test
    void aBadFileExitsWithStatusTwoAndListsEveryProblemOnItsOwnLine() throws IOException {
        Path file = file("""
                model:
                  artifact: /m.edrl
                  tokenizer: /t
                  name: n
                server:
                  port: 0
                  colour: red
                context:
                  max-tokens: many
                """);
        var err = new ByteArrayOutputStream();
        var launched = new AtomicBoolean();
        int status = EuhedralInferenceApplication.start(
                new String[] {"--config", file.toString()},
                Map.of(),
                new PrintStream(err, true, StandardCharsets.UTF_8),
                (settings, files) -> launched.set(true));
        assertEquals(2, status);
        assertFalse(launched.get());
        List<String> problems = lines(err).lines().toList();
        assertEquals(3, problems.size(), problems.toString());
        assertTrue(problems.stream().anyMatch(line -> line.contains("server.port")), problems.toString());
        assertTrue(problems.stream().anyMatch(line -> line.contains("server.colour")), problems.toString());
        assertTrue(problems.stream().anyMatch(line -> line.contains("context.max-tokens")), problems.toString());
    }

    @Test
    void aMissingRequiredKeyIsReportedBeforeAnythingStarts() {
        var err = new ByteArrayOutputStream();
        int status = EuhedralInferenceApplication.start(
                new String[0], Map.of(), new PrintStream(err, true, StandardCharsets.UTF_8), (s, f) -> {});
        assertEquals(2, status);
        assertTrue(lines(err).contains("model.artifact is required"), lines(err));
    }

    @Test
    void theServerPortAndRequestLimitsComeFromTheFile() throws IOException {
        int port = freePort();
        Path file = file("""
                model:
                  artifact: /m.edrl
                  tokenizer: /t
                  name: n
                server:
                  port: %d
                  requests:
                    default-max-tokens: 99
                    max-queued: 3
                    max-body-size: 2KiB
                    timeout: 5m
                """.formatted(port));
        Settings settings =
                SettingsLoader.load(SettingsLoader.locate(new String[] {"--config=" + file}, Map.of()), Map.of());
        NativeFiles files = NativeFiles.at(this.directory);
        try (ConfigurableApplicationContext context = EuhedralInferenceApplication.application(
                        settings, files, Probe.class)
                .run()) {
            assertEquals(String.valueOf(port), context.getEnvironment().getProperty("local.server.port"));
            assertSame(settings, context.getBean(Settings.class));
            assertSame(files, context.getBean(NativeFiles.class));
            assertEquals(
                    new RequestSettings(99, 3, 2048, Duration.ofMinutes(5)), context.getBean(RequestSettings.class));
            assertEquals(Optional.empty(), settings.host().nativeCache());
        }
    }

    @Test
    void configIsNotPassedOnToSpring() {
        assertEquals(List.of("--x=1", "--y"), List.of(EuhedralInferenceApplication.springArguments(new String[] {
            "--x=1", "--config", "f.yaml", "--config=g.yaml", "--y"
        })));
    }

    @SpringBootConfiguration
    @EnableAutoConfiguration
    static class Probe {}
}
