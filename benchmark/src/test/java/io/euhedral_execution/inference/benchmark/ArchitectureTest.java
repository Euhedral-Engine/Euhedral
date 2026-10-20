package io.euhedral_execution.inference.benchmark;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/// A guard on where settings come from: the settings loader of the core reads the settings file and its variables
/// once, and nothing in this module reads a system property or an environment variable of its own.
class ArchitectureTest {

    private static final Path MAIN = Path.of("src/main/java/io/euhedral_execution/inference/benchmark");

    private static final Pattern SETTING_READ = Pattern.compile("System\\.getProperty\\(|System\\.getenv\\("
            + "|Integer\\.getInteger\\(|Long\\.getLong\\(|Boolean\\.getBoolean\\(");

    /// The benchmark reads its own options file, and an engine settings file through the settings loader.
    /// As `path` to names on the read's line.
    private static final Map<String, Set<String>> EXCEPTIONS = Map.of();

    @Test
    void onlyTheSettingsReadPropertiesAndVariables() throws IOException {
        List<String> violations = new ArrayList<>();
        Set<String> used = new HashSet<>();
        try (Stream<Path> walk = Files.walk(MAIN)) {
            for (Path file : walk.filter(p -> p.toString().endsWith(".java")).toList()) {
                String path = MAIN.relativize(file).toString().replace('\\', '/');
                List<String> lines = Files.readAllLines(file);
                for (int i = 0; i < lines.size(); i++) {
                    String code = lines.get(i).strip();
                    if (code.startsWith("//")
                            || code.startsWith("*")
                            || !SETTING_READ.matcher(code).find()) continue;
                    var names = EXCEPTIONS.getOrDefault(path, Set.of()).stream()
                            .filter(code::contains)
                            .toList();
                    if (names.isEmpty()) violations.add(path + ":" + (i + 1) + ": " + code);
                    else names.forEach(name -> used.add(path + "|" + name));
                }
            }
        }
        assertTrue(violations.isEmpty(), "reads outside the settings:\n" + String.join("\n", violations));
        for (var entry : EXCEPTIONS.entrySet())
            for (String name : entry.getValue())
                assertTrue(used.contains(entry.getKey() + "|" + name), "no longer read: " + entry.getKey());
    }
}
