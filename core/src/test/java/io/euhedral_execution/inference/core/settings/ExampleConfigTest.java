package io.euhedral_execution.inference.core.settings;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

class ExampleConfigTest {

    private static Path example() {
        Path dir = Path.of("").toAbsolutePath();
        while (dir != null && !Files.isRegularFile(dir.resolve("config/euhedral.example.yaml"))) dir = dir.getParent();
        assertTrue(dir != null, "config/euhedral.example.yaml not found above the working directory");
        return dir.resolve("config/euhedral.example.yaml");
    }

    /// The key paths the file sets, read from its indentation (two spaces per level).
    private static Set<String> keysInFile(Path file) throws Exception {
        List<String> stack = new ArrayList<>();
        Set<String> keys = new HashSet<>();
        for (String line : Files.readAllLines(file)) {
            if (line.isBlank() || line.stripLeading().startsWith("#")) continue;
            int depth = (line.length() - line.stripLeading().length()) / 2;
            String key = line.strip().split(":")[0];
            while (stack.size() > depth) stack.remove(stack.size() - 1);
            stack.add(key);
            keys.add(String.join(".", stack));
        }
        return keys;
    }

    @Test
    void exampleSetsEveryKey() throws Exception {
        Set<String> inFile = keysInFile(example());
        for (String key : SettingsLoader.knownKeys()) assertTrue(inFile.contains(key), "example lacks " + key);
    }

    @Test
    void exampleEqualsDefaults() {
        Settings s = SettingsLoader.load(Optional.of(example()), Map.of());
        Settings expected = new Settings(
                s.model(),
                ServerSettings.defaults(),
                ContextSettings.defaults(),
                PrefixCacheSettings.defaults(),
                BatchingSettings.defaults(),
                MemorySettings.defaults(),
                HostSettings.defaults(),
                KernelSettings.defaults());
        assertEquals(expected, s);
    }
}
