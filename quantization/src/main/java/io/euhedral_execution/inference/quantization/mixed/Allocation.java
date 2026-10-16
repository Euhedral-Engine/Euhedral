package io.euhedral_execution.inference.quantization.mixed;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;

/// The option chosen for each projection object, by object name; an object not listed keeps the base's. The file is
/// `{"objects": {"text/layers/0/mlp/down": "trellis-3.0", ...}}`, keys sorted.
public record Allocation(Map<String, Option> choices) {

    private static final ObjectMapper JSON = new ObjectMapper();

    public Allocation {
        choices = Map.copyOf(choices);
    }

    /// The allocation in `json`; an unknown option key is refused naming it and its object.
    public static Allocation read(Path json) throws IOException {
        JsonNode objects = JSON.readTree(json.toFile()).get("objects");
        if (objects == null || !objects.isObject())
            throw new IllegalArgumentException(json + " has no \"objects\" map");
        Map<String, Option> choices = new TreeMap<>();
        for (var entry : objects.properties()) {
            if (!entry.getValue().isTextual())
                throw new IllegalArgumentException("object " + entry.getKey() + " has no option name");
            try {
                choices.put(entry.getKey(), Option.of(entry.getValue().asText()));
            } catch (IllegalArgumentException failure) {
                throw new IllegalArgumentException("object " + entry.getKey() + ": " + failure.getMessage(), failure);
            }
        }
        return new Allocation(choices);
    }

    /// Writes this allocation to `json`.
    public void write(Path json) throws IOException {
        ObjectNode root = JSON.createObjectNode();
        ObjectNode objects = root.putObject("objects");
        new TreeMap<>(choices).forEach((name, option) -> objects.put(name, option.key()));
        Files.writeString(json, JSON.writerWithDefaultPrettyPrinter().writeValueAsString(root) + "\n");
    }
}
