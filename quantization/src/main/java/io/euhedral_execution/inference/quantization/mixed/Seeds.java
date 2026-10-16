package io.euhedral_execution.inference.quantization.mixed;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/// Allocations to measure first. The file is `{"allocations": [{"objects": {"text/layers/0/mlp/down":
/// "trellis-3.0", ...}}, ...]}`, or one allocation file (`{"objects": {...}}`); an object an allocation does
/// not name keeps the base's option.
public final class Seeds {

    private static final ObjectMapper JSON = new ObjectMapper();

    private Seeds() {}

    public static List<Allocation> read(Path json) throws IOException {
        JsonNode root = JSON.readTree(json.toFile());
        List<Allocation> seeds = new ArrayList<>();
        if (root.has("allocations")) {
            for (JsonNode node : root.get("allocations")) seeds.add(allocation(json, node));
        } else {
            seeds.add(allocation(json, root));
        }
        return seeds;
    }

    private static Allocation allocation(Path json, JsonNode node) {
        JsonNode objects = node.get("objects");
        if (objects == null || !objects.isObject())
            throw new IllegalArgumentException(json + ": an allocation has no \"objects\" map");
        Map<String, Option> choices = new TreeMap<>();
        for (var entry : objects.properties()) {
            try {
                choices.put(entry.getKey(), Option.of(entry.getValue().asText()));
            } catch (IllegalArgumentException failure) {
                throw new IllegalArgumentException("object " + entry.getKey() + ": " + failure.getMessage(), failure);
            }
        }
        return new Allocation(choices);
    }
}
