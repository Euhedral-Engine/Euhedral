package io.euhedral_execution.inference.quantization.mixed;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/// The local error of each option of each object, a proxy the fidelity model starts from. The file is
/// `{"objects": {"text/layers/0/mlp/down": {"nvfp4": 0.0012, "trellis-2.5": 0.0031, ...}, ...}}`.
public record Priors(Map<String, Map<Option, Double>> objects) {

    private static final ObjectMapper JSON = new ObjectMapper();

    public Priors {
        Map<String, Map<Option, Double>> copy = new TreeMap<>();
        objects.forEach((name, byOption) -> copy.put(name, Map.copyOf(byOption)));
        objects = Map.copyOf(copy);
    }

    /// The priors in `json`; an unknown option key is refused naming it and its object.
    public static Priors read(Path json) throws IOException {
        JsonNode objects = JSON.readTree(json.toFile()).get("objects");
        if (objects == null || !objects.isObject())
            throw new IllegalArgumentException(json + " has no \"objects\" map");
        Map<String, Map<Option, Double>> read = new TreeMap<>();
        for (var entry : objects.properties()) {
            Map<Option, Double> byOption = new TreeMap<>();
            for (var error : entry.getValue().properties()) {
                if (!error.getValue().isNumber())
                    throw new IllegalArgumentException(
                            "object " + entry.getKey() + ": the error of " + error.getKey() + " is not a number");
                try {
                    byOption.put(Option.of(error.getKey()), error.getValue().asDouble());
                } catch (IllegalArgumentException failure) {
                    throw new IllegalArgumentException(
                            "object " + entry.getKey() + ": " + failure.getMessage(), failure);
                }
            }
            read.put(entry.getKey(), byOption);
        }
        return new Priors(read);
    }

    /// These priors of only `objects` and `options`; IllegalArgumentException naming what is missing.
    public Priors restrict(Collection<String> objects, Collection<Option> options) {
        Map<String, Map<Option, Double>> kept = new TreeMap<>();
        for (String object : objects) {
            Map<Option, Double> byOption = new TreeMap<>();
            for (Option option : options) byOption.put(option, error(object, option));
            kept.put(object, byOption);
        }
        return new Priors(kept);
    }

    /// The error of `option` at `object`; IllegalArgumentException naming both when there is none.
    public double error(String object, Option option) {
        Map<Option, Double> byOption = objects.get(object);
        if (byOption == null) throw new IllegalArgumentException("the priors have no object " + object);
        Double error = byOption.get(option);
        if (error == null)
            throw new IllegalArgumentException("the priors have no error of " + option.key() + " for " + object);
        return error;
    }

    /// The options every object has an error for, in the enum's order.
    public List<Option> options() {
        Set<Option> common = EnumSet.allOf(Option.class);
        objects.values().forEach(byOption -> common.retainAll(byOption.keySet()));
        return new ArrayList<>(common);
    }
}
