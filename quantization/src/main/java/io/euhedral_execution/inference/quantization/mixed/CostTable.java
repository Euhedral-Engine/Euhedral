package io.euhedral_execution.inference.quantization.mixed;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.euhedral_execution.inference.quantization.mixed.Cost.Key;
import io.euhedral_execution.inference.quantization.mixed.Cost.Shape;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/// The microseconds of one launch of a linear for each `(shape, option, rows)`, and the terms of a step that do not
/// depend on the allocation. The cost of an allocation is additive: the sum of its projections' launches plus the
/// fixed term.
///
/// The file is `{"fixed": {"prefillMicrosPerToken": x, "stepMicros": {"4": y}}, "entries": [{"in": k, "out": n,
/// "option": "trellis-3.0", "rows": 4, "micros": z}, ...]}`.
public final class CostTable {

    /// The rows of the launch a prefill token is charged at.
    public static final int PREFILL_ROWS = 512;

    private static final ObjectMapper JSON = new ObjectMapper();

    private final Map<Key, Double> micros;
    private final double fixedPrefillMicrosPerToken;
    private final Map<Integer, Double> fixedStepMicros;

    /// A table of `micros`; `fixedPrefillMicrosPerToken` is added to a prefill token's cost and `fixedStepMicros`
    /// (by rows, absent meaning 0) to a verification step's.
    public CostTable(Map<Key, Double> micros, double fixedPrefillMicrosPerToken, Map<Integer, Double> fixedStepMicros) {
        this.micros = Map.copyOf(micros);
        this.fixedPrefillMicrosPerToken = fixedPrefillMicrosPerToken;
        this.fixedStepMicros = Map.copyOf(fixedStepMicros);
    }

    /// This table with the fixed terms replaced, as measured end to end.
    public CostTable withFixed(double prefillPerToken, Map<Integer, Double> stepByRows) {
        return new CostTable(micros, prefillPerToken, stepByRows);
    }

    /// The microseconds of one launch of `key`; IllegalArgumentException naming it when it was not measured.
    public double micros(Key key) {
        Double value = micros.get(key);
        if (value == null)
            throw new IllegalArgumentException(
                    "no measured cost for " + key.option().key() + " at "
                            + key.shape().in() + " x " + key.shape().out() + " with " + key.rows() + " rows");
        return value;
    }

    /// The prefill cost of a token under `allocation`: each projection's launch at [#PREFILL_ROWS] rows over the
    /// rows, plus the fixed term.
    public double prefillMicrosPerToken(Sources sources, Allocation allocation) {
        double sum = 0;
        for (String name : sources.projections()) sum += micros(key(sources, allocation, name, PREFILL_ROWS));
        return sum / PREFILL_ROWS + fixedPrefillMicrosPerToken;
    }

    /// The cost of a `rows`-row step under `allocation`: each projection's launch at `rows`, plus the fixed term of
    /// that row count.
    public double verifyMicros(Sources sources, Allocation allocation, int rows) {
        double sum = 0;
        for (String name : sources.projections()) sum += micros(key(sources, allocation, name, rows));
        return sum + fixedStepMicros.getOrDefault(rows, 0.0);
    }

    private static Key key(Sources sources, Allocation allocation, String name, int rows) {
        Option option = allocation.choices().getOrDefault(name, sources.base());
        return new Key(Cost.shape(sources.descriptor(option, name)), option, rows);
    }

    /// The table in `json`.
    public static CostTable read(Path json) throws IOException {
        JsonNode root = JSON.readTree(json.toFile());
        Map<Key, Double> micros = new HashMap<>();
        JsonNode entries = root.get("entries");
        if (entries == null || !entries.isArray()) throw new IllegalArgumentException(json + " has no \"entries\"");
        for (JsonNode entry : entries)
            micros.put(
                    new Key(
                            new Shape(entry.get("in").asInt(), entry.get("out").asInt()),
                            Option.of(entry.get("option").asText()),
                            entry.get("rows").asInt()),
                    entry.get("micros").asDouble());
        JsonNode fixed = root.get("fixed");
        double prefill = 0;
        Map<Integer, Double> step = new HashMap<>();
        if (fixed != null) {
            prefill = fixed.path("prefillMicrosPerToken").asDouble(0);
            for (var entry : fixed.path("stepMicros").properties())
                step.put(Integer.parseInt(entry.getKey()), entry.getValue().asDouble());
        }
        return new CostTable(micros, prefill, step);
    }

    /// Writes this table to `json`, entries in a stable order.
    public void write(Path json) throws IOException {
        ObjectNode root = JSON.createObjectNode();
        ObjectNode fixed = root.putObject("fixed");
        fixed.put("prefillMicrosPerToken", fixedPrefillMicrosPerToken);
        ObjectNode step = fixed.putObject("stepMicros");
        new TreeMap<>(fixedStepMicros).forEach((rows, value) -> step.put(Integer.toString(rows), value));
        ArrayNode entries = root.putArray("entries");
        micros.entrySet().stream()
                .sorted(Comparator.comparingInt(
                                (Map.Entry<Key, Double> e) -> e.getKey().shape().in())
                        .thenComparingInt(e -> e.getKey().shape().out())
                        .thenComparing(e -> e.getKey().option())
                        .thenComparingInt(e -> e.getKey().rows()))
                .forEach(entry -> {
                    ObjectNode node = entries.addObject();
                    node.put("in", entry.getKey().shape().in());
                    node.put("out", entry.getKey().shape().out());
                    node.put("option", entry.getKey().option().key());
                    node.put("rows", entry.getKey().rows());
                    node.put("micros", entry.getValue());
                });
        Files.writeString(json, JSON.writerWithDefaultPrettyPrinter().writeValueAsString(root) + "\n");
    }
}
