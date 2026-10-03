package io.euhedral_execution.inference.benchmark.result;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/// Short human summary: medians over measured successful rows. Warmup rows are excluded; ineligible
/// and failed rows are only counted.
public final class Summary {
    private static final String LABEL = "implementation | artifact | scenario";

    private Summary() {}

    public static String format(List<BenchmarkResult> rows) {
        Map<String, List<BenchmarkResult>> groups = new LinkedHashMap<>();
        for (BenchmarkResult row : rows) {
            if (row.warmup()) continue;
            String artifact = row.engine() == null
                    ? "-"
                    : row.engine().path("configuration").path("artifact").asText("-");
            groups.computeIfAbsent(
                            row.implementation() + " | " + artifact + " | "
                                    + row.scenario().name(),
                            ignored -> new ArrayList<>())
                    .add(row);
        }
        int width = groups.keySet().stream().mapToInt(String::length).max().orElse(0);
        width = Math.max(width, LABEL.length());
        StringBuilder out = new StringBuilder();
        out.append(String.format(
                Locale.ROOT,
                "%-" + width + "s %4s %4s %4s %12s %10s %12s %10s%n",
                LABEL,
                "ok",
                "inel",
                "fail",
                "prefill t/s",
                "ttft ms",
                "decode t/s",
                "e2e ms"));
        for (var entry : groups.entrySet()) {
            List<BenchmarkResult> group = entry.getValue();
            List<BenchmarkResult> ok = group.stream()
                    .filter(row -> BenchmarkResult.SUCCESS.equals(row.status()))
                    .toList();
            out.append(String.format(
                    Locale.ROOT,
                    "%-" + width + "s %4d %4d %4d %12s %10s %12s %10s%n",
                    entry.getKey(),
                    ok.size(),
                    count(group, BenchmarkResult.INELIGIBLE),
                    count(group, BenchmarkResult.FAILED),
                    median(ok, row -> row.throughput().prefillTokensPerSecond(), 1.0),
                    median(ok, row -> nanos(row.timings().timeToFirstToken()), 1e-6),
                    median(ok, row -> row.throughput().decodeTokensPerSecond(), 1.0),
                    median(ok, row -> nanos(row.timings().endToEnd()), 1e-6)));
        }
        return out.toString();
    }

    private static long count(List<BenchmarkResult> rows, String status) {
        return rows.stream().filter(row -> status.equals(row.status())).count();
    }

    private static Double nanos(Long value) {
        return value == null ? null : value.doubleValue();
    }

    static String median(List<BenchmarkResult> rows, Function<BenchmarkResult, Double> metric, double scale) {
        double[] values = rows.stream()
                .map(metric)
                .filter(Objects::nonNull)
                .mapToDouble(Double::doubleValue)
                .sorted()
                .toArray();
        if (values.length == 0) return "-";
        int middle = values.length / 2;
        double median = values.length % 2 == 1 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
        return String.format(Locale.ROOT, "%.1f", median * scale);
    }
}
