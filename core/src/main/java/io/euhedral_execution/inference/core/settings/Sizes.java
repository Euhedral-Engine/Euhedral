package io.euhedral_execution.inference.core.settings;

import java.time.Duration;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/// Parsing of sizes (`4GiB`, `700MiB`, plain bytes) and durations (`10s`, `30m`, `1000ms`).
public final class Sizes {

    private static final Pattern SIZE = Pattern.compile("(\\d+)\\s*(B|KiB|MiB|GiB|TiB)?");
    private static final Pattern DURATION = Pattern.compile("(\\d+)\\s*(ms|s|m|h|d)");

    private Sizes() {}

    /// A size in bytes: whole number with an optional binary unit; no unit means bytes.
    public static long parseSize(String text) {
        Matcher m = SIZE.matcher(text.strip());
        if (!m.matches())
            throw new IllegalArgumentException("not a size (whole number, optional B/KiB/MiB/GiB/TiB): " + text);
        long value = number(m.group(1), text);
        int shift =
                switch (m.group(2) == null ? "B" : m.group(2)) {
                    case "KiB" -> 10;
                    case "MiB" -> 20;
                    case "GiB" -> 30;
                    case "TiB" -> 40;
                    default -> 0;
                };
        if (value > (Long.MAX_VALUE >> shift)) throw new IllegalArgumentException("size too large: " + text);
        return value << shift;
    }

    /// A duration: whole number with a unit (ms, s, m, h, d); a bare `0` is zero.
    public static Duration parseDuration(String text) {
        String t = text.strip();
        if (t.equals("0")) return Duration.ZERO;
        Matcher m = DURATION.matcher(t);
        if (!m.matches()) throw new IllegalArgumentException("not a duration (whole number with ms/s/m/h/d): " + text);
        long value = number(m.group(1), text);
        try {
            return switch (m.group(2)) {
                case "ms" -> Duration.ofMillis(value);
                case "s" -> Duration.ofSeconds(value);
                case "m" -> Duration.ofMinutes(value);
                case "h" -> Duration.ofHours(value);
                default -> Duration.ofDays(value);
            };
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("duration too large: " + text);
        }
    }

    private static long number(String digits, String text) {
        try {
            return Long.parseLong(digits);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("number too large: " + text);
        }
    }
}
