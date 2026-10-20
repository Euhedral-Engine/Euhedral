package io.euhedral_execution.inference.core.settings;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import io.euhedral_execution.inference.core.state.KvFormat;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Pattern;

/// Reads the deployer's settings: one YAML file, overridden key by key from the environment, validated in one place.
///
/// A variable named `EUHEDRAL_` plus the key path upper-cased with `.` and `-` turned into `_` replaces the file's
/// value for that key (`EUHEDRAL_MODEL_ARTIFACT` for `model.artifact`); `PORT` is accepted for `server.port`. An
/// unknown
/// key in the file and a variable named in {@link RetiredSettings} are errors; other variables are ignored. An optional
/// key written `auto` is resolved at startup from the machine. Every problem found is reported together.
public final class SettingsLoader {

    private static final String PREFIX = "EUHEDRAL_";
    private static final Pattern CPUS = Pattern.compile("\\d+(-\\d+)?(,\\d+(-\\d+)?)*");

    /// How one leaf key reads its text.
    private record Leaf(Function<String, Object> parser, boolean required, boolean auto) {}

    private static final Map<String, Leaf> LEAVES = leaves();

    private SettingsLoader() {}

    /// Every leaf key path, such as `memory.weight-streaming.copy-streams`.
    public static Set<String> knownKeys() {
        return Collections.unmodifiableSet(LEAVES.keySet());
    }

    /// The variable that overrides a key.
    public static String variableName(String key) {
        return PREFIX + key.toUpperCase(Locale.ROOT).replace('.', '_').replace('-', '_');
    }

    /// The configuration file named by `--config <path>` or `--config=<path>`, else by `EUHEDRAL_CONFIG`.
    public static Optional<Path> locate(String[] args, Map<String, String> environment) {
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--config") && i + 1 < args.length) return Optional.of(Path.of(args[i + 1]));
            if (args[i].startsWith("--config=")) return Optional.of(Path.of(args[i].substring("--config=".length())));
        }
        String variable = environment.get("EUHEDRAL_CONFIG");
        return variable == null || variable.isBlank() ? Optional.empty() : Optional.of(Path.of(variable));
    }

    /// Loads from the process environment.
    public static Settings load(Optional<Path> file) {
        return load(file, System.getenv());
    }

    /// Loads the file (if any), applies the environment, validates, and builds the settings.
    ///
    /// @throws SettingsException with every problem found
    public static Settings load(Optional<Path> file, Map<String, String> environment) {
        List<String> problems = new ArrayList<>();
        Map<String, String> raw = new LinkedHashMap<>();
        Map<String, String> source = new LinkedHashMap<>();
        file.ifPresent(f -> readFile(f, raw, source, problems));
        applyEnvironment(environment, raw, source, problems);

        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<String, Leaf> entry : LEAVES.entrySet()) {
            String key = entry.getKey();
            Leaf leaf = entry.getValue();
            String text = raw.get(key);
            if (text == null) {
                if (leaf.required())
                    problems.add(key + " is required (set it in the file or as " + variableName(key) + ")");
                continue;
            }
            if (text.strip().equalsIgnoreCase("auto")) {
                if (!leaf.auto()) problems.add(describe(key, source) + ": auto is not accepted, give a value");
                continue;
            }
            try {
                values.put(key, leaf.parser().apply(text.strip()));
            } catch (IllegalArgumentException e) {
                problems.add(describe(key, source) + ": " + e.getMessage());
            }
        }
        if (!problems.isEmpty()) throw new SettingsException(problems);
        try {
            return build(values);
        } catch (IllegalArgumentException e) {
            throw new SettingsException(List.of(e.getMessage()));
        }
    }

    private static void readFile(
            Path file, Map<String, String> raw, Map<String, String> source, List<String> problems) {
        if (!Files.isRegularFile(file)) {
            problems.add("configuration file not found: " + file);
            return;
        }
        JsonNode root;
        try {
            root = new ObjectMapper(new YAMLFactory()).readTree(Files.readString(file));
        } catch (IOException e) {
            problems.add("configuration file " + file + " is not valid YAML: " + e.getMessage());
            return;
        }
        if (root == null || root.isMissingNode() || root.isNull()) return;
        if (!root.isObject()) {
            problems.add("configuration file " + file + " must hold a mapping of settings");
            return;
        }
        walk("", root, raw, source, problems);
    }

    private static void walk(
            String prefix, JsonNode node, Map<String, String> raw, Map<String, String> source, List<String> problems) {
        for (Map.Entry<String, JsonNode> field : node.properties()) {
            String key = prefix.isEmpty() ? field.getKey() : prefix + "." + field.getKey();
            JsonNode value = field.getValue();
            boolean leaf = LEAVES.containsKey(key);
            boolean section = !leaf && LEAVES.keySet().stream().anyMatch(k -> k.startsWith(key + "."));
            if (!leaf && !section) {
                problems.add("unknown setting " + key);
            } else if (value.isNull()) {
                problems.add(key + " has no value");
            } else if (section) {
                if (value.isObject()) walk(key, value, raw, source, problems);
                else problems.add(key + " is a section, not a value");
            } else if (value.isContainerNode()) {
                problems.add(key + " must be a single value");
            } else {
                raw.put(key, value.asText());
            }
        }
    }

    private static void applyEnvironment(
            Map<String, String> environment,
            Map<String, String> raw,
            Map<String, String> source,
            List<String> problems) {
        for (String variable : RetiredSettings.variables())
            if (environment.containsKey(variable))
                problems.add(variable + " is retired: "
                        + RetiredSettings.message(variable).orElseThrow());
        for (String key : LEAVES.keySet()) {
            String variable = variableName(key);
            String value = environment.get(variable);
            if (value == null && key.equals("server.port")) {
                value = environment.get("PORT");
                variable = "PORT";
            }
            if (value == null) continue;
            if (value.isBlank()) {
                problems.add(variable + " is empty");
                continue;
            }
            raw.put(key, value);
            source.put(key, variable);
        }
    }

    private static String describe(String key, Map<String, String> source) {
        String variable = source.get(key);
        return variable == null ? key : key + " (" + variable + ")";
    }

    private static Settings build(Map<String, Object> v) {
        Get g = new Get(v);
        ModelSettings model =
                new ModelSettings(g.path("model.artifact"), g.path("model.tokenizer"), g.text("model.name"));

        RequestSettings rd = RequestSettings.defaults();
        ServerSettings sd = ServerSettings.defaults();
        ServerSettings server = new ServerSettings(
                g.integer("server.port", sd.port()),
                g.duration("server.shutdown-timeout", sd.shutdownTimeout()),
                new RequestSettings(
                        g.integer("server.requests.default-max-tokens", rd.defaultMaxTokens()),
                        g.integer("server.requests.max-queued", rd.maxQueued()),
                        g.size("server.requests.max-body-size", rd.maxBodySize()),
                        g.duration("server.requests.timeout", rd.timeout())));

        ContextSettings cd = ContextSettings.defaults();
        ContextSettings context = new ContextSettings(
                g.integer("context.max-tokens", cd.maxTokens()),
                Optional.ofNullable((KvFormat) v.get("context.kv-cache-format")),
                g.integer("context.kv-cache-on-gpu-tokens", cd.kvCacheOnGpuTokens()));

        PrefixCacheSettings pd = PrefixCacheSettings.defaults();
        PrefixCacheSettings prefixCache = new PrefixCacheSettings(
                g.size("prefix-cache.size", pd.size()),
                g.integer("prefix-cache.checkpoint-tokens", pd.checkpointTokens()));

        BatchingSettings bd = BatchingSettings.defaults();
        BatchingSettings batching = new BatchingSettings(
                g.integer("batching.max-sequences", bd.maxSequences()),
                g.optionalInt("batching.max-sequences-per-pass"),
                g.optionalInt("batching.max-tokens-per-pass"),
                g.integer("batching.prefill-chunk-tokens", bd.prefillChunkTokens()));

        MemorySettings md = MemorySettings.defaults();
        WeightStreamingSettings wd = WeightStreamingSettings.defaults();
        MemorySettings memory = new MemorySettings(
                g.optionalLong("memory.host-limit"),
                g.optionalLong("memory.host-weights-limit"),
                g.optionalLong("memory.gpu-limit"),
                g.size("memory.gpu-reserve", md.gpuReserve()),
                g.optionalLong("memory.pinned-buffer-pool"),
                new WeightStreamingSettings(
                        g.integer("memory.weight-streaming.concurrent-reads", wd.concurrentReads()),
                        g.integer("memory.weight-streaming.read-split", wd.readSplit()),
                        g.integer("memory.weight-streaming.copy-streams", wd.copyStreams()),
                        g.integer("memory.weight-streaming.gpu-streams", wd.gpuStreams())));

        HostSettings hd = HostSettings.defaults();
        HostSettings host = new HostSettings(
                Optional.ofNullable((String) v.get("host.cpus")),
                g.integer("host.task-parallelism", hd.taskParallelism()),
                g.duration("host.idle-release", hd.idleRelease()),
                Optional.ofNullable((Path) v.get("host.native-cache")));

        TrellisKernelSettings td = TrellisKernelSettings.defaults();
        KernelSettings kernels = new KernelSettings(new TrellisKernelSettings(
                g.integer("kernels.trellis.tensor-core-max-tokens", td.tensorCoreMaxTokens()),
                g.optionalInt("kernels.trellis.target-blocks"),
                g.integer("kernels.trellis.whole-k-min-tokens", td.wholeKMinTokens()),
                g.integer("kernels.trellis.wide-block-min-tokens", td.wideBlockMinTokens())));

        return new Settings(model, server, context, prefixCache, batching, memory, host, kernels);
    }

    /// Typed reads of the parsed values; a missing key reads as its default.
    private record Get(Map<String, Object> values) {
        Path path(String key) {
            return (Path) values.get(key);
        }

        String text(String key) {
            return (String) values.get(key);
        }

        int integer(String key, int fallback) {
            return values.containsKey(key) ? (Integer) values.get(key) : fallback;
        }

        long size(String key, long fallback) {
            return values.containsKey(key) ? (Long) values.get(key) : fallback;
        }

        Duration duration(String key, Duration fallback) {
            return values.containsKey(key) ? (Duration) values.get(key) : fallback;
        }

        OptionalInt optionalInt(String key) {
            return values.containsKey(key) ? OptionalInt.of((Integer) values.get(key)) : OptionalInt.empty();
        }

        OptionalLong optionalLong(String key) {
            return values.containsKey(key) ? OptionalLong.of((Long) values.get(key)) : OptionalLong.empty();
        }
    }

    private static Map<String, Leaf> leaves() {
        Map<String, Leaf> m = new LinkedHashMap<>();
        m.put("model.artifact", required(SettingsLoader::path));
        m.put("model.tokenizer", required(SettingsLoader::path));
        m.put("model.name", required(SettingsLoader::name));

        m.put("server.port", fixed(integer(1, 65535)));
        m.put("server.shutdown-timeout", fixed(duration(false)));
        m.put("server.requests.default-max-tokens", fixed(integer(1, Integer.MAX_VALUE)));
        m.put("server.requests.max-queued", fixed(integer(1, Integer.MAX_VALUE)));
        m.put("server.requests.max-body-size", fixed(size(1)));
        m.put("server.requests.timeout", fixed(duration(true)));

        m.put("context.max-tokens", fixed(integer(1, Integer.MAX_VALUE)));
        m.put("context.kv-cache-format", optional(SettingsLoader::kvFormat));
        m.put("context.kv-cache-on-gpu-tokens", fixed(integer(0, Integer.MAX_VALUE)));

        m.put("prefix-cache.size", fixed(size(0)));
        m.put("prefix-cache.checkpoint-tokens", fixed(SettingsLoader::checkpoint));

        m.put("batching.max-sequences", fixed(integer(0, Integer.MAX_VALUE)));
        m.put("batching.max-sequences-per-pass", optional(integer(1, Integer.MAX_VALUE)));
        m.put("batching.max-tokens-per-pass", optional(integer(1, Integer.MAX_VALUE)));
        m.put("batching.prefill-chunk-tokens", fixed(integer(1, Integer.MAX_VALUE)));

        m.put("memory.host-limit", optional(size(1)));
        m.put("memory.host-weights-limit", optional(size(1)));
        m.put("memory.gpu-limit", optional(size(1)));
        m.put("memory.gpu-reserve", fixed(size(0)));
        m.put("memory.pinned-buffer-pool", optional(size(0)));
        m.put("memory.weight-streaming.concurrent-reads", fixed(integer(1, Integer.MAX_VALUE)));
        m.put("memory.weight-streaming.read-split", fixed(integer(1, Integer.MAX_VALUE)));
        m.put("memory.weight-streaming.copy-streams", fixed(integer(1, Integer.MAX_VALUE)));
        m.put("memory.weight-streaming.gpu-streams", fixed(integer(1, Integer.MAX_VALUE)));

        m.put("host.cpus", optional(SettingsLoader::cpus));
        m.put("host.task-parallelism", fixed(integer(1, Integer.MAX_VALUE)));
        m.put("host.idle-release", fixed(duration(false)));
        m.put("host.native-cache", optional(SettingsLoader::path));

        m.put("kernels.trellis.tensor-core-max-tokens", fixed(integer(8, Integer.MAX_VALUE)));
        m.put("kernels.trellis.target-blocks", optional(integer(1, Integer.MAX_VALUE)));
        m.put("kernels.trellis.whole-k-min-tokens", fixed(integer(1, Integer.MAX_VALUE)));
        m.put("kernels.trellis.wide-block-min-tokens", fixed(integer(1, Integer.MAX_VALUE)));
        return m;
    }

    private static Leaf required(Function<String, Object> parser) {
        return new Leaf(parser, true, false);
    }

    private static Leaf fixed(Function<String, Object> parser) {
        return new Leaf(parser, false, false);
    }

    private static Leaf optional(Function<String, Object> parser) {
        return new Leaf(parser, false, true);
    }

    private static Function<String, Object> integer(int min, int max) {
        return text -> {
            int value;
            try {
                value = Integer.parseInt(text);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("not a whole number: " + text);
            }
            if (value < min || value > max)
                throw new IllegalArgumentException("must be " + min + " to " + max + ": " + text);
            return value;
        };
    }

    private static Function<String, Object> size(long min) {
        return text -> {
            long value = Sizes.parseSize(text);
            if (value < min) throw new IllegalArgumentException("must be at least " + min + " bytes: " + text);
            return value;
        };
    }

    private static Function<String, Object> duration(boolean positive) {
        return text -> {
            Duration value = Sizes.parseDuration(text);
            if (positive && value.isZero()) throw new IllegalArgumentException("must be positive: " + text);
            return value;
        };
    }

    private static Object path(String text) {
        return Path.of(text);
    }

    private static Object name(String text) {
        if (text.isBlank()) throw new IllegalArgumentException("must not be blank");
        return text;
    }

    private static Object kvFormat(String text) {
        return switch (text.toLowerCase(Locale.ROOT)) {
            case "nvfp4" -> KvFormat.NVFP4;
            case "bf16" -> KvFormat.BF16;
            default -> throw new IllegalArgumentException("must be nvfp4, bf16 or auto: " + text);
        };
    }

    private static Object cpus(String text) {
        if (!CPUS.matcher(text).matches())
            throw new IllegalArgumentException("must be a CPU list such as 2-5,8: " + text);
        return text;
    }

    private static Object checkpoint(String text) {
        int value = (Integer) integer(1, Integer.MAX_VALUE).apply(text);
        if (value % 512 != 0) throw new IllegalArgumentException("must be a positive multiple of 512: " + text);
        return value;
    }
}
