package io.euhedral_execution.inference.core.settings;

import java.util.Map;
import java.util.Optional;
import java.util.Set;

/// Environment variables (and file keys) that once configured the server and no longer do. Setting one fails startup,
/// so a stale environment or configuration file is caught instead of silently ignored.
public final class RetiredSettings {

    private static final String BUNDLED = "removed: the jar carries its native files";
    private static final String SETTLED = "removed: the setting is fixed";
    private static final String TOKENIZER = "removed: the artifact carries its tokenizer";

    private static final Map<String, String> RETIRED = Map.ofEntries(
            renamed("EUHEDRAL_INFERENCE_ARTIFACT_PATH", "model.artifact"),
            Map.entry("EUHEDRAL_INFERENCE_TOKENIZER_DIRECTORY", TOKENIZER),
            Map.entry("EUHEDRAL_MODEL_TOKENIZER", TOKENIZER),
            renamed("EUHEDRAL_INFERENCE_MODEL_ID", "model.name"),
            renamed("EUHEDRAL_INFERENCE_WORKER_CPUS", "host.cpus"),
            renamed("EUHEDRAL_INFERENCE_MAX_CONTEXT_TOKENS", "context.max-tokens"),
            renamed("EUHEDRAL_INFERENCE_SHUTDOWN_TIMEOUT", "server.shutdown-timeout"),
            renamed("EUHEDRAL_INFERENCE_PREFIX_CACHE_BYTES", "prefix-cache.size"),
            renamed("EUHEDRAL_INFERENCE_PREFIX_CACHE_CHECKPOINT_TOKENS", "prefix-cache.checkpoint-tokens"),
            renamed("EUHEDRAL_INFERENCE_KV_CACHE", "context.kv-cache-format"),
            renamed("EUHEDRAL_INFERENCE_MAX_SEQUENCES", "batching.max-sequences"),
            renamed("EUHEDRAL_INFERENCE_KV_DEVICE_MINIMUM_TOKENS", "context.kv-cache-on-gpu-tokens"),
            renamed("EUHEDRAL_API_DEFAULT_MAX_TOKENS", "server.requests.default-max-tokens"),
            renamed("EUHEDRAL_API_MAX_QUEUED_GENERATIONS", "server.requests.max-queued"),
            renamed("EUHEDRAL_API_MAX_REQUEST_BYTES", "server.requests.max-body-size"),
            renamed("EUHEDRAL_API_REQUEST_TIMEOUT", "server.requests.timeout"),
            resized("EUHEDRAL_HOST_MEMORY_MIB", "memory.host-limit"),
            resized("EUHEDRAL_GPU_SYSTEM_RESERVE_MIB", "memory.gpu-reserve"),
            renamed("EUHEDRAL_LAKE_SINKS", "host.task-parallelism"),
            renamed("EUHEDRAL_QWEN4_READS", "memory.weight-streaming.concurrent-reads"),
            renamed("EUHEDRAL_QWEN4_READ_PARTS", "memory.weight-streaming.read-split"),
            renamed("EUHEDRAL_QWEN4_COPY_STREAMS", "memory.weight-streaming.copy-streams"),
            renamed("EUHEDRAL_QWEN4_LANES", "memory.weight-streaming.gpu-streams"),
            Map.entry("EUHEDRAL_INFERENCE_CUDA_LIBRARY_PATH", BUNDLED),
            Map.entry("EUHEDRAL_CUDA_INCLUDE_DIR", BUNDLED),
            settled("EUHEDRAL_LAKE_PARTITIONS"),
            settled("EUHEDRAL_CUDA_GRAPHS"),
            settled("EUHEDRAL_CUDA_KV_CODEC"),
            settled("EUHEDRAL_QWEN4_PREFETCH"),
            settled("EUHEDRAL_QWEN4_AHEAD"),
            settled("EUHEDRAL_QWEN4_LINEAR_TC"),
            settled("EUHEDRAL_QWEN4_LINEAR_SPLIT"),
            settled("EUHEDRAL_QWEN4_TIER_POLICY"),
            settled("EUHEDRAL_QWEN4_GPU_POLICY"),
            settled("EUHEDRAL_QWEN4_TIER_PRELOAD"),
            settled("EUHEDRAL_QWEN4_PIN_TIER"),
            settled("EUHEDRAL_QWEN4_TIER_ADMISSION"),
            settled("EUHEDRAL_QWEN4_GROUP"),
            settled("EUHEDRAL_QWEN4_DIRECT_READS"));

    private RetiredSettings() {}

    /// What to do about a retired variable, or empty when the variable is not retired.
    public static Optional<String> message(String variable) {
        return Optional.ofNullable(RETIRED.get(variable));
    }

    /// What to do about a retired key of the file, such as `model.tokenizer`, or empty when the key is not retired.
    public static Optional<String> keyMessage(String key) {
        return message(SettingsLoader.variableName(key));
    }

    /// Every retired variable name.
    static Set<String> variables() {
        return RETIRED.keySet();
    }

    private static Map.Entry<String, String> renamed(String variable, String key) {
        return Map.entry(variable, "renamed to " + key + " (" + SettingsLoader.variableName(key) + ")");
    }

    private static Map.Entry<String, String> resized(String variable, String key) {
        return Map.entry(variable, renamed(variable, key).getValue() + "; give a size with a unit such as 24GiB");
    }

    private static Map.Entry<String, String> settled(String variable) {
        return Map.entry(variable, SETTLED);
    }
}
