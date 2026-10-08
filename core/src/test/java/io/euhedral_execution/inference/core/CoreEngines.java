package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.testing.Shared;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;

/// Engines that the CUDA tests of this package (and the dense model's tests) share, one per configuration. Two tests
/// that need the same configuration ask for the same key and so share one engine; asking for another key closes the
/// engine in use ([SharedEngines]). The prefix-cache tests have their own ([PrefixCacheEngines]).
///
/// A test does not assume a fresh engine: it reads allocated bytes and counters as differences, and ends its sessions.
public final class CoreEngines {
    static final String WORKSPACE_ROWS = "euhedral.workspace.rows";

    private CoreEngines() {}

    /// Closes the shared engine in use, for a test that loads an engine of its own (to check what closing it frees, or
    /// to load it on another device fixture): two engines do not fit the device.
    static void releaseDevice() {
        AutoCloseable none = () -> {};
        Shared.get("none", "core:none", () -> none);
    }

    /// The compact Q3 artifact with the default context and workspace.
    public static InferenceEngine q3() {
        return q3Held().engine();
    }

    static SharedEngines.Held q3Held() {
        return SharedEngines.get(
                "core:q3", () -> config(PrefixCacheEngines.Q3_PROPERTY, PrefixCacheEngines.Q3_DEFAULT, 0));
    }

    /// The compact Q3 artifact with a workspace wide enough for prefill chunks of up to `rows` tokens.
    static SharedEngines.Held q3Wide(int rows) {
        String previous = System.getProperty(WORKSPACE_ROWS);
        System.setProperty(WORKSPACE_ROWS, Integer.toString(rows));
        try {
            return SharedEngines.get(
                    "core:q3:rows" + rows,
                    () -> config(PrefixCacheEngines.Q3_PROPERTY, PrefixCacheEngines.Q3_DEFAULT, 0));
        } finally {
            if (previous == null) System.clearProperty(WORKSPACE_ROWS);
            else System.setProperty(WORKSPACE_ROWS, previous);
        }
    }

    /// The NVFP4 artifact with a context of `maxContextTokens` (0 for the default).
    static SharedEngines.Held nvfp4(int maxContextTokens) {
        return SharedEngines.get(
                "core:nvfp4:" + maxContextTokens,
                () -> config(PrefixCacheEngines.NVFP4_PROPERTY, PrefixCacheEngines.NVFP4_DEFAULT, maxContextTokens));
    }

    private static InferenceConfig config(String artifactProperty, String artifactDefault, int maxContextTokens) {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "CUDA library is required");
        Path artifact = Path.of(System.getProperty(artifactProperty, artifactDefault));
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(artifact) && Files.isRegularFile(tokenizer.resolve("tokenizer.json")));
        BitSet cpus = new BitSet();
        var available = SystemInfo.getPCpuSet();
        for (int cpu = available.nextSetBit(0); cpu >= 0 && cpus.cardinality() < 2; cpu = available.nextSetBit(cpu + 1))
            cpus.set(cpu);
        assumeTrue(cpus.cardinality() == 2);
        Duration shutdown = Duration.ofSeconds(10);
        return maxContextTokens == 0
                ? new InferenceConfig(artifact, tokenizer, Path.of(library), cpus, shutdown)
                : new InferenceConfig(artifact, tokenizer, Path.of(library), cpus, maxContextTokens, shutdown);
    }
}
