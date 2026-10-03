package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.core.InferenceConfig;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/// Engine inputs bound from `euhedral.inference.*` (or `EUHEDRAL_INFERENCE_*` environment variables).
///
/// `workerCpus` lists processor IDs accepted by Euhedral, not a worker count: comma-separated IDs and
/// inclusive ranges such as `2,3,8-11`. `modelId` is the public name clients send as `model`.
/// `maxContextTokens` is the longest prompt plus completion a request may use (default 32768); the
/// engine keeps device memory for that much context and holds back weights in pinned host memory when
/// the GPU cannot fit both.
@ConfigurationProperties("euhedral.inference")
public record InferenceProperties(
        Path artifactPath,
        Path tokenizerDirectory,
        Path cudaLibraryPath,
        String workerCpus,
        @DefaultValue("32768") int maxContextTokens,
        @DefaultValue("10s") Duration shutdownTimeout,
        String modelId) {

    public InferenceProperties {
        require(artifactPath, "artifact-path");
        require(tokenizerDirectory, "tokenizer-directory");
        require(cudaLibraryPath, "cuda-library-path");
        require(workerCpus, "worker-cpus");
        require(shutdownTimeout, "shutdown-timeout");
        require(modelId, "model-id");
        if (modelId.isBlank()) throw new IllegalArgumentException("euhedral.inference.model-id must not be blank");
        parseCpus(workerCpus);
        if (maxContextTokens <= 0)
            throw new IllegalArgumentException("euhedral.inference.max-context-tokens must be positive");
    }

    /// Converts to the core record, which performs its own lifetime and CPU-set validation.
    public InferenceConfig toInferenceConfig() {
        return new InferenceConfig(
                this.artifactPath,
                this.tokenizerDirectory,
                this.cudaLibraryPath,
                parseCpus(this.workerCpus),
                this.maxContextTokens,
                this.shutdownTimeout);
    }

    static BitSet parseCpus(String specification) {
        BitSet cpus = new BitSet();
        for (String part : specification.split(",")) {
            String item = part.strip();
            if (item.isEmpty()) continue;
            try {
                int dash = item.indexOf('-');
                int first = Integer.parseInt((dash < 0 ? item : item.substring(0, dash)).strip());
                int last = dash < 0
                        ? first
                        : Integer.parseInt(item.substring(dash + 1).strip());
                if (first < 0 || last < first) throw new NumberFormatException();
                cpus.set(first, last + 1);
            } catch (NumberFormatException invalid) {
                throw new IllegalArgumentException(
                        "euhedral.inference.worker-cpus has an invalid processor ID or range: " + item);
            }
        }
        if (cpus.isEmpty()) throw new IllegalArgumentException("euhedral.inference.worker-cpus selects no processors");
        return cpus;
    }

    private static void require(Object value, String property) {
        if (value == null) throw new IllegalArgumentException("euhedral.inference." + property + " must be set");
    }
}
