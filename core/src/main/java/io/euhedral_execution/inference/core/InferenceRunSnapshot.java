package io.euhedral_execution.inference.core;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;

/// Immutable record of one engine's effective experiment inputs, suitable for stable JSON.
///
/// Worker processor IDs appear only under `configuration`, in ascending order. `workerCoreIds` are the
/// Euhedral core IDs derived from them at load; one lattice worker runs per core. These are the
/// validated processors handed to Euhedral, not a live observation of active workers. Identity
/// values are measured once at load, outside any generation. `generation` is null unless supplied.
public record InferenceRunSnapshot(
        int schemaVersion,
        Configuration configuration,
        List<Integer> workerCoreIds,
        Model model,
        GenerationConfig generation,
        RuntimeIdentity runtime) {
    /// Version 3 replaced the tuning axes with `configuration`: the engine derives its policy from the artifact.
    public static final int SCHEMA_VERSION = 5;
    /// Explicit value for identity the runtime does not expose.
    public static final String UNAVAILABLE = "unavailable";

    private static final ObjectMapper JSON = new ObjectMapper();

    public InferenceRunSnapshot {
        Objects.requireNonNull(configuration, "configuration");
        workerCoreIds = List.copyOf(workerCoreIds);
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(runtime, "runtime");
    }

    /// Returns deterministic JSON: fields in declaration order and IDs ascending.
    public String toJson() {
        try {
            return JSON.writeValueAsString(this);
        } catch (JsonProcessingException failure) {
            throw new IllegalStateException("snapshot is not serializable", failure);
        }
    }

    /// The inputs the engine was loaded with and the policy it derived from the artifact.
    /// `artifact` is [ModelDescription#artifactName()], null when no artifact was profiled.
    /// `prefixCacheBytes` is the pinned host memory configured for the prefix cache (0 when off). `speculation` is
    /// the speculative strategy the artifact selects (`none`, `mtp`, `dflash2`), drafting `speculativeDepth` tokens
    /// per verification.
    public record Configuration(
            List<Integer> workerProcessorIds,
            int maxContextTokens,
            String artifact,
            int speculativeDepth,
            long prefixCacheBytes,
            String speculation) {
        public Configuration {
            workerProcessorIds = List.copyOf(workerProcessorIds);
        }

        public Configuration(
                List<Integer> workerProcessorIds,
                int maxContextTokens,
                String artifact,
                int speculativeDepth,
                long prefixCacheBytes) {
            this(workerProcessorIds, maxContextTokens, artifact, speculativeDepth, prefixCacheBytes, "none");
        }

        public static Configuration of(InferenceConfig config, ModelDescription description) {
            ModelDescription model = description == null ? ModelDescription.NONE : description;
            return new Configuration(
                    ids(config.workerCpus()),
                    config.maxContextTokens(),
                    model.artifactName(),
                    model.speculativeDepth(),
                    config.prefixCacheBytes(),
                    model.speculation());
        }
    }

    /// `artifactBytes` is null when the file size cannot be read; `artifactFormatVersion` is null
    /// when no artifact header was read. No content digest is computed.
    public record Model(String artifactPath, Long artifactBytes, Integer artifactFormatVersion, Dimensions dimensions) {
        public Model {
            Objects.requireNonNull(artifactPath, "artifactPath");
            Objects.requireNonNull(dimensions, "dimensions");
        }
    }

    /// Scalar projection of a model's configuration; layer kinds are counted rather than copying its array.
    public record Dimensions(
            int vocabSize,
            int hiddenSize,
            int numHiddenLayers,
            int fullAttentionLayers,
            int gatedDeltaNetLayers,
            int numAttentionHeads,
            int numKeyValueHeads,
            int attentionHeadDim,
            int intermediateSize,
            int linearNumKeyHeads,
            int linearNumValueHeads,
            int linearKeyHeadDim,
            int linearValueHeadDim,
            int maxPositionEmbeddings) {}

    /// `nativeRuntimeVersion` is [#UNAVAILABLE] because the CUDA library exposes no version query.
    public record RuntimeIdentity(
            String javaVersion,
            String javaVendor,
            String javaVmName,
            String osName,
            String osArch,
            String euhedralCoreVersion,
            String euhedralCoreArtifact,
            String nativeLibraryPath,
            String nativeRuntimeVersion) {
        public RuntimeIdentity {
            javaVersion = orUnavailable(javaVersion);
            javaVendor = orUnavailable(javaVendor);
            javaVmName = orUnavailable(javaVmName);
            osName = orUnavailable(osName);
            osArch = orUnavailable(osArch);
            euhedralCoreVersion = orUnavailable(euhedralCoreVersion);
            euhedralCoreArtifact = orUnavailable(euhedralCoreArtifact);
            nativeLibraryPath = orUnavailable(nativeLibraryPath);
            nativeRuntimeVersion = orUnavailable(nativeRuntimeVersion);
        }
    }

    static List<Integer> ids(BitSet set) {
        List<Integer> ids = new ArrayList<>(set.cardinality());
        set.stream().forEach(ids::add);
        return ids;
    }

    private static String orUnavailable(String value) {
        return value == null || value.isBlank() ? UNAVAILABLE : value;
    }
}
