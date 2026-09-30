package io.euhedral_execution.inference.core;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.core.gpu.Q3DispatchMode;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;

/// Immutable record of one engine's effective experiment inputs, suitable for stable JSON.
///
/// Worker processor IDs appear only under `tuning`, in ascending order. `workerCoreIds` are the
/// Euhedral core IDs derived from them at load; one lattice worker runs per core. These are the
/// validated processors handed to Euhedral, not a live observation of active workers. Identity
/// values are measured once at load, outside any generation. `generation` is null unless supplied.
public record InferenceRunSnapshot(
        int schemaVersion,
        Tuning tuning,
        List<Integer> workerCoreIds,
        Model model,
        GenerationConfig generation,
        RuntimeIdentity runtime) {
    /// Version 2 dropped `tuning.gpuExecutionMode`: every run submits asynchronously.
    public static final int SCHEMA_VERSION = 2;
    /// Explicit value for identity the runtime does not expose.
    public static final String UNAVAILABLE = "unavailable";

    private static final ObjectMapper JSON = new ObjectMapper();

    public InferenceRunSnapshot {
        Objects.requireNonNull(tuning, "tuning");
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

    /// Version-1 records carried `gpuExecutionMode`; it is ignored when they are read.
    @JsonIgnoreProperties({"gpuExecutionMode"})
    public record Tuning(
            List<Integer> workerProcessorIds,
            int prefillChunkTokens,
            Q3DispatchMode q3DispatchMode,
            int q3SmallRowThreshold) {
        public Tuning {
            workerProcessorIds = List.copyOf(workerProcessorIds);
            q3DispatchMode = q3DispatchMode == null ? Q3DispatchMode.SCALAR : q3DispatchMode;
        }

        public Tuning(List<Integer> workerProcessorIds, int prefillChunkTokens) {
            this(
                    workerProcessorIds,
                    prefillChunkTokens,
                    Q3DispatchMode.AUTO,
                    Q3DispatchMode.DEFAULT_SMALL_ROW_THRESHOLD);
        }

        public static Tuning of(InferenceTuning tuning) {
            return new Tuning(
                    ids(tuning.workerProcessorIds()),
                    tuning.prefillChunkTokens(),
                    tuning.q3DispatchMode(),
                    tuning.q3SmallRowThreshold());
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

    /// Scalar projection of [QwenConfig]; layer kinds are counted rather than copying its array.
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
            int maxPositionEmbeddings) {
        public static Dimensions of(QwenConfig config) {
            int attention = 0;
            int gdn = 0;
            QwenLayerType[] layers = config.layerTypes();
            if (layers != null) {
                for (QwenLayerType layer : layers) {
                    if (layer == QwenLayerType.FULL_ATTENTION) attention++;
                    else if (layer == QwenLayerType.GATED_DELTA_NET) gdn++;
                }
            }
            return new Dimensions(
                    config.vocabSize(),
                    config.hiddenSize(),
                    config.numHiddenLayers(),
                    attention,
                    gdn,
                    config.numAttentionHeads(),
                    config.numKeyValueHeads(),
                    config.attentionHeadDim(),
                    config.intermediateSize(),
                    config.linearNumKeyHeads(),
                    config.linearNumValueHeads(),
                    config.linearKeyHeadDim(),
                    config.linearValueHeadDim(),
                    config.maxPositionEmbeddings());
        }
    }

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
