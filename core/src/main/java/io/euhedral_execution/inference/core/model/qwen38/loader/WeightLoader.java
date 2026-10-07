package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightLoadException;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactHeader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/// Loads the runtime objects of a compact Qwen artifact and assembles the declared model structure.
public final class WeightLoader {

    private WeightLoader() {}

    public static Weights load(Path artifactPath, Artifact artifact, GpuMemory gpuMemory) throws IOException {
        return load(artifactPath, artifact, gpuMemory, ArtifactProfile.Speculation.NONE, Set.of());
    }

    /// Loads a compact artifact's objects, keeping those named in `hostBacked` in pinned host memory;
    /// `speculation` also loads the drafter it uses: the MTP layer and draft head, or the DFlash2 drafter.
    public static Weights load(
            Path artifactPath,
            Artifact artifact,
            GpuMemory gpuMemory,
            ArtifactProfile.Speculation speculation,
            Set<String> hostBacked)
            throws IOException {
        Objects.requireNonNull(artifactPath, "artifactPath");
        Objects.requireNonNull(artifact, "artifact");
        Objects.requireNonNull(gpuMemory, "gpuMemory");
        requireCompact(artifact);
        requireConfig(artifact);
        return CompactWeightLoader.load(
                artifactPath,
                artifact,
                gpuMemory,
                indexDescriptors(requireDescriptors(artifact)),
                speculation,
                hostBacked);
    }

    /// Loads the actual embedding and model-declared GDN layer zero without uploading later layers.
    ///
    /// The returned `Weights` retains the source config and a one-entry array containing only
    /// the actual layer-zero weights; it is intended for first-layer execution, not general inference.
    public static Weights loadFirstLayer(Path artifactPath, Artifact artifact, GpuMemory gpuMemory) throws IOException {
        Objects.requireNonNull(artifactPath, "artifactPath");
        Objects.requireNonNull(artifact, "artifact");
        Objects.requireNonNull(gpuMemory, "gpuMemory");
        requireCompact(artifact);
        return CompactWeightLoader.loadFirstLayer(
                artifactPath, requireConfig(artifact), gpuMemory, indexDescriptors(requireDescriptors(artifact)));
    }

    private static void requireCompact(Artifact artifact) throws WeightLoadException {
        if (artifact.header() == null || artifact.header().version() != ArtifactHeader.COMPACT_VERSION) {
            throw new WeightLoadException(
                    "Qwen artifact is not a compact version " + ArtifactHeader.COMPACT_VERSION + " artifact");
        }
    }

    private static Qwen38Config requireConfig(Artifact artifact) throws WeightLoadException {
        if (artifact.config() == null) {
            throw new WeightLoadException("Qwen artifact config is missing");
        }
        return artifact.config();
    }

    private static TensorDescriptor[] requireDescriptors(Artifact artifact) throws WeightLoadException {
        if (artifact.tensors() == null) {
            throw new WeightLoadException("Qwen artifact tensor table is missing");
        }
        return artifact.tensors();
    }

    private static Map<String, TensorDescriptor> indexDescriptors(TensorDescriptor[] descriptors)
            throws WeightLoadException {
        Map<String, TensorDescriptor> byName = new LinkedHashMap<>();
        for (TensorDescriptor descriptor : descriptors) {
            if (descriptor == null
                    || descriptor.name() == null
                    || descriptor.name().isBlank()) {
                throw new WeightLoadException("Qwen artifact contains a tensor with no name");
            }
            if (byName.putIfAbsent(descriptor.name(), descriptor) != null) {
                throw new WeightLoadException(
                        "Qwen artifact contains duplicate tensor name '" + descriptor.name() + "'");
            }
        }
        return byName;
    }
}
