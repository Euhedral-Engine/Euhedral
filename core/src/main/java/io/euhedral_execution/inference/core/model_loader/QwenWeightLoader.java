package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactHeader;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/// Loads the runtime objects of a compact Qwen artifact and assembles the declared model structure.
public final class QwenWeightLoader {

    private QwenWeightLoader() {}

    public static QwenWeights load(Path artifactPath, QwenArtifact artifact, GpuMemory gpuMemory) throws IOException {
        return load(artifactPath, artifact, gpuMemory, ArtifactProfile.Speculation.NONE, Set.of());
    }

    /// Loads a compact artifact's objects, keeping those named in `hostBacked` in pinned host memory;
    /// `speculation` also loads the drafter it uses: the MTP layer and draft head, or the DFlash2 drafter.
    public static QwenWeights load(
            Path artifactPath,
            QwenArtifact artifact,
            GpuMemory gpuMemory,
            ArtifactProfile.Speculation speculation,
            Set<String> hostBacked)
            throws IOException {
        Objects.requireNonNull(artifactPath, "artifactPath");
        Objects.requireNonNull(artifact, "artifact");
        Objects.requireNonNull(gpuMemory, "gpuMemory");
        requireCompact(artifact);
        requireConfig(artifact);
        return QwenCompactWeightLoader.load(
                artifactPath,
                artifact,
                gpuMemory,
                indexDescriptors(requireDescriptors(artifact)),
                speculation,
                hostBacked);
    }

    /// Loads the actual embedding and model-declared GDN layer zero without uploading later layers.
    ///
    /// The returned `QwenWeights` retains the source config and a one-entry array containing only
    /// the actual layer-zero weights; it is intended for first-layer execution, not general inference.
    public static QwenWeights loadFirstLayer(Path artifactPath, QwenArtifact artifact, GpuMemory gpuMemory)
            throws IOException {
        Objects.requireNonNull(artifactPath, "artifactPath");
        Objects.requireNonNull(artifact, "artifact");
        Objects.requireNonNull(gpuMemory, "gpuMemory");
        requireCompact(artifact);
        return QwenCompactWeightLoader.loadFirstLayer(
                artifactPath, requireConfig(artifact), gpuMemory, indexDescriptors(requireDescriptors(artifact)));
    }

    private static void requireCompact(QwenArtifact artifact) throws QwenWeightLoadException {
        if (artifact.header() == null || artifact.header().version() != QwenArtifactHeader.COMPACT_VERSION) {
            throw new QwenWeightLoadException(
                    "Qwen artifact is not a compact version " + QwenArtifactHeader.COMPACT_VERSION + " artifact");
        }
    }

    private static QwenConfig requireConfig(QwenArtifact artifact) throws QwenWeightLoadException {
        if (artifact.config() == null) {
            throw new QwenWeightLoadException("Qwen artifact config is missing");
        }
        return artifact.config();
    }

    private static TensorDescriptor[] requireDescriptors(QwenArtifact artifact) throws QwenWeightLoadException {
        if (artifact.tensors() == null) {
            throw new QwenWeightLoadException("Qwen artifact tensor table is missing");
        }
        return artifact.tensors();
    }

    private static Map<String, TensorDescriptor> indexDescriptors(TensorDescriptor[] descriptors)
            throws QwenWeightLoadException {
        Map<String, TensorDescriptor> byName = new LinkedHashMap<>();
        for (TensorDescriptor descriptor : descriptors) {
            if (descriptor == null
                    || descriptor.name() == null
                    || descriptor.name().isBlank()) {
                throw new QwenWeightLoadException("Qwen artifact contains a tensor with no name");
            }
            if (byName.putIfAbsent(descriptor.name(), descriptor) != null) {
                throw new QwenWeightLoadException(
                        "Qwen artifact contains duplicate tensor name '" + descriptor.name() + "'");
            }
        }
        return byName;
    }
}
