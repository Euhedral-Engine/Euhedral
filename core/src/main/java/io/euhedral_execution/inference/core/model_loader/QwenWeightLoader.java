package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactHeader;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenDenseFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenExpertWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenMixerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenMoeWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenMtpWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import java.io.IOException;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/// Loads every tensor in a Qwen artifact and assembles the declared model structure.
public final class QwenWeightLoader {

    private static final String LEGACY_MODEL_PREFIX = "model";
    private static final String LANGUAGE_MODEL_PREFIX = "model.language_model";

    private QwenWeightLoader() {}

    public static QwenWeights load(Path artifactPath, QwenArtifact artifact, GpuMemory gpuMemory) throws IOException {
        return load(artifactPath, artifact, gpuMemory, WeightResidency.ALL);
    }

    /// Loads with the given residency; [WeightResidency#EXECUTED] applies to compact artifacts.
    public static QwenWeights load(
            Path artifactPath, QwenArtifact artifact, GpuMemory gpuMemory, WeightResidency residency)
            throws IOException {
        Objects.requireNonNull(artifactPath, "artifactPath");
        Objects.requireNonNull(artifact, "artifact");
        Objects.requireNonNull(gpuMemory, "gpuMemory");

        QwenConfig config = requireConfig(artifact);
        validateConfig(config);
        TensorDescriptor[] descriptors = requireDescriptors(artifact);
        Map<String, TensorDescriptor> descriptorsByName = indexDescriptors(descriptors);
        if (isCompactArtifact(artifact)) {
            return QwenCompactWeightLoader.load(artifactPath, artifact, gpuMemory, descriptorsByName, residency);
        }
        String modelPrefix = modelPrefix(descriptorsByName.keySet());
        Set<String> requiredNames = requiredNames(config, modelPrefix);
        requireAllNamesPresent(requiredNames, descriptorsByName.keySet());

        Map<String, TensorHandle> handles = new LinkedHashMap<>();
        try {
            for (TensorDescriptor descriptor : descriptors) {
                handles.put(descriptor.name(), TensorLoader.load(artifactPath, descriptor, gpuMemory));
            }

            Assembly assembly = new Assembly(handles);
            TensorHandle tokenEmbedding = assembly.take(modelPrefix + ".embed_tokens.weight");
            TensorHandle finalNorm = assembly.take(modelPrefix + ".norm.weight");
            TensorHandle lmHead = assembly.take("lm_head.weight");

            QwenLayerWeights[] layers = new QwenLayerWeights[config.numHiddenLayers()];
            for (int index = 0; index < layers.length; index++) {
                layers[index] = buildLayer(
                        assembly, config, modelPrefix + ".layers." + index, index, config.layerTypes()[index]);
            }

            QwenMtpWeights mtp = config.mtpLayerCount() == 0 ? null : buildMtp(assembly, config);
            assembly.requireFullyConsumed();
            return new QwenWeights(config, tokenEmbedding, layers, finalNorm, lmHead, mtp);
        } catch (Throwable failure) {
            freeAll(handles.values(), gpuMemory, failure);
            return propagate(failure);
        }
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
        if (!isCompactArtifact(artifact)) {
            throw new QwenWeightLoadException("first-layer loading requires a compact Qwen artifact");
        }
        QwenConfig config = requireConfig(artifact);
        validateConfig(config);
        return QwenCompactWeightLoader.loadFirstLayer(
                artifactPath, config, gpuMemory, indexDescriptors(requireDescriptors(artifact)));
    }

    static boolean isCompactArtifact(QwenArtifact artifact) {
        return artifact.header() != null && artifact.header().version() == QwenArtifactHeader.COMPACT_VERSION;
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

    private static String modelPrefix(Set<String> names) throws QwenWeightLoadException {
        if (names.contains("model.language_model.embed_tokens.weight")) {
            return LANGUAGE_MODEL_PREFIX;
        }
        if (names.contains("model.embed_tokens.weight")) {
            return LEGACY_MODEL_PREFIX;
        }
        throw new QwenWeightLoadException(
                "Qwen artifact is missing required token embedding tensor 'model.embed_tokens.weight' "
                        + "or 'model.language_model.embed_tokens.weight'");
    }

    private static Set<String> requiredNames(QwenConfig config, String modelPrefix) {
        Set<String> names = new LinkedHashSet<>();
        names.add(modelPrefix + ".embed_tokens.weight");
        names.add(modelPrefix + ".norm.weight");
        names.add("lm_head.weight");
        for (int index = 0; index < config.numHiddenLayers(); index++) {
            addLayerNames(names, config, modelPrefix + ".layers." + index, config.layerTypes()[index]);
        }
        if (config.mtpLayerCount() > 0) {
            names.add("mtp.pre_fc_norm_embedding.weight");
            names.add("mtp.pre_fc_norm_hidden.weight");
            names.add("mtp.fc.weight");
            names.add("mtp.norm.weight");
            addLayerNames(names, config, "mtp.layers.0", QwenLayerType.FULL_ATTENTION);
        }
        return names;
    }

    private static void addLayerNames(Set<String> names, QwenConfig config, String prefix, QwenLayerType layerType) {
        names.add(prefix + ".input_layernorm.weight");
        if (layerType == QwenLayerType.FULL_ATTENTION) {
            names.addAll(attentionNames(prefix));
        } else if (layerType == QwenLayerType.GATED_DELTA_NET) {
            names.addAll(gdnNames(prefix));
        } else {
            throw new IllegalArgumentException("Unsupported Qwen layer type: " + layerType);
        }
        names.add(prefix + ".post_attention_layernorm.weight");
        addFfnNames(names, config, prefix);
    }

    private static Set<String> attentionNames(String prefix) {
        return Set.of(
                prefix + ".self_attn.q_proj.weight",
                prefix + ".self_attn.k_proj.weight",
                prefix + ".self_attn.v_proj.weight",
                prefix + ".self_attn.o_proj.weight",
                prefix + ".self_attn.q_norm.weight",
                prefix + ".self_attn.k_norm.weight");
    }

    private static Set<String> gdnNames(String prefix) {
        return Set.of(
                prefix + ".linear_attn.in_proj_qkv.weight",
                prefix + ".linear_attn.in_proj_z.weight",
                prefix + ".linear_attn.in_proj_b.weight",
                prefix + ".linear_attn.in_proj_a.weight",
                prefix + ".linear_attn.conv1d.weight",
                prefix + ".linear_attn.norm.weight",
                prefix + ".linear_attn.dt_bias",
                prefix + ".linear_attn.A_log",
                prefix + ".linear_attn.out_proj.weight");
    }

    private static void addFfnNames(Set<String> names, QwenConfig config, String prefix) {
        if (config.numExperts() == 0) {
            names.add(prefix + ".mlp.gate_proj.weight");
            names.add(prefix + ".mlp.up_proj.weight");
            names.add(prefix + ".mlp.down_proj.weight");
            return;
        }
        names.add(prefix + ".mlp.gate.weight");
        for (int expert = 0; expert < config.numExperts(); expert++) {
            names.add(prefix + ".mlp.experts." + expert + ".gate_proj.weight");
            names.add(prefix + ".mlp.experts." + expert + ".up_proj.weight");
            names.add(prefix + ".mlp.experts." + expert + ".down_proj.weight");
        }
        if (config.sharedExpertIntermediateSize() > 0) {
            names.add(prefix + ".mlp.shared_expert.gate_proj.weight");
            names.add(prefix + ".mlp.shared_expert.up_proj.weight");
            names.add(prefix + ".mlp.shared_expert.down_proj.weight");
            names.add(prefix + ".mlp.shared_expert_gate.weight");
        }
    }

    private static void requireAllNamesPresent(Set<String> requiredNames, Set<String> availableNames)
            throws QwenWeightLoadException {
        for (String requiredName : requiredNames) {
            if (!availableNames.contains(requiredName)) {
                throw new QwenWeightLoadException("Qwen artifact is missing required tensor '" + requiredName + "'");
            }
        }
    }

    private static QwenLayerWeights buildLayer(
            Assembly assembly, QwenConfig config, String prefix, int index, QwenLayerType layerType)
            throws QwenWeightLoadException {
        TensorHandle inputNorm = assembly.take(prefix + ".input_layernorm.weight");
        TensorHandle postAttentionNorm = assembly.take(prefix + ".post_attention_layernorm.weight");
        QwenMixerWeights mixer = layerType == QwenLayerType.FULL_ATTENTION
                ? buildAttention(assembly, prefix)
                : buildGdn(assembly, prefix);
        return new QwenLayerWeights(index, inputNorm, postAttentionNorm, mixer, buildFfn(assembly, config, prefix));
    }

    private static QwenAttentionWeights buildAttention(Assembly assembly, String prefix)
            throws QwenWeightLoadException {
        return new QwenAttentionWeights(
                assembly.take(prefix + ".self_attn.q_proj.weight"),
                assembly.take(prefix + ".self_attn.k_proj.weight"),
                assembly.take(prefix + ".self_attn.v_proj.weight"),
                assembly.take(prefix + ".self_attn.o_proj.weight"),
                assembly.take(prefix + ".self_attn.q_norm.weight"),
                assembly.take(prefix + ".self_attn.k_norm.weight"));
    }

    private static QwenGatedDeltaNetWeights buildGdn(Assembly assembly, String prefix) throws QwenWeightLoadException {
        return new QwenGatedDeltaNetWeights(
                assembly.take(prefix + ".linear_attn.in_proj_qkv.weight"),
                assembly.take(prefix + ".linear_attn.in_proj_z.weight"),
                assembly.take(prefix + ".linear_attn.in_proj_b.weight"),
                assembly.take(prefix + ".linear_attn.in_proj_a.weight"),
                assembly.take(prefix + ".linear_attn.conv1d.weight"),
                assembly.take(prefix + ".linear_attn.norm.weight"),
                assembly.take(prefix + ".linear_attn.dt_bias"),
                assembly.take(prefix + ".linear_attn.A_log"),
                assembly.take(prefix + ".linear_attn.out_proj.weight"));
    }

    private static QwenFfnWeights buildFfn(Assembly assembly, QwenConfig config, String prefix)
            throws QwenWeightLoadException {
        if (config.numExperts() == 0) {
            return new QwenDenseFfnWeights(
                    assembly.take(prefix + ".mlp.gate_proj.weight"),
                    assembly.take(prefix + ".mlp.up_proj.weight"),
                    assembly.take(prefix + ".mlp.down_proj.weight"));
        }

        QwenExpertWeights[] experts = new QwenExpertWeights[config.numExperts()];
        for (int expert = 0; expert < experts.length; expert++) {
            String expertPrefix = prefix + ".mlp.experts." + expert;
            experts[expert] = new QwenExpertWeights(
                    assembly.take(expertPrefix + ".gate_proj.weight"),
                    assembly.take(expertPrefix + ".up_proj.weight"),
                    assembly.take(expertPrefix + ".down_proj.weight"));
        }
        QwenDenseFfnWeights sharedExpert = null;
        TensorHandle sharedExpertGate = null;
        if (config.sharedExpertIntermediateSize() > 0) {
            String sharedPrefix = prefix + ".mlp.shared_expert";
            sharedExpert = new QwenDenseFfnWeights(
                    assembly.take(sharedPrefix + ".gate_proj.weight"),
                    assembly.take(sharedPrefix + ".up_proj.weight"),
                    assembly.take(sharedPrefix + ".down_proj.weight"));
            sharedExpertGate = assembly.take(prefix + ".mlp.shared_expert_gate.weight");
        }
        return new QwenMoeWeights(assembly.take(prefix + ".mlp.gate.weight"), experts, sharedExpert, sharedExpertGate);
    }

    private static QwenMtpWeights buildMtp(Assembly assembly, QwenConfig config) throws QwenWeightLoadException {
        return new QwenMtpWeights(
                assembly.take("mtp.pre_fc_norm_embedding.weight"),
                assembly.take("mtp.pre_fc_norm_hidden.weight"),
                assembly.take("mtp.fc.weight"),
                buildLayer(assembly, config, "mtp.layers.0", 0, QwenLayerType.FULL_ATTENTION),
                assembly.take("mtp.norm.weight"));
    }

    private static void validateConfig(QwenConfig config) throws QwenWeightLoadException {
        if (config.layerTypes() == null || config.layerTypes().length != config.numHiddenLayers()) {
            throw new QwenWeightLoadException("Qwen config layerTypes length "
                    + (config.layerTypes() == null ? "null" : config.layerTypes().length)
                    + " does not match numHiddenLayers "
                    + config.numHiddenLayers());
        }
        if (config.numHiddenLayers() < 0 || config.mtpLayerCount() < 0 || config.mtpLayerCount() > 1) {
            throw new QwenWeightLoadException("Qwen config has unsupported layer or MTP counts");
        }
        if (config.numExperts() < 0 || config.numExpertsPerToken() < 0) {
            throw new QwenWeightLoadException("Qwen config has negative MoE counts");
        }
        if (config.numExperts() == 0) {
            if (config.numExpertsPerToken() != 0
                    || config.moeIntermediateSize() != 0
                    || config.sharedExpertIntermediateSize() != 0) {
                throw new QwenWeightLoadException("Qwen dense config contains MoE fields");
            }
        } else if (config.numExpertsPerToken() == 0
                || config.numExpertsPerToken() > config.numExperts()
                || config.moeIntermediateSize() <= 0
                || config.sharedExpertIntermediateSize() < 0) {
            throw new QwenWeightLoadException("Qwen MoE config is inconsistent");
        }
        for (QwenLayerType layerType : config.layerTypes()) {
            if (layerType == null) {
                throw new QwenWeightLoadException("Qwen config contains a null layer type");
            }
        }
    }

    private static void freeAll(Iterable<TensorHandle> handles, GpuMemory gpuMemory, Throwable failure) {
        for (TensorHandle handle : handles) {
            try {
                gpuMemory.free(handle.deviceAddress());
            } catch (Throwable cleanupFailure) {
                failure.addSuppressed(cleanupFailure);
            }
        }
    }

    private static QwenWeights propagate(Throwable failure) throws IOException {
        if (failure instanceof IOException exception) {
            throw exception;
        }
        if (failure instanceof RuntimeException exception) {
            throw exception;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        throw new AssertionError(failure);
    }

    private static final class Assembly {

        private final Map<String, TensorHandle> handles;
        private final Set<String> consumed = new HashSet<>();

        private Assembly(Map<String, TensorHandle> handles) {
            this.handles = handles;
        }

        private TensorHandle take(String name) throws QwenWeightLoadException {
            TensorHandle handle = handles.get(name);
            if (handle == null) {
                throw new QwenWeightLoadException("Qwen artifact is missing loaded tensor '" + name + "'");
            }
            consumed.add(name);
            return handle;
        }

        private void requireFullyConsumed() throws QwenWeightLoadException {
            for (String name : handles.keySet()) {
                if (!consumed.contains(name)) {
                    throw new QwenWeightLoadException(
                            "Qwen artifact contains unexpected or unconsumed tensor '" + name + "'");
                }
            }
        }
    }
}
