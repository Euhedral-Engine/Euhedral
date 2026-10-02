package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactDenseFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactMtpAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenMtpWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/// Assembles the canonical Qwen model view from NInfer-compatible compact runtime objects.
///
/// The source Hugging Face names are intentionally absent here. Fused objects remain one
/// `TensorHandle`: consumers must use the descriptor's runtime layout rather than reconstructing
/// source projections at load time.
final class QwenCompactWeightLoader {

    private static final int EXPECTED_OBJECT_COUNT = 1_118;
    private static final int EXPECTED_VISION_OBJECT_COUNT = 333;

    private QwenCompactWeightLoader() {}

    static QwenWeights load(
            Path artifactPath, QwenArtifact artifact, GpuMemory gpuMemory, Map<String, TensorDescriptor> descriptors)
            throws IOException {
        QwenConfig config = artifact.config();
        validateConfig(config);
        validateInventory(config, descriptors);

        Map<String, TensorHandle> handles = new LinkedHashMap<>();
        try {
            for (TensorDescriptor descriptor : descriptors.values()) {
                handles.put(descriptor.name(), TensorLoader.load(artifactPath, descriptor, gpuMemory));
            }
            return assemble(config, handles);
        } catch (Throwable failure) {
            freeAll(handles.values(), gpuMemory, failure);
            return propagate(failure);
        }
    }

    static QwenWeights loadFirstLayer(
            Path artifactPath, QwenConfig config, GpuMemory gpuMemory, Map<String, TensorDescriptor> descriptors)
            throws IOException {
        validateConfig(config);
        validateInventory(config, descriptors);
        if (config.layerTypes()[0] != QwenLayerType.GATED_DELTA_NET) {
            throw new QwenWeightLoadException("actual layer zero is not a compact GDN layer");
        }

        String prefix = "text/layers/0";
        String[] selectedNames = {
            "text/token_embedding",
            prefix + "/input_norm",
            prefix + "/post_attention_norm",
            prefix + "/gdn/a_log",
            prefix + "/gdn/dt_bias",
            prefix + "/gdn/convolution",
            prefix + "/gdn/a_projection",
            prefix + "/gdn/b_projection",
            prefix + "/gdn/query_key",
            prefix + "/gdn/value_z",
            prefix + "/gdn/norm",
            prefix + "/gdn/output",
            prefix + "/mlp/gate_up",
            prefix + "/mlp/down"
        };

        Map<String, TensorHandle> handles = new LinkedHashMap<>();
        try {
            for (String name : selectedNames) {
                TensorDescriptor descriptor = descriptors.get(name);
                if (descriptor == null) {
                    throw new QwenWeightLoadException("compact artifact is missing runtime object '" + name + "'");
                }
                handles.put(name, TensorLoader.load(artifactPath, descriptor, gpuMemory));
            }

            QwenLayerWeights layer = new QwenLayerWeights(
                    0,
                    take(handles, prefix + "/input_norm"),
                    take(handles, prefix + "/post_attention_norm"),
                    buildGdn(handles, prefix),
                    new QwenCompactDenseFfnWeights(
                            take(handles, prefix + "/mlp/gate_up"), take(handles, prefix + "/mlp/down")));
            return new QwenWeights(
                    config,
                    take(handles, "text/token_embedding"),
                    new QwenLayerWeights[] {layer},
                    null,
                    null,
                    null,
                    handles);
        } catch (Throwable failure) {
            freeAll(handles.values(), gpuMemory, failure);
            return propagate(failure);
        }
    }

    static QwenWeights assemble(QwenConfig config, Map<String, TensorHandle> handles) throws QwenWeightLoadException {
        TensorHandle tokenEmbedding = take(handles, "text/token_embedding");
        TensorHandle finalNorm = take(handles, "text/final_norm");
        TensorHandle lmHead = take(handles, "text/output_head");

        QwenLayerWeights[] layers = new QwenLayerWeights[config.numHiddenLayers()];
        for (int index = 0; index < layers.length; index++) {
            String prefix = "text/layers/" + index;
            TensorHandle inputNorm = take(handles, prefix + "/input_norm");
            TensorHandle postNorm = take(handles, prefix + "/post_attention_norm");
            var mixer = config.layerTypes()[index] == QwenLayerType.FULL_ATTENTION
                    ? buildAttention(handles, prefix)
                    : buildGdn(handles, prefix);
            var ffn = new QwenCompactDenseFfnWeights(
                    take(handles, prefix + "/mlp/gate_up"), take(handles, prefix + "/mlp/down"));
            layers[index] = new QwenLayerWeights(index, inputNorm, postNorm, mixer, ffn);
        }

        QwenMtpWeights mtp = config.mtpLayerCount() == 0 ? null : buildMtp(handles);
        return new QwenWeights(config, tokenEmbedding, layers, finalNorm, lmHead, mtp, handles);
    }

    private static QwenCompactAttentionWeights buildAttention(Map<String, TensorHandle> handles, String prefix)
            throws QwenWeightLoadException {
        return new QwenCompactAttentionWeights(
                take(handles, prefix + "/attention/query_key"),
                take(handles, prefix + "/attention/gate_value"),
                take(handles, prefix + "/attention/query_norm"),
                take(handles, prefix + "/attention/key_norm"),
                take(handles, prefix + "/attention/output"));
    }

    private static QwenCompactGatedDeltaNetWeights buildGdn(Map<String, TensorHandle> handles, String prefix)
            throws QwenWeightLoadException {
        return new QwenCompactGatedDeltaNetWeights(
                take(handles, prefix + "/gdn/a_log"),
                take(handles, prefix + "/gdn/dt_bias"),
                take(handles, prefix + "/gdn/convolution"),
                take(handles, prefix + "/gdn/a_projection"),
                take(handles, prefix + "/gdn/b_projection"),
                take(handles, prefix + "/gdn/query_key"),
                take(handles, prefix + "/gdn/value_z"),
                take(handles, prefix + "/gdn/norm"),
                take(handles, prefix + "/gdn/output"));
    }

    private static QwenMtpWeights buildMtp(Map<String, TensorHandle> handles) throws QwenWeightLoadException {
        String prefix = "mtp/layer";
        QwenLayerWeights layer = new QwenLayerWeights(
                0,
                take(handles, prefix + "/input_norm"),
                take(handles, prefix + "/post_attention_norm"),
                new QwenCompactMtpAttentionWeights(
                        take(handles, prefix + "/attention/query_key_gate_value"),
                        take(handles, prefix + "/attention/query_norm"),
                        take(handles, prefix + "/attention/key_norm"),
                        take(handles, prefix + "/attention/output")),
                new QwenCompactDenseFfnWeights(
                        take(handles, prefix + "/mlp/gate_up"), take(handles, prefix + "/mlp/down")));
        return new QwenMtpWeights(
                take(handles, "mtp/embedding_norm"),
                take(handles, "mtp/hidden_norm"),
                take(handles, "mtp/input_projection"),
                layer,
                take(handles, "mtp/final_norm"));
    }

    /// The text inventory is required. The vision tower is optional as a whole: the NVFP4 profile omits
    /// it, the compact Q3 profile carries all of it.
    static void validateInventory(QwenConfig config, Map<String, TensorDescriptor> descriptors)
            throws QwenWeightLoadException {
        int textObjects = EXPECTED_OBJECT_COUNT - EXPECTED_VISION_OBJECT_COUNT;
        if (descriptors.size() != EXPECTED_OBJECT_COUNT && descriptors.size() != textObjects) {
            throw new QwenWeightLoadException("compact Qwen artifact must contain " + EXPECTED_OBJECT_COUNT
                    + " runtime objects, or " + textObjects + " without the vision tower, found "
                    + descriptors.size());
        }
        Set<String> expected = new LinkedHashSet<>();
        expected.add("text/token_embedding");
        expected.add("text/final_norm");
        expected.add("text/output_head");
        expected.add("text/draft_head");
        expected.add("text/draft_head_token_ids");
        for (int index = 0; index < config.numHiddenLayers(); index++) {
            String prefix = "text/layers/" + index;
            expected.add(prefix + "/input_norm");
            expected.add(prefix + "/post_attention_norm");
            expected.add(prefix + "/mlp/gate_up");
            expected.add(prefix + "/mlp/down");
            if (config.layerTypes()[index] == QwenLayerType.FULL_ATTENTION) {
                expected.add(prefix + "/attention/query_key");
                expected.add(prefix + "/attention/gate_value");
                expected.add(prefix + "/attention/query_norm");
                expected.add(prefix + "/attention/key_norm");
                expected.add(prefix + "/attention/output");
            } else {
                expected.add(prefix + "/gdn/a_log");
                expected.add(prefix + "/gdn/dt_bias");
                expected.add(prefix + "/gdn/convolution");
                expected.add(prefix + "/gdn/a_projection");
                expected.add(prefix + "/gdn/b_projection");
                expected.add(prefix + "/gdn/query_key");
                expected.add(prefix + "/gdn/value_z");
                expected.add(prefix + "/gdn/norm");
                expected.add(prefix + "/gdn/output");
            }
        }
        if (config.mtpLayerCount() > 0) {
            expected.add("mtp/input_projection");
            expected.add("mtp/embedding_norm");
            expected.add("mtp/hidden_norm");
            expected.add("mtp/final_norm");
            expected.add("mtp/layer/input_norm");
            expected.add("mtp/layer/post_attention_norm");
            expected.add("mtp/layer/attention/query_key_gate_value");
            expected.add("mtp/layer/attention/query_norm");
            expected.add("mtp/layer/attention/key_norm");
            expected.add("mtp/layer/attention/output");
            expected.add("mtp/layer/mlp/gate_up");
            expected.add("mtp/layer/mlp/down");
        }
        for (String name : expected) {
            TensorDescriptor descriptor = descriptors.get(name);
            if (descriptor == null) {
                throw new QwenWeightLoadException("compact artifact is missing runtime object '" + name + "'");
            }
            validateExpectedDescriptor(descriptor);
        }
        int visionCount = 0;
        Set<String> visionNames = new LinkedHashSet<>();
        for (String name : descriptors.keySet()) {
            if (name.startsWith("vision/")) {
                visionCount++;
                visionNames.add(name);
            } else if (!expected.contains(name)) {
                throw new QwenWeightLoadException("compact artifact contains unknown runtime object '" + name + "'");
            }
        }
        if (visionCount == 0) return;
        if (visionCount != EXPECTED_VISION_OBJECT_COUNT) {
            throw new QwenWeightLoadException("compact artifact must contain " + EXPECTED_VISION_OBJECT_COUNT
                    + " vision runtime objects, found " + visionCount);
        }
        Set<String> expectedVisionNames = expectedVisionNames();
        if (!visionNames.equals(expectedVisionNames)) {
            Set<String> missing = new LinkedHashSet<>(expectedVisionNames);
            missing.removeAll(visionNames);
            Set<String> unexpected = new LinkedHashSet<>(visionNames);
            unexpected.removeAll(expectedVisionNames);
            throw new QwenWeightLoadException("compact vision inventory differs from the registered reference; missing="
                    + missing.stream().findFirst().orElse("none")
                    + ", unexpected=" + unexpected.stream().findFirst().orElse("none"));
        }
        for (String name : visionNames) {
            validateExpectedDescriptor(descriptors.get(name));
        }
    }

    private static void validateExpectedDescriptor(TensorDescriptor descriptor) throws QwenWeightLoadException {
        String name = descriptor.name();
        if (descriptor.layout() == null) {
            throw new QwenWeightLoadException("runtime object has no persistent layout: " + name);
        }
        boolean quantized = descriptor.format() == WeightFormat.Q3_G64_FP16
                || descriptor.format() == WeightFormat.Q4_G64_FP16
                || descriptor.format() == WeightFormat.Q5_G64_FP16
                || descriptor.format() == WeightFormat.Q6_G64_FP16
                || descriptor.format() == WeightFormat.W8_G32_FP16
                || descriptor.format() == WeightFormat.NVFP4;
        boolean p2e2 = descriptor.layout() == WeightLayout.ROW_SPLIT_P2E2_V1
                && descriptor.format() == WeightFormat.Q3_G64_FP16;
        if (quantized && descriptor.layout() != WeightLayout.ROW_SPLIT_K128_V1 && !p2e2) {
            throw new QwenWeightLoadException("quantized runtime object has unsupported layout: " + name);
        }
        if (!quantized && descriptor.layout() != WeightLayout.CONTIGUOUS_LE_V1) {
            throw new QwenWeightLoadException("direct runtime object has unsupported layout: " + name);
        }
        Expected expected = expectedDescriptor(name);
        if (expected != null
                && (!java.util.Arrays.equals(expected.shape(), descriptor.shape())
                        || expected.dataType() != descriptor.dataType()
                        || !(expected.format() == descriptor.format()
                                || (descriptor.format() == WeightFormat.NVFP4
                                        && expected.layout() == WeightLayout.ROW_SPLIT_K128_V1))
                        || !(expected.layout() == descriptor.layout()
                                || (p2e2 && expected.layout() == WeightLayout.ROW_SPLIT_K128_V1)))) {
            throw new QwenWeightLoadException(
                    "compact runtime metadata conflicts with the registered layout for '" + name + "'");
        }
    }

    static Expected expectedDescriptor(String name) {
        if (name.equals("vision/patch_embedding")) {
            return quantized(new long[] {1152, 1536}, TensorDataType.BF16, WeightFormat.Q6_G64_FP16);
        }
        if (name.equals("vision/patch_embedding_bias")) {
            return direct(new long[] {1152}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.equals("vision/position_embedding")) {
            return direct(new long[] {2304, 1152}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.startsWith("vision/layers/") && name.endsWith("/attention/qkv")) {
            return quantized(new long[] {3456, 1152}, TensorDataType.BF16, WeightFormat.Q4_G64_FP16);
        }
        if (name.startsWith("vision/layers/") && name.endsWith("/attention/qkv_bias")) {
            return direct(new long[] {3456}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.startsWith("vision/layers/") && name.endsWith("/attention/output")) {
            return quantized(new long[] {1152, 1152}, TensorDataType.BF16, WeightFormat.Q5_G64_FP16);
        }
        if (name.startsWith("vision/layers/") && name.endsWith("/attention/output_bias")) {
            return direct(new long[] {1152}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.startsWith("vision/layers/") && name.endsWith("/mlp/fc1")) {
            return quantized(new long[] {4304, 1152}, TensorDataType.BF16, WeightFormat.Q4_G64_FP16);
        }
        if (name.startsWith("vision/layers/") && name.endsWith("/mlp/fc1_bias")) {
            return direct(new long[] {4304}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.startsWith("vision/layers/") && name.endsWith("/mlp/fc2")) {
            return quantized(new long[] {1152, 4304}, TensorDataType.BF16, WeightFormat.Q5_G64_FP16);
        }
        if (name.startsWith("vision/layers/") && name.endsWith("/mlp/fc2_bias")) {
            return direct(new long[] {1152}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.startsWith("vision/layers/")
                && (name.endsWith("/norm1/weight")
                        || name.endsWith("/norm1/bias")
                        || name.endsWith("/norm2/weight")
                        || name.endsWith("/norm2/bias"))) {
            return direct(new long[] {1152}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.equals("vision/merger/fc1")) {
            return quantized(new long[] {4608, 4608}, TensorDataType.BF16, WeightFormat.W8_G32_FP16);
        }
        if (name.equals("vision/merger/fc1_bias")) {
            return direct(new long[] {4608}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.equals("vision/merger/fc2")) {
            return quantized(new long[] {5120, 4608}, TensorDataType.BF16, WeightFormat.W8_G32_FP16);
        }
        if (name.equals("vision/merger/fc2_bias")) {
            return direct(new long[] {5120}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.equals("vision/merger/norm/weight") || name.equals("vision/merger/norm/bias")) {
            return direct(new long[] {1152}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.equals("text/token_embedding") || name.equals("text/output_head")) {
            return q3(new long[] {248320, 5120});
        }
        if (name.equals("text/final_norm")) {
            return direct(new long[] {5120}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.equals("text/draft_head")) {
            return q3(new long[] {131072, 5120});
        }
        if (name.equals("text/draft_head_token_ids")) {
            return direct(new long[] {131072}, TensorDataType.INT32, WeightFormat.I32);
        }
        if (name.equals("mtp/input_projection")) {
            return q3(new long[] {5120, 10240});
        }
        if (name.startsWith("mtp/")
                && (name.endsWith("embedding_norm")
                        || name.endsWith("hidden_norm")
                        || name.endsWith("final_norm")
                        || name.endsWith("layer/input_norm")
                        || name.endsWith("layer/post_attention_norm"))) {
            return direct(new long[] {5120}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.equals("mtp/layer/attention/query_key_gate_value")) {
            return quantized(new long[] {14336, 5120}, TensorDataType.BF16, WeightFormat.W8_G32_FP16);
        }
        if (name.equals("mtp/layer/attention/query_norm") || name.equals("mtp/layer/attention/key_norm")) {
            return direct(new long[] {256}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.equals("mtp/layer/attention/output")) {
            return q3(new long[] {5120, 6144});
        }
        if (name.equals("mtp/layer/mlp/gate_up")) {
            return q3(new long[] {34816, 5120});
        }
        if (name.equals("mtp/layer/mlp/down")) {
            return q3(new long[] {5120, 17408});
        }
        if (name.startsWith("text/layers/") && name.endsWith("/input_norm")) {
            return direct(new long[] {5120}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.startsWith("text/layers/") && name.endsWith("/post_attention_norm")) {
            return direct(new long[] {5120}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.startsWith("text/layers/") && name.endsWith("/mlp/gate_up")) {
            return q3(new long[] {34816, 5120});
        }
        if (name.startsWith("text/layers/") && name.endsWith("/mlp/down")) {
            return q3(new long[] {5120, 17408});
        }
        if (name.endsWith("/attention/query_key")) {
            return quantized(new long[] {7168, 5120}, TensorDataType.BF16, WeightFormat.Q4_G64_FP16);
        }
        if (name.endsWith("/attention/gate_value")) {
            return quantized(new long[] {7168, 5120}, TensorDataType.BF16, WeightFormat.Q5_G64_FP16);
        }
        if (name.endsWith("/attention/query_norm") || name.endsWith("/attention/key_norm")) {
            return direct(new long[] {256}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.endsWith("/attention/output")) {
            return q3(new long[] {5120, 6144});
        }
        if (name.endsWith("/gdn/a_log") || name.endsWith("/gdn/dt_bias")) {
            return direct(new long[] {48}, TensorDataType.BF16, WeightFormat.FP32);
        }
        if (name.endsWith("/gdn/convolution")) {
            return direct(new long[] {4, 10240}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.endsWith("/gdn/a_projection") || name.endsWith("/gdn/b_projection")) {
            return direct(new long[] {48, 5120}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.endsWith("/gdn/query_key")) {
            return quantized(new long[] {4096, 5120}, TensorDataType.BF16, WeightFormat.Q4_G64_FP16);
        }
        if (name.endsWith("/gdn/value_z")) {
            return quantized(new long[] {12288, 5120}, TensorDataType.BF16, WeightFormat.Q5_G64_FP16);
        }
        if (name.endsWith("/gdn/norm")) {
            return direct(new long[] {128}, TensorDataType.BF16, WeightFormat.BF16);
        }
        if (name.endsWith("/gdn/output")) {
            return q3(new long[] {5120, 6144});
        }
        return null;
    }

    private static Set<String> expectedVisionNames() {
        Set<String> names = new LinkedHashSet<>();
        names.add("vision/patch_embedding");
        names.add("vision/patch_embedding_bias");
        names.add("vision/position_embedding");
        for (int layer = 0; layer < 27; layer++) {
            String prefix = "vision/layers/" + layer + "/";
            names.add(prefix + "attention/qkv");
            names.add(prefix + "attention/qkv_bias");
            names.add(prefix + "attention/output");
            names.add(prefix + "attention/output_bias");
            names.add(prefix + "mlp/fc1");
            names.add(prefix + "mlp/fc1_bias");
            names.add(prefix + "mlp/fc2");
            names.add(prefix + "mlp/fc2_bias");
            names.add(prefix + "norm1/weight");
            names.add(prefix + "norm1/bias");
            names.add(prefix + "norm2/weight");
            names.add(prefix + "norm2/bias");
        }
        names.add("vision/merger/fc1");
        names.add("vision/merger/fc1_bias");
        names.add("vision/merger/fc2");
        names.add("vision/merger/fc2_bias");
        names.add("vision/merger/norm/weight");
        names.add("vision/merger/norm/bias");
        return names;
    }

    private static Expected q3(long[] shape) {
        return quantized(shape, TensorDataType.BF16, WeightFormat.Q3_G64_FP16);
    }

    private static Expected quantized(long[] shape, TensorDataType dataType, WeightFormat format) {
        return new Expected(shape, dataType, format, WeightLayout.ROW_SPLIT_K128_V1);
    }

    private static Expected direct(long[] shape, TensorDataType dataType, WeightFormat format) {
        return new Expected(shape, dataType, format, WeightLayout.CONTIGUOUS_LE_V1);
    }

    record Expected(long[] shape, TensorDataType dataType, WeightFormat format, WeightLayout layout) {}

    private static TensorHandle take(Map<String, TensorHandle> handles, String name) throws QwenWeightLoadException {
        TensorHandle handle = handles.get(name);
        if (handle == null) {
            throw new QwenWeightLoadException("compact artifact is missing loaded runtime object '" + name + "'");
        }
        return handle;
    }

    private static void validateConfig(QwenConfig config) throws QwenWeightLoadException {
        if (config == null
                || config.layerTypes() == null
                || config.layerTypes().length != 64
                || config.numHiddenLayers() != 64
                || config.mtpLayerCount() > 1) {
            throw new QwenWeightLoadException("compact Qwen artifact has unsupported model topology");
        }
        if (config.numExperts() != 0) {
            throw new QwenWeightLoadException("compact Q3 loader only supports the dense Qwen reference topology");
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
}
