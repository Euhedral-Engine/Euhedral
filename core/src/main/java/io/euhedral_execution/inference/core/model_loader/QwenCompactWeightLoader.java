package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model_loader.artifact.Nvfp4Layout;
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
        return load(artifactPath, artifact, gpuMemory, descriptors, false, Set.of());
    }

    static QwenWeights load(
            Path artifactPath,
            QwenArtifact artifact,
            GpuMemory gpuMemory,
            Map<String, TensorDescriptor> descriptors,
            boolean speculative,
            Set<String> hostBacked)
            throws IOException {
        QwenConfig config = artifact.config();
        validateConfig(config);
        validateInventory(config, descriptors);

        Map<String, TensorHandle> handles = new LinkedHashMap<>();
        long hostArena = 0;
        try {
            // Host-backed objects share one pinned arena, allocated before any payload is read: huge
            // pages are far likelier to be available then than between reads.
            long hostBytes = 0;
            for (TensorDescriptor descriptor : descriptors.values()) {
                if (uploads(descriptor.name(), speculative) && hostBacked.contains(descriptor.name()))
                    hostBytes += hostSlot(descriptor.byteSize());
            }
            if (hostBytes > 0) hostArena = gpuMemory.allocateHostWeights(hostBytes);
            long hostOffset = 0;
            for (TensorDescriptor descriptor : descriptors.values()) {
                if (!uploads(descriptor.name(), speculative)) continue;
                if (hostBacked.contains(descriptor.name())) {
                    TensorHandle host = TensorLoader.loadToHost(artifactPath, descriptor, hostArena + hostOffset);
                    // The embedding is a gather: kernels read its rows in place instead of staging it.
                    if (descriptor.name().equals(HostWeightSelection.EMBEDDING))
                        host = mapped(host, gpuMemory.hostWeightsDeviceAddress(host.hostAddress()));
                    handles.put(descriptor.name(), host);
                    hostOffset += hostSlot(descriptor.byteSize());
                } else handles.put(descriptor.name(), TensorLoader.load(artifactPath, descriptor, gpuMemory));
            }
            // Only a speculative load prepares the MTP layer for execution.
            if (speculative) splitMtpAttention(handles, gpuMemory);
            return assemble(config, handles);
        } catch (Throwable failure) {
            freeAll(handles.values(), gpuMemory, failure);
            if (hostArena != 0) {
                try {
                    gpuMemory.freeHostWeights(hostArena);
                } catch (Throwable cleanupFailure) {
                    failure.addSuppressed(cleanupFailure);
                }
            }
            return propagate(failure);
        }
    }

    /// Whether a load places the named object on the device: never the vision tower, and the MTP layer
    /// and draft head only for speculative decoding.
    static boolean uploads(String name, boolean speculative) {
        if (name.startsWith("vision/")) return false;
        return speculative || !(name.startsWith("mtp/") || name.startsWith("text/draft_head"));
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

        QwenMtpWeights mtp =
                config.mtpLayerCount() == 0 || !handles.containsKey("mtp/input_projection") ? null : buildMtp(handles);
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

    private static final String MTP_ATTENTION = "mtp/layer/attention/query_key_gate_value";

    /// The MTP layer packs its attention projection as the base layers' query/key rows (q, k) followed by
    /// their gate/value rows (gate, v) (tools/convert_qwen_safetensors_to_compact_edrl.py). An NVFP4
    /// pack is split here into those two tensors, each with its rows' codes and scales and the shared
    /// global scale, so the MTP layer executes as an ordinary attention layer (docs/MTP_CONTRACT.md §1).
    private static void splitMtpAttention(Map<String, TensorHandle> handles, GpuMemory gpu) {
        TensorHandle packed = handles.get(MTP_ATTENTION);
        if (packed == null || packed.hostBacked()) return;
        if (packed.format() == WeightFormat.W8_G32_FP16) {
            requantizeW8MtpAttention(handles, packed, gpu);
            return;
        }
        if (packed.format() != WeightFormat.NVFP4) return;
        long rows = packed.shape()[0], k = packed.shape()[1];
        long half = rows / 2;
        handles.put("mtp/layer/attention/query_key", nvfp4Rows(packed, 0, half, "mtp/layer/attention/query_key", gpu));
        handles.put(
                "mtp/layer/attention/gate_value", nvfp4Rows(packed, half, half, "mtp/layer/attention/gate_value", gpu));
        handles.remove(MTP_ATTENTION);
        gpu.free(packed.deviceAddress());
    }

    /// The compact Q3 artifact stores the MTP attention pack as W8G32 (int8 codes [rows][K], then FP16
    /// scales [rows][K/32] at a 256-aligned offset), for which no linear kernel exists. Each half is
    /// dequantized and re-quantized to NVFP4 (Nvfp4WeightQuantizer). This affects drafts only: verification
    /// by the base model decides every output token.
    private static void requantizeW8MtpAttention(
            Map<String, TensorHandle> handles, TensorHandle packed, GpuMemory gpu) {
        int rows = Math.toIntExact(packed.shape()[0]), k = Math.toIntExact(packed.shape()[1]);
        int padded = (k + 127) / 128 * 128, groups = padded / 32;
        long scales = align256((long) rows * padded);
        float[] values = new float[rows * k];
        try (var arena = java.lang.foreign.Arena.ofConfined()) {
            var host = arena.allocate(packed.byteSize(), 16);
            gpu.copyDeviceToHost(host, packed.deviceAddress(), packed.byteSize());
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < k; col++) {
                    byte code = host.get(java.lang.foreign.ValueLayout.JAVA_BYTE, (long) row * padded + col);
                    short half = host.get(
                            java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED,
                            scales + ((long) row * groups + col / 32) * 2);
                    values[row * k + col] = code * Float.float16ToFloat(half);
                }
            }
        }
        int half = rows / 2;
        handles.put(
                "mtp/layer/attention/query_key",
                uploadNvfp4(
                        java.util.Arrays.copyOfRange(values, 0, half * k),
                        half,
                        k,
                        "mtp/layer/attention/query_key",
                        packed,
                        gpu));
        handles.put(
                "mtp/layer/attention/gate_value",
                uploadNvfp4(
                        java.util.Arrays.copyOfRange(values, half * k, rows * k),
                        half,
                        k,
                        "mtp/layer/attention/gate_value",
                        packed,
                        gpu));
        handles.remove(MTP_ATTENTION);
        gpu.free(packed.deviceAddress());
    }

    private static TensorHandle uploadNvfp4(
            float[] values, int rows, int k, String name, TensorHandle source, GpuMemory gpu) {
        byte[] payload = Nvfp4WeightQuantizer.quantize(values, rows, k);
        long address = gpu.allocate(payload.length);
        try (var arena = java.lang.foreign.Arena.ofConfined()) {
            var host = arena.allocate(payload.length, 16);
            host.copyFrom(java.lang.foreign.MemorySegment.ofArray(payload));
            gpu.copyHostToDevice(address, host, payload.length);
        } catch (RuntimeException | Error failure) {
            gpu.free(address);
            throw failure;
        }
        return new TensorHandle(
                name,
                new long[] {rows, k},
                source.dataType(),
                WeightFormat.NVFP4,
                WeightLayout.ROW_SPLIT_K128_V1,
                address,
                payload.length);
    }

    private static TensorHandle nvfp4Rows(TensorHandle source, long first, long rows, String name, GpuMemory gpu) {
        long k = source.shape()[1];
        WeightLayout layout = source.layout();
        long rowBytes = Nvfp4Layout.paddedK(k) / 2, rowScales = Nvfp4Layout.rowScaleBytes(k, layout);
        long sourceScales = Nvfp4Layout.scaleOffset(source.shape()[0], k);
        long sourceTrailer = Nvfp4Layout.trailerOffset(source.shape()[0], k, layout);
        long scales = Nvfp4Layout.scaleOffset(rows, k), trailer = Nvfp4Layout.trailerOffset(rows, k, layout);
        long bytes = Nvfp4Layout.byteSize(rows, k, layout);
        long address = gpu.allocate(bytes);
        try {
            long base = source.deviceAddress();
            gpu.copyDeviceToDevice(address, base + first * rowBytes, rows * rowBytes);
            gpu.copyDeviceToDevice(address + scales, base + sourceScales + first * rowScales, rows * rowScales);
            gpu.copyDeviceToDevice(address + trailer, base + sourceTrailer, Nvfp4Layout.trailerBytes(layout));
        } catch (RuntimeException | Error failure) {
            gpu.free(address);
            throw failure;
        }
        return new TensorHandle(
                name, new long[] {rows, k}, source.dataType(), WeightFormat.NVFP4, source.layout(), address, bytes);
    }

    private static long align256(long value) {
        return (value + 255) & ~255L;
    }

    private static QwenMtpWeights buildMtp(Map<String, TensorHandle> handles) throws QwenWeightLoadException {
        String prefix = "mtp/layer";
        if (handles.containsKey("mtp/layer/attention/query_key")) {
            QwenLayerWeights layer = new QwenLayerWeights(
                    0,
                    take(handles, prefix + "/input_norm"),
                    take(handles, prefix + "/post_attention_norm"),
                    new QwenCompactAttentionWeights(
                            take(handles, prefix + "/attention/query_key"),
                            take(handles, prefix + "/attention/gate_value"),
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
        boolean sd4 =
                descriptor.layout() == WeightLayout.ROW_SPLIT_K128_SD4_V1 && descriptor.format() == WeightFormat.NVFP4;
        if (quantized && descriptor.layout() != WeightLayout.ROW_SPLIT_K128_V1 && !p2e2 && !sd4) {
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
                                || ((p2e2 || sd4) && expected.layout() == WeightLayout.ROW_SPLIT_K128_V1)))) {
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

    private static TensorHandle mapped(TensorHandle host, long deviceAddress) {
        return new TensorHandle(
                host.name(),
                host.shape(),
                host.dataType(),
                host.format(),
                host.layout(),
                deviceAddress,
                host.byteSize(),
                host.hostAddress(),
                true);
    }

    /// Host-backed objects start on 4 KiB boundaries within the arena.
    private static long hostSlot(long byteSize) {
        return (byteSize + 4095) & ~4095L;
    }

    private static void freeAll(Iterable<TensorHandle> handles, GpuMemory gpuMemory, Throwable failure) {
        for (TensorHandle handle : handles) {
            if (handle.hostAddress() != 0) continue; // in the host arena
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
