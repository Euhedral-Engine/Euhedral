package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model_loader.artifact.Nvfp4Layout;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.config.DFlash2Config;
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

    private static final int EXPECTED_OBJECT_COUNT = 785;

    private QwenCompactWeightLoader() {}

    static QwenWeights load(
            Path artifactPath,
            QwenArtifact artifact,
            GpuMemory gpuMemory,
            Map<String, TensorDescriptor> descriptors,
            ArtifactProfile.Speculation speculation,
            Set<String> hostBacked)
            throws IOException {
        QwenConfig config = artifact.config();
        validateConfig(config);
        DFlash2Config dflash2 = DFlash2Inventory.read(artifactPath, descriptors);
        validateInventory(config, descriptors, dflash2);
        // The DFlash2 selector's codebooks are always read in place from mapped host memory.
        Set<String> mapped = new java.util.HashSet<>();
        if (speculation == ArtifactProfile.Speculation.DFLASH2) mapped.addAll(DFlash2Inventory.MAPPED);
        Set<String> host = new java.util.HashSet<>(hostBacked);
        host.addAll(mapped);

        Map<String, TensorHandle> handles = new LinkedHashMap<>();
        long hostArena = 0;
        try {
            // Host-backed objects share one pinned arena, allocated before any payload is read: huge
            // pages are far likelier to be available then than between reads.
            long hostBytes = 0;
            for (TensorDescriptor descriptor : descriptors.values()) {
                if (uploads(descriptor.name(), speculation) && host.contains(descriptor.name()))
                    hostBytes += hostSlot(descriptor.byteSize());
            }
            if (hostBytes > 0) hostArena = gpuMemory.allocateHostWeights(hostBytes);
            long hostOffset = 0;
            for (TensorDescriptor descriptor : descriptors.values()) {
                if (!uploads(descriptor.name(), speculation)) continue;
                if (host.contains(descriptor.name())) {
                    TensorHandle loaded = TensorLoader.loadToHost(artifactPath, descriptor, hostArena + hostOffset);
                    // The embedding and the codebooks are gathers: kernels read their rows in place instead of
                    // staging them.
                    if (descriptor.name().equals(HostWeightSelection.EMBEDDING) || mapped.contains(descriptor.name()))
                        loaded = mapped(loaded, gpuMemory.hostWeightsDeviceAddress(loaded.hostAddress()));
                    handles.put(descriptor.name(), loaded);
                    hostOffset += hostSlot(descriptor.byteSize());
                } else handles.put(descriptor.name(), TensorLoader.load(artifactPath, descriptor, gpuMemory));
            }
            // Only an MTP load prepares the MTP layer for execution.
            if (speculation == ArtifactProfile.Speculation.MTP) splitMtpAttention(handles, gpuMemory);
            QwenWeights weights = assemble(config, handles);
            if (speculation != ArtifactProfile.Speculation.DFLASH2) return weights;
            return new QwenWeights(
                    weights.config(),
                    weights.tokenEmbedding(),
                    weights.layers(),
                    weights.finalNorm(),
                    weights.lmHead(),
                    null,
                    weights.runtimeObjects(),
                    DFlash2Inventory.assemble(dflash2, handles));
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

    /// Whether a load places the named object on the device (or in host memory for it): the MTP layer and draft
    /// head only for MTP, the DFlash2 drafter (its configuration aside, which the loader reads on the host) only
    /// for DFlash2.
    static boolean uploads(String name, ArtifactProfile.Speculation speculation) {
        if (name.startsWith("mtp/") || name.startsWith("text/draft_head"))
            return speculation == ArtifactProfile.Speculation.MTP;
        if (name.startsWith(DFlash2Inventory.PREFIX))
            return speculation == ArtifactProfile.Speculation.DFLASH2 && !name.equals(DFlash2Inventory.CONFIG);
        return true;
    }

    static QwenWeights loadFirstLayer(
            Path artifactPath, QwenConfig config, GpuMemory gpuMemory, Map<String, TensorDescriptor> descriptors)
            throws IOException {
        validateConfig(config);
        validateInventory(config, descriptors, DFlash2Inventory.read(artifactPath, descriptors));
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
    /// their gate/value rows (gate, v) (tools/euhedral_artifacts). The NVFP4 pack is split here into those two
    /// tensors, each with its rows' codes and scales and the shared global scale, so the MTP layer executes as an
    /// ordinary attention layer (docs/MTP_CONTRACT.md §1).
    private static void splitMtpAttention(Map<String, TensorHandle> handles, GpuMemory gpu) {
        TensorHandle packed = handles.get(MTP_ATTENTION);
        if (packed == null || packed.hostBacked()) return;
        if (packed.format() != WeightFormat.NVFP4) return;
        long rows = packed.shape()[0], k = packed.shape()[1];
        long half = rows / 2;
        handles.put("mtp/layer/attention/query_key", nvfp4Rows(packed, 0, half, "mtp/layer/attention/query_key", gpu));
        handles.put(
                "mtp/layer/attention/gate_value", nvfp4Rows(packed, half, half, "mtp/layer/attention/gate_value", gpu));
        handles.remove(MTP_ATTENTION);
        gpu.free(packed.deviceAddress());
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

    /// The text inventory: every object, and no other.
    static void validateInventory(QwenConfig config, Map<String, TensorDescriptor> descriptors)
            throws QwenWeightLoadException {
        validateInventory(config, descriptors, null);
    }

    /// The text inventory and, with `dflash2`, the DFlash2 drafter's objects: every object, and no other.
    static void validateInventory(QwenConfig config, Map<String, TensorDescriptor> descriptors, DFlash2Config dflash2)
            throws QwenWeightLoadException {
        int count = EXPECTED_OBJECT_COUNT + (dflash2 == null ? 0 : DFlash2Inventory.objectCount(dflash2));
        if (descriptors.size() != count) {
            throw new QwenWeightLoadException(
                    "compact Qwen artifact must contain " + count + " runtime objects, found " + descriptors.size());
        }
        if (dflash2 != null) DFlash2Inventory.validate(dflash2, descriptors);
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
        for (String name : descriptors.keySet()) {
            if (dflash2 != null && name.startsWith(DFlash2Inventory.PREFIX)) continue;
            if (!expected.contains(name))
                throw new QwenWeightLoadException("compact artifact contains unknown runtime object '" + name + "'");
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
            return quantized(new long[] {14336, 5120}, TensorDataType.BF16, WeightFormat.Q4_G64_FP16);
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
