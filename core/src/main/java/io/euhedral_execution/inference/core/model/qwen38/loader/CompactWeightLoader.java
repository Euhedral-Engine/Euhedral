package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorDataReader;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.TensorLoader;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.artifact.WeightLoadException;
import io.euhedral_execution.inference.core.artifact.format.ExpectedTensor;
import io.euhedral_execution.inference.core.artifact.format.LinearFormat;
import io.euhedral_execution.inference.core.artifact.format.LinearFormats;
import io.euhedral_execution.inference.core.artifact.format.TrellisFormat;
import io.euhedral_execution.inference.core.artifact.format.TrellisLayout;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/// Assembles the canonical Qwen model view from NInfer-compatible compact runtime objects.
///
/// The source Hugging Face names are intentionally absent here. Fused objects remain one
/// `TensorHandle`: consumers must use the descriptor's runtime layout rather than reconstructing
/// source projections at load time.
final class CompactWeightLoader {

    private static final int EXPECTED_OBJECT_COUNT = 785;

    private CompactWeightLoader() {}

    static Weights load(
            Path artifactPath,
            Artifact artifact,
            GpuMemory gpuMemory,
            Map<String, TensorDescriptor> descriptors,
            ArtifactProfile.Speculation speculation,
            Set<String> hostBacked,
            Set<String> diskBacked)
            throws IOException {
        Qwen38Config config = artifact.config();
        validateConfig(config);
        DFlash2Config dflash2 = DFlash2Inventory.read(artifactPath, descriptors);
        validateInventory(config, descriptors, dflash2);
        validateTrellisSegments(artifactPath, descriptors);
        Map<String, TensorHandle> handles =
                loadObjects(artifactPath, descriptors.values(), gpuMemory, speculation, hostBacked, diskBacked);
        try {
            // Only an MTP load prepares the MTP layer for execution.
            if (speculation == ArtifactProfile.Speculation.MTP) splitMtpAttention(handles, gpuMemory);
            Weights weights = assemble(config, handles);
            if (speculation != ArtifactProfile.Speculation.DFLASH2) return weights;
            return new Weights(
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
            freeArena(handles.values(), gpuMemory, failure);
            return propagate(failure);
        }
    }

    /// Places each object `speculation` loads: those named in `hostBacked` in one pinned host arena, those named in
    /// `diskBacked` nowhere (their handles name their offset in the file, which execution reads them from), every
    /// other one on the device. A disk-backed object must be one a transfer stages
    /// ([HostWeightSelection#stageable]).
    static Map<String, TensorHandle> loadObjects(
            Path artifactPath,
            Collection<TensorDescriptor> descriptors,
            GpuMemory gpuMemory,
            ArtifactProfile.Speculation speculation,
            Set<String> hostBacked,
            Set<String> diskBacked)
            throws IOException {
        // The DFlash2 selector's codebooks are always read in place from mapped host memory.
        Set<String> mapped = new HashSet<>();
        if (speculation == ArtifactProfile.Speculation.DFLASH2) mapped.addAll(DFlash2Inventory.MAPPED);
        Set<String> host = new HashSet<>(hostBacked);
        host.addAll(mapped);
        // A trellis companion is a few KB that every launch of its object reads: it never leaves the device.
        Set<String> companions = trellisCompanions(descriptors);
        for (String name : host)
            if (companions.contains(name)) throw new IllegalArgumentException(name + " cannot be host-backed");
        for (TensorDescriptor descriptor : descriptors) {
            if (!diskBacked.contains(descriptor.name())) continue;
            if (host.contains(descriptor.name()) || !HostWeightSelection.stageable(descriptor))
                throw new IllegalArgumentException(descriptor.name() + " cannot be read from disk");
        }

        Map<String, TensorHandle> handles = new LinkedHashMap<>();
        long hostArena = 0;
        try {
            // Host-backed objects share one pinned arena, allocated before any payload is read: huge
            // pages are far likelier to be available then than between reads.
            long hostBytes = 0;
            for (TensorDescriptor descriptor : descriptors) {
                if (uploads(descriptor.name(), speculation) && host.contains(descriptor.name()))
                    hostBytes += hostSlot(descriptor.byteSize());
            }
            if (hostBytes > 0) hostArena = gpuMemory.allocateHostWeights(hostBytes);
            long hostOffset = 0;
            for (TensorDescriptor descriptor : descriptors) {
                if (!uploads(descriptor.name(), speculation)) continue;
                if (diskBacked.contains(descriptor.name())) {
                    handles.put(descriptor.name(), TensorHandle.disk(descriptor));
                } else if (host.contains(descriptor.name())) {
                    TensorHandle loaded = TensorLoader.loadToHost(artifactPath, descriptor, hostArena + hostOffset);
                    // The embedding and the codebooks are gathers, and the output head a streamed matrix: kernels
                    // read them in place over the bus instead of staging them.
                    if (HostWeightSelection.mapped(descriptor.name()) || mapped.contains(descriptor.name()))
                        loaded = mapped(loaded, gpuMemory.hostWeightsDeviceAddress(loaded.hostAddress()));
                    handles.put(descriptor.name(), loaded);
                    hostOffset += hostSlot(descriptor.byteSize());
                } else handles.put(descriptor.name(), TensorLoader.load(artifactPath, descriptor, gpuMemory));
            }
            return handles;
        } catch (Throwable failure) {
            freeAll(handles.values(), gpuMemory, failure);
            if (hostArena != 0) {
                try {
                    gpuMemory.freeHostWeights(hostArena);
                } catch (Throwable cleanupFailure) {
                    failure.addSuppressed(cleanupFailure);
                }
            }
            throw propagateObjects(failure);
        }
    }

    /// Frees the host arena of `handles`: the lowest host address among them, where the arena starts.
    private static void freeArena(Collection<TensorHandle> handles, GpuMemory gpuMemory, Throwable failure) {
        long arena = 0;
        for (TensorHandle handle : handles)
            if (handle.hostAddress() != 0 && (arena == 0 || Long.compareUnsigned(handle.hostAddress(), arena) < 0))
                arena = handle.hostAddress();
        if (arena == 0) return;
        try {
            gpuMemory.freeHostWeights(arena);
        } catch (Throwable cleanupFailure) {
            failure.addSuppressed(cleanupFailure);
        }
    }

    private static IOException propagateObjects(Throwable failure) throws IOException {
        if (failure instanceof IOException exception) throw exception;
        if (failure instanceof RuntimeException exception) throw exception;
        if (failure instanceof Error error) throw error;
        throw new AssertionError(failure);
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

    static Weights loadFirstLayer(
            Path artifactPath, Qwen38Config config, GpuMemory gpuMemory, Map<String, TensorDescriptor> descriptors)
            throws IOException {
        validateConfig(config);
        validateInventory(config, descriptors, DFlash2Inventory.read(artifactPath, descriptors));
        if (config.layerTypes()[0] != LayerType.GATED_DELTA_NET) {
            throw new WeightLoadException("actual layer zero is not a compact GDN layer");
        }
        if (descriptors.values().stream().anyMatch(CompactWeightLoader::trellis)) {
            throw new WeightLoadException("the first-layer load does not carry trellis companions");
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
                    throw new WeightLoadException("compact artifact is missing runtime object '" + name + "'");
                }
                handles.put(name, TensorLoader.load(artifactPath, descriptor, gpuMemory));
            }

            LayerWeights layer = new LayerWeights(
                    0,
                    take(handles, prefix + "/input_norm"),
                    take(handles, prefix + "/post_attention_norm"),
                    buildGdn(handles, prefix),
                    new DenseFfnWeights(take(handles, prefix + "/mlp/gate_up"), take(handles, prefix + "/mlp/down")));
            return new Weights(
                    config,
                    take(handles, "text/token_embedding"),
                    new LayerWeights[] {layer},
                    null,
                    null,
                    null,
                    handles);
        } catch (Throwable failure) {
            freeAll(handles.values(), gpuMemory, failure);
            return propagate(failure);
        }
    }

    static Weights assemble(Qwen38Config config, Map<String, TensorHandle> handles) throws WeightLoadException {
        TensorHandle tokenEmbedding = take(handles, "text/token_embedding");
        TensorHandle finalNorm = take(handles, "text/final_norm");
        TensorHandle lmHead = take(handles, "text/output_head");

        LayerWeights[] layers = new LayerWeights[config.numHiddenLayers()];
        for (int index = 0; index < layers.length; index++) {
            String prefix = "text/layers/" + index;
            TensorHandle inputNorm = take(handles, prefix + "/input_norm");
            TensorHandle postNorm = take(handles, prefix + "/post_attention_norm");
            var mixer = config.layerTypes()[index] == LayerType.FULL_ATTENTION
                    ? buildAttention(handles, prefix)
                    : buildGdn(handles, prefix);
            var ffn = new DenseFfnWeights(take(handles, prefix + "/mlp/gate_up"), take(handles, prefix + "/mlp/down"));
            layers[index] = new LayerWeights(index, inputNorm, postNorm, mixer, ffn);
        }

        MtpWeights mtp =
                config.mtpLayerCount() == 0 || !handles.containsKey("mtp/input_projection") ? null : buildMtp(handles);
        return new Weights(config, tokenEmbedding, layers, finalNorm, lmHead, mtp, handles);
    }

    private static AttentionWeights buildAttention(Map<String, TensorHandle> handles, String prefix)
            throws WeightLoadException {
        return new AttentionWeights(
                take(handles, prefix + "/attention/query_key"),
                take(handles, prefix + "/attention/gate_value"),
                take(handles, prefix + "/attention/query_norm"),
                take(handles, prefix + "/attention/key_norm"),
                take(handles, prefix + "/attention/output"));
    }

    private static GdnWeights buildGdn(Map<String, TensorHandle> handles, String prefix) throws WeightLoadException {
        return new GdnWeights(
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
        if (packed == null || packed.hostBacked() || packed.diskBacked()) return;
        LinearFormat format =
                LinearFormats.find(packed.format(), packed.layout()).orElse(null);
        if (format == null || !format.splitsRows()) return;
        List<TensorHandle> object = group(handles, MTP_ATTENTION);
        // Rows are the first dimension of the grouped formats; a trellis payload is [k/16, n/16, w].
        long half = (packed.format() == WeightFormat.TRELLIS ? packed.shape()[1] * 16 : packed.shape()[0]) / 2;
        List<TensorHandle> first = format.rows(object, 0, half, "mtp/layer/attention/query_key", gpu);
        List<TensorHandle> second;
        try {
            second = format.rows(object, half, half, "mtp/layer/attention/gate_value", gpu);
        } catch (RuntimeException | Error failure) {
            for (TensorHandle handle : first) gpu.free(handle.deviceAddress());
            throw failure;
        }
        for (TensorHandle handle : first) handles.put(handle.name(), handle);
        for (TensorHandle handle : second) handles.put(handle.name(), handle);
        for (TensorHandle handle : object) {
            handles.remove(handle.name());
            gpu.free(handle.deviceAddress());
        }
    }

    /// `name`'s tensor and its companions, in the order its format lists them.
    static List<TensorHandle> group(Map<String, TensorHandle> handles, String name) {
        TensorHandle object = handles.get(name);
        if (object == null) throw new IllegalArgumentException("no loaded runtime object '" + name + "'");
        List<TensorHandle> group = new ArrayList<>(List.of(object));
        for (String companion : LinearFormats.of(object).companions(name)) {
            TensorHandle handle = handles.get(companion);
            if (handle == null)
                throw new IllegalArgumentException("no loaded companion '" + companion + "' of '" + name + "'");
            group.add(handle);
        }
        return group;
    }

    private static MtpWeights buildMtp(Map<String, TensorHandle> handles) throws WeightLoadException {
        String prefix = "mtp/layer";
        if (handles.containsKey("mtp/layer/attention/query_key")) {
            LayerWeights layer = new LayerWeights(
                    0,
                    take(handles, prefix + "/input_norm"),
                    take(handles, prefix + "/post_attention_norm"),
                    new AttentionWeights(
                            take(handles, prefix + "/attention/query_key"),
                            take(handles, prefix + "/attention/gate_value"),
                            take(handles, prefix + "/attention/query_norm"),
                            take(handles, prefix + "/attention/key_norm"),
                            take(handles, prefix + "/attention/output")),
                    new DenseFfnWeights(take(handles, prefix + "/mlp/gate_up"), take(handles, prefix + "/mlp/down")));
            return new MtpWeights(
                    take(handles, "mtp/embedding_norm"),
                    take(handles, "mtp/hidden_norm"),
                    take(handles, "mtp/input_projection"),
                    layer,
                    take(handles, "mtp/final_norm"));
        }
        LayerWeights layer = new LayerWeights(
                0,
                take(handles, prefix + "/input_norm"),
                take(handles, prefix + "/post_attention_norm"),
                new MtpAttentionWeights(
                        take(handles, prefix + "/attention/query_key_gate_value"),
                        take(handles, prefix + "/attention/query_norm"),
                        take(handles, prefix + "/attention/key_norm"),
                        take(handles, prefix + "/attention/output")),
                new DenseFfnWeights(take(handles, prefix + "/mlp/gate_up"), take(handles, prefix + "/mlp/down")));
        return new MtpWeights(
                take(handles, "mtp/embedding_norm"),
                take(handles, "mtp/hidden_norm"),
                take(handles, "mtp/input_projection"),
                layer,
                take(handles, "mtp/final_norm"));
    }

    private static boolean trellis(TensorDescriptor descriptor) {
        return descriptor.format() == WeightFormat.TRELLIS && descriptor.layout() == WeightLayout.TRELLIS_TILE16_V1;
    }

    /// The names of the companion tensors of every trellis object in `descriptors`.
    static Set<String> trellisCompanions(Map<String, TensorDescriptor> descriptors) {
        return trellisCompanions(descriptors.values());
    }

    static Set<String> trellisCompanions(Collection<TensorDescriptor> descriptors) {
        Set<String> names = new LinkedHashSet<>();
        for (TensorDescriptor descriptor : descriptors)
            if (trellis(descriptor)) names.addAll(LinearFormats.of(descriptor).companions(descriptor.name()));
        return names;
    }

    private static final List<String> COMPANION_SUFFIXES =
            List.of(TrellisFormat.SUH, TrellisFormat.SVH, TrellisFormat.SEGMENTS);

    /// Every trellis object has exactly its three companions, with the dtype, format, layout and shape the
    /// format names, and no companion stands without its object. Checked on descriptors alone, so a malformed
    /// artifact is refused before any payload is read or allocated.
    private static void validateTrellisCompanions(Map<String, TensorDescriptor> descriptors)
            throws WeightLoadException {
        for (TensorDescriptor descriptor : descriptors.values()) {
            if (!trellis(descriptor)) continue;
            String name = descriptor.name();
            if (descriptor.shape().length != 3)
                throw new WeightLoadException("trellis object '" + name + "' is not [k/16, n/16, w]");
            long n = descriptor.shape()[1] * 16, k = descriptor.shape()[0] * 16;
            TensorDescriptor suh = companion(descriptors, name, TrellisFormat.SUH, WeightFormat.FP16);
            TensorDescriptor svh = companion(descriptors, name, TrellisFormat.SVH, WeightFormat.FP16);
            TensorDescriptor segments = companion(descriptors, name, TrellisFormat.SEGMENTS, WeightFormat.I32);
            if (suh.shape().length != 2 || suh.shape()[1] != k || suh.shape()[0] < 1)
                throw new WeightLoadException(
                        "trellis companion '" + suh.name() + "' must have shape [segments, " + k + "]");
            if (svh.shape().length != 1 || svh.shape()[0] != n)
                throw new WeightLoadException("trellis companion '" + svh.name() + "' must have shape [" + n + "]");
            if (segments.shape().length != 1 || segments.shape()[0] != suh.shape()[0])
                throw new WeightLoadException("trellis companion '" + segments.name()
                        + "' must have one end per suh row (" + suh.shape()[0] + ")");
        }
        for (TensorDescriptor descriptor : descriptors.values())
            for (String suffix : COMPANION_SUFFIXES) {
                String name = descriptor.name();
                if (!name.endsWith(suffix)) continue;
                TensorDescriptor owner = descriptors.get(name.substring(0, name.length() - suffix.length()));
                if (owner == null || !trellis(owner))
                    throw new WeightLoadException("trellis companion '" + name + "' has no trellis object");
            }
    }

    private static TensorDescriptor companion(
            Map<String, TensorDescriptor> descriptors, String object, String suffix, WeightFormat format)
            throws WeightLoadException {
        String name = object + suffix;
        TensorDescriptor companion = descriptors.get(name);
        if (companion == null)
            throw new WeightLoadException("trellis object '" + object + "' is missing its companion '" + name + "'");
        TensorDataType dataType = format == WeightFormat.I32 ? TensorDataType.INT32 : TensorDataType.FP16;
        if (companion.format() != format
                || companion.dataType() != dataType
                || companion.layout() != WeightLayout.CONTIGUOUS_LE_V1)
            throw new WeightLoadException(
                    "trellis companion '" + name + "' must be " + dataType + "/" + format + "/CONTIGUOUS_LE_V1");
        return companion;
    }

    /// Reads each trellis object's `.segments` on the host and checks them against the object: strictly
    /// increasing multiples of 128, the last equal to n, one per `suh` row, at most 8; the packed MTP
    /// attention projection must also have n/2 as a boundary. Before any device allocation.
    static void validateTrellisSegments(Path artifactPath, Map<String, TensorDescriptor> descriptors)
            throws IOException {
        try (Arena arena = Arena.ofConfined()) {
            for (TensorDescriptor descriptor : descriptors.values()) {
                if (!trellis(descriptor)) continue;
                String name = descriptor.name();
                TensorDescriptor suh = descriptors.get(name + TrellisFormat.SUH);
                TensorDescriptor segments = descriptors.get(name + TrellisFormat.SEGMENTS);
                if (suh == null || segments == null) continue;
                long n = descriptor.shape()[1] * 16;
                try {
                    int[] ends = TrellisLayout.readEnds(TensorDataReader.read(artifactPath, segments, arena));
                    TrellisLayout.validateSegments(ends, n, suh.shape()[0]);
                    if (name.equals(MTP_ATTENTION) && Arrays.stream(ends).noneMatch(end -> end == n / 2))
                        throw new IllegalArgumentException("n/2=" + n / 2 + " is not a segment boundary");
                } catch (IllegalArgumentException invalid) {
                    throw new WeightLoadException(
                            "invalid trellis segments '" + segments.name() + "': " + invalid.getMessage());
                }
            }
        }
    }

    /// The text inventory: every object, and no other.
    static void validateInventory(Qwen38Config config, Map<String, TensorDescriptor> descriptors)
            throws WeightLoadException {
        validateInventory(config, descriptors, null);
    }

    /// The text inventory and, with `dflash2`, the DFlash2 drafter's objects: every object, and no other.
    static void validateInventory(Qwen38Config config, Map<String, TensorDescriptor> descriptors, DFlash2Config dflash2)
            throws WeightLoadException {
        validateTrellisCompanions(descriptors);
        int count = EXPECTED_OBJECT_COUNT
                + (dflash2 == null ? 0 : DFlash2Inventory.objectCount(dflash2))
                + trellisCompanions(descriptors).size();
        if (descriptors.size() != count) {
            throw new WeightLoadException(
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
            if (config.layerTypes()[index] == LayerType.FULL_ATTENTION) {
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
        expected.addAll(trellisCompanions(descriptors));
        for (String name : expected) {
            TensorDescriptor descriptor = descriptors.get(name);
            if (descriptor == null) {
                throw new WeightLoadException("compact artifact is missing runtime object '" + name + "'");
            }
            validateExpectedDescriptor(descriptor);
        }
        for (String name : descriptors.keySet()) {
            if (dflash2 != null && name.startsWith(DFlash2Inventory.PREFIX)) continue;
            if (!expected.contains(name))
                throw new WeightLoadException("compact artifact contains unknown runtime object '" + name + "'");
        }
    }

    /// Objects read by row or mapped in place, which have no trellis route.
    private static final Set<String> TRELLIS_REFUSED =
            Set.of("text/token_embedding", "text/output_head", "text/draft_head");

    private static void validateExpectedDescriptor(TensorDescriptor descriptor) throws WeightLoadException {
        String name = descriptor.name();
        if (descriptor.layout() == null) {
            throw new WeightLoadException("runtime object has no persistent layout: " + name);
        }
        boolean registered =
                LinearFormats.find(descriptor.format(), descriptor.layout()).isPresent();
        if (!registered && LinearFormats.quantized(descriptor.format())) {
            throw new WeightLoadException("quantized runtime object has unsupported layout: " + name);
        }
        if (!registered
                && !LinearFormats.quantized(descriptor.format())
                && descriptor.layout() != WeightLayout.CONTIGUOUS_LE_V1) {
            throw new WeightLoadException("direct runtime object has unsupported layout: " + name);
        }
        if (descriptor.format() == WeightFormat.TRELLIS && TRELLIS_REFUSED.contains(name)) {
            throw new WeightLoadException(
                    "'" + name + "' cannot be trellis: the token embedding and the heads are" + " Q3, NVFP4 or BF16");
        }
        Expected expected = expectedDescriptor(name);
        if (expected != null && !accepts(expected, descriptor)) {
            throw new WeightLoadException(
                    "compact runtime metadata conflicts with the registered layout for '" + name + "'");
        }
    }

    /// Whether `descriptor` may stand for the registered `expected` object: the same shape and data type, and the
    /// expected format and layout, or what the descriptor's [LinearFormat] allows in their place (any grouped
    /// width where a grouped one is expected, NVFP4 plain or SD4 where a row-split object is, P2E2 for row-split
    /// Q3, BF16 values for any quantized object).
    static boolean accepts(Expected expected, TensorDescriptor descriptor) {
        ExpectedTensor registered =
                new ExpectedTensor(expected.shape(), expected.dataType(), expected.format(), expected.layout());
        return LinearFormats.find(descriptor.format(), descriptor.layout())
                .map(format -> format.accepts(descriptor, registered))
                .orElseGet(() -> Arrays.equals(expected.shape(), descriptor.shape())
                        && expected.dataType() == descriptor.dataType()
                        && expected.format() == descriptor.format()
                        && expected.layout() == descriptor.layout());
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

    private static TensorHandle take(Map<String, TensorHandle> handles, String name) throws WeightLoadException {
        TensorHandle handle = handles.get(name);
        if (handle == null) {
            throw new WeightLoadException("compact artifact is missing loaded runtime object '" + name + "'");
        }
        return handle;
    }

    private static void validateConfig(Qwen38Config config) throws WeightLoadException {
        if (config == null
                || config.layerTypes() == null
                || config.layerTypes().length != 64
                || config.numHiddenLayers() != 64
                || config.mtpLayerCount() > 1) {
            throw new WeightLoadException("compact Qwen artifact has unsupported model topology");
        }
        if (config.numExperts() != 0) {
            throw new WeightLoadException("compact Q3 loader only supports the dense Qwen reference topology");
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
            // In the host arena, or in the file.
            if (handle.hostAddress() != 0 || handle.diskBacked()) continue;
            try {
                gpuMemory.free(handle.deviceAddress());
            } catch (Throwable cleanupFailure) {
                failure.addSuppressed(cleanupFailure);
            }
        }
    }

    private static Weights propagate(Throwable failure) throws IOException {
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
