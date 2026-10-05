package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.model_loader.artifact.TensorDataReader;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.config.DFlash2Config;
import io.euhedral_execution.inference.core.model_loader.layer_weights.DFlash2Weights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/// The DFlash2 drafter's objects in an artifact (`dflash2/`, tools/euhedral_artifacts/dflash2.py): which there are,
/// their shapes, and how they assemble into [DFlash2Weights]. The matrices that feed DFlash2 linears (each layer's five
/// projections and two convolution kernel projections, and `fc`) are BF16 or NVFP4; every other object is BF16, and
/// `dflash2/config` (I32) is read on the host only.
final class DFlash2Inventory {
    static final String PREFIX = "dflash2/";
    static final String CONFIG = PREFIX + "config";
    static final String PREDECESSOR = PREFIX + "selector/predecessor";
    static final String SUCCESSOR = PREFIX + "selector/successor";
    /// The selector's codebooks stay in mapped host memory: a draft gathers only its anchor's and candidates' rows.
    static final Set<String> MAPPED = Set.of(PREDECESSOR, SUCCESSOR);

    private DFlash2Inventory() {}

    /// One expected object: its shape, and whether it is a projection (BF16 or NVFP4) or a direct BF16 tensor.
    record Entry(long[] shape, boolean projection) {}

    /// The drafter configuration the artifact records, or null when it holds no drafter.
    static DFlash2Config read(Path artifactPath, Map<String, TensorDescriptor> descriptors) throws IOException {
        TensorDescriptor descriptor = descriptors.get(CONFIG);
        if (descriptor == null) return null;
        if (descriptor.format() != WeightFormat.I32 || descriptor.shape().length != 1)
            throw new QwenWeightLoadException("dflash2/config must be an I32 vector");
        try (Arena arena = Arena.ofConfined()) {
            int[] words = TensorDataReader.read(artifactPath, descriptor, arena)
                    .asSlice(0, descriptor.shape()[0] * Integer.BYTES)
                    .toArray(ValueLayout.JAVA_INT_UNALIGNED);
            try {
                return DFlash2Config.parse(words);
            } catch (IllegalArgumentException invalid) {
                throw new QwenWeightLoadException(invalid.getMessage());
            }
        }
    }

    /// Every drafter object but the configuration, by name.
    static Map<String, Entry> expected(DFlash2Config c) {
        long hidden = c.hiddenSize();
        Map<String, Entry> entries = new LinkedHashMap<>();
        entries.put(PREFIX + "fc", new Entry(new long[] {hidden, c.tapWidth()}, true));
        entries.put(PREFIX + "hidden_norm", new Entry(new long[] {hidden}, false));
        entries.put(PREFIX + "final_norm", new Entry(new long[] {hidden}, false));
        for (int layer = 0; layer < c.layers(); layer++) {
            String p = PREFIX + "layers/" + layer + "/";
            entries.put(p + "input_norm", new Entry(new long[] {hidden}, false));
            entries.put(p + "post_attention_norm", new Entry(new long[] {hidden}, false));
            entries.put(p + "attention/query", new Entry(new long[] {c.queryWidth(), hidden}, true));
            entries.put(p + "attention/key_value", new Entry(new long[] {2L * c.keyValueWidth(), hidden}, true));
            entries.put(p + "attention/output", new Entry(new long[] {hidden, c.queryWidth()}, true));
            entries.put(p + "attention/query_norm", new Entry(new long[] {c.headDim()}, false));
            entries.put(p + "attention/key_norm", new Entry(new long[] {c.headDim()}, false));
            entries.put(p + "mlp/gate_up", new Entry(new long[] {2L * c.intermediateSize(), hidden}, true));
            entries.put(p + "mlp/down", new Entry(new long[] {hidden, c.intermediateSize()}, true));
            for (String conv : new String[] {"attention_conv", "mlp_conv"}) {
                entries.put(p + conv + "/base", new Entry(new long[] {2, c.convKernel(), hidden}, false));
                entries.put(p + conv + "/projection", new Entry(new long[] {c.convProjectionWidth(), hidden}, true));
            }
        }
        entries.put(PREFIX + "selector/hidden_projection", new Entry(new long[] {c.selectorRank(), hidden}, false));
        entries.put(PREDECESSOR, new Entry(new long[] {c.vocabSize(), c.selectorRank()}, false));
        entries.put(SUCCESSOR, new Entry(new long[] {c.vocabSize(), c.selectorRank()}, false));
        return entries;
    }

    /// Checks the artifact's `dflash2/` objects: exactly the expected ones, each in its shape and an accepted format.
    static void validate(DFlash2Config config, Map<String, TensorDescriptor> descriptors)
            throws QwenWeightLoadException {
        Map<String, Entry> expected = expected(config);
        for (Map.Entry<String, Entry> entry : expected.entrySet()) {
            TensorDescriptor descriptor = descriptors.get(entry.getKey());
            if (descriptor == null)
                throw new QwenWeightLoadException("artifact is missing DFlash2 object '" + entry.getKey() + "'");
            boolean bf16 = descriptor.format() == WeightFormat.BF16
                    && descriptor.layout() == WeightLayout.CONTIGUOUS_LE_V1
                    && descriptor.dataType() == TensorDataType.BF16;
            boolean nvfp4 = descriptor.format() == WeightFormat.NVFP4
                    && descriptor.layout() == WeightLayout.ROW_SPLIT_K128_V1
                    && entry.getValue().projection();
            if (!Arrays.equals(entry.getValue().shape(), descriptor.shape()) || !(bf16 || nvfp4))
                throw new QwenWeightLoadException(
                        "DFlash2 object '" + entry.getKey() + "' has an unexpected shape or format");
        }
        for (String name : descriptors.keySet())
            if (name.startsWith(PREFIX) && !name.equals(CONFIG) && !expected.containsKey(name))
                throw new QwenWeightLoadException("artifact contains unknown DFlash2 object '" + name + "'");
    }

    /// The number of `dflash2/` objects in an artifact that holds the drafter, the configuration included.
    static int objectCount(DFlash2Config config) {
        return expected(config).size() + 1;
    }

    static DFlash2Weights assemble(DFlash2Config c, Map<String, TensorHandle> handles) throws QwenWeightLoadException {
        DFlash2Weights.Layer[] layers = new DFlash2Weights.Layer[c.layers()];
        for (int layer = 0; layer < layers.length; layer++) {
            String p = PREFIX + "layers/" + layer + "/";
            layers[layer] = new DFlash2Weights.Layer(
                    take(handles, p + "input_norm"),
                    take(handles, p + "post_attention_norm"),
                    take(handles, p + "attention/query"),
                    take(handles, p + "attention/key_value"),
                    take(handles, p + "attention/output"),
                    take(handles, p + "attention/query_norm"),
                    take(handles, p + "attention/key_norm"),
                    take(handles, p + "mlp/gate_up"),
                    take(handles, p + "mlp/down"),
                    take(handles, p + "attention_conv/base"),
                    take(handles, p + "attention_conv/projection"),
                    take(handles, p + "mlp_conv/base"),
                    take(handles, p + "mlp_conv/projection"));
        }
        return new DFlash2Weights(
                c,
                take(handles, PREFIX + "fc"),
                take(handles, PREFIX + "hidden_norm"),
                take(handles, PREFIX + "final_norm"),
                layers,
                take(handles, PREFIX + "selector/hidden_projection"),
                take(handles, PREDECESSOR),
                take(handles, SUCCESSOR));
    }

    private static TensorHandle take(Map<String, TensorHandle> handles, String name) throws QwenWeightLoadException {
        TensorHandle handle = handles.get(name);
        if (handle == null) throw new QwenWeightLoadException("DFlash2 object '" + name + "' was not loaded");
        return handle;
    }
}
