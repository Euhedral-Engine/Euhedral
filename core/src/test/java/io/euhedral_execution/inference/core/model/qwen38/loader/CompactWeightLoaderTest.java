package io.euhedral_execution.inference.core.model.qwen38.loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.artifact.WeightLoadException;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class CompactWeightLoaderTest {

    @Test
    void assemblesFusedRuntimeObjectsWithoutSplittingHandles() throws Exception {
        Qwen38Config config = config();
        Map<String, TensorHandle> handles = new LinkedHashMap<>();
        handles.put("text/token_embedding", handle("text/token_embedding"));
        handles.put("text/final_norm", handle("text/final_norm"));
        handles.put("text/output_head", handle("text/output_head"));
        for (int index = 0; index < 64; index++) {
            String prefix = "text/layers/" + index;
            handles.put(prefix + "/input_norm", handle(prefix + "/input_norm"));
            handles.put(prefix + "/post_attention_norm", handle(prefix + "/post_attention_norm"));
            handles.put(prefix + "/mlp/gate_up", handle(prefix + "/mlp/gate_up"));
            handles.put(prefix + "/mlp/down", handle(prefix + "/mlp/down"));
            if (config.layerTypes()[index] == LayerType.FULL_ATTENTION) {
                for (String suffix : new String[] {
                    "/attention/query_key",
                    "/attention/gate_value",
                    "/attention/query_norm",
                    "/attention/key_norm",
                    "/attention/output"
                }) {
                    handles.put(prefix + suffix, handle(prefix + suffix));
                }
            } else {
                for (String suffix : new String[] {
                    "/gdn/a_log",
                    "/gdn/dt_bias",
                    "/gdn/convolution",
                    "/gdn/a_projection",
                    "/gdn/b_projection",
                    "/gdn/query_key",
                    "/gdn/value_z",
                    "/gdn/norm",
                    "/gdn/output"
                }) {
                    handles.put(prefix + suffix, handle(prefix + suffix));
                }
            }
        }
        handles.put("mtp/input_projection", handle("mtp/input_projection"));
        handles.put("mtp/embedding_norm", handle("mtp/embedding_norm"));
        handles.put("mtp/hidden_norm", handle("mtp/hidden_norm"));
        handles.put("mtp/final_norm", handle("mtp/final_norm"));
        handles.put("mtp/layer/input_norm", handle("mtp/layer/input_norm"));
        handles.put("mtp/layer/post_attention_norm", handle("mtp/layer/post_attention_norm"));
        for (String suffix : new String[] {
            "/attention/query_key_gate_value",
            "/attention/query_norm",
            "/attention/key_norm",
            "/attention/output",
            "/mlp/gate_up",
            "/mlp/down"
        }) {
            handles.put("mtp/layer" + suffix, handle("mtp/layer" + suffix));
        }
        handles.put("text/draft_head", handle("text/draft_head"));
        handles.put("text/draft_head_token_ids", handle("text/draft_head_token_ids"));
        handles.put("vision/sentinel", handle("vision/sentinel"));

        Weights weights = CompactWeightLoader.assemble(config, handles);

        LayerWeights gdnLayer = weights.layers()[0];
        assertInstanceOf(GdnWeights.class, gdnLayer.mixer());
        assertInstanceOf(DenseFfnWeights.class, gdnLayer.ffn());
        assertInstanceOf(AttentionWeights.class, weights.layers()[3].mixer());
        assertEquals(
                "text/layers/3/attention/query_key",
                ((AttentionWeights) weights.layers()[3].mixer()).queryKey().name());
        assertInstanceOf(MtpAttentionWeights.class, weights.mtp().layer().mixer());
        assertEquals(handles.size(), weights.runtimeObjects().size());
        assertThrows(
                UnsupportedOperationException.class,
                () -> weights.runtimeObjects().put("unexpected", handle("unexpected")));
    }

    private static TensorHandle handle(String name) {
        return new TensorHandle(name, new long[] {1}, TensorDataType.BF16, WeightFormat.BF16, 1L, 2L);
    }

    @Test
    void acceptsTheNvfp4TextInventoryWithoutVision() throws Exception {
        CompactWeightLoader.validateInventory(config(), nvfp4Inventory());
    }

    private static final io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Config DFLASH2 =
            io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Config.parse(new int[] {
                1,
                5,
                5120,
                17408,
                32,
                8,
                128,
                8,
                248070,
                2,
                16,
                16,
                256,
                2048,
                248320,
                Float.floatToIntBits(1e-6f),
                Float.floatToIntBits(1e7f),
                5,
                5,
                19,
                33,
                47,
                61
            });

    private static Map<String, TensorDescriptor> withDFlash2(boolean nvfp4Projections) {
        Map<String, TensorDescriptor> descriptors = nvfp4Inventory();
        descriptors.put(
                "dflash2/config",
                new TensorDescriptor(
                        "dflash2/config",
                        new long[] {23},
                        TensorDataType.INT32,
                        WeightFormat.I32,
                        WeightLayout.CONTIGUOUS_LE_V1,
                        0,
                        92));
        DFlash2Inventory.expected(DFLASH2)
                .forEach((name, entry) -> descriptors.put(
                        name,
                        nvfp4Projections && entry.projection()
                                ? descriptor(name, entry.shape(), WeightFormat.NVFP4)
                                : new TensorDescriptor(
                                        name,
                                        entry.shape(),
                                        TensorDataType.BF16,
                                        WeightFormat.BF16,
                                        WeightLayout.CONTIGUOUS_LE_V1,
                                        0,
                                        1)));
        return descriptors;
    }

    @Test
    void acceptsTheDFlash2DrafterBesideTheTextInventory() throws Exception {
        CompactWeightLoader.validateInventory(config(), withDFlash2(false), DFLASH2);
        CompactWeightLoader.validateInventory(config(), withDFlash2(true), DFLASH2);
        assertEquals(785 + 72, withDFlash2(false).size(), "5 x 13 layer objects, 3 shared, 3 selector, config");
    }

    @Test
    void rejectsADFlash2ArtifactReadWithoutItsConfigurationOrWithAnUnknownObject() {
        assertThrows(
                WeightLoadException.class, () -> CompactWeightLoader.validateInventory(config(), withDFlash2(false)));
        Map<String, TensorDescriptor> extra = withDFlash2(false);
        extra.put(
                "dflash2/layers/5/input_norm",
                descriptor("dflash2/layers/5/input_norm", new long[] {5120}, WeightFormat.BF16));
        extra.remove("dflash2/final_norm");
        assertThrows(WeightLoadException.class, () -> CompactWeightLoader.validateInventory(config(), extra, DFLASH2));
        Map<String, TensorDescriptor> quantizedNorm = withDFlash2(false);
        quantizedNorm.put(
                "dflash2/hidden_norm", descriptor("dflash2/hidden_norm", new long[] {5120}, WeightFormat.NVFP4));
        assertThrows(
                WeightLoadException.class,
                () -> CompactWeightLoader.validateInventory(config(), quantizedNorm, DFLASH2));
    }

    @Test
    void rejectsAPartialVisionTower() throws Exception {
        Map<String, TensorDescriptor> descriptors = nvfp4Inventory();
        descriptors.put(
                "vision/merger/fc1", descriptor("vision/merger/fc1", new long[] {4608, 4608}, WeightFormat.NVFP4));
        assertThrows(WeightLoadException.class, () -> CompactWeightLoader.validateInventory(config(), descriptors));
    }

    @Test
    void rejectsNvfp4WhereADirectTensorIsRegistered() throws Exception {
        Map<String, TensorDescriptor> descriptors = nvfp4Inventory();
        descriptors.put("text/final_norm", descriptor("text/final_norm", new long[] {5120}, WeightFormat.NVFP4));
        assertThrows(WeightLoadException.class, () -> CompactWeightLoader.validateInventory(config(), descriptors));
    }

    /// The registered text and MTP inventory with every row-split object except the embedding in NVFP4.
    private static Map<String, TensorDescriptor> nvfp4Inventory() {
        Qwen38Config config = config();
        List<String> names = new ArrayList<>(List.of(
                "text/token_embedding",
                "text/final_norm",
                "text/output_head",
                "text/draft_head",
                "text/draft_head_token_ids",
                "mtp/input_projection",
                "mtp/embedding_norm",
                "mtp/hidden_norm",
                "mtp/final_norm",
                "mtp/layer/input_norm",
                "mtp/layer/post_attention_norm",
                "mtp/layer/attention/query_key_gate_value",
                "mtp/layer/attention/query_norm",
                "mtp/layer/attention/key_norm",
                "mtp/layer/attention/output",
                "mtp/layer/mlp/gate_up",
                "mtp/layer/mlp/down"));
        for (int index = 0; index < 64; index++) {
            String prefix = "text/layers/" + index;
            List<String> suffixes =
                    new ArrayList<>(List.of("/input_norm", "/post_attention_norm", "/mlp/gate_up", "/mlp/down"));
            suffixes.addAll(
                    config.layerTypes()[index] == LayerType.FULL_ATTENTION
                            ? List.of(
                                    "/attention/query_key",
                                    "/attention/gate_value",
                                    "/attention/query_norm",
                                    "/attention/key_norm",
                                    "/attention/output")
                            : List.of(
                                    "/gdn/a_log",
                                    "/gdn/dt_bias",
                                    "/gdn/convolution",
                                    "/gdn/a_projection",
                                    "/gdn/b_projection",
                                    "/gdn/query_key",
                                    "/gdn/value_z",
                                    "/gdn/norm",
                                    "/gdn/output"));
            for (String suffix : suffixes) names.add(prefix + suffix);
        }
        Map<String, TensorDescriptor> descriptors = new LinkedHashMap<>();
        for (String name : names) {
            var expected = CompactWeightLoader.expectedDescriptor(name);
            WeightFormat format =
                    expected.layout() == WeightLayout.ROW_SPLIT_K128_V1 && !name.equals("text/token_embedding")
                            ? WeightFormat.NVFP4
                            : expected.format();
            descriptors.put(
                    name,
                    new TensorDescriptor(name, expected.shape(), expected.dataType(), format, expected.layout(), 0, 1));
        }
        assertEquals(785, descriptors.size());
        return descriptors;
    }

    private static TensorDescriptor descriptor(String name, long[] shape, WeightFormat format) {
        return new TensorDescriptor(name, shape, TensorDataType.BF16, format, WeightLayout.ROW_SPLIT_K128_V1, 0, 1);
    }

    private static Qwen38Config config() {
        LayerType[] types = new LayerType[64];
        for (int index = 0; index < types.length; index++) {
            types[index] = index % 4 == 3 ? LayerType.FULL_ATTENTION : LayerType.GATED_DELTA_NET;
        }
        return new Qwen38Config(
                248320,
                5120,
                64,
                24,
                4,
                256,
                17408,
                16,
                48,
                128,
                128,
                4,
                1.0e-6,
                10_000_000.0,
                0.25,
                262144,
                "silu",
                types,
                0,
                0,
                0,
                0,
                false,
                true,
                1);
    }
}
