package io.euhedral_execution.inference.core.model_loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactDenseFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactMtpAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class QwenCompactWeightLoaderTest {

    @Test
    void assemblesFusedRuntimeObjectsWithoutSplittingHandles() throws Exception {
        QwenConfig config = config();
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
            if (config.layerTypes()[index] == QwenLayerType.FULL_ATTENTION) {
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

        QwenWeights weights = QwenCompactWeightLoader.assemble(config, handles);

        QwenLayerWeights gdnLayer = weights.layers()[0];
        assertInstanceOf(QwenCompactGatedDeltaNetWeights.class, gdnLayer.mixer());
        assertInstanceOf(QwenCompactDenseFfnWeights.class, gdnLayer.ffn());
        assertInstanceOf(QwenCompactAttentionWeights.class, weights.layers()[3].mixer());
        assertEquals(
                "text/layers/3/attention/query_key",
                ((QwenCompactAttentionWeights) weights.layers()[3].mixer())
                        .queryKey()
                        .name());
        assertInstanceOf(
                QwenCompactMtpAttentionWeights.class, weights.mtp().layer().mixer());
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
        QwenCompactWeightLoader.validateInventory(config(), nvfp4Inventory());
    }

    @Test
    void rejectsAPartialVisionTower() throws Exception {
        Map<String, TensorDescriptor> descriptors = nvfp4Inventory();
        descriptors.put(
                "vision/merger/fc1", descriptor("vision/merger/fc1", new long[] {4608, 4608}, WeightFormat.NVFP4));
        assertThrows(
                QwenWeightLoadException.class, () -> QwenCompactWeightLoader.validateInventory(config(), descriptors));
    }

    @Test
    void rejectsNvfp4WhereADirectTensorIsRegistered() throws Exception {
        Map<String, TensorDescriptor> descriptors = nvfp4Inventory();
        descriptors.put("text/final_norm", descriptor("text/final_norm", new long[] {5120}, WeightFormat.NVFP4));
        assertThrows(
                QwenWeightLoadException.class, () -> QwenCompactWeightLoader.validateInventory(config(), descriptors));
    }

    /// The registered text and MTP inventory with every row-split object except the embedding in NVFP4.
    private static Map<String, TensorDescriptor> nvfp4Inventory() {
        QwenConfig config = config();
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
                    config.layerTypes()[index] == QwenLayerType.FULL_ATTENTION
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
            var expected = QwenCompactWeightLoader.expectedDescriptor(name);
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

    private static QwenConfig config() {
        QwenLayerType[] types = new QwenLayerType[64];
        for (int index = 0; index < types.length; index++) {
            types[index] = index % 4 == 3 ? QwenLayerType.FULL_ATTENTION : QwenLayerType.GATED_DELTA_NET;
        }
        return new QwenConfig(
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
