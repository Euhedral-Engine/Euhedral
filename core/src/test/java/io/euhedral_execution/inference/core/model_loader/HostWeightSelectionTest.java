package io.euhedral_execution.inference.core.model_loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class HostWeightSelectionTest {

    private static TensorDescriptor tensor(String name, WeightLayout layout, long bytes) {
        return new TensorDescriptor(name, new long[] {1, 1}, TensorDataType.BF16, WeightFormat.NVFP4, layout, 0, bytes);
    }

    private static QwenArtifact artifact(WeightLayout layout) {
        List<TensorDescriptor> tensors = new ArrayList<>();
        tensors.add(tensor("text/token_embedding", WeightLayout.ROW_SPLIT_K128_V1, 500));
        tensors.add(tensor("text/output_head", layout, 700));
        for (int layer = 0; layer < 4; layer++) {
            tensors.add(tensor("text/layers/" + layer + "/gdn/query_key", layout, 10));
            tensors.add(tensor("text/layers/" + layer + "/mlp/down", layout, 100));
        }
        return new QwenArtifact(null, null, tensors.toArray(TensorDescriptor[]::new));
    }

    @Test
    void theEmbeddingGoesFirstThenTheSmallestProjections() {
        for (WeightLayout layout : List.of(WeightLayout.ROW_SPLIT_K128_V1, WeightLayout.ROW_SPLIT_K128_SD4_V1)) {
            QwenArtifact artifact = artifact(layout);
            assertEquals(List.of(), List.copyOf(HostWeightSelection.select(artifact, 0)));
            assertEquals(List.of("text/token_embedding"), List.copyOf(HostWeightSelection.select(artifact, 500)));
            assertEquals(
                    List.of(
                            "text/token_embedding",
                            "text/layers/0/gdn/query_key",
                            "text/layers/1/gdn/query_key",
                            "text/layers/2/gdn/query_key",
                            "text/layers/3/gdn/query_key",
                            "text/layers/0/mlp/down",
                            "text/layers/2/mlp/down"),
                    List.copyOf(HostWeightSelection.select(artifact, 500 + 40 + 150)));
            var everything = HostWeightSelection.select(artifact, 1L << 40);
            assertEquals(9, everything.size());
            assertTrue(!everything.contains("text/output_head"));
        }
    }
}
