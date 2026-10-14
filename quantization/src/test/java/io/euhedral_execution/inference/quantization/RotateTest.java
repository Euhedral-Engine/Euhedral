package io.euhedral_execution.inference.quantization;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/// The residual rotation is orthogonal and each tensor gets the change its place in the residual stream needs.
class RotateTest {

    @Test
    void hadamard20IsHadamard() {
        int[][] h = Rotate.hadamard20();
        for (int i = 0; i < 20; i++)
            for (int j = 0; j < 20; j++) {
                assertEquals(1, Math.abs(h[i][j]));
                int dot = 0;
                for (int k = 0; k < 20; k++) dot += h[i][k] * h[j][k];
                assertEquals(i == j ? 20 : 0, dot);
            }
    }

    @Test
    void rotationRowsAreOrthonormal() {
        float[] q = Rotate.rotation(7);
        int n = Rotate.HIDDEN;
        int[] rows = {0, 1, 255, 256, 2600, 4864, 5119};
        for (int a : rows)
            for (int b : rows) {
                double dot = 0;
                for (int k = 0; k < n; k++) dot += (double) q[a * n + k] * q[b * n + k];
                assertEquals(a == b ? 1 : 0, dot, 1e-6, a + "·" + b);
            }
    }

    @Test
    void rolesFollowTheResidualStream() {
        String layer = "model.language_model.layers.3.";
        assertEquals(Rotate.Role.EMBEDDING, Rotate.role("model.language_model.embed_tokens.weight"));
        assertEquals(Rotate.Role.INPUT, Rotate.role("lm_head.weight"));
        assertEquals(Rotate.Role.INPUT, Rotate.role(layer + "self_attn.q_proj.weight"));
        assertEquals(Rotate.Role.INPUT, Rotate.role("model.language_model.layers.0.linear_attn.in_proj_b.weight"));
        assertEquals(Rotate.Role.INPUT, Rotate.role(layer + "mlp.up_proj.weight"));
        assertEquals(Rotate.Role.OUTPUT, Rotate.role(layer + "self_attn.o_proj.weight"));
        assertEquals(Rotate.Role.OUTPUT, Rotate.role("model.language_model.layers.0.linear_attn.out_proj.weight"));
        assertEquals(Rotate.Role.OUTPUT, Rotate.role(layer + "mlp.down_proj.weight"));
        assertEquals(Rotate.Role.NORM, Rotate.role(layer + "post_attention_layernorm.weight"));
        assertEquals(Rotate.Role.NORM, Rotate.role("model.language_model.norm.weight"));
        assertEquals(Rotate.Role.COPY, Rotate.role(layer + "self_attn.q_norm.weight"));
        assertEquals(Rotate.Role.COPY, Rotate.role("model.language_model.layers.0.linear_attn.conv1d.weight"));
        assertEquals(Rotate.Role.COPY, Rotate.role("mtp.fc.weight"));
        assertEquals(layer + "post_attention_layernorm.weight", Rotate.normOf(layer + "mlp.gate_proj.weight"));
        assertEquals(layer + "input_layernorm.weight", Rotate.normOf(layer + "self_attn.v_proj.weight"));
        assertEquals("model.language_model.norm.weight", Rotate.normOf("lm_head.weight"));
    }
}
