package io.euhedral_execution.inference.quantization.trellis;

import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/// Qwen3.8's text linears: which object's Hessian each one shares, where its rows sit in the layer's fused
/// artifact objects, and the allocation inputs (priority and promotion group) of the budgeted ones.
public final class Linears {

    private Linears() {}

    private static final String TEXT = "model.language_model.layers.";

    /// A linear of a decoder layer: `tensor` is its name without `.weight`, `[rows, columns]` = `[n, k]`,
    /// `hessianObject` the object whose observed Hessian it shares, `quantized` false when `n` or `k` is not a
    /// multiple of 128.
    public record Linear(String tensor, int rows, int columns, String hessianObject, boolean quantized) {}

    /// Where `rows` rows of a linear (from `linearRow`) sit in an object (from `objectRow`).
    public record Placement(String object, long objectRow, long linearRow, long rows) {}

    /// The linears of decoder layer `index`, in module order.
    public static List<Linear> layer(Qwen38Objects checkpoint, int index) {
        String p = TEXT + index + ".";
        String o = "text/layers/" + index + "/";
        List<Linear> all = new ArrayList<>();
        if (checkpoint.config.fullAttention()[index]) {
            for (String role : new String[] {"q_proj", "k_proj", "v_proj"})
                all.add(linear(checkpoint, p + "self_attn." + role, o + "attention/query_key"));
            all.add(linear(checkpoint, p + "self_attn.o_proj", o + "attention/output"));
        } else {
            all.add(linear(checkpoint, p + "linear_attn.in_proj_qkv", o + "gdn/query_key"));
            all.add(linear(checkpoint, p + "linear_attn.in_proj_z", o + "gdn/query_key"));
            all.add(linear(checkpoint, p + "linear_attn.out_proj", o + "gdn/output"));
            all.add(linear(checkpoint, p + "linear_attn.in_proj_a", o + "gdn/query_key"));
            all.add(linear(checkpoint, p + "linear_attn.in_proj_b", o + "gdn/query_key"));
        }
        all.add(linear(checkpoint, p + "mlp.gate_proj", o + "mlp/gate_up"));
        all.add(linear(checkpoint, p + "mlp.up_proj", o + "mlp/gate_up"));
        all.add(linear(checkpoint, p + "mlp.down_proj", o + "mlp/down"));
        return all;
    }

    private static Linear linear(Qwen38Objects checkpoint, String tensor, String hessianObject) {
        long[] shape = checkpoint.store().tensor(tensor + ".weight").shape();
        return new Linear(
                tensor, (int) shape[0], (int) shape[1], hessianObject, shape[0] % 128 == 0 && shape[1] % 128 == 0);
    }

    /// The rows of `tensor` (a name from [#layer]) inside the layer's objects, in object order. Empty for the
    /// GDN a/b projections, which no object holds.
    public static List<Placement> placements(Qwen38Objects checkpoint, int index, String tensor) {
        String weight = tensor + ".weight";
        List<Placement> found = new ArrayList<>();
        for (String name : checkpoint.layerObjects(index)) {
            long objectRow = 0;
            for (Qwen38Objects.Piece piece : checkpoint.object(name).pieces()) {
                if (piece.tensor().equals(weight))
                    found.add(new Placement(name, objectRow, piece.firstRow(), piece.rows()));
                objectRow += piece.rows();
            }
        }
        return found;
    }

    /// Swift-1.5's linear shapes `[n, k]` by role, for tests without a checkpoint.
    public static final Map<String, long[]> SWIFT_SHAPES = Map.of(
            "q_proj", new long[] {12288, 5120},
            "k_proj", new long[] {1024, 5120},
            "v_proj", new long[] {1024, 5120},
            "o_proj", new long[] {5120, 6144},
            "in_proj_qkv", new long[] {10240, 5120},
            "in_proj_z", new long[] {6144, 5120},
            "out_proj", new long[] {5120, 6144},
            "gate_proj", new long[] {17408, 5120},
            "up_proj", new long[] {17408, 5120},
            "down_proj", new long[] {5120, 17408});

    /// The budgeted linears of a Qwen3.8 text model with `layers` layers (every fourth layer, from the fourth, is
    /// full attention), in exllamav3's module-tree order, with its allocation inputs: the attention module has
    /// priority 2 (`q_priority = 2 + select_hq_bits`) and promotes q, k and v together (`.qkv`) and o alone (`.o`);
    /// the GDN module has priority 1 and promotes in_proj_qkv and in_proj_z together (`.qkvz`) and out_proj alone
    /// (`.o`); the gated MLP has priority 0 and promotes up and gate together (`.gu`) and down alone (`.d`). The
    /// GDN a/b projections are not quantized and are not listed.
    public static List<Allocation.Linear> qwen38(int layers, Map<String, long[]> shapes) {
        List<Allocation.Linear> all = new ArrayList<>();
        for (int i = 0; i < layers; i++) {
            String p = TEXT + i + ".";
            boolean full = i % 4 == 3;
            String attention = p + (full ? "self_attn" : "linear_attn");
            int priority = full ? 2 : 1;
            String[][] projections = full
                    ? new String[][] {{"q_proj", ".qkv"}, {"k_proj", ".qkv"}, {"v_proj", ".qkv"}, {"o_proj", ".o"}}
                    : new String[][] {{"in_proj_qkv", ".qkvz"}, {"in_proj_z", ".qkvz"}, {"out_proj", ".o"}};
            for (String[] projection : projections)
                all.add(budgeted(
                        attention + "." + projection[0],
                        i,
                        shapes.get(projection[0]),
                        priority,
                        attention + projection[1]));
            String mlp = p + "mlp";
            all.add(budgeted(mlp + ".up_proj", i, shapes.get("up_proj"), 0, mlp + ".gu"));
            all.add(budgeted(mlp + ".gate_proj", i, shapes.get("gate_proj"), 0, mlp + ".gu"));
            all.add(budgeted(mlp + ".down_proj", i, shapes.get("down_proj"), 0, mlp + ".d"));
        }
        return all;
    }

    private static Allocation.Linear budgeted(String name, int layer, long[] shape, int priority, String group) {
        return new Allocation.Linear(name, layer, shape[0] * shape[1], priority, group);
    }
}
