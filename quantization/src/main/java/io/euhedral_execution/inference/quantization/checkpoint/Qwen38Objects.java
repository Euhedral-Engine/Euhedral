package io.euhedral_execution.inference.quantization.checkpoint;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/// The quantized objects of a Qwen3.8 artifact as row ranges of the BF16 checkpoint tensors, fused exactly as the
/// converter fuses them (tools/euhedral_artifacts/inventory.py, tools/gptq/qwen38.py):
/// - full attention: q_proj interleaves per head [query 256 rows; gate 256 rows], split into `query_key` =
///   [query (6144); k_proj (1024)] and `gate_value` = [gate (6144); v_proj (1024)];
/// - GDN: `query_key` = in_proj_qkv rows 0..4095, `value_z` = [in_proj_qkv rows 4096..10239; in_proj_z];
/// - MLP: `gate_up` = [gate_proj; up_proj];
/// - MTP: `query_key_gate_value` = [query; k; gate; v], `input_projection` = mtp.fc.
public final class Qwen38Objects {

    /// Rows `[firstRow, firstRow + rows)` of a checkpoint tensor.
    public record Piece(String tensor, long firstRow, long rows) {}

    /// An object: its pieces in order, and its shape [rows, columns].
    public record ObjectSource(String name, List<Piece> pieces, long rows, long columns) {

        public long bytes() {
            return rows * columns * Short.BYTES;
        }
    }

    private static final String TEXT = "model.language_model.";

    private final Safetensors store;
    public final Config config;

    /// The dimensions the calibration needs, from `config.json`'s text_config.
    public record Config(
            int hidden,
            int intermediate,
            int layers,
            int heads,
            int kvHeads,
            int headDim,
            int rotaryDim,
            int keyHeads,
            int valueHeads,
            int keyDim,
            int valueDim,
            int convKernel,
            double eps,
            double theta,
            int vocabulary,
            boolean[] fullAttention) {}

    public Qwen38Objects(Safetensors store, Path checkpoint) throws IOException {
        this.store = store;
        JsonNode text = new ObjectMapper()
                .readTree(checkpoint.resolve("config.json").toFile())
                .get("text_config");
        int layers = text.get("num_hidden_layers").asInt();
        boolean[] full = new boolean[layers];
        JsonNode types = text.get("layer_types");
        for (int i = 0; i < layers; i++) full[i] = types.get(i).asText().equals("full_attention");
        int headDim = text.get("head_dim").asInt();
        this.config = new Config(
                text.get("hidden_size").asInt(),
                text.get("intermediate_size").asInt(),
                layers,
                text.get("num_attention_heads").asInt(),
                text.get("num_key_value_heads").asInt(),
                headDim,
                (int) Math.round(headDim * text.get("partial_rotary_factor").asDouble()),
                text.get("linear_num_key_heads").asInt(),
                text.get("linear_num_value_heads").asInt(),
                text.get("linear_key_head_dim").asInt(),
                text.get("linear_value_head_dim").asInt(),
                text.get("linear_conv_kernel_dim").asInt(),
                text.get("rms_norm_eps").asDouble(),
                text.get("rope_parameters").get("rope_theta").asDouble(),
                text.get("vocab_size").asInt(),
                full);
    }

    /// The quantized object names of decoder layer `index`, in the order the layer reads them.
    public List<String> layerObjects(int index) {
        String o = "text/layers/" + index + "/";
        return config.fullAttention[index]
                ? List.of(
                        o + "attention/query_key",
                        o + "attention/gate_value",
                        o + "attention/output",
                        o + "mlp/gate_up",
                        o + "mlp/down")
                : List.of(o + "gdn/query_key", o + "gdn/value_z", o + "gdn/output", o + "mlp/gate_up", o + "mlp/down");
    }

    public static final List<String> MTP_OBJECTS = List.of(
            "mtp/input_projection",
            "mtp/layer/attention/query_key_gate_value",
            "mtp/layer/attention/output",
            "mtp/layer/mlp/gate_up",
            "mtp/layer/mlp/down");

    public ObjectSource object(String name) {
        int hidden = config.hidden;
        if (name.startsWith("text/layers/")) {
            String[] parts = name.split("/", 4);
            int index = Integer.parseInt(parts[2]);
            return layerObject(name, TEXT + "layers." + index + ".", parts[3]);
        }
        if (name.startsWith("mtp/layer/"))
            return layerObject(name, "mtp.layers.0.", name.substring("mtp/layer/".length()));
        return switch (name) {
            case "mtp/input_projection" -> whole(name, "mtp.fc.weight");
            case "text/output_head" -> whole(name, "lm_head.weight");
            default -> throw new IllegalArgumentException("unknown object " + name + " (hidden " + hidden + ")");
        };
    }

    private ObjectSource layerObject(String name, String p, String role) {
        int qRows = config.heads * config.headDim;
        int kvRows = config.kvHeads * config.headDim;
        return switch (role) {
            case "attention/query_key" ->
                fuse(name, concat(attentionHalf(p, false), List.of(full(p + "self_attn.k_proj.weight"))));
            case "attention/gate_value" ->
                fuse(name, concat(attentionHalf(p, true), List.of(full(p + "self_attn.v_proj.weight"))));
            case "attention/query_key_gate_value" ->
                fuse(
                        name,
                        concat(
                                concat(attentionHalf(p, false), List.of(full(p + "self_attn.k_proj.weight"))),
                                concat(attentionHalf(p, true), List.of(full(p + "self_attn.v_proj.weight")))));
            case "attention/output" -> whole(name, p + "self_attn.o_proj.weight");
            case "gdn/query_key" ->
                fuse(
                        name,
                        List.of(new Piece(
                                p + "linear_attn.in_proj_qkv.weight", 0, 2L * config.keyHeads * config.keyDim)));
            case "gdn/value_z" -> {
                long qk = 2L * config.keyHeads * config.keyDim;
                long v = (long) config.valueHeads * config.valueDim;
                yield fuse(
                        name,
                        List.of(
                                new Piece(p + "linear_attn.in_proj_qkv.weight", qk, v),
                                full(p + "linear_attn.in_proj_z.weight")));
            }
            case "gdn/output" -> whole(name, p + "linear_attn.out_proj.weight");
            case "mlp/gate_up" -> fuse(name, List.of(full(p + "mlp.gate_proj.weight"), full(p + "mlp.up_proj.weight")));
            case "mlp/down" -> whole(name, p + "mlp.down_proj.weight");
            default ->
                throw new IllegalArgumentException(
                        "unknown layer object " + name + " (q " + qRows + ", kv " + kvRows + ")");
        };
    }

    /// The query (or gate) rows of a per-head interleaved q_proj: head h owns rows h*2*headDim + [0, headDim) for
    /// the query and + [headDim, 2*headDim) for the gate.
    private List<Piece> attentionHalf(String p, boolean gate) {
        String tensor = p + "self_attn.q_proj.weight";
        List<Piece> pieces = new ArrayList<>();
        for (int h = 0; h < config.heads; h++) {
            pieces.add(new Piece(tensor, (long) h * 2 * config.headDim + (gate ? config.headDim : 0), config.headDim));
        }
        return pieces;
    }

    private Piece full(String tensor) {
        return new Piece(tensor, 0, store.tensor(tensor).shape()[0]);
    }

    private ObjectSource whole(String name, String tensor) {
        return fuse(name, List.of(full(tensor)));
    }

    private ObjectSource fuse(String name, List<Piece> pieces) {
        long columns = -1, rows = 0;
        for (Piece piece : pieces) {
            Safetensors.Tensor tensor = store.tensor(piece.tensor());
            if (!tensor.dtype().equals("BF16") || tensor.shape().length != 2) {
                throw new IllegalArgumentException(piece.tensor() + " is not a BF16 matrix");
            }
            if (columns >= 0 && tensor.shape()[1] != columns)
                throw new IllegalArgumentException(name + ": mixed widths");
            columns = tensor.shape()[1];
            if (piece.firstRow() + piece.rows() > tensor.shape()[0])
                throw new IllegalArgumentException(name + ": rows out of range");
            rows += piece.rows();
        }
        return new ObjectSource(name, List.copyOf(pieces), rows, columns);
    }

    private static <T> List<T> concat(List<T> a, List<T> b) {
        List<T> all = new ArrayList<>(a);
        all.addAll(b);
        return all;
    }

    /// Reads object `source` (BF16, row-major [rows, columns]) into `destination`.
    public void read(ObjectSource source, MemorySegment destination) throws IOException {
        if (destination.byteSize() != source.bytes())
            throw new IllegalArgumentException(source.name() + ": wrong size");
        long rowBytes = source.columns() * Short.BYTES;
        long at = 0;
        for (Piece piece : source.pieces()) {
            long bytes = piece.rows() * rowBytes;
            store.read(piece.tensor(), piece.firstRow() * rowBytes, destination.asSlice(at, bytes));
            at += bytes;
        }
    }

    /// Reads a whole BF16 vector or matrix tensor of the checkpoint (norms, convolution, a/b projections).
    public void readTensor(String tensor, MemorySegment destination) throws IOException {
        store.read(tensor, destination);
    }

    public Safetensors store() {
        return store;
    }
}
