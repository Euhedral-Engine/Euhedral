package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightStaging;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;

/// The fixed objects of a loaded [Qwen4Model] as operators use them, wherever the plan put them: a resident or
/// host-mapped tensor is read at its address, a host-staged one is copied into the next slot of the model's staging
/// ring when its address is asked for ([Weight]). One thread submits to one stream at a time; ring slots are
/// reused in turn, which stream order makes safe: a copy into a slot queues behind the kernels that read the slot's
/// previous tensor.
public final class Weights {

    private final Qwen4Model model;
    private final ExecutionGpu gpu;
    private int nextSlot;

    public Weights(Qwen4Model model, ExecutionGpu gpu) {
        this.model = model;
        this.gpu = gpu;
    }

    /// The object named `name` (for example `text/layers/3/attention/q_proj`).
    public Weight of(String name) {
        TensorHandle handle = this.model.tensor(name);
        if (handle.deviceAddress() != 0) return Weight.of(handle.deviceAddress(), handle.byteSize());
        if (handle.hostBacked()) return new Staged(handle);
        throw new IllegalStateException(name + " is neither on the device nor staged");
    }

    private final class Staged implements Weight {
        private final TensorHandle handle;

        private Staged(TensorHandle handle) {
            this.handle = handle;
        }

        @Override
        public long address() {
            WeightStaging ring = Weights.this.model.staging();
            if (ring == null || this.handle.byteSize() > ring.slotBytes())
                throw new IllegalStateException("no staging slot holds " + this.handle.name());
            long slot = ring.slotAddress(Weights.this.nextSlot);
            Weights.this.nextSlot = (Weights.this.nextSlot + 1) % ring.slots();
            Weights.this.gpu.copyHostWeightsToDevice(slot, this.handle.hostAddress(), this.handle.byteSize());
            return slot;
        }

        @Override
        public long bytes() {
            return this.handle.byteSize();
        }
    }

    private static String layer(int layer) {
        return "text/layers/" + layer + "/";
    }

    /// The attention-side gated residual of `layer`.
    public HyperConnection.Weights attentionResidual(int layer) {
        return hyperConnection(layer(layer) + "attn_hc", true);
    }

    /// The MoE-side gated residual of `layer`.
    public HyperConnection.Weights moeResidual(int layer) {
        return hyperConnection(layer(layer) + "mlp_hc", true);
    }

    /// The final mixer after the last layer: no injection projection.
    public HyperConnection.Weights finalMixer() {
        return hyperConnection("text/hyper_connection_mixer", false);
    }

    private HyperConnection.Weights hyperConnection(String prefix, boolean inject) {
        return new HyperConnection.Weights(
                of(prefix + "/hc_norm"),
                of(prefix + "/input_mix_down"),
                of(prefix + "/input_mix_up"),
                inject ? of(prefix + "/block_inject") : null);
    }

    public GdnLayer.Weights gdn(int layer) {
        String p = layer(layer) + "gdn/";
        return new GdnLayer.Weights(
                of(p + "in_proj_qkv"),
                of(p + "in_proj_z"),
                of(p + "out_proj"),
                of(p + "in_proj_a"),
                of(p + "in_proj_b"),
                of(p + "conv1d"),
                of(p + "a_log"),
                of(p + "dt_bias"),
                of(p + "norm"));
    }

    public QsaLayer.Weights qsa(int layer) {
        String p = layer(layer) + "attention/";
        return new QsaLayer.Weights(
                of(p + "q_proj"),
                of(p + "k_proj"),
                of(p + "v_proj"),
                of(p + "o_proj"),
                of(p + "q_norm"),
                of(p + "k_norm"),
                of(p + "indexer/index_qk_proj"),
                of(p + "indexer/q_layernorm"),
                of(p + "indexer/k_layernorm"));
    }

    public MoeLayer.Weights moe(int layer) {
        String p = layer(layer) + "moe/";
        return new MoeLayer.Weights(
                of(p + "router"),
                of(p + "shared_expert/gate_proj"),
                of(p + "shared_expert/up_proj"),
                of(p + "shared_expert/down_proj"),
                of(p + "shared_expert_gate"),
                this.paddedSharedDown.get(layer));
    }

    /// Each layer's shared-expert down projection padded to [MoeLayer#paddedWidth] input columns, for prefill;
    /// none when the width needs no padding or the tensor is not plain NVFP4 on the device.
    private final java.util.Map<Integer, Weight> paddedSharedDown = new java.util.HashMap<>();
    private final java.util.List<Long> owned = new java.util.ArrayList<>();

    /// Builds the padded shared-expert down projections of `layers` layers (`hidden` outputs, `width` inputs) on
    /// the device and waits for them. Before any graph runs.
    void padSharedDown(int layers, int hidden, int width) {
        int padded = MoeLayer.paddedWidth(width);
        if (padded == width) return;
        for (int layer = 0; layer < layers; layer++) {
            TensorHandle handle = this.model.tensor(layer(layer) + "moe/shared_expert/down_proj");
            if (handle.deviceAddress() == 0 || handle.byteSize() != MoeOps.nvfp4Bytes(hidden, width)) continue;
            long bytes = MoeOps.nvfp4Bytes(hidden, padded);
            long target = this.gpu.allocate(bytes);
            this.owned.add(target);
            MoeOps.nvfp4PadK(this.gpu, handle.deviceAddress(), target, hidden, width, padded);
            this.paddedSharedDown.put(layer, Weight.of(target, bytes));
        }
        this.gpu.synchronize();
    }

    /// Frees the padded copies. After every graph retired.
    void close() {
        for (long address : this.owned) this.gpu.free(address);
        this.owned.clear();
        this.paddedSharedDown.clear();
    }

    public Ple.Weights ple(int layer) {
        String p = layer(layer) + "ple/";
        return new Ple.Weights(
                of(p + "key_proj"),
                of(p + "value_proj"),
                of(p + "norm_key"),
                of(p + "norm_query"),
                of(p + "norm_conv"),
                of(p + "conv1d"));
    }

    /// The token embedding table (device or host mapped).
    public Weight embedding() {
        return of("text/token_embedding");
    }

    /// The output head, `[vocabulary][hidden]` BF16 (device or host mapped).
    public Weight head() {
        return of("text/output_head");
    }
}
