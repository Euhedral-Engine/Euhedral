package io.euhedral_execution.inference.core.model.qwen4.loader;

import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen4.Qwen4Config;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/// The objects a `qwen4_exp` artifact must hold, derived from its configuration alone: every name, shape,
/// storage format and group. The artifact is valid only if its tables hold exactly this inventory
/// ([Validator]).
///
/// Names: `text/...` for the text model (`text/layers/N/{attn_hc,mlp_hc,gdn,attention,moe,ple}/...`), `mtp/...` for
/// the multi-token-prediction layer and `vision/...` for the vision tower. NVFP4 shapes are the logical `[rows, K]`.
public final class ExpectedInventory {

    private ExpectedInventory() {}

    /// One expected fixed object.
    public record Expected(long[] shape, WeightFormat format, WeightLayout layout, ComponentGroup group) {}

    /// One expected expert bank: its projections' shapes, in record order.
    public record ExpectedBank(
            String name, ComponentGroup group, int layer, int experts, Map<String, long[]> projections) {}

    public record Inventory(Map<String, Expected> tensors, List<ExpectedBank> banks) {}

    public static Inventory expected(Qwen4Config c) {
        Map<String, Expected> tensors = new LinkedHashMap<>();
        List<ExpectedBank> banks = new ArrayList<>();
        long hidden = c.text().hiddenSize();
        long width = (long) c.hyperConnection().count() * hidden;
        add(
                tensors,
                "text/token_embedding",
                ComponentGroup.TOKEN_EMBEDDING,
                c.text().vocabSize(),
                hidden);
        add(tensors, "text/output_head", ComponentGroup.OUTPUT_HEAD, c.text().vocabSize(), hidden);
        hyperConnectionMixer(tensors, "text/hyper_connection_mixer", c, ComponentGroup.HYPER_CONNECTION);
        for (int layer = 0; layer < c.text().numLayers(); layer++) {
            String prefix = "text/layers/" + layer;
            layer(tensors, banks, prefix, layer, c, c.layerType(layer), ComponentGroup.ROUTED_EXPERT, null);
            if (c.hasNgram(layer)) ngram(tensors, prefix + "/ple", c, hidden, width);
        }
        for (int layer = 0; layer < c.mtp().numLayers(); layer++) {
            String prefix = "mtp/layers/" + layer;
            layer(
                    tensors,
                    banks,
                    prefix,
                    layer,
                    c,
                    c.mtp().layerTypes()[layer],
                    ComponentGroup.MTP,
                    ComponentGroup.MTP);
        }
        if (c.mtp().numLayers() > 0) {
            quant(tensors, "mtp/fc_embedding", ComponentGroup.MTP, hidden, hidden);
            quant(tensors, "mtp/fc_hidden", ComponentGroup.MTP, hidden, hidden);
            add(tensors, "mtp/pre_fc_norm_embedding", ComponentGroup.MTP, hidden);
            add(tensors, "mtp/pre_fc_norm_hidden", ComponentGroup.MTP, width);
            hyperConnectionMixer(tensors, "mtp/hyper_connection_mixer", c, ComponentGroup.MTP);
        }
        vision(tensors, c.vision());
        return new Inventory(tensors, banks);
    }

    private static void layer(
            Map<String, Expected> tensors,
            List<ExpectedBank> banks,
            String prefix,
            int layer,
            Qwen4Config c,
            LayerType type,
            ComponentGroup expertGroup,
            ComponentGroup overrideGroup) {
        long hidden = c.text().hiddenSize();
        long width = (long) c.hyperConnection().count() * hidden;
        long lowrank = c.hyperConnection().lowrank();
        ComponentGroup hc = overrideGroup != null ? overrideGroup : ComponentGroup.HYPER_CONNECTION;
        ComponentGroup small = overrideGroup != null ? overrideGroup : ComponentGroup.NORM_SMALL_STATE;
        for (String site : new String[] {"attn_hc", "mlp_hc"}) {
            add(tensors, prefix + "/" + site + "/hc_norm", hc, width);
            add(tensors, prefix + "/" + site + "/input_mix_down", hc, lowrank, width);
            add(tensors, prefix + "/" + site + "/input_mix_up", hc, width, lowrank);
            add(
                    tensors,
                    prefix + "/" + site + "/block_inject",
                    hc,
                    c.hyperConnection().count(),
                    width);
        }
        if (type == LayerType.GATED_DELTA_NET) {
            Qwen4Config.Gdn g = c.gdn();
            ComponentGroup gdn = overrideGroup != null ? overrideGroup : ComponentGroup.GDN;
            long values = (long) g.numValueHeads() * g.valueHeadDim();
            add(tensors, prefix + "/gdn/a_log", small, g.numValueHeads());
            add(tensors, prefix + "/gdn/dt_bias", small, g.numValueHeads());
            add(tensors, prefix + "/gdn/conv1d", gdn, g.convChannels(), 1, g.convKernelDim());
            add(tensors, prefix + "/gdn/in_proj_a", gdn, g.numValueHeads(), hidden);
            add(tensors, prefix + "/gdn/in_proj_b", gdn, g.numValueHeads(), hidden);
            quant(tensors, prefix + "/gdn/in_proj_qkv", gdn, g.convChannels(), hidden);
            quant(tensors, prefix + "/gdn/in_proj_z", gdn, values, hidden);
            quant(tensors, prefix + "/gdn/out_proj", gdn, hidden, values);
            add(tensors, prefix + "/gdn/norm", small, g.valueHeadDim());
        } else {
            Qwen4Config.Attention a = c.attention();
            Qwen4Config.Qsa q = c.qsa();
            ComponentGroup qsa = overrideGroup != null ? overrideGroup : ComponentGroup.QSA;
            long queries = (long) a.numHeads() * a.headDim();
            long keys = (long) a.numKvHeads() * a.headDim();
            quant(tensors, prefix + "/attention/q_proj", qsa, 2 * queries, hidden);
            quant(tensors, prefix + "/attention/k_proj", qsa, keys, hidden);
            quant(tensors, prefix + "/attention/v_proj", qsa, keys, hidden);
            quant(tensors, prefix + "/attention/o_proj", qsa, hidden, queries);
            add(tensors, prefix + "/attention/q_norm", small, a.headDim());
            add(tensors, prefix + "/attention/k_norm", small, a.headDim());
            add(
                    tensors,
                    prefix + "/attention/indexer/index_qk_proj",
                    qsa,
                    (long) (q.numHeads() + q.kvHeads()) * q.headDim(),
                    hidden);
            add(tensors, prefix + "/attention/indexer/q_layernorm", small, q.headDim());
            add(tensors, prefix + "/attention/indexer/k_layernorm", small, q.headDim());
        }
        Qwen4Config.Moe m = c.moe();
        ComponentGroup router = overrideGroup != null ? overrideGroup : ComponentGroup.ROUTER;
        ComponentGroup shared = overrideGroup != null ? overrideGroup : ComponentGroup.SHARED_EXPERT;
        add(tensors, prefix + "/moe/router", router, m.numExperts(), hidden);
        quant(tensors, prefix + "/moe/shared_expert/gate_proj", shared, m.sharedExpertIntermediateSize(), hidden);
        quant(tensors, prefix + "/moe/shared_expert/up_proj", shared, m.sharedExpertIntermediateSize(), hidden);
        quant(tensors, prefix + "/moe/shared_expert/down_proj", shared, hidden, m.sharedExpertIntermediateSize());
        add(tensors, prefix + "/moe/shared_expert_gate", shared, 1, hidden);
        Map<String, long[]> projections = new LinkedHashMap<>();
        projections.put("gate_up", new long[] {2L * m.intermediateSize(), hidden});
        projections.put("down", new long[] {hidden, m.intermediateSize()});
        banks.add(new ExpectedBank(prefix + "/moe/experts", expertGroup, layer, m.numExperts(), projections));
    }

    private static void hyperConnectionMixer(
            Map<String, Expected> tensors, String prefix, Qwen4Config c, ComponentGroup group) {
        long width = (long) c.hyperConnection().count() * c.text().hiddenSize();
        long lowrank = c.hyperConnection().lowrank();
        add(tensors, prefix + "/hc_norm", group, width);
        add(tensors, prefix + "/input_mix_down", group, lowrank, width);
        add(tensors, prefix + "/input_mix_up", group, width, lowrank);
    }

    /// The n-gram tables of one layer with the projections that feed them.
    private static void ngram(Map<String, Expected> tensors, String prefix, Qwen4Config c, long hidden, long width) {
        quant(tensors, prefix + "/key_proj", ComponentGroup.FIXED_TEXT, width, hidden);
        quant(tensors, prefix + "/value_proj", ComponentGroup.FIXED_TEXT, hidden, hidden);
        add(
                tensors,
                prefix + "/conv1d",
                ComponentGroup.FIXED_TEXT,
                width,
                1,
                c.ple().convKernelSize());
        add(tensors, prefix + "/norm_conv", ComponentGroup.NORM_SMALL_STATE, width);
        add(tensors, prefix + "/norm_key", ComponentGroup.NORM_SMALL_STATE, width);
        add(tensors, prefix + "/norm_query", ComponentGroup.NORM_SMALL_STATE, width);
        long rowWidth = hidden / c.ngram().heads();
        for (int shard = 0; shard < c.ngram().splitParts(); shard++)
            tensors.put(
                    prefix + "/ngram/" + shardName(shard),
                    new Expected(
                            new long[] {c.ngram().shardRows(), rowWidth},
                            WeightFormat.NVFP4,
                            WeightLayout.ROW_INTERLEAVED_NVFP4_V1,
                            ComponentGroup.NGRAM));
    }

    /// `shard_000`, `shard_001`, ...
    public static String shardName(int shard) {
        return String.format("shard_%03d", shard);
    }

    private static void vision(Map<String, Expected> tensors, Qwen4Config.Vision v) {
        long hidden = v.hiddenSize();
        long merged = hidden * v.spatialMergeSize() * v.spatialMergeSize();
        for (int block = 0; block < v.depth(); block++) {
            String prefix = "vision/blocks/" + block;
            add(tensors, prefix + "/attn/qkv/weight", ComponentGroup.VISION, 3 * hidden, hidden);
            add(tensors, prefix + "/attn/qkv/bias", ComponentGroup.VISION, 3 * hidden);
            add(tensors, prefix + "/attn/proj/weight", ComponentGroup.VISION, hidden, hidden);
            add(tensors, prefix + "/attn/proj/bias", ComponentGroup.VISION, hidden);
            add(tensors, prefix + "/mlp/linear_fc1/weight", ComponentGroup.VISION, v.intermediateSize(), hidden);
            add(tensors, prefix + "/mlp/linear_fc1/bias", ComponentGroup.VISION, v.intermediateSize());
            add(tensors, prefix + "/mlp/linear_fc2/weight", ComponentGroup.VISION, hidden, v.intermediateSize());
            add(tensors, prefix + "/mlp/linear_fc2/bias", ComponentGroup.VISION, hidden);
            for (String norm : new String[] {"norm1", "norm2"}) {
                add(tensors, prefix + "/" + norm + "/weight", ComponentGroup.VISION, hidden);
                add(tensors, prefix + "/" + norm + "/bias", ComponentGroup.VISION, hidden);
            }
        }
        add(tensors, "vision/merger/linear_fc1/weight", ComponentGroup.VISION, merged, merged);
        add(tensors, "vision/merger/linear_fc1/bias", ComponentGroup.VISION, merged);
        add(tensors, "vision/merger/linear_fc2/weight", ComponentGroup.VISION, v.outHiddenSize(), merged);
        add(tensors, "vision/merger/linear_fc2/bias", ComponentGroup.VISION, v.outHiddenSize());
        add(tensors, "vision/merger/norm/weight", ComponentGroup.VISION, hidden);
        add(tensors, "vision/merger/norm/bias", ComponentGroup.VISION, hidden);
        add(
                tensors,
                "vision/patch_embed/proj/weight",
                ComponentGroup.VISION,
                hidden,
                v.inChannels(),
                v.temporalPatchSize(),
                v.patchSize(),
                v.patchSize());
        add(tensors, "vision/patch_embed/proj/bias", ComponentGroup.VISION, hidden);
        add(tensors, "vision/pos_embed/weight", ComponentGroup.VISION, v.numPositionEmbeddings(), hidden);
    }

    private static void add(Map<String, Expected> tensors, String name, ComponentGroup group, long... shape) {
        if (tensors.put(name, new Expected(shape, WeightFormat.BF16, WeightLayout.CONTIGUOUS_LE_V1, group)) != null)
            throw new IllegalStateException("duplicate expected object " + name);
    }

    private static void quant(Map<String, Expected> tensors, String name, ComponentGroup group, long rows, long k) {
        if (tensors.put(
                        name,
                        new Expected(new long[] {rows, k}, WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1, group))
                != null) throw new IllegalStateException("duplicate expected object " + name);
    }
}
