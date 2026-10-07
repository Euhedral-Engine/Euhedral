package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.inference.core.artifact.ArtifactFormatException;
import io.euhedral_execution.inference.core.model.qwen4.loader.LayerType;
import io.euhedral_execution.inference.core.model.qwen4.loader.Metadata;
import java.util.LinkedHashMap;
import java.util.Map;

/// The topology and hyper-parameters of a `qwen4_exp` model, as an artifact records them. Everything the runtime
/// needs is here: no `config.json` is read after conversion.
public record Qwen4Config(
        Text text,
        Attention attention,
        Gdn gdn,
        Qsa qsa,
        Moe moe,
        HyperConnection hyperConnection,
        Ngram ngram,
        Ple ple,
        Mtp mtp,
        Vision vision) {

    public static final String ARCHITECTURE = "qwen4_exp";

    public record Text(
            int vocabSize,
            int hiddenSize,
            int numLayers,
            int maxPositionEmbeddings,
            double rmsNormEpsilon,
            String hiddenActivation,
            boolean tieWordEmbeddings,
            int bosTokenId,
            int eosTokenId,
            LayerType[] layerTypes) {}

    public record Attention(
            int numHeads,
            int numKvHeads,
            int headDim,
            double partialRotaryFactor,
            double ropeTheta,
            int[] mropeSection,
            boolean mropeInterleaved,
            String outputGate,
            int fullAttentionInterval) {}

    public record Gdn(
            int numKeyHeads, int numValueHeads, int keyHeadDim, int valueHeadDim, int convKernelDim, String ssmDtype) {

        /// Channels the input convolution runs over: query, key and value.
        public int convChannels() {
            return 2 * numKeyHeads * keyHeadDim + numValueHeads * valueHeadDim;
        }
    }

    /// Sparse attention: each query attends to at most `indexerBudget` keys chosen by an indexer with `numHeads`
    /// query heads and `kvHeads` key heads of `headDim`, over keys compressed by `compressRatio`.
    public record Qsa(int indexerBudget, int compressRatio, int headDim, int kvHeads, int numHeads) {}

    public record Moe(int numExperts, int expertsPerToken, int intermediateSize, int sharedExpertIntermediateSize) {}

    /// Hyper-connections: `count` parallel residual streams, mixed through a rank-`lowrank` projection.
    public record HyperConnection(int count, int lowrank) {}

    /// The n-gram embedding: `(size - 1) * headsPerNgram` hash heads, each addressing a prime-sized slice of one
    /// table split row-wise into `splitParts` shards of `shardRows` rows.
    public record Ngram(
            int size,
            long vocabSizeBase,
            int headsPerNgram,
            int vocabDivisibleBy,
            int splitParts,
            long shardRows,
            long[] layerMultipliers,
            long[] headsOffsets,
            long[] headsVocabSizes) {

        public int heads() {
            return (size - 1) * headsPerNgram;
        }

        public long totalRows() {
            return shardRows * splitParts;
        }
    }

    /// The layers that hold the n-gram tables and their projections, from 0.
    public record Ple(int[] layers, int embedDim, int convKernelSize) {}

    public record Mtp(
            int numLayers,
            boolean hybrid,
            LayerType[] layerTypes,
            double ropeTheta,
            boolean useDedicatedEmbeddings,
            int useHiddenStateFromLayer) {}

    public record Vision(
            int depth,
            int hiddenSize,
            int intermediateSize,
            int numHeads,
            int inChannels,
            int patchSize,
            int spatialMergeSize,
            int temporalPatchSize,
            int outHiddenSize,
            int numPositionEmbeddings,
            String hiddenActivation,
            int[] deepstackVisualIndexes,
            int imageTokenId,
            int videoTokenId,
            int startTokenId,
            int endTokenId) {}

    public LayerType layerType(int layer) {
        return this.text.layerTypes()[layer];
    }

    public int sparseAttentionLayers() {
        int count = 0;
        for (LayerType type : this.text.layerTypes()) if (type == LayerType.SPARSE_ATTENTION) count++;
        return count;
    }

    public int gdnLayers() {
        return this.text.numLayers() - sparseAttentionLayers();
    }

    /// Whether `layer` holds the n-gram tables.
    public boolean hasNgram(int layer) {
        for (int candidate : this.ple.layers()) if (candidate == layer) return true;
        return false;
    }

    public static Qwen4Config fromMetadata(Map<String, Object> metadata) throws ArtifactFormatException {
        var in = new Metadata.Reader(metadata);
        if (!ARCHITECTURE.equals(in.string("architecture")))
            throw new ArtifactFormatException("metadata architecture is not " + ARCHITECTURE);
        in.string("source.model_type");
        String[] layerNames = in.strings("text.layer_types");
        LayerType[] layerTypes = layerTypes(layerNames);
        Text text = new Text(
                in.count("text.vocab_size"),
                in.count("text.hidden_size"),
                in.count("text.num_hidden_layers"),
                in.count("text.max_position_embeddings"),
                in.real("text.rms_norm_eps"),
                in.string("text.hidden_act"),
                in.bool("text.tie_word_embeddings"),
                in.signed("text.bos_token_id"),
                in.signed("text.eos_token_id"),
                layerTypes);
        Attention attention = new Attention(
                in.count("attention.num_heads"),
                in.count("attention.num_kv_heads"),
                in.count("attention.head_dim"),
                in.real("attention.partial_rotary_factor"),
                in.real("attention.rope_theta"),
                in.ints("attention.mrope_section"),
                in.bool("attention.mrope_interleaved"),
                in.string("attention.output_gate"),
                in.count("attention.full_attention_interval"));
        Gdn gdn = new Gdn(
                in.count("gdn.num_key_heads"),
                in.count("gdn.num_value_heads"),
                in.count("gdn.key_head_dim"),
                in.count("gdn.value_head_dim"),
                in.count("gdn.conv_kernel_dim"),
                in.string("gdn.ssm_dtype"));
        Qsa qsa = new Qsa(
                in.count("qsa.indexer_budget"),
                in.count("qsa.compress_ratio"),
                in.count("qsa.head_dim"),
                in.count("qsa.kv_heads"),
                in.count("qsa.num_heads"));
        Moe moe = new Moe(
                in.count("moe.num_experts"),
                in.count("moe.experts_per_token"),
                in.count("moe.intermediate_size"),
                in.count("moe.shared_expert_intermediate_size"));
        HyperConnection hc = new HyperConnection(in.count("hc.count"), in.count("hc.lowrank"));
        Ngram ngram = new Ngram(
                in.count("ngram.size"),
                in.integer("ngram.vocab_size_base"),
                in.count("ngram.heads_per_ngram"),
                in.count("ngram.vocab_divisible_by"),
                in.count("ngram.split_parts"),
                in.integer("ngram.shard_rows"),
                in.longs("ngram.layer_multipliers"),
                in.longs("ngram.heads_offsets"),
                in.longs("ngram.heads_vocab_sizes"));
        Ple ple = new Ple(in.ints("ple.layers"), in.count("ple.embed_dim"), in.count("ple.conv_kernel_size"));
        Mtp mtp = new Mtp(
                in.count("mtp.num_layers"),
                in.bool("mtp.hybrid"),
                layerTypes(in.strings("mtp.layer_types")),
                in.real("mtp.rope_theta"),
                in.bool("mtp.use_dedicated_embeddings"),
                in.signed("mtp.use_hidden_state_from_layer"));
        Vision vision = new Vision(
                in.count("vision.depth"),
                in.count("vision.hidden_size"),
                in.count("vision.intermediate_size"),
                in.count("vision.num_heads"),
                in.count("vision.in_channels"),
                in.count("vision.patch_size"),
                in.count("vision.spatial_merge_size"),
                in.count("vision.temporal_patch_size"),
                in.count("vision.out_hidden_size"),
                in.count("vision.num_position_embeddings"),
                in.string("vision.hidden_act"),
                in.ints("vision.deepstack_visual_indexes"),
                in.signed("vision.image_token_id"),
                in.signed("vision.video_token_id"),
                in.signed("vision.start_token_id"),
                in.signed("vision.end_token_id"));
        String format = in.string("quant.format");
        long block = in.integer("quant.block");
        String scale = in.string("quant.scale_dtype");
        if (!format.equals("nvfp4") || block != 16 || !scale.equals("e4m3"))
            throw new ArtifactFormatException(
                    "unsupported quantization " + format + " block " + block + " scales " + scale);
        in.finish();
        Qwen4Config config = new Qwen4Config(text, attention, gdn, qsa, moe, hc, ngram, ple, mtp, vision);
        config.validate();
        return config;
    }

    /// The metadata an artifact stores for this configuration.
    public Map<String, Object> toMetadata() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("architecture", ARCHITECTURE);
        m.put("source.model_type", ARCHITECTURE);
        m.put("text.vocab_size", (long) text.vocabSize());
        m.put("text.hidden_size", (long) text.hiddenSize());
        m.put("text.num_hidden_layers", (long) text.numLayers());
        m.put("text.max_position_embeddings", (long) text.maxPositionEmbeddings());
        m.put("text.rms_norm_eps", text.rmsNormEpsilon());
        m.put("text.hidden_act", text.hiddenActivation());
        m.put("text.tie_word_embeddings", text.tieWordEmbeddings());
        m.put("text.bos_token_id", (long) text.bosTokenId());
        m.put("text.eos_token_id", (long) text.eosTokenId());
        m.put("text.layer_types", sourceNames(text.layerTypes()));
        m.put("attention.num_heads", (long) attention.numHeads());
        m.put("attention.num_kv_heads", (long) attention.numKvHeads());
        m.put("attention.head_dim", (long) attention.headDim());
        m.put("attention.partial_rotary_factor", attention.partialRotaryFactor());
        m.put("attention.rope_theta", attention.ropeTheta());
        m.put("attention.mrope_section", widen(attention.mropeSection()));
        m.put("attention.mrope_interleaved", attention.mropeInterleaved());
        m.put("attention.output_gate", attention.outputGate());
        m.put("attention.full_attention_interval", (long) attention.fullAttentionInterval());
        m.put("gdn.num_key_heads", (long) gdn.numKeyHeads());
        m.put("gdn.num_value_heads", (long) gdn.numValueHeads());
        m.put("gdn.key_head_dim", (long) gdn.keyHeadDim());
        m.put("gdn.value_head_dim", (long) gdn.valueHeadDim());
        m.put("gdn.conv_kernel_dim", (long) gdn.convKernelDim());
        m.put("gdn.ssm_dtype", gdn.ssmDtype());
        m.put("qsa.indexer_budget", (long) qsa.indexerBudget());
        m.put("qsa.compress_ratio", (long) qsa.compressRatio());
        m.put("qsa.head_dim", (long) qsa.headDim());
        m.put("qsa.kv_heads", (long) qsa.kvHeads());
        m.put("qsa.num_heads", (long) qsa.numHeads());
        m.put("moe.num_experts", (long) moe.numExperts());
        m.put("moe.experts_per_token", (long) moe.expertsPerToken());
        m.put("moe.intermediate_size", (long) moe.intermediateSize());
        m.put("moe.shared_expert_intermediate_size", (long) moe.sharedExpertIntermediateSize());
        m.put("hc.count", (long) hyperConnection.count());
        m.put("hc.lowrank", (long) hyperConnection.lowrank());
        m.put("ngram.size", (long) ngram.size());
        m.put("ngram.vocab_size_base", ngram.vocabSizeBase());
        m.put("ngram.heads_per_ngram", (long) ngram.headsPerNgram());
        m.put("ngram.vocab_divisible_by", (long) ngram.vocabDivisibleBy());
        m.put("ngram.split_parts", (long) ngram.splitParts());
        m.put("ngram.shard_rows", ngram.shardRows());
        m.put("ngram.layer_multipliers", ngram.layerMultipliers().clone());
        m.put("ngram.heads_offsets", ngram.headsOffsets().clone());
        m.put("ngram.heads_vocab_sizes", ngram.headsVocabSizes().clone());
        m.put("ple.layers", widen(ple.layers()));
        m.put("ple.embed_dim", (long) ple.embedDim());
        m.put("ple.conv_kernel_size", (long) ple.convKernelSize());
        m.put("mtp.num_layers", (long) mtp.numLayers());
        m.put("mtp.hybrid", mtp.hybrid());
        m.put("mtp.layer_types", sourceNames(mtp.layerTypes()));
        m.put("mtp.rope_theta", mtp.ropeTheta());
        m.put("mtp.use_dedicated_embeddings", mtp.useDedicatedEmbeddings());
        m.put("mtp.use_hidden_state_from_layer", (long) mtp.useHiddenStateFromLayer());
        m.put("vision.depth", (long) vision.depth());
        m.put("vision.hidden_size", (long) vision.hiddenSize());
        m.put("vision.intermediate_size", (long) vision.intermediateSize());
        m.put("vision.num_heads", (long) vision.numHeads());
        m.put("vision.in_channels", (long) vision.inChannels());
        m.put("vision.patch_size", (long) vision.patchSize());
        m.put("vision.spatial_merge_size", (long) vision.spatialMergeSize());
        m.put("vision.temporal_patch_size", (long) vision.temporalPatchSize());
        m.put("vision.out_hidden_size", (long) vision.outHiddenSize());
        m.put("vision.num_position_embeddings", (long) vision.numPositionEmbeddings());
        m.put("vision.hidden_act", vision.hiddenActivation());
        m.put("vision.deepstack_visual_indexes", widen(vision.deepstackVisualIndexes()));
        m.put("vision.image_token_id", (long) vision.imageTokenId());
        m.put("vision.video_token_id", (long) vision.videoTokenId());
        m.put("vision.start_token_id", (long) vision.startTokenId());
        m.put("vision.end_token_id", (long) vision.endTokenId());
        m.put("quant.format", "nvfp4");
        m.put("quant.block", 16L);
        m.put("quant.scale_dtype", "e4m3");
        return m;
    }

    /// Checks the relations between the fields that every later size and shape computation relies on.
    void validate() throws ArtifactFormatException {
        require(
                text.numLayers() > 0 && text.layerTypes().length == text.numLayers(),
                "layer types do not match the layer count");
        require(text.hiddenSize() % 128 == 0, "hidden size must be a multiple of 128");
        require(
                attention.numHeads() % attention.numKvHeads() == 0,
                "attention heads are not a multiple of the KV heads");
        require(
                attention.outputGate().equals("sigmoid"),
                "unsupported attention output gate " + attention.outputGate());
        require(gdn.numValueHeads() % gdn.numKeyHeads() == 0, "GDN value heads are not a multiple of the key heads");
        require(gdn.ssmDtype().equals("float32"), "unsupported GDN state type " + gdn.ssmDtype());
        require(
                moe.expertsPerToken() > 0 && moe.expertsPerToken() <= moe.numExperts(),
                "experts per token is out of range");
        require(moe.numExperts() > 0 && moe.numExperts() <= 65536, "expert count is out of range");
        require(
                hyperConnection.count() > 0 && hyperConnection.lowrank() > 0,
                "hyper-connection sizes must be positive");
        require(ngram.size() >= 2 && ngram.splitParts() > 0, "n-gram configuration is invalid");
        int heads = ngram.heads();
        require(
                ngram.headsOffsets().length == heads && ngram.headsVocabSizes().length == heads,
                "n-gram head tables do not have " + heads + " entries");
        require(
                ngram.layerMultipliers().length == ngram.size(),
                "n-gram multipliers do not have one entry per n-gram size");
        require(text.hiddenSize() % heads == 0, "hidden size is not a multiple of the n-gram heads");
        require(ple.layers().length > 0, "no layer holds the n-gram tables");
        for (int layer : ple.layers()) require(layer >= 0 && layer < text.numLayers(), "n-gram layer out of range");
        long end = 0;
        for (int head = 0; head < heads; head++) {
            require(ngram.headsOffsets()[head] == end, "n-gram head offsets are not cumulative at head " + head);
            require(ngram.headsVocabSizes()[head] > 0, "n-gram head vocabulary is empty");
            end += ngram.headsVocabSizes()[head];
        }
        require(
                end <= ngram.totalRows(),
                "n-gram heads address " + end + " rows but the shards hold " + ngram.totalRows());
        require(mtp.layerTypes().length == mtp.numLayers(), "MTP layer types do not match the MTP layer count");
        require(text.maxPositionEmbeddings() > 0, "maximum position embeddings must be positive");
    }

    private static void require(boolean condition, String message) throws ArtifactFormatException {
        if (!condition) throw new ArtifactFormatException("invalid configuration: " + message);
    }

    private static LayerType[] layerTypes(String[] names) throws ArtifactFormatException {
        LayerType[] types = new LayerType[names.length];
        for (int i = 0; i < names.length; i++) {
            try {
                types[i] = LayerType.parse(names[i]);
            } catch (IllegalArgumentException exception) {
                throw new ArtifactFormatException(exception.getMessage());
            }
        }
        return types;
    }

    private static String[] sourceNames(LayerType[] types) {
        String[] names = new String[types.length];
        for (int i = 0; i < types.length; i++) names[i] = types[i].sourceName();
        return names;
    }

    private static long[] widen(int[] values) {
        long[] widened = new long[values.length];
        for (int i = 0; i < values.length; i++) widened[i] = values[i];
        return widened;
    }
}
