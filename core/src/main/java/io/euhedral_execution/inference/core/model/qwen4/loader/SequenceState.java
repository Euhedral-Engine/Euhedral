package io.euhedral_execution.inference.core.model.qwen4.loader;

import io.euhedral_execution.inference.core.model.qwen4.Qwen4Config;

/// The device bytes a running sequence of a given length needs, from the configuration alone: the accounting the
/// residency planner reserves. The runtime state that holds them is
/// `io.euhedral_execution.inference.core.model.qwen4.Sequence`.
///
/// - Attention keys and values live in NVFP4 pages of 256 tokens: a row of 256 values takes 144 bytes, and a token
///   stores one K row and one V row per 256 values of each KV head, in every sparse-attention layer.
/// - The indexer keeps one BF16 key of `qsa.headDim` per `qsa.kvHeads` for every `qsa.compressRatio` tokens of each
///   sparse-attention layer.
/// - Gated DeltaNet keeps a FP32 `keyHeadDim x valueHeadDim` state per value head and the last `convKernelDim - 1`
///   inputs of its convolution in BF16, for each GDN layer, whatever the length.
public final class SequenceState {

    /// Tokens per KV page.
    public static final int PAGE_TOKENS = 256;

    private static final long KV_ROW_BYTES = 144;

    private SequenceState() {}

    /// Key/value bytes of one sequence of `tokens` positions, in whole pages.
    public static long kvBytes(Qwen4Config config, int tokens) {
        long pages = (tokens + PAGE_TOKENS - 1L) / PAGE_TOKENS;
        long rowsPerToken =
                (long) config.attention().numKvHeads() * config.attention().headDim() / 256L;
        return config.sparseAttentionLayers() * pages * PAGE_TOKENS * 2L * KV_ROW_BYTES * rowsPerToken;
    }

    /// Indexer key bytes of one sequence.
    public static long indexerBytes(Qwen4Config config, int tokens) {
        Qwen4Config.Qsa qsa = config.qsa();
        long keys = (tokens + qsa.compressRatio() - 1L) / qsa.compressRatio();
        return config.sparseAttentionLayers() * keys * qsa.kvHeads() * qsa.headDim() * 2L;
    }

    /// Recurrent state and convolution history of one sequence.
    public static long gdnStateBytes(Qwen4Config config) {
        Qwen4Config.Gdn gdn = config.gdn();
        long state = (long) gdn.numValueHeads() * gdn.keyHeadDim() * gdn.valueHeadDim() * Float.BYTES;
        long convolution = (long) gdn.convChannels() * (gdn.convKernelDim() - 1) * 2L;
        return config.gdnLayers() * (state + convolution);
    }

    /// Everything that grows with, or is held for, one sequence of `tokens` positions.
    public static long contextBytes(Qwen4Config config, int tokens) {
        return kvBytes(config, tokens) + indexerBytes(config, tokens) + gdnStateBytes(config);
    }

    /// Prefill tokens an execution step holds at once unless the residency plan finds room for more.
    public static final int PREFILL_CHUNK_TOKENS = 512;
    /// Rows of the workspaces that serve a decode token and a short chunk, beside the full chunk's.
    private static final int SMALL_WORKSPACE_ROWS = 1 + 16;
    /// Rows of one decode or verification step's logits.
    private static final int LOGIT_ROWS = 8;

    /// Activation storage of one execution step, from the configuration: for each token of a prefill chunk the
    /// hyper-connection streams (the layer's input, its mixed input and its output), the widest token mixer's
    /// intermediates (double buffered, BF16), the router logits and the selected experts' gate/up and down
    /// activations; plus the logits of a decode step. A bound sized from the topology, not a measurement.
    public static long workspaceBytes(Qwen4Config config) {
        return workspaceBytes(config, PREFILL_CHUNK_TOKENS);
    }

    /// As [#workspaceBytes(Qwen4Config)] for chunks of `chunkTokens` tokens, with the smaller workspaces that
    /// serve a decode token and a short chunk: every workspace the execution plan allocates.
    public static long workspaceBytes(Qwen4Config config, int chunkTokens) {
        long hidden = config.text().hiddenSize();
        long stream = (long) config.hyperConnection().count() * hidden * 2L;
        Qwen4Config.Gdn gdn = config.gdn();
        Qwen4Config.Attention attention = config.attention();
        long gdnWidth = gdn.convChannels() + 2L * gdn.numValueHeads() * gdn.valueHeadDim();
        long attentionWidth =
                2L * attention.numHeads() * attention.headDim() + 2L * attention.numKvHeads() * attention.headDim();
        long mixer = 2L * Math.max(gdnWidth, attentionWidth) * 2L;
        Qwen4Config.Moe moe = config.moe();
        long experts = (long) moe.expertsPerToken() * (2L * moe.intermediateSize() + hidden) * 2L
                + (long) moe.numExperts() * Float.BYTES
                + (long) moe.expertsPerToken() * 8L;
        long perToken = 3 * stream + mixer + experts + 4 * hidden * 2L;
        long logits = (long) LOGIT_ROWS * config.text().vocabSize() * Float.BYTES;
        return ((long) chunkTokens + SMALL_WORKSPACE_ROWS) * perToken + logits;
    }
}
