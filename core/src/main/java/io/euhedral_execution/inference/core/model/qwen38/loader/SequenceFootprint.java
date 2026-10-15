package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import io.euhedral_execution.inference.core.state.KvFormat;

/// The device memory a sequence needs beyond its attention KV, derived from the model configuration and the sizes
/// the sequence's states allocate: the GDN state and its verification buffers, the draft seed rows, the MTP layer's
/// cache and the device words of a host-logits selection.
public final class SequenceFootprint {

    /// Rows `GdnState#speculative` allocates at least, however few are verified.
    private static final int MINIMUM_ROWS = 8;
    /// A scored selection on the device (`HostLogits`): the 8-byte key then the log-probability, padded.
    private static final long SELECTION_BYTES = 2L * Long.BYTES;

    private SequenceFootprint() {}

    /// GDN state of every GDN layer (recurrent and convolution), its verification checkpoint and its row buffers
    /// at `verifiedRows` rows (at least 8, as `GdnState#speculative` allocates).
    public static long gdnBytes(Qwen38Config config, int verifiedRows) {
        long layers = 0;
        for (LayerType type : config.layerTypes()) if (type == LayerType.GATED_DELTA_NET) layers++;
        long queryKeyWidth = 2L * config.linearNumKeyHeads() * config.linearKeyHeadDim();
        long valueWidth = (long) config.linearNumValueHeads() * config.linearValueHeadDim();
        long convolvedWidth = queryKeyWidth + valueWidth;
        long convolution = convolvedWidth * (config.linearConvKernelDim() - 1L) * Short.BYTES;
        long recurrent = (long) config.linearNumValueHeads()
                * config.linearKeyHeadDim()
                * config.linearValueHeadDim()
                * Float.BYTES;
        long rows = Math.max(verifiedRows, MINIMUM_ROWS);
        long verification = recurrent
                + convolution
                + rows * queryKeyWidth * Short.BYTES
                + rows * 2 * valueWidth * Short.BYTES
                + 2 * rows * config.linearNumValueHeads() * Float.BYTES
                + rows * convolvedWidth * Short.BYTES
                + rows * valueWidth * Short.BYTES;
        return Math.multiplyExact(layers, recurrent + convolution + verification);
    }

    /// Device bytes one more sequence needs beyond its attention KV: gdnBytes, the draft seed rows of a prefill
    /// chunk, the MTP layer's cache for `contextTokens` (when it drafts), and its host-logits device words.
    public static long extraSequenceBytes(
            Qwen38Config config, KvFormat format, int contextTokens, int verifiedRows, boolean drafts) {
        long total = gdnBytes(config, verifiedRows) + SELECTION_BYTES + (long) verifiedRows * Long.BYTES;
        if (!drafts) return total;
        long seedRows = Math.max(InferenceConfig.PREFILL_CHUNK_TOKENS, MINIMUM_ROWS);
        total += seedRows * config.hiddenSize() * Short.BYTES;
        // The MTP layer is one full-attention layer's cache.
        long pages = (contextTokens + AttentionKvState.PAGE_TOKENS - 1L) / AttentionKvState.PAGE_TOKENS;
        int width = config.numKeyValueHeads() * config.attentionHeadDim();
        total += pages * 2 * AttentionKvState.planePageBytes(width, format);
        return total;
    }
}
