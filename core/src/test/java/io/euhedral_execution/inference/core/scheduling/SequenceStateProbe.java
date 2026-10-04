package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;

/// Test support: labelled digests of the device bytes of a sequence's state that a restore must reproduce,
/// in layer order. For GDN layers every buffer; for attention layers the pages that are full below the
/// committed length (a partly written page holds unwritten rows). Digests keep the comparison small enough
/// for the test JVM's heap and say which buffer differs.
public final class SequenceStateProbe {
    private SequenceStateProbe() {}

    public static List<String> committedDigests(
            ExecutionGpu gpu, QwenSequenceState sequence, QwenLayerType[] layerTypes) {
        var gdn = (GdnSequenceStates) sequence.recurrentState();
        var attention = (AttentionSequenceStates) sequence.kvCacheState();
        List<String> digests = new ArrayList<>();
        for (int layer = 0; layer < layerTypes.length; layer++) {
            if (layerTypes[layer] == QwenLayerType.GATED_DELTA_NET) {
                QwenGdnSequenceState state = gdn.forLayer(layer);
                digests.add("layer " + layer + " conv "
                        + digest(gpu, state.convolutionStateAddress(), state.convolutionBytes()));
                digests.add("layer " + layer + " recurrent "
                        + digest(gpu, state.recurrentStateAddress(), state.recurrentBytes()));
            } else {
                AttentionKvState state = attention.forLayer(layer);
                List<Long> pages = state.pageAddresses();
                int full = state.length() / AttentionKvState.PAGE_TOKENS;
                for (int page = 0; page < full; page++)
                    digests.add("layer " + layer + " page " + page + " "
                            + digest(gpu, pages.get(page), 2 * state.planePageBytes()));
            }
        }
        return digests;
    }

    /// Digests of the MTP cache's pages that are full below `rows` rows, or an empty list when the sequence has no
    /// MTP cache.
    public static List<String> mtpDigests(
            ExecutionGpu gpu, QwenSequenceState sequence, QwenLayerType[] layerTypes, int rows) {
        var attention = (AttentionSequenceStates) sequence.kvCacheState();
        AttentionKvState mtp;
        try {
            mtp = attention.forLayer(layerTypes.length);
        } catch (IllegalArgumentException none) {
            return List.of();
        }
        List<String> digests = new ArrayList<>();
        List<Long> pages = mtp.pageAddresses();
        for (int page = 0; page < rows / AttentionKvState.PAGE_TOKENS; page++)
            digests.add("mtp page " + page + " " + digest(gpu, pages.get(page), 2 * mtp.planePageBytes()));
        return digests;
    }

    private static String digest(ExecutionGpu gpu, long address, long bytes) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(bytes);
            gpu.copyDeviceToHost(host, address, bytes);
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            sha.update(host.asByteBuffer());
            return HexFormat.of().formatHex(sha.digest(), 0, 8);
        } catch (NoSuchAlgorithmException unavailable) {
            throw new IllegalStateException(unavailable);
        }
    }
}
