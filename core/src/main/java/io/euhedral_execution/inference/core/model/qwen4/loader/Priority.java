package io.euhedral_execution.inference.core.model.qwen4.loader;

import io.euhedral_execution.inference.core.artifact.WeightFormat;

/// Which fixed objects leave the device first when the context needs the room, and how.
///
/// Device-residency priority, highest first: routers, norms and small control weights, hyper-connections, GDN
/// weights, QSA and indexer weights, shared experts, the output head, then the remaining always-used text weights. A
/// byte of any fixed weight is read on every token, so each byte moved to the host costs the same bus time; the order
/// therefore follows the priority above, and within a rank the planner moves the fewest and smallest tensors.
///
/// Never leave the device: routers, norms and every object under [#SMALL_BYTES] (the control vectors, the GDN
/// convolutions and gates, the hyper-connection norms and inject weights): they are tiny and read first in each
/// layer. The output head is the one big matrix that is read in place from mapped host memory instead of staged,
/// since a staging slot would have to hold all of it.
public final class Priority {

    /// Objects at or below this size are always device resident.
    public static final long SMALL_BYTES = 4L << 20;

    private Priority() {}

    /// How a fixed object that leaves the device is placed on the host.
    public enum Move {
        /// Pinned host memory, staged into a device ring on use.
        STAGED,
        /// Pinned host memory that kernels read in place.
        MAPPED
    }

    /// The order in which ranks leave the device: index 0 first.
    private static final ComponentGroup[] OFFLOAD_ORDER = {
        ComponentGroup.FIXED_TEXT,
        ComponentGroup.OUTPUT_HEAD,
        ComponentGroup.SHARED_EXPERT,
        ComponentGroup.QSA,
        ComponentGroup.GDN,
        ComponentGroup.HYPER_CONNECTION
    };

    /// The rank at which the tensor may leave the device (0 is first), or -1 when it never does.
    public static int offloadRank(Tensor tensor) {
        if (tensor.byteSize() <= SMALL_BYTES) return -1;
        for (int rank = 0; rank < OFFLOAD_ORDER.length; rank++) if (OFFLOAD_ORDER[rank] == tensor.group()) return rank;
        return -1;
    }

    public static int offloadRanks() {
        return OFFLOAD_ORDER.length;
    }

    public static Move move(Tensor tensor) {
        return tensor.group() == ComponentGroup.OUTPUT_HEAD ? Move.MAPPED : Move.STAGED;
    }

    /// Whether the token embedding is a gather over a vocabulary-sized table: its rows are read in place.
    public static boolean isTokenEmbedding(Tensor tensor) {
        return tensor.group() == ComponentGroup.TOKEN_EMBEDDING;
    }

    static boolean isProjection(Tensor tensor) {
        return tensor.format() == WeightFormat.NVFP4;
    }
}
