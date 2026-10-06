package io.euhedral_execution.inference.core.model_loader.qwen4;

import io.euhedral_execution.inference.core.model_loader.ResidencyPlanner;
import io.euhedral_execution.inference.core.model_loader.WeightStaging;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Pattern;

/// The fixed-memory half of Flash-Next residency: from the device's free memory, the host's
/// pinnable memory, the requested maximum context and the artifact's actual object sizes it decides
/// what lives on the device for good, what is host-backed, what is mapped, what stays unloaded, and
/// how many bytes are left for the routed-expert cache. It keeps no replacement state; the cache
/// ([io.euhedral_execution.inference.core.model_loader.qwen4.expert]) manages the budget this
/// planner hands it.
///
/// ```
/// free device memory
///   - runtime reserve (kernels, CUDA context, graph pools)
///   - workspace and the sequence state of `maxContextTokens`  (KV, indexer keys, GDN state)
///   - fixed objects placed on the device in priority order    (Qwen4Priority)
///   - the staging ring of any fixed object moved to the host
///   = the expert cache, in whole slots
/// ```
///
/// Context has priority: a longer context takes memory from the expert cache first, and only when
/// the cache would fall below its minimum does a fixed object move to the host, lowest priority
/// first and the fewest bytes that suffice. A fixed byte is read on every token while a cached
/// expert is hit only on some, so the planner never moves fixed objects to enlarge the cache. A
/// plan that cannot meet the minimum even with every movable object on the host is returned with
/// `fits` false and the reason.
public final class Qwen4ResidencyPlanner {

    /// The CUDA context, kernel modules and graph pools that exist beside the model (about 0.9 GiB
    /// measured on the dense model's engine, docs/NVFP4_RESIDENCY.md).
    public static final long KERNEL_RESERVE_BYTES = 1024L << 20;

    /// Staging slots for host-backed fixed objects, shared with the dense engine's ring.
    public static final int STAGING_SLOTS = ResidencyPlanner.STAGING_SLOTS;

    private static final long SELECTION_STEP = 64L << 20;
    private static final Pattern LAYER_INDEX = Pattern.compile("/layers/\\d+/");

    private Qwen4ResidencyPlanner() {}

    /// The smallest expert cache that can serve one token: its selected experts plus a second set
    /// in flight.
    public static int minimumSlots(Qwen4Config config) {
        return 2 * config.moe().expertsPerToken();
    }

    /// Pinned staging slots when expert records pass through the host from the artifact file: a
    /// wave of experts (at most 32, and no more than a layer has) loads at once, and each load
    /// holds a slot until its copy retires.
    public static int fileStagingSlots(Qwen4Config config) {
        return Math.max(
                Math.min(32, config.moe().numExperts()), 2 * config.moe().expertsPerToken());
    }

    /// Experts of one wave at most: a wave's experts are pinned until its kernels are submitted and
    /// the next wave's claim slots at the same time, so a shard must hold two waves.
    public static int expertWave(int slotsOfAShard) {
        return Math.max(1, Math.min(32, slotsOfAShard / 2));
    }

    /// Shards (independent partitions of the expert cache, each its own lattice source) for a cache
    /// of `slots`: as many as can each hold two full waves, at most 8.
    public static int expertShards(int slots) {
        int wave = expertWave(slots);
        return Math.max(1, Math.min(8, slots / (2 * wave)));
    }

    public static Qwen4ResidencyPlan plan(
            Qwen4Artifact artifact, Qwen4Mode mode, long freeBytes, HostBudget host, int maxContextTokens) {
        Qwen4Config config = artifact.config();
        if (maxContextTokens <= 0) throw new IllegalArgumentException("maxContextTokens must be positive");

        // What each object is for this mode.
        Map<String, Qwen4ResidencyPlan.Placement> placements = new LinkedHashMap<>();
        List<Qwen4Tensor> fixed = new ArrayList<>();
        Qwen4Tensor embedding = null;
        long ngramBytes = 0;
        long deferredMtp = 0;
        long deferredVision = 0;
        for (Qwen4Tensor tensor : artifact.tensors()) {
            switch (tensor.group()) {
                case MTP -> {
                    if (mode.mtp()) fixed.add(tensor);
                    else {
                        deferredMtp += tensor.byteSize();
                        placements.put(tensor.name(), deferred());
                    }
                }
                case VISION -> {
                    if (mode.vision()) fixed.add(tensor);
                    else {
                        deferredVision += tensor.byteSize();
                        placements.put(tensor.name(), deferred());
                    }
                }
                case NGRAM -> {
                    ngramBytes += tensor.byteSize();
                    placements.put(tensor.name(), new Qwen4ResidencyPlan.Placement(StorageClass.HOST_STAGED, null));
                }
                case TOKEN_EMBEDDING -> embedding = tensor;
                default -> fixed.add(tensor);
            }
        }
        Map<String, StorageClass> banks = new LinkedHashMap<>();
        long expertBytes = 0;
        long totalExperts = 0;
        long slotBytes = 0;
        for (ExpertBank bank : artifact.banks()) {
            if (bank.group() == ComponentGroup.MTP && !mode.mtp()) {
                deferredMtp += bank.totalBytes();
                banks.put(bank.name(), StorageClass.DEFERRED);
                continue;
            }
            banks.put(bank.name(), StorageClass.DEVICE_CACHED);
            expertBytes += bank.totalBytes();
            totalExperts += bank.expertCount();
            slotBytes = Math.max(slotBytes, bank.maxRecordBytes());
        }
        if (totalExperts == 0) throw new IllegalArgumentException("the artifact has no expert banks to cache");
        int minimumSlots = minimumSlots(config);

        // Device state that does not depend on placement.
        long kv = Qwen4SequenceState.kvBytes(config, maxContextTokens);
        long indexer = Qwen4SequenceState.indexerBytes(config, maxContextTokens);
        long gdn = Qwen4SequenceState.gdnStateBytes(config);
        long workspace = Qwen4SequenceState.workspaceBytes(config);
        long reserved = KERNEL_RESERVE_BYTES + workspace + kv + indexer + gdn;

        // The token embedding is a gather: its rows are read in place from pinned host memory.
        long pinnedLeft = host.pinnableBytes();
        boolean embeddingMapped = embedding != null && embedding.byteSize() <= pinnedLeft;
        if (embeddingMapped) {
            pinnedLeft -= embedding.byteSize();
            placements.put(
                    embedding.name(),
                    new Qwen4ResidencyPlan.Placement(StorageClass.HOST_MAPPED, Qwen4Priority.Move.MAPPED));
        } else if (embedding != null) {
            fixed.add(embedding);
        }

        long fixedBytes = 0;
        List<Qwen4Tensor> movable = new ArrayList<>();
        for (Qwen4Tensor tensor : fixed) {
            fixedBytes += tensor.byteSize();
            if (Qwen4Priority.offloadRank(tensor) >= 0 && !(mode.mtp() && tensor.group() == ComponentGroup.MTP))
                movable.add(tensor);
        }
        long minimumCache = minimumSlots * slotBytes;

        // The fewest bytes moved to the host that leave the minimum cache.
        long movableBytes = 0;
        for (Qwen4Tensor tensor : movable) movableBytes += tensor.byteSize();
        Selection selection = Selection.NONE;
        boolean placed = false;
        for (long target = 0; ; target += SELECTION_STEP) {
            selection = select(movable, target);
            long need = deviceBytes(fixedBytes, selection);
            if (need + reserved + minimumCache <= freeBytes) {
                placed = true;
                break;
            }
            if (target >= movableBytes) break;
        }
        long need = deviceBytes(fixedBytes, selection);
        long ring = selection.stagedMax == 0 ? 0 : STAGING_SLOTS * WeightStaging.slotBytesFor(selection.stagedMax);
        long resident = need - ring;

        String problem = null;
        if (maxContextTokens > config.text().maxPositionEmbeddings())
            problem = "a context of " + maxContextTokens + " tokens is longer than the "
                    + config.text().maxPositionEmbeddings() + " positions the model supports";
        else if (!placed)
            problem = String.format(
                    Locale.ROOT,
                    "a context of %d tokens needs %d MiB of sequence state and workspace and %d MiB of runtime reserve, "
                            + "the objects that must stay on the device take %d MiB (%d MiB after moving every movable "
                            + "object to the host, with a %d MiB staging ring) and the smallest expert cache of %d slots "
                            + "takes %d MiB: %d MiB in all against %d MiB free",
                    maxContextTokens,
                    (kv + indexer + gdn + workspace) >> 20,
                    KERNEL_RESERVE_BYTES >> 20,
                    (fixedBytes - movableBytes) >> 20,
                    need >> 20,
                    ring >> 20,
                    minimumSlots,
                    minimumCache >> 20,
                    (need + reserved + minimumCache) >> 20,
                    freeBytes >> 20);

        // Placements of fixed objects.
        for (Qwen4Tensor tensor : fixed) {
            if (tensor == embedding) {
                placements.put(tensor.name(), new Qwen4ResidencyPlan.Placement(StorageClass.DEVICE_RESIDENT, null));
                continue;
            }
            if (selection.names.contains(tensor.name())) {
                Qwen4Priority.Move move = Qwen4Priority.move(tensor);
                placements.put(
                        tensor.name(),
                        new Qwen4ResidencyPlan.Placement(
                                move == Qwen4Priority.Move.MAPPED ? StorageClass.HOST_MAPPED : StorageClass.HOST_STAGED,
                                move));
            } else placements.put(tensor.name(), new Qwen4ResidencyPlan.Placement(StorageClass.DEVICE_RESIDENT, null));
        }

        long cacheBudget = freeBytes - need - reserved;
        ExpertCacheGeometry geometry = ExpertCacheGeometry.derive(cacheBudget, slotBytes, totalExperts, minimumSlots);
        long cacheBytes = geometry.bytes();
        long slack = Math.max(0, cacheBudget - cacheBytes);

        // The host: pinned memory goes to the gathered embedding, staged fixed objects, then the experts.
        long mapped = (embeddingMapped ? embedding.byteSize() : 0) + selection.mappedBytes;
        long staged = selection.stagedBytes;
        pinnedLeft -= selection.stagedBytes + selection.mappedBytes;
        Qwen4ResidencyPlan.ExpertStoreMode expertStore;
        long expertPinned = 0;
        long expertFile = 0;
        long stagingPinned = fileStagingSlots(config) * alignUp(slotBytes, 4096);
        if (expertBytes <= pinnedLeft) {
            expertStore = Qwen4ResidencyPlan.ExpertStoreMode.PINNED_ARENA;
            expertPinned = expertBytes;
        } else {
            expertStore = Qwen4ResidencyPlan.ExpertStoreMode.FILE_BACKED;
            expertPinned = stagingPinned;
            expertFile = expertBytes;
        }
        pinnedLeft -= expertPinned;
        if (pinnedLeft < 0 && problem == null)
            problem = "the host cannot pin " + ((host.pinnableBytes() - pinnedLeft) >> 20)
                    + " MiB for host-backed weights and expert staging; " + (host.pinnableBytes() >> 20)
                    + " MiB is pinnable";
        // The tables are sparse gathers: they go to pinned memory only when the experts already do and room is left;
        // otherwise they are read from the mapped file and the unpinned memory serves the experts' page cache.
        Qwen4ResidencyPlan.NgramMode ngram = Qwen4ResidencyPlan.NgramMode.MAPPED_FILE;
        long ngramPinned = 0;
        long ngramFile = ngramBytes;
        if (expertStore == Qwen4ResidencyPlan.ExpertStoreMode.PINNED_ARENA && ngramBytes <= pinnedLeft) {
            ngram = Qwen4ResidencyPlan.NgramMode.PINNED_ARENA;
            ngramPinned = ngramBytes;
            ngramFile = 0;
        }

        var device = new Qwen4ResidencyPlan.Device(
                freeBytes, kv, indexer, gdn, workspace, KERNEL_RESERVE_BYTES, resident, ring, cacheBytes, slack);
        var hostPlan = new Qwen4ResidencyPlan.Host(
                mapped, staged, expertPinned, expertFile, ngramPinned, ngramFile, deferredMtp, deferredVision);
        boolean fits = problem == null && geometry.viable();
        return new Qwen4ResidencyPlan(
                maxContextTokens,
                mode,
                fits,
                problem,
                placements,
                banks,
                device,
                hostPlan,
                geometry,
                expertStore,
                ngram);
    }

    private static Qwen4ResidencyPlan.Placement deferred() {
        return new Qwen4ResidencyPlan.Placement(StorageClass.DEFERRED, null);
    }

    private static long alignUp(long value, long alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    /// Device bytes of the fixed objects that stay, plus the staging ring of those that are staged.
    private static long deviceBytes(long fixedBytes, Selection selection) {
        long resident = fixedBytes - selection.stagedBytes - selection.mappedBytes;
        long ring = selection.stagedMax == 0 ? 0 : STAGING_SLOTS * WeightStaging.slotBytesFor(selection.stagedMax);
        return resident + ring;
    }

    /// The objects chosen to leave the device, with the sums the plan needs.
    private record Selection(Set<String> names, long stagedBytes, long stagedMax, long mappedBytes) {
        static final Selection NONE = new Selection(Set.of(), 0, 0, 0);
    }

    /// At least `bytes` of movable objects (or all of them), in offload-rank order. Within a rank,
    /// whole families (the same object in every layer) go smallest tensor first, and the last
    /// family takes layers spread evenly, so that transfers interleave with resident work and the
    /// staging ring holds the smallest possible tensor.
    private static Selection select(List<Qwen4Tensor> movable, long bytes) {
        if (bytes <= 0) return Selection.NONE;
        Map<Integer, Map<String, List<Qwen4Tensor>>> ranks = new TreeMap<>();
        for (Qwen4Tensor tensor : movable)
            ranks.computeIfAbsent(Qwen4Priority.offloadRank(tensor), rank -> new LinkedHashMap<>())
                    .computeIfAbsent(
                            LAYER_INDEX.matcher(tensor.name()).replaceFirst("/layers/*/"), family -> new ArrayList<>())
                    .add(tensor);
        Set<String> names = new LinkedHashSet<>();
        long staged = 0;
        long stagedMax = 0;
        long mapped = 0;
        long remaining = bytes;
        for (Map<String, List<Qwen4Tensor>> families : ranks.values()) {
            List<Map.Entry<String, List<Qwen4Tensor>>> ordered = new ArrayList<>(families.entrySet());
            ordered.sort(Comparator.<Map.Entry<String, List<Qwen4Tensor>>>comparingLong(
                            entry -> entry.getValue().getFirst().byteSize())
                    .thenComparing(Map.Entry::getKey));
            for (Map.Entry<String, List<Qwen4Tensor>> family : ordered) {
                List<Qwen4Tensor> tensors = family.getValue();
                long familyBytes =
                        tensors.stream().mapToLong(Qwen4Tensor::byteSize).sum();
                List<Qwen4Tensor> taken = new ArrayList<>();
                if (familyBytes <= remaining) taken.addAll(tensors);
                else {
                    long size = tensors.getFirst().byteSize();
                    int count = (int) Math.min(tensors.size(), (remaining + size - 1) / size);
                    for (int index = 0; index < count; index++)
                        taken.add(tensors.get((int) ((long) index * tensors.size() / count)));
                }
                for (Qwen4Tensor tensor : taken) {
                    names.add(tensor.name());
                    remaining -= tensor.byteSize();
                    if (Qwen4Priority.move(tensor) == Qwen4Priority.Move.MAPPED) mapped += tensor.byteSize();
                    else {
                        staged += tensor.byteSize();
                        stagedMax = Math.max(stagedMax, tensor.byteSize());
                    }
                }
                if (remaining <= 0) return new Selection(names, staged, stagedMax, mapped);
            }
        }
        return new Selection(names, staged, stagedMax, mapped);
    }
}
