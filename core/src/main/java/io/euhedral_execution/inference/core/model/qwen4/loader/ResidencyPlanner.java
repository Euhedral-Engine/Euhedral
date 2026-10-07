package io.euhedral_execution.inference.core.model.qwen4.loader;

import io.euhedral_execution.inference.core.artifact.WeightStaging;
import io.euhedral_execution.inference.core.model.qwen4.Qwen4Config;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertBank;
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
///   - fixed objects placed on the device in priority order    (Priority)
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
public final class ResidencyPlanner {

    /// The CUDA context, kernel modules and graph pools that exist beside the model (about 0.9 GiB
    /// measured on the dense model's engine, docs/NVFP4_RESIDENCY.md).
    public static final long KERNEL_RESERVE_BYTES = 1024L << 20;

    /// Device memory left free for the rest of the system (the desktop and other programs on the GPU), which the
    /// plan never fills: 700 MiB, or `EUHEDRAL_GPU_SYSTEM_RESERVE_MIB`.
    public static final long SYSTEM_RESERVE_BYTES =
            Long.parseLong(System.getenv().getOrDefault("EUHEDRAL_GPU_SYSTEM_RESERVE_MIB", "700")) << 20;

    /// What the plan keeps beside the model: the runtime's own memory and the system's share.
    static final long RUNTIME_RESERVE_BYTES = KERNEL_RESERVE_BYTES + SYSTEM_RESERVE_BYTES;

    /// The shared expert's width as a prefill's down projection reads it: a multiple of 256, as the native NVFP4
    /// kernels need for more than 64 rows (they fetch the weight scales 256 values at a time).
    public static int paddedSharedWidth(int width) {
        return (width + 255) / 256 * 256;
    }

    /// Device bytes of the shared expert's down projections padded to [#paddedSharedWidth] (one per
    /// layer, plain NVFP4: codes, 256-aligned scales, the 256-aligned global) and of the padding in the
    /// activation of the longest prefill chunk; 0 when the width needs none.
    public static long sharedDownPaddingBytes(Qwen4Config config) {
        int width = config.moe().sharedExpertIntermediateSize();
        int padded = paddedSharedWidth(width);
        if (padded == width) return 0;
        long hidden = config.text().hiddenSize();
        long scales = (hidden * padded / 2 + 255) & ~255L;
        long tensor = ((scales + hidden * padded / 16 + 255) & ~255L) + 4;
        long activation = (long) LARGEST_PREFILL_CHUNK_TOKENS * (padded - width) * 2L;
        return config.text().numLayers() * ((tensor + 255) & ~255L) + activation;
    }

    /// The largest prefill chunk the planner will make room for.
    public static final int LARGEST_PREFILL_CHUNK_TOKENS = 4096;

    /// The share of the expert cache's memory that a larger prefill chunk may take: a chunk of many tokens
    /// reads most of the experts of every layer once however many tokens it holds, so its cost is
    /// amortized over its tokens, and the cache gives up a tenth of its slots at most for that.
    private static final int CHUNK_SHARE_DIVISOR = 10;

    /// Staging slots for host-backed fixed objects, shared with the dense engine's ring.
    public static final int STAGING_SLOTS = WeightStaging.SLOTS;

    private static final long SELECTION_STEP = 64L << 20;
    private static final Pattern LAYER_INDEX = Pattern.compile("/layers/\\d+/");

    private ResidencyPlanner() {}

    /// The smallest expert cache that can serve one token: its selected experts plus a second set
    /// in flight.
    public static int minimumSlots(Qwen4Config config) {
        return 2 * config.moe().expertsPerToken();
    }

    /// Records read from the artifact at once at most. A load holds its read only while it reads, and the disk
    /// reads fastest with a few records in flight: past about 30 MB it slows down (docs/FLASH_NEXT_DISK.md).
    /// `EUHEDRAL_QWEN4_READS` overrides it for benchmarks.
    public static final int DISK_READS = Integer.parseInt(System.getenv().getOrDefault("EUHEDRAL_QWEN4_READS", "8"));

    /// Pinned staging slots when expert records pass through the host from the artifact file: one per read the disk
    /// has in flight ([#DISK_READS]) but no more than a layer has experts, as each load holds its slot until its
    /// copy retires, and at least two tokens' experts.
    public static int fileStagingSlots(Qwen4Config config) {
        return Math.max(
                Math.min(DISK_READS, config.moe().numExperts()),
                2 * config.moe().expertsPerToken());
    }

    public static ResidencyPlan plan(
            Artifact artifact, Mode mode, long freeBytes, HostBudget host, int maxContextTokens) {
        Qwen4Config config = artifact.config();
        if (maxContextTokens <= 0) throw new IllegalArgumentException("maxContextTokens must be positive");

        // What each object is for this mode.
        Map<String, ResidencyPlan.Placement> placements = new LinkedHashMap<>();
        List<Tensor> fixed = new ArrayList<>();
        Tensor embedding = null;
        long ngramBytes = 0;
        long deferredMtp = 0;
        long deferredVision = 0;
        for (Tensor tensor : artifact.tensors()) {
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
                    placements.put(tensor.name(), new ResidencyPlan.Placement(StorageClass.HOST_STAGED, null));
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
        long kv = SequenceState.kvBytes(config, maxContextTokens);
        long indexer = SequenceState.indexerBytes(config, maxContextTokens);
        long gdn = SequenceState.gdnStateBytes(config);
        // The workspace counts the shared expert's padded down projections, which the plan keeps beside it.
        long padding = sharedDownPaddingBytes(config);
        long workspace = SequenceState.workspaceBytes(config) + padding;
        long reserved = RUNTIME_RESERVE_BYTES + workspace + kv + indexer + gdn;

        // The token embedding is a gather: its rows are read in place from pinned host memory.
        long pinnedLeft = host.pinnableBytes();
        boolean embeddingMapped = embedding != null && embedding.byteSize() <= pinnedLeft;
        if (embeddingMapped) {
            pinnedLeft -= embedding.byteSize();
            placements.put(
                    embedding.name(), new ResidencyPlan.Placement(StorageClass.HOST_MAPPED, Priority.Move.MAPPED));
        } else if (embedding != null) {
            fixed.add(embedding);
        }

        long fixedBytes = 0;
        List<Tensor> movable = new ArrayList<>();
        for (Tensor tensor : fixed) {
            fixedBytes += tensor.byteSize();
            if (Priority.offloadRank(tensor) >= 0 && !(mode.mtp() && tensor.group() == ComponentGroup.MTP))
                movable.add(tensor);
        }
        long minimumCache = minimumSlots * slotBytes;

        // The fewest bytes moved to the host that leave the minimum cache.
        long movableBytes = 0;
        for (Tensor tensor : movable) movableBytes += tensor.byteSize();
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
                    RUNTIME_RESERVE_BYTES >> 20,
                    (fixedBytes - movableBytes) >> 20,
                    need >> 20,
                    ring >> 20,
                    minimumSlots,
                    minimumCache >> 20,
                    (need + reserved + minimumCache) >> 20,
                    freeBytes >> 20);

        // Placements of fixed objects.
        for (Tensor tensor : fixed) {
            if (tensor == embedding) {
                placements.put(tensor.name(), new ResidencyPlan.Placement(StorageClass.DEVICE_RESIDENT, null));
                continue;
            }
            if (selection.names.contains(tensor.name())) {
                Priority.Move move = Priority.move(tensor);
                placements.put(
                        tensor.name(),
                        new ResidencyPlan.Placement(
                                move == Priority.Move.MAPPED ? StorageClass.HOST_MAPPED : StorageClass.HOST_STAGED,
                                move));
            } else placements.put(tensor.name(), new ResidencyPlan.Placement(StorageClass.DEVICE_RESIDENT, null));
        }

        // A longer chunk spends the device's memory on workspace instead of expert slots. A prefill chunk reads
        // most of the experts of every layer once, whatever its length, so the longest chunk the cache can
        // afford to give up memory for is the one that serves a prompt best. A decode token needs none of it.
        int chunk = SequenceState.PREFILL_CHUNK_TOKENS;
        long budget512 = freeBytes - need - reserved;
        long base = SequenceState.workspaceBytes(config, chunk);
        for (int candidate = chunk * 2;
                candidate <= LARGEST_PREFILL_CHUNK_TOKENS && candidate / 2 < maxContextTokens;
                candidate *= 2) {
            long extra = SequenceState.workspaceBytes(config, candidate) - base;
            if (extra > budget512 / CHUNK_SHARE_DIVISOR || budget512 - extra < minimumCache) break;
            chunk = candidate;
        }
        workspace = SequenceState.workspaceBytes(config, chunk) + padding;
        reserved = RUNTIME_RESERVE_BYTES + workspace + kv + indexer + gdn;
        long cacheBudget = freeBytes - need - reserved;
        ExpertCacheGeometry geometry = ExpertCacheGeometry.derive(cacheBudget, slotBytes, totalExperts, minimumSlots);
        long cacheBytes = geometry.bytes();
        long slack = Math.max(0, cacheBudget - cacheBytes);

        // The host. Pinned memory is a small tier: the gathered embedding, staged fixed objects, and the staging slots
        // every expert record passes through. The routed experts themselves live in ordinary memory, in what the
        // resident budget has left after the pinned part (pinned memory comes out of the same physical memory).
        long mapped = (embeddingMapped ? embedding.byteSize() : 0) + selection.mappedBytes;
        long staged = selection.stagedBytes;
        pinnedLeft -= selection.stagedBytes + selection.mappedBytes;
        long stagingPinned = fileStagingSlots(config) * alignUp(slotBytes, 4096);
        pinnedLeft -= stagingPinned;
        if (pinnedLeft < 0 && problem == null)
            problem = "the host cannot pin " + ((host.pinnableBytes() - pinnedLeft) >> 20)
                    + " MiB for host-backed weights and expert staging; " + (host.pinnableBytes() >> 20)
                    + " MiB is pinnable";
        long pinnedPlanned = host.pinnableBytes() - Math.max(0, pinnedLeft);
        long residentLeft = host.residentBytes() - pinnedPlanned;
        long ramSlotBytes = alignUp(slotBytes, 4096);
        long ramSlots = residentLeft <= 0 ? 0 : Math.min(totalExperts, residentLeft / ramSlotBytes);
        ResidencyPlan.ExpertStoreMode expertStore;
        long expertRamBytes = 0;
        if (ramSlots >= totalExperts) {
            expertStore = ResidencyPlan.ExpertStoreMode.RAM_RESIDENT;
            expertRamBytes = totalExperts * ramSlotBytes;
        } else if (ramSlots > geometry.slotCount()) {
            // A tier no larger than the device cache would only repeat it.
            expertStore = ResidencyPlan.ExpertStoreMode.RAM_CACHED;
            expertRamBytes = ramSlots * ramSlotBytes;
        } else {
            expertStore = ResidencyPlan.ExpertStoreMode.FILE_BACKED;
            ramSlots = 0;
        }
        residentLeft -= expertRamBytes;
        // The tables are sparse gathers: they are pinned only when every expert is already resident and memory is
        // left over; otherwise they are read from the mapped file, and what is left serves the experts' page cache.
        ResidencyPlan.NgramMode ngram = ResidencyPlan.NgramMode.MAPPED_FILE;
        long ngramPinned = 0;
        long ngramFile = ngramBytes;
        if (expertStore == ResidencyPlan.ExpertStoreMode.RAM_RESIDENT
                && ngramBytes <= pinnedLeft
                && ngramBytes <= residentLeft) {
            ngram = ResidencyPlan.NgramMode.PINNED_ARENA;
            ngramPinned = ngramBytes;
            ngramFile = 0;
        }

        var device = new ResidencyPlan.Device(
                freeBytes, kv, indexer, gdn, workspace, RUNTIME_RESERVE_BYTES, resident, ring, cacheBytes, slack);
        var hostPlan = new ResidencyPlan.Host(
                mapped,
                staged,
                stagingPinned,
                expertRamBytes,
                (int) ramSlots,
                expertBytes,
                ngramPinned,
                ngramFile,
                deferredMtp,
                deferredVision);
        boolean fits = problem == null && geometry.viable();
        return new ResidencyPlan(
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
                ngram,
                chunk,
                host);
    }

    private static ResidencyPlan.Placement deferred() {
        return new ResidencyPlan.Placement(StorageClass.DEFERRED, null);
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
    private static Selection select(List<Tensor> movable, long bytes) {
        if (bytes <= 0) return Selection.NONE;
        Map<Integer, Map<String, List<Tensor>>> ranks = new TreeMap<>();
        for (Tensor tensor : movable)
            ranks.computeIfAbsent(Priority.offloadRank(tensor), rank -> new LinkedHashMap<>())
                    .computeIfAbsent(
                            LAYER_INDEX.matcher(tensor.name()).replaceFirst("/layers/*/"), family -> new ArrayList<>())
                    .add(tensor);
        Set<String> names = new LinkedHashSet<>();
        long staged = 0;
        long stagedMax = 0;
        long mapped = 0;
        long remaining = bytes;
        for (Map<String, List<Tensor>> families : ranks.values()) {
            List<Map.Entry<String, List<Tensor>>> ordered = new ArrayList<>(families.entrySet());
            ordered.sort(Comparator.<Map.Entry<String, List<Tensor>>>comparingLong(
                            entry -> entry.getValue().getFirst().byteSize())
                    .thenComparing(Map.Entry::getKey));
            for (Map.Entry<String, List<Tensor>> family : ordered) {
                List<Tensor> tensors = family.getValue();
                long familyBytes = tensors.stream().mapToLong(Tensor::byteSize).sum();
                List<Tensor> taken = new ArrayList<>();
                if (familyBytes <= remaining) taken.addAll(tensors);
                else {
                    long size = tensors.getFirst().byteSize();
                    int count = (int) Math.min(tensors.size(), (remaining + size - 1) / size);
                    for (int index = 0; index < count; index++)
                        taken.add(tensors.get((int) ((long) index * tensors.size() / count)));
                }
                for (Tensor tensor : taken) {
                    names.add(tensor.name());
                    remaining -= tensor.byteSize();
                    if (Priority.move(tensor) == Priority.Move.MAPPED) mapped += tensor.byteSize();
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
