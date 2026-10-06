package io.euhedral_execution.inference.core.model_loader.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.WeightStaging;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCache;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCacheStats;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.FileExpertStore;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.FileRecordSource;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.GpuExpertTransfer;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.RamTier;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ReplacementPolicy;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// A Flash-Next model whose storage is established and whose execution is not: every object has a
/// known location, the fixed objects the plan keeps on the device are loaded, the routed experts
/// sit behind a bounded device cache over a host store, the n-gram tables are on the host, and MTP
/// and vision stay in the artifact unless the mode selects them.
///
/// The GPU is borrowed and must outlive the model. Close the model only after every lease on its
/// expert cache is returned; closing releases the device slab, the staging ring, the fixed tensors
/// and every host allocation.
public final class Qwen4Model implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(Qwen4Model.class);

    /// Threads that read and check fixed objects while loading.
    static final int LOAD_THREADS = 8;

    /// Readers that fill a resident expert tier at startup, each taking about 32 MiB of contiguous
    /// records at a time.
    static final int PRELOAD_READERS = 16;

    /// How the device cache shares its slots among the layers. `EUHEDRAL_QWEN4_GPU_POLICY`
    /// (`global` or `partitioned`) overrides it for benchmarks.
    static final ReplacementPolicy DEVICE_POLICY = "partitioned".equals(System.getenv("EUHEDRAL_QWEN4_GPU_POLICY"))
            ? ReplacementPolicy.BANK_PARTITIONED
            : ReplacementPolicy.GLOBAL_LRU;

    /// Parts one record's artifact read is split into, read side by side as frames of the lattice.
    /// `euhedral.qwen4.read-parts` (or `EUHEDRAL_QWEN4_READ_PARTS`) overrides it for benchmarks.
    static final int READ_PARTS = Integer.getInteger(
            "euhedral.qwen4.read-parts",
            Integer.parseInt(System.getenv().getOrDefault("EUHEDRAL_QWEN4_READ_PARTS", "4")));

    private final Qwen4Artifact artifact;
    private final Qwen4ResidencyPlan plan;
    private final Qwen4FixedLoader.Loaded fixed;
    private final ExecutionGpu gpu;
    private final NgramStore ngram;
    private final ExpertCache cache;
    private final ExpertBank[] cachedBanks;
    private final FileExpertStore fileStore;
    private final FileRecordSource artifactSource;
    private final GpuExpertTransfer gpuTransfer;
    private boolean closed;

    private Qwen4Model(
            Qwen4Artifact artifact,
            Qwen4ResidencyPlan plan,
            Qwen4FixedLoader.Loaded fixed,
            ExecutionGpu gpu,
            NgramStore ngram,
            ExpertCache cache,
            ExpertBank[] cachedBanks,
            FileExpertStore fileStore,
            FileRecordSource artifactSource,
            GpuExpertTransfer gpuTransfer) {
        this.artifact = artifact;
        this.plan = plan;
        this.fixed = fixed;
        this.gpu = gpu;
        this.ngram = ngram;
        this.cache = cache;
        this.cachedBanks = cachedBanks;
        this.fileStore = fileStore;
        this.artifactSource = artifactSource;
        this.gpuTransfer = gpuTransfer;
    }

    /// Reads and validates the artifact, plans its residency for `freeDeviceBytes`, `host` and
    /// `maxContextTokens`, and loads the plan. Fails with the plan's explanation when no placement
    /// can serve the context.
    public static Qwen4Model open(
            Path path, ExecutionGpu gpu, long freeDeviceBytes, HostBudget host, Qwen4Mode mode, int maxContextTokens)
            throws IOException {
        Qwen4Artifact artifact = Qwen4ArtifactReader.read(path);
        Qwen4Validator.validateInventory(artifact);
        Qwen4ResidencyPlan plan = Qwen4ResidencyPlanner.plan(artifact, mode, freeDeviceBytes, host, maxContextTokens);
        return load(path, artifact, plan, gpu);
    }

    /// Loads `plan`, which must have been made for `artifact`.
    public static Qwen4Model load(Path path, Qwen4Artifact artifact, Qwen4ResidencyPlan plan, ExecutionGpu gpu)
            throws IOException {
        LOG.info("{}", plan.report());
        if (!plan.fits())
            throw new IOException(
                    "a context of " + plan.maxContextTokens() + " tokens cannot be placed: " + plan.explanation());
        List<ExpertBank> banks = new ArrayList<>();
        for (ExpertBank bank : artifact.banks())
            if (plan.banks().get(bank.name()) == StorageClass.DEVICE_CACHED) banks.add(bank);
        ExpertBank[] cachedBanks = banks.toArray(ExpertBank[]::new);

        Qwen4FixedLoader.Loaded fixed = Qwen4FixedLoader.load(path, artifact, plan, gpu, LOAD_THREADS);
        NgramStore ngram = null;
        FileRecordSource artifactSource = null;
        RamTier tier = null;
        FileExpertStore store = null;
        GpuExpertTransfer transfer = null;
        try {
            ngram = NgramStore.open(path, artifact, plan.ngram(), gpu);
            int lanes = Qwen4ResidencyPlanner.fileStagingSlots(artifact.config());
            int shards = Qwen4ResidencyPlanner.expertShards(plan.expertCache().slotCount());
            artifactSource = new FileRecordSource(path, cachedBanks);
            if (plan.expertStore() != Qwen4ResidencyPlan.ExpertStoreMode.FILE_BACKED) {
                tier = new RamTier(
                        cachedBanks, plan.host().expertRamSlots(), shards, ReplacementPolicy.BANK_PARTITIONED);
                if (tier.isResident()) {
                    long begin = System.nanoTime();
                    tier.preload(artifactSource, PRELOAD_READERS);
                    LOG.info(
                            "Expert tier loaded: {} records, {} MiB in {} ms ({} MiB/s)",
                            tier.slotCount(),
                            tier.capacityBytes() >> 20,
                            (System.nanoTime() - begin) / 1_000_000,
                            (long) (tier.stats().preloadBytesPerSecond() / 1048576.0));
                }
            }
            // The store owns the source and the tier from here on, and closes them.
            FileRecordSource owned = artifactSource;
            RamTier ownedTier = tier;
            store = new FileExpertStore(gpu, owned, ownedTier, cachedBanks, lanes, READ_PARTS);
            transfer = new GpuExpertTransfer(gpu, lanes);
            ExpertCache cache = new ExpertCache(
                    store,
                    transfer,
                    gpu,
                    plan.expertCache().slotCount(),
                    plan.expertCache().slotBytes(),
                    shards);
            return new Qwen4Model(artifact, plan, fixed, gpu, ngram, cache, cachedBanks, store, owned, transfer);
        } catch (Throwable failure) {
            closeQuietly(transfer, failure);
            if (store != null) closeQuietly(store, failure);
            else {
                closeQuietly(artifactSource, failure);
                closeQuietly(tier, failure);
            }
            closeQuietly(ngram, failure);
            Qwen4FixedLoader.release(fixed.handles(), fixed.hostArena(), staging(fixed), gpu, failure);
            if (failure instanceof IOException io) throw io;
            if (failure instanceof RuntimeException runtime) throw runtime;
            if (failure instanceof Error error) throw error;
            throw new IOException(failure);
        }
    }

    private static long staging(Qwen4FixedLoader.Loaded fixed) {
        return fixed.staging() == null ? 0 : fixed.staging().baseAddress();
    }

    private static void closeQuietly(AutoCloseable resource, Throwable failure) {
        if (resource == null) return;
        try {
            resource.close();
        } catch (Exception cleanup) {
            failure.addSuppressed(cleanup);
        }
    }

    public Qwen4Artifact artifact() {
        return this.artifact;
    }

    public Qwen4ResidencyPlan plan() {
        return this.plan;
    }

    /// The fixed objects that are loaded (device resident, host staged or host mapped), by name;
    /// deferred objects and the n-gram tables are not here.
    public Map<String, TensorHandle> tensors() {
        return Collections.unmodifiableMap(this.fixed.handles());
    }

    public TensorHandle tensor(String name) {
        TensorHandle handle = this.fixed.handles().get(name);
        if (handle == null) throw new IllegalArgumentException("not a loaded object: " + name);
        return handle;
    }

    /// Pinned staging slots every expert record passes through when it is read from the artifact:
    /// the most loads that can be outstanding at once.
    public int stagingSlots() {
        return Qwen4ResidencyPlanner.fileStagingSlots(this.artifact.config());
    }

    /// The device ring that host-staged fixed objects pass through, or null when none is
    /// host-staged.
    public WeightStaging staging() {
        return this.fixed.staging();
    }

    /// The routed experts' device cache. A bank's ordinal in it is its index in [#expertBanks()].
    public ExpertCache expertCache() {
        return this.cache;
    }

    public ExpertBank[] expertBanks() {
        return this.cachedBanks.clone();
    }

    /// The ordinal of the named bank in [#expertCache()].
    public int bankOrdinal(String bankName) {
        for (int i = 0; i < this.cachedBanks.length; i++)
            if (this.cachedBanks[i].name().equals(bankName)) return i;
        throw new IllegalArgumentException("not a cached expert bank: " + bankName);
    }

    public NgramStore ngram() {
        return this.ngram;
    }

    public Qwen4Telemetry telemetry() {
        ExpertCacheStats.Snapshot cacheStats = this.cache.stats().snapshot();
        NgramStore.Stats ngramStats = this.ngram.stats();
        var device = this.plan.device();
        long hostBacked = this.plan.host().stagedBytes() + mappedFixedBytes();
        return new Qwen4Telemetry(
                this.fixed.residentBytes(),
                hostBacked,
                this.plan.host().mappedBytes(),
                device.contextBytes() + device.workspaceBytes() + device.runtimeReserveBytes(),
                this.cache.capacityBytes(),
                this.cache.slotCount(),
                cacheStats.hits(),
                cacheStats.misses(),
                cacheStats.evictions(),
                cacheStats.transferBytes(),
                cacheStats.transferNanos(),
                cacheStats.waitNanos(),
                ngramStats.bytesStagedToDevice(),
                ngramStats.rowsGathered(),
                this.ngram.hostBytes());
    }

    /// A reading of every tier of the expert hierarchy.
    public ExpertHierarchyStats hierarchyStats() {
        RamTier tier = this.fileStore.ramTier();
        ExpertCacheStats.Snapshot gpuStats = this.cache.stats().snapshot();
        return new ExpertHierarchyStats(
                gpuStats,
                tier == null ? null : tier.stats(),
                new ExpertHierarchyStats.Artifact(
                        this.artifactSource.recordReads(),
                        this.artifactSource.bytesRead(),
                        this.artifactSource.readNanos(),
                        this.artifactSource.concurrentReadsHighWater()),
                new ExpertHierarchyStats.Staging(
                        this.fileStore.lanes(),
                        this.fileStore.recordOpens(),
                        this.fileStore.ramCopyBytes(),
                        this.fileStore.ramCopyNanos()),
                new ExpertHierarchyStats.H2d(
                        gpuStats.transferBytes(), gpuStats.transferNanos(), this.gpuTransfer.submittedCopies()));
    }

    /// Mapped bytes of fixed objects other than the token embedding.
    private long mappedFixedBytes() {
        long total = 0;
        for (TensorHandle handle : this.fixed.handles().values())
            if (handle.hostMapped() && !handle.name().equals("text/token_embedding")) total += handle.byteSize();
        return total;
    }

    @Override
    public synchronized void close() {
        if (this.closed) return;
        this.closed = true;
        RuntimeException failure = null;
        try {
            this.cache.close();
        } catch (RuntimeException e) {
            failure = e;
        }
        try {
            this.ngram.close();
        } catch (RuntimeException e) {
            if (failure == null) failure = e;
            else failure.addSuppressed(e);
        }
        Throwable release = new Throwable("release");
        Qwen4FixedLoader.release(this.fixed.handles(), this.fixed.hostArena(), staging(this.fixed), this.gpu, release);
        if (failure == null && release.getSuppressed().length > 0) {
            failure = new IllegalStateException("releasing fixed objects failed", release.getSuppressed()[0]);
        }
        if (failure != null) throw failure;
    }
}
