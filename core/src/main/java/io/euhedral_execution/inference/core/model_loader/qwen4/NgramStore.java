package io.euhedral_execution.inference.core.model_loader.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model_loader.artifact.ArtifactFileAccess;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactFormatException;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.atomic.LongAdder;

/// The n-gram embedding tables, held on the host and addressed by row. They are sparse lookup data (a few rows per
/// token out of hundreds of millions), not expert weights: nothing here touches the expert cache, and the device only
/// ever receives the rows a step asks for.
///
/// The table is `splitParts` shards of `shardRows` rows
/// ([NgramLayout]); the configuration's hash heads each address a contiguous slice of `headsVocabSizes[head]` rows
/// starting at `headsOffsets[head]`. A global row is `shard * shardRows + localRow`. Rows stay where the artifact has
/// them: [Qwen4ResidencyPlan.NgramMode#MAPPED_FILE] maps the shards read-only (the operating system keeps the rows
/// that are used in memory), [Qwen4ResidencyPlan.NgramMode#PINNED_ARENA] reads them into one pinned arena.
public final class NgramStore implements AutoCloseable {

    /// Counters for the next stage of the engine.
    public record Stats(long gathers, long rowsGathered, long bytesGathered, long bytesStagedToDevice) {}

    private final Qwen4Config.Ngram config;
    private final Qwen4ResidencyPlan.NgramMode mode;
    private final int rowWidth;
    private final int rowBytes;
    private final MemorySegment[] shards;
    private final float[] globalScales;
    private final long hostBytes;
    private final Arena mapping;
    private final GpuMemory gpu;
    private final long arenaAddress;
    private final LongAdder gathers = new LongAdder();
    private final LongAdder rowsGathered = new LongAdder();
    private final LongAdder bytesStaged = new LongAdder();
    private boolean closed;

    private NgramStore(
            Qwen4Config.Ngram config,
            Qwen4ResidencyPlan.NgramMode mode,
            int rowWidth,
            MemorySegment[] shards,
            float[] globalScales,
            long hostBytes,
            Arena mapping,
            GpuMemory gpu,
            long arenaAddress) {
        this.config = config;
        this.mode = mode;
        this.rowWidth = rowWidth;
        this.rowBytes = (int) NgramLayout.rowBytes(rowWidth);
        this.shards = shards;
        this.globalScales = globalScales;
        this.hostBytes = hostBytes;
        this.mapping = mapping;
        this.gpu = gpu;
        this.arenaAddress = arenaAddress;
    }

    /// Opens the tables of `artifact` as `mode` says. `gpu` supplies the pinned arena for
    /// [Qwen4ResidencyPlan.NgramMode#PINNED_ARENA] and may be null for a mapped file.
    public static NgramStore open(Path path, Qwen4Artifact artifact, Qwen4ResidencyPlan.NgramMode mode, GpuMemory gpu)
            throws IOException {
        Qwen4Config config = artifact.config();
        Qwen4Config.Ngram ngram = config.ngram();
        int layer = config.ple().layers()[0];
        Qwen4Tensor[] tensors = new Qwen4Tensor[ngram.splitParts()];
        int rowWidth = config.text().hiddenSize() / ngram.heads();
        long total = 0;
        for (int shard = 0; shard < tensors.length; shard++) {
            String name = "text/layers/" + layer + "/ple/ngram/" + Qwen4Inventory.shardName(shard);
            tensors[shard] = artifact.tensor(name)
                    .orElseThrow(() -> new QwenArtifactFormatException("artifact has no n-gram shard " + name));
            total += tensors[shard].byteSize();
        }
        MemorySegment[] segments = new MemorySegment[tensors.length];
        float[] scales = new float[tensors.length];
        Arena mapping = Arena.ofShared();
        long arenaAddress = 0;
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            if (mode == Qwen4ResidencyPlan.NgramMode.PINNED_ARENA) {
                if (gpu == null) throw new IllegalArgumentException("a pinned n-gram arena needs a GPU");
                long[] starts = new long[tensors.length];
                long cursor = 0;
                for (int i = 0; i < tensors.length; i++) {
                    starts[i] = cursor;
                    cursor += alignUp(tensors[i].byteSize(), 4096);
                }
                arenaAddress = gpu.allocateHostWeights(cursor);
                MemorySegment arena = MemorySegment.ofAddress(arenaAddress).reinterpret(cursor);
                // Shards are independent ranges: read them concurrently.
                var failures = new java.util.concurrent.ConcurrentLinkedQueue<Throwable>();
                var threads = new java.util.ArrayList<Thread>();
                int workers = Math.min(8, tensors.length);
                for (int worker = 0; worker < workers; worker++) {
                    final int first = worker;
                    Thread thread = Thread.ofPlatform()
                            .name("ngram-load-" + worker)
                            .start(() -> {
                                try {
                                    for (int shard = first; shard < tensors.length; shard += workers) {
                                        MemorySegment target = arena.asSlice(starts[shard], tensors[shard].byteSize());
                                        ArtifactFileAccess.readFully(
                                                channel, tensors[shard].dataOffset(), target, tensors[shard].name());
                                        segments[shard] = target;
                                    }
                                } catch (Throwable failure) {
                                    failures.add(failure);
                                }
                            });
                    threads.add(thread);
                }
                for (Thread thread : threads) join(thread);
                if (!failures.isEmpty()) {
                    Throwable first = failures.peek();
                    if (first instanceof IOException io) throw io;
                    throw new IOException("reading the n-gram tables failed", first);
                }
            } else {
                for (int shard = 0; shard < tensors.length; shard++)
                    segments[shard] = channel.map(
                            FileChannel.MapMode.READ_ONLY,
                            tensors[shard].dataOffset(),
                            tensors[shard].byteSize(),
                            mapping);
            }
            for (int shard = 0; shard < tensors.length; shard++) {
                long trailer = NgramLayout.trailerOffset(tensors[shard].shape()[0], rowWidth);
                float scale = segments[shard].get(
                        ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), trailer);
                if (!Float.isFinite(scale) || scale < 0)
                    throw new QwenArtifactFormatException(
                            "global scale of " + tensors[shard].name() + " is not a finite non-negative number");
                scales[shard] = scale;
            }
        } catch (Throwable failure) {
            mapping.close();
            if (arenaAddress != 0) {
                try {
                    gpu.freeHostWeights(arenaAddress);
                } catch (RuntimeException cleanup) {
                    failure.addSuppressed(cleanup);
                }
            }
            if (failure instanceof IOException io) throw io;
            if (failure instanceof RuntimeException runtime) throw runtime;
            if (failure instanceof Error error) throw error;
            throw new IOException(failure);
        }
        return new NgramStore(ngram, mode, rowWidth, segments, scales, total, mapping, gpu, arenaAddress);
    }

    private static void join(Thread thread) throws IOException {
        try {
            thread.join();
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while reading the n-gram tables", interrupted);
        }
    }

    private static long alignUp(long value, long alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    public Qwen4ResidencyPlan.NgramMode mode() {
        return this.mode;
    }

    /// Bytes of the tables on the host.
    public long hostBytes() {
        return this.hostBytes;
    }

    public int heads() {
        return this.config.heads();
    }

    /// Values per row.
    public int rowWidth() {
        return this.rowWidth;
    }

    /// Bytes per row: the row's codes then its block scales.
    public int rowBytes() {
        return this.rowBytes;
    }

    public long totalRows() {
        return this.config.totalRows();
    }

    public long shardRows() {
        return this.config.shardRows();
    }

    public int shardCount() {
        return this.shards.length;
    }

    /// The global row of entry `index` of hash head `head`.
    public long globalRow(int head, long index) {
        if (head < 0 || head >= heads()) throw new IndexOutOfBoundsException("head " + head);
        if (index < 0 || index >= this.config.headsVocabSizes()[head])
            throw new IndexOutOfBoundsException(
                    "entry " + index + " of head " + head + " (" + this.config.headsVocabSizes()[head] + " entries)");
        return this.config.headsOffsets()[head] + index;
    }

    public int shardOf(long globalRow) {
        return (int) (checked(globalRow) / this.config.shardRows());
    }

    public long localRow(long globalRow) {
        return checked(globalRow) % this.config.shardRows();
    }

    /// Bytes of the mapped tables the operating system holds in memory now (`mincore` over each shard's pages), or
    /// [#hostBytes] for a pinned arena: how much of a mapped table a gather finds without a read from the disk.
    public long residentBytes() {
        ensureOpen();
        if (this.mode != Qwen4ResidencyPlan.NgramMode.MAPPED_FILE) return this.hostBytes;
        long page = 4096, resident = 0;
        try (Arena scratch = Arena.ofConfined()) {
            for (MemorySegment shard : this.shards) {
                long start = shard.address() & -page;
                long length = shard.address() + shard.byteSize() - start;
                long pages = (length + page - 1) / page;
                MemorySegment vector = scratch.allocate(pages);
                int status = (int) MINCORE.invokeExact(MemorySegment.ofAddress(start), length, vector);
                if (status != 0) throw new IllegalStateException("mincore failed");
                for (long i = 0; i < pages; i++) if ((vector.get(ValueLayout.JAVA_BYTE, i) & 1) != 0) resident++;
            }
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new IllegalStateException(failure);
        }
        return Math.min(resident * page, this.hostBytes);
    }

    private static final java.lang.invoke.MethodHandle MINCORE = java.lang.foreign.Linker.nativeLinker()
            .downcallHandle(
                    java.lang.foreign.Linker.nativeLinker()
                            .defaultLookup()
                            .find("mincore")
                            .orElseThrow(),
                    java.lang.foreign.FunctionDescriptor.of(
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));

    /// The table's FP32 global scale for the shard.
    public float globalScale(int shard) {
        return this.globalScales[shard];
    }

    /// A read-only view of the row's `rowBytes()` bytes in the host tables.
    public MemorySegment row(long globalRow) {
        long row = checked(globalRow);
        long local = row % this.config.shardRows();
        return this.shards[(int) (row / this.config.shardRows())]
                .asSlice(local * this.rowBytes, this.rowBytes)
                .asReadOnly();
    }

    /// Copies the rows `globalRows[0..count)` one after another into `destination` (host memory of at least
    /// `count * rowBytes()` bytes): the gather a step does before staging.
    public void gather(long[] globalRows, int count, MemorySegment destination) {
        ensureOpen();
        if (count < 0 || count > globalRows.length) throw new IllegalArgumentException("count");
        if (destination.byteSize() < (long) count * this.rowBytes)
            throw new IllegalArgumentException("destination holds " + destination.byteSize() + " bytes");
        for (int i = 0; i < count; i++)
            MemorySegment.copy(row(globalRows[i]), 0, destination, (long) i * this.rowBytes, this.rowBytes);
        this.gathers.increment();
        this.rowsGathered.add(count);
    }

    /// Gathers the rows into pinned host memory and queues their copy to the device at `deviceAddress` on the
    /// selected
    /// stream. The caller closes the returned buffer once the copy has retired.
    public ExecutionGpu.UploadBuffer stage(ExecutionGpu gpu, long[] globalRows, int count, long deviceAddress) {
        ensureOpen();
        if (count <= 0) throw new IllegalArgumentException("count must be positive");
        long bytes = (long) count * this.rowBytes;
        ExecutionGpu.UploadBuffer upload = gpu.allocateUploadBuffer(bytes);
        try {
            gather(globalRows, count, upload.segment());
            gpu.copyUploadToDevice(deviceAddress, upload);
        } catch (Throwable failure) {
            upload.close();
            throw failure;
        }
        this.bytesStaged.add(bytes);
        return upload;
    }

    /// Bytes of one staged record: the row, zero padding to a multiple of four, then the row's shard's FP32 global
    /// scale in
    /// the last four bytes. A kernel decodes a record on its own.
    public int recordBytes() {
        return (this.rowBytes + 4 + 3) / 4 * 4;
    }

    /// Like [#gather], but each row becomes a [#recordBytes] record that carries its shard's global scale.
    public void gatherRecords(long[] globalRows, int count, MemorySegment destination) {
        ensureOpen();
        if (count < 0 || count > globalRows.length) throw new IllegalArgumentException("count");
        if (destination.byteSize() < (long) count * recordBytes())
            throw new IllegalArgumentException("destination holds " + destination.byteSize() + " bytes");
        gatherRecordsRange(globalRows, 0, count, destination);
        this.gathers.increment();
    }

    /// The records of rows `[from, to)` of `globalRows`, each at its own place in `destination` (record `i`
    /// at `i * recordBytes()`): callers that gather disjoint ranges of one list side by side fill one buffer.
    public void gatherRecordsRange(long[] globalRows, int from, int to, MemorySegment destination) {
        ensureOpen();
        if (from < 0 || to < from || to > globalRows.length) throw new IllegalArgumentException("range");
        int recordBytes = recordBytes();
        if (destination.byteSize() < (long) to * recordBytes)
            throw new IllegalArgumentException("destination holds " + destination.byteSize() + " bytes");
        for (int i = from; i < to; i++) {
            long row = checked(globalRows[i]);
            long at = (long) i * recordBytes;
            MemorySegment.copy(row(row), 0, destination, at, this.rowBytes);
            destination
                    .asSlice(at + this.rowBytes, recordBytes - 4 - this.rowBytes)
                    .fill((byte) 0);
            destination.set(
                    ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                    at + recordBytes - 4,
                    this.globalScales[(int) (row / this.config.shardRows())]);
        }
        this.rowsGathered.add(to - from);
    }

    /// [#gatherRecords] into pinned memory and a queued copy to `deviceAddress` on the selected stream. The caller
    /// closes the returned buffer once the copy has retired.
    public ExecutionGpu.UploadBuffer stageRecords(ExecutionGpu gpu, long[] globalRows, int count, long deviceAddress) {
        ensureOpen();
        if (count <= 0) throw new IllegalArgumentException("count must be positive");
        long bytes = (long) count * recordBytes();
        ExecutionGpu.UploadBuffer upload = gpu.allocateUploadBuffer(bytes);
        try {
            gatherRecords(globalRows, count, upload.segment());
            gpu.copyUploadToDevice(deviceAddress, upload);
        } catch (Throwable failure) {
            upload.close();
            throw failure;
        }
        this.bytesStaged.add(bytes);
        return upload;
    }

    /// Records that `bytes` of staged records were copied to the device by a caller that gathered them itself.
    public void recordStaged(long bytes) {
        this.bytesStaged.add(bytes);
    }

    public Stats stats() {
        return new Stats(
                this.gathers.sum(),
                this.rowsGathered.sum(),
                this.rowsGathered.sum() * this.rowBytes,
                this.bytesStaged.sum());
    }

    private long checked(long globalRow) {
        if (globalRow < 0 || globalRow >= this.config.totalRows())
            throw new IndexOutOfBoundsException("row " + globalRow + " of " + this.config.totalRows());
        return globalRow;
    }

    private void ensureOpen() {
        if (this.closed) throw new IllegalStateException("the n-gram store is closed");
    }

    @Override
    public synchronized void close() {
        if (this.closed) return;
        this.closed = true;
        this.mapping.close();
        if (this.arenaAddress != 0) this.gpu.freeHostWeights(this.arenaAddress);
    }
}
