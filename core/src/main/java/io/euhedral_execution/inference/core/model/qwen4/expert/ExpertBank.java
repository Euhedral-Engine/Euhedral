package io.euhedral_execution.inference.core.model.qwen4.expert;

import io.euhedral_execution.inference.core.model.qwen4.loader.ComponentGroup;
import java.util.List;
import java.util.Objects;

/// The routed experts of one layer: `expertCount` records, each holding every projection of one expert in one
/// contiguous byte range of the artifact (`fileOffset(expert)`, `recordBytes(expert)`), so a cache miss is one
/// copy. Records need not be equal in size; `maxRecordBytes` is the slot size that holds any of them.
///
/// A bank keeps its record table in primitive arrays: a model has tens of thousands of experts and none of them
/// is an object.
public final class ExpertBank {
    private final String name;
    private final ComponentGroup group;
    private final int layer;
    private final List<ExpertProjection> projections;
    private final long[] offsets;
    private final long[] sizes;
    private final int[] crcs;
    private final long maxRecordBytes;
    private final long totalBytes;

    public ExpertBank(
            String name,
            ComponentGroup group,
            int layer,
            List<ExpertProjection> projections,
            long[] offsets,
            long[] sizes,
            int[] crcs) {
        this.name = Objects.requireNonNull(name, "name");
        this.group = Objects.requireNonNull(group, "group");
        this.layer = layer;
        this.projections = List.copyOf(projections);
        if (this.projections.isEmpty()) throw new IllegalArgumentException("an expert bank needs projections");
        if (offsets.length == 0 || offsets.length != sizes.length || offsets.length != crcs.length)
            throw new IllegalArgumentException("expert index arrays must be non-empty and equally long");
        this.offsets = offsets.clone();
        this.sizes = sizes.clone();
        this.crcs = crcs.clone();
        long recordEnd = 0;
        for (ExpertProjection projection : this.projections)
            recordEnd = Math.max(recordEnd, Math.addExact(projection.recordOffset(), projection.byteSize()));
        long max = 0;
        long total = 0;
        for (int expert = 0; expert < sizes.length; expert++) {
            if (offsets[expert] < 0 || sizes[expert] < recordEnd)
                throw new IllegalArgumentException(
                        "expert " + expert + " of " + name + " has a record smaller than its projections");
            max = Math.max(max, sizes[expert]);
            total = Math.addExact(total, sizes[expert]);
        }
        this.maxRecordBytes = max;
        this.totalBytes = total;
    }

    public String name() {
        return this.name;
    }

    public ComponentGroup group() {
        return this.group;
    }

    /// The layer within its stack (the text layers, or the MTP layer).
    public int layer() {
        return this.layer;
    }

    public int expertCount() {
        return this.offsets.length;
    }

    public List<ExpertProjection> projections() {
        return this.projections;
    }

    public ExpertProjection projection(String projectionName) {
        for (ExpertProjection projection : this.projections)
            if (projection.name().equals(projectionName)) return projection;
        throw new IllegalArgumentException(this.name + " has no projection " + projectionName);
    }

    /// Offset of the expert's record in the artifact file.
    public long fileOffset(int expert) {
        return this.offsets[checked(expert)];
    }

    public long recordBytes(int expert) {
        return this.sizes[checked(expert)];
    }

    /// CRC-32 (IEEE) of the expert's record.
    public int crc32(int expert) {
        return this.crcs[checked(expert)];
    }

    /// The bytes of the largest record: the slot size that holds any expert of the bank.
    public long maxRecordBytes() {
        return this.maxRecordBytes;
    }

    public long totalBytes() {
        return this.totalBytes;
    }

    private int checked(int expert) {
        if (expert < 0 || expert >= this.offsets.length)
            throw new IndexOutOfBoundsException(
                    "expert " + expert + " of " + this.name + " (" + this.offsets.length + ")");
        return expert;
    }
}
