package io.euhedral_execution.inference.core.model.qwen38.offload;

/// One use of a disk-backed weight in a view: its payload in the artifact file (`byteSize` bytes at `fileOffset`)
/// and the window slot ([DiskWindow]) it is read into. The read covers the whole aligned blocks around the payload,
/// which starts `lead` bytes into the slot.
public record DiskUse(String name, long fileOffset, long byteSize, int slot) {

    public DiskUse {
        if (fileOffset < 0 || byteSize <= 0 || slot < 0 || slot >= DiskWindow.SLOTS)
            throw new IllegalArgumentException("invalid disk use of " + name);
    }

    /// The file offset the read starts at: the payload's, rounded down to a block.
    public long alignedOffset() {
        return this.fileOffset & -DiskWindow.ALIGNMENT;
    }

    /// Where the payload starts within the slot.
    public long lead() {
        return this.fileOffset - alignedOffset();
    }

    /// Bytes of the aligned read: whole blocks from [#alignedOffset] past the payload's end.
    public long readBytes() {
        return DiskWindow.align(lead() + this.byteSize);
    }
}
