package io.euhedral_execution.inference.core.model.qwen38.artifact;

/// The fixed 48-byte header of an EDRL artifact.
///
/// All fields are encoded in big-endian byte order. The on-disk reserved field is always zero and
/// is intentionally not part of this record.
public record ArtifactHeader(
        int magic,
        int version,
        long metadataOffset,
        long metadataSize,
        long tensorTableOffset,
        int tensorCount,
        long tensorDataOffset) {

    public static final int MAGIC = 0x5157454E;
    /// The only supported container version: tensor descriptors carry a persistent weight layout.
    public static final int COMPACT_VERSION = 2;
    public static final int BYTE_SIZE = 48;
}
