package io.euhedral_execution.inference.core.model_loader.qwen4;

/// The fixed 64-byte header of a version 3 artifact, big-endian:
/// `int32 magic, int32 version, int32 architecture, int32 reserved (0), int64 metadataOffset, metadataSize,
/// tablesOffset, tablesSize, dataOffset, fileSize`.
public record Qwen4Header(
        int magic,
        int version,
        int architecture,
        long metadataOffset,
        long metadataSize,
        long tablesOffset,
        long tablesSize,
        long dataOffset,
        long fileSize) {

    public static final int MAGIC = 0x5157454E;
    public static final int VERSION = 3;
    public static final int BYTE_SIZE = 64;
    /// The only architecture of version 3 so far.
    public static final int ARCHITECTURE_QWEN4_EXP = 1;
    /// Payload data starts on this boundary.
    public static final long DATA_ALIGNMENT = 4096;
    /// Every object starts on this boundary.
    public static final long OBJECT_ALIGNMENT = 256;
}
