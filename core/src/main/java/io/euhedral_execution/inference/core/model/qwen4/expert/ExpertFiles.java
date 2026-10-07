package io.euhedral_execution.inference.core.model.qwen4.expert;

import java.io.EOFException;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

/// Checks and reads shared by the stores that serve expert records from an artifact file.
final class ExpertFiles {
    /// The largest single positional read: a direct buffer over a segment is limited to an `int` length.
    private static final long READ_CHUNK = 1L << 30;

    private ExpertFiles() {}

    /// Requires every record of every bank to lie inside a file of `fileSize` bytes.
    static void validate(ExpertBank[] banks, long fileSize) {
        for (int ordinal = 0; ordinal < banks.length; ordinal++) {
            ExpertBank bank = banks[ordinal];
            for (int expert = 0; expert < bank.expertCount(); expert++) {
                long offset = bank.fileOffset(expert);
                long size = bank.recordBytes(expert);
                if (offset < 0 || size <= 0 || offset > fileSize || size > fileSize - offset)
                    throw new IllegalArgumentException("expert " + expert + " of " + bank.name() + " (bank " + ordinal
                            + ") lies outside the file: offset " + offset + ", " + size + " bytes, file " + fileSize
                            + " bytes");
            }
        }
    }

    static long maxRecordBytes(ExpertBank[] banks) {
        long max = 0;
        for (ExpertBank bank : banks) max = Math.max(max, bank.maxRecordBytes());
        return max;
    }

    static long alignUp(long value, long alignment) {
        return Math.multiplyExact(Math.addExact(value, alignment - 1) / alignment, alignment);
    }

    /// Fills `destination` from `channel` starting at `position` with positional reads, so concurrent callers
    /// share the channel.
    static void readFully(FileChannel channel, MemorySegment destination, long position) throws IOException {
        long done = 0;
        long size = destination.byteSize();
        while (done < size) {
            ByteBuffer buffer =
                    destination.asSlice(done, Math.min(READ_CHUNK, size - done)).asByteBuffer();
            int read = channel.read(buffer, position + done);
            if (read < 0) throw new EOFException("artifact ends " + (size - done) + " bytes before the record does");
            done += read;
        }
    }
}
