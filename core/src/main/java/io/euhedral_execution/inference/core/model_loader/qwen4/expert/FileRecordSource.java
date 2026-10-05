package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.channels.ClosedByInterruptException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.LongAdder;

/// Reads records straight from the artifact file with positional reads. Every record of every bank is checked
/// against the file's size when the source opens.
///
/// Each read opens its own channel. A [FileChannel] is closed for everyone when a thread blocked in it is
/// interrupted, and records are read on the threads that ask for them, which callers may interrupt: a channel
/// shared between reads would be lost to the first interrupted acquirer. An interrupted read fails with
/// [InterruptedException] and affects nothing else; opening a channel is negligible beside a record's read.
public final class FileRecordSource implements RecordSource {
    private final Path file;
    private final LongAdder bytesRead = new LongAdder();
    private final AtomicBoolean closed = new AtomicBoolean();

    public FileRecordSource(Path file, ExpertBank[] banks) throws IOException {
        this.file = file;
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            ExpertFiles.validate(banks, channel.size());
        }
    }

    @Override
    public void read(ExpertBank bank, int expert, MemorySegment destination) throws IOException, InterruptedException {
        long size = bank.recordBytes(expert);
        if (destination.byteSize() != size)
            throw new IllegalArgumentException(
                    "destination holds " + destination.byteSize() + " bytes, record " + size);
        if (this.closed.get()) throw new IllegalStateException("the record source is closed");
        try (FileChannel channel = FileChannel.open(this.file, StandardOpenOption.READ)) {
            ExpertFiles.readFully(channel, destination, bank.fileOffset(expert));
        } catch (ClosedByInterruptException interrupted) {
            // The interrupt closed this read's channel and left the flag set; the exception consumes it.
            Thread.interrupted();
            throw new InterruptedException("interrupted while reading expert " + expert + " of " + bank.name());
        }
        this.bytesRead.add(size);
    }

    @Override
    public long bytesRead() {
        return this.bytesRead.sum();
    }

    @Override
    public void close() {
        this.closed.set(true);
    }
}
