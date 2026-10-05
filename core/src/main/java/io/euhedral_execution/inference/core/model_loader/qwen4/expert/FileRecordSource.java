package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.atomic.LongAdder;

/// Reads records straight from the artifact file with positional reads, which share one channel between
/// threads. Every record of every bank is checked against the file's size when the source opens.
public final class FileRecordSource implements RecordSource {
    private final FileChannel channel;
    private final LongAdder bytesRead = new LongAdder();

    public FileRecordSource(Path file, ExpertBank[] banks) throws IOException {
        this.channel = FileChannel.open(file, StandardOpenOption.READ);
        try {
            ExpertFiles.validate(banks, this.channel.size());
        } catch (IOException | RuntimeException failure) {
            closeQuietly(failure);
            throw failure;
        }
    }

    @Override
    public void read(ExpertBank bank, int expert, MemorySegment destination) throws IOException {
        long size = bank.recordBytes(expert);
        if (destination.byteSize() != size)
            throw new IllegalArgumentException(
                    "destination holds " + destination.byteSize() + " bytes, record " + size);
        ExpertFiles.readFully(this.channel, destination, bank.fileOffset(expert));
        this.bytesRead.add(size);
    }

    @Override
    public long bytesRead() {
        return this.bytesRead.sum();
    }

    @Override
    public void close() {
        try {
            this.channel.close();
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }

    private void closeQuietly(Throwable failure) {
        try {
            this.channel.close();
        } catch (IOException closeFailure) {
            failure.addSuppressed(closeFailure);
        }
    }
}
