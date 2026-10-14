package io.euhedral_execution.inference.quantization.checkpoint;

import java.io.EOFException;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Properties;

/// The one file a long run overwrites in place and resumes from: a header of properties (what the run is, how
/// far it got, which sections hold whole data) and the sections after it at fixed offsets. A writer marks a
/// section invalid in the header before overwriting it and valid once the data is on disk, so a stop part way
/// through leaves nothing to resume from rather than a mix.
public final class ResumeFile implements AutoCloseable {

    /// Bytes of the header; the sections start after it.
    public static final long HEADER = 4096;

    /// Bytes one buffer covers (a ByteBuffer covers at most 2 GiB).
    private static final long SLICE = 1L << 30;

    private final FileChannel channel;

    public ResumeFile(Path file) throws IOException {
        this.channel =
                FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE);
    }

    /// The header's properties; empty for a new file.
    public Properties header() throws IOException {
        Properties properties = new Properties();
        if (channel.size() < HEADER) return properties;
        ByteBuffer buffer = ByteBuffer.allocate((int) HEADER);
        while (buffer.hasRemaining()) if (channel.read(buffer, buffer.position()) < 0) return properties;
        String text = new String(buffer.array(), StandardCharsets.UTF_8);
        int end = text.indexOf('\0');
        properties.load(new StringReader(end < 0 ? text : text.substring(0, end)));
        return properties;
    }

    /// Replaces the header and forces it, with every section written before it, to disk.
    public void header(Properties properties) throws IOException {
        StringWriter text = new StringWriter();
        properties.store(text, null);
        byte[] bytes = text.toString().getBytes(StandardCharsets.UTF_8);
        if (bytes.length > HEADER) throw new IOException("resume header of " + bytes.length + " bytes");
        channel.force(false);
        ByteBuffer buffer = ByteBuffer.allocate((int) HEADER).put(bytes).flip().limit((int) HEADER);
        while (buffer.hasRemaining()) channel.write(buffer, buffer.position());
        channel.force(false);
    }

    /// Writes `segment` at `offset`; returns the offset after it.
    public long write(long offset, MemorySegment segment) throws IOException {
        for (long at = 0; at < segment.byteSize(); at += SLICE) {
            ByteBuffer buffer = segment.asSlice(at, Math.min(SLICE, segment.byteSize() - at))
                    .asByteBuffer();
            long position = offset + at;
            while (buffer.hasRemaining()) position += channel.write(buffer, position);
        }
        return offset + segment.byteSize();
    }

    /// Fills `segment` from `offset`; returns the offset after it.
    public long read(long offset, MemorySegment segment) throws IOException {
        for (long at = 0; at < segment.byteSize(); at += SLICE) {
            ByteBuffer buffer = segment.asSlice(at, Math.min(SLICE, segment.byteSize() - at))
                    .asByteBuffer();
            long position = offset + at;
            while (buffer.hasRemaining()) {
                int read = channel.read(buffer, position);
                if (read < 0) throw new EOFException("resume file ends at " + position);
                position += read;
            }
        }
        return offset + segment.byteSize();
    }

    @Override
    public void close() throws IOException {
        channel.close();
    }
}
