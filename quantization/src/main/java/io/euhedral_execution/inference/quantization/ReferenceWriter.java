package io.euhedral_execution.inference.quantization;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;

/// A llama.cpp logits file (`llama-perplexity --kl-divergence-base`) written by the engine, in the layout
/// [Scorer.Reference] reads: "_logits_", int32 context, vocabulary and chunk count, every chunk's tokens, then per
/// chunk the records of its second half (`context - 1 - context / 2` of them), each `width` uint16: scale and
/// minimum as FP32, a code per vocabulary entry, padded to an even count.
///
/// A record is llama.cpp's `log_softmax` quantization of one row of logits (`tools/perplexity/perplexity.cpp`):
/// the logits are clamped below at `low = max(min, max - 16)`; with `lse = ln(sum exp(logit - max))` the record holds
/// `scale = (max - low) / 65535`, `minimum = low - max - lse` and `code[i] = rint((max(logit[i], low) - low) / scale)`.
/// The inverse is `lp[i] = scale * code[i] + minimum` in float64, which `score.cu` applies ([#decode] is its host
/// twin); clamped entries decode to `minimum`, at most -16, and `score.cu` leaves them out of the KL sum. Records of
/// files llama.cpp wrote all span exactly 16 (`scale * 65535`) with their minimum at `-16 - lse`.
///
/// The file is written as `OUT.partial` of the final size plus an 8-byte count of the chunks complete, past the
/// end. A writer opened on an existing partial file continues after the last complete chunk, the count and the
/// rows overwritten in place; [#finish] cuts the count off and renames the file to `OUT`.
final class ReferenceWriter implements AutoCloseable {

    private static final byte[] MAGIC = "_logits_".getBytes(StandardCharsets.US_ASCII);
    private static final float FLOOR = -16f;
    private static final int HEADER = 20;

    private final Path out, partial;
    private final FileChannel channel;
    private final int context, vocabulary, chunks, records, width;
    private final long dataOffset, dataBytes;
    private int completed;

    private ReferenceWriter(
            Path out, Path partial, FileChannel channel, int context, int vocabulary, int chunks, int completed) {
        this.out = out;
        this.partial = partial;
        this.channel = channel;
        this.context = context;
        this.vocabulary = vocabulary;
        this.chunks = chunks;
        this.records = records(context);
        this.width = width(vocabulary);
        this.dataOffset = HEADER + 4L * chunks * context;
        this.dataBytes = 2L * width * records * chunks;
        this.completed = completed;
    }

    /// The records of a chunk of `context` tokens: the second half's, as [Scorer.Reference] counts them.
    static int records(int context) {
        return context - 1 - context / 2;
    }

    /// The uint16 values of one record.
    static int width(int vocabulary) {
        return 2 * ((vocabulary + 1) / 2) + 4;
    }

    /// Starts `out`'s partial file for `tokens` (a chunk each, `context` tokens long), or continues the one a
    /// stopped run left.
    static ReferenceWriter open(Path out, int context, int vocabulary, int[][] tokens) throws IOException {
        if (Files.exists(out)) throw new IllegalStateException(out + " exists; remove it to write it again");
        Path partial = out.resolveSibling(out.getFileName() + ".partial");
        int chunks = tokens.length;
        ByteBuffer head = ByteBuffer.allocate(HEADER + 4 * chunks * context).order(ByteOrder.LITTLE_ENDIAN);
        head.put(MAGIC).putInt(context).putInt(vocabulary).putInt(chunks);
        for (int[] chunk : tokens) {
            if (chunk.length != context) throw new IllegalArgumentException("a chunk is not " + context + " tokens");
            for (int token : chunk) head.putInt(token);
        }
        head.flip();
        boolean resume = Files.exists(partial);
        FileChannel channel =
                FileChannel.open(partial, StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE);
        try {
            int completed = 0;
            var writer = new ReferenceWriter(out, partial, channel, context, vocabulary, chunks, 0);
            long size = writer.dataOffset + writer.dataBytes + Long.BYTES;
            if (resume) {
                ByteBuffer found = ByteBuffer.allocate(head.remaining());
                readFully(channel, found, 0);
                if (channel.size() != size || !found.flip().equals(head))
                    throw new IllegalStateException(partial + " was started for other tokens, another context or "
                            + "vocabulary; remove it to start again");
                ByteBuffer count = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
                readFully(channel, count, size - Long.BYTES);
                long stored = count.flip().getLong();
                if (stored < 0 || stored > chunks)
                    throw new IllegalStateException(partial + " counts " + stored + " complete chunks of " + chunks);
                completed = (int) stored;
            } else {
                writeFully(channel, head, 0);
                writeFully(channel, ByteBuffer.allocate(Long.BYTES), size - Long.BYTES);
            }
            writer.completed = completed;
            return writer;
        } catch (IOException | RuntimeException failure) {
            channel.close();
            throw failure;
        }
    }

    /// Chunks complete: the next to write is this one.
    int completed() {
        return completed;
    }

    int records() {
        return records;
    }

    /// Writes `record` (as [#encode] makes it) as record `index` of `chunk`.
    void write(int chunk, int index, byte[] record) throws IOException {
        if (chunk < 0 || chunk >= chunks || index < 0 || index >= records || record.length != 2 * width)
            throw new IllegalArgumentException("record " + index + " of chunk " + chunk + " does not fit");
        writeFully(channel, ByteBuffer.wrap(record), dataOffset + 2L * width * ((long) chunk * records + index));
    }

    /// Records that chunks `0` to `count - 1` are all written; durable before this returns.
    void complete(int count) throws IOException {
        channel.force(false);
        ByteBuffer stored = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        stored.putLong(count).flip();
        writeFully(channel, stored, dataOffset + dataBytes);
        channel.force(false);
        completed = count;
    }

    /// Cuts the count off the partial file and renames it to the reference.
    void finish() throws IOException {
        if (completed != chunks) throw new IllegalStateException(completed + " of " + chunks + " chunks written");
        channel.truncate(dataOffset + dataBytes);
        channel.force(true);
        channel.close();
        Files.move(partial, out, StandardCopyOption.ATOMIC_MOVE);
    }

    @Override
    public void close() throws IOException {
        channel.close();
    }

    /// llama.cpp's record of one row of logits.
    static byte[] encode(float[] logits) {
        int n = logits.length;
        float max = logits[0], min = logits[0];
        for (float value : logits) {
            max = Math.max(max, value);
            min = Math.min(min, value);
        }
        float low = Math.max(min, max + FLOOR);
        double sum = 0;
        for (float value : logits) sum += Math.exp(value - max);
        float logSumExp = (float) Math.log(sum);
        float scale = (max - low) / 65535f;
        float minimum = low - max - logSumExp;
        ByteBuffer record = ByteBuffer.allocate(2 * width(n)).order(ByteOrder.LITTLE_ENDIAN);
        record.putFloat(scale).putFloat(minimum);
        float inverse = scale > 0 ? 1 / scale : 0;
        for (float value : logits) {
            int code = (int) Math.rint(inverse * (Math.max(value, low) - low));
            record.putShort((short) Math.min(65535, Math.max(0, code)));
        }
        return record.array();
    }

    /// The log-probabilities a record stands for, as `score.cu` reads them: `scale * code + minimum`.
    static double[] decode(byte[] record, int vocabulary) {
        ByteBuffer buffer = ByteBuffer.wrap(record).order(ByteOrder.LITTLE_ENDIAN);
        double scale = buffer.getFloat(0), minimum = buffer.getFloat(4);
        double[] logProbabilities = new double[vocabulary];
        for (int i = 0; i < vocabulary; i++)
            logProbabilities[i] = scale * (buffer.getShort(8 + 2 * i) & 0xffff) + minimum;
        return logProbabilities;
    }

    private static void readFully(FileChannel channel, ByteBuffer target, long position) throws IOException {
        while (target.hasRemaining()) {
            int read = channel.read(target, position + target.position());
            if (read < 0) throw new IOException("truncated");
        }
    }

    private static void writeFully(FileChannel channel, ByteBuffer source, long position) throws IOException {
        long at = position;
        while (source.hasRemaining()) at += channel.write(source, at);
    }
}
