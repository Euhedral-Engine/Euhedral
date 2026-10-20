package io.euhedral_execution.inference.quantization.mtp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/// The MTP drafter's training corpus, a directory of four append-only files:
///
/// - `tokens.bin`: every sequence's tokens (int32, little endian), the prompt then the answer, the last token the one
///   the base model chose but never ran;
/// - `labels.bin`: for each token of `tokens.bin`, the base model's greedy choice at its position (the token itself
///   in the prompt): what a draft of that position should be;
/// - `hidden.bin`: the post-final-norm hidden (BF16 [hidden]) of every row the base model ran: all of a sequence's
///   tokens but the last;
/// - `index.jsonl`: one line per sequence, written after its rows: its id, first token and first hidden row, token
///   count, prompt length and whether its answer was sampled (otherwise greedy, its labels its tokens).
///
/// A corpus written before labels were kept is greedy throughout: opening it for writing makes its labels its
/// tokens.
public final class Corpus {

    /// One sequence of the corpus.
    public record Entry(String id, long token, long row, int length, int prompt, boolean sampled) {}

    private Corpus() {}

    /// The index of the corpus in `directory`.
    public static List<Entry> entries(Path directory) throws IOException {
        Path index = directory.resolve("index.jsonl");
        List<Entry> entries = new ArrayList<>();
        if (!Files.exists(index)) return entries;
        ObjectMapper json = new ObjectMapper();
        for (String line : Files.readAllLines(index)) {
            if (line.isBlank()) continue;
            JsonNode node = json.readTree(line);
            entries.add(new Entry(
                    node.path("id").asText(),
                    node.path("token").asLong(),
                    node.path("row").asLong(),
                    node.path("length").asInt(),
                    node.path("prompt").asInt(),
                    node.path("sampled").asBoolean(false)));
        }
        return entries;
    }

    /// Appends sequences to a corpus, from one owner; a corpus left with rows beyond its index (a stopped run) is cut
    /// back to it.
    public static final class Writer implements AutoCloseable {
        private final FileChannel tokens, labels, hidden, index;
        private final int width;
        private final Set<String> ids = new HashSet<>();
        private long tokenCount, rowCount;

        private Writer(Path directory, int width) throws IOException {
            this.width = width;
            for (Entry entry : entries(directory)) {
                ids.add(entry.id());
                tokenCount = Math.max(tokenCount, entry.token() + entry.length());
                rowCount = Math.max(rowCount, entry.row() + entry.length() - 1);
            }
            var options = new StandardOpenOption[] {
                StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.READ
            };
            tokens = FileChannel.open(directory.resolve("tokens.bin"), options);
            labels = FileChannel.open(directory.resolve("labels.bin"), options);
            hidden = FileChannel.open(directory.resolve("hidden.bin"), options);
            index = FileChannel.open(
                    directory.resolve("index.jsonl"),
                    StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE,
                    StandardOpenOption.APPEND);
            tokens.truncate(4 * tokenCount);
            labels.truncate(4 * tokenCount);
            // Sequences written before labels were kept were answered greedily: their labels are their tokens.
            labels.position(labels.size());
            for (long at = labels.size(); at < 4 * tokenCount; )
                at += tokens.transferTo(at, 4 * tokenCount - at, labels);
            if (width > 0) hidden.truncate(2L * width * rowCount);
        }

        /// Opens the corpus in `directory` for rows of `width` hidden values (0: only to read its ids).
        public static Writer open(Path directory, int width) throws IOException {
            return new Writer(directory, width);
        }

        /// The ids already in the corpus.
        public Set<String> ids() {
            return Set.copyOf(ids);
        }

        /// Appends a sequence: `length` tokens of which `prompt` are the prompt, their labels, whether its answer was
        /// sampled, and `rows` hidden rows from `source`.
        public void write(
                String id,
                int[] values,
                int[] labelValues,
                int length,
                int prompt,
                boolean sampled,
                MemorySegment source,
                int rows) {
            if (rows != length - 1) throw new IllegalArgumentException("a sequence keeps a hidden row per token run");
            try {
                ByteBuffer buffer = ByteBuffer.allocate(4 * length).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < length; i++) buffer.putInt(values[i]);
                writeFully(tokens, buffer.flip(), 4 * tokenCount);
                buffer.clear();
                for (int i = 0; i < length; i++) buffer.putInt(labelValues[i]);
                writeFully(labels, buffer.flip(), 4 * tokenCount);
                writeFully(hidden, source.asSlice(0, 2L * width * rows).asByteBuffer(), 2L * width * rowCount);
                tokens.force(false);
                labels.force(false);
                hidden.force(false);
                String line = String.format(
                        "{\"id\":\"%s\",\"token\":%d,\"row\":%d,\"length\":%d,\"prompt\":%d,\"sampled\":%b}%n",
                        id.replace("\\", "\\\\").replace("\"", "\\\""), tokenCount, rowCount, length, prompt, sampled);
                index.write(ByteBuffer.wrap(line.getBytes(StandardCharsets.UTF_8)));
                index.force(false);
                ids.add(id);
                tokenCount += length;
                rowCount += rows;
            } catch (IOException failure) {
                throw new UncheckedIOException(failure);
            }
        }

        private static void writeFully(FileChannel channel, ByteBuffer buffer, long position) throws IOException {
            while (buffer.hasRemaining()) position += channel.write(buffer, position);
        }

        @Override
        public void close() throws IOException {
            tokens.close();
            labels.close();
            hidden.close();
            index.close();
        }
    }
}
