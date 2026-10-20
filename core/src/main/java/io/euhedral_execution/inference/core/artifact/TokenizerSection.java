package io.euhedral_execution.inference.core.artifact;

import io.euhedral_execution.inference.core.tokenizer.TokenizerFiles;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Optional;
import java.util.zip.CRC32;

/// The tokenizer an artifact carries: a section after the artifact's own content, found from a footer at the very
/// end of the file, so an artifact gains or replaces it without any other byte moving. Both container versions
/// carry it the same way. Big-endian:
///
/// ```
/// body:   int32 entryCount (4), then per entry: int32 nameLength, UTF-8 name, int32 dataLength, UTF-8 data,
///         in the order tokenizer.json, tokenizer_config.json, chat_template.jinja, special_tokens.json
/// footer: int64 bodyLength, int32 CRC-32 of the body, int32 version (1), 8 bytes "EDRLTOKN"
/// ```
///
/// `special_tokens.json` is [TokenizerFiles#specialTokens].
public final class TokenizerSection {

    /// The bytes of the footer.
    public static final int FOOTER_BYTES = 24;

    public static final int VERSION = 1;

    /// "EDRLTOKN" in ASCII.
    public static final long MAGIC = 0x4544524C544F4B4EL;

    /// The entries of the body, in their order.
    public static final List<String> ENTRIES =
            List.of("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens.json");

    private static final long MAX_BODY_BYTES = 1L << 30;
    private static final ByteOrder ORDER = ByteOrder.BIG_ENDIAN;

    private TokenizerSection() {}

    /// The section (body and footer) holding `files`.
    public static byte[] encode(TokenizerFiles files) {
        ByteArrayOutputStream body = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(body)) {
            List<String> data =
                    List.of(files.tokenizer(), files.tokenizerConfig(), files.chatTemplate(), files.specialTokens());
            out.writeInt(ENTRIES.size());
            for (int i = 0; i < ENTRIES.size(); i++) {
                writeText(out, ENTRIES.get(i));
                writeText(out, data.get(i));
            }
        } catch (IOException exception) {
            throw new AssertionError(exception);
        }
        byte[] bytes = body.toByteArray();
        ByteBuffer section = ByteBuffer.allocate(bytes.length + FOOTER_BYTES).order(ORDER);
        section.put(bytes);
        section.putLong(bytes.length);
        section.putInt(crc(bytes));
        section.putInt(VERSION);
        section.putLong(MAGIC);
        return section.array();
    }

    /// Where the content of the artifact at `path`, open as `channel`, ends: the start of its tokenizer section, or
    /// the end of the file when it has none.
    ///
    /// @throws ArtifactFormatException naming `path` when the footer is present but names a section the file cannot
    ///     hold
    public static long contentEnd(Path path, FileChannel channel) throws IOException {
        return footer(path, channel).map(Footer::sectionStart).orElse(channel.size());
    }

    /// The tokenizer files of the artifact at `path`, or empty when it carries none.
    ///
    /// @throws ArtifactFormatException naming `path` when the section is damaged
    public static Optional<TokenizerFiles> read(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            Optional<Footer> footer = footer(path, channel);
            if (footer.isEmpty()) return Optional.empty();
            ByteBuffer body = ByteBuffer.allocate((int) footer.get().bodyLength());
            ArtifactFileAccess.readFully(channel, footer.get().sectionStart(), body, "tokenizer section");
            if (crc(body.array()) != footer.get().crc()) throw damaged(path, "the tokenizer section fails its CRC-32");
            try {
                return Optional.of(decode(body.array()));
            } catch (ArtifactFormatException damaged) {
                throw damaged(path, damaged.getMessage());
            }
        }
    }

    /// The tokenizer files the artifact at `artifact` carries.
    ///
    /// @throws ArtifactFormatException naming the command that adds them when the artifact carries none
    public static TokenizerFiles require(Path artifact) throws IOException {
        return read(artifact)
                .orElseThrow(() -> new ArtifactFormatException(
                        artifact + " carries no tokenizer; add the checkpoint's with: " + addCommand(artifact)));
    }

    /// The command that adds the tokenizer of a checkpoint directory to the artifact at `artifact`.
    public static String addCommand(Path artifact) {
        return "./gradlew :quantization:run --args=\"package-tokenizer --artifact " + artifact
                + " --tokenizer CHECKPOINT_DIR\"";
    }

    /// Writes `files` as the section after the first `contentEnd` bytes of the file, dropping whatever followed
    /// them. The footer is written last, so an interrupted write leaves no section.
    public static void write(FileChannel channel, long contentEnd, TokenizerFiles files) throws IOException {
        byte[] section = encode(files);
        channel.truncate(contentEnd);
        int bodyLength = section.length - FOOTER_BYTES;
        writeFully(channel, ByteBuffer.wrap(section, 0, bodyLength), contentEnd);
        channel.force(true);
        writeFully(channel, ByteBuffer.wrap(section, bodyLength, FOOTER_BYTES), contentEnd + bodyLength);
        channel.force(true);
    }

    private record Footer(long sectionStart, long bodyLength, int crc) {}

    private static Optional<Footer> footer(Path path, FileChannel channel) throws IOException {
        long size = channel.size();
        if (size < FOOTER_BYTES) return Optional.empty();
        ByteBuffer footer = ByteBuffer.allocate(FOOTER_BYTES).order(ORDER);
        ArtifactFileAccess.readFully(channel, size - FOOTER_BYTES, footer, "tokenizer footer");
        footer.flip();
        long bodyLength = footer.getLong();
        int crc = footer.getInt();
        int version = footer.getInt();
        if (footer.getLong() != MAGIC) return Optional.empty();
        if (version != VERSION) throw damaged(path, "unsupported tokenizer section version " + version + REPAIR);
        if (bodyLength < 0 || bodyLength > MAX_BODY_BYTES || bodyLength > size - FOOTER_BYTES)
            throw damaged(path, "the tokenizer section's length is out of range: " + bodyLength + REPAIR);
        return Optional.of(new Footer(size - FOOTER_BYTES - bodyLength, bodyLength, crc));
    }

    private static final String REPAIR = "; running package-tokenizer on the artifact again replaces the section";

    private static ArtifactFormatException damaged(Path path, String message) {
        return new ArtifactFormatException(path + ": " + message);
    }

    private static TokenizerFiles decode(byte[] body) throws ArtifactFormatException {
        ByteBuffer in = ByteBuffer.wrap(body).order(ORDER);
        int count = intValue(in);
        if (count != ENTRIES.size())
            throw new ArtifactFormatException(
                    "the tokenizer section holds " + count + " entries, not " + ENTRIES.size());
        String[] data = new String[count];
        for (int i = 0; i < count; i++) {
            String name = text(in);
            if (!name.equals(ENTRIES.get(i)))
                throw new ArtifactFormatException(
                        "tokenizer section entry " + i + " is " + name + ", not " + ENTRIES.get(i));
            data[i] = text(in);
        }
        if (in.hasRemaining()) throw new ArtifactFormatException("the tokenizer section has trailing bytes");
        return new TokenizerFiles(data[0], data[1], data[2], data[3]);
    }

    private static int intValue(ByteBuffer in) throws ArtifactFormatException {
        if (in.remaining() < Integer.BYTES) throw new ArtifactFormatException("the tokenizer section is truncated");
        return in.getInt();
    }

    private static String text(ByteBuffer in) throws ArtifactFormatException {
        int length = intValue(in);
        if (length < 0 || length > in.remaining())
            throw new ArtifactFormatException("the tokenizer section is truncated");
        ByteBuffer bytes = in.slice(in.position(), length);
        in.position(in.position() + length);
        try {
            return StandardCharsets.UTF_8
                    .newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(bytes)
                    .toString();
        } catch (CharacterCodingException exception) {
            throw new ArtifactFormatException("the tokenizer section holds text that is not UTF-8");
        }
    }

    private static void writeText(DataOutputStream out, String text) throws IOException {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        out.writeInt(bytes.length);
        out.write(bytes);
    }

    private static int crc(byte[] bytes) {
        CRC32 crc = new CRC32();
        crc.update(bytes);
        return (int) crc.getValue();
    }

    private static void writeFully(FileChannel channel, ByteBuffer bytes, long offset) throws IOException {
        long at = offset;
        while (bytes.hasRemaining()) at += channel.write(bytes, at);
    }
}
