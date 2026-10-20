package io.euhedral_execution.inference.core.artifact;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.tokenizer.TokenizerFiles;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TokenizerSectionTest {

    /// The files whose encoding `tools/test_tokenizer_section.py` checks against the same bytes.
    static final TokenizerFiles FILES =
            new TokenizerFiles("{\"model\":{}}", "{}", "{{ \"é\" }}", "{\"eos_token_id\":[2]}");

    static final String ENCODED = "000000040000000e746f6b656e697a65722e6a736f6e0000000c7b226d6f64656c223a7b7d7d"
            + "00000015746f6b656e697a65725f636f6e6669672e6a736f6e000000027b7d00000013636861745f74656d706c6174652e"
            + "6a696e6a610000000a7b7b2022c3a922207d7d000000137370656369616c5f746f6b656e732e6a736f6e000000147b2265"
            + "6f735f746f6b656e5f6964223a5b325d7d00000000000000998fa57935000000014544524c544f4b4e";

    @TempDir
    Path directory;

    @Test
    void encodesTheDocumentedLayout() {
        assertEquals(ENCODED, HexFormat.of().formatHex(TokenizerSection.encode(FILES)));
    }

    @Test
    void writesAfterTheContentAndReadsBack() throws IOException {
        byte[] content = content(5000);
        Path path = Files.write(directory.resolve("a.edrl"), content);
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            TokenizerSection.write(channel, content.length, FILES);
            assertEquals(content.length, TokenizerSection.contentEnd(path, channel));
        }
        byte[] file = Files.readAllBytes(path);
        assertArrayEquals(content, Arrays.copyOf(file, content.length));
        assertArrayEquals(TokenizerSection.encode(FILES), Arrays.copyOfRange(file, content.length, file.length));
        assertEquals(Optional.of(FILES), TokenizerSection.read(path));
    }

    @Test
    void replacingKeepsTheContent() throws IOException {
        byte[] content = content(777);
        Path path = Files.write(directory.resolve("a.edrl"), content);
        TokenizerFiles other = new TokenizerFiles("{}", "{}", "template", "{}");
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            TokenizerSection.write(channel, content.length, FILES);
            TokenizerSection.write(channel, TokenizerSection.contentEnd(path, channel), other);
        }
        byte[] file = Files.readAllBytes(path);
        assertEquals(content.length + TokenizerSection.encode(other).length, file.length);
        assertArrayEquals(content, Arrays.copyOf(file, content.length));
        assertEquals(Optional.of(other), TokenizerSection.read(path));
    }

    @Test
    void aFileWithoutTheSectionHasNone() throws IOException {
        Path path = Files.write(directory.resolve("a.edrl"), content(100));
        assertEquals(Optional.empty(), TokenizerSection.read(path));
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            assertEquals(100, TokenizerSection.contentEnd(path, channel));
        }
    }

    @Test
    void aDamagedSectionIsRejected() throws IOException {
        Path path = Files.write(directory.resolve("a.edrl"), content(64));
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            TokenizerSection.write(channel, 64, FILES);
            channel.write(ByteBuffer.wrap(new byte[] {'X'}), 64 + 30);
        }
        ArtifactFormatException failure =
                assertThrows(ArtifactFormatException.class, () -> TokenizerSection.read(path));
        assertTrue(failure.getMessage().contains("CRC-32"), failure.getMessage());
        assertTrue(failure.getMessage().startsWith(path + ": "), failure.getMessage());
    }

    private static byte[] content(int size) {
        byte[] bytes = new byte[size];
        for (int i = 0; i < size; i++) bytes[i] = (byte) (i * 31 + 7);
        return bytes;
    }
}
