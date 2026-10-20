package io.euhedral_execution.inference.core.tokenizer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class QwenTokenizerTest {
    private static QwenTokenizer tokenizer;

    @BeforeAll
    static void loadCheckpointTokenizer() throws Exception {
        Path checkpoint =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "unset-euhedral.qwen.tokenizer-dir"));
        assumeTrue(Files.isRegularFile(checkpoint.resolve("tokenizer.json")));
        tokenizer = QwenTokenizer.load(TokenizerFiles.fromCheckpoint(checkpoint));
    }

    @Test
    void matchesReferenceVectorsFromTheTargetCheckpoint() {
        assertArrayEquals(new int[] {9419, 11, 1814, 0}, tokenizer.encodeText("Hello, world!"));
        assertArrayEquals(new int[] {220, 11173, 256, 12258}, tokenizer.encodeText("  repeated   spaces"));
        assertArrayEquals(new int[] {1021, 799, 198, 1021, 1330, 198}, tokenizer.encodeText("line one\nline two\n"));
        assertArrayEquals(new int[] {895, 56868, 220, 99986, 87209}, tokenizer.encodeText("café 中文 😀"));
        assertArrayEquals(new int[] {32, 169171, 33}, tokenizer.encodeText("A🙂B"));
        assertArrayEquals(new int[0], tokenizer.encodeText(""));
        assertArrayEquals(new int[] {3267, 36328, 571, 3825}, tokenizer.encodeText("naïve é"));
        assertArrayEquals(new int[] {127, 123, 3966, 4326}, tokenizer.encodeText("ÿ \t\r\n"));
        assertArrayEquals(
                new int[] {64, 1076, 11726, 31921, 220, 16, 17, 18, 19, 20}, tokenizer.encodeText("a...!!!??? 12345"));
        assertArrayEquals(new int[] {14572, 914, 1165, 2224, 353, 3172}, tokenizer.encodeText("don't We're I'll"));
        assertArrayEquals(new int[] {174675, 30061, 171405, 211075}, tokenizer.encodeText("한국어 العربية русский"));
        assertArrayEquals(new int[] {248068, 79420, 248069}, tokenizer.encodeText("<think>thinking</think>"));
        assertArrayEquals(
                new int[] {248045, 846, 198, 12675, 248046}, tokenizer.encodeText("<|im_start|>user\nHi<|im_end|>"));
        assertArrayEquals(
                new int[] {760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13},
                tokenizer.encodeText("The quick brown fox jumps over the lazy dog."));
        assertEquals(" world", tokenizer.decode(new int[] {1814}));
    }

    @Test
    void exposesControlEosAndModelBosSeparately() {
        assertEquals(248045, tokenizer.specialTokenId("<|im_start|>").orElseThrow());
        assertEquals(248046, tokenizer.specialTokenId("<|im_end|>").orElseThrow());
        assertEquals(248068, tokenizer.controlTokenId("<think>").orElseThrow());
        assertTrue(tokenizer.specialTokenId("<think>").isEmpty());
        assertEquals(248046, tokenizer.eosTokenId());
        assertTrue(tokenizer.isEosToken(248046));
        assertFalse(tokenizer.isEosToken(248044));
        assertEquals(Set.of(248044, 248046), tokenizer.generationEosTokenIds());
        assertTrue(tokenizer.isGenerationEosToken(248044));
        assertTrue(tokenizer.isGenerationEosToken(248046));
        assertFalse(tokenizer.isGenerationEosToken(248045));
        assertEquals(248044, tokenizer.generationBosTokenId().orElseThrow());
        assertFalse(tokenizer.isEosToken(248045));
        assertEquals(TokenKind.EOS, tokenizer.tokenKind(248046));
        assertEquals(TokenKind.CONTROL, tokenizer.tokenKind(248045));
        assertEquals(TokenKind.CONTROL, tokenizer.tokenKind(248068));
        assertEquals(TokenKind.NORMAL, tokenizer.tokenKind(1814));
        assertFalse(tokenizer.bosTokenId().isPresent());
        assertArrayEquals(tokenizer.encodeText("hello"), tokenizer.encodeWithModelSpecialTokens("hello"));
        assertEquals("<|im_end|>", tokenizer.decode(new int[] {tokenizer.eosTokenId()}));
        assertEquals("<|endoftext|>", tokenizer.decode(new int[] {248044}));
    }

    @Test
    void decodesCheckpointTextIncludingUnicodeAndControlTokens() {
        int[] tokens = {248045, 846, 198, 12675, 248046};
        assertEquals("<|im_start|>user\nHi<|im_end|>", tokenizer.decode(tokens));
        assertEquals("Hello, world!", tokenizer.decode(tokenizer.encodeText("Hello, world!")));
        assertEquals("café 中文 😀", tokenizer.decode(tokenizer.encodeText("café 中文 😀")));
    }

    @Test
    void incrementalDecodeBuffersUtf8UntilCompleteAndMatchesEveryChunking() {
        int[] splitEmoji = {172, 253, 246, 222};
        IncrementalDecoder oneAtATime = tokenizer.newIncrementalDecoder();
        assertEquals("", oneAtATime.append(172));
        assertEquals("", oneAtATime.append(253));
        assertEquals("", oneAtATime.append(246));
        assertEquals("😀", oneAtATime.append(222));
        assertEquals("", oneAtATime.finish());

        var allAtOnce = tokenizer.newIncrementalDecoder();
        assertEquals("😀", allAtOnce.append(splitEmoji));
        assertEquals("", allAtOnce.finish());

        var arbitraryChunks = tokenizer.newIncrementalDecoder();
        List<String> emitted = new ArrayList<>();
        emitted.add(arbitraryChunks.append(new int[] {172, 253}));
        emitted.add(arbitraryChunks.append(new int[] {246}));
        emitted.add(arbitraryChunks.append(new int[] {222}));
        emitted.add(arbitraryChunks.finish());
        assertEquals("😀", String.join("", emitted));
        assertEquals(tokenizer.decode(splitEmoji), String.join("", emitted));
    }

    @Test
    void incrementalDecodePreservesTextAroundUtf8BoundariesAndEos() {
        int[] tokens = {9419, 11, 1814, 0, 248046};
        String expected = tokenizer.decode(tokens);
        var decoder = tokenizer.newIncrementalDecoder();
        StringBuilder actual = new StringBuilder();
        for (int token : tokens) actual.append(decoder.append(token));
        actual.append(decoder.finish());
        assertEquals(expected, actual.toString());
        assertTrue(expected.endsWith("<|im_end|>"));
    }

    @Test
    void incrementalDecodeHandlesTruncatedInvalidAndControlSeparatedUtf8() {
        assertIncrementalPathsMatch(new int[] {172}, "�");
        assertIncrementalPathsMatch(new int[] {172, 32}, "�A");
        assertIncrementalPathsMatch(new int[] {172, 248046}, "�<|im_end|>");

        var decoder = tokenizer.newIncrementalDecoder();
        assertEquals("", decoder.append(172));
        assertEquals("�<|im_end|>", decoder.append(248046));
        assertEquals("", decoder.finish());
    }

    @Test
    void incrementalDecoderRejectsEmptyChunksAfterFinish() {
        IncrementalDecoder decoder = tokenizer.newIncrementalDecoder();
        decoder.finish();
        assertThrows(IllegalStateException.class, () -> decoder.append(new int[0]));
    }

    private void assertIncrementalPathsMatch(int[] tokenIds, String expected) {
        assertEquals(expected, tokenizer.decode(tokenIds));

        var oneAtATime = tokenizer.newIncrementalDecoder();
        StringBuilder streamed = new StringBuilder();
        for (int tokenId : tokenIds) streamed.append(oneAtATime.append(tokenId));
        streamed.append(oneAtATime.finish());
        assertEquals(expected, streamed.toString());

        var arbitraryChunks = tokenizer.newIncrementalDecoder();
        StringBuilder chunked = new StringBuilder();
        for (int start = 0; start < tokenIds.length; start += 2) {
            chunked.append(
                    arbitraryChunks.append(Arrays.copyOfRange(tokenIds, start, Math.min(start + 2, tokenIds.length))));
        }
        chunked.append(arbitraryChunks.finish());
        assertEquals(expected, chunked.toString());
    }
}
