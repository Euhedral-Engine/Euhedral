package io.euhedral_execution.inference.core.runtime;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/// Prompt tokenization as frames equals the tokenizer's own encoding, whatever order and threads the frames
/// run on.
class PromptTokenizationTest {

    private static final Path TOKENIZER_DIRECTORY =
            Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));

    private static QwenTokenizer tokenizer;
    private static ExecutorService workers;

    @BeforeAll
    static void load() throws Exception {
        assumeTrue(Files.isRegularFile(TOKENIZER_DIRECTORY.resolve("tokenizer.json")), "no tokenizer assets");
        tokenizer = QwenTokenizer.load(TOKENIZER_DIRECTORY);
        workers = Executors.newFixedThreadPool(8);
    }

    @AfterAll
    static void stop() {
        if (workers != null) workers.shutdownNow();
    }

    private static List<String> texts() throws Exception {
        String document = Files.readString(Path.of("..", "docs", "FRAME_MODEL.md"));
        StringBuilder pieces = new StringBuilder();
        for (int index = 0; index < 3 * PromptTokenization.CHUNK_PRETOKENS; index++)
            pieces.append(" w").append(index);
        return List.of(
                "",
                "a",
                document,
                document.substring(0, 5000),
                // Control tokens between and inside chunks, as the chat template renders them.
                "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\n" + document.substring(0, 20000)
                        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                // NFC-normalized input (decomposed accents), CJK, emoji and digits.
                "Café été 你好世界 🚀🚀 12345678 ",
                pieces.toString(),
                pieces.substring(0, pieces.length() / 3));
    }

    /// Runs every published frame in a shuffled order on the calling thread, including frames they publish.
    private static int[] runShuffled(String text, boolean special, long seed) throws Exception {
        List<AbstractFrame> queue = new ArrayList<>();
        AtomicInteger terminations = new AtomicInteger();
        CompletableFuture<int[]> ids =
                PromptTokenization.start(tokenizer, text, special, queue::add, terminations::incrementAndGet);
        Random random = new Random(seed);
        while (!queue.isEmpty()) {
            Collections.shuffle(queue, random);
            AbstractFrame frame = queue.removeFirst();
            frame.execute();
            frame.doFinally();
        }
        assertEquals(1, terminations.get(), "the job terminates once");
        return ids.get();
    }

    @Test
    void framesInAnyOrderEncodeExactlyAsTheTokenizer() throws Exception {
        for (String text : texts()) {
            for (boolean special : new boolean[] {false, true}) {
                int[] expected = special ? tokenizer.encodeWithModelSpecialTokens(text) : tokenizer.encodeText(text);
                for (long seed = 0; seed < 4; seed++)
                    assertArrayEquals(
                            expected, runShuffled(text, special, seed), "text of " + text.length() + " chars");
            }
        }
    }

    @Test
    void framesOnConcurrentThreadsEncodeExactlyAsTheTokenizer() throws Exception {
        String text = texts().get(2);
        int[] expected = tokenizer.encodeWithModelSpecialTokens(text);
        for (int round = 0; round < 8; round++) {
            AtomicInteger frames = new AtomicInteger();
            CompletableFuture<int[]> ids = PromptTokenization.start(
                    tokenizer,
                    text,
                    true,
                    new java.util.function.Consumer<>() {
                        @Override
                        public void accept(AbstractFrame frame) {
                            frames.incrementAndGet();
                            workers.execute(() -> {
                                frame.execute();
                                frame.doFinally();
                            });
                        }
                    },
                    () -> {});
            assertArrayEquals(expected, ids.get(30, TimeUnit.SECONDS));
            assertTrue(frames.get() > 4, "a long prompt fans out: " + frames.get() + " frames");
        }
    }

    @Test
    void aRejectedFrameFailsTheJobOnce() throws Exception {
        String text = texts().get(2);
        List<AbstractFrame> queue = new ArrayList<>();
        AtomicInteger terminations = new AtomicInteger();
        CompletableFuture<int[]> ids =
                PromptTokenization.start(tokenizer, text, false, queue::add, terminations::incrementAndGet);
        AbstractFrame split = queue.removeFirst();
        split.execute();
        RuntimeException rejection = new RuntimeException("worker cache retired");
        queue.removeFirst().doFinallyWithError(rejection);
        while (!queue.isEmpty()) queue.removeFirst().execute();
        ExecutionException failure = assertThrows(ExecutionException.class, ids::get);
        assertSame(rejection, failure.getCause().getCause());
        assertEquals(1, terminations.get());
    }

    @Test
    void aPublisherThatThrowsFailsTheJob() {
        RuntimeException closed = new IllegalStateException("closed");
        CompletableFuture<int[]> ids = PromptTokenization.start(
                tokenizer,
                "text",
                false,
                frame -> {
                    throw closed;
                },
                () -> {});
        ExecutionException failure = assertThrows(ExecutionException.class, ids::get);
        assertSame(closed, failure.getCause());
    }
}
