package io.euhedral_execution.inference.benchmark.prompt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ChatPromptCorpusTest {

    /// One token per whitespace-separated word: enough to exercise sizing deterministically.
    private static int count(String text) {
        return text.split("\\s+").length;
    }

    @Test
    void largeTargetsQuoteTheDocumentWithinTheTargetAndShortTargetsUseStandaloneTasks() {
        var large = ChatPromptCorpus.build(2000, ChatPromptCorpusTest::count);
        assertEquals(ChatPromptCorpus.size(), large.size());
        for (var prompt : large) {
            assertTrue(prompt.actualTokens() <= 2000);
            assertTrue(prompt.actualTokens() > 1500, "fills most of the target: " + prompt.actualTokens());
            assertTrue(prompt.text().startsWith("<|im_start|>user\nHere is a document:"));
            assertTrue(prompt.text().endsWith("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
            assertEquals(ChatPromptCorpus.GENERATOR, prompt.generator());
            assertEquals(PromptMaterial.sha256(prompt.text()), prompt.sha256());
        }
        var small = ChatPromptCorpus.build(40, ChatPromptCorpusTest::count);
        for (var prompt : small) assertFalse(prompt.text().contains("Here is a document"));
        assertEquals(large, ChatPromptCorpus.build(2000, ChatPromptCorpusTest::count), "deterministic");
        assertThrows(IllegalArgumentException.class, () -> ChatPromptCorpus.build(5, ChatPromptCorpusTest::count));
    }

    @Test
    void crlfCheckoutsBuildTheSamePrompts() {
        String document = ChatPromptCorpus.document().replace("\r\n", "\n");
        assertEquals(
                ChatPromptCorpus.build(2000, ChatPromptCorpusTest::count, document),
                ChatPromptCorpus.build(2000, ChatPromptCorpusTest::count, document.replace("\n", "\r\n")));
        assertEquals(
                ChatPromptCorpus.build(2000, ChatPromptCorpusTest::count, document),
                ChatPromptCorpus.build(2000, ChatPromptCorpusTest::count));
    }
}
