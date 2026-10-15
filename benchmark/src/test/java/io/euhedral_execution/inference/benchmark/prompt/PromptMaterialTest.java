package io.euhedral_execution.inference.benchmark.prompt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class PromptMaterialTest {
    private static final Path TOKENIZER =
            Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "unset-euhedral.qwen.tokenizer-dir"));

    /// One token per word plus one per trailing punctuation mark.
    private static int count(String text) {
        int punctuation = text.endsWith(".") ? 1 : 0;
        return text.split(" ").length + punctuation;
    }

    @Test
    void reachesTheTargetDeterministically() {
        var first = PromptMaterial.build(100, 5L, PromptMaterialTest::count);
        var second = PromptMaterial.build(100, 5L, PromptMaterialTest::count);
        assertEquals(first, second);
        assertEquals(100, first.actualTokens());
        assertEquals(PromptMaterial.sha256(first.text()), first.sha256());
        assertNotEquals(
                first.sha256(),
                PromptMaterial.build(100, 6L, PromptMaterialTest::count).sha256());
    }

    @Test
    void reportsTheActualCountWhenTheTargetCannotBeHitExactly() {
        // Every word costs two tokens, so an odd target is unreachable and must not be claimed.
        var prompt = PromptMaterial.build(7, 1L, text -> text.split(" ").length * 2);
        assertEquals(7, prompt.targetTokens());
        assertEquals(6, prompt.actualTokens());
        assertThrows(IllegalArgumentException.class, () -> PromptMaterial.build(1, 1L, text -> 2));
    }

    @Test
    void matchesTheQwenTokenizerCountUsedByTheSession() throws Exception {
        assumeTrue(Files.isRegularFile(TOKENIZER.resolve("tokenizer.json")), "Qwen tokenizer assets unavailable");
        QwenTokenizer tokenizer = QwenTokenizer.load(TOKENIZER);
        for (int target : new int[] {32, 256, 1024}) {
            var prompt = PromptMaterial.build(
                    target, PromptMaterial.DEFAULT_SEED, text -> tokenizer.encodeWithModelSpecialTokens(text).length);
            assertEquals(target, prompt.actualTokens(), "target " + target);
            assertEquals(target, tokenizer.encodeWithModelSpecialTokens(prompt.text()).length);
        }
    }
}
