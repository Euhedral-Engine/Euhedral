package io.euhedral_execution.inference.core.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class ReasoningConstraintTest {
    private static QwenTokenizer tokenizer;
    private static int thinkEnd;

    @BeforeAll
    static void loadCheckpointTokenizer() throws Exception {
        Path checkpoint =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(checkpoint.resolve("tokenizer.json")));
        tokenizer = QwenTokenizer.load(checkpoint);
        thinkEnd = tokenizer.controlTokenId("</think>").orElseThrow();
    }

    @Test
    void reasoningIsFreeThenTheAnswerFollowsItsGrammarAfterABlankLine() {
        var constraint = new ReasoningConstraint(
                tokenizer, Integer.MAX_VALUE, new JsonEnvelopeConstraint(tokenizer, List.of("read_file"), true));
        accept(constraint, "I should read the file </tool_call> {not json");
        assertFalse(constraint.allows(tokenizer.eosTokenId()), "a terminator cannot end reasoning before an answer");
        accept(constraint, thinkEnd);
        assertFalse(constraint.allows(tokenizer.encodeText("Sure")[0]), "the answer follows its grammar");
        accept(constraint, "\n\n{\"tool_calls\":[{\"name\":\"read_file\",\"arguments\":{}}]}");
        assertTrue(constraint.allows(tokenizer.eosTokenId()));
        assertEquals(
                tokenizer.encodeText("I should read the file </tool_call> {not json").length,
                constraint.reasoningTokens());
    }

    @Test
    void theGapBeforeTheAnswerGrammarIsBounded() {
        var constraint = new ReasoningConstraint(
                tokenizer, Integer.MAX_VALUE, new JsonEnvelopeConstraint(tokenizer, List.of("f"), true));
        accept(constraint, thinkEnd);
        accept(constraint, " ".repeat(ReasoningConstraint.MAX_GAP_BYTES));
        assertFalse(constraint.allows(tokenizer.encodeText(" ")[0]));
        assertTrue(constraint.allows(tokenizer.encodeText("{")[0]));
    }

    @Test
    void theBudgetForcesTheEndOfReasoning() {
        var constraint = new ReasoningConstraint(tokenizer, 3, null);
        int[] words = tokenizer.encodeText(" one two three four");
        for (int index = 0; index < 3; index++) accept(constraint, words[index]);
        assertFalse(constraint.allows(words[3]));
        assertFalse(constraint.allows(tokenizer.eosTokenId()));
        assertThrows(IllegalArgumentException.class, () -> constraint.accept(words[3]));
        accept(constraint, thinkEnd);
        assertTrue(constraint.allows(words[3]), "the answer is free without a grammar");
        assertTrue(constraint.allows(tokenizer.eosTokenId()));
    }

    @Test
    void withoutAnAnswerGrammarATerminatorMayEndTheReasoning() {
        var constraint = new ReasoningConstraint(tokenizer, 10, null);
        assertTrue(constraint.allows(tokenizer.eosTokenId()));
    }

    private static void accept(ReasoningConstraint constraint, String text) {
        for (int id : tokenizer.encodeText(text)) accept(constraint, id);
    }

    private static void accept(ReasoningConstraint constraint, int id) {
        assertTrue(constraint.allows(id), "token " + id + " rejected");
        constraint.accept(id);
    }
}
