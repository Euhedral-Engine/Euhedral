package io.euhedral_execution.inference.core.guidance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.core.tokenizer.ReasoningConstraint;
import org.junit.jupiter.api.Test;

class ReasoningConstraintTest {
    private static final String ANSWER =
            "start: GAP? answer\nGAP: /[ \\t\\r\\n]{1,8}/\nanswer: %json {\"type\":\"object\"}";

    @Test
    void reasoningIsFreeThenTheAnswerFollowsItsGrammar() throws Exception {
        QwenTokenizer tokenizer = GuidanceFixtures.tokenizer();
        int thinkEnd = tokenizer.controlTokenId("</think>").orElseThrow();
        try (var constraint = new ReasoningConstraint(
                tokenizer, Integer.MAX_VALUE, GuidanceFixtures.compiler().constraint(ANSWER))) {
            String reasoning = "I should answer </tool_call> {not json";
            accept(constraint, tokenizer, reasoning);
            assertFalse(constraint.allows(tokenizer.eosTokenId()), "no terminator before an answer exists");
            float[] row = new float[GuidanceFixtures.VOCABULARY];
            constraint.maskDisallowed(row);
            assertEquals(Float.NEGATIVE_INFINITY, row[tokenizer.eosTokenId()]);
            assertEquals(0.0f, row[thinkEnd]);
            assertTrue(constraint.allows(thinkEnd));
            constraint.accept(thinkEnd);
            assertFalse(constraint.allows(tokenizer.encodeText("Sure")[0]), "the answer follows its grammar");
            accept(constraint, tokenizer, "\n\n{\"a\": 1}");
            assertTrue(constraint.allows(tokenizer.eosTokenId()));
            assertEquals(tokenizer.encodeText(reasoning).length, constraint.reasoningTokens());
        }
    }

    @Test
    void theBudgetForcesTheEndOfReasoning() throws Exception {
        QwenTokenizer tokenizer = GuidanceFixtures.tokenizer();
        int thinkEnd = tokenizer.controlTokenId("</think>").orElseThrow();
        var constraint = new ReasoningConstraint(tokenizer, 3, null);
        int[] words = tokenizer.encodeText(" one two three four");
        for (int index = 0; index < 3; index++) {
            assertTrue(constraint.allows(words[index]));
            constraint.accept(words[index]);
        }
        assertFalse(constraint.allows(words[3]));
        assertFalse(constraint.allows(tokenizer.eosTokenId()));
        assertThrows(IllegalArgumentException.class, () -> constraint.accept(words[3]));
        float[] row = new float[GuidanceFixtures.VOCABULARY];
        constraint.maskDisallowed(row);
        for (int id = 0; id < row.length; id++) assertEquals(id == thinkEnd ? 0.0f : Float.NEGATIVE_INFINITY, row[id]);
        constraint.accept(thinkEnd);
        assertTrue(constraint.allows(words[3]), "the answer is free without a grammar");
        assertTrue(constraint.allows(tokenizer.eosTokenId()));
    }

    @Test
    void withoutAnAnswerGrammarATerminatorMayEndTheReasoning() throws Exception {
        QwenTokenizer tokenizer = GuidanceFixtures.tokenizer();
        assertTrue(new ReasoningConstraint(tokenizer, 10, null).allows(tokenizer.eosTokenId()));
    }

    private static void accept(ReasoningConstraint constraint, QwenTokenizer tokenizer, String text) {
        for (int id : tokenizer.encodeText(text)) {
            assertTrue(constraint.allows(id), "token " + id + " rejected");
            constraint.accept(id);
        }
    }
}
