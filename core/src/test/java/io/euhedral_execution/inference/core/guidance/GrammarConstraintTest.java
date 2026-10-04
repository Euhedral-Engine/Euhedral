package io.euhedral_execution.inference.core.guidance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

/// llguidance as the engine uses it: the checkpoint vocabulary, its control tokens marked special, its
/// terminators as grammar ends.
class GrammarConstraintTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final String PERSON = """
            {"type":"object","properties":{
              "name":{"type":"string","minLength":1,"maxLength":12},
              "age":{"type":"integer","minimum":0,"maximum":150},
              "role":{"enum":["admin","user"]},
              "tags":{"type":"array","items":{"type":"string","maxLength":5},"maxItems":3},
              "email":{"anyOf":[{"type":"string","pattern":"^[a-z]+@[a-z]+\\\\.com$"},{"type":"null"}]}},
             "required":["name","age","role"],"additionalProperties":false}""";

    /// Whatever tokens a generation picks among those the mask allows, a finished text satisfies the schema.
    @Test
    void randomWalksThroughTheMaskOnlyFinishValidDocuments() throws Exception {
        // A uniform walk almost never leaves a string, so walks prefer short punctuation and digit tokens while still
        // taking arbitrary allowed tokens three times in ten.
        QwenTokenizer tokenizer = GuidanceFixtures.tokenizer();
        GrammarCompiler compiler = GuidanceFixtures.compiler();
        var random = new SplittableRandom(7);
        int finished = 0;
        for (int walk = 0; walk < 40; walk++) {
            try (GrammarConstraint constraint = compiler.constraint("start: %json " + PERSON)) {
                List<Integer> chosen = new ArrayList<>();
                for (int step = 0; step < 200; step++) {
                    List<Integer> allowed = allowedTokens(constraint);
                    assertFalse(allowed.isEmpty(), "a live grammar always allows a token");
                    boolean canEnd = allowed.contains(tokenizer.eosTokenId());
                    if (canEnd && random.nextInt(3) == 0) break;
                    List<Integer> structural = allowed.stream()
                            .filter(id -> isStructural(tokenizer, id))
                            .toList();
                    int choice = !structural.isEmpty() && random.nextInt(10) < 7
                            ? structural.get(random.nextInt(structural.size()))
                            : allowed.get(random.nextInt(allowed.size()));
                    if (tokenizer.isGenerationEosToken(choice)) break;
                    constraint.accept(choice);
                    chosen.add(choice);
                }
                if (!constraint.complete()) continue;
                finished++;
                String text = tokenizer.decode(
                        chosen.stream().mapToInt(Integer::intValue).toArray());
                assertPerson(JSON.readTree(text), text);
            }
        }
        assertTrue(finished > 10, "too few walks finished: " + finished);
    }

    @Test
    void theBulkMaskAgreesWithTokenQueriesAndNeverAllowsControlTokens() throws Exception {
        QwenTokenizer tokenizer = GuidanceFixtures.tokenizer();
        try (GrammarConstraint constraint =
                GuidanceFixtures.compiler().constraint("start: %json {\"type\":\"string\"}")) {
            for (int id : tokenizer.encodeText("\"inside")) constraint.accept(id);
            float[] row = new float[GuidanceFixtures.VOCABULARY];
            constraint.maskDisallowed(row);
            for (int id = 0; id < row.length; id++) assertEquals(constraint.allows(id), row[id] == 0.0f, "token " + id);
            for (String control : List.of("</think>", "<think>", "<tool_call>", "<|im_start|>"))
                assertFalse(constraint.allows(tokenizer.controlTokenId(control).orElseThrow()), control);
            assertFalse(constraint.allows(tokenizer.eosTokenId()), "the string is unfinished");
        }
    }

    @Test
    void anEndTokenIsAllowedOnlyOnceTheDocumentIsWhole() throws Exception {
        QwenTokenizer tokenizer = GuidanceFixtures.tokenizer();
        try (GrammarConstraint constraint = GuidanceFixtures.compiler().constraint("start: %json " + PERSON)) {
            for (int id : tokenizer.encodeText("{\"name\":\"Ann\",\"age\":42,\"role\":\"user\"")) {
                assertTrue(constraint.allows(id));
                constraint.accept(id);
            }
            assertFalse(constraint.complete());
            for (int end : tokenizer.generationEosTokenIds()) assertFalse(constraint.allows(end));
            constraint.accept(tokenizer.encodeText("}")[0]);
            assertTrue(constraint.complete());
            for (int end : tokenizer.generationEosTokenIds()) assertTrue(constraint.allows(end));
            assertFalse(constraint.allows(tokenizer.encodeText(" ")[0]), "nothing follows the document");
            // Generation commits the terminator it selected; the finished grammar takes it without consuming.
            constraint.accept(tokenizer.eosTokenId());
        }
    }

    @Test
    void schemaViolationsCannotBeGenerated() throws Exception {
        QwenTokenizer tokenizer = GuidanceFixtures.tokenizer();
        GrammarCompiler compiler = GuidanceFixtures.compiler();
        for (String invalid : List.of(
                "{\"name\":\"Ann\",\"age\":151",
                "{\"name\":\"Ann\",\"age\":-1",
                "{\"name\":\"Ann\",\"age\":4.5",
                "{\"name\":\"Ann\",\"age\":42,\"role\":\"root\"",
                "{\"name\":\"Ann\",\"age\":42,\"role\":\"user\",\"extra\":1",
                "{\"name\":\"\",",
                "{\"name\":\"Ann\",\"age\":42,\"role\":\"user\",\"email\":\"Ann@x.com\"",
                "{\"age\":42}",
                "{\"name\":\"Ann\",\"age\":42,\"role\":\"user\"}}")) {
            try (GrammarConstraint constraint = compiler.constraint("start: %json " + PERSON)) {
                boolean refused = false;
                for (int id : tokenizer.encodeText(invalid)) {
                    if (!constraint.allows(id)) {
                        refused = true;
                        break;
                    }
                    constraint.accept(id);
                }
                assertTrue(refused || !constraint.complete(), "accepted " + invalid);
                if (!refused) assertTrue(!constraint.complete(), invalid);
            }
        }
    }

    @Test
    void unsupportedSchemasAreRefusedWithLlguidancesReason() throws Exception {
        GrammarCompiler compiler = GuidanceFixtures.compiler();
        var notKeyword = assertThrows(GrammarException.class, () -> compiler.checkJsonSchema("{\"not\":{}}"));
        assertTrue(notKeyword.getMessage().contains("Unimplemented keys: [\"not\"]"), notKeyword.getMessage());
        assertThrows(GrammarException.class, () -> compiler.checkJsonSchema("{\"type\":\"string\",\"format\":\"x\"}"));
        assertThrows(
                GrammarException.class,
                () -> compiler.checkJsonSchema("{\"type\":\"integer\",\"minimum\":5,\"maximum\":1}"));
        assertThrows(GrammarException.class, () -> compiler.check("start: nothing"));
        compiler.checkJsonSchema(PERSON);
    }

    @Test
    void propertiesAppearInTheSchemasOrderAndOverlappingOneOfIsRefused() throws Exception {
        QwenTokenizer tokenizer = GuidanceFixtures.tokenizer();
        GrammarCompiler compiler = GuidanceFixtures.compiler();
        String ordered = "start: %json {\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"},"
                + "\"b\":{\"type\":\"integer\"}},\"required\":[\"a\",\"b\"]}";
        try (GrammarConstraint constraint = compiler.constraint(ordered)) {
            boolean refused = false;
            for (int id : tokenizer.encodeText("{\"b\":1,\"a\":2}")) {
                if (!constraint.allows(id)) {
                    refused = true;
                    break;
                }
                constraint.accept(id);
            }
            assertTrue(refused, "llguidance generates properties in the schema's order");
        }
        assertThrows(
                GrammarException.class,
                () -> compiler.checkJsonSchema("{\"oneOf\":[{\"type\":\"integer\"},{\"type\":\"number\"}]}"));
        compiler.checkJsonSchema("{\"oneOf\":[{\"type\":\"string\"},{\"type\":\"integer\"}]}");
    }

    @Test
    void constraintsFromOneCachedGrammarAreIndependent() throws Exception {
        QwenTokenizer tokenizer = GuidanceFixtures.tokenizer();
        GrammarCompiler compiler = GuidanceFixtures.compiler();
        try (GrammarConstraint first = compiler.constraint("start: %json {\"type\":\"boolean\"}");
                GrammarConstraint second = compiler.constraint("start: %json {\"type\":\"boolean\"}")) {
            for (int id : tokenizer.encodeText("true")) first.accept(id);
            assertTrue(first.complete());
            assertFalse(second.complete());
            assertTrue(second.allows(tokenizer.encodeText("false")[0]));
        }
    }

    private static boolean isStructural(QwenTokenizer tokenizer, int id) {
        if (tokenizer.isGenerationEosToken(id)) return false;
        String text = tokenizer.decode(new int[] {id});
        return text.length() <= 2 && text.chars().allMatch(c -> "\"{}[],:-0123456789aeu".indexOf(c) >= 0);
    }

    private static List<Integer> allowedTokens(GrammarConstraint constraint) {
        return IntStream.range(0, GuidanceFixtures.VOCABULARY)
                .filter(constraint::allows)
                .boxed()
                .collect(java.util.stream.Collectors.toCollection(ArrayList::new));
    }

    private static void assertPerson(JsonNode person, String text) {
        assertTrue(person.isObject(), text);
        person.fieldNames()
                .forEachRemaining(key -> assertTrue(
                        List.of("name", "age", "role", "tags", "email").contains(key), text));
        String name = person.get("name").asText();
        assertTrue(name.codePointCount(0, name.length()) >= 1 && name.codePointCount(0, name.length()) <= 12, text);
        assertTrue(person.get("age").isIntegralNumber(), text);
        long age = person.get("age").asLong();
        assertTrue(age >= 0 && age <= 150, text);
        assertTrue(Map.of("admin", 1, "user", 1).containsKey(person.get("role").asText()), text);
        if (person.has("tags")) {
            assertTrue(person.get("tags").isArray() && person.get("tags").size() <= 3, text);
            for (JsonNode tag : person.get("tags"))
                assertTrue(tag.isTextual() && tag.asText().length() <= 5, text);
        }
        if (person.has("email") && !person.get("email").isNull())
            assertTrue(person.get("email").asText().matches("[a-z]+@[a-z]+\\.com"), text);
    }
}
