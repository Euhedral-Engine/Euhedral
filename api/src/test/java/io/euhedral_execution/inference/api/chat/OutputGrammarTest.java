package io.euhedral_execution.inference.api.chat;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.guidance.GrammarCompiler;
import io.euhedral_execution.inference.core.guidance.GrammarConstraint;
import io.euhedral_execution.inference.core.guidance.Llguidance;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.json.JsonMapper;

/// The answer grammars over the checkpoint vocabulary: what each lets a generation finish with.
class OutputGrammarTest {
    private static QwenTokenizer tokenizer;
    private static GrammarCompiler compiler;

    private static final Map<String, Object> WEATHER = schema("""
            {"type":"object","properties":{"city":{"type":"string"},"units":{"enum":["metric","imperial"]}},
             "required":["city"],"additionalProperties":false}""");

    @BeforeAll
    static void loadVocabulary() throws Exception {
        Path checkpoint =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(checkpoint.resolve("tokenizer.json")));
        tokenizer = QwenTokenizer.load(checkpoint);
        compiler = GrammarCompiler.forTokenizer(
                Llguidance.load(Path.of(System.getProperty("euhedral.llguidance.library"))), tokenizer, 248320);
    }

    @Test
    void autoToolsAllowACallOrADirectAnswerAndNothingElse() {
        String grammar = OutputGrammar.tools(calling(ToolCalling.Choice.AUTO, true, false), null, false);
        assertTrue(
                finishes(grammar, "{\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"}}]}"));
        assertTrue(finishes(grammar, "{\"content\":\"It is cold.\"}"));
        assertFalse(finishes(grammar, "{\"tool_calls\":[{\"name\":\"get_time\",\"arguments\":{}}]}"));
        assertFalse(finishes(grammar, "It is cold."));
        assertFalse(finishes(grammar, "<tool_call>"));
        assertFalse(finishes(grammar, " {\"content\":\"x\"}"), "without reasoning the answer starts at once");
        assertTrue(
                finishes(
                        grammar,
                        "{\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"city\": 1, \"x\": []}}]}"),
                "a non-strict function takes any object");
    }

    @Test
    void strictArgumentsFollowTheSchema() {
        String grammar = OutputGrammar.tools(calling(ToolCalling.Choice.REQUIRED, true, true), null, false);
        assertTrue(finishes(grammar, call("{\"city\":\"Oslo\",\"units\":\"metric\"}")));
        assertTrue(finishes(grammar, call("{\"city\": \"Oslo\"}")));
        assertFalse(finishes(grammar, call("{\"units\":\"metric\"}")), "required property missing");
        assertFalse(finishes(grammar, call("{\"city\":\"Oslo\",\"units\":\"kelvin\"}")), "enum");
        assertFalse(finishes(grammar, call("{\"city\":\"Oslo\",\"wind\":true}")), "additional property");
        assertFalse(finishes(grammar, call("{\"city\":7}")), "type");
        assertFalse(finishes(grammar, "{\"content\":\"x\"}"), "a required call offers no answer");
    }

    @Test
    void aSingleCallLimitEndsTheArrayAfterOneCall() {
        String grammar = OutputGrammar.tools(calling(ToolCalling.Choice.AUTO, false, false), null, false);
        String one = "{\"name\":\"get_weather\",\"arguments\":{}}";
        assertTrue(finishes(grammar, "{\"tool_calls\":[" + one + "]}"));
        assertFalse(finishes(grammar, "{\"tool_calls\":[" + one + "," + one + "]}"));
        String parallel = OutputGrammar.tools(calling(ToolCalling.Choice.AUTO, true, false), null, false);
        assertTrue(finishes(parallel, "{\"tool_calls\":[" + one + "," + one + "]}"));
    }

    @Test
    void strictSchemasKeepTheirOwnReferences() {
        Map<String, Object> recursive = schema("""
                {"type":"object","$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},
                 "next":{"$ref":"#/$defs/node"}},"required":["v"],"additionalProperties":false}},
                 "properties":{"head":{"$ref":"#/$defs/node"}},"required":["head"],"additionalProperties":false}""");
        var tool = new FunctionTool("walk", null, recursive, true);
        String grammar = OutputGrammar.tools(
                new ToolCalling(List.of(tool), ToolCalling.Choice.REQUIRED, null, true), null, false);
        compiler.check(grammar);
        assertTrue(finishes(
                grammar,
                "{\"tool_calls\":[{\"name\":\"walk\",\"arguments\":" + "{\"head\":{\"v\":1,\"next\":{\"v\":2}}}}]}"));
        assertFalse(finishes(
                grammar,
                "{\"tool_calls\":[{\"name\":\"walk\",\"arguments\":" + "{\"head\":{\"v\":1,\"next\":{\"w\":2}}}}]}"));
    }

    @Test
    void aStructuredAnswerBesideToolsIsADocumentOfTheResponseSchema() {
        String grammar = OutputGrammar.tools(
                calling(ToolCalling.Choice.AUTO, true, false),
                schema("{\"type\":\"object\",\"properties\":"
                        + "{\"answer\":{\"type\":\"integer\"}},\"required\":[\"answer\"]}"),
                false);
        assertTrue(finishes(grammar, "{\"content\":{\"answer\":42}}"));
        assertFalse(finishes(grammar, "{\"content\":\"42\"}"));
    }

    @Test
    void afterReasoningTheAnswerMayFollowABlankLine() {
        String grammar = OutputGrammar.json(schema("{\"type\":\"array\",\"items\":{\"type\":\"number\"}}"), true);
        assertTrue(finishes(grammar, "\n\n[1, 2.5]"));
        assertFalse(finishes(grammar, " ".repeat(9) + "[1]"), "the gap is bounded");
        String plain = OutputGrammar.json(schema("{\"type\":\"array\"}"), false);
        assertFalse(finishes(plain, "\n[1]"));
        assertTrue(finishes(plain, "[1,\n  2]"));
        assertFalse(finishes(plain, "[1," + " ".repeat(41) + "2]"), "whitespace runs are bounded");
    }

    private static String call(String arguments) {
        return "{\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":" + arguments + "}]}";
    }

    private static ToolCalling calling(ToolCalling.Choice choice, boolean parallel, boolean strict) {
        return new ToolCalling(
                List.of(new FunctionTool("get_weather", "Weather.", WEATHER, strict)), choice, null, parallel);
    }

    /// Whether a generation could produce exactly `text` and then end.
    private static boolean finishes(String grammar, String text) {
        try (GrammarConstraint constraint = compiler.constraint(grammar)) {
            for (int id : tokenizer.encodeText(text)) {
                if (!constraint.allows(id)) return false;
                constraint.accept(id);
            }
            return constraint.allows(tokenizer.eosTokenId());
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> schema(String json) {
        return JsonMapper.shared().readValue(json, Map.class);
    }
}
