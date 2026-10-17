package io.euhedral_execution.inference.api.chat;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "unset-euhedral.qwen.tokenizer-dir"));
        assumeTrue(Files.isRegularFile(checkpoint.resolve("tokenizer.json")));
        tokenizer = QwenTokenizer.load(checkpoint);
        compiler = GrammarCompiler.forTokenizer(
                Llguidance.load(Path.of(System.getProperty("euhedral.llguidance.library"))), tokenizer, 248320);
    }

    private static final String OSLO = "<parameter=city>\nOslo\n</parameter>\n";

    @Test
    void autoWithAJsonAnswerAllowsACallOrTheDocumentAndNothingElse() {
        Map<String, Object> answer = schema("{\"type\":\"object\",\"properties\":"
                + "{\"answer\":{\"type\":\"integer\"}},\"required\":[\"answer\"]}");
        String grammar = OutputGrammar.tools(calling(ToolCalling.Choice.AUTO, true, false), answer, false);
        compiler.check(grammar);
        assertTrue(finishes(grammar, call("get_weather", OSLO)));
        assertTrue(finishes(grammar, "{\"answer\":42}"));
        assertFalse(finishes(grammar, "{\"answer\":\"42\"}"));
        assertFalse(finishes(grammar, "It is cold."));
        assertFalse(finishes(grammar, call("get_time", "")), "only offered functions");
        assertFalse(finishes(grammar, " {\"answer\":42}"), "without reasoning the answer starts at once");
        String reasoning = OutputGrammar.tools(calling(ToolCalling.Choice.AUTO, true, false), answer, true);
        assertTrue(finishes(reasoning, "\n\n" + call("get_weather", OSLO)));
        assertTrue(finishes(reasoning, "\n\n{\"answer\":1}"));
    }

    @Test
    void aRequiredCallOffersNoAnswer() {
        String grammar = OutputGrammar.tools(calling(ToolCalling.Choice.REQUIRED, true, false), null, false);
        compiler.check(grammar);
        assertTrue(finishes(grammar, call("get_weather", OSLO)));
        assertFalse(finishes(grammar, "It is cold."));
        assertFalse(finishes(grammar, ""));
    }

    @Test
    void aNonStrictFunctionTakesAnyNamesAndTextValues() {
        String grammar = OutputGrammar.tools(calling(ToolCalling.Choice.REQUIRED, true, false), null, false);
        assertTrue(finishes(grammar, call("get_weather", "")));
        assertTrue(finishes(grammar, call("get_weather", OSLO + "<parameter=x-1>\n[1, 2]\n</parameter>\n")));
        assertTrue(finishes(grammar, call("get_weather", "<parameter=a>\n\n</parameter>\n")));
        assertTrue(finishes(grammar, call("get_weather", "<parameter=a>\nline\nline two\n</parameter>\n")));
        assertFalse(finishes(grammar, call("get_weather", "<parameter=a b>\nx\n</parameter>\n")), "key characters");
        assertFalse(finishes(grammar, "<tool_call>\n<function=get_weather>\n</function>"), "unclosed call");
    }

    @Test
    void aStringValueCannotContainTheClosingTag() {
        String grammar = OutputGrammar.tools(calling(ToolCalling.Choice.REQUIRED, true, false), null, false);
        assertFalse(finishes(grammar, call("get_weather", "<parameter=a>\nx</parameter>y\n</parameter>\n")));
        assertFalse(finishes(grammar, call("get_weather", "<parameter=a>\nx</parameter>\ny\n</parameter>\n")));
        assertTrue(finishes(grammar, call("get_weather", "<parameter=a>\nx</paramete\n</parameter>\n")));
        assertTrue(finishes(grammar, call("get_weather", "<parameter=a>\nif a < b && c</d> then\n</parameter>\n")));
        assertTrue(finishes(grammar, call("get_weather", "<parameter=a>\n</parameter\n</parameter>\n")));
        String strict = OutputGrammar.tools(calling(ToolCalling.Choice.REQUIRED, true, true), null, false);
        assertFalse(finishes(strict, call("get_weather", "<parameter=city>\nOs</parameter>lo\n</parameter>\n")));
        assertTrue(finishes(strict, call("get_weather", "<parameter=city>\nOs</paramete lo<\n</parameter>\n")));
    }

    @Test
    void strictParametersFollowTheSchemaInOrder() {
        String grammar = OutputGrammar.tools(calling(ToolCalling.Choice.REQUIRED, true, true), null, false);
        compiler.check(grammar);
        String units = "<parameter=units>\nmetric\n</parameter>\n";
        assertTrue(finishes(grammar, call("get_weather", OSLO + units)));
        assertTrue(finishes(grammar, call("get_weather", OSLO)));
        assertFalse(finishes(grammar, call("get_weather", units)), "required property missing");
        assertFalse(finishes(grammar, call("get_weather", units + OSLO)), "schema order");
        assertFalse(finishes(grammar, call("get_weather", OSLO + "<parameter=units>\nkelvin\n</parameter>\n")), "enum");
        assertFalse(finishes(grammar, call("get_weather", OSLO + "<parameter=wind>\n1\n</parameter>\n")), "extra");
        assertFalse(finishes(grammar, call("get_weather", OSLO + OSLO)), "repeated");
    }

    @Test
    void strictNonStringParametersAreJsonOfTheirSchema() {
        Map<String, Object> schema = schema("""
                {"type":"object","properties":{"count":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}},
                 "note":{"type":["string","null"]},"free":{},"mode":{"$ref":"#/$defs/mode"}},
                 "required":["count","mode"],"additionalProperties":false,
                 "$defs":{"mode":{"type":"integer","enum":[1,2]}}}""");
        var tool = new FunctionTool("record", null, schema, true);
        String grammar = OutputGrammar.tools(
                new ToolCalling(List.of(tool), ToolCalling.Choice.FUNCTION, "record", true), null, false);
        compiler.check(grammar);
        String mode = "<parameter=mode>\n2\n</parameter>\n";
        String count = "<parameter=count>\n3\n</parameter>\n";
        assertTrue(finishes(grammar, call("record", count + mode)));
        assertTrue(finishes(
                grammar,
                call(
                        "record",
                        count + "<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n"
                                + "<parameter=note>\nfree text\n</parameter>\n<parameter=free>\nx y\n</parameter>\n"
                                + mode)));
        assertFalse(finishes(grammar, call("record", "<parameter=count>\nthree\n</parameter>\n" + mode)));
        assertFalse(finishes(grammar, call("record", count + "<parameter=mode>\n5\n</parameter>\n")), "$ref enum");
        assertFalse(finishes(grammar, call("record", count)), "required property missing");
    }

    @Test
    void aNamedFunctionIsTheOnlyChoice() {
        var weather = new FunctionTool("get_weather", null, WEATHER, true);
        var time = new FunctionTool("get_time", null, null, false);
        String grammar = OutputGrammar.tools(
                new ToolCalling(List.of(weather, time), ToolCalling.Choice.FUNCTION, "get_time", true), null, false);
        assertTrue(finishes(grammar, call("get_time", "")));
        assertFalse(finishes(grammar, call("get_weather", OSLO)));
        String either = OutputGrammar.tools(
                new ToolCalling(List.of(weather, time), ToolCalling.Choice.REQUIRED, null, true), null, false);
        assertTrue(finishes(either, call("get_weather", OSLO)));
        assertTrue(finishes(either, call("get_time", "")));
    }

    @Test
    void parallelCallsAreSeparatedByANewline() {
        String one = call("get_weather", OSLO);
        String single = OutputGrammar.tools(calling(ToolCalling.Choice.REQUIRED, false, true), null, false);
        assertTrue(finishes(single, one));
        assertFalse(finishes(single, one + "\n" + one));
        String parallel = OutputGrammar.tools(calling(ToolCalling.Choice.REQUIRED, true, true), null, false);
        assertTrue(finishes(parallel, one + "\n" + one));
        assertFalse(finishes(parallel, one + one));
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
        assertTrue(finishes(grammar, call("walk", "<parameter=head>\n{\"v\":1,\"next\":{\"v\":2}}\n</parameter>\n")));
        assertFalse(finishes(grammar, call("walk", "<parameter=head>\n{\"v\":1,\"next\":{\"w\":2}}\n</parameter>\n")));
    }

    @Test
    void aStringKeywordTheGrammarCannotEnforceIsRefused() {
        Map<String, Object> schema = schema("""
                {"type":"object","properties":{"id":{"type":"string","pattern":"^a+$"}},"required":["id"]}""");
        var tool = new FunctionTool("lookup", null, schema, true);
        var calling = new ToolCalling(List.of(tool), ToolCalling.Choice.REQUIRED, null, true);
        assertThrows(ApiException.class, () -> OutputGrammar.tools(calling, null, false));
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

    private static String call(String name, String parameters) {
        return "<tool_call>\n<function=" + name + ">\n" + parameters + "</function>\n</tool_call>";
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
