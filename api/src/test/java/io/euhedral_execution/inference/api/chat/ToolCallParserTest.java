package io.euhedral_execution.inference.api.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.json.JsonMapper;

/// Argument typing and framing of generated `<tool_call>` blocks, independent of HTTP.
class ToolCallParserTest {

    /// Each parameter value is read back the way the template wrote it: strings verbatim, other JSON as JSON.
    @Test
    void parameterValuesFollowTheDeclaredSchemaTypes() throws Exception {
        String schema = """
                {"type":"object","properties":{
                  "s":{"type":"string"},
                  "i":{"type":"integer"},
                  "n":{"type":"number"},
                  "b":{"type":"boolean"},
                  "a":{"type":"array"},
                  "o":{"type":"object"},
                  "sn":{"type":["string","null"]},
                  "e":{"enum":["1","2"]},
                  "any":{"anyOf":[{"type":"integer"},{"type":"string"}]},
                  "free":{}
                }}""";
        assertEquals(
                "{\"s\":\"42\",\"i\":42,\"n\":3,\"b\":false,\"a\":[1,\"x\"],\"o\":{\"k\":null},\"sn\":null,"
                        + "\"e\":\"1\",\"any\":\"forty\",\"free\":\"plain words\"}",
                arguments(
                        schema,
                        "s",
                        "42",
                        "i",
                        "42",
                        "n",
                        "3",
                        "b",
                        "false",
                        "a",
                        "[1, \"x\"]",
                        "o",
                        "{\"k\": null}",
                        "sn",
                        "null",
                        "e",
                        "1",
                        "any",
                        "forty",
                        "free",
                        "plain words"));
        // A quoted value keeps its quotes: the template never JSON-encodes strings.
        assertEquals(
                "{\"sn\":\"\\\"quoted\\\"\",\"free\":\"\\\"q\\\"\"}",
                arguments(schema, "sn", "\"quoted\"", "free", "\"q\""));
        // Untyped values take the JSON reading when it parses.
        assertEquals("{\"free\":[1,2]}", arguments(schema, "free", "[1, 2]"));
        // Python literals for booleans read as booleans; anything else of the wrong type names the value.
        assertEquals("{\"b\":true}", arguments(schema, "b", "True"));
        assertEquals("{\"b\":false}", arguments(schema, "b", " FALSE "));
        var refused =
                assertThrows(ToolCallParser.MalformedToolCallException.class, () -> arguments(schema, "b", "yes"));
        assertEquals("parameter b of f is not valid JSON of type [boolean]: 'yes'", refused.getMessage());
    }

    @Test
    void literalClosingCallTagInsideAStringArgumentDoesNotEndTheCall() throws Exception {
        assertEquals(
                "{\"s\":\"before </tool_call> after\"}",
                arguments(
                        "{\"type\":\"object\",\"properties\":{\"s\":{\"type\":\"string\"}}}",
                        "s",
                        "before </tool_call> after"));
    }

    @Test
    void valuesThatAreNotTheDeclaredTypeAreMalformed() {
        String schema =
                "{\"type\":\"object\",\"properties\":{\"i\":{\"type\":\"integer\"},\"b\":{\"type\":\"boolean\"},"
                        + "\"o\":{\"type\":\"object\"},\"n\":{\"type\":\"number\"},\"free\":{}}}";
        for (List<String> invalid : List.of(
                List.of("i", "3 4"),
                List.of("n", "1e400"),
                List.of("free", "[1e400]"),
                List.of("i", "3.5"),
                List.of("i", "\"3\""),
                List.of("i", ""),
                List.of("b", "yes"),
                List.of("o", "[1]"),
                List.of("o", "{\"k\": 1} trailing"))) {
            assertThrows(
                    ToolCallParser.MalformedToolCallException.class,
                    () -> arguments(schema, invalid.toArray(String[]::new)),
                    invalid.toString());
        }
    }

    @Test
    void aRequiredChoiceWithoutACallIsMalformedOnceTheModelStops() throws Exception {
        var tools =
                new ToolCalling(List.of(new FunctionTool("f", null, null)), ToolCalling.Choice.REQUIRED, null, true);
        var parser = new ToolCallParser(tools);
        parser.accept("No call.", new RecordingOutput());
        assertThrows(ToolCallParser.MalformedToolCallException.class, () -> parser.finish(true, new RecordingOutput()));
        var truncated = new ToolCallParser(tools);
        truncated.accept("No call.", new RecordingOutput());
        truncated.finish(false, new RecordingOutput());
    }

    @Test
    void answerTextStreamsAsItArrivesAndOnlyATagPrefixIsHeldBack() throws Exception {
        var tools = new ToolCalling(List.of(new FunctionTool("f", null, null)), ToolCalling.Choice.AUTO, null, true);
        var parser = new ToolCallParser(tools);
        var output = new RecordingOutput();
        parser.accept("It is", output);
        parser.accept(" cold <tool", output);
        assertEquals(List.of("It is", " cold"), output.content);
        parser.accept("_call>\n<function=f>\n</function>\n</tool_call>", output);
        parser.finish(true, output);
        assertEquals(List.of("It is", " cold"), output.content);
        assertEquals(List.of("{}"), output.arguments);
    }

    @Test
    void withoutParallelCallsTextAfterTheFirstCallIsDiscarded() throws Exception {
        var tools = new ToolCalling(List.of(new FunctionTool("f", null, null)), ToolCalling.Choice.AUTO, null, false);
        var parser = new ToolCallParser(tools);
        var output = new RecordingOutput();
        String call = "<tool_call>\n<function=f>\n</function>\n</tool_call>";
        parser.accept(call + "\n" + call + " more", output);
        parser.finish(true, output);
        assertEquals(List.of("{}"), output.arguments);
        assertEquals(List.of(), output.content);
    }

    private static String arguments(String parametersJson, String... keyValues) throws Exception {
        @SuppressWarnings("unchecked")
        Map<String, Object> parameters = JsonMapper.shared().readValue(parametersJson, Map.class);
        var tools =
                new ToolCalling(List.of(new FunctionTool("f", null, parameters)), ToolCalling.Choice.AUTO, null, true);
        StringBuilder block = new StringBuilder("<tool_call>\n<function=f>\n");
        for (int index = 0; index < keyValues.length; index += 2)
            block.append("<parameter=")
                    .append(keyValues[index])
                    .append(">\n")
                    .append(keyValues[index + 1])
                    .append("\n</parameter>\n");
        block.append("</function>\n</tool_call>");
        var output = new RecordingOutput();
        var parser = new ToolCallParser(tools);
        parser.accept(block.toString(), output);
        parser.finish(true, output);
        assertEquals(List.of(), output.content);
        assertEquals(1, output.arguments.size());
        return output.arguments.getFirst();
    }

    private static final class RecordingOutput implements ToolCallParser.Output {
        final List<String> content = new ArrayList<>();
        final List<String> arguments = new ArrayList<>();

        @Override
        public void content(String text) {
            this.content.add(text);
        }

        @Override
        public void toolCall(String name, String arguments) {
            this.arguments.add(arguments);
        }
    }
}
