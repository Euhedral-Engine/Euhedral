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
