package io.euhedral_execution.inference.api.chat;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import tools.jackson.databind.json.JsonMapper;

/// The llguidance grammars (Lark) that constrain an answer: a JSON document for `response_format`, or the JSON
/// tool-call envelope. llguidance compiles each `%json` schema in strict mode, so a schema keyword it cannot
/// enforce is refused rather than ignored.
///
/// - Whitespace between JSON tokens is bounded to runs of 40 bytes, so a model cannot pad forever.
/// - After reasoning the answer may begin with up to 8 whitespace bytes, the blank line the template puts after
///   `</think>`; without reasoning it begins at its first byte.
/// - The envelope is spelled exactly, without whitespace: `{"tool_calls":[{"name":"f","arguments":{...}}]}` or,
///   when the model may answer instead, `{"content":"..."}`. Each strict function's arguments are its own `%json`
///   rule, so its schema keeps its own root for `$ref`; other functions take any JSON object.
final class OutputGrammar {
    private static final JsonMapper JSON = JsonMapper.shared();
    /// The skipped whitespace between JSON tokens.
    static final String WHITESPACE = "[\\x20\\x0A\\x0D\\x09]{1,40}";
    private static final Map<String, Object> ANY_OBJECT = Map.of("type", "object");
    private static final Map<String, Object> STRING = Map.of("type", "string");

    private OutputGrammar() {}

    /// A JSON document of `schema`.
    static String json(Map<String, Object> schema, boolean reasoning) {
        return start(reasoning, "answer") + "answer: %json " + schemaText(schema) + "\n";
    }

    /// The tool-call envelope for `calling`'s callable functions. `content` is the schema of a direct answer
    /// (null: a JSON string); a choice that requires a call offers no answer.
    static String tools(ToolCalling calling, Map<String, Object> content, boolean reasoning) {
        List<FunctionTool> callable = calling.callable();
        StringBuilder grammar = new StringBuilder(start(reasoning, "body"));
        boolean answer = calling.choice() == ToolCalling.Choice.AUTO;
        grammar.append("body: calls").append(answer ? " | content" : "").append('\n');
        grammar.append("calls: \"{\\\"tool_calls\\\":[\" call")
                .append(calling.parallel() ? " (\",\" call)*" : "")
                .append(" \"]}\"\n");
        grammar.append("call: ");
        for (int index = 0; index < callable.size(); index++) {
            if (index > 0) grammar.append(" | ");
            // Function names are [a-zA-Z0-9_-]{1,64}: nothing in them needs escaping.
            grammar.append("\"{\\\"name\\\":\\\"")
                    .append(callable.get(index).name())
                    .append("\\\",\\\"arguments\\\":\" arguments_")
                    .append(index)
                    .append(" \"}\"");
        }
        grammar.append('\n');
        for (int index = 0; index < callable.size(); index++) {
            FunctionTool tool = callable.get(index);
            Map<String, Object> arguments = tool.strict() ? tool.parametersOrEmpty() : ANY_OBJECT;
            grammar.append("arguments_")
                    .append(index)
                    .append(": %json ")
                    .append(schemaText(arguments))
                    .append('\n');
        }
        if (answer) {
            grammar.append("content: \"{\\\"content\\\":\" content_value \"}\"\n");
            grammar.append("content_value: %json ")
                    .append(schemaText(content == null ? STRING : content))
                    .append('\n');
        }
        return grammar.toString();
    }

    /// `schema` as llguidance reads it: compact JSON with the bounded whitespace option.
    static String schemaText(Map<String, Object> schema) {
        Map<String, Object> options = new LinkedHashMap<>();
        options.put("x-guidance", Map.of("whitespace_pattern", WHITESPACE));
        options.putAll(schema);
        return JSON.writeValueAsString(options);
    }

    private static String start(boolean reasoning, String rule) {
        return reasoning ? "start: GAP? " + rule + "\nGAP: /[ \\t\\r\\n]{1,8}/\n" : "start: " + rule + "\n";
    }
}
