package io.euhedral_execution.inference.api.chat;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import tools.jackson.databind.json.JsonMapper;

/// The llguidance grammars (Lark) that constrain an answer: a JSON document for `response_format`, or the
/// checkpoint's native tool calls. llguidance compiles each `%json` schema in strict mode, so a schema keyword it
/// cannot enforce is refused rather than ignored.
///
/// - Whitespace between JSON tokens is bounded to runs of 40 bytes, so a model cannot pad forever.
/// - After reasoning the answer may begin with up to 8 whitespace bytes, the blank line the template puts after
///   `</think>`; without reasoning it begins at its first byte.
/// - A call is spelled as the template renders it,
///   `<tool_call>\n<function=NAME>\n(<parameter=KEY>\nVALUE\n</parameter>\n)*</function>\n</tool_call>`, and
///   consecutive calls are separated by `\n`. A string VALUE is a terminal that includes its closing
///   `\n</parameter>\n` and cannot contain `</parameter>`, so any other text, `<` and `</paramete` included, is
///   admitted. A strict function's parameters follow its schema's `properties` in schema order: required ones are
///   present, optional ones may be omitted, and a parameter the schema types as a string is text (an `enum` or
///   `const` of strings, its literals). A parameter of another type is a `%json` of its schema, which keeps the
///   function's `$defs`; an untyped one, or a union that includes a string, is any text. Names outside `properties`
///   are generated only when the schema does not set `additionalProperties: false`, or for a required name it
///   does not declare. A function that is not strict takes any parameter names and any text values.
final class OutputGrammar {
    private static final JsonMapper JSON = JsonMapper.shared();
    /// The skipped whitespace between JSON tokens.
    static final String WHITESPACE = "[\\x20\\x0A\\x0D\\x09]{1,40}";
    private static final Set<String> ANNOTATIONS = Set.of("description", "title", "default", "examples", "$comment");
    private static final List<String> UNENFORCEABLE_STRING_KEYWORDS =
            List.of("pattern", "format", "minLength", "maxLength");

    private OutputGrammar() {}

    /// A JSON document of `schema`.
    static String json(Map<String, Object> schema, boolean reasoning) {
        return start(reasoning, "answer") + "answer: %json " + schemaText(schema) + "\n";
    }

    /// The native calls of `calling`'s callable functions. `answer` is the schema of a direct JSON answer that may
    /// replace the calls (null: a choice that requires a call, which offers no answer).
    static String tools(ToolCalling calling, Map<String, Object> answer, boolean reasoning) {
        boolean auto = calling.choice() == ToolCalling.Choice.AUTO;
        if (auto != (answer != null))
            throw new IllegalArgumentException("only a choice that may answer takes an answer schema");
        List<FunctionTool> callable = calling.callable();
        StringBuilder grammar = new StringBuilder(start(reasoning, auto ? "(calls | answer)" : "calls"));
        grammar.append("calls: call")
                .append(calling.parallel() ? " (\"\\n\" call)*" : "")
                .append('\n');
        grammar.append("call: <tool_call> \"\\n\" (");
        for (int index = 0; index < callable.size(); index++) {
            if (index > 0) grammar.append(" | ");
            grammar.append("function_").append(index);
        }
        grammar.append(") </tool_call>\n");
        StringBuilder rules = new StringBuilder();
        for (int index = 0; index < callable.size(); index++) function(callable.get(index), index, grammar, rules);
        grammar.append(rules);
        if (auto) grammar.append("answer: %json ").append(schemaText(answer)).append('\n');
        grammar.append(TEXT);
        grammar.append("KEY: /[a-zA-Z0-9_-]+/\n");
        return grammar.toString();
    }

    /// Text up to the first `</parameter>`, which it consumes together with the newline before it and the one after.
    private static final String TEXT =
            "TEXT: (ANY & ~(ANY \"</parameter>\" ANY)) \"\\n</parameter>\\n\"\n" + "ANY: /[\\s\\S]*/\n";

    private static void function(FunctionTool tool, int index, StringBuilder grammar, StringBuilder rules) {
        // Function names are [a-zA-Z0-9_-]{1,64}: nothing in them needs escaping.
        grammar.append("function_")
                .append(index)
                .append(": \"<function=")
                .append(tool.name())
                .append(">\\n\"");
        if (!tool.strict()) {
            grammar.append(" parameter_").append(index).append("* \"</function>\\n\"\n");
            rules.append("parameter_").append(index).append(": \"<parameter=\" KEY \">\\n\" TEXT\n");
            return;
        }
        Map<String, Object> schema = tool.parametersOrEmpty();
        Map<?, ?> properties = schema.get("properties") instanceof Map<?, ?> declared ? declared : Map.of();
        Set<String> required = new LinkedHashSet<>(tool.requiredParameters());
        int parameter = 0;
        for (var property : properties.entrySet()) {
            String key = (String) property.getKey();
            if (property.getValue() instanceof Boolean allowed && !allowed) continue;
            String name = "parameter_" + index + "_" + parameter++;
            grammar.append(' ').append(name).append(required.contains(key) ? "" : "?");
            rules.append(name)
                    .append(": \"<parameter=\" ")
                    .append(escaped(key))
                    .append(" \">\\n\" ")
                    .append(value(tool, schema, key, property.getValue(), name))
                    .append('\n');
        }
        for (String key : required) {
            if (properties.containsKey(key)) continue;
            String name = "parameter_" + index + "_" + parameter++;
            grammar.append(' ').append(name);
            rules.append(name).append(": \"<parameter=\" ").append(escaped(key)).append(" \">\\n\" TEXT\n");
        }
        if (!Boolean.FALSE.equals(schema.get("additionalProperties"))) {
            grammar.append(" parameter_").append(index).append("*");
            rules.append("parameter_").append(index).append(": \"<parameter=\" KEY \">\\n\" TEXT\n");
        }
        grammar.append(" \"</function>\\n\"\n");
    }

    /// The right-hand side of a parameter after its opening tag, including its closing tag.
    private static String value(FunctionTool tool, Map<String, Object> root, String key, Object property, String rule) {
        if (!(property instanceof Map<?, ?> schema)) return "TEXT";
        Set<String> types = tool.parameterTypes(key);
        if (types.equals(Set.of("string"))) {
            for (String keyword : UNENFORCEABLE_STRING_KEYWORDS)
                if (schema.containsKey(keyword))
                    throw ApiException.unenforceableSchema(
                            "The JSON Schema cannot be enforced: a string parameter of a native tool call cannot"
                                    + " use '" + keyword + "'; parameter '" + key + "' of " + tool.name() + ".",
                            tool.parametersParam());
            List<?> literals = schema.get("enum") instanceof List<?> values
                    ? values
                    : schema.containsKey("const") ? List.of(schema.get("const")) : null;
            if (literals == null) return "TEXT";
            StringBuilder choices = new StringBuilder("(");
            for (Object literal : literals) {
                if (choices.length() > 1) choices.append(" | ");
                choices.append(escaped((String) literal));
            }
            return choices + ") \"\\n</parameter>\\n\"";
        }
        if (types.contains("string") || (types.isEmpty() && ANNOTATIONS.containsAll(schema.keySet()))) return "TEXT";
        // The parameter's own schema, with the function's definitions so its references resolve.
        Map<String, Object> standalone = new LinkedHashMap<>();
        for (var entry : schema.entrySet()) standalone.put((String) entry.getKey(), entry.getValue());
        for (String definitions : List.of("$defs", "definitions"))
            if (root.containsKey(definitions)) standalone.putIfAbsent(definitions, root.get(definitions));
        return rule + "_json \"\\n</parameter>\\n\"\n" + rule + "_json: %json " + schemaText(standalone);
    }

    private static String escaped(String text) {
        return JSON.writeValueAsString(text);
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
