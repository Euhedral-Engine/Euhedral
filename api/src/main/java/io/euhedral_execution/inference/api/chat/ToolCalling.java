package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.openai.OpenAiException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/// The functions one request offers and how the model may call them, validated at admission.
///
/// `NONE` omits the definitions from the prompt, as if no tools were sent, so the model is never told it
/// can call and its output is plain text. `AUTO` constrains sampling to a JSON call or a JSON answer,
/// `REQUIRED` and `FUNCTION` to a call ([OutputGrammar#tools]); the parser rejects a malformed or incomplete
/// envelope. `FUNCTION` also restricts every call in the response to `function`. Without `parallel`,
/// decoding ends once the first call is complete.
public record ToolCalling(List<FunctionTool> tools, Choice choice, String function, boolean parallel) {
    static final ToolCalling DISABLED = new ToolCalling(List.of(), Choice.NONE, null, true);
    private static final Pattern FUNCTION_NAME = Pattern.compile("[a-zA-Z0-9_-]{1,64}");

    public enum Choice {
        NONE,
        AUTO,
        REQUIRED,
        FUNCTION
    }

    public ToolCalling {
        tools = List.copyOf(tools);
        if ((choice == Choice.FUNCTION) != (function != null))
            throw new IllegalArgumentException("a named function goes with the FUNCTION choice only");
    }

    /// Validates the request's `tools`, `tool_choice`, and `parallel_tool_calls`.
    static ToolCalling fromRequest(Object tools, Object toolChoice, Boolean parallelToolCalls) {
        if (tools == null) {
            // Without tools only choices that request no behavior are meaningful.
            if (toolChoice == null || "none".equals(toolChoice) || "auto".equals(toolChoice)) return DISABLED;
            throw OpenAiException.invalidRequest(
                    "'tool_choice' is only allowed when 'tools' are specified.", "tool_choice");
        }
        if (!(tools instanceof List<?> definitions))
            throw OpenAiException.invalidRequest("'tools' must be an array of tool definitions.", "tools");
        List<FunctionTool> functions = new ArrayList<>(definitions.size());
        Set<String> names = new HashSet<>();
        for (int index = 0; index < definitions.size(); index++) {
            FunctionTool tool = functionTool(definitions.get(index), "tools[" + index + "]");
            if (!names.add(tool.name()))
                throw OpenAiException.invalidRequest(
                        "Tool names must be unique; '" + tool.name() + "' is defined twice.",
                        "tools[" + index + "].function.name");
            functions.add(tool);
        }
        if (functions.isEmpty()) {
            if (toolChoice == null || "none".equals(toolChoice) || "auto".equals(toolChoice)) return DISABLED;
            throw OpenAiException.invalidRequest(
                    "'tool_choice' is only allowed when 'tools' are specified.", "tool_choice");
        }
        boolean parallel = !Boolean.FALSE.equals(parallelToolCalls);
        return switch (toolChoice) {
            case null -> new ToolCalling(functions, Choice.AUTO, null, parallel);
            case String mode ->
                switch (mode) {
                    case "none" -> new ToolCalling(functions, Choice.NONE, null, parallel);
                    case "auto" -> new ToolCalling(functions, Choice.AUTO, null, parallel);
                    case "required" -> new ToolCalling(functions, Choice.REQUIRED, null, parallel);
                    default ->
                        throw OpenAiException.invalidRequest(
                                "'tool_choice' must be 'none', 'auto', 'required', or a function choice.",
                                "tool_choice");
                };
            case Map<?, ?> named -> new ToolCalling(functions, Choice.FUNCTION, namedFunction(named, names), parallel);
            default ->
                throw OpenAiException.invalidRequest(
                        "'tool_choice' must be 'none', 'auto', 'required', or a function choice.", "tool_choice");
        };
    }

    private static String namedFunction(Map<?, ?> choice, Set<String> names) {
        if (!(choice.get("type") instanceof String type))
            throw OpenAiException.invalidRequest("'tool_choice.type' is required.", "tool_choice.type");
        if (!type.equals("function")) throw OpenAiException.unsupportedParameter("tool_choice.type");
        checkKeys(choice, Set.of("type", "function"), "tool_choice");
        if (!(choice.get("function") instanceof Map<?, ?> function))
            throw OpenAiException.invalidRequest("'tool_choice.function' is required.", "tool_choice.function");
        checkKeys(function, Set.of("name"), "tool_choice.function");
        if (!(function.get("name") instanceof String name))
            throw OpenAiException.invalidRequest(
                    "'tool_choice.function.name' is required.", "tool_choice.function.name");
        if (!names.contains(name))
            throw OpenAiException.invalidRequest(
                    "'tool_choice' names function '" + name + "', which is not in 'tools'.", "tool_choice");
        return name;
    }

    private static FunctionTool functionTool(Object definition, String path) {
        if (!(definition instanceof Map<?, ?> tool))
            throw OpenAiException.invalidRequest("Each tool must be an object.", path);
        Object type = tool.get("type");
        if (!(type instanceof String typeName))
            throw OpenAiException.invalidRequest("Tool 'type' is required.", path + ".type");
        if (!typeName.equals("function")) throw OpenAiException.unsupportedParameter(path + ".type");
        checkKeys(tool, Set.of("type", "function"), path);
        if (!(tool.get("function") instanceof Map<?, ?> function))
            throw OpenAiException.invalidRequest("Function tools require a 'function' object.", path + ".function");
        String functionPath = path + ".function";
        checkKeys(function, Set.of("name", "description", "parameters", "strict"), functionPath);
        if (!(function.get("name") instanceof String name) || !isFunctionName(name))
            throw OpenAiException.invalidRequest(
                    "Function names must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes.",
                    functionPath + ".name");
        Object description = function.get("description");
        if (description != null && !(description instanceof String))
            throw OpenAiException.invalidRequest(
                    "Function 'description' must be a string.", functionPath + ".description");
        // Strict arguments are generated under the full schema, which the mapper has llguidance check.
        Object strict = function.get("strict");
        if (strict != null && !(strict instanceof Boolean))
            throw OpenAiException.invalidRequest("Function 'strict' must be a boolean.", functionPath + ".strict");
        return new FunctionTool(
                name,
                (String) description,
                parameters(function.get("parameters"), functionPath),
                Boolean.TRUE.equals(strict));
    }

    /// Checks only what the call format depends on; the rest of the JSON Schema passes through verbatim.
    @SuppressWarnings("unchecked")
    private static Map<String, Object> parameters(Object value, String functionPath) {
        String path = functionPath + ".parameters";
        if (value == null) return null;
        if (!(value instanceof Map<?, ?> schema))
            throw OpenAiException.invalidRequest("Function 'parameters' must be a JSON Schema object.", path);
        Object type = schema.get("type");
        if (type != null && !"object".equals(type))
            throw OpenAiException.invalidRequest(
                    "Function 'parameters' must describe an object; tool calls pass named arguments.", path + ".type");
        Object properties = schema.get("properties");
        if (properties != null) {
            if (!(properties instanceof Map<?, ?> named))
                throw OpenAiException.invalidRequest("'properties' must be an object.", path + ".properties");
            // JSON object keys may contain delimiter-looking text; they do not use XML parameters.
            for (Object name : named.keySet()) {
                if (!isParameterName((String) name))
                    throw OpenAiException.invalidRequest("Parameter names must be non-empty.", path + ".properties");
            }
        }
        Object required = schema.get("required");
        if (required != null) {
            if (!(required instanceof List<?> list) || !list.stream().allMatch(String.class::isInstance))
                throw OpenAiException.invalidRequest("'required' must be an array of strings.", path + ".required");
            Set<String> names = new HashSet<>();
            for (Object entry : list) {
                String name = (String) entry;
                if (!isParameterName(name) || !names.add(name))
                    throw OpenAiException.invalidRequest(
                            "Required parameter names must be unique and non-empty.", path + ".required");
                if (Boolean.FALSE.equals(schema.get("additionalProperties"))
                        && (!(properties instanceof Map<?, ?> declared) || !declared.containsKey(name)))
                    throw OpenAiException.invalidRequest(
                            "Required parameters must be declared when additional properties are forbidden.",
                            path + ".required");
            }
        }
        return (Map<String, Object>) schema;
    }

    /// OpenAI's function-name rule.
    static boolean isFunctionName(String name) {
        return FUNCTION_NAME.matcher(name).matches();
    }

    static boolean isParameterName(String name) {
        return !name.isEmpty();
    }

    private static void checkKeys(Map<?, ?> object, Set<String> known, String path) {
        for (Object key : object.keySet()) {
            if (!known.contains(key)) throw OpenAiException.unrecognizedArgument(path + "." + key);
        }
    }

    /// True when the model's output may contain calls and must be parsed.
    boolean parsesOutput() {
        return this.choice != Choice.NONE;
    }

    /// Functions the model may call in this response.
    List<FunctionTool> callable() {
        if (this.choice == Choice.NONE) return List.of();
        if (this.choice != Choice.FUNCTION) return this.tools;
        return this.tools.stream()
                .filter(tool -> tool.name().equals(this.function))
                .toList();
    }

    /// Do not prefill an incomplete JSON object: the checkpoint may stop before closing it.
    String generationPrefix() {
        return "";
    }

    /// Tool objects for the prompt's tool-definition block; empty renders the template's no-tools branch.
    List<Map<String, Object>> promptTools() {
        if (this.choice == Choice.NONE) return List.of();
        List<Map<String, Object>> rendered = new ArrayList<>(this.tools.size());
        for (FunctionTool tool : this.tools) rendered.add(tool.templateObject());
        return rendered;
    }
}
