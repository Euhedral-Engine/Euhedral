package io.euhedral_execution.inference.api.chat;

import java.math.BigInteger;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import tools.jackson.core.JacksonException;
import tools.jackson.core.StreamReadFeature;
import tools.jackson.databind.json.JsonMapper;

/// Parses a single JSON tool-call envelope without interpreting delimiter-like text inside JSON strings.
/// Plain content streams until the envelope begins; calls are delivered only after the entire JSON object
/// is validated. Confined to the generation thread.
final class JsonToolCallParser {
    private static final String KEY = "\"tool_calls\"";
    private static final String CONTENT_KEY = "\"content\"";
    private static final String LEGACY_CALL = QwenChatTemplate.TOOL_CALL_START;
    private static final JsonMapper JSON = JsonMapper.builder()
            .enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
            .build();

    private enum State {
        CONTENT,
        OBJECT,
        DONE
    }

    private final Map<String, FunctionTool> callable = new HashMap<>();
    private final boolean parallel;
    private final boolean requiresCall;
    private final StringBuilder buffer = new StringBuilder();
    private State state = State.CONTENT;
    private int scanned;
    private int depth;
    private boolean quoted;
    private boolean escaped;
    private int calls;

    JsonToolCallParser(ToolCalling tools) {
        for (FunctionTool tool : tools.callable()) this.callable.put(tool.name(), tool);
        this.parallel = tools.parallel();
        this.requiresCall =
                tools.choice() == ToolCalling.Choice.REQUIRED || tools.choice() == ToolCalling.Choice.FUNCTION;
        this.buffer.append(tools.generationPrefix());
    }

    int calls() {
        return this.calls;
    }

    void accept(String chunk, ToolCallParser.Output output) throws ToolCallParser.MalformedToolCallException {
        this.buffer.append(chunk);
        drain(output);
    }

    void finish(boolean stopTokenReached, ToolCallParser.Output output)
            throws ToolCallParser.MalformedToolCallException {
        drain(output);
        if (this.state == State.CONTENT) {
            for (int index = 0; index < this.buffer.length(); index++)
                if (this.buffer.charAt(index) == '{' && openingAt(index) >= 0)
                    throw malformed("generation ended inside a JSON tool call");
            if (this.requiresCall) throw malformed("the model did not make the required function call");
            if (!this.buffer.isEmpty()) output.content(this.buffer.toString());
        } else if (this.state == State.OBJECT || !this.buffer.isEmpty()) {
            throw malformed("generation ended inside a JSON tool call");
        }
        this.buffer.setLength(0);
    }

    private void drain(ToolCallParser.Output output) throws ToolCallParser.MalformedToolCallException {
        while (true) {
            switch (this.state) {
                case CONTENT -> {
                    int marker = -1;
                    boolean complete = false;
                    for (int index = 0; index < this.buffer.length(); index++) {
                        if (this.buffer.charAt(index) != '{') continue;
                        int opening = openingAt(index);
                        if (opening < 0) continue;
                        marker = index;
                        complete = opening == 1;
                        break;
                    }
                    int legacy = this.buffer.indexOf(LEGACY_CALL);
                    if (legacy >= 0 && (marker < 0 || legacy < marker))
                        throw malformed("the model used the unsupported XML tool-call format");
                    if (marker < 0) {
                        int hold = suffixPrefix(LEGACY_CALL);
                        int safe = trailingSpaceStart(this.buffer.length() - hold);
                        if (safe > 0) output.content(this.buffer.substring(0, safe));
                        this.buffer.delete(0, safe);
                        return;
                    }
                    int end = trailingSpaceStart(marker);
                    if (end > 0) output.content(this.buffer.substring(0, end));
                    if (!complete) {
                        this.buffer.delete(0, end);
                        return;
                    }
                    this.buffer.delete(0, marker);
                    this.state = State.OBJECT;
                }
                case OBJECT -> {
                    for (; this.scanned < this.buffer.length(); this.scanned++) {
                        char c = this.buffer.charAt(this.scanned);
                        if (this.escaped) {
                            this.escaped = false;
                        } else if (this.quoted && c == '\\') {
                            this.escaped = true;
                        } else if (c == '"') {
                            this.quoted = !this.quoted;
                        } else if (!this.quoted) {
                            if (c == '{' || c == '[') this.depth++;
                            else if (c == '}' || c == ']') this.depth--;
                            if (this.depth < 0) throw malformed("JSON tool-call delimiters are unbalanced");
                            if (this.depth == 0) {
                                String object = this.buffer.substring(0, this.scanned + 1);
                                this.buffer.delete(0, this.scanned + 1);
                                validateAndDeliver(object, output);
                                this.state = State.DONE;
                                break;
                            }
                        }
                    }
                    if (this.state == State.OBJECT) return;
                }
                case DONE -> {
                    if (!this.buffer.toString().isBlank())
                        throw malformed("the model produced text after a JSON tool call");
                    this.buffer.setLength(0);
                    return;
                }
            }
        }
    }

    private void validateAndDeliver(String text, ToolCallParser.Output output)
            throws ToolCallParser.MalformedToolCallException {
        Object decoded;
        try {
            decoded = JSON.readValue(text, Object.class);
        } catch (JacksonException invalid) {
            throw malformed("the model produced invalid JSON tool calls");
        }
        if (decoded instanceof Map<?, ?> content
                && content.keySet().equals(Set.of("content"))
                && content.get("content") instanceof String answer) {
            if (this.requiresCall) throw malformed("the model did not make the required function call");
            output.content(answer);
            return;
        }
        if (!(decoded instanceof Map<?, ?> root)
                || !root.keySet().equals(Set.of("tool_calls"))
                || !(root.get("tool_calls") instanceof List<?> entries)
                || entries.isEmpty()) throw malformed("the tool-call object must contain a nonempty tool_calls array");
        if (!this.parallel && entries.size() != 1)
            throw malformed("parallel_tool_calls=false allows one function call");
        for (Object entry : entries) {
            if (!(entry instanceof Map<?, ?> call)
                    || !call.keySet().equals(Set.of("name", "arguments"))
                    || !(call.get("name") instanceof String name))
                throw malformed("each tool call needs a name and an arguments object");
            FunctionTool tool = this.callable.get(name);
            if (tool == null) throw malformed("the model called a function that was not offered: " + name);
            if (!(call.get("arguments") instanceof Map<?, ?> arguments))
                throw malformed("the arguments of " + name + " must be an object");
            for (var argument : arguments.entrySet()) {
                if (!(argument.getKey() instanceof String key)
                        || !ToolCalling.isParameterName(key)
                        || !tool.acceptsParameter(key)) throw malformed(name + " has an invalid parameter name");
                Set<String> types = tool.parameterTypes(key);
                String actual = jsonType(argument.getValue());
                if (!types.isEmpty()
                        && !types.contains(actual)
                        && !(actual.equals("integer") && types.contains("number")))
                    throw malformed("parameter " + key + " of " + name + " is not of type " + types);
            }
            for (String required : tool.requiredParameters())
                if (!arguments.containsKey(required)) throw malformed("required parameter " + required + " is missing");
        }
        for (Object entry : entries) {
            Map<?, ?> call = (Map<?, ?>) entry;
            this.calls++;
            output.toolCall((String) call.get("name"), JSON.writeValueAsString(call.get("arguments")));
        }
    }

    private static String jsonType(Object value) {
        return switch (value) {
            case null -> "null";
            case String _ -> "string";
            case Boolean _ -> "boolean";
            case Integer _, Long _, BigInteger _ -> "integer";
            case Number _ -> "number";
            case List<?> _ -> "array";
            case Map<?, ?> _ -> "object";
            default -> throw new IllegalStateException("unexpected JSON value " + value.getClass());
        };
    }

    /// A JSON envelope can contain whitespace between its structural tokens, including across chunks.
    /// Returns 1 for a complete opening, 0 for an incomplete possible opening, -1 otherwise.
    private int openingAt(int start) {
        int tool = openingFor(start, KEY, '[');
        int content = openingFor(start, CONTENT_KEY, '"');
        if (tool == 1 || content == 1) return 1;
        return tool == 0 || content == 0 ? 0 : -1;
    }

    private int openingFor(int start, String key, char valueStart) {
        int index = start + 1;
        while (index < this.buffer.length() && jsonSpace(this.buffer.charAt(index))) index++;
        for (int offset = 0; offset < key.length(); offset++) {
            if (index == this.buffer.length()) return 0;
            if (this.buffer.charAt(index++) != key.charAt(offset)) return -1;
        }
        while (index < this.buffer.length() && jsonSpace(this.buffer.charAt(index))) index++;
        if (index == this.buffer.length()) return 0;
        if (this.buffer.charAt(index++) != ':') return -1;
        while (index < this.buffer.length() && jsonSpace(this.buffer.charAt(index))) index++;
        if (index == this.buffer.length()) return 0;
        return this.buffer.charAt(index) == valueStart ? 1 : -1;
    }

    private static boolean jsonSpace(char value) {
        return value == ' ' || value == '\t' || value == '\n' || value == '\r';
    }

    private int trailingSpaceStart(int end) {
        while (end > 0 && QwenChatTemplate.isPythonSpace(this.buffer.charAt(end - 1))) end--;
        return end;
    }

    private int suffixPrefix(String marker) {
        for (int length = Math.min(marker.length() - 1, this.buffer.length()); length > 0; length--)
            if (marker.startsWith(this.buffer.substring(this.buffer.length() - length))) return length;
        return 0;
    }

    private static ToolCallParser.MalformedToolCallException malformed(String message) {
        return new ToolCallParser.MalformedToolCallException(message);
    }
}
