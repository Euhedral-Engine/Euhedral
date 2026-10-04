package io.euhedral_execution.inference.api.chat;

import java.math.BigInteger;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.json.JsonMapper;

/// Splits decoded model output into assistant text and the checkpoint's `<tool_call>` blocks.
///
/// The format is the inverse of the template's assistant rendering:
/// `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>`.
/// A block is parsed only once its closing tag has arrived, so every emitted call is complete and its
/// arguments are a valid JSON object. Text that could still begin a `<tool_call>` tag, and whitespace that
/// could still precede one, is held back until the next chunk decides it.
/// Confined to the generation thread.
final class ToolCallParser {
    private static final String CALL_START = QwenChatTemplate.TOOL_CALL_START;
    private static final String CALL_END = QwenChatTemplate.TOOL_CALL_END;
    private static final String FUNCTION_START = "<function=";
    private static final String FUNCTION_END = "</function>";
    private static final String PARAMETER_START = "<parameter=";
    private static final String PARAMETER_END = "</parameter>";
    private static final JsonMapper JSON = JsonMapper.shared();

    /// Receives parsed output in generation order.
    interface Output {
        void content(String text);

        void toolCall(String name, String arguments);
    }

    /// The model's output claims to be a tool call but is not one this request can accept.
    static final class MalformedToolCallException extends Exception {
        MalformedToolCallException(String message) {
            super(message);
        }
    }

    private enum State {
        CONTENT,
        CALL,
        BETWEEN_CALLS,
        // The single call parallel_tool_calls=false allows is complete; everything after it is discarded.
        DONE
    }

    private final Map<String, FunctionTool> callable;
    private final boolean parallel;
    private final StringBuilder buffer = new StringBuilder();
    private State state = State.CONTENT;
    private int calls;

    /// Starts after the request's generation prefix, which the prompt already ends with.
    ToolCallParser(ToolCalling tools) {
        this.callable = new LinkedHashMap<>();
        for (FunctionTool tool : tools.callable()) this.callable.put(tool.name(), tool);
        this.parallel = tools.parallel();
        this.buffer.append(tools.generationPrefix());
    }

    int calls() {
        return this.calls;
    }

    void accept(String chunk, Output output) throws MalformedToolCallException {
        this.buffer.append(chunk);
        drain(output);
    }

    /// Releases held-back text once generation has ended. After a stop token an unfinished call is
    /// malformed; after an exhausted budget it is truncated output and is dropped.
    void finish(boolean stopTokenReached, Output output) throws MalformedToolCallException {
        drain(output);
        if (this.state == State.CONTENT) {
            if (!this.buffer.isEmpty()) output.content(this.buffer.toString());
        } else if (stopTokenReached && (this.state == State.CALL || !this.buffer.isEmpty())) {
            throw malformed("generation ended inside a tool call");
        }
        this.buffer.setLength(0);
    }

    private void drain(Output output) throws MalformedToolCallException {
        while (true) {
            switch (this.state) {
                case CONTENT -> {
                    int start = this.buffer.indexOf(CALL_START);
                    if (start < 0) {
                        int safe = trailingSpaceStart(this.buffer.length() - heldTagPrefix());
                        if (safe > 0) output.content(this.buffer.substring(0, safe));
                        this.buffer.delete(0, safe);
                        return;
                    }
                    // The template renders content|trim before calls, so trailing space before one is dropped.
                    int end = trailingSpaceStart(start);
                    if (end > 0) output.content(this.buffer.substring(0, end));
                    this.buffer.delete(0, start + CALL_START.length());
                    this.state = State.CALL;
                }
                case CALL -> {
                    int end = callEnd();
                    if (end < 0) return;
                    String block = this.buffer.substring(0, end);
                    this.buffer.delete(0, end + CALL_END.length());
                    parseBlock(block, output);
                    this.state = this.parallel ? State.BETWEEN_CALLS : State.DONE;
                }
                case BETWEEN_CALLS -> {
                    int text = 0;
                    while (text < this.buffer.length() && Character.isWhitespace(this.buffer.charAt(text))) text++;
                    this.buffer.delete(0, text);
                    if (this.buffer.isEmpty()) return;
                    if (this.buffer.indexOf(CALL_START) == 0) {
                        this.buffer.delete(0, CALL_START.length());
                        this.state = State.CALL;
                    } else if (CALL_START.startsWith(this.buffer.toString())) {
                        return;
                    } else {
                        throw new MalformedToolCallException("the model produced text after a tool call");
                    }
                }
                case DONE -> {
                    this.buffer.setLength(0);
                    return;
                }
            }
        }
    }

    /// The template closes the function before the call. Ignore `</tool_call>` inside a parameter value.
    private int callEnd() {
        int end = this.buffer.indexOf(CALL_END);
        while (end >= 0) {
            // Without any function opening, this cannot be parameter text; reject the closed block now.
            if (this.buffer.indexOf(FUNCTION_START) < 0) return end;
            int before = end;
            while (before > 0 && Character.isWhitespace(this.buffer.charAt(before - 1))) before--;
            if (before >= FUNCTION_END.length()
                    && this.buffer
                            .substring(before - FUNCTION_END.length(), before)
                            .equals(FUNCTION_END)) return end;
            end = this.buffer.indexOf(CALL_END, end + CALL_END.length());
        }
        return -1;
    }

    /// Start of the whitespace run ending at `end`; that text is held until it is known not to precede a call.
    private int trailingSpaceStart(int end) {
        while (end > 0 && QwenChatTemplate.isPythonSpace(this.buffer.charAt(end - 1))) end--;
        return end;
    }

    /// Length of the longest buffer suffix that is a proper prefix of `<tool_call>`.
    private int heldTagPrefix() {
        int limit = Math.min(CALL_START.length() - 1, this.buffer.length());
        for (int length = limit; length > 0; length--) {
            if (CALL_START.startsWith(this.buffer.substring(this.buffer.length() - length))) return length;
        }
        return 0;
    }

    private void parseBlock(String block, Output output) throws MalformedToolCallException {
        int position = skipWhitespace(block, 0);
        if (!block.startsWith(FUNCTION_START, position)) throw malformed("a tool call has no <function=...> tag");
        int nameEnd = block.indexOf('>', position);
        if (nameEnd < 0) throw malformed("a tool call has an unterminated <function=...> tag");
        String name = block.substring(position + FUNCTION_START.length(), nameEnd);
        FunctionTool tool = this.callable.get(name);
        if (tool == null) throw malformed("the model called a function that was not offered: " + name);
        position = nameEnd + 1;
        Map<String, Object> arguments = new LinkedHashMap<>();
        while (true) {
            position = skipWhitespace(block, position);
            if (block.startsWith(FUNCTION_END, position)) {
                if (skipWhitespace(block, position + FUNCTION_END.length()) != block.length())
                    throw malformed("a tool call has text after </function>");
                break;
            }
            if (!block.startsWith(PARAMETER_START, position))
                throw malformed("a tool call for " + name + " has text outside its parameters");
            int keyEnd = block.indexOf('>', position);
            if (keyEnd < 0) throw malformed("a tool call for " + name + " has an unterminated <parameter=...> tag");
            String key = block.substring(position + PARAMETER_START.length(), keyEnd);
            int valueEnd = parameterEnd(block, keyEnd + 1);
            if (valueEnd < 0) throw malformed("parameter " + key + " of " + name + " is not closed");
            if (arguments.containsKey(key)) throw malformed("parameter " + key + " of " + name + " was repeated");
            if (!tool.acceptsParameter(key)) throw malformed(name + " has no parameter " + key);
            String raw = stripOneNewline(block.substring(keyEnd + 1, valueEnd));
            arguments.put(key, argumentValue(tool, key, raw));
            position = valueEnd + PARAMETER_END.length();
        }
        for (String required : tool.requiredParameters()) {
            if (!arguments.containsKey(required))
                throw malformed("required parameter " + required + " of " + name + " is missing");
        }
        this.calls++;
        output.toolCall(name, JSON.writeValueAsString(arguments));
    }

    /// The first `</parameter>` followed by another parameter or `</function>`; values may contain the tag.
    private static int parameterEnd(String block, int from) {
        int end = block.indexOf(PARAMETER_END, from);
        while (end >= 0) {
            int next = skipWhitespace(block, end + PARAMETER_END.length());
            if (block.startsWith(PARAMETER_START, next) || block.startsWith(FUNCTION_END, next)) return end;
            end = block.indexOf(PARAMETER_END, end + 1);
        }
        return -1;
    }

    /// Inverts the template: strings render verbatim, every other value as JSON. The schema's declared
    /// types decide between the two readings; an untyped parameter takes the JSON reading when it parses.
    private static Object argumentValue(FunctionTool tool, String key, String raw) throws MalformedToolCallException {
        Set<String> types = tool.parameterTypes(key);
        if (types.equals(Set.of("string"))) return raw;
        Object parsed;
        try {
            parsed = JSON.readValue(raw, Object.class);
        } catch (JacksonException notJson) {
            if (types.isEmpty() || types.contains("string")) return raw;
            throw malformed("parameter " + key + " of " + tool.name() + " is not valid JSON of type " + types);
        }
        // Out-of-range literals parse as infinities, which no JSON arguments string can carry.
        if (!isFinite(parsed))
            throw malformed("parameter " + key + " of " + tool.name() + " has a number outside the double range");
        String type = jsonType(parsed);
        // A JSON string literal is never how the template renders a string, so keep the model's exact text.
        if (type.equals("string") && (types.isEmpty() || types.contains("string"))) return raw;
        if (types.isEmpty() || types.contains(type) || (type.equals("integer") && types.contains("number")))
            return parsed;
        if (types.contains("string")) return raw;
        throw malformed("parameter " + key + " of " + tool.name() + " is not of type " + types);
    }

    private static boolean isFinite(Object value) {
        return switch (value) {
            case Double number -> Double.isFinite(number);
            case List<?> list -> list.stream().allMatch(ToolCallParser::isFinite);
            case Map<?, ?> map -> map.values().stream().allMatch(ToolCallParser::isFinite);
            case null, default -> true;
        };
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

    private static String stripOneNewline(String value) {
        int start = value.startsWith("\n") ? 1 : 0;
        int end = value.length() > start && value.endsWith("\n") ? value.length() - 1 : value.length();
        return value.substring(start, end);
    }

    private static int skipWhitespace(String text, int position) {
        while (position < text.length() && Character.isWhitespace(text.charAt(position))) position++;
        return position;
    }

    private static MalformedToolCallException malformed(String message) {
        return new MalformedToolCallException(message);
    }
}
