package io.euhedral_execution.inference.api.chat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/// Renders conversations in the ChatML-with-think format of the loaded Qwen checkpoint.
///
/// The checkpoint's Jinja template is the source of truth, but it relies on Python-specific Jinja
/// (reverse slicing, `str.startswith`, namespaces) that Java template engines do not reproduce. This class
/// therefore implements the template's text branches with `add_generation_prompt=true`: system/user/assistant
/// text, the thinking controls (`enable_thinking`, `reasoning_effort`, `preserve_thinking`) and replayed
/// `reasoning_content`, the tool-definition system block, assistant `<tool_call>` XML, and grouped
/// `<tool_response>` turns. It refuses at startup any template that no longer contains the fragments it
/// reproduces. Golden tests compare its output against the template rendered by jinja2
/// (`tools/render_qwen_chat_template_golden.py`).
public final class QwenChatTemplate {
    public static final String IM_START = "<|im_start|>";
    public static final String IM_END = "<|im_end|>";
    public static final String THINK_START = "<think>";
    public static final String THINK_END = "</think>";
    public static final String TOOL_CALL_START = "<tool_call>";
    public static final String TOOL_CALL_END = "</tool_call>";
    public static final String TOOL_RESPONSE_START = "<tool_response>";
    public static final String TOOL_RESPONSE_END = "</tool_response>";
    private static final Pattern CONTROL_TOKEN_OPEN =
            Pattern.compile("<(?=\\|[^<>\\s]+\\|>|/?(?:tool_call|tool_response|think|tts_[a-z_]+)>)");

    /// The tool-calling format instructions exactly as the Jinja string literal spells them (`\n` escapes).
    private static final String TOOL_INSTRUCTIONS_SOURCE =
            "\\n\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n"
                    + "<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n"
                    + "</parameter>\\n<parameter=example_parameter_2>\\nThis is the value for the second parameter\\n"
                    + "that can span\\nmultiple lines\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\n"
                    + "Reminder:\\n"
                    + "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n"
                    + "- Required parameters MUST be specified\\n"
                    + "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n"
                    + "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n"
                    + "</IMPORTANT>";
    private static final String TOOL_INSTRUCTIONS = TOOL_INSTRUCTIONS_SOURCE.replace("\\n", "\n");
    /// The template's `reasoning_instructions` for each effort it names; `medium` has none.
    private static final String XHIGH_INSTRUCTIONS = "Reasoning effort is set to xhigh. Please think carefully through"
            + " the task, validate key assumptions, consider plausible alternatives, and prioritize correctness,"
            + " consistency, and clarity in the final answer.";
    private static final String LOW_INSTRUCTIONS = "Reasoning effort is set to low. Keep your thinking brief and"
            + " focused, moving directly to the conclusion without unnecessary elaboration.";
    private static final String JSON_TOOL_INSTRUCTIONS = "\n\nIf a function is needed, reply ONLY with one JSON object"
            + " in this exact form: {\"tool_calls\":[{\"name\":\"function_name\",\"arguments\":{\"parameter\":\"value\"}}]}."
            + " Use valid JSON with arguments as an object, function names from the tools above. Tool results"
            + " arrive as a JSON tool_results array in call order."
            + " If no function is needed, reply ONLY with {\"content\":\"answer\"}, putting the full answer"
            + " in the JSON string. Never emit markdown, XML, or text outside the JSON object.";

    /// Exact Jinja source fragments whose behavior `render` reproduces. `\n` is the Jinja escape.
    private static final List<String> REQUIRED_FRAGMENTS = List.of(
            "{%- if tools and tools is iterable and tools is not mapping %}",
            "{{- \"# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>\" }}",
            "{{- \"\\n\" }}\n        {{- tool | tojson }}",
            "{{- \"\\n</tools>\" }}",
            "{{- '" + TOOL_INSTRUCTIONS_SOURCE + "' }}",
            "{{- '\\n\\n' + content }}",
            "{%- if messages[0].role == 'system' %}",
            "render_content(messages[0].content, false, true)|trim",
            "'<|im_start|>system\\n' + (reasoning_instructions + '\\n\\n' if reasoning_instructions else '')"
                    + "  + content + '<|im_end|>\\n'",
            "raise_exception('No user query found in messages.')",
            "render_content(message.content, true)|trim",
            "raise_exception('System message must be at the beginning.')",
            "'<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n'",
            "'<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content",
            "{%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}",
            "{{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}",
            "{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}",
            "{{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}",
            "{%- for args_name, args_value in tool_call.arguments|items %}",
            "{{- '<parameter=' + args_name + '>\\n' }}",
            "{%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}",
            "{{- '\\n</parameter>\\n' }}",
            "{{- '</function>\\n</tool_call>' }}",
            "{%- if loop.previtem and loop.previtem.role != \"tool\" %}",
            "{{- '<|im_start|>user' }}",
            "{{- '\\n<tool_response>\\n' }}",
            "{{- '\\n</tool_response>' }}",
            "{%- if not loop.last and loop.nextitem.role != \"tool\" %}",
            "{%- if enable_thinking is undefined or enable_thinking is true %}",
            "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}",
            "{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}",
            "{%- if resolved_reasoning_effort == 'xhigh' %}",
            "{%- set reasoning_instructions = '" + XHIGH_INSTRUCTIONS + "' %}",
            "{%- elif resolved_reasoning_effort == 'low' %}",
            "{%- set reasoning_instructions = '" + LOW_INSTRUCTIONS + "' %}",
            "{{- reasoning_instructions + '\\n\\n' }}",
            "{{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}",
            "{%- if message.reasoning_content is string %}",
            "{%- set reasoning_content = reasoning_content|trim %}",
            "{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}",
            "{{- '<|im_start|>' + message.role + '\\n' + content }}",
            "{%- if enable_thinking is defined and enable_thinking is false %}",
            "{{- '<think>\\n' }}",
            "{{- '<|im_start|>assistant\\n' }}",
            "{{- '<think>\\n\\n</think>\\n\\n' }}");

    private QwenChatTemplate() {}

    public enum Role {
        SYSTEM,
        USER,
        ASSISTANT,
        TOOL
    }

    /// One turn. `content` is the already-concatenated text of the message's text parts; only assistant
    /// turns carry `toolCalls`, rendered after the content in list order, and `reasoning`, the replayed
    /// `reasoning_content` of an earlier response (empty when the client sent none).
    public record Turn(Role role, String content, List<ToolCall> toolCalls, String reasoning) {
        public Turn {
            Objects.requireNonNull(role, "role");
            Objects.requireNonNull(content, "content");
            Objects.requireNonNull(reasoning, "reasoning");
            toolCalls = List.copyOf(toolCalls);
            if (!toolCalls.isEmpty() && role != Role.ASSISTANT)
                throw new IllegalArgumentException("only assistant turns carry tool calls");
            if (!reasoning.isEmpty() && role != Role.ASSISTANT)
                throw new IllegalArgumentException("only assistant turns carry reasoning");
        }

        public Turn(Role role, String content, List<ToolCall> toolCalls) {
            this(role, content, toolCalls, "");
        }

        public Turn(Role role, String content) {
            this(role, content, List.of(), "");
        }
    }

    /// The template's thinking variables. `effort` null is `enable_thinking=false`; otherwise thinking is on
    /// with the template's `reasoning_effort`. `preserveThinking` false drops the think block of assistant
    /// turns before the last user query, as `preserve_thinking=false` does.
    public record Thinking(Effort effort, boolean preserveThinking) {
        /// What the template renders when the caller sets none of its thinking variables.
        public static final Thinking MODEL_DEFAULT = new Thinking(Effort.XHIGH, true);

        public static final Thinking DISABLED = new Thinking(null, true);

        public boolean enabled() {
            return this.effort != null;
        }
    }

    /// The template's `reasoning_effort` values. Only `xhigh` and `low` add system instructions.
    public enum Effort {
        LOW,
        MEDIUM,
        XHIGH;

        String instructions() {
            return switch (this) {
                case LOW -> LOW_INSTRUCTIONS;
                case MEDIUM -> "";
                case XHIGH -> XHIGH_INSTRUCTIONS;
            };
        }
    }

    /// A function call as the template reads it: `arguments` is the parsed JSON object, in its key order.
    public record ToolCall(String name, Map<String, Object> arguments) {
        public ToolCall {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(arguments, "arguments");
        }
    }

    /// Thrown for conversations the checkpoint template itself rejects; messages mirror the template.
    public static final class InvalidConversationException extends IllegalArgumentException {
        InvalidConversationException(String message) {
            super(message);
        }
    }

    /// Loads `chat_template.jinja`, falling back to `tokenizer_config.json`'s `chat_template` string.
    public static QwenChatTemplate load(Path tokenizerDirectory) throws IOException {
        Path jinja = tokenizerDirectory.resolve("chat_template.jinja");
        if (Files.isRegularFile(jinja)) return fromTemplateSource(Files.readString(jinja));
        Path config = tokenizerDirectory.resolve("tokenizer_config.json");
        if (Files.isRegularFile(config)) {
            JsonNode template =
                    JsonMapper.shared().readTree(Files.readString(config)).path("chat_template");
            if (template.isString()) return fromTemplateSource(template.asString());
        }
        throw new IOException("checkpoint has no chat template in " + tokenizerDirectory);
    }

    /// Verifies that `source` is the Qwen template this formatter reproduces.
    public static QwenChatTemplate fromTemplateSource(String source) throws IOException {
        Objects.requireNonNull(source, "source");
        String normalizedSource = source.replace("\r\n", "\n");
        for (String fragment : REQUIRED_FRAGMENTS) {
            if (!normalizedSource.contains(fragment))
                throw new IOException(
                        "unsupported checkpoint chat template; the Qwen formatter no longer matches fragment: "
                                + fragment);
        }
        return new QwenChatTemplate();
    }

    /// Control tokens this formatter emits; the tokenizer must encode each as one control token.
    public List<String> controlTokens() {
        return List.of(
                IM_START,
                IM_END,
                THINK_START,
                THINK_END,
                TOOL_CALL_START,
                TOOL_CALL_END,
                TOOL_RESPONSE_START,
                TOOL_RESPONSE_END);
    }

    /// Renders the conversation followed by the assistant generation prompt.
    public String render(List<Turn> turns, Thinking thinking) {
        return render(turns, List.of(), thinking);
    }

    /// Renders the conversation with tool definitions, each a JSON object serialized as the template's
    /// `tool | tojson`. An empty list renders exactly as the template does without tools.
    public String render(List<Turn> turns, List<Map<String, Object>> tools, Thinking thinking) {
        return render(turns, tools, thinking, false, "");
    }

    /// Uses JSON tool-call envelopes instead of unescaped XML parameter values.
    public String renderJsonTools(List<Turn> turns, ToolCalling calling, Thinking thinking) {
        List<Map<String, Object>> tools = calling.promptTools();
        if (tools.isEmpty()) throw new IllegalArgumentException("JSON tool rendering requires offered tools");
        String constraint =
                switch (calling.choice()) {
                    case REQUIRED ->
                        " For this turn you MUST call one of the offered functions; do not answer directly.";
                    case FUNCTION -> " For this turn you must call " + calling.function() + "; do not answer directly.";
                    case AUTO -> "";
                    case NONE -> throw new IllegalArgumentException("none cannot call functions");
                };
        String callCount = calling.parallel()
                ? " Multiple calls belong in the same array."
                : " At most one function call is allowed in tool_calls.";
        return render(turns, tools, thinking, true, callCount + constraint);
    }

    private String render(
            List<Turn> turns,
            List<Map<String, Object>> tools,
            Thinking thinking,
            boolean jsonTools,
            String constraint) {
        Objects.requireNonNull(turns, "turns");
        Objects.requireNonNull(tools, "tools");
        Objects.requireNonNull(thinking, "thinking");
        if (turns.isEmpty()) throw new InvalidConversationException("No messages provided.");
        StringBuilder prompt = new StringBuilder();
        String instructions = thinking.enabled() ? thinking.effort().instructions() : "";
        String system =
                turns.getFirst().role() == Role.SYSTEM ? trim(turns.getFirst().content()) : "";
        if (!tools.isEmpty()) {
            prompt.append(IM_START).append("system\n");
            if (!instructions.isEmpty()) prompt.append(instructions).append("\n\n");
            prompt.append("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for (Map<String, Object> tool : tools)
                prompt.append('\n').append(escapeControlTokens(PythonJson.dumps(tool)));
            prompt.append("\n</tools>");
            if (!jsonTools) prompt.append(TOOL_INSTRUCTIONS);
            if (!system.isEmpty()) prompt.append("\n\n").append(system);
            if (jsonTools) prompt.append(JSON_TOOL_INSTRUCTIONS).append(constraint);
            prompt.append(IM_END).append('\n');
        } else if (!system.isEmpty() || !instructions.isEmpty()) {
            prompt.append(IM_START).append("system\n").append(instructions);
            if (!system.isEmpty() && !instructions.isEmpty()) prompt.append("\n\n");
            prompt.append(system).append(IM_END).append('\n');
        }
        int lastQuery = lastUserQuery(turns);
        if (lastQuery < 0) throw new InvalidConversationException("No user query found in messages.");
        for (int index = 0; index < turns.size(); index++) {
            Turn turn = turns.get(index);
            String content = trim(turn.content());
            switch (turn.role()) {
                case SYSTEM -> {
                    if (index != 0) throw new InvalidConversationException("System message must be at the beginning.");
                }
                case USER ->
                    prompt.append(IM_START)
                            .append("user\n")
                            .append(escapeControlTokens(content))
                            .append(IM_END)
                            .append('\n');
                // Every assistant turn keeps its think block unless preserve_thinking=false drops those before the
                // last user query.
                case ASSISTANT -> {
                    prompt.append(IM_START).append("assistant\n");
                    if (thinking.preserveThinking() || index > lastQuery)
                        prompt.append(THINK_START)
                                .append('\n')
                                .append(escapeControlTokens(trim(turn.reasoning())))
                                .append('\n')
                                .append(THINK_END)
                                .append("\n\n");
                    prompt.append(escapeControlTokens(content));
                    if (jsonTools) appendJsonToolCalls(turn.toolCalls(), !content.isEmpty(), prompt);
                    else appendToolCalls(turn.toolCalls(), !content.isEmpty(), prompt);
                    prompt.append(IM_END).append('\n');
                }
                // Consecutive tool results share one user turn; the template opens it only after a non-tool turn.
                case TOOL -> {
                    if (jsonTools) {
                        if (index > 0 && turns.get(index - 1).role() != Role.TOOL)
                            prompt.append(IM_START).append("user\n{\"tool_results\": [");
                        else prompt.append(", ");
                        prompt.append(escapeControlTokens(PythonJson.dumps(content)));
                        if (index == turns.size() - 1 || turns.get(index + 1).role() != Role.TOOL)
                            prompt.append("]}").append(IM_END).append('\n');
                    } else {
                        if (index > 0 && turns.get(index - 1).role() != Role.TOOL)
                            prompt.append(IM_START).append("user");
                        prompt.append('\n')
                                .append(TOOL_RESPONSE_START)
                                .append('\n')
                                .append(escapeControlTokens(content))
                                .append('\n')
                                .append(TOOL_RESPONSE_END);
                        if (index == turns.size() - 1 || turns.get(index + 1).role() != Role.TOOL)
                            prompt.append(IM_END).append('\n');
                    }
                }
            }
        }
        prompt.append(IM_START).append("assistant\n").append(THINK_START).append('\n');
        if (!thinking.enabled()) prompt.append('\n').append(THINK_END).append("\n\n");
        return prompt.toString();
    }

    /// String arguments render verbatim; every other JSON value renders as `tojson`.
    private static void appendToolCalls(List<ToolCall> calls, boolean afterContent, StringBuilder prompt) {
        for (int index = 0; index < calls.size(); index++) {
            ToolCall call = calls.get(index);
            if (index > 0 || afterContent) prompt.append(index == 0 ? "\n\n" : "\n");
            prompt.append(TOOL_CALL_START)
                    .append("\n<function=")
                    .append(escapeControlTokens(call.name()))
                    .append(">\n");
            for (var argument : call.arguments().entrySet()) {
                Object value = argument.getValue();
                prompt.append("<parameter=")
                        .append(escapeControlTokens(argument.getKey()))
                        .append(">\n")
                        .append(escapeControlTokens(value instanceof String text ? text : PythonJson.dumps(value)))
                        .append("\n</parameter>\n");
            }
            prompt.append("</function>\n").append(TOOL_CALL_END);
        }
    }

    private static void appendJsonToolCalls(List<ToolCall> calls, boolean afterContent, StringBuilder prompt) {
        if (calls.isEmpty()) return;
        if (afterContent) prompt.append("\n\n");
        List<Map<String, Object>> entries = new ArrayList<>(calls.size());
        for (ToolCall call : calls) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("name", call.name());
            entry.put("arguments", call.arguments());
            entries.add(entry);
        }
        prompt.append(escapeControlTokens(PythonJson.dumps(Map.of("tool_calls", entries))));
    }

    /// Protect untrusted text from checkpoint control-token recognition without changing ordinary
    /// angle-bracket content. All added-token spellings in the supported Qwen checkpoint match this form.
    private static String escapeControlTokens(String text) {
        return CONTROL_TOKEN_OPEN.matcher(text).replaceAll(Matcher.quoteReplacement("\\u003c"));
    }

    /// Jinja's `trim` is Python `str.strip`, whose whitespace set includes NBSP and NEL unlike `String.strip`.
    static String trim(String text) {
        int start = 0;
        int end = text.length();
        while (start < end && isPythonSpace(text.charAt(start))) start++;
        while (end > start && isPythonSpace(text.charAt(end - 1))) end--;
        return text.substring(start, end);
    }

    static boolean isPythonSpace(char value) {
        return Character.isWhitespace(value) || Character.isSpaceChar(value) || value == '\u0085';
    }

    /// Mirrors the template's search for the last user turn that is not a wrapped tool response; -1 when
    /// there is none.
    private static int lastUserQuery(List<Turn> turns) {
        for (int index = turns.size() - 1; index >= 0; index--) {
            Turn turn = turns.get(index);
            if (turn.role() != Role.USER) continue;
            String content = trim(turn.content());
            if (!(content.startsWith("<tool_response>") && content.endsWith("</tool_response>"))) return index;
        }
        return -1;
    }
}
