package io.euhedral_execution.inference.api.responses;

import io.euhedral_execution.inference.api.chat.ApiException;
import io.euhedral_execution.inference.api.chat.Conversation;
import io.euhedral_execution.inference.api.chat.ConversationPlanner;
import io.euhedral_execution.inference.api.chat.FunctionTool;
import io.euhedral_execution.inference.api.chat.QwenChatTemplate;
import io.euhedral_execution.inference.api.chat.Reasoning;
import io.euhedral_execution.inference.api.chat.ResponseFormat;
import io.euhedral_execution.inference.api.chat.SamplingDefaults;
import io.euhedral_execution.inference.api.chat.ToolCalling;
import io.euhedral_execution.inference.api.chat.ToolResults;
import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ThreadLocalRandom;
import org.springframework.stereotype.Component;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.json.JsonMapper;

/// Validates an OpenAI Responses request and maps it to the shared [Conversation].
///
/// The server keeps no state between requests: a conversation is sent whole as `input`, `previous_response_id` and
/// `conversation` are refused, and `store` has no effect. Reasoning the server returned comes back as `reasoning`
/// items, by their `content` or, with `include: ["reasoning.encrypted_content"]`, their opaque `encrypted_content`.
@Component
public class ResponsesMapper {
    /// The prefix of the opaque reasoning token: its UTF-8 text in URL-safe base64, not encrypted.
    static final String ENCRYPTED_PREFIX = "euhedral:";

    private static final JsonMapper JSON = JsonMapper.shared();
    private static final Set<String> IGNORED =
            Set.of("user", "metadata", "store", "service_tier", "safety_identifier", "prompt_cache_key", "truncation");
    private static final Set<String> REFUSED =
            Set.of("previous_response_id", "conversation", "background", "prompt", "max_tool_calls", "top_logprobs");

    private final InferenceBackend backend;
    private final SamplingDefaults samplingDefaults;
    private final ConversationPlanner planner;

    public ResponsesMapper(InferenceBackend backend, SamplingDefaults samplingDefaults, ConversationPlanner planner) {
        this.backend = backend;
        this.samplingDefaults = samplingDefaults;
        this.planner = planner;
    }

    /// Validates and plans a request on the backend's workers; the future fails with an [ApiException].
    public CompletableFuture<ConversationPlanner.Planned> planAsync(ResponsesRequest request) {
        return this.planner.planAsync(() -> map(request));
    }

    ConversationPlanner.Mapped map(ResponsesRequest request) {
        if (request == null) throw ApiException.invalidRequest("Request body is required.", null);
        for (var field : request.otherFields().entrySet()) {
            String name = field.getKey();
            Object value = field.getValue();
            if (name.equals("truncation") && value != null && !"disabled".equals(value))
                throw ApiException.unsupportedParameter("truncation");
            if (name.equals("background") && Boolean.FALSE.equals(value)) continue;
            if (IGNORED.contains(name)) continue;
            if (REFUSED.contains(name)) {
                if (value == null) continue;
                throw ApiException.unsupportedParameter(name);
            }
            throw ApiException.unrecognizedArgument(name);
        }
        if (request.model() == null || request.model().isBlank())
            throw ApiException.invalidRequest("You must provide a model parameter.", "model");
        if (!request.model().equals(this.backend.modelId())) throw ApiException.modelNotFound(request.model());
        boolean includeEncrypted = include(request.include());
        ToolCalling tools = tools(request.tools(), request.toolChoice(), request.parallelToolCalls());
        Map<?, ?> reasoningOptions = reasoningOptions(request.reasoning());
        QwenChatTemplate.Thinking thinking =
                Reasoning.thinking(reasoningOptions == null ? null : reasoningOptions.get("effort"), null);
        ResponseFormat format = format(request.text());
        var conversation = new Conversation(
                turns(request.instructions(), request.input()),
                tools,
                thinking,
                Integer.MAX_VALUE,
                format,
                List.of(),
                sampling(request),
                request.maxOutputTokens(),
                "max_output_tokens",
                Boolean.TRUE.equals(request.stream()));
        return new ConversationPlanner.Mapped(
                conversation,
                new ResponsesResponder(
                        Instant.now().getEpochSecond(),
                        this.backend.modelId(),
                        includeEncrypted,
                        echo(request, tools, thinking)));
    }

    // Input.

    private static List<QwenChatTemplate.Turn> turns(String instructions, Object input) {
        if (input == null) throw ApiException.invalidRequest("'input' is required.", "input");
        List<Object> items = input instanceof String text
                ? List.of(Map.of("type", "message", "role", "user", "content", text))
                : input instanceof List<?> list ? new ArrayList<>(list) : null;
        if (items == null) throw ApiException.invalidRequest("'input' must be a string or a list of items.", "input");
        if (items.isEmpty()) throw ApiException.invalidRequest("'input' must not be empty.", "input");
        List<QwenChatTemplate.Turn> turns = new ArrayList<>();
        StringBuilder system = new StringBuilder(instructions == null ? "" : instructions);
        boolean leading = true;
        Assistant assistant = null;
        ToolResults pending = null;
        for (int index = 0; index < items.size(); index++) {
            String path = "input[" + index + "]";
            if (!(items.get(index) instanceof Map<?, ?> item))
                throw ApiException.invalidRequest("Each input item must be an object.", path);
            String type = item.get("type") instanceof String name ? name : item.containsKey("role") ? "message" : null;
            if (type == null) throw ApiException.invalidRequest("Input item 'type' is required.", path + ".type");
            String role = "message".equals(type) && item.get("role") instanceof String name ? name : null;
            boolean assistantItem =
                    type.equals("reasoning") || type.equals("function_call") || "assistant".equals(role);
            if (!assistantItem && assistant != null) {
                pending = assistant.closeInto(turns);
                assistant = null;
            }
            if (!type.equals("function_call_output") && !assistantItem && pending != null) {
                pending.closeInto(turns);
                pending = null;
            }
            switch (type) {
                case "message" -> {
                    checkKeys(item, Set.of("type", "role", "content", "id", "status"), path);
                    if (role == null) throw ApiException.invalidRequest("Message 'role' is required.", path + ".role");
                    String text = content(item.get("content"), path + ".content");
                    switch (role) {
                        case "system", "developer" -> {
                            if (!leading)
                                throw ApiException.invalidRequest(
                                        "System and developer messages must come before the conversation.", path);
                            if (!system.isEmpty()) system.append("\n\n");
                            system.append(text);
                        }
                        case "user" -> turns.add(new QwenChatTemplate.Turn(QwenChatTemplate.Role.USER, text));
                        case "assistant" -> {
                            if (pending != null) {
                                pending.closeInto(turns);
                                pending = null;
                            }
                            if (assistant == null) assistant = new Assistant(path);
                            assistant.text.append(text);
                        }
                        default ->
                            throw ApiException.invalidRequest("Invalid message role '" + role + "'.", path + ".role");
                    }
                }
                case "reasoning" -> {
                    checkKeys(item, Set.of("type", "id", "summary", "content", "encrypted_content", "status"), path);
                    if (pending != null) {
                        pending.closeInto(turns);
                        pending = null;
                    }
                    if (assistant == null) assistant = new Assistant(path);
                    assistant.reasoning.append(reasoning(item, path));
                }
                case "function_call" -> {
                    checkKeys(item, Set.of("type", "id", "call_id", "name", "arguments", "status"), path);
                    if (pending != null) {
                        pending.closeInto(turns);
                        pending = null;
                    }
                    if (assistant == null) assistant = new Assistant(path);
                    if (!(item.get("call_id") instanceof String id) || id.isEmpty())
                        throw ApiException.invalidRequest("'call_id' is required.", path + ".call_id");
                    if (assistant.ids.contains(id))
                        throw ApiException.invalidRequest("Call IDs must be unique within a turn.", path + ".call_id");
                    if (!(item.get("name") instanceof String name) || !ToolCalling.isFunctionName(name))
                        throw ApiException.invalidRequest(
                                "Function names must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes.",
                                path + ".name");
                    assistant.ids.add(id);
                    assistant.calls.add(new QwenChatTemplate.ToolCall(name, arguments(item.get("arguments"), path)));
                }
                case "function_call_output" -> {
                    checkKeys(item, Set.of("type", "id", "call_id", "output", "status"), path);
                    if (pending == null)
                        throw ApiException.invalidRequest(
                                "A function_call_output must answer a function_call of the preceding turn.", path);
                    pending.answer(
                            item.get("call_id"),
                            content(item.get("output"), path + ".output"),
                            path + ".call_id",
                            "call_id");
                }
                default -> throw ApiException.unsupportedParameter(path + ".type=" + type);
            }
            if (!type.equals("message") || !("system".equals(role) || "developer".equals(role))) leading = false;
        }
        if (assistant != null) pending = assistant.closeInto(turns);
        if (pending != null) pending.closeInto(turns);
        if (!system.isEmpty())
            turns.addFirst(new QwenChatTemplate.Turn(QwenChatTemplate.Role.SYSTEM, system.toString()));
        return turns;
    }

    /// The items of one assistant turn: reasoning, text and calls, in that order in the prompt.
    private static final class Assistant {
        final String path;
        final StringBuilder reasoning = new StringBuilder();
        final StringBuilder text = new StringBuilder();
        final List<QwenChatTemplate.ToolCall> calls = new ArrayList<>();
        final List<String> ids = new ArrayList<>();

        Assistant(String path) {
            this.path = path;
        }

        /// Adds the turn; returns the results its calls are owed, or null.
        ToolResults closeInto(List<QwenChatTemplate.Turn> turns) {
            turns.add(new QwenChatTemplate.Turn(
                    QwenChatTemplate.Role.ASSISTANT, this.text.toString(), this.calls, this.reasoning.toString()));
            return this.calls.isEmpty() ? null : new ToolResults(this.ids, this.path, "call_id");
        }
    }

    /// A message's or output's text: a string, or `input_text`/`output_text` parts concatenated.
    private static String content(Object content, String path) {
        if (content == null) return "";
        if (content instanceof String text) return text;
        if (!(content instanceof List<?> parts))
            throw ApiException.invalidRequest("Content must be a string or a list of content parts.", path);
        StringBuilder text = new StringBuilder();
        for (int index = 0; index < parts.size(); index++) {
            String partPath = path + "[" + index + "]";
            if (!(parts.get(index) instanceof Map<?, ?> part))
                throw ApiException.invalidRequest("Content part must be an object.", partPath);
            Object type = part.get("type");
            if (!"input_text".equals(type) && !"output_text".equals(type))
                throw ApiException.unsupportedParameter(partPath + ".type=" + type);
            checkKeys(part, Set.of("type", "text", "annotations", "logprobs"), partPath);
            if (part.get("annotations") instanceof List<?> notes && !notes.isEmpty())
                throw ApiException.unsupportedParameter(partPath + ".annotations");
            if (!(part.get("text") instanceof String partText))
                throw ApiException.invalidRequest("Text content part requires 'text'.", partPath + ".text");
            text.append(partText);
        }
        return text.toString();
    }

    /// The reasoning a `reasoning` item carries: its `reasoning_text` content, else its opaque token, else its summary.
    private static String reasoning(Map<?, ?> item, String path) {
        if (item.get("content") instanceof List<?> parts && !parts.isEmpty()) {
            StringBuilder text = new StringBuilder();
            for (Object part : parts) {
                if (!(part instanceof Map<?, ?> entry)
                        || !"reasoning_text".equals(entry.get("type"))
                        || !(entry.get("text") instanceof String partText))
                    throw ApiException.invalidRequest(
                            "Reasoning content must be reasoning_text parts.", path + ".content");
                text.append(partText);
            }
            return text.toString();
        }
        if (item.get("encrypted_content") instanceof String token) {
            if (!token.startsWith(ENCRYPTED_PREFIX))
                throw ApiException.invalidRequest(
                        "'encrypted_content' was not issued by this server.", path + ".encrypted_content");
            try {
                return new String(
                        Base64.getUrlDecoder().decode(token.substring(ENCRYPTED_PREFIX.length())),
                        StandardCharsets.UTF_8);
            } catch (IllegalArgumentException invalid) {
                throw ApiException.invalidRequest("'encrypted_content' is malformed.", path + ".encrypted_content");
            }
        }
        StringBuilder text = new StringBuilder();
        if (item.get("summary") instanceof List<?> parts)
            for (Object part : parts)
                if (part instanceof Map<?, ?> entry && entry.get("text") instanceof String partText)
                    text.append(partText);
        return text.toString();
    }

    /// The opaque token of a reasoning text, which a client sends back in a later request.
    static String encrypt(String reasoning) {
        return ENCRYPTED_PREFIX
                + Base64.getUrlEncoder().withoutPadding().encodeToString(reasoning.getBytes(StandardCharsets.UTF_8));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> arguments(Object value, String path) {
        if (!(value instanceof String text))
            throw ApiException.invalidRequest("'arguments' must be a JSON string.", path + ".arguments");
        if (text.isEmpty()) return Map.of();
        Object parsed;
        try {
            parsed = JSON.readValue(text, Object.class);
        } catch (JacksonException invalid) {
            throw ApiException.invalidRequest("'arguments' must be valid JSON.", path + ".arguments");
        }
        if (!(parsed instanceof Map<?, ?> object))
            throw ApiException.invalidRequest("'arguments' must encode a JSON object.", path + ".arguments");
        return (Map<String, Object>) object;
    }

    // Options.

    private static boolean include(Object value) {
        if (value == null) return false;
        if (!(value instanceof List<?> list)) throw ApiException.invalidRequest("'include' must be a list.", "include");
        boolean encrypted = false;
        for (Object entry : list) {
            if ("reasoning.encrypted_content".equals(entry)) encrypted = true;
            else throw ApiException.unsupportedParameter("include=" + entry);
        }
        return encrypted;
    }

    private static ToolCalling tools(Object value, Object choice, Boolean parallelToolCalls) {
        List<FunctionTool> functions = new ArrayList<>();
        if (value != null) {
            if (!(value instanceof List<?> list)) throw ApiException.invalidRequest("'tools' must be a list.", "tools");
            Set<String> names = new HashSet<>();
            for (int index = 0; index < list.size(); index++) {
                String path = "tools[" + index + "]";
                if (!(list.get(index) instanceof Map<?, ?> tool))
                    throw ApiException.invalidRequest("Each tool must be an object.", path);
                if (!"function".equals(tool.get("type")))
                    throw ApiException.unsupportedParameter(path + ".type=" + tool.get("type"));
                checkKeys(tool, Set.of("type", "name", "description", "parameters", "strict"), path);
                if (!(tool.get("name") instanceof String name) || !ToolCalling.isFunctionName(name))
                    throw ApiException.invalidRequest(
                            "Function names must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes.",
                            path + ".name");
                if (!names.add(name)) throw ApiException.invalidRequest("Tool names must be unique.", path + ".name");
                Object description = tool.get("description");
                if (description != null && !(description instanceof String))
                    throw ApiException.invalidRequest("'description' must be a string.", path + ".description");
                Object strict = tool.get("strict");
                if (strict != null && !(strict instanceof Boolean))
                    throw ApiException.invalidRequest("'strict' must be a boolean.", path + ".strict");
                // In this API function tools are strict unless they say otherwise.
                functions.add(new FunctionTool(
                        name,
                        (String) description,
                        ToolCalling.parameters(tool.get("parameters"), path + ".parameters"),
                        !Boolean.FALSE.equals(strict),
                        path + ".parameters"));
            }
        }
        boolean parallel = !Boolean.FALSE.equals(parallelToolCalls);
        if (functions.isEmpty()) {
            if (choice == null || "none".equals(choice) || "auto".equals(choice))
                return new ToolCalling(List.of(), ToolCalling.Choice.NONE, null, parallel);
            throw ApiException.invalidRequest(
                    "'tool_choice' is only allowed when 'tools' are specified.", "tool_choice");
        }
        return switch (choice) {
            case null -> new ToolCalling(functions, ToolCalling.Choice.AUTO, null, parallel);
            case String mode
            when mode.equals("auto") -> new ToolCalling(functions, ToolCalling.Choice.AUTO, null, parallel);
            case String mode
            when mode.equals("none") -> new ToolCalling(functions, ToolCalling.Choice.NONE, null, parallel);
            case String mode
            when mode.equals("required") -> new ToolCalling(functions, ToolCalling.Choice.REQUIRED, null, parallel);
            case Map<?, ?> named
            when "function".equals(named.get("type")) -> {
                checkKeys(named, Set.of("type", "name"), "tool_choice");
                if (!(named.get("name") instanceof String name)
                        || functions.stream().noneMatch(tool -> tool.name().equals(name)))
                    throw ApiException.invalidRequest(
                            "'tool_choice.name' must name one of the tools.", "tool_choice.name");
                yield new ToolCalling(functions, ToolCalling.Choice.FUNCTION, name, parallel);
            }
            case Map<?, ?> other -> throw ApiException.unsupportedParameter("tool_choice.type=" + other.get("type"));
            default ->
                throw ApiException.invalidRequest(
                        "'tool_choice' must be 'none', 'auto', 'required', or a function choice.", "tool_choice");
        };
    }

    /// `reasoning`: `effort` selects the template's level; `summary` is accepted, but no summary is generated (the
    /// reasoning itself is returned as `reasoning_text`).
    private static Map<?, ?> reasoningOptions(Object value) {
        if (value == null) return null;
        if (!(value instanceof Map<?, ?> options))
            throw ApiException.invalidRequest("'reasoning' must be an object.", "reasoning");
        checkKeys(options, Set.of("effort", "summary", "generate_summary"), "reasoning");
        for (String key : List.of("summary", "generate_summary")) {
            Object summary = options.get(key);
            if (summary != null && !Set.of("auto", "concise", "detailed").contains(summary))
                throw ApiException.invalidRequest(
                        "'reasoning." + key + "' must be auto, concise or detailed.", "reasoning." + key);
        }
        return options;
    }

    @SuppressWarnings("unchecked")
    private static ResponseFormat format(Object text) {
        if (text == null) return ResponseFormat.TEXT;
        if (!(text instanceof Map<?, ?> options))
            throw ApiException.invalidRequest("'text' must be an object.", "text");
        checkKeys(options, Set.of("format", "verbosity"), "text");
        if (options.get("verbosity") != null) throw ApiException.unsupportedParameter("text.verbosity");
        Object format = options.get("format");
        if (format == null) return ResponseFormat.TEXT;
        if (!(format instanceof Map<?, ?> definition) || !(definition.get("type") instanceof String type))
            throw ApiException.invalidRequest("'text.format.type' is required.", "text.format.type");
        return switch (type) {
            case "text" -> {
                checkKeys(definition, Set.of("type"), "text.format");
                yield ResponseFormat.TEXT;
            }
            case "json_object" -> {
                checkKeys(definition, Set.of("type"), "text.format");
                yield new ResponseFormat(ResponseFormat.Kind.JSON_OBJECT, null, null, null);
            }
            case "json_schema" -> {
                // The flattened Responses form: the json_schema object's fields sit in the format itself.
                checkKeys(definition, Set.of("type", "name", "schema", "strict", "description"), "text.format");
                Map<String, Object> chat = new LinkedHashMap<>((Map<String, Object>) definition);
                chat.remove("type");
                ResponseFormat parsed = ResponseFormat.fromRequest(Map.of("type", "json_schema", "json_schema", chat));
                yield new ResponseFormat(parsed.kind(), parsed.name(), parsed.schema(), "text.format.schema");
            }
            default -> throw ApiException.unsupportedParameter("text.format.type=" + type);
        };
    }

    private GenerationConfig sampling(ResponsesRequest request) {
        float temperature = this.samplingDefaults.temperature();
        if (request.temperature() != null) {
            double value = request.temperature();
            if (!(value >= 0.0 && value <= 2.0))
                throw ApiException.invalidRequest("'temperature' must be between 0 and 2.", "temperature");
            temperature = (float) value;
        }
        float topP = this.samplingDefaults.topP();
        if (request.topP() != null) {
            double value = request.topP();
            if (!(value >= 0.0 && value <= 1.0))
                throw ApiException.invalidRequest("'top_p' must be between 0 and 1.", "top_p");
            topP = (float) value;
        }
        boolean greedy = temperature == 0.0f || topP == 0.0f;
        return new GenerationConfig(
                temperature,
                this.samplingDefaults.topK(),
                greedy ? 1.0f : topP,
                ThreadLocalRandom.current().nextLong(),
                greedy);
    }

    /// The request's settings, as a response repeats them.
    private static Map<String, Object> echo(
            ResponsesRequest request, ToolCalling tools, QwenChatTemplate.Thinking thinking) {
        Map<String, Object> echo = new LinkedHashMap<>();
        echo.put("instructions", request.instructions());
        echo.put("max_output_tokens", request.maxOutputTokens());
        echo.put("parallel_tool_calls", tools.parallel());
        echo.put("temperature", request.temperature());
        echo.put("top_p", request.topP());
        echo.put("tool_choice", request.toolChoice() == null ? "auto" : request.toolChoice());
        echo.put("tools", request.tools() == null ? List.of() : request.tools());
        echo.put("text", request.text() == null ? Map.of("format", Map.of("type", "text")) : request.text());
        Map<String, Object> reasoning = new LinkedHashMap<>();
        reasoning.put(
                "effort", thinking.enabled() ? thinking.effort().name().toLowerCase(java.util.Locale.ROOT) : "none");
        reasoning.put("summary", null);
        echo.put("reasoning", reasoning);
        echo.put("store", false);
        echo.put("metadata", request.otherFields().getOrDefault("metadata", Map.of()));
        return echo;
    }

    private static void checkKeys(Map<?, ?> object, Set<String> known, String path) {
        for (Object key : object.keySet())
            if (!known.contains(key)) throw ApiException.unrecognizedArgument(path + "." + key);
    }
}
