package io.euhedral_execution.inference.api.openai;

import io.euhedral_execution.inference.api.chat.ApiException;
import io.euhedral_execution.inference.api.chat.Conversation;
import io.euhedral_execution.inference.api.chat.ConversationPlanner;
import io.euhedral_execution.inference.api.chat.QwenChatTemplate;
import io.euhedral_execution.inference.api.chat.Reasoning;
import io.euhedral_execution.inference.api.chat.ResponseFormat;
import io.euhedral_execution.inference.api.chat.SamplingDefaults;
import io.euhedral_execution.inference.api.chat.ToolCalling;
import io.euhedral_execution.inference.api.chat.ToolResults;
import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Predicate;
import org.springframework.stereotype.Component;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.json.JsonMapper;

/// Validates an OpenAI Chat Completions request and maps it to the shared [Conversation].
///
/// Fields fall into four classes: supported; metadata that cannot change generation and is ignored;
/// behavioral features accepted only at their neutral value; and everything else, which is rejected.
@Component
public class ChatRequestMapper {
    static final int MAX_STOP_SEQUENCES = 4;
    private static final JsonMapper JSON = JsonMapper.shared();

    /// Accepted and ignored: these never influence the generated tokens.
    private static final Set<String> IGNORED_METADATA =
            Set.of("user", "metadata", "store", "service_tier", "safety_identifier", "prompt_cache_key");

    /// Behavioral features that are not implemented; accepted only when the value requests no behavior.
    private static final Map<String, Predicate<Object>> NEUTRAL_VALUES = Map.ofEntries(
            Map.entry("n", value -> value instanceof Number number && number.doubleValue() == 1.0),
            Map.entry("logprobs", Boolean.FALSE::equals),
            Map.entry("top_logprobs", value -> false),
            Map.entry("frequency_penalty", ChatRequestMapper::isZero),
            Map.entry("presence_penalty", ChatRequestMapper::isZero),
            Map.entry("repeat_penalty", value -> value instanceof Number number && number.doubleValue() == 1.0),
            Map.entry("logit_bias", value -> value instanceof Map<?, ?> map && map.isEmpty()),
            Map.entry("functions", value -> value instanceof List<?> list && list.isEmpty()),
            Map.entry("function_call", "none"::equals),
            Map.entry("modalities", List.of("text")::equals),
            Map.entry("audio", value -> false),
            Map.entry("prediction", value -> false),
            Map.entry("web_search_options", value -> false),
            Map.entry("verbosity", value -> false));

    private static final Map<String, Predicate<Object>> MESSAGE_NEUTRAL_VALUES = Map.of(
            "refusal", value -> false,
            "function_call", value -> false,
            "audio", value -> false);

    private final InferenceBackend backend;
    private final SamplingDefaults samplingDefaults;
    private final ConversationPlanner planner;

    public ChatRequestMapper(InferenceBackend backend, SamplingDefaults samplingDefaults, ConversationPlanner planner) {
        this.backend = backend;
        this.samplingDefaults = samplingDefaults;
        this.planner = planner;
    }

    /// Validates and plans a request on the backend's workers. The future fails with an [ApiException] for an
    /// invalid request.
    public CompletableFuture<ConversationPlanner.Planned> planAsync(ChatCompletionRequest request) {
        return this.planner.planAsync(() -> map(request));
    }

    /// Validates everything the request states and maps it to the shared conversation and an OpenAI responder.
    public ConversationPlanner.Mapped map(ChatCompletionRequest request) {
        if (request == null) throw ApiException.invalidRequest("Request body is required.", null);
        checkOtherFields(request.otherFields(), NEUTRAL_VALUES, "");
        boolean stream = Boolean.TRUE.equals(request.stream());
        boolean includeUsage = streamOptionsIncludeUsage(request, stream);
        requireServedModel(request.model());
        ToolCalling tools = ToolCalling.fromRequest(request.tools(), request.toolChoice(), request.parallelToolCalls());
        QwenChatTemplate.Thinking thinking =
                Reasoning.thinking(request.reasoningEffort(), request.chatTemplateKwargs());
        ResponseFormat format = ResponseFormat.fromRequest(request.responseFormat());
        if (format.json() && request.stop() != null)
            throw ApiException.invalidRequest(
                    "'stop' cannot be combined with a JSON 'response_format': a stop inside the JSON would end it"
                            + " invalid.",
                    "stop");
        List<QwenChatTemplate.Turn> turns = turns(request.messages());
        var conversation = new Conversation(
                turns,
                tools,
                thinking,
                Integer.MAX_VALUE,
                format,
                stops(request.stop()),
                sampling(request),
                maxTokens(request),
                request.maxTokens() != null ? "max_tokens" : "max_completion_tokens",
                stream);
        return new ConversationPlanner.Mapped(
                conversation,
                new OpenAiResponder(Instant.now().getEpochSecond(), this.backend.modelId(), includeUsage));
    }

    private void requireServedModel(String model) {
        if (model == null || model.isBlank())
            throw ApiException.invalidRequest("You must provide a model parameter.", "model");
        if (!model.equals(this.backend.modelId())) throw ApiException.modelNotFound(model);
    }

    private static boolean streamOptionsIncludeUsage(ChatCompletionRequest request, boolean stream) {
        var options = request.streamOptions();
        if (options == null) return false;
        if (!stream)
            throw ApiException.invalidRequest(
                    "The 'stream_options' parameter is only allowed when 'stream' is enabled.", "stream_options");
        for (var field : options.otherFields().entrySet()) {
            // Obfuscation padding only guards against network side channels; omitting it changes no content.
            if (field.getKey().equals("include_obfuscation")) continue;
            throw ApiException.unrecognizedArgument("stream_options." + field.getKey());
        }
        return Boolean.TRUE.equals(options.includeUsage());
    }

    private static List<QwenChatTemplate.Turn> turns(List<ChatMessage> messages) {
        if (messages == null || messages.isEmpty())
            throw ApiException.invalidRequest("'messages' must contain at least one message.", "messages");
        List<QwenChatTemplate.Turn> turns = new ArrayList<>(messages.size());
        ToolResults pending = null;
        for (int index = 0; index < messages.size(); index++) {
            ChatMessage message = messages.get(index);
            String path = "messages[" + index + "]";
            if (message == null) throw ApiException.invalidRequest("Message must be an object.", path);
            checkOtherFields(message.otherFields(), MESSAGE_NEUTRAL_VALUES, path + ".");
            QwenChatTemplate.Role role = role(message.role(), path);
            if (role != QwenChatTemplate.Role.ASSISTANT && message.toolCalls() != null)
                throw ApiException.invalidRequest(
                        "Only assistant messages may contain 'tool_calls'.", path + ".tool_calls");
            if (role != QwenChatTemplate.Role.TOOL && message.toolCallId() != null)
                throw ApiException.invalidRequest(
                        "Only tool messages may contain 'tool_call_id'.", path + ".tool_call_id");
            if (role != QwenChatTemplate.Role.ASSISTANT && message.reasoningContent() != null)
                throw ApiException.invalidRequest(
                        "Only assistant messages may contain 'reasoning_content'.", path + ".reasoning_content");
            if (message.reasoningContent() != null && !(message.reasoningContent() instanceof String))
                throw ApiException.invalidRequest("'reasoning_content' must be a string.", path + ".reasoning_content");
            String content = content(message.content(), role, path);
            if (role == QwenChatTemplate.Role.TOOL) {
                if (pending == null)
                    throw ApiException.invalidRequest(
                            "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'.",
                            path + ".role");
                pending.answer(message.toolCallId(), content, path + ".tool_call_id", "tool_call_id");
                continue;
            }
            if (pending != null) pending.closeInto(turns);
            List<String> ids = new ArrayList<>();
            List<QwenChatTemplate.ToolCall> calls = toolCalls(message.toolCalls(), path + ".tool_calls", ids);
            String reasoning = message.reasoningContent() instanceof String text ? text : "";
            turns.add(new QwenChatTemplate.Turn(role, content, calls, reasoning));
            pending = calls.isEmpty() ? null : new ToolResults(ids, path + ".tool_calls", "tool_call_id");
        }
        if (pending != null) pending.closeInto(turns);
        return turns;
    }

    /// Parses OpenAI assistant `tool_calls`, whose `arguments` are JSON-object strings, into template calls.
    /// Parses OpenAI assistant `tool_calls`, adding each call's ID to `callIds`.
    private static List<QwenChatTemplate.ToolCall> toolCalls(Object value, String path, List<String> callIds) {
        if (value == null) return List.of();
        if (!(value instanceof List<?> calls))
            throw ApiException.invalidRequest("'tool_calls' must be an array.", path);
        List<QwenChatTemplate.ToolCall> parsed = new ArrayList<>(calls.size());
        Set<String> ids = new HashSet<>();
        for (int index = 0; index < calls.size(); index++) {
            String callPath = path + "[" + index + "]";
            if (!(calls.get(index) instanceof Map<?, ?> call))
                throw ApiException.invalidRequest("Each tool call must be an object.", callPath);
            checkKeys(call, Set.of("id", "type", "function"), callPath);
            if (!(call.get("id") instanceof String id) || id.isEmpty())
                throw ApiException.invalidRequest("Tool call 'id' is required.", callPath + ".id");
            if (!ids.add(id))
                throw ApiException.invalidRequest("Tool call IDs must be unique within a message.", callPath + ".id");
            callIds.add(id);
            if (!(call.get("type") instanceof String type))
                throw ApiException.invalidRequest("Tool call 'type' is required.", callPath + ".type");
            if (!type.equals("function")) throw ApiException.unsupportedParameter(callPath + ".type");
            if (!(call.get("function") instanceof Map<?, ?> function))
                throw ApiException.invalidRequest("Tool call 'function' is required.", callPath + ".function");
            String functionPath = callPath + ".function";
            checkKeys(function, Set.of("name", "arguments"), functionPath);
            if (!(function.get("name") instanceof String name) || !ToolCalling.isFunctionName(name))
                throw ApiException.invalidRequest(
                        "Function names must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes.",
                        functionPath + ".name");
            parsed.add(new QwenChatTemplate.ToolCall(name, arguments(function.get("arguments"), functionPath)));
        }
        return parsed;
    }

    /// The template iterates the parsed object; it renders nothing for the empty string, as for `{}`.
    @SuppressWarnings("unchecked")
    private static Map<String, Object> arguments(Object value, String functionPath) {
        String path = functionPath + ".arguments";
        if (!(value instanceof String text))
            throw ApiException.invalidRequest("Tool call 'arguments' must be a JSON string.", path);
        if (text.isEmpty()) return Map.of();
        Object parsed;
        try {
            parsed = JSON.readValue(text, Object.class);
        } catch (JacksonException invalid) {
            throw ApiException.invalidRequest("Tool call 'arguments' must be valid JSON.", path);
        }
        if (!(parsed instanceof Map<?, ?> object))
            throw ApiException.invalidRequest("Tool call 'arguments' must encode a JSON object.", path);
        for (Object key : object.keySet()) {
            if (!ToolCalling.isParameterName((String) key))
                throw ApiException.invalidRequest("Argument names must be non-empty.", path);
        }
        return (Map<String, Object>) object;
    }

    private static QwenChatTemplate.Role role(String role, String path) {
        if (role == null) throw ApiException.invalidRequest("Message role is required.", path + ".role");
        return switch (role) {
            // OpenAI's newer name for instructions formerly sent as system messages.
            case "system", "developer" -> QwenChatTemplate.Role.SYSTEM;
            case "user" -> QwenChatTemplate.Role.USER;
            case "assistant" -> QwenChatTemplate.Role.ASSISTANT;
            case "tool" -> QwenChatTemplate.Role.TOOL;
            // Legacy function calling predates tool_call IDs and is not implemented.
            case "function" -> throw ApiException.unsupportedParameter(path + ".role");
            default -> throw ApiException.invalidRequest("Invalid message role '" + role + "'.", path + ".role");
        };
    }

    /// Mirrors the template's `render_content` for text: strings pass through, text parts concatenate.
    private static String content(Object content, QwenChatTemplate.Role role, String path) {
        if (content == null) {
            if (role == QwenChatTemplate.Role.ASSISTANT) return "";
            throw ApiException.invalidRequest("Message content is required.", path + ".content");
        }
        if (content instanceof String text) return text;
        if (!(content instanceof List<?> parts))
            throw ApiException.invalidRequest(
                    "Message content must be a string or an array of content parts.", path + ".content");
        StringBuilder text = new StringBuilder();
        for (int index = 0; index < parts.size(); index++) {
            String partPath = path + ".content[" + index + "]";
            if (!(parts.get(index) instanceof Map<?, ?> part))
                throw ApiException.invalidRequest("Content part must be an object.", partPath);
            Object type = part.get("type");
            if (!"text".equals(type)) {
                if (type instanceof String) throw ApiException.unsupportedParameter(partPath + ".type=" + type);
                throw ApiException.invalidRequest("Content part type is required.", partPath + ".type");
            }
            if (!(part.get("text") instanceof String partText))
                throw ApiException.invalidRequest("Text content part requires 'text'.", partPath + ".text");
            for (Object key : part.keySet()) {
                if (!key.equals("type") && !key.equals("text"))
                    throw ApiException.unrecognizedArgument(partPath + "." + key);
            }
            text.append(partText);
        }
        return text.toString();
    }

    /// The requested completion budget, or null for the server's default; the planner checks it against the context.
    private static Integer maxTokens(ChatCompletionRequest request) {
        Integer requested = request.maxCompletionTokens();
        if (request.maxTokens() != null) {
            if (requested != null && !requested.equals(request.maxTokens()))
                throw ApiException.invalidRequest(
                        "'max_tokens' and 'max_completion_tokens' conflict; send only one.", "max_tokens");
            requested = request.maxTokens();
        }
        return requested;
    }

    /// Omitted fields take the checkpoint's generation_config values; temperature or top_p of 0 is greedy.
    private GenerationConfig sampling(ChatCompletionRequest request) {
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
        long seed = request.seed() != null
                ? request.seed()
                : ThreadLocalRandom.current().nextLong();
        boolean greedy = temperature == 0.0f || topP == 0.0f;
        return new GenerationConfig(temperature, this.samplingDefaults.topK(), greedy ? 1.0f : topP, seed, greedy);
    }

    private static List<String> stops(Object stop) {
        if (stop == null) return List.of();
        List<?> values = stop instanceof String single ? List.of(single) : stop instanceof List<?> list ? list : null;
        if (values == null)
            throw ApiException.invalidRequest("'stop' must be a string or an array of strings.", "stop");
        if (values.size() > MAX_STOP_SEQUENCES)
            throw ApiException.invalidRequest("'stop' accepts at most " + MAX_STOP_SEQUENCES + " sequences.", "stop");
        List<String> stops = new ArrayList<>(values.size());
        for (Object value : values) {
            if (!(value instanceof String text) || text.isEmpty())
                throw ApiException.invalidRequest("Each stop sequence must be a non-empty string.", "stop");
            stops.add(text);
        }
        return stops;
    }

    private static void checkOtherFields(
            Map<String, Object> fields, Map<String, Predicate<Object>> neutralValues, String prefix) {
        for (var field : fields.entrySet()) {
            String name = field.getKey();
            if (prefix.isEmpty() && IGNORED_METADATA.contains(name)) continue;
            Predicate<Object> neutral = neutralValues.get(name);
            if (neutral == null) throw ApiException.unrecognizedArgument(prefix + name);
            if (field.getValue() != null && !neutral.test(field.getValue()))
                throw ApiException.unsupportedParameter(prefix + name);
        }
    }

    private static void checkKeys(Map<?, ?> object, Set<String> known, String path) {
        for (Object key : object.keySet()) {
            if (!known.contains(key)) throw ApiException.unrecognizedArgument(path + "." + key);
        }
    }

    private static boolean isZero(Object value) {
        return value instanceof Number number && number.doubleValue() == 0.0;
    }
}
