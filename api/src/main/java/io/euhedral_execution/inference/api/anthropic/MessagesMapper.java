package io.euhedral_execution.inference.api.anthropic;

import io.euhedral_execution.inference.api.chat.ApiException;
import io.euhedral_execution.inference.api.chat.Conversation;
import io.euhedral_execution.inference.api.chat.ConversationPlanner;
import io.euhedral_execution.inference.api.chat.FunctionTool;
import io.euhedral_execution.inference.api.chat.QwenChatTemplate;
import io.euhedral_execution.inference.api.chat.ResponseFormat;
import io.euhedral_execution.inference.api.chat.SamplingDefaults;
import io.euhedral_execution.inference.api.chat.ToolCalling;
import io.euhedral_execution.inference.api.chat.ToolResults;
import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ThreadLocalRandom;
import java.util.regex.Pattern;
import org.springframework.stereotype.Component;

/// Validates an Anthropic Messages request and maps it to the shared [Conversation].
///
/// Supported: `system`, `messages` with text, `tool_use`, `tool_result` and `thinking` blocks, `tools` (custom),
/// `tool_choice`, `max_tokens`, `stop_sequences`, `temperature`, `top_p`, `top_k`, `stream`, `thinking`
/// (`enabled` with an enforced `budget_tokens`, `adaptive`, `disabled`), `output_config.effort` and
/// `output_config.format` (also the earlier `output_format`). Accepted without effect: `metadata`, `service_tier`,
/// `cache_control` on blocks and tools (prefix reuse is automatic). Refused: images, documents, server tools,
/// redacted thinking, a final assistant message (prefill), MCP servers, containers, and anything unknown.
@Component
public class MessagesMapper {
    private static final Pattern TOOL_NAME = Pattern.compile("[a-zA-Z0-9_-]{1,128}");
    private static final int MAX_STOP_SEQUENCES = 32;
    private static final Set<String> IGNORED = Set.of("metadata", "service_tier");

    private final InferenceBackend backend;
    private final SamplingDefaults samplingDefaults;
    private final ConversationPlanner planner;

    public MessagesMapper(InferenceBackend backend, SamplingDefaults samplingDefaults, ConversationPlanner planner) {
        this.backend = backend;
        this.samplingDefaults = samplingDefaults;
        this.planner = planner;
    }

    /// Validates and plans a request on the backend's workers; the future fails with an [ApiException].
    public CompletableFuture<ConversationPlanner.Planned> planAsync(MessagesRequest request) {
        return this.planner.planAsync(() -> {
            Conversation conversation = conversation(request, true);
            return new ConversationPlanner.Mapped(
                    conversation, new AnthropicResponder(this.backend.modelId(), conversation.stream()));
        });
    }

    /// The prompt tokens of a request (`count_tokens`), which needs no `max_tokens`.
    public CompletableFuture<Integer> countTokensAsync(MessagesRequest request) {
        return this.planner.countTokensAsync(() -> conversation(request, false));
    }

    private Conversation conversation(MessagesRequest request, boolean generating) {
        if (request == null) throw ApiException.invalidRequest("Request body is required.", null);
        for (var field : request.otherFields().entrySet()) {
            String name = field.getKey();
            if (IGNORED.contains(name)) continue;
            if (name.equals("mcp_servers") || name.equals("container") || name.equals("context_management"))
                throw ApiException.unsupportedParameter(name);
            throw ApiException.unrecognizedArgument(name);
        }
        if (request.model() == null || request.model().isBlank())
            throw ApiException.invalidRequest("model: Field required", "model");
        if (!request.model().equals(this.backend.modelId())) throw ApiException.modelNotFound(request.model());
        Integer maxTokens = request.maxTokens();
        if (generating) {
            if (maxTokens == null) throw ApiException.invalidRequest("max_tokens: Field required", "max_tokens");
            if (maxTokens < 1) throw ApiException.invalidRequest("max_tokens: must be at least 1", "max_tokens");
        } else if (maxTokens != null) throw ApiException.unrecognizedArgument("max_tokens");
        boolean stream = Boolean.TRUE.equals(request.stream());
        if (!generating && request.stream() != null) throw ApiException.unrecognizedArgument("stream");

        ToolCalling tools = tools(request.tools(), request.toolChoice());
        Thinking thinking = thinking(request.thinking(), request.outputConfig(), maxTokens, generating);
        ResponseFormat format = format(request.outputConfig(), request.outputFormat());
        List<String> stops = stops(request.stopSequences());
        if (format.json() && !stops.isEmpty())
            throw ApiException.invalidRequest(
                    "stop_sequences cannot be combined with a JSON output format: a stop inside the JSON would end it"
                            + " invalid.",
                    "stop_sequences");
        return new Conversation(
                turns(request.system(), request.messages()),
                tools,
                thinking.template(),
                thinking.budget(),
                format,
                stops,
                sampling(request),
                maxTokens,
                "max_tokens",
                stream);
    }

    // Messages.

    private static List<QwenChatTemplate.Turn> turns(Object system, List<Object> messages) {
        List<QwenChatTemplate.Turn> turns = new ArrayList<>();
        if (system != null) {
            String text = system instanceof String plain ? plain : textBlocks(system, "system");
            turns.add(new QwenChatTemplate.Turn(QwenChatTemplate.Role.SYSTEM, text));
        }
        if (messages == null || messages.isEmpty())
            throw ApiException.invalidRequest("messages: at least one message is required", "messages");
        ToolResults pending = null;
        for (int index = 0; index < messages.size(); index++) {
            String path = "messages[" + index + "]";
            if (!(messages.get(index) instanceof Map<?, ?> message))
                throw ApiException.invalidRequest(path + ": must be an object", path);
            checkKeys(message, Set.of("role", "content"), path);
            Object content = message.get("content");
            if (content == null)
                throw ApiException.invalidRequest(path + ".content: Field required", path + ".content");
            List<Map<?, ?>> blocks = blocks(content, path + ".content");
            switch (message.get("role") instanceof String role ? role : null) {
                case "user" -> {
                    List<ToolResult> results = new ArrayList<>();
                    StringBuilder text = new StringBuilder();
                    for (int block = 0; block < blocks.size(); block++) {
                        String blockPath = path + ".content[" + block + "]";
                        Map<?, ?> item = blocks.get(block);
                        switch (type(item, blockPath)) {
                            case "text" -> text.append(text(item, blockPath));
                            case "tool_result" -> {
                                if (!text.isEmpty())
                                    throw ApiException.invalidRequest(
                                            blockPath + ": tool_result blocks must come before text in a user message",
                                            blockPath);
                                results.add(toolResult(item, blockPath));
                            }
                            default -> throw unsupportedBlock(item, blockPath);
                        }
                    }
                    if (!results.isEmpty()) {
                        if (pending == null)
                            throw ApiException.invalidRequest(
                                    path + ": tool_result blocks must answer tool_use blocks of the preceding"
                                            + " assistant message",
                                    path);
                        for (ToolResult result : results)
                            pending.answer(result.id(), result.content(), result.path(), "tool_use_id");
                        pending.closeInto(turns);
                        pending = null;
                    } else if (pending != null) pending.closeInto(turns);
                    if (!text.isEmpty() || results.isEmpty())
                        turns.add(new QwenChatTemplate.Turn(QwenChatTemplate.Role.USER, text.toString()));
                }
                case "assistant" -> {
                    if (pending != null) pending.closeInto(turns);
                    if (index == messages.size() - 1)
                        throw ApiException.unsupportedParameter(path + " (a final assistant message to continue)");
                    StringBuilder reasoning = new StringBuilder();
                    StringBuilder text = new StringBuilder();
                    List<QwenChatTemplate.ToolCall> calls = new ArrayList<>();
                    List<String> ids = new ArrayList<>();
                    for (int block = 0; block < blocks.size(); block++) {
                        String blockPath = path + ".content[" + block + "]";
                        Map<?, ?> item = blocks.get(block);
                        switch (type(item, blockPath)) {
                            case "thinking" -> {
                                checkKeys(item, Set.of("type", "thinking", "signature", "cache_control"), blockPath);
                                if (!(item.get("thinking") instanceof String thought))
                                    throw ApiException.invalidRequest(
                                            blockPath + ".thinking: Field required", blockPath + ".thinking");
                                reasoning.append(thought);
                            }
                            case "text" -> text.append(text(item, blockPath));
                            case "tool_use" -> {
                                checkKeys(item, Set.of("type", "id", "name", "input", "cache_control"), blockPath);
                                if (!(item.get("id") instanceof String id) || id.isEmpty())
                                    throw ApiException.invalidRequest(
                                            blockPath + ".id: Field required", blockPath + ".id");
                                if (ids.contains(id))
                                    throw ApiException.invalidRequest(
                                            blockPath + ".id: tool_use IDs must be unique", blockPath + ".id");
                                if (!(item.get("name") instanceof String name)
                                        || !TOOL_NAME.matcher(name).matches())
                                    throw ApiException.invalidRequest(
                                            blockPath + ".name: must match ^[a-zA-Z0-9_-]{1,128}$",
                                            blockPath + ".name");
                                if (!(item.get("input") instanceof Map<?, ?> input))
                                    throw ApiException.invalidRequest(
                                            blockPath + ".input: must be an object", blockPath + ".input");
                                ids.add(id);
                                calls.add(new QwenChatTemplate.ToolCall(name, stringKeys(input, blockPath + ".input")));
                            }
                            default -> throw unsupportedBlock(item, blockPath);
                        }
                    }
                    turns.add(new QwenChatTemplate.Turn(
                            QwenChatTemplate.Role.ASSISTANT, text.toString(), calls, reasoning.toString()));
                    pending = calls.isEmpty() ? null : new ToolResults(ids, path + ".content", "tool_use_id");
                }
                case null -> throw ApiException.invalidRequest(path + ".role: Field required", path + ".role");
                default ->
                    throw ApiException.invalidRequest(path + ".role: must be 'user' or 'assistant'", path + ".role");
            }
        }
        if (pending != null) pending.closeInto(turns);
        return turns;
    }

    private record ToolResult(Object id, String content, String path) {}

    private static ToolResult toolResult(Map<?, ?> block, String path) {
        checkKeys(block, Set.of("type", "tool_use_id", "content", "is_error", "cache_control"), path);
        Object error = block.get("is_error");
        if (error != null && !(error instanceof Boolean))
            throw ApiException.invalidRequest(path + ".is_error: must be a boolean", path + ".is_error");
        Object content = block.get("content");
        String text =
                content == null ? "" : content instanceof String plain ? plain : textBlocks(content, path + ".content");
        return new ToolResult(block.get("tool_use_id"), text, path + ".tool_use_id");
    }

    /// The text of an array of text blocks, concatenated as the template concatenates text parts.
    private static String textBlocks(Object value, String path) {
        StringBuilder text = new StringBuilder();
        List<Map<?, ?>> blocks = blocks(value, path);
        for (int index = 0; index < blocks.size(); index++) {
            String blockPath = path + "[" + index + "]";
            Map<?, ?> block = blocks.get(index);
            if (!"text".equals(type(block, blockPath))) throw unsupportedBlock(block, blockPath);
            text.append(text(block, blockPath));
        }
        return text.toString();
    }

    private static List<Map<?, ?>> blocks(Object value, String path) {
        if (value instanceof String text) return List.of(Map.of("type", "text", "text", text));
        if (!(value instanceof List<?> list))
            throw ApiException.invalidRequest(path + ": must be a string or a list of content blocks", path);
        List<Map<?, ?>> blocks = new ArrayList<>(list.size());
        for (int index = 0; index < list.size(); index++) {
            if (!(list.get(index) instanceof Map<?, ?> block))
                throw ApiException.invalidRequest(
                        path + "[" + index + "]: must be an object", path + "[" + index + "]");
            blocks.add(block);
        }
        return blocks;
    }

    private static String type(Map<?, ?> block, String path) {
        if (!(block.get("type") instanceof String type))
            throw ApiException.invalidRequest(path + ".type: Field required", path + ".type");
        return type;
    }

    private static String text(Map<?, ?> block, String path) {
        checkKeys(block, Set.of("type", "text", "cache_control", "citations"), path);
        Object citations = block.get("citations");
        if (citations != null && !(citations instanceof List<?> list && list.isEmpty()))
            throw ApiException.unsupportedParameter(path + ".citations");
        if (!(block.get("text") instanceof String text))
            throw ApiException.invalidRequest(path + ".text: Field required", path + ".text");
        return text;
    }

    private static ApiException unsupportedBlock(Map<?, ?> block, String path) {
        return ApiException.unsupportedParameter(path + ".type=" + block.get("type"));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> stringKeys(Map<?, ?> input, String path) {
        for (Object key : input.keySet())
            if (!ToolCalling.isParameterName((String) key))
                throw ApiException.invalidRequest(path + ": argument names must be non-empty", path);
        return (Map<String, Object>) input;
    }

    // Tools.

    private static ToolCalling tools(Object value, Object choice) {
        List<FunctionTool> functions = new ArrayList<>();
        if (value != null) {
            if (!(value instanceof List<?> list)) throw ApiException.invalidRequest("tools: must be a list", "tools");
            Set<String> names = new HashSet<>();
            for (int index = 0; index < list.size(); index++) {
                String path = "tools[" + index + "]";
                if (!(list.get(index) instanceof Map<?, ?> tool))
                    throw ApiException.invalidRequest(path + ": must be an object", path);
                Object type = tool.get("type");
                if (type != null && !"custom".equals(type))
                    throw ApiException.unsupportedParameter(path + ".type=" + type);
                checkKeys(tool, Set.of("type", "name", "description", "input_schema", "cache_control", "strict"), path);
                if (!(tool.get("name") instanceof String name)
                        || !TOOL_NAME.matcher(name).matches())
                    throw ApiException.invalidRequest(
                            path + ".name: must match ^[a-zA-Z0-9_-]{1,128}$", path + ".name");
                if (!names.add(name))
                    throw ApiException.invalidRequest(path + ".name: tool names must be unique", path + ".name");
                Object description = tool.get("description");
                if (description != null && !(description instanceof String))
                    throw ApiException.invalidRequest(path + ".description: must be a string", path + ".description");
                Object strict = tool.get("strict");
                if (strict != null && !(strict instanceof Boolean))
                    throw ApiException.invalidRequest(path + ".strict: must be a boolean", path + ".strict");
                if (tool.get("input_schema") == null)
                    throw ApiException.invalidRequest(path + ".input_schema: Field required", path + ".input_schema");
                String schemaPath = path + ".input_schema";
                functions.add(new FunctionTool(
                        name,
                        (String) description,
                        ToolCalling.parameters(tool.get("input_schema"), schemaPath),
                        Boolean.TRUE.equals(strict),
                        schemaPath));
            }
        }
        if (choice == null)
            return functions.isEmpty()
                    ? toolCalling(List.of(), ToolCalling.Choice.NONE, null, true)
                    : toolCalling(functions, ToolCalling.Choice.AUTO, null, true);
        if (!(choice instanceof Map<?, ?> selected))
            throw ApiException.invalidRequest("tool_choice: must be an object", "tool_choice");
        checkKeys(selected, Set.of("type", "name", "disable_parallel_tool_use"), "tool_choice");
        Object disable = selected.get("disable_parallel_tool_use");
        if (disable != null && !(disable instanceof Boolean))
            throw ApiException.invalidRequest(
                    "tool_choice.disable_parallel_tool_use: must be a boolean",
                    "tool_choice.disable_parallel_tool_use");
        boolean parallel = !Boolean.TRUE.equals(disable);
        String type = selected.get("type") instanceof String name ? name : null;
        if ("none".equals(type)) return toolCalling(functions, ToolCalling.Choice.NONE, null, parallel);
        if (functions.isEmpty()) throw ApiException.invalidRequest("tool_choice: requires tools", "tool_choice");
        return switch (type) {
            case "auto" -> toolCalling(functions, ToolCalling.Choice.AUTO, null, parallel);
            case "any" -> toolCalling(functions, ToolCalling.Choice.REQUIRED, null, parallel);
            case "tool" -> {
                if (!(selected.get("name") instanceof String name)
                        || functions.stream().noneMatch(tool -> tool.name().equals(name)))
                    throw ApiException.invalidRequest(
                            "tool_choice.name: must name one of the tools", "tool_choice.name");
                yield toolCalling(functions, ToolCalling.Choice.FUNCTION, name, parallel);
            }
            case null -> throw ApiException.invalidRequest("tool_choice.type: Field required", "tool_choice.type");
            default ->
                throw ApiException.invalidRequest(
                        "tool_choice.type: must be auto, any, tool or none", "tool_choice.type");
        };
    }

    private static ToolCalling toolCalling(
            List<FunctionTool> functions, ToolCalling.Choice choice, String function, boolean parallel) {
        return new ToolCalling(functions, choice, function, parallel);
    }

    // Thinking and output.

    private record Thinking(QwenChatTemplate.Thinking template, int budget) {}

    /// Extended thinking is opt-in in this API: without `thinking` the model answers directly.
    private static Thinking thinking(Object value, Object outputConfig, Integer maxTokens, boolean generating) {
        QwenChatTemplate.Effort effort = effort(outputConfig);
        if (value == null) {
            if (effort != null)
                throw ApiException.invalidRequest(
                        "output_config.effort: selects how much the model thinks, which needs thinking enabled",
                        "output_config.effort");
            return new Thinking(QwenChatTemplate.Thinking.DISABLED, Integer.MAX_VALUE);
        }
        if (!(value instanceof Map<?, ?> thinking))
            throw ApiException.invalidRequest("thinking: must be an object", "thinking");
        QwenChatTemplate.Effort level = effort == null ? QwenChatTemplate.Effort.XHIGH : effort;
        return switch (thinking.get("type") instanceof String type ? type : null) {
            case "disabled" -> {
                checkKeys(thinking, Set.of("type"), "thinking");
                if (effort != null)
                    throw ApiException.invalidRequest(
                            "output_config.effort: selects how much the model thinks, which needs thinking enabled",
                            "output_config.effort");
                yield new Thinking(QwenChatTemplate.Thinking.DISABLED, Integer.MAX_VALUE);
            }
            case "adaptive" -> {
                checkKeys(thinking, Set.of("type"), "thinking");
                yield new Thinking(new QwenChatTemplate.Thinking(level, true), Integer.MAX_VALUE);
            }
            case "enabled" -> {
                checkKeys(thinking, Set.of("type", "budget_tokens"), "thinking");
                if (!(thinking.get("budget_tokens") instanceof Integer budget) || budget < 1)
                    throw ApiException.invalidRequest(
                            "thinking.budget_tokens: must be a positive integer", "thinking.budget_tokens");
                if (generating && budget >= maxTokens)
                    throw ApiException.invalidRequest(
                            "thinking.budget_tokens: must be less than max_tokens", "thinking.budget_tokens");
                yield new Thinking(new QwenChatTemplate.Thinking(level, true), budget);
            }
            case null -> throw ApiException.invalidRequest("thinking.type: Field required", "thinking.type");
            default ->
                throw ApiException.invalidRequest(
                        "thinking.type: must be enabled, adaptive or disabled", "thinking.type");
        };
    }

    /// `output_config.effort`: `low` and `medium` are the template's; `high` and `max` its highest, `xhigh`.
    private static QwenChatTemplate.Effort effort(Object outputConfig) {
        if (!(outputConfig instanceof Map<?, ?> config) || config.get("effort") == null) return null;
        return switch (config.get("effort") instanceof String effort ? effort : "") {
            case "low" -> QwenChatTemplate.Effort.LOW;
            case "medium" -> QwenChatTemplate.Effort.MEDIUM;
            case "high", "max" -> QwenChatTemplate.Effort.XHIGH;
            default ->
                throw ApiException.invalidRequest(
                        "output_config.effort: must be low, medium, high or max", "output_config.effort");
        };
    }

    @SuppressWarnings("unchecked")
    private static ResponseFormat format(Object outputConfig, Object legacyFormat) {
        Object format = null;
        String path = null;
        if (outputConfig != null) {
            if (!(outputConfig instanceof Map<?, ?> config))
                throw ApiException.invalidRequest("output_config: must be an object", "output_config");
            checkKeys(config, Set.of("effort", "format"), "output_config");
            format = config.get("format");
            path = "output_config.format";
        }
        if (legacyFormat != null) {
            if (format != null)
                throw ApiException.invalidRequest(
                        "output_format: conflicts with output_config.format; send only one", "output_format");
            format = legacyFormat;
            path = "output_format";
        }
        if (format == null) return ResponseFormat.TEXT;
        if (!(format instanceof Map<?, ?> definition) || !"json_schema".equals(definition.get("type")))
            throw ApiException.invalidRequest(path + ": type must be json_schema", path);
        checkKeys(definition, Set.of("type", "schema"), path);
        if (!(definition.get("schema") instanceof Map<?, ?> schema))
            throw ApiException.invalidRequest(path + ".schema: must be a JSON Schema object", path + ".schema");
        if (schema.containsKey("x-guidance")) throw ApiException.unsupportedParameter(path + ".schema.x-guidance");
        return new ResponseFormat(
                ResponseFormat.Kind.JSON_SCHEMA, null, (Map<String, Object>) schema, path + ".schema");
    }

    private static List<String> stops(Object value) {
        if (value == null) return List.of();
        if (!(value instanceof List<?> list))
            throw ApiException.invalidRequest("stop_sequences: must be a list of strings", "stop_sequences");
        if (list.size() > MAX_STOP_SEQUENCES)
            throw ApiException.invalidRequest(
                    "stop_sequences: at most " + MAX_STOP_SEQUENCES + " sequences", "stop_sequences");
        List<String> stops = new ArrayList<>(list.size());
        for (Object stop : list) {
            if (!(stop instanceof String text) || text.isEmpty())
                throw ApiException.invalidRequest(
                        "stop_sequences: each sequence must be a non-empty string", "stop_sequences");
            stops.add(text);
        }
        return stops;
    }

    /// Omitted values take the checkpoint's generation_config; a temperature of 0 is greedy.
    private GenerationConfig sampling(MessagesRequest request) {
        float temperature = this.samplingDefaults.temperature();
        if (request.temperature() != null) {
            double value = request.temperature();
            if (!(value >= 0.0 && value <= 1.0))
                throw ApiException.invalidRequest("temperature: must be between 0 and 1", "temperature");
            temperature = (float) value;
        }
        float topP = this.samplingDefaults.topP();
        if (request.topP() != null) {
            double value = request.topP();
            if (!(value > 0.0 && value <= 1.0))
                throw ApiException.invalidRequest("top_p: must be greater than 0 and at most 1", "top_p");
            topP = (float) value;
        }
        int topK = this.samplingDefaults.topK();
        if (request.topK() != null) {
            if (request.topK() < 1) throw ApiException.invalidRequest("top_k: must be at least 1", "top_k");
            topK = request.topK();
        }
        boolean greedy = temperature == 0.0f;
        return new GenerationConfig(
                temperature,
                topK,
                greedy ? 1.0f : topP,
                ThreadLocalRandom.current().nextLong(),
                greedy);
    }

    private static void checkKeys(Map<?, ?> object, Set<String> known, String path) {
        for (Object key : object.keySet())
            if (!known.contains(key)) throw ApiException.unrecognizedArgument(path + "." + key);
    }
}
