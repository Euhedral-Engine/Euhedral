package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.engine.ApiProperties;
import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.core.guidance.GrammarException;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;
import org.springframework.stereotype.Component;

/// Turns a [Conversation] into a [GenerationPlan], the same way for every API surface.
///
/// On the backend's workers: the surface's mapping (validation) runs first, then the answer's grammar is compiled
/// (each schema on its own first, so a refusal names its request field), the prompt is rendered with the
/// checkpoint template and encoded by the backend's tokenization frames, and the completion budget is checked
/// against the context. A prompt is encoded once.
@Component
public class ConversationPlanner {
    private final InferenceBackend backend;
    private final QwenChatTemplate chatTemplate;
    private final ApiProperties properties;

    public ConversationPlanner(InferenceBackend backend, QwenChatTemplate chatTemplate, ApiProperties properties) {
        this.backend = backend;
        this.chatTemplate = chatTemplate;
        this.properties = properties;
    }

    /// A surface's validated request and the responder that answers it in the surface's format.
    public record Mapped(Conversation conversation, GenerationService.Responder responder) {}

    /// A plan ready to generate and its responder.
    public record Planned(GenerationPlan plan, GenerationService.Responder responder) {}

    /// Maps and plans a request on the workers. The future fails with an [ApiException] for an invalid request.
    public CompletableFuture<Planned> planAsync(Supplier<Mapped> mapping) {
        return CompletableFuture.supplyAsync(
                        () -> {
                            Mapped mapped = mapping.get();
                            Conversation conversation = mapped.conversation();
                            String grammar = grammar(conversation);
                            return new Prepared(mapped, grammar, render(conversation));
                        },
                        this.backend.workers())
                .thenCompose(prepared -> this.backend
                        .encodePrompt(prepared.prompt())
                        .thenApply(prompt -> new Planned(
                                plan(prepared, prompt), prepared.mapped().responder())));
    }

    /// The prompt tokens a request would use, without generating.
    public CompletableFuture<Integer> countTokensAsync(Supplier<Conversation> mapping) {
        return CompletableFuture.supplyAsync(() -> render(mapping.get()), this.backend.workers())
                .thenCompose(this.backend::encodePrompt)
                .thenApply(InferenceBackend.EncodedPrompt::tokenCount);
    }

    private record Prepared(Mapped mapped, String grammar, String prompt) {}

    private GenerationPlan plan(Prepared prepared, InferenceBackend.EncodedPrompt prompt) {
        Conversation conversation = prepared.mapped().conversation();
        return new GenerationPlan(
                UUID.randomUUID().toString().replace("-", ""),
                prompt,
                maxTokens(conversation, prompt.tokenCount()),
                conversation.sampling(),
                conversation.stops(),
                conversation.stream(),
                conversation.tools(),
                conversation.thinking().enabled(),
                conversation.reasoningBudget(),
                conversation.format(),
                prepared.grammar());
    }

    /// The conversation rendered by the checkpoint template, ending in the assistant's generation prompt.
    private String render(Conversation conversation) {
        ToolCalling tools = conversation.tools();
        try {
            return (tools.parsesOutput()
                            ? this.chatTemplate.renderJsonTools(
                                    conversation.turns(),
                                    tools,
                                    conversation.format().json(),
                                    conversation.thinking())
                            : this.chatTemplate.render(
                                    conversation.turns(), tools.promptTools(), conversation.thinking()))
                    + tools.generationPrefix();
        } catch (QwenChatTemplate.InvalidConversationException invalid) {
            throw ApiException.invalidRequest(invalid.getMessage(), "messages");
        }
    }

    /// The answer's grammar, or null for free text.
    private String grammar(Conversation conversation) {
        ToolCalling tools = conversation.tools();
        ResponseFormat format = conversation.format();
        boolean reasoning = conversation.thinking().enabled();
        if (format.kind() == ResponseFormat.Kind.JSON_SCHEMA) checkSchema(format.schema(), format.schemaParam());
        String grammar;
        if (tools.parsesOutput()) {
            for (FunctionTool tool : tools.callable())
                if (tool.strict()) checkSchema(tool.parametersOrEmpty(), tool.parametersParam());
            grammar = OutputGrammar.tools(tools, format.answerSchema(), reasoning);
        } else if (format.json()) {
            grammar = OutputGrammar.json(format.answerSchema(), reasoning);
        } else return null;
        try {
            this.backend.checkGrammar(grammar);
        } catch (GrammarException refused) {
            throw ApiException.invalidRequest("The requested output cannot be enforced: " + refused.getMessage(), null);
        }
        return grammar;
    }

    private void checkSchema(Map<String, Object> schema, String param) {
        try {
            this.backend.checkJsonSchema(OutputGrammar.schemaText(schema));
        } catch (GrammarException refused) {
            throw ApiException.unenforceableSchema(
                    "The JSON Schema cannot be enforced: " + refused.getMessage(), param);
        }
    }

    private int maxTokens(Conversation conversation, int promptTokens) {
        Integer requested = conversation.maxTokens();
        String param = conversation.maxTokensParam();
        int context = this.backend.contextLength();
        int remaining = context - promptTokens;
        if (requested != null) {
            if (requested < 1) throw ApiException.invalidRequest("'" + param + "' must be at least 1.", param);
            if (requested > remaining)
                throw ApiException.contextLengthExceeded(
                        "This model's maximum context length is " + context + " tokens. However, you requested "
                                + (promptTokens + requested) + " tokens (" + promptTokens + " in the messages, "
                                + requested + " in the completion).",
                        param);
            return requested;
        }
        if (remaining < 1)
            throw ApiException.contextLengthExceeded(
                    "This model's maximum context length is " + context + " tokens, but the messages use "
                            + promptTokens + " tokens.",
                    "messages");
        return Math.min(this.properties.defaultMaxTokens(), remaining);
    }
}
