package io.euhedral_execution.inference.api.web;

import io.euhedral_execution.inference.api.anthropic.MessagesMapper;
import io.euhedral_execution.inference.api.chat.ConversationPlanner;
import io.euhedral_execution.inference.api.chat.GenerationService;
import io.euhedral_execution.inference.api.chat.QwenChatTemplate;
import io.euhedral_execution.inference.api.metrics.ServerMetrics;
import io.euhedral_execution.inference.api.openai.ChatRequestMapper;
import io.euhedral_execution.inference.api.responses.ResponsesMapper;
import io.euhedral_execution.inference.core.settings.RequestSettings;
import io.euhedral_execution.inference.core.settings.SamplingSettings;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Import;

/// The production web layer with the engine replaced by `ScriptedInferenceBackend`.
/// Imports components explicitly so the engine configuration is never scanned.
@SpringBootConfiguration
@ConditionalOnProperty(name = "euhedral.test.scripted-api", havingValue = "true")
@EnableAutoConfiguration
@Import({
    OpenAiController.class,
    WebConfiguration.class,
    AnthropicController.class,
    MessagesMapper.class,
    ResponsesMapper.class,
    HealthController.class,
    ApiErrorHandler.class,
    ChatRequestBodyLimit.class,
    GenerationService.class,
    ServerMetrics.class,
    ConversationPlanner.class,
    ChatRequestMapper.class
})
class ScriptedApiApplication {
    /// The `sampling` settings, apart from the defaults so a test sees they reach the request mapping.
    static final SamplingSettings SAMPLING = new SamplingSettings(0.6f, 30, 0.9f);

    /// The request limits, with `euhedral.test.max-queued` as the queue length.
    @Bean
    RequestSettings requestSettings(@Value("${euhedral.test.max-queued:16}") int maxQueued) {
        RequestSettings defaults = RequestSettings.defaults();
        return new RequestSettings(defaults.defaultMaxTokens(), maxQueued, defaults.maxBodySize(), defaults.timeout());
    }

    @Bean
    ScriptedInferenceBackend scriptedInferenceBackend() {
        return new ScriptedInferenceBackend();
    }

    @Bean
    QwenChatTemplate qwenChatTemplate() throws IOException {
        try (var template = ScriptedApiApplication.class.getResourceAsStream("/qwen-chat-template.jinja")) {
            return QwenChatTemplate.fromTemplateSource(new String(template.readAllBytes(), StandardCharsets.UTF_8));
        }
    }

    @Bean
    SamplingSettings sampling() {
        return SAMPLING;
    }
}
