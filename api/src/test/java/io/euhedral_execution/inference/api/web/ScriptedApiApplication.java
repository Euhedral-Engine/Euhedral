package io.euhedral_execution.inference.api.web;

import io.euhedral_execution.inference.api.anthropic.MessagesMapper;
import io.euhedral_execution.inference.api.chat.ConversationPlanner;
import io.euhedral_execution.inference.api.chat.GenerationService;
import io.euhedral_execution.inference.api.chat.QwenChatTemplate;
import io.euhedral_execution.inference.api.chat.SamplingDefaults;
import io.euhedral_execution.inference.api.engine.ApiProperties;
import io.euhedral_execution.inference.api.openai.ChatRequestMapper;
import io.euhedral_execution.inference.api.responses.ResponsesMapper;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Import;

/// The production web layer with the engine replaced by `ScriptedInferenceBackend`.
/// Imports components explicitly so the engine configuration is never scanned.
@SpringBootConfiguration
@ConditionalOnProperty(name = "euhedral.test.scripted-api", havingValue = "true")
@EnableAutoConfiguration
@EnableConfigurationProperties(ApiProperties.class)
@Import({
    OpenAiController.class,
    AnthropicController.class,
    MessagesMapper.class,
    ResponsesMapper.class,
    HealthController.class,
    ApiErrorHandler.class,
    ChatRequestBodyLimit.class,
    GenerationService.class,
    ConversationPlanner.class,
    ChatRequestMapper.class
})
class ScriptedApiApplication {
    static final SamplingDefaults CHECKPOINT_DEFAULTS = new SamplingDefaults(1.0f, 20, 0.95f);

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
    SamplingDefaults samplingDefaults() {
        return CHECKPOINT_DEFAULTS;
    }
}
