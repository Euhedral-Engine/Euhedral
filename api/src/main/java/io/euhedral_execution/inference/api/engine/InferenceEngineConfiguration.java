package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.api.chat.QwenChatTemplate;
import io.euhedral_execution.inference.api.chat.SamplingDefaults;
import io.euhedral_execution.inference.api.metrics.EngineMetrics;
import io.euhedral_execution.inference.api.metrics.ServerMetrics;
import io.euhedral_execution.inference.core.InferenceEngine;
import java.io.IOException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/// Creates the process's single `InferenceEngine` and the checkpoint-derived chat adapters.
///
/// Singletons are created before the embedded server's connector starts, so a load failure aborts
/// startup without ever serving HTTP. Spring destroys the engine after the web server and the
/// generation service have stopped; `InferenceEngine.close` then cancels and closes any residual session.
@Configuration(proxyBeanMethods = false)
@EnableConfigurationProperties(InferenceProperties.class)
public class InferenceEngineConfiguration {
    private static final Logger LOG = LoggerFactory.getLogger(InferenceEngineConfiguration.class);

    /// Loaded before the engine so an unsupported template fails fast without touching the GPU.
    @Bean
    QwenChatTemplate qwenChatTemplate(InferenceProperties properties) throws IOException {
        return QwenChatTemplate.load(properties.tokenizerDirectory());
    }

    @Bean
    SamplingDefaults samplingDefaults(InferenceProperties properties) throws IOException {
        return SamplingDefaults.load(properties.tokenizerDirectory());
    }

    /// Depends on the template bean only to order checkpoint validation before the GPU load.
    @Bean(destroyMethod = "close")
    InferenceEngine inferenceEngine(InferenceProperties properties, QwenChatTemplate chatTemplate) throws IOException {
        LOG.info("Loading inference engine for model {}", properties.modelId());
        try {
            return InferenceEngine.load(properties.toInferenceConfig());
        } catch (InferenceEngine.StartupFailure failure) {
            // Rollback failed inside load; retry once so a failed startup does not strand device resources.
            try {
                failure.close();
            } catch (RuntimeException | Error cleanup) {
                failure.addSuppressed(cleanup);
            }
            throw failure;
        }
    }

    @Bean(destroyMethod = "close")
    EngineInferenceBackend inferenceBackend(
            InferenceEngine engine,
            InferenceProperties properties,
            QwenChatTemplate chatTemplate,
            ServerMetrics metrics) {
        return new EngineInferenceBackend(engine, properties.modelId(), chatTemplate, metrics);
    }

    @Bean
    EngineMetrics engineMetrics(
            InferenceEngine engine, InferenceProperties properties, EngineInferenceBackend backend) {
        return new EngineMetrics(engine, properties.modelId(), backend.contextLength());
    }
}
