package io.euhedral_execution.inference.api;

import io.euhedral_execution.inference.api.engine.ApiProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

/// OpenAI-compatible HTTP entry point for Euhedral-Inference.
///
/// Spring only adapts HTTP to the core engine. The single `InferenceEngine` bean owns the model, GPU,
/// lattice, and runtime; each request owns one generation session for its lifetime.
@SpringBootApplication
@EnableConfigurationProperties(ApiProperties.class)
public class EuhedralInferenceApplication {

    static void main(String[] args) {
        ShutdownSignals.install(SpringApplication.run(EuhedralInferenceApplication.class, args));
    }
}
