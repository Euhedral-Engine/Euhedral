package io.euhedral_execution.inference.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.InferenceEngine;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.Timeout;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/// The real application on the real Flash-Next artifact, driven over HTTP: the shared conversation pipeline, the OpenAI
/// and Anthropic requests and the streamed response run unchanged over the Flash-Next text engine, which is selected by
/// the artifact alone. Run through `gradle :api:cudaIntegrationTest`.
@SpringBootTest(
        classes = EuhedralInferenceApplication.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@DirtiesContext
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@Timeout(1800)
class Qwen4ChatCompletionsCudaIntegrationTest {
    private static final String MODEL = "flash-next-test";
    private static final JsonMapper JSON = JsonMapper.shared();
    private static Path library;
    private static Path artifact;
    private static Path tokenizer;

    @LocalServerPort
    private int port;

    @Autowired
    private InferenceEngine engine;

    @BeforeAll
    static void requireAssets() {
        String configuredLibrary = System.getProperty("euhedral.cuda.library");
        assumeTrue(configuredLibrary != null && Files.isRegularFile(Path.of(configuredLibrary)));
        library = Path.of(configuredLibrary);
        artifact = Path.of(System.getProperty(
                "euhedral.qwen4.artifact", "/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl"));
        tokenizer = Path.of(System.getProperty("euhedral.qwen4.tokenizer-dir", "/mnt/shared/qwen38-flash-next/nvfp4"));
        assumeTrue(Files.isRegularFile(artifact) && Files.isRegularFile(tokenizer.resolve("chat_template.jinja")));
    }

    @DynamicPropertySource
    static void engineProperties(DynamicPropertyRegistry registry) {
        registry.add("euhedral.inference.artifact-path", () -> artifact.toString());
        registry.add("euhedral.inference.tokenizer-directory", () -> tokenizer.toString());
        registry.add("euhedral.inference.cuda-library-path", () -> library.toString());
        registry.add("euhedral.inference.worker-cpus", () -> "0");
        registry.add("euhedral.inference.max-context-tokens", () -> "4096");
        registry.add("euhedral.inference.model-id", () -> MODEL);
    }

    @Test
    @Order(1)
    void modelsAndHealth() throws Exception {
        var models = JSON.readTree(get("/v1/models").body());
        assertEquals(MODEL, models.get("data").get(0).get("id").asString());
        var health = get("/health");
        assertEquals(200, health.statusCode());
        assertFalse(health.body().contains("/models/") || models.toString().contains(".edrl"));
    }

    @Test
    @Order(2)
    void openAiChatCompletionAnswersFromTheModel() throws Exception {
        long before = this.engine.allocatedDeviceBytes();
        var response = post(
                "/v1/chat/completions",
                "{\"model\":\"" + MODEL
                        + "\",\"reasoning_effort\":\"none\",\"temperature\":0,\"max_tokens\":24,\"messages\":["
                        + "{\"role\":\"system\",\"content\":\"Answer with a single word.\"},"
                        + "{\"role\":\"user\",\"content\":\"What is the capital of France?\"}]}");
        assertEquals(200, response.statusCode(), response.body());
        JsonNode completion = JSON.readTree(response.body());
        System.out.println("Flash-Next completion: " + completion);
        String content = completion.at("/choices/0/message/content").asString();
        assertTrue(content.toLowerCase(java.util.Locale.ROOT).contains("paris"), content);
        assertFalse(content.contains("<|im_end|>"), content);
        assertTrue(completion.at("/usage/completion_tokens").asInt() >= 1);
        // The sequence's storage is returned; only the engine's retained scratch may have grown.
        assertTrue(this.engine.allocatedDeviceBytes() - before <= (64L << 20), "sequence state was not released");
    }

    @Test
    @Order(3)
    void openAiStreamingDeliversTextAndUsage() throws Exception {
        var request = HttpRequest.newBuilder(uri("/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString("{\"model\":\"" + MODEL
                        + "\",\"reasoning_effort\":\"none\",\"stream\":true,\"stream_options\":{\"include_usage\":true},"
                        + "\"temperature\":0,\"max_tokens\":24,\"messages\":["
                        + "{\"role\":\"user\",\"content\":\"Count from one to five in words.\"}]}"))
                .build();
        List<String> data = new ArrayList<>();
        try (var client = HttpClient.newHttpClient()) {
            var response = client.send(request, HttpResponse.BodyHandlers.ofLines());
            assertEquals(200, response.statusCode());
            response.body()
                    .filter(line -> line.startsWith("data:"))
                    .forEach(line -> data.add(line.substring("data:".length()).strip()));
        }
        assertEquals("[DONE]", data.getLast());
        StringBuilder text = new StringBuilder();
        boolean usage = false;
        for (String chunk : data.subList(0, data.size() - 1)) {
            JsonNode node = JSON.readTree(chunk);
            JsonNode content = node.at("/choices/0/delta/content");
            if (!content.isMissingNode() && !content.isNull()) text.append(content.asString());
            if (node.has("usage") && !node.get("usage").isNull()) usage = true;
        }
        System.out.println("Flash-Next streamed: " + text);
        assertFalse(text.toString().isBlank());
        assertTrue(usage, "usage chunk");
    }

    @Test
    @Order(4)
    void anthropicMessagesAnswersFromTheModel() throws Exception {
        var response = post(
                "/v1/messages",
                "{\"model\":\"" + MODEL
                        + "\",\"max_tokens\":24,\"temperature\":0,\"messages\":["
                        + "{\"role\":\"user\",\"content\":\"What is the capital of France? Answer with one word.\"}]}");
        assertEquals(200, response.statusCode(), response.body());
        JsonNode message = JSON.readTree(response.body());
        System.out.println("Flash-Next Anthropic message: " + message);
        assertEquals("message", message.get("type").asString());
        StringBuilder text = new StringBuilder();
        for (JsonNode block : message.get("content"))
            if (block.has("text")) text.append(block.get("text").asString());
        assertTrue(text.toString().toLowerCase(java.util.Locale.ROOT).contains("paris"), text.toString());
    }

    private URI uri(String path) {
        return URI.create("http://localhost:" + this.port + path);
    }

    private HttpResponse<String> get(String path) throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            return client.send(HttpRequest.newBuilder(uri(path)).build(), HttpResponse.BodyHandlers.ofString());
        }
    }

    private HttpResponse<String> post(String path, String body) throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            return client.send(
                    HttpRequest.newBuilder(uri(path))
                            .header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(body))
                            .build(),
                    HttpResponse.BodyHandlers.ofString());
        }
    }
}
