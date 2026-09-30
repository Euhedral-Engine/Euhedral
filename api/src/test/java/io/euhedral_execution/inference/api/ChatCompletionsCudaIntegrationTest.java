package io.euhedral_execution.inference.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.SystemInfo;
import io.euhedral_execution.inference.api.engine.EngineInferenceBackend;
import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.core.InferenceEngine;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.Socket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.StringJoiner;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.Timeout;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/// Starts the real application against the real `InferenceEngine` and drives it over HTTP.
///
/// Run through `gradle :api:cudaIntegrationTest`. The production backend is wrapped only to count session
/// open/close/cancel; generation runs entirely through the engine. The engine's device allocations are
/// compared before and after each request to confirm sequence state was released.
@SpringBootTest(
        classes = {EuhedralInferenceApplication.class, ChatCompletionsCudaIntegrationTest.Tracking.class},
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@DirtiesContext
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@Timeout(1800)
class ChatCompletionsCudaIntegrationTest {
    private static final String MODEL = "qwen-test";
    private static final JsonMapper JSON = JsonMapper.shared();
    private static Path library;
    private static Path artifact;
    private static Path tokenizer;

    @LocalServerPort
    private int port;

    @Autowired
    private InferenceEngine engine;

    @Autowired
    private TrackingBackend tracking;

    /// Runs before the Spring context loads, so a machine without CUDA assets skips instead of failing.
    @BeforeAll
    static void requireCudaAssets() {
        String configuredLibrary = System.getProperty("euhedral.cuda.library");
        assumeTrue(configuredLibrary != null && Files.isRegularFile(Path.of(configuredLibrary)));
        library = Path.of(configuredLibrary);
        artifact = Path.of(System.getProperty(
                "euhedral.qwen.artifact", "/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl"));
        tokenizer = Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(artifact) && Files.isRegularFile(tokenizer.resolve("chat_template.jinja")));
    }

    @DynamicPropertySource
    static void engineProperties(DynamicPropertyRegistry registry) {
        registry.add("euhedral.inference.artifact-path", () -> artifact.toString());
        registry.add("euhedral.inference.tokenizer-directory", () -> tokenizer.toString());
        registry.add("euhedral.inference.cuda-library-path", () -> library.toString());
        registry.add("euhedral.inference.worker-cpus", ChatCompletionsCudaIntegrationTest::twoPerformanceCpus);
        registry.add("euhedral.inference.model-id", () -> MODEL);
    }

    @Test
    @Order(1)
    void modelsAndHealthExposeOnlyThePublicModel() throws Exception {
        var models = JSON.readTree(get("/v1/models").body());
        assertEquals(MODEL, models.get("data").get(0).get("id").asString());
        var health = get("/health");
        assertEquals(200, health.statusCode());
        assertEquals("ready", JSON.readTree(health.body()).get("engine").asString());
        assertFalse(health.body().contains("/mnt/") || models.toString().contains(".edrl"));
    }

    @Test
    @Order(2)
    void chatCompletionReturnsARealAssistantAnswer() throws Exception {
        long before = allocatedDeviceBytes();
        var response = post("{\"model\":\"" + MODEL + "\",\"temperature\":0,\"max_tokens\":24,\"messages\":["
                + "{\"role\":\"system\",\"content\":\"Answer with a single word.\"},"
                + "{\"role\":\"user\",\"content\":\"What is the capital of France?\"}]}");
        assertEquals(200, response.statusCode(), response.body());
        JsonNode completion = JSON.readTree(response.body());
        System.out.println("Non-streaming completion: " + completion);
        assertEquals("chat.completion", completion.get("object").asString());
        assertEquals(MODEL, completion.get("model").asString());
        JsonNode choice = completion.get("choices").get(0);
        assertEquals("assistant", choice.get("message").get("role").asString());
        String content = choice.get("message").get("content").asString();
        assertTrue(content.toLowerCase(java.util.Locale.ROOT).contains("paris"), content);
        assertFalse(content.contains("<|im_end|>") || content.contains("<think>"), content);
        assertTrue(
                List.of("stop", "length").contains(choice.get("finish_reason").asString()));

        JsonNode usage = completion.get("usage");
        int promptTokens = usage.get("prompt_tokens").asInt();
        int completionTokens = usage.get("completion_tokens").asInt();
        assertTrue(promptTokens > 10 && completionTokens > 0 && completionTokens <= 24, usage.toString());
        assertEquals(promptTokens + completionTokens, usage.get("total_tokens").asInt());

        assertSessionsReleased(before);
    }

    @Test
    @Order(3)
    void streamingEmitsIncrementalChunksAndCleansUp() throws Exception {
        long before = allocatedDeviceBytes();
        var request = HttpRequest.newBuilder(uri("/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString("{\"model\":\"" + MODEL + "\",\"stream\":true,"
                        + "\"stream_options\":{\"include_usage\":true},\"temperature\":0,\"max_tokens\":32,"
                        + "\"messages\":[{\"role\":\"user\",\"content\":\"Count from one to ten in words.\"}]}"))
                .build();
        List<String> data = new ArrayList<>();
        try (var client = HttpClient.newHttpClient()) {
            var response = client.send(request, HttpResponse.BodyHandlers.ofLines());
            assertEquals(200, response.statusCode());
            assertTrue(
                    response.headers().firstValue("Content-Type").orElseThrow().startsWith("text/event-stream"));
            boolean expectData = true;
            for (String line : (Iterable<String>) response.body()::iterator) {
                if (expectData) {
                    assertTrue(line.startsWith("data:"), line);
                    data.add(line.substring("data:".length()).strip());
                } else {
                    assertEquals("", line);
                }
                expectData = !expectData;
            }
        }
        assertEquals("[DONE]", data.getLast());
        List<JsonNode> chunks = new ArrayList<>();
        for (String event : data.subList(0, data.size() - 1)) chunks.add(JSON.readTree(event));
        assertEquals(
                "assistant",
                chunks.getFirst().get("choices").get(0).get("delta").get("role").asString());
        var text = new StringJoiner("");
        int contentChunks = 0;
        String finishReason = null;
        JsonNode usage = null;
        for (JsonNode chunk : chunks.subList(1, chunks.size())) {
            assertEquals("chat.completion.chunk", chunk.get("object").asString());
            if (chunk.get("choices").isEmpty()) {
                usage = chunk.get("usage");
                continue;
            }
            JsonNode choice = chunk.get("choices").get(0);
            if (!choice.get("finish_reason").isNull())
                finishReason = choice.get("finish_reason").asString();
            else if (choice.get("delta").has("content")) {
                text.add(choice.get("delta").get("content").asString());
                contentChunks++;
            }
        }
        System.out.println("Streamed " + contentChunks + " chunks: " + text);
        assertTrue(contentChunks > 1, "expected incremental content, got " + contentChunks + " chunk(s)");
        assertTrue(text.toString().toLowerCase(java.util.Locale.ROOT).contains("three"), text.toString());
        assertTrue(List.of("stop", "length").contains(finishReason), String.valueOf(finishReason));
        assertTrue(usage != null && usage.get("completion_tokens").asInt() > 1);

        assertSessionsReleased(before);
    }

    @Test
    @Order(4)
    void clientAbortCancelsGenerationAndClosesTheSession() throws Exception {
        long before = allocatedDeviceBytes();
        int maxTokens = 2000;
        String body = "{\"model\":\"" + MODEL + "\",\"stream\":true,\"max_tokens\":" + maxTokens
                + ",\"messages\":[{\"role\":\"user\",\"content\":\"Write a very long story about a lighthouse.\"}]}";
        this.tracking.resetCounts();
        try (var socket = new Socket("localhost", this.port)) {
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            socket.getOutputStream()
                    .write(("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                                    + "Content-Type: application/json\r\nContent-Length: " + bytes.length + "\r\n\r\n")
                            .getBytes(StandardCharsets.US_ASCII));
            socket.getOutputStream().write(bytes);
            socket.getOutputStream().flush();
            var reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
            int contentChunks = 0;
            for (String line; contentChunks < 3 && (line = reader.readLine()) != null; ) {
                if (line.contains("\"content\":\"") && !line.contains("\"role\"")) contentChunks++;
            }
            assertEquals(3, contentChunks, "stream ended before the client aborted");
            socket.setSoLinger(true, 0);
        }
        long aborted = System.nanoTime();
        assertTrue(this.tracking.closed.await(60, TimeUnit.SECONDS), "session was not closed after abort");
        System.out.println("Session closed "
                + Duration.ofNanos(System.nanoTime() - aborted).toMillis() + " ms after abort; tokens sampled: "
                + this.tracking.lastCompletionTokens.get());
        assertEquals(1, this.tracking.cancelled.get(), "abort must cancel the session");
        assertTrue(
                this.tracking.lastCompletionTokens.get() < maxTokens,
                "generation ran to its limit after the client left");
        assertSessionsReleased(before);
    }

    private void assertSessionsReleased(long allocatedBefore) {
        assertEquals(this.tracking.opened.get(), this.tracking.closedCount.get(), "every session must be closed");
        assertEquals(allocatedBefore, allocatedDeviceBytes(), "sequence device memory was not released");
    }

    private long allocatedDeviceBytes() {
        return this.engine.allocatedDeviceBytes();
    }

    private HttpResponse<String> get(String path) throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            return client.send(HttpRequest.newBuilder(uri(path)).build(), HttpResponse.BodyHandlers.ofString());
        }
    }

    private HttpResponse<String> post(String body) throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            return client.send(
                    HttpRequest.newBuilder(uri("/v1/chat/completions"))
                            .header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(body))
                            .build(),
                    HttpResponse.BodyHandlers.ofString());
        }
    }

    private URI uri(String path) {
        return URI.create("http://localhost:" + this.port + path);
    }

    private static String twoPerformanceCpus() {
        var available = SystemInfo.getPCpuSet();
        var selected = new StringJoiner(",");
        int count = 0;
        for (int cpu = available.nextSetBit(0); cpu >= 0 && count < 2; cpu = available.nextSetBit(cpu + 1)) {
            selected.add(Integer.toString(cpu));
            count++;
        }
        return selected.toString();
    }

    @TestConfiguration(proxyBeanMethods = false)
    static class Tracking {
        @Bean
        @Primary
        TrackingBackend trackingBackend(EngineInferenceBackend engineBackend) {
            return new TrackingBackend(engineBackend);
        }
    }

    /// Delegates every call to the production backend and records session lifecycle events.
    static final class TrackingBackend implements InferenceBackend {
        private final InferenceBackend delegate;
        final AtomicInteger opened = new AtomicInteger();
        final AtomicInteger closedCount = new AtomicInteger();
        final AtomicInteger cancelled = new AtomicInteger();
        final AtomicInteger lastCompletionTokens = new AtomicInteger();
        volatile CountDownLatch closed = new CountDownLatch(1);

        TrackingBackend(InferenceBackend delegate) {
            this.delegate = delegate;
        }

        void resetCounts() {
            this.cancelled.set(0);
            this.closed = new CountDownLatch(1);
        }

        @Override
        public String modelId() {
            return this.delegate.modelId();
        }

        @Override
        public boolean isAvailable() {
            return this.delegate.isAvailable();
        }

        @Override
        public int contextLength() {
            return this.delegate.contextLength();
        }

        @Override
        public int countPromptTokens(String prompt) {
            return this.delegate.countPromptTokens(prompt);
        }

        @Override
        public Generation openGeneration(GenerationConfig config, ToolConstraint constraint) {
            Generation generation = this.delegate.openGeneration(config, constraint);
            this.opened.incrementAndGet();
            return new Generation() {
                @Override
                public Result generate(String prompt, int maxNewTokens, Consumer<String> output)
                        throws InterruptedException, ExecutionException {
                    Result result = generation.generate(prompt, maxNewTokens, output);
                    TrackingBackend.this.lastCompletionTokens.set(result.completionTokens());
                    return result;
                }

                @Override
                public void cancel() {
                    if (!generation.isCancelled()) TrackingBackend.this.cancelled.incrementAndGet();
                    generation.cancel();
                }

                @Override
                public boolean isCancelled() {
                    return generation.isCancelled();
                }

                @Override
                public void close() {
                    generation.close();
                    TrackingBackend.this.closedCount.incrementAndGet();
                    TrackingBackend.this.closed.countDown();
                }
            };
        }
    }
}
