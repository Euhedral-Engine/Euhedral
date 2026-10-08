package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.micrometer.core.instrument.MeterRegistry;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;

/// With one generation slot and no queue, a second concurrent request is refused before it opens a session,
/// and the client receives an OpenAI 503.
// Contexts with equal configuration are cached and shared, backend included: one such class at a time.
@ResourceLock("scripted-api-context")
@SpringBootTest(
        classes = ScriptedApiApplication.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = {"euhedral.test.scripted-api=true", "euhedral.api.max-queued-generations=0"})
@Timeout(60)
class GenerationCapacityTest {
    private static final String BODY = "{\"model\":\"" + ScriptedInferenceBackend.MODEL_ID
            + "\",\"reasoning_effort\":\"none\",\"stream\":true,\"max_tokens\":3000,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}";

    @LocalServerPort
    private int port;

    @Autowired
    private ScriptedInferenceBackend backend;

    @Autowired
    private MeterRegistry registry;

    /// The previous test's generation may still hold the slot for a moment after its session closed.
    @BeforeEach
    void freeSlot() throws InterruptedException {
        var active = this.registry.get("euhedral.generations.active").gauge();
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (active.value() > 0) {
            if (System.nanoTime() > deadline) throw new AssertionError("the generation slot was never released");
            Thread.sleep(5);
        }
        this.backend.reset();
    }

    @Test
    void saturatedServerRejectsWith503WithoutOpeningASession() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(20);
        try (var busy = SseTestClient.post(this.port, "/v1/chat/completions", BODY);
                var client = HttpClient.newHttpClient()) {
            busy.readUntil(line -> line.contains("\"tok0 \""));
            var response = client.send(
                    HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/chat/completions"))
                            .header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(BODY))
                            .build(),
                    HttpResponse.BodyHandlers.ofString());
            assertEquals(503, response.statusCode());
            assertTrue(response.body().contains("\"type\":\"service_unavailable_error\""), response.body());
            assertEquals(1, this.backend.generations.size(), "a rejected request opens no session");
        }
        var running = this.backend.generations.getFirst();
        assertTrue(running.closed.await(10, TimeUnit.SECONDS));
    }

    /// Anthropic's clients read 529 as overloaded and retry it.
    @Test
    void messagesAreRefusedWithAnthropicsOverloadedStatus() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(20);
        try (var busy = SseTestClient.post(this.port, "/v1/chat/completions", BODY);
                var client = HttpClient.newHttpClient()) {
            busy.readUntil(line -> line.contains("\"tok0 \""));
            var response = client.send(
                    HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/messages"))
                            .header("Content-Type", "application/json")
                            .POST(
                                    HttpRequest.BodyPublishers.ofString(
                                            "{\"model\":\"" + ScriptedInferenceBackend.MODEL_ID
                                                    + "\",\"max_tokens\":5,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"))
                            .build(),
                    HttpResponse.BodyHandlers.ofString());
            assertEquals(529, response.statusCode());
            assertTrue(response.body().contains("\"type\":\"overloaded_error\""), response.body());
        }
        assertTrue(this.backend.generations.getFirst().closed.await(10, TimeUnit.SECONDS));
    }
}
