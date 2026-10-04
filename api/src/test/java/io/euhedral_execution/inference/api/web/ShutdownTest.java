package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.micrometer.core.instrument.MeterRegistry;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;

/// A shutdown first waits for the requests in flight, queued ones included, up to the drain window
/// (`spring.lifecycle.timeout-per-shutdown-phase`); what is left when it ends is answered before the server stops:
/// the running generation is stopped and its stream ends with an error event, and a queued request is refused
/// with 503. Each test starts its own server, because it closes it.
@Timeout(60)
class ShutdownTest {
    private static final String STREAM = "{\"model\":\"" + ScriptedInferenceBackend.MODEL_ID
            + "\",\"reasoning_effort\":\"none\",\"stream\":true,\"max_tokens\":%d,"
            + "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}";

    private ConfigurableApplicationContext context;
    private ScriptedInferenceBackend backend;
    private int port;

    private void start(String drainWindow) {
        this.context = new SpringApplicationBuilder(ScriptedApiApplication.class)
                .properties(
                        "euhedral.test.scripted-api=true",
                        "euhedral.api.max-queued-generations=1",
                        "server.port=0",
                        "spring.lifecycle.timeout-per-shutdown-phase=" + drainWindow)
                .run();
        this.backend = this.context.getBean(ScriptedInferenceBackend.class);
        this.port = Integer.parseInt(this.context.getEnvironment().getProperty("local.server.port"));
    }

    @AfterEach
    void close() {
        if (this.context != null) this.context.close();
    }

    @Test
    void runningAndQueuedRequestsFinishWithinTheDrainWindow() throws Exception {
        start("20s");
        // 20 tokens 50 ms apart: each generation takes a second.
        this.backend.script = ScriptedInferenceBackend.endless(50);
        try (var running = SseTestClient.post(this.port, "/v1/chat/completions", STREAM.formatted(20));
                var client = HttpClient.newHttpClient()) {
            running.readUntil(line -> line.contains("\"tok0 \""));
            CompletableFuture<HttpResponse<String>> queued =
                    client.sendAsync(json(20), HttpResponse.BodyHandlers.ofString());
            awaitQueued();
            CompletableFuture<Void> shutdown = CompletableFuture.runAsync(this.context::close);

            List<String> stream = running.readUntil(line -> line.contains("[DONE]"));
            assertTrue(
                    stream.stream().anyMatch(line -> line.contains("\"finish_reason\":\"length\"")), stream.toString());
            HttpResponse<String> answer = queued.get(20, TimeUnit.SECONDS);
            assertEquals(200, answer.statusCode(), answer.body());
            assertTrue(answer.body().contains("tok19"), answer.body());
            shutdown.get(20, TimeUnit.SECONDS);
        }
        assertEquals(2, this.backend.generations.size());
    }

    @Test
    void whatIsLeftAfterTheDrainWindowIsAnsweredBeforeTheServerStops() throws Exception {
        start("1s");
        this.backend.script = ScriptedInferenceBackend.endless(20);
        try (var running = SseTestClient.post(this.port, "/v1/chat/completions", STREAM.formatted(3000));
                var client = HttpClient.newHttpClient()) {
            running.readUntil(line -> line.contains("\"tok0 \""));
            CompletableFuture<HttpResponse<String>> queued =
                    client.sendAsync(json(3000), HttpResponse.BodyHandlers.ofString());
            awaitQueued();
            CompletableFuture<Void> shutdown = CompletableFuture.runAsync(this.context::close);

            List<String> stream = running.readUntil(line -> line.contains("\"error\""));
            assertTrue(stream.getLast().contains("interrupted by server shutdown"), stream.getLast());
            HttpResponse<String> refused = queued.get(20, TimeUnit.SECONDS);
            assertEquals(503, refused.statusCode(), refused.body());
            assertTrue(refused.body().contains("shutting down"), refused.body());
            shutdown.get(20, TimeUnit.SECONDS);
        }
        assertEquals(1, this.backend.generations.size(), "the queued request opened no session");
        assertTrue(this.backend.generations.getFirst().awaitClosed());
    }

    /// Until the second request waits behind the first. A connection the server has not yet accepted when the
    /// shutdown begins is reset instead of answered.
    private void awaitQueued() throws InterruptedException {
        var queued = this.context
                .getBean(MeterRegistry.class)
                .get("euhedral.generations.queued")
                .gauge();
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20);
        while (queued.value() < 1) {
            if (System.nanoTime() > deadline) throw new AssertionError("the second request was never queued");
            Thread.sleep(5);
        }
    }

    private HttpRequest json(int maxTokens) {
        return HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(
                        STREAM.formatted(maxTokens).replace("\"stream\":true", "\"stream\":false")))
                .build();
    }
}
