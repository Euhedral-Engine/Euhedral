package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;

/// A client that leaves stops its generation before the next quantum, however it leaves and whatever its
/// request is doing: a JSON response being generated, a stream still prefilling its prompt, or a place in the queue.
// Contexts with equal configuration are cached and shared, backend included: one such class at a time.
@ResourceLock("scripted-api-context")
@SpringBootTest(
        classes = ScriptedApiApplication.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = {"euhedral.test.scripted-api=true", "euhedral.api.max-queued-generations=1"})
@Timeout(60)
class DisconnectTest {
    private static final String MODEL = ScriptedInferenceBackend.MODEL_ID;

    @LocalServerPort
    private int port;

    @Autowired
    private ScriptedInferenceBackend backend;

    @BeforeEach
    void resetBackend() {
        this.backend.reset();
    }

    @Test
    void aJsonClientThatResetsItsConnectionStopsTheGeneration() throws Exception {
        assertStopsOnLeaving(false);
    }

    @Test
    void aJsonClientThatClosesItsConnectionStopsTheGeneration() throws Exception {
        assertStopsOnLeaving(true);
    }

    @Test
    void aStreamThatLeavesDuringPrefillStopsBeforeItsFirstToken() throws Exception {
        assertPrefillStopsOnLeaving("/v1/chat/completions", body(true, 3000));
    }

    @Test
    void anAnthropicStreamThatLeavesDuringPrefillStops() throws Exception {
        assertPrefillStopsOnLeaving(
                "/v1/messages",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":3000,\"stream\":true,"
                        + "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}");
    }

    @Test
    void aResponsesStreamThatLeavesDuringPrefillStops() throws Exception {
        assertPrefillStopsOnLeaving(
                "/v1/responses",
                "{\"model\":\"" + MODEL + "\",\"reasoning\":{\"effort\":\"none\"},\"max_output_tokens\":3000,"
                        + "\"stream\":true,\"input\":\"Hi\"}");
    }

    @Test
    void keepAlivesAreCommentsAndPingsAClientSkips() throws Exception {
        this.backend.script = ScriptedInferenceBackend.prefilling(3, 5);
        try (var client = SseTestClient.post(this.port, "/v1/chat/completions", body(true, 2))) {
            java.util.List<String> lines = client.readUntil(line -> line.contains("[DONE]"));
            assertTrue(lines.stream().anyMatch(line -> line.equals(":keep-alive")), lines.toString());
            int firstData = -1;
            for (int index = 0; index < lines.size(); index++)
                if (lines.get(index).startsWith("data:")) {
                    firstData = index;
                    break;
                }
            for (String line : lines.subList(firstData, lines.size()))
                assertTrue(!line.startsWith(":"), "keep-alives stop once the stream began: " + lines);
        }
        this.backend.generations.clear();
        this.backend.script = ScriptedInferenceBackend.prefilling(2, 5);
        try (var client = SseTestClient.post(
                this.port,
                "/v1/messages",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":2,\"stream\":true,"
                        + "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}")) {
            java.util.List<String> lines = client.readUntil(line -> line.contains("message_stop"));
            assertTrue(lines.contains("event:ping"), lines.toString());
            assertTrue(lines.indexOf("event:ping") < lines.indexOf("event:message_start"), lines.toString());
        }
    }

    private void assertPrefillStopsOnLeaving(String path, String body) throws Exception {
        this.backend.script = ScriptedInferenceBackend.prefilling(1000, 10);
        try (var client = SseTestClient.post(this.port, path, body)) {
            client.readUntil(line -> line.startsWith("HTTP/1.1 200"));
            var generation = this.backend.awaitFirst();
            while (generation.prefills.get() < 3) Thread.sleep(5);
        }
        var generation = this.backend.only();
        assertTrue(generation.awaitClosed());
        assertTrue(generation.isCancelled());
        assertEquals(0, generation.emitted.get(), "no token is generated for a client that left");
        assertTrue(generation.prefills.get() < 100, "prefill went on: " + generation.prefills.get());
    }

    @Test
    void aQueuedClientThatLeavesGivesUpItsPlace() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(10);
        try (var running = SseTestClient.post(this.port, "/v1/chat/completions", body(true, 3000))) {
            running.readUntil(line -> line.contains("tok1"));
            // The one place in the queue: taken, then given up.
            try (var queued = SseTestClient.post(this.port, "/v1/chat/completions", body(false, 5))) {
                Thread.sleep(200);
                queued.closeGracefully();
            }
            Thread.sleep(200);
            // The place is free again: this request waits instead of being refused.
            this.backend.script = ScriptedInferenceBackend.tokens(java.util.List.of("done"), true);
            CompletableFuture<HttpResponse<String>> next;
            try (var client = HttpClient.newHttpClient()) {
                next = client.sendAsync(
                        HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/chat/completions"))
                                .header("Content-Type", "application/json")
                                .timeout(Duration.ofSeconds(30))
                                .POST(HttpRequest.BodyPublishers.ofString(body(false, 5)))
                                .build(),
                        HttpResponse.BodyHandlers.ofString());
                Thread.sleep(200);
                assertTrue(!next.isDone(), "the next request was refused instead of queued");
                running.close();
                HttpResponse<String> response = next.get(30, TimeUnit.SECONDS);
                assertEquals(200, response.statusCode(), response.body());
            }
        }
        assertEquals(2, this.backend.generations.size(), "the client that left never opened a session");
    }

    private void assertStopsOnLeaving(boolean graceful) throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(10);
        try (var client = SseTestClient.post(this.port, "/v1/chat/completions", body(false, 3000))) {
            var generation = this.backend.awaitFirst();
            while (generation.emitted.get() < 3) Thread.sleep(5);
            if (graceful) client.closeGracefully();
        }
        var generation = this.backend.only();
        assertTrue(generation.awaitClosed(), "the session closes after the client left");
        assertTrue(generation.isCancelled(), "leaving cancels the generation");
        int emitted = generation.emitted.get();
        assertTrue(emitted < 100, "generation went on after the client left: " + emitted + " tokens");
        Thread.sleep(100);
        assertEquals(emitted, generation.emitted.get(), "no quantum runs after the session closed");
    }

    private static String body(boolean stream, int maxTokens) {
        return "{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"stream\":" + stream + ",\"max_tokens\":"
                + maxTokens + ",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}";
    }
}
