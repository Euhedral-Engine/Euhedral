package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.micrometer.core.instrument.MeterRegistry;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Import;

/// A backend that holds two sequences runs two generations at once; the next request waits or, with no queue, is
/// refused, and a slot freed by completion or by a client leaving admits it.
// Contexts with equal configuration are cached and shared, backend included: one such class at a time.
@ResourceLock("scripted-api-context")
@SpringBootTest(
        classes = ScriptedApiApplication.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = {"euhedral.test.scripted-api=true", "euhedral.test.max-queued=0"})
@Import(GenerationSlotsTest.TwoSlots.class)
@Timeout(60)
class GenerationSlotsTest {
    private static final String BODY = "{\"model\":\"" + ScriptedInferenceBackend.MODEL_ID
            + "\",\"reasoning_effort\":\"none\",\"stream\":true,\"max_tokens\":3000,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}";
    private static final String SHORT = "{\"model\":\"" + ScriptedInferenceBackend.MODEL_ID
            + "\",\"reasoning_effort\":\"none\",\"max_tokens\":3,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}";

    /// Raises the slots before any request, as the engine reports them when it loads.
    @TestConfiguration
    static class TwoSlots {
        @Bean
        Object twoSlots(ScriptedInferenceBackend backend) {
            backend.maxActive = 2;
            return new Object();
        }
    }

    @LocalServerPort
    private int port;

    @Autowired
    private ScriptedInferenceBackend backend;

    @Autowired
    private MeterRegistry registry;

    @BeforeEach
    void freeSlots() throws InterruptedException {
        await(() -> this.registry.get("euhedral.generations.active").gauge().value() == 0);
        this.backend.reset();
    }

    private static void await(BooleanSupplier condition) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (!condition.getAsBoolean()) {
            if (System.nanoTime() > deadline) throw new AssertionError("condition never held");
            Thread.sleep(5);
        }
    }

    private int status(HttpClient client, String body) throws Exception {
        return client.send(
                        HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/chat/completions"))
                                .header("Content-Type", "application/json")
                                .POST(HttpRequest.BodyPublishers.ofString(body))
                                .build(),
                        HttpResponse.BodyHandlers.ofString())
                .statusCode();
    }

    private double active() {
        return this.registry.get("euhedral.generations.active").gauge().value();
    }

    @Test
    void twoSlotsRunTwoGenerationsAndQueueTheThird() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(20);
        try (var first = SseTestClient.post(this.port, "/v1/chat/completions", BODY);
                var second = SseTestClient.post(this.port, "/v1/chat/completions", BODY);
                var client = HttpClient.newHttpClient()) {
            first.readUntil(line -> line.contains("\"tok0 \""));
            second.readUntil(line -> line.contains("\"tok0 \""));
            assertEquals(2.0, active());
            assertEquals(503, status(client, SHORT));
            assertEquals(2, this.backend.generations.size(), "a refused request opens no session");
            // The client of the first leaves: its slot admits a new request.
            first.close();
            // Which of the two opened first is not fixed; exactly one of them closes.
            await(() -> this.backend.generations.stream().anyMatch(generation -> generation.closeCount.get() > 0));
            this.backend.script = ScriptedInferenceBackend.tokens(java.util.List.of("ok"), true);
            await(() -> this.active() == 1);
            assertEquals(200, status(client, SHORT));
            assertEquals(3, this.backend.generations.size());
        }
        for (var generation : this.backend.generations) assertTrue(generation.closed.await(10, TimeUnit.SECONDS));
        await(() -> active() == 0);
    }

    /// The job cannot be built after the generation opened: the session closes, the request fails, the slot frees.
    @Test
    void aStartThatFailsAfterOpeningReleasesItsSlot() throws Exception {
        this.backend.failNextStart = true;
        try (var client = HttpClient.newHttpClient()) {
            assertEquals(500, status(client, SHORT));
            assertTrue(this.backend.only().closed.await(10, TimeUnit.SECONDS));
            await(() -> active() == 0);
            assertEquals(200, status(client, SHORT));
        }
    }

    /// A stream already answered 200 when its job fails to build: the stream ends instead of idling to the timeout.
    @Test
    void aStreamWhoseStartFailsAfterOpeningEndsAndReleasesItsSlot() throws Exception {
        this.backend.failNextStart = true;
        try (var stream = SseTestClient.post(this.port, "/v1/chat/completions", BODY)) {
            assertThrows(java.io.IOException.class, () -> stream.readUntil(line -> line.contains("never")));
        }
        assertTrue(this.backend.only().closed.await(10, TimeUnit.SECONDS));
        await(() -> active() == 0);
    }

    @Test
    void completionsReleaseTheirSlots() throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            for (int request = 0; request < 5; request++) assertEquals(200, status(client, SHORT));
        }
        await(() -> active() == 0);
        assertEquals(5, this.backend.generations.size());
    }
}
