package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.api.chat.GenerationService;
import io.micrometer.core.instrument.MeterRegistry;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.context.annotation.Import;

/// Two slots and a queue: a freed slot starts a waiting request, and a waiting request started after shutdown
/// began is refused.
// Contexts with equal configuration are cached and shared, backend included: one such class at a time.
@ResourceLock("scripted-api-context")
@SpringBootTest(
        classes = ScriptedApiApplication.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = {"euhedral.test.scripted-api=true", "euhedral.api.max-queued-generations=2"})
@Import(GenerationSlotsTest.TwoSlots.class)
@Timeout(60)
class GenerationQueueTest {
    private static final String BODY = "{\"model\":\"" + ScriptedInferenceBackend.MODEL_ID
            + "\",\"reasoning_effort\":\"none\",\"stream\":true,\"max_tokens\":3000,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}";
    private static final String SHORT = "{\"model\":\"" + ScriptedInferenceBackend.MODEL_ID
            + "\",\"reasoning_effort\":\"none\",\"max_tokens\":3,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}";

    @LocalServerPort
    private int port;

    @Autowired
    private ScriptedInferenceBackend backend;

    @Autowired
    private MeterRegistry registry;

    @Autowired
    private GenerationService service;

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

    private CompletableFuture<HttpResponse<String>> queued(HttpClient client) {
        return client.sendAsync(
                HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/chat/completions"))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(SHORT))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private double queuedCount() {
        return this.registry.get("euhedral.generations.queued").gauge().value();
    }

    @Test
    void aFreedSlotStartsTheWaitingRequestWithTwoSlotsBusy() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(20);
        try (var first = SseTestClient.post(this.port, "/v1/chat/completions", BODY);
                var second = SseTestClient.post(this.port, "/v1/chat/completions", BODY);
                var client = HttpClient.newHttpClient()) {
            first.readUntil(line -> line.contains("\"tok0 \""));
            second.readUntil(line -> line.contains("\"tok0 \""));
            var third = queued(client);
            await(() -> queuedCount() == 1);
            this.backend.script = ScriptedInferenceBackend.tokens(java.util.List.of("ok"), true);
            first.close();
            assertEquals(200, third.get(10, TimeUnit.SECONDS).statusCode());
        }
    }

    @Test
    void aWaitingRequestStartedAfterShutdownBeganIsRefusedAndFreesItsSlot() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(20);
        var hold = new CountDownLatch(1);
        try (var first = SseTestClient.post(this.port, "/v1/chat/completions", BODY);
                var second = SseTestClient.post(this.port, "/v1/chat/completions", BODY);
                var client = HttpClient.newHttpClient()) {
            first.readUntil(line -> line.contains("\"tok0 \""));
            second.readUntil(line -> line.contains("\"tok0 \""));
            var third = queued(client);
            await(() -> queuedCount() == 1);
            // The freed slot's start is held on a worker while the service shuts down.
            this.backend.entered = new CountDownLatch(1);
            this.backend.holdStarts = hold;
            first.close();
            assertTrue(this.backend.entered.await(10, TimeUnit.SECONDS), "the freed slot's start was submitted");
            var stopping = CompletableFuture.runAsync(this.service::stop);
            await(() -> !this.service.isRunning());
            hold.countDown();
            assertEquals(503, third.get(10, TimeUnit.SECONDS).statusCode());
            stopping.get(40, TimeUnit.SECONDS);
            assertEquals(2, this.backend.generations.size(), "the refused request opens no session");
        } finally {
            hold.countDown();
            this.backend.holdStarts = null;
            this.service.start();
        }
        await(() -> this.registry.get("euhedral.generations.active").gauge().value() == 0);
    }
}
