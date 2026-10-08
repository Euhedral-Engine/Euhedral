package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.api.engine.InferenceBackend;
import java.io.ByteArrayInputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/// Streaming and disconnect behavior against a real Tomcat connector, without CUDA.
// Contexts with equal configuration are cached and shared, backend included: one such class at a time.
@ResourceLock("scripted-api-context")
@SpringBootTest(
        classes = ScriptedApiApplication.class,
        properties = "euhedral.test.scripted-api=true",
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Timeout(60)
class ChatCompletionStreamingTest {
    private static final String MODEL = ScriptedInferenceBackend.MODEL_ID;
    private static final JsonMapper JSON = JsonMapper.shared();

    @LocalServerPort
    private int port;

    @Autowired
    private ScriptedInferenceBackend backend;

    @BeforeEach
    void resetBackend() {
        this.backend.reset();
    }

    @Test
    void chunkedOversizedToolSchemaIsRejectedWithoutAContentLength() throws Exception {
        String body = "{\"model\":\"" + MODEL
                + "\",\"reasoning_effort\":\"none\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],"
                + "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"description\":\""
                + "x".repeat(1_048_576) + "\"}}]}";
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        var request = HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofInputStream(() -> new ByteArrayInputStream(bytes)))
                .build();
        try (var client = HttpClient.newHttpClient()) {
            var response = client.send(request, HttpResponse.BodyHandlers.ofString());
            assertEquals(413, response.statusCode());
            assertEquals(
                    "request_too_large",
                    JSON.readTree(response.body()).get("error").get("code").asString());
        }
        assertTrue(this.backend.generations.isEmpty(), "oversized requests must not open sessions");
    }

    @Test
    void streamEmitsFramedChunksIncrementallyThenDone() throws Exception {
        var firstChunkSeen = new CountDownLatch(1);
        this.backend.script = (generation, max, output) -> {
            generation.emit(output, "Hel");
            // Generation cannot continue until the client has received "Hel": proves nothing is buffered.
            if (!firstChunkSeen.await(10, TimeUnit.SECONDS))
                throw new ExecutionException(new AssertionError("first chunk was not delivered while generating"));
            generation.emit(output, "lo");
            return new InferenceBackend.Result(3, true);
        };
        var request = HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(
                        "{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"stream\":true,"
                                + "\"stream_options\":{\"include_usage\":true},"
                                + "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"))
                .build();
        HttpResponse<java.util.stream.Stream<String>> response;
        try (var client = HttpClient.newHttpClient()) {
            response = client.send(request, HttpResponse.BodyHandlers.ofLines());
            assertEquals(200, response.statusCode());
            assertTrue(
                    response.headers().firstValue("Content-Type").orElseThrow().startsWith("text/event-stream"));

            List<String> data = new ArrayList<>();
            var lines = response.body().iterator();
            boolean expectData = true;
            while (lines.hasNext()) {
                String line = lines.next();
                if (expectData) {
                    assertTrue(line.startsWith("data:"), "each event must be a single data field: " + line);
                    data.add(line.substring("data:".length()).strip());
                    if (data.getLast().contains("\"Hel\"")) firstChunkSeen.countDown();
                } else {
                    assertEquals("", line, "events must be separated by a blank line");
                }
                expectData = !expectData;
            }

            assertEquals("[DONE]", data.getLast());
            List<JsonNode> chunks = new ArrayList<>();
            for (String event : data.subList(0, data.size() - 1)) chunks.add(JSON.readTree(event));
            assertEquals(5, chunks.size(), data.toString());
            String id = chunks.getFirst().get("id").asString();
            for (JsonNode chunk : chunks) {
                assertEquals(id, chunk.get("id").asString());
                assertEquals("chat.completion.chunk", chunk.get("object").asString());
                assertEquals(MODEL, chunk.get("model").asString());
            }
            JsonNode role = chunks.get(0).get("choices").get(0);
            assertEquals("assistant", role.get("delta").get("role").asString());
            assertTrue(role.get("finish_reason").isNull());
            assertEquals(
                    "Hel",
                    chunks.get(1)
                            .get("choices")
                            .get(0)
                            .get("delta")
                            .get("content")
                            .asString());
            assertEquals(
                    "lo",
                    chunks.get(2)
                            .get("choices")
                            .get(0)
                            .get("delta")
                            .get("content")
                            .asString());
            JsonNode terminal = chunks.get(3).get("choices").get(0);
            assertEquals("stop", terminal.get("finish_reason").asString());
            assertTrue(terminal.get("delta").isEmpty());
            JsonNode usage = chunks.get(4);
            assertTrue(usage.get("choices").isEmpty());
            assertEquals(3, usage.get("usage").get("completion_tokens").asInt());
        }
        var generation = this.backend.only();
        assertTrue(generation.awaitClosed());
        assertEquals(1, generation.closeCount.get());
    }

    /// Text streams as it is decoded; JSON tool calls arrive complete in `delta.tool_calls` chunks.
    @Test
    void streamDeliversTextThenToolCallDeltasThenToolCallsFinish() throws Exception {
        var textSeen = new CountDownLatch(1);
        String call = "{\"tool_calls\":[{\"name\":\"f\",\"arguments\":{\"n\":7}},"
                + "{\"name\":\"g\",\"arguments\":{\"n\":7}}]}";
        this.backend.script = (generation, max, output) -> {
            generation.emit(output, "Checking.");
            // The call cannot begin until the client holds the text: tool mode must not buffer content.
            if (!textSeen.await(10, TimeUnit.SECONDS))
                throw new ExecutionException(new AssertionError("text was not delivered while generating"));
            generation.emit(output, "\n\n{\"tool_");
            generation.emit(output, call.substring("{\"tool_".length()));
            return new InferenceBackend.Result(4, true);
        };
        String tools = "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\","
                + "\"properties\":{\"n\":{\"type\":\"integer\"}}}}},"
                + "{\"type\":\"function\",\"function\":{\"name\":\"g\"}}]";
        List<String> data = streamEvents(
                "{\"model\":\"" + MODEL
                        + "\",\"reasoning_effort\":\"none\",\"stream\":true,\"stream_options\":{\"include_usage\":true},"
                        + "\"tools\":" + tools + ",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}",
                event -> {
                    if (event.contains("\"Checking.\"")) textSeen.countDown();
                });

        assertEquals("[DONE]", data.getLast());
        List<JsonNode> chunks = new ArrayList<>();
        for (String event : data.subList(0, data.size() - 1)) chunks.add(JSON.readTree(event));
        assertEquals(6, chunks.size(), data.toString());
        assertEquals(
                "assistant",
                chunks.get(0).get("choices").get(0).get("delta").get("role").asString());
        assertEquals(
                "Checking.",
                chunks.get(1).get("choices").get(0).get("delta").get("content").asString());
        List<String> ids = new ArrayList<>();
        for (int index = 0; index < 2; index++) {
            JsonNode choice = chunks.get(2 + index).get("choices").get(0);
            assertTrue(choice.get("finish_reason").isNull());
            JsonNode delta = choice.get("delta");
            assertTrue(delta.get("content") == null, delta.toString());
            assertEquals(1, delta.get("tool_calls").size());
            JsonNode toolCall = delta.get("tool_calls").get(0);
            assertEquals(index, toolCall.get("index").asInt());
            assertEquals("function", toolCall.get("type").asString());
            assertTrue(toolCall.get("id").asString().startsWith("call_"));
            ids.add(toolCall.get("id").asString());
            assertEquals(
                    index == 0 ? "f" : "g", toolCall.get("function").get("name").asString());
            assertEquals(
                    index == 0 ? "{\"n\":7}" : "{\"n\":7}",
                    toolCall.get("function").get("arguments").asString());
        }
        assertTrue(!ids.get(0).equals(ids.get(1)), "tool call IDs must be unique");
        JsonNode terminal = chunks.get(4).get("choices").get(0);
        assertEquals("tool_calls", terminal.get("finish_reason").asString());
        assertTrue(terminal.get("delta").isEmpty());
        assertEquals(4, chunks.get(5).get("usage").get("completion_tokens").asInt());
        for (String event : data) assertTrue(!event.contains("<tool_call>") && !event.contains("<function"), event);
        var generation = this.backend.only();
        assertTrue(generation.awaitClosed());
        assertEquals(1, generation.closeCount.get());
    }

    /// Text held back as a possible stop-sequence prefix is released before the call that follows it.
    @Test
    void streamReleasesHeldBackTextBeforeTheToolCall() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(
                List.of("Answer ST", "{\"tool_calls\":[{\"name\":\"f\",\"arguments\":{}}]}"), true);
        List<String> data = streamEvents(
                "{\"model\":\"" + MODEL
                        + "\",\"reasoning_effort\":\"none\",\"stream\":true,\"stop\":[\"STOP\"],\"tools\":[{\"type\":\"function\","
                        + "\"function\":{\"name\":\"f\"}}],\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}",
                event -> {});
        List<String> order = new ArrayList<>();
        StringBuilder text = new StringBuilder();
        for (String event : data.subList(1, data.size() - 1)) {
            JsonNode choice = JSON.readTree(event).get("choices").get(0);
            JsonNode delta = choice.get("delta");
            if (delta.has("content")) {
                assertTrue(order.isEmpty(), "text must arrive before the call");
                text.append(delta.get("content").asString());
            } else if (delta.has("tool_calls")) order.add("call");
            else order.add("finish:" + choice.get("finish_reason").asString());
        }
        assertEquals("Answer ST", text.toString());
        assertEquals(List.of("call", "finish:tool_calls"), order);
    }

    /// An invalid call after streamed text ends the stream with an in-band error and no `[DONE]`.
    @Test
    void streamReportsAnInvalidToolCallInBand() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(
                List.of("Sure.", " <tool_call>\n<function=nope>\n</function>\n</tool_call>", "tail"), false);
        List<String> data = streamEvents(
                "{\"model\":\"" + MODEL
                        + "\",\"reasoning_effort\":\"none\",\"stream\":true,\"tools\":[{\"type\":\"function\","
                        + "\"function\":{\"name\":\"f\"}}],\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}",
                event -> {});
        JsonNode error = JSON.readTree(data.getLast());
        assertEquals("invalid_tool_call", error.get("error").get("code").asString());
        assertTrue(data.stream().noneMatch(event -> event.equals("[DONE]")), data.toString());
        assertEquals(
                "Sure.",
                JSON.readTree(data.get(1))
                        .get("choices")
                        .get(0)
                        .get("delta")
                        .get("content")
                        .asString());
        var generation = this.backend.only();
        assertTrue(generation.awaitClosed());
        assertTrue(generation.isCancelled(), "an invalid call must stop decoding");
        assertEquals(2, generation.emitted.get());
    }

    private List<String> streamEvents(String body, Consumer<String> onEvent) throws Exception {
        var request = HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
        try (var client = HttpClient.newHttpClient()) {
            HttpResponse<java.util.stream.Stream<String>> response =
                    client.send(request, HttpResponse.BodyHandlers.ofLines());
            assertEquals(200, response.statusCode());
            List<String> data = new ArrayList<>();
            var lines = response.body().iterator();
            boolean expectData = true;
            while (lines.hasNext()) {
                String line = lines.next();
                if (expectData) {
                    assertTrue(line.startsWith("data:"), "each event must be a single data field: " + line);
                    data.add(line.substring("data:".length()).strip());
                    onEvent.accept(data.getLast());
                } else {
                    assertEquals("", line, "events must be separated by a blank line");
                }
                expectData = !expectData;
            }
            return data;
        }
    }

    @Test
    void streamingClientAbortCancelsAndClosesTheSession() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(20);
        try (var client = SseTestClient.post(
                this.port,
                "/v1/chat/completions",
                "{\"model\":\"" + MODEL
                        + "\",\"reasoning_effort\":\"none\",\"stream\":true,\"max_tokens\":3000,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}")) {
            client.readUntil(line -> line.contains("\"tok1 \""));
        }
        var generation = this.backend.only();
        assertTrue(generation.awaitClosed(), "session must be closed after the client disconnects");
        assertTrue(generation.isCancelled(), "disconnect must cancel generation");
        int emitted = generation.emitted.get();
        assertTrue(emitted < 500, "generation kept running after disconnect: " + emitted + " tokens");
        Thread.sleep(200);
        assertEquals(emitted, generation.emitted.get(), "no quantum may run after the session closed");
        assertEquals(1, generation.closeCount.get());
    }

    /// Tomcat does not watch an idle async connection, so a JSON-mode disconnect surfaces only when the
    /// response is written. The session must still be closed exactly once when generation ends.
    @Test
    void nonStreamingClientAbortStillClosesTheSession() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(10);
        ScriptedInferenceBackend.ScriptedGeneration generation;
        try (var client = SseTestClient.post(
                this.port,
                "/v1/chat/completions",
                "{\"model\":\"" + MODEL
                        + "\",\"reasoning_effort\":\"none\",\"max_tokens\":50,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}")) {
            generation = awaitGeneration();
            assertTrue(generation.started.await(10, TimeUnit.SECONDS));
        }
        assertTrue(generation.awaitClosed(), "session must be closed after an abandoned request finishes");
        assertEquals(1, generation.closeCount.get());
    }

    private ScriptedInferenceBackend.ScriptedGeneration awaitGeneration() throws InterruptedException {
        long deadline = System.nanoTime() + Duration.ofSeconds(10).toNanos();
        while (this.backend.generations.isEmpty() && System.nanoTime() < deadline) Thread.sleep(5);
        return this.backend.only();
    }
}
