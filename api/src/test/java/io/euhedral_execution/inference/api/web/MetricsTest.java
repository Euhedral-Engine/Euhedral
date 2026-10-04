package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.test.annotation.DirtiesContext;

/// `/metrics` in Prometheus text format, after requests that succeed, are refused and fail.
@SpringBootTest(
        classes = ScriptedApiApplication.class,
        properties = "euhedral.test.scripted-api=true",
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@DirtiesContext(classMode = DirtiesContext.ClassMode.BEFORE_CLASS)
@Timeout(60)
class MetricsTest {
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
    void requestsTokensAndTimingsAreExported() throws Exception {
        String hello = ",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]";
        assertEquals(
                200,
                post(
                        "/v1/chat/completions",
                        "{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\"" + hello + "}"));
        assertEquals(200, post("/v1/messages", "{\"model\":\"" + MODEL + "\",\"max_tokens\":8" + hello + "}"));
        assertEquals(400, post("/v1/chat/completions", "{\"model\":\"" + MODEL + "\",\"n\":3" + hello + "}"));
        assertEquals(
                400,
                post(
                        "/v1/chat/completions",
                        "{\"model\":\"" + MODEL + "\",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
                                + "{\"name\":\"x\",\"schema\":{\"not\":{}}}}" + hello + "}"));
        this.backend.script =
                ScriptedInferenceBackend.tokens(List.of("{\"tool_calls\":[{\"name\":\"f\",\"arguments\":"), true);
        assertEquals(
                500,
                post(
                        "/v1/chat/completions",
                        "{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"tools\":[{\"type\":\"function\","
                                + "\"function\":{\"name\":\"f\"}}]" + hello + "}"));

        String metrics = metrics();
        assertEquals(1.0, value(metrics, "euhedral_requests_total", "api=\"chat_completions\"", "outcome=\"success\""));
        assertEquals(1.0, value(metrics, "euhedral_requests_total", "api=\"messages\"", "outcome=\"success\""));
        assertEquals(2.0, value(metrics, "euhedral_requests_total", "api=\"chat_completions\"", "outcome=\"invalid\""));
        assertEquals(1.0, value(metrics, "euhedral_requests_total", "api=\"chat_completions\"", "outcome=\"failed\""));
        assertEquals(1.0, value(metrics, "euhedral_schema_rejections_total", "api=\"chat_completions\""));
        assertEquals(1.0, value(metrics, "euhedral_constrained_failures_total", "reason=\"invalid_tool_call\""));
        assertEquals(2.0, value(metrics, "euhedral_finishes_total", "reason=\"end\""));
        assertEquals(8.0, value(metrics, "euhedral_completion_tokens_total"), "four tokens per answer");
        assertTrue(value(metrics, "euhedral_prompt_tokens_total") > 0);
        assertEquals(2.0, value(metrics, "euhedral_time_to_first_token_seconds_count", "api="));
        assertEquals(0.0, value(metrics, "euhedral_generations_active"));
        assertEquals(0.0, value(metrics, "euhedral_generations_queued"));
        assertTrue(metrics.contains("jvm_memory_used_bytes"), "the JVM's own metrics are exported too");
    }

    private int post(String path, String body) throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            return client.send(
                            HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + path))
                                    .header("Content-Type", "application/json")
                                    .POST(HttpRequest.BodyPublishers.ofString(body))
                                    .build(),
                            HttpResponse.BodyHandlers.ofString())
                    .statusCode();
        }
    }

    private String metrics() throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            var response = client.send(
                    HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/metrics"))
                            .build(),
                    HttpResponse.BodyHandlers.ofString());
            assertEquals(200, response.statusCode());
            assertTrue(
                    response.headers().firstValue("Content-Type").orElseThrow().startsWith("text/plain"));
            return response.body();
        }
    }

    /// The sum of the samples of `name` whose labels contain every fragment.
    private static double value(String metrics, String name, String... labels) {
        double sum = 0;
        boolean found = false;
        Matcher matcher = Pattern.compile("(?m)^" + Pattern.quote(name) + "(\\{[^}]*})? (\\S+)$")
                .matcher(metrics);
        outer:
        while (matcher.find()) {
            String labelText = matcher.group(1) == null ? "" : matcher.group(1);
            for (String label : labels) if (!labelText.contains(label)) continue outer;
            sum += Double.parseDouble(matcher.group(2));
            found = true;
        }
        if (!found) throw new AssertionError("no sample of " + name + " " + List.of(labels) + " in\n" + metrics);
        return sum;
    }
}
