package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/// The Anthropic Messages surface over the shared pipeline, against the scripted backend: request mapping, the
/// rendered prompt, response blocks, stream events, errors.
@SpringBootTest(
        classes = ScriptedApiApplication.class,
        properties = "euhedral.test.scripted-api=true",
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Timeout(60)
class AnthropicMessagesTest {
    private static final String MODEL = ScriptedInferenceBackend.MODEL_ID;
    private static final JsonMapper JSON = JsonMapper.shared();
    private static final String HI = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]";
    private static final String WEATHER = "\"tools\":[{\"name\":\"get_weather\",\"description\":\"Weather.\","
            + "\"input_schema\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
            + "\"required\":[\"city\"]}}]";

    @LocalServerPort
    private int port;

    @Autowired
    private ScriptedInferenceBackend backend;

    @BeforeEach
    void resetBackend() {
        this.backend.reset();
    }

    @Test
    void aMessageWithASystemPromptRendersLikeTheEquivalentChatCompletion() throws Exception {
        int encoded = this.backend.encoded.get();
        JsonNode message = post(
                200,
                "/v1/messages",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":64,\"system\":[{\"type\":\"text\",\"text\":\"Be brief.\","
                        + "\"cache_control\":{\"type\":\"ephemeral\"}}]," + HI + "}");
        assertEquals("message", message.get("type").asString());
        assertEquals("assistant", message.get("role").asString());
        assertTrue(message.get("id").asString().startsWith("msg_"));
        assertEquals(MODEL, message.get("model").asString());
        assertEquals("Hello, world", message.at("/content/0/text").asString());
        assertEquals(encoded + 1, this.backend.encoded.get(), "a request's prompt is encoded once");
        assertEquals(1, message.get("content").size());
        assertEquals("end_turn", message.get("stop_reason").asString());
        assertTrue(message.get("stop_sequence").isNull());
        assertEquals(4, message.at("/usage/output_tokens").asInt());
        assertEquals(0, message.at("/usage/cache_read_input_tokens").asInt());
        String anthropicPrompt = this.backend.only().prompt;
        assertEquals(
                "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
                        + "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                anthropicPrompt,
                "without thinking the model answers directly");
        // The same conversation through Chat Completions renders the same prompt, so they share cached prefixes.
        this.backend.generations.clear();
        post(
                200,
                "/v1/chat/completions",
                "{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"messages\":[{\"role\":\"system\","
                        + "\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"Hi\"}]}");
        assertEquals(anthropicPrompt, this.backend.only().prompt);
        assertEquals(
                this.backend.only().prompt.length(),
                message.at("/usage/input_tokens").asInt());
    }

    @Test
    void thinkingBecomesAThinkingBlockWithItsBudgetEnforced() throws Exception {
        this.backend.script =
                ScriptedInferenceBackend.tokens(List.of("Let me think.", "</think>", "\n\n", "Hello."), true);
        JsonNode message = post(
                200,
                "/v1/messages",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":2048,\"thinking\":{\"type\":\"enabled\","
                        + "\"budget_tokens\":1024}," + HI + "}");
        assertEquals("thinking", message.at("/content/0/type").asString());
        assertEquals("Let me think.", message.at("/content/0/thinking").asString());
        assertTrue(message.at("/content/0/signature").asString().startsWith("euhedral-"));
        assertEquals("text", message.at("/content/1/type").asString());
        assertEquals("Hello.", message.at("/content/1/text").asString());
        var generation = this.backend.only();
        assertTrue(generation.output.reasoning());
        assertEquals(1024, generation.output.reasoningBudget(), "budget_tokens caps the reasoning");
        assertTrue(generation.prompt.endsWith("<|im_start|>assistant\n<think>\n"));
        assertTrue(generation.prompt.contains("Reasoning effort is set to xhigh."));
    }

    @Test
    void adaptiveThinkingWithAnEffortUsesTheTemplatesLevel() throws Exception {
        this.backend.emptyReasoning = true;
        post(
                200,
                "/v1/messages",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":64,\"thinking\":{\"type\":\"adaptive\"},"
                        + "\"output_config\":{\"effort\":\"low\"}," + HI + "}");
        var generation = this.backend.only();
        assertEquals(Integer.MAX_VALUE, generation.output.reasoningBudget());
        assertTrue(generation.prompt.contains("Reasoning effort is set to low."));
    }

    @Test
    void aToolUseAndItsResultContinueTheConversation() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(
                List.of("{\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"}}]}"), true);
        JsonNode message =
                post(200, "/v1/messages", "{\"model\":\"" + MODEL + "\",\"max_tokens\":64," + WEATHER + "," + HI + "}");
        assertEquals("tool_use", message.get("stop_reason").asString());
        JsonNode use = message.at("/content/0");
        assertEquals("tool_use", use.get("type").asString());
        assertTrue(use.get("id").asString().startsWith("toolu_"));
        assertEquals("get_weather", use.get("name").asString());
        assertEquals("Oslo", use.at("/input/city").asString());
        assertTrue(this.backend.only().grammar != null);

        this.backend.generations.clear();
        this.backend.script = ScriptedInferenceBackend.tokens(List.of("{\"content\":\"It is cold.\"}"), true);
        String id = use.get("id").asString();
        JsonNode next = post(
                200,
                "/v1/messages",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":64," + WEATHER + ",\"messages\":["
                        + "{\"role\":\"user\",\"content\":\"Weather in Oslo?\"},"
                        + "{\"role\":\"assistant\",\"content\":[" + use + "]},"
                        + "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"" + id
                        + "\",\"content\":[{\"type\":\"text\",\"text\":\"-3C\"}]}]}]}");
        assertEquals("It is cold.", next.at("/content/0/text").asString());
        assertEquals("end_turn", next.get("stop_reason").asString());
        String prompt = this.backend.only().prompt;
        assertTrue(
                prompt.contains("{\"tool_calls\": [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}]}"),
                prompt);
        assertTrue(prompt.contains("{\"tool_results\": [\"-3C\"]}"), prompt);
    }

    @Test
    void parallelToolUseCanBeDisabledAndAnyRequiresACall() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(
                List.of("{\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"A\"}}]}"), true);
        post(
                200,
                "/v1/messages",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":64," + WEATHER
                        + ",\"tool_choice\":{\"type\":\"any\",\"disable_parallel_tool_use\":true}," + HI + "}");
        String grammar = this.backend.only().grammar;
        assertTrue(grammar.contains("body: calls\n"), "any requires a call");
        assertFalse(grammar.contains("(\",\" call)*"), "parallel tool use is disabled");
    }

    @Test
    void stopSequencesAreReported() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(List.of("one ", "two ", "END", " three"), true);
        JsonNode message = post(
                200,
                "/v1/messages",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":64,\"stop_sequences\":[\"END\",\"STOP\"]," + HI + "}");
        assertEquals("stop_sequence", message.get("stop_reason").asString());
        assertEquals("END", message.get("stop_sequence").asString());
        assertEquals("one two ", message.at("/content/0/text").asString());
    }

    @Test
    void anExhaustedBudgetEndsWithMaxTokens() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(0);
        JsonNode message = post(200, "/v1/messages", "{\"model\":\"" + MODEL + "\",\"max_tokens\":3," + HI + "}");
        assertEquals("max_tokens", message.get("stop_reason").asString());
        assertEquals(3, message.at("/usage/output_tokens").asInt());
    }

    @Test
    void aStreamFollowsTheMessagesEventSequence() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(
                List.of(
                        "Plan.",
                        "</think>",
                        "\n\n",
                        "{\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":" + "{\"city\":\"Oslo\"}}]}"),
                true);
        List<String[]> events = stream(
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":2048,\"stream\":true,\"thinking\":{\"type\":\"enabled\","
                        + "\"budget_tokens\":1000}," + WEATHER + "," + HI + "}");
        List<String> names = new ArrayList<>();
        for (String[] event : events) names.add(event[0]);
        assertEquals(
                List.of(
                        "message_start",
                        "content_block_start",
                        "content_block_delta",
                        "content_block_delta",
                        "content_block_stop",
                        "content_block_start",
                        "content_block_delta",
                        "content_block_stop",
                        "message_delta",
                        "message_stop"),
                names);
        JsonNode start = JSON.readTree(events.get(0)[1]);
        assertEquals("message_start", start.get("type").asString());
        assertTrue(start.at("/message/content").isEmpty());
        assertTrue(start.at("/message/usage/input_tokens").asInt() > 0);
        assertEquals(
                "thinking",
                JSON.readTree(events.get(1)[1]).at("/content_block/type").asString());
        assertEquals(
                "Plan.", JSON.readTree(events.get(2)[1]).at("/delta/thinking").asString());
        assertEquals(
                "signature_delta",
                JSON.readTree(events.get(3)[1]).at("/delta/type").asString());
        JsonNode toolStart = JSON.readTree(events.get(5)[1]);
        assertEquals(1, toolStart.get("index").asInt());
        assertEquals("tool_use", toolStart.at("/content_block/type").asString());
        assertEquals("get_weather", toolStart.at("/content_block/name").asString());
        JsonNode arguments = JSON.readTree(events.get(6)[1]);
        assertEquals("input_json_delta", arguments.at("/delta/type").asString());
        assertEquals("{\"city\":\"Oslo\"}", arguments.at("/delta/partial_json").asString());
        JsonNode delta = JSON.readTree(events.get(8)[1]);
        assertEquals("tool_use", delta.at("/delta/stop_reason").asString());
        assertEquals(5, delta.at("/usage/output_tokens").asInt(), "four chunks and the terminator");
    }

    @Test
    void textStreamsAsTextDeltas() throws Exception {
        List<String[]> events = stream("{\"model\":\"" + MODEL + "\",\"max_tokens\":64,\"stream\":true," + HI + "}");
        StringBuilder text = new StringBuilder();
        for (String[] event : events)
            if (event[0].equals("content_block_delta"))
                text.append(JSON.readTree(event[1]).at("/delta/text").asString());
        assertEquals("Hello, world", text.toString());
        assertEquals("message_stop", events.getLast()[0]);
    }

    @Test
    void aClientThatDisconnectsCancelsTheGeneration() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(20);
        try (var client = SseTestClient.post(
                this.port,
                "/v1/messages",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":3000,\"stream\":true," + HI + "}")) {
            client.readUntil(line -> line.contains("tok1"));
        }
        var generation = this.backend.only();
        assertTrue(generation.awaitClosed(), "the session closes after the client left");
        assertTrue(generation.isCancelled());
        int emitted = generation.emitted.get();
        Thread.sleep(200);
        assertEquals(emitted, generation.emitted.get(), "no quantum runs after the session closed");
    }

    @Test
    void countTokensRendersWithoutGenerating() throws Exception {
        JsonNode count = post(200, "/v1/messages/count_tokens", "{\"model\":\"" + MODEL + "\"," + HI + "}");
        assertEquals(
                "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".length(),
                count.get("input_tokens").asInt());
        assertTrue(this.backend.generations.isEmpty());
    }

    @Test
    void errorsUseAnthropicsErrorObject() throws Exception {
        JsonNode missing = post(400, "/v1/messages", "{\"model\":\"" + MODEL + "\"," + HI + "}");
        assertEquals("error", missing.get("type").asString());
        assertEquals("invalid_request_error", missing.at("/error/type").asString());
        assertTrue(missing.at("/error/message").asString().contains("max_tokens"));
        assertEquals(
                "not_found_error",
                post(404, "/v1/messages", "{\"model\":\"claude\",\"max_tokens\":5," + HI + "}")
                        .at("/error/type")
                        .asString());
        for (String body : List.of(
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":5,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":"
                        + "\"image\",\"source\":{}}]}]}",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":5,\"tools\":[{\"type\":\"web_search_20250305\","
                        + "\"name\":\"web_search\"}]," + HI + "}",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":5,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"},"
                        + "{\"role\":\"assistant\",\"content\":\"Prefill\"}]}",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":5,\"mcp_servers\":[]," + HI + "}",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":5,\"temperature\":1.5," + HI + "}",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":100,\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":"
                        + "200}," + HI + "}",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":5,\"output_config\":{\"effort\":\"low\"}," + HI + "}",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":5,\"bogus\":1," + HI + "}",
                "{\"model\":\"" + MODEL + "\",\"max_tokens\":5,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":"
                        + "\"tool_result\",\"tool_use_id\":\"x\",\"content\":\"r\"}]}]}"))
            assertEquals(
                    "invalid_request_error",
                    post(400, "/v1/messages", body).at("/error/type").asString(),
                    body);
        assertTrue(this.backend.generations.isEmpty());
    }

    @Test
    void structuredOutputConstrainsTheTextBlock() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(List.of("{\"answer\":4}"), true);
        JsonNode message = post(
                200,
                "/v1/messages",
                "{\"model\":\"" + MODEL
                        + "\",\"max_tokens\":64,\"output_config\":{\"format\":{\"type\":\"json_schema\","
                        + "\"schema\":{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"integer\"}},"
                        + "\"required\":[\"answer\"]}}}," + HI + "}");
        assertEquals("{\"answer\":4}", message.at("/content/0/text").asString());
        assertTrue(this.backend.only().grammar.contains("\"answer\""));
        JsonNode refused = post(
                400,
                "/v1/messages",
                "{\"model\":\"" + MODEL
                        + "\",\"max_tokens\":64,\"output_config\":{\"format\":{\"type\":\"json_schema\","
                        + "\"schema\":{\"not\":{}}}}," + HI + "}");
        assertTrue(refused.at("/error/message").asString().contains("not"));
    }

    @Test
    void modelsListInAnthropicsShapeForAnthropicClients() throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            var response = client.send(
                    HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/models"))
                            .header("anthropic-version", "2023-06-01")
                            .build(),
                    HttpResponse.BodyHandlers.ofString());
            JsonNode list = JSON.readTree(response.body());
            assertEquals("model", list.at("/data/0/type").asString());
            assertEquals(MODEL, list.at("/data/0/id").asString());
            assertFalse(list.get("has_more").asBoolean());
            assertNull(list.get("object"));
        }
    }

    private JsonNode post(int status, String path, String body) throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            var response = client.send(request(path, body), HttpResponse.BodyHandlers.ofString());
            assertEquals(status, response.statusCode(), response.body());
            return JSON.readTree(response.body());
        }
    }

    /// Each event's name and data.
    private List<String[]> stream(String body) throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            var response = client.send(request("/v1/messages", body), HttpResponse.BodyHandlers.ofLines());
            assertEquals(200, response.statusCode());
            List<String[]> events = new ArrayList<>();
            String name = null;
            for (String line : (Iterable<String>) response.body()::iterator) {
                if (line.startsWith("event:"))
                    name = line.substring("event:".length()).strip();
                else if (line.startsWith("data:")) events.add(new String[] {name, line.substring("data:".length())});
            }
            return events;
        }
    }

    private HttpRequest request(String path, String body) {
        return HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + path))
                .header("Content-Type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .header("x-api-key", "unused")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
    }
}
