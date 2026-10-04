package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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

/// The Responses surface over the shared pipeline, against the scripted backend.
@SpringBootTest(
        classes = ScriptedApiApplication.class,
        properties = "euhedral.test.scripted-api=true",
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Timeout(60)
class ResponsesApiTest {
    private static final String MODEL = ScriptedInferenceBackend.MODEL_ID;
    private static final JsonMapper JSON = JsonMapper.shared();
    private static final String WEATHER = "\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\","
            + "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
            + "\"required\":[\"city\"],\"additionalProperties\":false}}]";

    @LocalServerPort
    private int port;

    @Autowired
    private ScriptedInferenceBackend backend;

    @BeforeEach
    void resetBackend() {
        this.backend.reset();
    }

    @Test
    void aStringInputReturnsAMessageItem() throws Exception {
        this.backend.emptyReasoning = true;
        JsonNode response = post(200, "{\"model\":\"" + MODEL + "\",\"input\":\"Hi\",\"instructions\":\"Be brief.\"}");
        assertEquals("response", response.get("object").asString());
        assertTrue(response.get("id").asString().startsWith("resp_"));
        assertEquals("completed", response.get("status").asString());
        assertEquals(1, response.get("output").size(), "an empty reasoning produces no reasoning item");
        JsonNode message = response.at("/output/0");
        assertEquals("message", message.get("type").asString());
        assertEquals("assistant", message.get("role").asString());
        assertEquals("output_text", message.at("/content/0/type").asString());
        assertEquals("Hello, world", message.at("/content/0/text").asString());
        assertEquals(5, response.at("/usage/output_tokens").asInt());
        assertEquals(0, response.at("/usage/input_tokens_details/cached_tokens").asInt());
        assertEquals("Be brief.", response.get("instructions").asString());
        String prompt = this.backend.only().prompt;
        assertTrue(prompt.contains("Be brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>"), prompt);
        assertTrue(prompt.endsWith("<think>\n"), "the template default thinks");
    }

    @Test
    void reasoningIsAnItemWhoseOpaqueTokenReplays() throws Exception {
        this.backend.script =
                ScriptedInferenceBackend.tokens(List.of("Thinking it over.", "</think>", "\n\n", "Done."), true);
        JsonNode response = post(
                200,
                "{\"model\":\"" + MODEL + "\",\"input\":\"Hi\",\"reasoning\":{\"effort\":\"low\",\"summary\":\"auto\"},"
                        + "\"include\":[\"reasoning.encrypted_content\"],\"store\":false}");
        JsonNode reasoning = response.at("/output/0");
        assertEquals("reasoning", reasoning.get("type").asString());
        assertEquals("Thinking it over.", reasoning.at("/content/0/text").asString());
        assertEquals(0, reasoning.get("summary").size(), "no summary is generated");
        String token = reasoning.get("encrypted_content").asString();
        assertEquals("Done.", response.at("/output/1/content/0/text").asString());
        assertEquals(
                1, response.at("/usage/output_tokens_details/reasoning_tokens").asInt());
        assertTrue(this.backend.only().prompt.contains("Reasoning effort is set to low."));

        this.backend.generations.clear();
        this.backend.emptyReasoning = true;
        post(
                200,
                "{\"model\":\"" + MODEL + "\",\"input\":[{\"role\":\"user\",\"content\":\"Hi\"},"
                        + "{\"type\":\"reasoning\",\"summary\":[],\"encrypted_content\":\"" + token + "\"},"
                        + "{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\","
                        + "\"text\":\"Done.\"}]},{\"role\":\"user\",\"content\":\"Again\"}]}");
        assertTrue(
                this.backend
                        .only()
                        .prompt
                        .contains("<|im_start|>assistant\n<think>\nThinking it over.\n</think>\n\nDone.<|im_end|>"),
                this.backend.only().prompt);
    }

    @Test
    void functionCallsAndOutputsContinueTheConversation() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(
                List.of("{\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"}}]}"), true);
        JsonNode response = post(
                200,
                "{\"model\":\"" + MODEL + "\",\"reasoning\":{\"effort\":\"none\"},\"input\":\"Weather?\"," + WEATHER
                        + "}");
        JsonNode call = response.at("/output/0");
        assertEquals("function_call", call.get("type").asString());
        assertEquals("get_weather", call.get("name").asString());
        assertEquals("{\"city\":\"Oslo\"}", call.get("arguments").asString());
        assertTrue(call.get("call_id").asString().startsWith("call_"));
        String grammar = this.backend.only().grammar;
        assertTrue(grammar.contains("arguments_0: %json {\"x-guidance\""), grammar);
        assertTrue(grammar.contains("\"additionalProperties\":false"), "function tools are strict by default");

        this.backend.generations.clear();
        this.backend.script = ScriptedInferenceBackend.tokens(List.of("{\"content\":\"Cold.\"}"), true);
        JsonNode next = post(
                200,
                "{\"model\":\"" + MODEL + "\",\"reasoning\":{\"effort\":\"none\"}," + WEATHER + ",\"input\":["
                        + "{\"role\":\"user\",\"content\":\"Weather?\"}," + call + ","
                        + "{\"type\":\"function_call_output\",\"call_id\":\""
                        + call.get("call_id").asString()
                        + "\",\"output\":\"-3C\"}]}");
        assertEquals("Cold.", next.at("/output/0/content/0/text").asString());
        assertTrue(this.backend.only().prompt.contains("{\"tool_results\": [\"-3C\"]}"));
        post(
                400,
                "{\"model\":\"" + MODEL + "\"," + WEATHER + ",\"input\":[{\"role\":\"user\",\"content\":\"Weather?\"},"
                        + call + ",{\"role\":\"user\",\"content\":\"Never mind\"}]}");
    }

    @Test
    void aNonStrictToolTakesAnyObject() throws Exception {
        this.backend.emptyReasoning = true;
        post(
                200,
                "{\"model\":\"" + MODEL + "\",\"input\":\"x\",\"tools\":[{\"type\":\"function\",\"name\":\"f\","
                        + "\"strict\":false,\"parameters\":{\"type\":\"object\",\"properties\":{\"a\":{\"not\":{}}}}}]}");
        assertTrue(this.backend.only().grammar.contains("arguments_0: %json {\"x-guidance\":{\"whitespace_pattern\""));
        assertTrue(this.backend.only().grammar.contains("\"type\":\"object\"}"));
    }

    @Test
    void aJsonSchemaFormatConstrainsTheText() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(List.of("{\"n\":1}"), true);
        JsonNode response = post(
                200,
                "{\"model\":\"" + MODEL + "\",\"reasoning\":{\"effort\":\"none\"},\"input\":\"x\",\"text\":{\"format\":"
                        + "{\"type\":\"json_schema\",\"name\":\"n\",\"strict\":true,\"schema\":{\"type\":\"object\","
                        + "\"properties\":{\"n\":{\"type\":\"integer\"}},\"required\":[\"n\"]}}}}");
        assertEquals("{\"n\":1}", response.at("/output/0/content/0/text").asString());
        assertTrue(this.backend.only().grammar.contains("\"required\":[\"n\"]"));
    }

    @Test
    void anExhaustedBudgetIsIncomplete() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(0);
        JsonNode response = post(
                200,
                "{\"model\":\"" + MODEL
                        + "\",\"reasoning\":{\"effort\":\"none\"},\"input\":\"x\",\"max_output_tokens\":3}");
        assertEquals("incomplete", response.get("status").asString());
        assertEquals(
                "max_output_tokens", response.at("/incomplete_details/reason").asString());
    }

    @Test
    void aStreamFollowsTheResponsesEventSequence() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(List.of("Plan.", "</think>", "\n\n", "A", "B"), true);
        List<JsonNode> events = stream("{\"model\":\"" + MODEL + "\",\"input\":\"x\",\"stream\":true}");
        List<String> types = new ArrayList<>();
        for (JsonNode event : events) types.add(event.get("type").asString());
        assertEquals(
                List.of(
                        "response.created",
                        "response.in_progress",
                        "response.output_item.added",
                        "response.content_part.added",
                        "response.reasoning_text.delta",
                        "response.reasoning_text.done",
                        "response.content_part.done",
                        "response.output_item.done",
                        "response.output_item.added",
                        "response.content_part.added",
                        "response.output_text.delta",
                        "response.output_text.delta",
                        "response.output_text.done",
                        "response.content_part.done",
                        "response.output_item.done",
                        "response.completed"),
                types);
        for (int index = 0; index < events.size(); index++)
            assertEquals(index, events.get(index).get("sequence_number").asInt());
        assertEquals("AB", events.get(12).get("text").asString());
        assertEquals(1, events.get(8).get("output_index").asInt());
        JsonNode completed = events.getLast().get("response");
        assertEquals(2, completed.get("output").size());
        assertEquals(
                events.get(2).at("/item/id").asString(),
                completed.at("/output/0/id").asString());
    }

    @Test
    void aStreamedFunctionCallCarriesItsArguments() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(
                List.of("{\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"}}]}"), true);
        List<JsonNode> events = stream("{\"model\":\"" + MODEL
                + "\",\"reasoning\":{\"effort\":\"none\"},\"input\":\"x\",\"stream\":true," + WEATHER + "}");
        List<String> types = new ArrayList<>();
        for (JsonNode event : events) types.add(event.get("type").asString());
        assertEquals(
                List.of(
                        "response.created",
                        "response.in_progress",
                        "response.output_item.added",
                        "response.function_call_arguments.delta",
                        "response.function_call_arguments.done",
                        "response.output_item.done",
                        "response.completed"),
                types);
        assertEquals("{\"city\":\"Oslo\"}", events.get(4).get("arguments").asString());
    }

    @Test
    void statefulAndUnsupportedFeaturesAreRefused() throws Exception {
        for (String extra : List.of(
                "\"previous_response_id\":\"resp_1\"",
                "\"background\":true",
                "\"tools\":[{\"type\":\"web_search\"}]",
                "\"include\":[\"message.output_text.logprobs\"]",
                "\"text\":{\"verbosity\":\"low\"}",
                "\"truncation\":\"auto\"",
                "\"reasoning\":{\"effort\":\"minimal\"}",
                "\"bogus\":1")) {
            post(400, "{\"model\":\"" + MODEL + "\",\"input\":\"x\"," + extra + "}");
        }
        post(
                400,
                "{\"model\":\"" + MODEL + "\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\","
                        + "\"image_url\":\"http://x\"}]}]}");
        post(
                400,
                "{\"model\":\"" + MODEL + "\",\"input\":[{\"type\":\"reasoning\",\"encrypted_content\":\"gAAAA\"},"
                        + "{\"role\":\"user\",\"content\":\"x\"}]}");
        JsonNode error = post(404, "{\"model\":\"gpt-5\",\"input\":\"x\"}");
        assertEquals("model_not_found", error.at("/error/code").asString());
        assertTrue(this.backend.generations.isEmpty());
    }

    @Test
    void aConversationRendersAsTheEquivalentChatCompletionAndEncodesOnce() throws Exception {
        this.backend.emptyReasoning = true;
        int encoded = this.backend.encoded.get();
        post(
                200,
                "{\"model\":\"" + MODEL + "\",\"instructions\":\"Be brief.\",\"input\":[{\"role\":\"user\","
                        + "\"content\":\"a\"},{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":"
                        + "\"output_text\",\"text\":\"b\"}]},{\"role\":\"user\",\"content\":\"c\"}]}");
        assertEquals(encoded + 1, this.backend.encoded.get(), "a request's prompt is encoded once");
        String responses = this.backend.only().prompt;
        this.backend.generations.clear();
        try (var client = HttpClient.newHttpClient()) {
            var chat = client.send(
                    HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/chat/completions"))
                            .header("Content-Type", "application/json")
                            .POST(
                                    HttpRequest.BodyPublishers.ofString(
                                            "{\"model\":\"" + MODEL + "\",\"messages\":["
                                                    + "{\"role\":\"system\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"a\"},"
                                                    + "{\"role\":\"assistant\",\"content\":\"b\"},{\"role\":\"user\",\"content\":\"c\"}]}"))
                            .build(),
                    HttpResponse.BodyHandlers.ofString());
            assertEquals(200, chat.statusCode(), chat.body());
        }
        assertEquals(responses, this.backend.only().prompt, "the two surfaces share cached prefixes");
    }

    @Test
    void developerMessagesJoinTheInstructions() throws Exception {
        this.backend.emptyReasoning = true;
        post(
                200,
                "{\"model\":\"" + MODEL + "\",\"reasoning\":{\"effort\":\"none\"},\"instructions\":\"A.\",\"input\":["
                        + "{\"role\":\"developer\",\"content\":\"B.\"},{\"role\":\"user\",\"content\":\"x\"}]}");
        assertTrue(this.backend.only().prompt.startsWith("<|im_start|>system\nA.\n\nB.<|im_end|>"));
        assertFalse(this.backend.only().prompt.contains("<think>\n\n</think>\n\nx"));
    }

    private JsonNode post(int status, String body) throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            var response = client.send(request(body), HttpResponse.BodyHandlers.ofString());
            assertEquals(status, response.statusCode(), body + " -> " + response.body());
            return JSON.readTree(response.body());
        }
    }

    private List<JsonNode> stream(String body) throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            var response = client.send(request(body), HttpResponse.BodyHandlers.ofLines());
            assertEquals(200, response.statusCode());
            List<JsonNode> events = new ArrayList<>();
            for (String line : (Iterable<String>) response.body()::iterator)
                if (line.startsWith("data:")) events.add(JSON.readTree(line.substring("data:".length())));
            return events;
        }
    }

    private HttpRequest request(String body) {
        return HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/responses"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
    }
}
