package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
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

/// Reasoning controls and the split of reasoning from the answer, against the scripted backend.
@SpringBootTest(
        classes = ScriptedApiApplication.class,
        properties = "euhedral.test.scripted-api=true",
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Timeout(60)
class ReasoningTest {
    private static final String MODEL = ScriptedInferenceBackend.MODEL_ID;
    private static final JsonMapper JSON = JsonMapper.shared();
    private static final String HELLO = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]";
    private static final String XHIGH = "Reasoning effort is set to xhigh.";
    private static final String LOW = "Reasoning effort is set to low.";

    @LocalServerPort
    private int port;

    @Autowired
    private ScriptedInferenceBackend backend;

    @BeforeEach
    void resetBackend() {
        this.backend.reset();
    }

    @Test
    void omittedSettingsThinkAtTheTemplateDefaultAndSplitReasoningFromTheAnswer() throws Exception {
        this.backend.script =
                ScriptedInferenceBackend.tokens(List.of("\nLet me", " think.\n", "</think>", "\n\n", "Hello."), true);
        JsonNode response = json(200, "{\"model\":\"" + MODEL + "\"," + HELLO + "}");
        JsonNode message = response.at("/choices/0/message");
        assertEquals("Let me think.", message.get("reasoning_content").asString());
        assertEquals("Hello.", message.get("content").asString());
        assertEquals("stop", response.at("/choices/0/finish_reason").asString());
        assertEquals(6, response.at("/usage/completion_tokens").asInt());
        assertEquals(
                2,
                response.at("/usage/completion_tokens_details/reasoning_tokens").asInt());
        String prompt = this.backend.only().prompt;
        assertTrue(prompt.startsWith("<|im_start|>system\n" + XHIGH), prompt);
        assertTrue(prompt.endsWith("<|im_start|>assistant\n<think>\n"), prompt);
        assertTrue(this.backend.only().output.reasoning());
        assertNull(this.backend.only().constraint, "free reasoning needs no sampling constraint");
    }

    @Test
    void noneTurnsThinkingOffAndReportsNoReasoning() throws Exception {
        JsonNode response = json(200, "{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\"," + HELLO + "}");
        assertEquals("Hello, world", response.at("/choices/0/message/content").asString());
        assertFalse(response.at("/choices/0/message").has("reasoning_content"));
        assertFalse(response.get("usage").has("completion_tokens_details"));
        assertEquals(
                "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                this.backend.only().prompt);
        assertFalse(this.backend.only().output.reasoning());
    }

    @Test
    void eachEffortRendersTheTemplatesInstructions() throws Exception {
        this.backend.emptyReasoning = true;
        String system =
                "\"messages\":[{\"role\":\"system\",\"content\":\"Be terse.\"},{\"role\":\"user\",\"content\":\"Hi\"}]";
        List<String> expected = List.of(
                "<|im_start|>system\n" + LOW,
                "<|im_start|>system\nBe terse.<|im_end|>",
                "<|im_start|>system\n" + XHIGH,
                "<|im_start|>system\n" + XHIGH);
        List<String> efforts = List.of("low", "medium", "high", "xhigh");
        for (int index = 0; index < efforts.size(); index++) {
            this.backend.generations.clear();
            JsonNode response = json(
                    200,
                    "{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"" + efforts.get(index) + "\"," + system + "}");
            assertEquals(
                    "Hello, world", response.at("/choices/0/message/content").asString());
            String prompt = this.backend.only().prompt;
            assertTrue(prompt.startsWith(expected.get(index)), efforts.get(index) + ": " + prompt);
            assertTrue(prompt.contains("Be terse."));
            assertTrue(prompt.endsWith("<think>\n"));
        }
    }

    @Test
    void templateKwargsAreTheTemplatesOwnSwitches() throws Exception {
        this.backend.emptyReasoning = true;
        json(200, "{\"model\":\"" + MODEL + "\",\"chat_template_kwargs\":{\"enable_thinking\":false}," + HELLO + "}");
        assertTrue(this.backend.only().prompt.endsWith("<think>\n\n</think>\n\n"));
        this.backend.generations.clear();
        json(
                200,
                "{\"model\":\"" + MODEL + "\",\"chat_template_kwargs\":{\"reasoning_effort\":\"low\"}," + HELLO + "}");
        assertTrue(this.backend.only().prompt.startsWith("<|im_start|>system\n" + LOW));
    }

    @Test
    void unknownOrConflictingReasoningSettingsAreRejected() throws Exception {
        assertEquals(
                "reasoning_effort",
                json(400, "{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"minimal\"," + HELLO + "}")
                        .at("/error/param")
                        .asString());
        json(400, "{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"max\"," + HELLO + "}");
        json(400, "{\"model\":\"" + MODEL + "\",\"reasoning_effort\":3," + HELLO + "}");
        json(
                400,
                "{\"model\":\"" + MODEL
                        + "\",\"reasoning_effort\":\"low\",\"chat_template_kwargs\":{\"enable_thinking\":false},"
                        + HELLO + "}");
        json(
                400,
                "{\"model\":\"" + MODEL
                        + "\",\"reasoning_effort\":\"none\",\"chat_template_kwargs\":{\"enable_thinking\":true},"
                        + HELLO + "}");
        assertEquals(
                "unsupported_parameter",
                json(
                                400,
                                "{\"model\":\"" + MODEL + "\",\"chat_template_kwargs\":{\"add_vision_id\":true},"
                                        + HELLO + "}")
                        .at("/error/code")
                        .asString());
        assertTrue(this.backend.generations.isEmpty());
    }

    @Test
    void replayedReasoningRendersIntoTheConversationAndOnlyOnAssistantTurns() throws Exception {
        this.backend.emptyReasoning = true;
        json(
                200,
                "{\"model\":\"" + MODEL + "\",\"messages\":[{\"role\":\"user\",\"content\":\"a\"},"
                        + "{\"role\":\"assistant\",\"reasoning_content\":\"  why  \",\"content\":\"b\"},"
                        + "{\"role\":\"user\",\"content\":\"c\"}]}");
        assertTrue(this.backend.only().prompt.contains("<|im_start|>assistant\n<think>\nwhy\n</think>\n\nb<|im_end|>"));
        json(
                400,
                "{\"model\":\"" + MODEL
                        + "\",\"messages\":[{\"role\":\"user\",\"content\":\"a\",\"reasoning_content\":\"x\"}]}");
    }

    @Test
    void stopSequencesApplyToTheAnswerOnly() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(
                List.of("I will say STOP", " later.", "</think>\n\n", "Go ", "STOP", " now"), true);
        JsonNode response = json(200, "{\"model\":\"" + MODEL + "\",\"stop\":[\"STOP\"]," + HELLO + "}");
        assertEquals(
                "I will say STOP later.",
                response.at("/choices/0/message/reasoning_content").asString());
        assertEquals("Go ", response.at("/choices/0/message/content").asString());
        assertEquals("stop", response.at("/choices/0/finish_reason").asString());
        assertEquals(5, this.backend.only().emitted.get(), "the stop ends decoding");
    }

    @Test
    void anExhaustedBudgetDuringReasoningEndsWithLength() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(0);
        JsonNode response = json(200, "{\"model\":\"" + MODEL + "\",\"max_tokens\":3," + HELLO + "}");
        assertEquals(
                "tok0 tok1 tok2",
                response.at("/choices/0/message/reasoning_content").asString());
        assertEquals("", response.at("/choices/0/message/content").asString());
        assertEquals("length", response.at("/choices/0/finish_reason").asString());
        assertEquals(
                3,
                response.at("/usage/completion_tokens_details/reasoning_tokens").asInt());
    }

    @Test
    void reasoningThenAToolCall() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(
                List.of(
                        "Need the weather.",
                        "\n</think>\n\n",
                        "{\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]}"),
                true);
        JsonNode response = json(
                200,
                "{\"model\":\"" + MODEL + "\",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
                        + "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}],"
                        + HELLO + "}");
        JsonNode message = response.at("/choices/0/message");
        assertEquals("Need the weather.", message.get("reasoning_content").asString());
        assertTrue(message.get("content").isNull());
        assertEquals("get_weather", message.at("/tool_calls/0/function/name").asString());
        assertEquals(
                "{\"city\":\"Paris\"}",
                message.at("/tool_calls/0/function/arguments").asString());
        assertEquals("tool_calls", response.at("/choices/0/finish_reason").asString());
        var generation = this.backend.only();
        assertTrue(generation.output.reasoning());
        assertNotNull(generation.constraint, "the answer after the reasoning is constrained");
        assertTrue(generation.prompt.startsWith("<|im_start|>system\n" + XHIGH), generation.prompt);
        assertTrue(generation.prompt.contains("in the final answer.\n\n# Tools"), generation.prompt);
    }

    @Test
    void streamingDeliversReasoningInOrderBeforeTheAnswer() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(
                List.of("One", " two ", "three\n", "</think>", "\n", "\nFour", " five"), true);
        List<JsonNode> chunks = stream("{\"model\":\"" + MODEL
                + "\",\"stream\":true,\"stream_options\":{\"include_usage\":true}," + HELLO + "}");
        StringBuilder reasoning = new StringBuilder();
        StringBuilder content = new StringBuilder();
        boolean answerStarted = false;
        for (JsonNode chunk : chunks) {
            JsonNode choices = chunk.get("choices");
            if (choices.isEmpty()) continue;
            JsonNode delta = choices.get(0).get("delta");
            if (delta.has("reasoning_content")) {
                assertFalse(answerStarted, "reasoning must precede the answer");
                reasoning.append(delta.get("reasoning_content").asString());
            }
            if (delta.has("content") && !delta.get("content").asString().isEmpty()) {
                answerStarted = true;
                content.append(delta.get("content").asString());
            }
        }
        assertEquals("One two three", reasoning.toString());
        assertEquals("Four five", content.toString());
        JsonNode usage = chunks.getLast().get("usage");
        assertEquals(3, usage.at("/completion_tokens_details/reasoning_tokens").asInt());
    }

    private JsonNode json(int status, String body) throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            var response = client.send(request(body), HttpResponse.BodyHandlers.ofString());
            assertEquals(status, response.statusCode(), response.body());
            return JSON.readTree(response.body());
        }
    }

    private List<JsonNode> stream(String body) throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            var response = client.send(request(body), HttpResponse.BodyHandlers.ofLines());
            assertEquals(200, response.statusCode());
            List<JsonNode> chunks = new ArrayList<>();
            for (String line : (Iterable<String>) response.body()::iterator) {
                if (!line.startsWith("data:")) continue;
                String data = line.substring("data:".length()).strip();
                if (data.equals("[DONE]")) break;
                chunks.add(JSON.readTree(data));
            }
            return chunks;
        }
    }

    private HttpRequest request(String body) {
        return HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
    }
}
