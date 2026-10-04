package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/// `response_format` and strict tools over HTTP: what is compiled, refused, and returned. The scripted backend
/// does not sample under the grammar; the grammars themselves are tested over the real vocabulary
/// (`OutputGrammarTest`).
@SpringBootTest(
        classes = ScriptedApiApplication.class,
        properties = "euhedral.test.scripted-api=true",
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Timeout(60)
class StructuredOutputTest {
    private static final String MODEL = ScriptedInferenceBackend.MODEL_ID;
    private static final JsonMapper JSON = JsonMapper.shared();
    private static final String HELLO = "\"messages\":[{\"role\":\"user\",\"content\":\"Describe Ann as JSON.\"}]";
    private static final String PERSON = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},"
            + "\"age\":{\"type\":\"integer\",\"minimum\":0}},\"required\":[\"name\",\"age\"],\"additionalProperties\":false}";

    @LocalServerPort
    private int port;

    @Autowired
    private ScriptedInferenceBackend backend;

    @BeforeEach
    void resetBackend() {
        this.backend.reset();
    }

    @Test
    void aJsonSchemaFormatConstrainsTheAnswerAndReturnsTheDocument() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(List.of("{\"name\":", "\"Ann\",\"age\":", "42}"), true);
        JsonNode response = json(
                200,
                request(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
                        + "{\"name\":\"person\",\"strict\":true,\"schema\":" + PERSON + "}}"));
        assertEquals(
                "{\"name\":\"Ann\",\"age\":42}",
                response.at("/choices/0/message/content").asString());
        assertEquals("stop", response.at("/choices/0/finish_reason").asString());
        String grammar = this.backend.only().grammar;
        assertTrue(grammar.startsWith("start: answer\nanswer: %json {\"x-guidance\":"), grammar);
        assertTrue(grammar.contains("\"required\":[\"name\",\"age\"]"), grammar);
    }

    @Test
    void jsonObjectModeAcceptsAnyObject() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(List.of("{\"a\":[1,2]}"), true);
        JsonNode response = json(200, request(",\"response_format\":{\"type\":\"json_object\"}"));
        assertEquals("{\"a\":[1,2]}", response.at("/choices/0/message/content").asString());
        assertTrue(this.backend.only().grammar.contains("%json {\"x-guidance\":"));
        assertTrue(this.backend.only().grammar.contains("\"type\":\"object\""));
    }

    @Test
    void aTextFormatIsFreeText() throws Exception {
        json(200, request(",\"response_format\":{\"type\":\"text\"}"));
        assertNull(this.backend.only().grammar);
    }

    @Test
    void schemasThatCannotBeEnforcedAreRefusedWithTheReason() throws Exception {
        JsonNode refused = json(
                400,
                request(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
                        + "{\"name\":\"p\",\"schema\":{\"type\":\"object\",\"if\":{}}}}"));
        assertEquals(
                "response_format.json_schema.schema", refused.at("/error/param").asString());
        assertTrue(refused.at("/error/message").asString().contains("if"), refused.toString());
        JsonNode unsatisfiable = json(
                400,
                request(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
                        + "{\"name\":\"p\",\"schema\":{\"type\":\"integer\",\"minimum\":3,\"maximum\":2}}}"));
        assertEquals(
                "response_format.json_schema.schema",
                unsatisfiable.at("/error/param").asString());
        assertTrue(this.backend.generations.isEmpty(), "a refused request opens no generation");
    }

    @Test
    void malformedFormatsAreRefused() throws Exception {
        assertEquals(
                "response_format.type=image",
                json(400, request(",\"response_format\":{\"type\":\"image\"}"))
                        .at("/error/param")
                        .asString());
        json(400, request(",\"response_format\":{\"type\":\"json_schema\"}"));
        json(400, request(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"schema\":{}}}"));
        json(400, request(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"a b\"}}"));
        json(400, request(",\"response_format\":{\"type\":\"json_object\",\"extra\":1}"));
        assertEquals(
                "response_format.json_schema.schema.x-guidance",
                json(
                                400,
                                request(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
                                        + "{\"name\":\"p\",\"schema\":{\"x-guidance\":{}}}}"))
                        .at("/error/param")
                        .asString());
    }

    @Test
    void stopSequencesCannotCutAJsonAnswer() throws Exception {
        JsonNode refused = json(400, request(",\"stop\":[\"}\"],\"response_format\":{\"type\":\"json_object\"}"));
        assertEquals("stop", refused.at("/error/param").asString());
    }

    @Test
    void aFinishedAnswerThatIsNotJsonIsNeverReturnedAsSuccess() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(List.of("{\"name\":"), true);
        JsonNode failed = json(500, request(",\"response_format\":{\"type\":\"json_object\"}"));
        assertEquals("invalid_structured_output", failed.at("/error/code").asString());
    }

    @Test
    void aTruncatedAnswerEndsWithLength() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(List.of("{\"name\":", "\"An"), true);
        JsonNode response = json(200, request(",\"max_tokens\":2,\"response_format\":{\"type\":\"json_object\"}"));
        assertEquals("length", response.at("/choices/0/finish_reason").asString());
        assertEquals("{\"name\":\"An", response.at("/choices/0/message/content").asString());
    }

    @Test
    void besideToolsADirectAnswerIsTheDocument() throws Exception {
        this.backend.script =
                ScriptedInferenceBackend.tokens(List.of("{\"content\":{\"name\":\"Ann\",\"age\":42}}"), true);
        JsonNode response = json(
                200,
                request(",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}],"
                        + "\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"p\",\"schema\":"
                        + PERSON
                        + "}}"));
        assertEquals(
                "{\"name\":\"Ann\",\"age\":42}",
                response.at("/choices/0/message/content").asString());
        String prompt = this.backend.only().prompt;
        assertTrue(prompt.contains("{\"content\":RESPONSE}"), prompt);
        assertTrue(this.backend.only().grammar.contains("content_value: %json {\"x-guidance\":"));
    }

    @Test
    void strictToolsCompileTheirSchemas() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(
                List.of("{\"tool_calls\":[{\"name\":\"save\",\"arguments\":{\"name\":\"Ann\",\"age\":42}}]}"), true);
        JsonNode response = json(
                200,
                request(",\"tool_choice\":\"required\",\"tools\":[{\"type\":\"function\","
                        + "\"function\":{\"name\":\"save\",\"strict\":true,\"parameters\":" + PERSON + "}}]"));
        assertEquals(
                "save",
                response.at("/choices/0/message/tool_calls/0/function/name").asString());
        String grammar = this.backend.only().grammar;
        assertTrue(grammar.contains("arguments_0: %json {\"x-guidance\":"), grammar);
        assertTrue(grammar.contains("\"minimum\":0"), "a strict tool's arguments follow its schema");
    }

    private String request(String extra) {
        return "{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\"," + HELLO + extra + "}";
    }

    private JsonNode json(int status, String body) throws Exception {
        try (var client = HttpClient.newHttpClient()) {
            var response = client.send(
                    HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/chat/completions"))
                            .header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(body))
                            .build(),
                    HttpResponse.BodyHandlers.ofString());
            assertEquals(status, response.statusCode(), response.body());
            return JSON.readTree(response.body());
        }
    }
}
