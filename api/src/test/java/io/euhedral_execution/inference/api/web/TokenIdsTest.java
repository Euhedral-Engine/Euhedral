package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.asyncDispatch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/// `return_token_ids` returns the prompt's and the completion's exact token IDs, only without streaming.
// Contexts with equal configuration are cached and shared, backend included: one such class at a time.
@ResourceLock("scripted-api-context")
@SpringBootTest(classes = ScriptedApiApplication.class, properties = "euhedral.test.scripted-api=true")
@AutoConfigureMockMvc
class TokenIdsTest {
    private static final String MODEL = ScriptedInferenceBackend.MODEL_ID;
    private static final JsonMapper JSON = JsonMapper.shared();

    @Autowired
    private MockMvc mvc;

    @Autowired
    private ScriptedInferenceBackend backend;

    @BeforeEach
    void resetBackend() {
        this.backend.reset();
    }

    @Test
    void theFlagReturnsThePromptAndCompletionIdsTerminatorIncluded() throws Exception {
        JsonNode body = complete("\"return_token_ids\":true");
        String prompt = this.backend.only().prompt;
        int[] encoded =
                this.backend.encodePrompt(prompt).get(10, TimeUnit.SECONDS).tokenIds();
        assertTrue(encoded.length > 0);
        assertEquals(Arrays.stream(encoded).boxed().toList(), ids(body.get("prompt_token_ids")));
        // "Hello", ", ", "world", then the stop token.
        assertEquals(
                List.of(
                        ScriptedInferenceBackend.COMPLETION_ID_BASE,
                        ScriptedInferenceBackend.COMPLETION_ID_BASE + 1,
                        ScriptedInferenceBackend.COMPLETION_ID_BASE + 2,
                        ScriptedInferenceBackend.COMPLETION_ID_BASE + 3),
                ids(body.get("choices").get(0).get("token_ids")));
    }

    @Test
    void withoutTheFlagBothKeysAreAbsent() throws Exception {
        JsonNode body = complete("\"return_token_ids\":false");
        assertFalse(body.has("prompt_token_ids"));
        assertFalse(body.get("choices").get(0).has("token_ids"));
        JsonNode omitted = complete("\"temperature\":0");
        assertFalse(omitted.has("prompt_token_ids"));
        assertFalse(omitted.get("choices").get(0).has("token_ids"));
    }

    @Test
    void streamingWithTheFlagIsRejectedNamingTheParameter() throws Exception {
        MvcResult started = this.mvc
                .perform(post("/v1/chat/completions")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(request("\"stream\":true,\"return_token_ids\":true")))
                .andReturn();
        assertTrue(started.getRequest().isAsyncStarted());
        this.mvc
                .perform(asyncDispatch(started))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error.param").value("return_token_ids"));
    }

    private JsonNode complete(String extra) throws Exception {
        MvcResult started = this.mvc
                .perform(post("/v1/chat/completions")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(request(extra)))
                .andReturn();
        assertTrue(started.getRequest().isAsyncStarted());
        MvcResult done = this.mvc
                .perform(asyncDispatch(started))
                .andExpect(status().isOk())
                .andReturn();
        return JSON.readTree(done.getResponse().getContentAsString());
    }

    private static String request(String extra) {
        return "{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\"," + extra
                + ",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}";
    }

    private static List<Integer> ids(JsonNode array) {
        return array.valueStream().map(JsonNode::intValue).toList();
    }
}
