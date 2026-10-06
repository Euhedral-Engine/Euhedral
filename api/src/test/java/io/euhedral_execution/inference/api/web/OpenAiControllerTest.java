package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.asyncDispatch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import io.euhedral_execution.inference.api.engine.InferenceBackend;
import jakarta.servlet.AsyncEvent;
import jakarta.servlet.AsyncListener;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import org.hamcrest.Matchers;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockAsyncContext;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.ResultActions;

/// Controller, mapping, and error-envelope behavior through the real Spring MVC stack, without CUDA.
@SpringBootTest(classes = ScriptedApiApplication.class, properties = "euhedral.test.scripted-api=true")
@AutoConfigureMockMvc
class OpenAiControllerTest {
    private static final String MODEL = ScriptedInferenceBackend.MODEL_ID;
    private static final String HELLO = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]";

    @Autowired
    private MockMvc mvc;

    @Autowired
    private ScriptedInferenceBackend backend;

    @BeforeEach
    void resetBackend() {
        this.backend.reset();
    }

    @Test
    void modelsListsOnlyThePublicModelId() throws Exception {
        this.mvc
                .perform(get("/v1/models"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.object").value("list"))
                .andExpect(jsonPath("$.data.length()").value(1))
                .andExpect(jsonPath("$.data[0].id").value(MODEL))
                .andExpect(jsonPath("$.data[0].object").value("model"))
                .andExpect(jsonPath("$.data[0].owned_by").value("euhedral"));
        this.mvc.perform(get("/v1/models/" + MODEL)).andExpect(status().isOk());
        this.mvc
                .perform(get("/v1/models/other"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error.code").value("model_not_found"));
    }

    @Test
    void nonStreamingCompletionReturnsOneAssistantChoiceWithUsage() throws Exception {
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\"," + HELLO + "}")
                .andExpect(status().isOk())
                .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.id").value(Matchers.startsWith("chatcmpl-")))
                .andExpect(jsonPath("$.object").value("chat.completion"))
                .andExpect(jsonPath("$.created").isNumber())
                .andExpect(jsonPath("$.model").value(MODEL))
                .andExpect(jsonPath("$.choices.length()").value(1))
                .andExpect(jsonPath("$.choices[0].index").value(0))
                .andExpect(jsonPath("$.choices[0].message.role").value("assistant"))
                .andExpect(jsonPath("$.choices[0].message.content").value("Hello, world"))
                .andExpect(jsonPath("$.choices[0].finish_reason").value("stop"))
                .andExpect(jsonPath("$.usage.completion_tokens").value(4))
                .andExpect(jsonPath("$.usage.total_tokens").value(Matchers.greaterThan(4)));

        var generation = this.backend.only();
        String expectedPrompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        assertEquals(expectedPrompt, generation.prompt);
        // The response's last write may reach the client before the completion closes the session.
        assertTrue(generation.awaitClosed());
        assertEquals(1, generation.closeCount.get(), "request session must be closed exactly once");
        assertTrue(generation.generatingThread.startsWith("scripted-worker-"), generation.generatingThread);
    }

    @Test
    void usageCountsPromptTokensWithTheBackendTokenizer() throws Exception {
        int encodedBefore = this.backend.encoded.get();
        var result = completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\"," + HELLO + "}")
                .andExpect(status().isOk())
                .andReturn();
        int promptTokens = this.backend.countPromptTokens(this.backend.only().prompt);
        String body = result.getResponse().getContentAsString();
        assertTrue(body.contains("\"prompt_tokens\":" + promptTokens), body);
        assertTrue(body.contains("\"total_tokens\":" + (promptTokens + 4)), body);
        assertEquals(
                1,
                this.backend.encoded.get() - encodedBefore,
                "the request encoded its prompt once, for usage and generation");
    }

    @Test
    void exhaustedBudgetFinishesWithLength() throws Exception {
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"max_tokens\":2," + HELLO + "}")
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.choices[0].message.content").value("Hello, "))
                .andExpect(jsonPath("$.choices[0].finish_reason").value("length"))
                .andExpect(jsonPath("$.usage.completion_tokens").value(2));
        assertEquals(2, this.backend.only().maxNewTokens);
    }

    @Test
    void stopSequenceTruncatesTextAndCancelsFurtherDecoding() throws Exception {
        this.backend.script = ScriptedInferenceBackend.tokens(List.of("one ", "tw", "o STO", "P three", " four"), true);
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"stop\":[\"STOP\"]," + HELLO + "}")
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.choices[0].message.content").value("one two "))
                .andExpect(jsonPath("$.choices[0].finish_reason").value("stop"));
        var generation = this.backend.only();
        assertTrue(generation.isCancelled(), "a matched stop sequence must cancel the session");
        assertEquals(4, generation.emitted.get(), "no quantum may run after the stop sequence completed");
        assertTrue(generation.awaitClosed());
        assertEquals(1, generation.closeCount.get());
    }

    @Test
    void samplingMapsOntoGenerationConfigWithCheckpointDefaults() throws Exception {
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"seed\":42," + HELLO + "}")
                .andExpect(status().isOk());
        var defaults = this.backend.only().config;
        assertEquals(1.0f, defaults.temperature());
        assertEquals(20, defaults.topK());
        assertEquals(0.95f, defaults.topP());
        assertEquals(42L, defaults.seed());
        assertFalse(defaults.greedy());

        this.backend.reset();
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"temperature\":0.7,\"top_p\":0.5,"
                        + HELLO + "}")
                .andExpect(status().isOk());
        var explicit = this.backend.only().config;
        assertEquals(0.7f, explicit.temperature());
        assertEquals(0.5f, explicit.topP());

        this.backend.reset();
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"temperature\":0," + HELLO + "}")
                .andExpect(status().isOk());
        assertTrue(this.backend.only().config.greedy(), "temperature 0 selects argmax");

        this.backend.reset();
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"top_k\":5," + HELLO + "}")
                .andExpect(status().isOk());
        assertEquals(5, this.backend.only().config.topK(), "top_k reaches the sampler");
        this.backend.reset();
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"top_k\":-1," + HELLO + "}")
                .andExpect(status().isOk());
        assertEquals(0, this.backend.only().config.topK(), "-1 turns top-k off");
        rejected("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"top_k\":-2," + HELLO + "}", 400)
                .andExpect(jsonPath("$.error.param").value("top_k"));
    }

    @Test
    void neutralSamplingExtrasAndCacheMetadataAreAccepted() throws Exception {
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"top_logprobs\":0,\"min_p\":0,"
                        + "\"repetition_penalty\":1.0,\"prompt_cache_retention\":\"24h\"," + HELLO + "}")
                .andExpect(status().isOk());
        for (String field : List.of("\"min_p\":0.05", "\"repetition_penalty\":1.1"))
            rejected("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\"," + field + "," + HELLO + "}", 400)
                    .andExpect(jsonPath("$.error.code").value("unsupported_parameter"));
    }

    @Test
    void maxCompletionTokensAndMaxTokensMustAgree() throws Exception {
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"max_completion_tokens\":3," + HELLO
                        + "}")
                .andExpect(status().isOk());
        assertEquals(3, this.backend.only().maxNewTokens);
        rejected(
                        "{\"model\":\"" + MODEL
                                + "\",\"reasoning_effort\":\"none\",\"max_tokens\":2,\"max_completion_tokens\":3,"
                                + HELLO + "}",
                        400)
                .andExpect(jsonPath("$.error.param").value("max_tokens"));
        rejected("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"max_tokens\":0," + HELLO + "}", 400);
    }

    @Test
    void contextLimitIsEnforcedBeforeGeneration() throws Exception {
        this.backend.contextLength = 80;
        rejected("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"max_tokens\":50," + HELLO + "}", 400)
                .andExpect(jsonPath("$.error.code").value("context_length_exceeded"));
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\"," + HELLO + "}")
                .andExpect(status().isOk());
        assertEquals(80 - this.backend.countPromptTokens(this.backend.only().prompt), this.backend.only().maxNewTokens);
    }

    @Test
    void unknownModelIsNotFound() throws Exception {
        rejected("{\"model\":\"gpt-4o\"," + HELLO + "}", 404)
                .andExpect(jsonPath("$.error.type").value("invalid_request_error"))
                .andExpect(jsonPath("$.error.code").value("model_not_found"))
                .andExpect(jsonPath("$.error.param").value("model"));
        rejected("{" + HELLO + "}", 400).andExpect(jsonPath("$.error.param").value("model"));
        assertTrue(this.backend.generations.isEmpty());
    }

    @Test
    void unsupportedBehaviorIsRejectedWithTheParameterName() throws Exception {
        for (String field : List.of(
                "\"functions\":[{\"name\":\"f\"}]",
                "\"n\":2",
                "\"logprobs\":true",
                "\"top_logprobs\":2",
                "\"frequency_penalty\":0.5",
                "\"presence_penalty\":1",
                "\"repeat_penalty\":1.1",
                "\"logit_bias\":{\"1\":5}",
                "\"audio\":{\"voice\":\"alloy\"}")) {
            String name = field.substring(1, field.indexOf('"', 1));
            rejected("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\"," + field + "," + HELLO + "}", 400)
                    .andExpect(jsonPath("$.error.code").value("unsupported_parameter"))
                    .andExpect(jsonPath("$.error.param").value(name));
        }
        rejected("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"bogus\":1," + HELLO + "}", 400)
                .andExpect(jsonPath("$.error.code").value("unrecognized_argument"))
                .andExpect(jsonPath("$.error.param").value("bogus"));
        rejected(
                        "{\"model\":\"" + MODEL
                                + "\",\"reasoning_effort\":\"none\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":"
                                + "\"image_url\",\"image_url\":{\"url\":\"http://x\"}}]}]}",
                        400)
                .andExpect(jsonPath("$.error.code").value("unsupported_parameter"));
        rejected(
                        "{\"model\":\"" + MODEL
                                + "\",\"reasoning_effort\":\"none\",\"messages\":[{\"role\":\"tool\",\"content\":\"x\"}]}",
                        400)
                .andExpect(jsonPath("$.error.param").value("messages[0].role"));
        assertTrue(this.backend.generations.isEmpty(), "rejected requests must not open sessions");
    }

    @Test
    void neutralRepeatPenaltyFromPiIsAccepted() throws Exception {
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"repeat_penalty\":1.0," + HELLO + "}")
                .andExpect(status().isOk());
        assertEquals(1, this.backend.generations.size());
        assertEquals(20, this.backend.only().config.topK(), "neutral penalty must not change sampling");
        assertEquals(0.95f, this.backend.only().config.topP());
    }

    @Test
    void nullRepeatPenaltyIsEquivalentToOmission() throws Exception {
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"repeat_penalty\":null," + HELLO + "}")
                .andExpect(status().isOk());
        assertEquals(1, this.backend.generations.size());
    }

    @Test
    void neutralValuesAndMetadataAreAccepted() throws Exception {
        completion("{\"model\":\"" + MODEL
                        + "\",\"reasoning_effort\":\"none\",\"n\":1,\"logprobs\":false,\"frequency_penalty\":0,"
                        + "\"presence_penalty\":0.0,\"tools\":[],\"response_format\":{\"type\":\"text\"},"
                        + "\"user\":\"u-1\",\"metadata\":{\"k\":\"v\"},\"store\":false,"
                        + "\"messages\":[{\"role\":\"developer\",\"content\":\"Be brief.\"},"
                        + "{\"role\":\"user\",\"name\":\"ann\",\"content\":[{\"type\":\"text\",\"text\":\"Hi\"}]}]}")
                .andExpect(status().isOk());
        assertTrue(this.backend.only().prompt.startsWith("<|im_start|>system\nBe brief.<|im_end|>\n"));
    }

    @Test
    void malformedRequestsGetOpenAiErrors() throws Exception {
        rejected("{\"model\":", 400).andExpect(jsonPath("$.error.type").value("invalid_request_error"));
        rejected(
                "{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"temperature\":\"hot\"," + HELLO + "}",
                400);
        rejected("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"temperature\":3," + HELLO + "}", 400)
                .andExpect(jsonPath("$.error.param").value("temperature"));
        rejected("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"messages\":[]}", 400)
                .andExpect(jsonPath("$.error.param").value("messages"));
        rejected(
                        "{\"model\":\"" + MODEL
                                + "\",\"reasoning_effort\":\"none\",\"messages\":[{\"role\":\"system\",\"content\":\"s\"}]}",
                        400)
                .andExpect(jsonPath("$.error.message").value("No user query found in messages."));
        rejected(
                        "{\"model\":\"" + MODEL
                                + "\",\"reasoning_effort\":\"none\",\"stream_options\":{\"include_usage\":true},"
                                + HELLO + "}",
                        400)
                .andExpect(jsonPath("$.error.param").value("stream_options"));
        this.mvc
                .perform(post("/v1/chat/completions")
                        .contentType(MediaType.TEXT_PLAIN)
                        .content("hi"))
                .andExpect(status().isUnsupportedMediaType())
                .andExpect(jsonPath("$.error.type").value("invalid_request_error"));
        this.mvc
                .perform(get("/v1/nothing"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error.message").value("Unknown request URL: GET /v1/nothing."));
    }

    @Test
    void unavailableEngineIs503() throws Exception {
        this.backend.available = false;
        rejected("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\"," + HELLO + "}", 503)
                .andExpect(jsonPath("$.error.code").value("engine_unavailable"));
        this.mvc
                .perform(get("/health"))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.engine").value("closed"));
    }

    @Test
    void engineShutdownDuringGenerationIs503AndClosesTheSession() throws Exception {
        this.backend.script = (generation, max, output) -> {
            generation.emit(output, "partial");
            generation.cancel(); // what InferenceEngine.close does to live sessions
            return new InferenceBackend.Result(1, false);
        };
        completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\"," + HELLO + "}")
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.error.type").value("service_unavailable_error"));
        assertTrue(this.backend.only().awaitClosed());
        assertEquals(1, this.backend.only().closeCount.get());
    }

    @Test
    void inferenceFailureIs500WithoutInternalDetail() throws Exception {
        this.backend.script = (generation, max, output) -> {
            throw new ExecutionException(new IllegalStateException("device 0x7f00deadbeef lattice shard 3 failed"));
        };
        var result = completion("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\"," + HELLO + "}")
                .andExpect(status().isInternalServerError())
                .andExpect(jsonPath("$.error.type").value("server_error"))
                .andReturn();
        String body = result.getResponse().getContentAsString();
        assertFalse(body.contains("0x7f00") || body.contains("lattice") || body.contains("Exception"), body);
        // The session closes on the generation's thread, possibly after the response was written.
        assertTrue(this.backend.only().awaitClosed(), "the failed generation's session must close");
        assertEquals(1, this.backend.only().closeCount.get());
    }

    /// Servlet containers report a detected disconnect as an async error; it must cancel and close the session.
    @Test
    void nonStreamingDisconnectSignalCancelsGeneration() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(10);
        MvcResult started = this.mvc
                .perform(post("/v1/chat/completions")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"model\":\"" + MODEL + "\",\"reasoning_effort\":\"none\",\"max_tokens\":3000,"
                                + HELLO + "}"))
                .andReturn();
        var generation = this.backend.awaitFirst();
        assertTrue(generation.started.await(10, TimeUnit.SECONDS));
        var async = (MockAsyncContext) started.getRequest().getAsyncContext();
        for (AsyncListener listener : async.getListeners())
            listener.onError(new AsyncEvent(async, new IOException("Broken pipe")));
        assertTrue(generation.awaitClosed(), "disconnect must close the session");
        assertTrue(generation.isCancelled(), "disconnect must cancel generation");
        assertTrue(generation.emitted.get() < 3000, "generation ran to completion after disconnect");
    }

    @Test
    void healthReportsEngineAndModel() throws Exception {
        this.mvc
                .perform(get("/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("up"))
                .andExpect(jsonPath("$.engine").value("ready"))
                .andExpect(jsonPath("$.model").value(MODEL));
    }

    private ResultActions completion(String body) throws Exception {
        MvcResult started = this.mvc
                .perform(post("/v1/chat/completions")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andReturn();
        assertTrue(
                started.getRequest().isAsyncStarted(),
                "expected asynchronous processing, got status "
                        + started.getResponse().getStatus());
        return this.mvc.perform(asyncDispatch(started));
    }

    /// Rejections before planning (malformed JSON) are synchronous; the rest arrive through the deferred result.
    private ResultActions rejected(String body, int status) throws Exception {
        MvcResult started = this.mvc
                .perform(post("/v1/chat/completions")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andReturn();
        ResultActions response =
                started.getRequest().isAsyncStarted() ? this.mvc.perform(asyncDispatch(started)) : actions(started);
        return response.andExpect(status().is(status))
                .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.error.message").isString());
    }

    private static ResultActions actions(MvcResult result) {
        return new ResultActions() {
            @Override
            public ResultActions andExpect(org.springframework.test.web.servlet.ResultMatcher matcher)
                    throws Exception {
                matcher.match(result);
                return this;
            }

            @Override
            public ResultActions andDo(org.springframework.test.web.servlet.ResultHandler handler) throws Exception {
                handler.handle(result);
                return this;
            }

            @Override
            public MvcResult andReturn() {
                return result;
            }
        };
    }
}
