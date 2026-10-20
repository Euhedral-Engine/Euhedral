package io.euhedral_execution.inference.quantization.mtp;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.api.chat.QwenChatTemplate;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class GeneratePromptTest {
    private final ObjectMapper json = new ObjectMapper();

    @Test
    void assistantToolCallsReadArgumentsAsObjectsOrTextInTheirKeyOrder() throws Exception {
        var message = json.readTree("""
                {"role": "assistant", "content": "", "tool_calls": [
                  {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": {"path": "src/x.py"}}},
                  {"id": "b", "type": "function", "function": {"name": "search", "arguments": "{\\"q\\": \\"x\\", \\"limit\\": 3}"}}
                ]}
                """);
        List<QwenChatTemplate.ToolCall> calls = Generate.Prompt.toolCalls(json, message);
        assertEquals(2, calls.size());
        assertEquals(new QwenChatTemplate.ToolCall("read_file", Map.of("path", "src/x.py")), calls.get(0));
        assertEquals("search", calls.get(1).name());
        assertEquals(List.of("q", "limit"), List.copyOf(calls.get(1).arguments().keySet()));
        assertEquals(3, calls.get(1).arguments().get("limit"));
    }

    @Test
    void aMessageWithoutToolCallsHasNone() throws Exception {
        assertTrue(Generate.Prompt.toolCalls(json, json.readTree("{\"role\": \"user\", \"content\": \"hi\"}"))
                .isEmpty());
    }
}
