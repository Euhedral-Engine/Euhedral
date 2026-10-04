package io.euhedral_execution.inference.api.chat;

import java.util.ArrayList;
import java.util.List;

/// The tool results owed to one assistant turn's calls, from any API surface. The template renders results without
/// their IDs, so they are emitted in call order, which is how the model pairs them with its calls; every call needs
/// exactly one result before the conversation goes on. `idField` names the request field that carries the ID.
public final class ToolResults {
    private final List<String> ids;
    private final QwenChatTemplate.Turn[] results;
    private final String callsPath;
    private final String idField;

    public ToolResults(List<String> ids, String callsPath, String idField) {
        this.ids = List.copyOf(ids);
        this.results = new QwenChatTemplate.Turn[this.ids.size()];
        this.callsPath = callsPath;
        this.idField = idField;
    }

    /// Records the result `content` for call `id`; `path` names the ID's request field.
    public void answer(Object id, String content, String path, String idName) {
        if (!(id instanceof String callId) || callId.isEmpty())
            throw ApiException.invalidRequest("Tool results require '" + idName + "'.", path);
        int position = this.ids.indexOf(callId);
        if (position < 0)
            throw ApiException.invalidRequest(
                    "'" + idName + "' " + callId + " does not match a tool call of the preceding assistant message.",
                    path);
        if (this.results[position] != null)
            throw ApiException.invalidRequest("Tool call " + callId + " already has a response.", path);
        this.results[position] = new QwenChatTemplate.Turn(QwenChatTemplate.Role.TOOL, content);
    }

    /// Appends the results in call order; refuses a conversation that goes on before every call was answered.
    public void closeInto(List<QwenChatTemplate.Turn> turns) {
        List<String> missing = new ArrayList<>();
        for (int index = 0; index < this.results.length; index++)
            if (this.results[index] == null) missing.add(this.ids.get(index));
        if (!missing.isEmpty())
            throw ApiException.invalidRequest(
                    "An assistant message with tool calls must be followed by a result for each call, matched by '"
                            + this.idField + "'. Missing responses: " + String.join(", ", missing) + ".",
                    this.callsPath);
        turns.addAll(List.of(this.results));
    }
}
