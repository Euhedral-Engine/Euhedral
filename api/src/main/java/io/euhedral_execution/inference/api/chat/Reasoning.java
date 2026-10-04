package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.openai.OpenAiException;
import java.util.Map;

/// Resolves a request's reasoning controls into the checkpoint template's thinking variables.
///
/// The Qwen3.8 template knows thinking on or off and, when on, three efforts: `low` (instructions to keep the
/// thinking brief), `medium` (no instructions), and `xhigh` (instructions to think carefully, the default). The
/// API's `reasoning_effort` names them `low`, `medium` and `high` (`xhigh` is accepted as the same level), and
/// `none` turns thinking off; `minimal` has no counterpart and is refused. The template's own variables are
/// also accepted as `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`, `preserve_thinking`), as
/// other servers for this model family accept them. A request that sets neither gets the template's default.
public final class Reasoning {
    private Reasoning() {}

    /// `effort` is the request's `reasoning_effort`; `templateKwargs` its `chat_template_kwargs`.
    public static QwenChatTemplate.Thinking thinking(Object effort, Object templateKwargs) {
        Map<?, ?> kwargs = Map.of();
        if (templateKwargs != null) {
            if (!(templateKwargs instanceof Map<?, ?> map))
                throw OpenAiException.invalidRequest(
                        "'chat_template_kwargs' must be an object.", "chat_template_kwargs");
            kwargs = map;
            for (Object key : kwargs.keySet()) {
                if (!key.equals("enable_thinking")
                        && !key.equals("reasoning_effort")
                        && !key.equals("preserve_thinking"))
                    throw OpenAiException.unsupportedParameter("chat_template_kwargs." + key);
            }
        }
        Boolean enable = flag(kwargs.get("enable_thinking"), "chat_template_kwargs.enable_thinking");
        Boolean preserve = flag(kwargs.get("preserve_thinking"), "chat_template_kwargs.preserve_thinking");
        Level level = effort == null ? null : level(effort, "reasoning_effort");
        Level templateLevel = kwargs.get("reasoning_effort") == null
                ? null
                : level(kwargs.get("reasoning_effort"), "chat_template_kwargs.reasoning_effort");
        if (templateLevel == Level.NONE)
            throw OpenAiException.invalidRequest(
                    "'chat_template_kwargs.reasoning_effort' must be low, medium or xhigh; use enable_thinking to turn"
                            + " thinking off.",
                    "chat_template_kwargs.reasoning_effort");
        if (level != null && templateLevel != null && level != templateLevel)
            throw OpenAiException.invalidRequest(
                    "'reasoning_effort' and 'chat_template_kwargs.reasoning_effort' conflict; send only one.",
                    "reasoning_effort");
        if (level == null) level = templateLevel;
        if (enable != null && level != null && enable != (level != Level.NONE))
            throw OpenAiException.invalidRequest(
                    "'chat_template_kwargs.enable_thinking' conflicts with 'reasoning_effort'.", "reasoning_effort");
        boolean enabled = enable != null ? enable : level != Level.NONE;
        QwenChatTemplate.Effort resolved =
                level == null || level == Level.NONE ? QwenChatTemplate.Effort.XHIGH : level.effort;
        return new QwenChatTemplate.Thinking(enabled ? resolved : null, preserve == null || preserve);
    }

    private enum Level {
        NONE(null),
        LOW(QwenChatTemplate.Effort.LOW),
        MEDIUM(QwenChatTemplate.Effort.MEDIUM),
        HIGH(QwenChatTemplate.Effort.XHIGH);

        private final QwenChatTemplate.Effort effort;

        Level(QwenChatTemplate.Effort effort) {
            this.effort = effort;
        }
    }

    private static Level level(Object value, String param) {
        if (value instanceof String name) {
            switch (name) {
                case "none":
                    return Level.NONE;
                case "low":
                    return Level.LOW;
                case "medium":
                    return Level.MEDIUM;
                case "high", "xhigh":
                    return Level.HIGH;
                case "minimal":
                    throw OpenAiException.invalidRequest(
                            "'" + param + "' 'minimal' is not supported: this model's reasoning levels are none, low,"
                                    + " medium and high.",
                            param);
                default:
                    break;
            }
        }
        throw OpenAiException.invalidRequest(
                "'" + param + "' must be one of none, low, medium, high (or xhigh).", param);
    }

    private static Boolean flag(Object value, String param) {
        if (value == null) return null;
        if (value instanceof Boolean flag) return flag;
        throw OpenAiException.invalidRequest("'" + param + "' must be a boolean.", param);
    }
}
