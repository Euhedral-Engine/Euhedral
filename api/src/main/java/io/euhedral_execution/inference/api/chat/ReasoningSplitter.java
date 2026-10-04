package io.euhedral_execution.inference.api.chat;

/// Splits output that opens inside the model's think block into reasoning and answer at `</think>`.
///
/// It inverts the template's `'<think>\n' + reasoning|trim + '\n</think>\n\n' + content`: the reasoning is
/// trimmed and the answer's leading whitespace dropped. Reasoning text that could still be part of `</think>`,
/// or whitespace that could still precede it, is held until the next chunk decides it, so reasoning streams in
/// order with at most that much delay. Confined to the generation's callbacks.
final class ReasoningSplitter {
    private static final String THINK_END = QwenChatTemplate.THINK_END;

    /// Receives the two parts in generation order: all reasoning, then all answer.
    interface Output {
        void reasoning(String text);

        void answer(String text);
    }

    private final StringBuilder held = new StringBuilder();
    private boolean answering;
    private boolean reasoningStarted;
    private boolean answerStarted;

    /// True once `</think>` has been generated.
    boolean answering() {
        return this.answering;
    }

    void accept(String chunk, Output output) {
        if (this.answering) {
            answer(chunk, output);
            return;
        }
        this.held.append(chunk);
        int end = this.held.indexOf(THINK_END);
        if (end >= 0) {
            reasoning(this.held.substring(0, trailingSpaceStart(end)), output);
            String rest = this.held.substring(end + THINK_END.length());
            this.held.setLength(0);
            this.answering = true;
            answer(rest, output);
            return;
        }
        int safe = trailingSpaceStart(this.held.length() - heldTagPrefix());
        reasoning(this.held.substring(0, safe), output);
        this.held.delete(0, safe);
    }

    /// Generation ended: what is held is reasoning that never reached `</think>`, less its trailing whitespace.
    void finish(Output output) {
        if (!this.answering) reasoning(this.held.substring(0, trailingSpaceStart(this.held.length())), output);
        this.held.setLength(0);
    }

    private void reasoning(String text, Output output) {
        if (!this.reasoningStarted) {
            text = stripLeading(text);
            if (text.isEmpty()) return;
            this.reasoningStarted = true;
        }
        if (!text.isEmpty()) output.reasoning(text);
    }

    private void answer(String text, Output output) {
        if (!this.answerStarted) {
            text = stripLeading(text);
            if (text.isEmpty()) return;
            this.answerStarted = true;
        }
        output.answer(text);
    }

    private static String stripLeading(String text) {
        int start = 0;
        while (start < text.length() && QwenChatTemplate.isPythonSpace(text.charAt(start))) start++;
        return text.substring(start);
    }

    private int trailingSpaceStart(int end) {
        while (end > 0 && QwenChatTemplate.isPythonSpace(this.held.charAt(end - 1))) end--;
        return end;
    }

    /// Length of the longest held suffix that is a proper prefix of `</think>`.
    private int heldTagPrefix() {
        for (int length = Math.min(THINK_END.length() - 1, this.held.length()); length > 0; length--)
            if (THINK_END.startsWith(this.held.substring(this.held.length() - length))) return length;
        return 0;
    }
}
