package io.euhedral_execution.inference.benchmark.prompt;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.ToIntFunction;

/// Deterministic chat prompts: real instructions in the model's chat template, with thinking disabled,
/// so generation behaves like a served request (draft acceptance in particular depends on it; random word
/// prompts make the model repeat itself).
///
/// A scenario gets one prompt per task. When the target leaves room, a prompt quotes the longest
/// paragraph-aligned prefix of a frozen document that keeps it within the target, and its task refers to
/// that document; otherwise it carries a standalone task. The document is a snapshot of docs/FRAME_MODEL.md
/// followed by other repository documents and Java sources (about 60K tokens), so 16K and 32K prompts are
/// real long contexts. v2 only appended to v1's document: prompts that fit in its first ~10K tokens are
/// byte-identical to v1's.
/// `actualTokens` is at most the target. Every task asks for long-form output.
public final class ChatPromptCorpus {
    public static final String GENERATOR = "euhedral-chat-v2";
    /// The least document a prompt quotes; shorter targets use standalone tasks.
    static final int MIN_DOCUMENT_TOKENS = 64;

    static final List<String> DOCUMENT_TASKS = List.of(
            "Summarize the document above in detail, section by section.",
            "Write a Java implementation of the main mechanism the document above describes, with comments"
                    + " explaining each part.",
            "List the design decisions in the document above and explain the trade-off behind each one.",
            "Write a tutorial for a new engineer that explains the ideas in the document above, with examples.");
    static final List<String> STANDALONE_TASKS = List.of(
            "Write a Java class that implements an LRU cache, with comments, and explain how it works.",
            "Explain in detail how a CPU executes an instruction, from fetch to write-back.",
            "Write a Python script that reads a CSV file of sales and prints totals per month, then explain it"
                    + " line by line.",
            "Write a short story about a lighthouse keeper who finds a message in a bottle.");

    private ChatPromptCorpus() {}

    public static int size() {
        return DOCUMENT_TASKS.size();
    }

    public static List<PromptMaterial> build(int targetTokens, ToIntFunction<String> tokenCount) {
        return build(targetTokens, tokenCount, document());
    }

    static List<PromptMaterial> build(int targetTokens, ToIntFunction<String> tokenCount, String document) {
        if (targetTokens <= 0) throw new IllegalArgumentException("targetTokens must be positive");
        // A Windows checkout may hold the document with CRLF line endings; prompts are built from its LF
        // form so they, and their hashes, are the same on every platform.
        String[] paragraphs = document.replace("\r\n", "\n").split("\n\n");
        List<PromptMaterial> prompts = new ArrayList<>();
        for (int task = 0; task < size(); task++) {
            String standalone = chat(null, STANDALONE_TASKS.get(task));
            int standaloneTokens = tokenCount.applyAsInt(standalone);
            if (standaloneTokens > targetTokens)
                throw new IllegalArgumentException("target " + targetTokens + " is below the shortest chat prompt");
            String text = standalone;
            int count = standaloneTokens;
            String instruction = DOCUMENT_TASKS.get(task);
            if (tokenCount.applyAsInt(chat(paragraphs[0], instruction)) <= targetTokens) {
                int low = 1, high = paragraphs.length;
                while (low < high) {
                    int middle = (low + high + 1) >>> 1;
                    if (tokenCount.applyAsInt(chat(join(paragraphs, middle), instruction)) <= targetTokens)
                        low = middle;
                    else high = middle - 1;
                }
                String quoted = chat(join(paragraphs, low), instruction);
                int quotedCount = tokenCount.applyAsInt(quoted);
                if (quotedCount - standaloneTokens >= MIN_DOCUMENT_TOKENS) {
                    text = quoted;
                    count = quotedCount;
                }
            }
            prompts.add(new PromptMaterial(text, targetTokens, count, PromptMaterial.sha256(text), GENERATOR));
        }
        return List.copyOf(prompts);
    }

    private static String chat(String document, String task) {
        String user = document == null ? task : "Here is a document:\n\n" + document + "\n\n" + task;
        return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }

    private static String join(String[] paragraphs, int count) {
        return String.join("\n\n", java.util.Arrays.asList(paragraphs).subList(0, count));
    }

    static String document() {
        try (InputStream in = ChatPromptCorpus.class.getResourceAsStream("chat-corpus-v1-document.md")) {
            if (in == null) throw new IllegalStateException("missing chat-corpus-v1-document.md");
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }
}
