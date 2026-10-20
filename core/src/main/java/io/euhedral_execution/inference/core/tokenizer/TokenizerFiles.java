package io.euhedral_execution.inference.core.tokenizer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/// The tokenizer files an artifact carries, as text: the checkpoint's `tokenizer.json` and `tokenizer_config.json`,
/// its chat template, and its special-token ids.
///
/// @param tokenizer `tokenizer.json`
/// @param tokenizerConfig `tokenizer_config.json`
/// @param chatTemplate the Jinja chat template
/// @param specialTokens a JSON object with the `bos_token_id`, `eos_token_id` and `pad_token_id` of
///     `generation_config.json` (those it sets), written without spaces
public record TokenizerFiles(String tokenizer, String tokenizerConfig, String chatTemplate, String specialTokens) {

    /// The `generation_config.json` keys [#specialTokens] keeps, in its order.
    public static final List<String> SPECIAL_TOKEN_KEYS = List.of("bos_token_id", "eos_token_id", "pad_token_id");

    private static final ObjectMapper JSON = new ObjectMapper();

    public TokenizerFiles {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(tokenizerConfig, "tokenizerConfig");
        Objects.requireNonNull(chatTemplate, "chatTemplate");
        Objects.requireNonNull(specialTokens, "specialTokens");
    }

    /// Reads the files from a checkpoint directory: `tokenizer.json`, `tokenizer_config.json`,
    /// `generation_config.json`, and `chat_template.jinja` or else the `chat_template` string of
    /// `tokenizer_config.json`.
    public static TokenizerFiles fromCheckpoint(Path directory) throws IOException {
        Objects.requireNonNull(directory, "directory");
        String tokenizer = read(directory, "tokenizer.json");
        String tokenizerConfig = read(directory, "tokenizer_config.json");
        String chatTemplate;
        Path jinja = directory.resolve("chat_template.jinja");
        if (Files.isRegularFile(jinja)) chatTemplate = Files.readString(jinja, StandardCharsets.UTF_8);
        else {
            JsonNode template = JSON.readTree(tokenizerConfig).path("chat_template");
            if (!template.isTextual()) throw new IOException("checkpoint has no chat template in " + directory);
            chatTemplate = template.asText();
        }
        JsonNode generation = JSON.readTree(read(directory, "generation_config.json"));
        ObjectNode special = JSON.createObjectNode();
        for (String key : SPECIAL_TOKEN_KEYS) if (generation.has(key)) special.set(key, generation.get(key));
        return new TokenizerFiles(tokenizer, tokenizerConfig, chatTemplate, JSON.writeValueAsString(special));
    }

    private static String read(Path directory, String name) throws IOException {
        Path file = directory.resolve(name);
        if (!Files.isRegularFile(file)) throw new IOException("checkpoint has no " + name + " in " + directory);
        return Files.readString(file, StandardCharsets.UTF_8);
    }
}
