package io.euhedral_execution.inference.api.chat;

/// Token accounting for one generation. `cachedPromptTokens` of the prompt were restored from the prefix cache
/// instead of prefilled; `completionTokens` counts every sampled token, a terminator included, of which
/// `reasoningTokens` (null without a think block) were reasoning.
public record TokenUsage(int promptTokens, int cachedPromptTokens, int completionTokens, Integer reasoningTokens) {}
