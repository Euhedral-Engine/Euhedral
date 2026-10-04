package io.euhedral_execution.inference.api.chat;

/// A complete function call the model made: the function's name and its arguments, a JSON object as text.
public record GeneratedCall(String name, String arguments) {}
