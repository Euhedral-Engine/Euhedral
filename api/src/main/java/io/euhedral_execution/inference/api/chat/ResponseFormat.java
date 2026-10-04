package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.openai.OpenAiException;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/// A request's `response_format`: free text, any JSON object, or a JSON document of a schema.
///
/// A JSON format is enforced while sampling, by the schema's llguidance grammar; `strict: false` is enforced as
/// well, since a schema that cannot be enforced is refused outright.
public record ResponseFormat(Kind kind, String name, Map<String, Object> schema) {
    static final ResponseFormat TEXT = new ResponseFormat(Kind.TEXT, null, null);
    private static final Pattern NAME = Pattern.compile("[a-zA-Z0-9_-]{1,64}");

    public enum Kind {
        TEXT,
        JSON_OBJECT,
        JSON_SCHEMA
    }

    boolean json() {
        return this.kind != Kind.TEXT;
    }

    /// The schema of the answer: any object for `json_object`, the request's for `json_schema`.
    Map<String, Object> answerSchema() {
        return switch (this.kind) {
            case TEXT -> null;
            case JSON_OBJECT -> Map.of("type", "object");
            case JSON_SCHEMA -> this.schema;
        };
    }

    @SuppressWarnings("unchecked")
    static ResponseFormat fromRequest(Object value) {
        if (value == null) return TEXT;
        if (!(value instanceof Map<?, ?> format))
            throw OpenAiException.invalidRequest("'response_format' must be an object.", "response_format");
        if (!(format.get("type") instanceof String type))
            throw OpenAiException.invalidRequest("'response_format.type' is required.", "response_format.type");
        switch (type) {
            case "text" -> {
                checkKeys(format, Set.of("type"), "response_format");
                return TEXT;
            }
            case "json_object" -> {
                checkKeys(format, Set.of("type"), "response_format");
                return new ResponseFormat(Kind.JSON_OBJECT, null, null);
            }
            case "json_schema" -> {
                checkKeys(format, Set.of("type", "json_schema"), "response_format");
                if (!(format.get("json_schema") instanceof Map<?, ?> definition))
                    throw OpenAiException.invalidRequest(
                            "'response_format.json_schema' is required.", "response_format.json_schema");
                String path = "response_format.json_schema";
                checkKeys(definition, Set.of("name", "description", "schema", "strict"), path);
                if (!(definition.get("name") instanceof String name)
                        || !NAME.matcher(name).matches())
                    throw OpenAiException.invalidRequest(
                            "'" + path + ".name' must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes.",
                            path + ".name");
                Object description = definition.get("description");
                if (description != null && !(description instanceof String))
                    throw OpenAiException.invalidRequest(
                            "'" + path + ".description' must be a string.", path + ".description");
                Object strict = definition.get("strict");
                if (strict != null && !(strict instanceof Boolean))
                    throw OpenAiException.invalidRequest("'" + path + ".strict' must be a boolean.", path + ".strict");
                Object schema = definition.get("schema");
                if (schema == null) return new ResponseFormat(Kind.JSON_SCHEMA, name, Map.of());
                if (!(schema instanceof Map<?, ?> object))
                    throw OpenAiException.invalidRequest(
                            "'" + path + ".schema' must be a JSON Schema object.", path + ".schema");
                if (object.containsKey("x-guidance"))
                    throw OpenAiException.unsupportedParameter(path + ".schema.x-guidance");
                return new ResponseFormat(Kind.JSON_SCHEMA, name, (Map<String, Object>) object);
            }
            default -> throw OpenAiException.unsupportedParameter("response_format.type=" + type);
        }
    }

    private static void checkKeys(Map<?, ?> object, Set<String> known, String path) {
        for (Object key : object.keySet())
            if (!known.contains(key)) throw OpenAiException.unrecognizedArgument(path + "." + key);
    }
}
