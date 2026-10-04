package io.euhedral_execution.inference.api.chat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/// One validated OpenAI function tool. `description` and `parameters` are null when the request omits
/// them; `parameters` keeps the request's JSON Schema object, key order included. A `strict` tool's arguments
/// are generated under its schema; the others' only as a JSON object.
public record FunctionTool(String name, String description, Map<String, Object> parameters, boolean strict) {

    public FunctionTool {
        Objects.requireNonNull(name, "name");
        if (parameters != null) parameters = Collections.unmodifiableMap(new LinkedHashMap<>(parameters));
    }

    public FunctionTool(String name, String description, Map<String, Object> parameters) {
        this(name, description, parameters, false);
    }

    /// The schema arguments follow when strict: `parameters`, or an empty object when there are none.
    Map<String, Object> parametersOrEmpty() {
        if (this.parameters != null) return this.parameters;
        return Map.of("type", "object", "properties", Map.of(), "additionalProperties", false);
    }

    /// JSON types the schema declares for a parameter through `type`, `anyOf`/`oneOf` branch types, or an
    /// all-string `enum`/`const`. Empty when the schema does not say, which admits any JSON value.
    Set<String> parameterTypes(String parameter) {
        if (!(property(parameter) instanceof Map<?, ?> schema)) return Set.of();
        Set<String> types = new HashSet<>();
        addTypes(schema, types);
        for (String combinator : List.of("anyOf", "oneOf")) {
            if (schema.get(combinator) instanceof List<?> branches)
                for (Object branch : branches)
                    if (branch instanceof Map<?, ?> branchSchema) addTypes(branchSchema, types);
        }
        if (types.isEmpty() && isAllStrings(schema)) types.add("string");
        return Set.copyOf(types);
    }

    /// Names the schema's `required` array lists; validated as strings at admission.
    List<String> requiredParameters() {
        if (this.parameters == null || !(this.parameters.get("required") instanceof List<?> required)) return List.of();
        List<String> names = new ArrayList<>(required.size());
        for (Object name : required) names.add((String) name);
        return names;
    }

    /// False only for a name outside `properties` when the schema sets `additionalProperties: false`.
    boolean acceptsParameter(String parameter) {
        if (this.parameters == null || !Boolean.FALSE.equals(this.parameters.get("additionalProperties"))) return true;
        return property(parameter) != null;
    }

    private Object property(String parameter) {
        if (this.parameters == null || !(this.parameters.get("properties") instanceof Map<?, ?> properties))
            return null;
        return properties.get(parameter);
    }

    private static void addTypes(Map<?, ?> schema, Set<String> types) {
        switch (schema.get("type")) {
            case String type -> types.add(type);
            case List<?> list -> {
                for (Object type : list) if (type instanceof String name) types.add(name);
            }
            case null, default -> {}
        }
    }

    private static boolean isAllStrings(Map<?, ?> schema) {
        if (schema.get("const") instanceof String) return true;
        if (!(schema.get("enum") instanceof List<?> values) || values.isEmpty()) return false;
        for (Object value : values) if (!(value instanceof String)) return false;
        return true;
    }

    /// The object the checkpoint template serializes with `tojson`, in canonical key order so equal tools
    /// always render identical prompt bytes: `type`, then `function` with `name`, `description`, `parameters`.
    Map<String, Object> templateObject() {
        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", this.name);
        if (this.description != null) function.put("description", this.description);
        if (this.parameters != null) function.put("parameters", this.parameters);
        Map<String, Object> tool = new LinkedHashMap<>();
        tool.put("type", "function");
        tool.put("function", function);
        return tool;
    }
}
