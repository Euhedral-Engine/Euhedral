//! A function the model may call, as far as parsing its calls needs to know it.

use std::collections::BTreeSet;

use crate::pipeline::json_value::JsonValue;

/// One function tool: its name and the JSON Schema object of its parameters, when the request gave one.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionTool {
    pub name: String,
    pub parameters: Option<JsonValue>,
}

impl FunctionTool {
    pub fn new(name: &str, parameters: Option<JsonValue>) -> Self {
        Self {
            name: name.to_string(),
            parameters,
        }
    }

    /// JSON types the schema declares for a parameter through `type`, `anyOf`/`oneOf` branch types, or an
    /// all-string `enum`/`const`. Empty when the schema does not say, which admits any JSON value.
    pub fn parameter_types(&self, parameter: &str) -> BTreeSet<String> {
        let mut types = BTreeSet::new();
        let Some(schema @ JsonValue::Object(_)) = self.property(parameter) else {
            return types;
        };
        add_types(schema, &mut types);
        for combinator in ["anyOf", "oneOf"] {
            if let Some(JsonValue::Array(branches)) = schema.get(combinator) {
                for branch in branches {
                    if matches!(branch, JsonValue::Object(_)) {
                        add_types(branch, &mut types);
                    }
                }
            }
        }
        if types.is_empty() && is_all_strings(schema) {
            types.insert("string".to_string());
        }
        types
    }

    /// Names the schema's `required` array lists.
    pub fn required_parameters(&self) -> Vec<String> {
        match self.parameters.as_ref().and_then(|schema| schema.get("required")) {
            Some(JsonValue::Array(names)) => names
                .iter()
                .filter_map(|name| match name {
                    JsonValue::String(name) => Some(name.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// False only for a name outside `properties` when the schema sets `additionalProperties: false`.
    pub fn accepts_parameter(&self, parameter: &str) -> bool {
        let closed = matches!(
            self.parameters
                .as_ref()
                .and_then(|schema| schema.get("additionalProperties")),
            Some(JsonValue::Bool(false))
        );
        !closed || self.property(parameter).is_some()
    }

    fn property(&self, parameter: &str) -> Option<&JsonValue> {
        match self.parameters.as_ref()?.get("properties")? {
            properties @ JsonValue::Object(_) => match properties.get(parameter)? {
                JsonValue::Null => None,
                schema => Some(schema),
            },
            _ => None,
        }
    }
}

fn add_types(schema: &JsonValue, types: &mut BTreeSet<String>) {
    match schema.get("type") {
        Some(JsonValue::String(name)) => {
            types.insert(name.clone());
        }
        Some(JsonValue::Array(names)) => {
            for name in names {
                if let JsonValue::String(name) = name {
                    types.insert(name.clone());
                }
            }
        }
        _ => {}
    }
}

fn is_all_strings(schema: &JsonValue) -> bool {
    if matches!(schema.get("const"), Some(JsonValue::String(_))) {
        return true;
    }
    match schema.get("enum") {
        Some(JsonValue::Array(values)) => {
            !values.is_empty() && values.iter().all(|value| matches!(value, JsonValue::String(_)))
        }
        _ => false,
    }
}
