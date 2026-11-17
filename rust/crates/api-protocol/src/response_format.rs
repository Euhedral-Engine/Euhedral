//! A request's answer format: free text, any JSON object, or a JSON document of a schema.

use crate::error::ApiError;
use crate::json::JsonValue;
use crate::tool_calling::{check_keys, is_function_name, member};

/// A request's answer format. A JSON format is enforced while sampling, by the schema's grammar; `strict: false` is
/// enforced as well, since a schema that cannot be enforced is refused outright. `schema_param` names the schema's
/// request field in errors.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum ResponseFormat {
    #[default]
    Text,
    /// Any JSON object.
    JsonObject,
    /// A JSON document of `schema`, an object (empty when the request gave none).
    JsonSchema {
        name: String,
        schema: JsonValue,
        schema_param: String,
    },
}

impl ResponseFormat {
    /// True for every format but free text.
    pub fn is_json(&self) -> bool {
        !matches!(self, ResponseFormat::Text)
    }

    /// The schema of the answer: any object for `JsonObject`, the request's for `JsonSchema`, none for text.
    pub fn answer_schema(&self) -> Option<JsonValue> {
        match self {
            ResponseFormat::Text => None,
            ResponseFormat::JsonObject => Some(JsonValue::Object(vec![(
                "type".to_owned(),
                JsonValue::String("object".to_owned()),
            )])),
            ResponseFormat::JsonSchema { schema, .. } => Some(schema.clone()),
        }
    }

    /// Reads OpenAI's `response_format`; absent or null is free text.
    pub fn from_request(value: Option<&JsonValue>) -> Result<ResponseFormat, ApiError> {
        let Some(format) = value.filter(|value| **value != JsonValue::Null) else {
            return Ok(ResponseFormat::Text);
        };
        if !matches!(format, JsonValue::Object(_)) {
            return Err(ApiError::invalid_request(
                "'response_format' must be an object.",
                Some("response_format"),
            ));
        }
        let Some(JsonValue::String(kind)) = member(format, "type") else {
            return Err(ApiError::invalid_request(
                "'response_format.type' is required.",
                Some("response_format.type"),
            ));
        };
        match kind.as_str() {
            "text" => {
                check_keys(format, &["type"], "response_format")?;
                Ok(ResponseFormat::Text)
            }
            "json_object" => {
                check_keys(format, &["type"], "response_format")?;
                Ok(ResponseFormat::JsonObject)
            }
            "json_schema" => {
                check_keys(format, &["type", "json_schema"], "response_format")?;
                json_schema(format.get("json_schema"))
            }
            other => Err(ApiError::unsupported_parameter(&format!(
                "response_format.type={other}"
            ))),
        }
    }
}

fn json_schema(definition: Option<&JsonValue>) -> Result<ResponseFormat, ApiError> {
    let path = "response_format.json_schema";
    let Some(definition @ JsonValue::Object(_)) = definition else {
        return Err(ApiError::invalid_request(
            "'response_format.json_schema' is required.",
            Some(path),
        ));
    };
    check_keys(definition, &["name", "description", "schema", "strict"], path)?;
    let name = match member(definition, "name") {
        Some(JsonValue::String(name)) if is_function_name(name) => name.clone(),
        _ => {
            return Err(ApiError::invalid_request(
                format!("'{path}.name' must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes."),
                Some(&format!("{path}.name")),
            ));
        }
    };
    if member(definition, "description").is_some_and(|value| !matches!(value, JsonValue::String(_))) {
        return Err(ApiError::invalid_request(
            format!("'{path}.description' must be a string."),
            Some(&format!("{path}.description")),
        ));
    }
    if member(definition, "strict").is_some_and(|value| !matches!(value, JsonValue::Bool(_))) {
        return Err(ApiError::invalid_request(
            format!("'{path}.strict' must be a boolean."),
            Some(&format!("{path}.strict")),
        ));
    }
    let schema_param = format!("{path}.schema");
    let schema = match member(definition, "schema") {
        None => JsonValue::Object(Vec::new()),
        Some(schema @ JsonValue::Object(_)) => {
            if schema.get("x-guidance").is_some() {
                return Err(ApiError::unsupported_parameter(&format!("{path}.schema.x-guidance")));
            }
            schema.clone()
        }
        Some(_) => {
            return Err(ApiError::invalid_request(
                format!("'{path}.schema' must be a JSON Schema object."),
                Some(&schema_param),
            ));
        }
    };
    Ok(ResponseFormat::JsonSchema {
        name,
        schema,
        schema_param,
    })
}
