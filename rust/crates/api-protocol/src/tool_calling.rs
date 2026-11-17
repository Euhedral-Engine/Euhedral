//! The functions one request offers and how the model may call them, validated at admission.
//!
//! [`ToolCalling::from_request`] reads the OpenAI shape (`tools`, `tool_choice`, `parallel_tool_calls`); the other
//! surfaces build the same value from their own request and share [`FunctionTool`] and the name rules.

use std::collections::HashSet;

use crate::error::ApiError;
use crate::json::JsonValue;
use crate::pipeline::{ToolChoice, ToolOptions};
use crate::tool_definition::ToolDefinition;

/// One validated function tool, from any API surface. `strict` tools have their arguments generated under their
/// full schema, the others only as a JSON object. `parameters_param` names the schema's request field in errors.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionTool {
    pub definition: ToolDefinition,
    pub strict: bool,
    pub parameters_param: String,
}

impl FunctionTool {
    /// A tool whose schema field is named `parameters`.
    pub fn new(definition: ToolDefinition, strict: bool) -> FunctionTool {
        FunctionTool {
            definition,
            strict,
            parameters_param: "parameters".to_owned(),
        }
    }

    /// The tool's name.
    pub fn name(&self) -> &str {
        &self.definition.name
    }

    /// The schema arguments follow when strict: `parameters`, or an empty closed object when there are none.
    pub fn parameters_or_empty(&self) -> JsonValue {
        self.definition.parameters.clone().unwrap_or_else(|| {
            JsonValue::Object(vec![
                ("type".to_owned(), JsonValue::String("object".to_owned())),
                ("properties".to_owned(), JsonValue::Object(Vec::new())),
                ("additionalProperties".to_owned(), JsonValue::Bool(false)),
            ])
        })
    }
}

/// The request's tools and how the model may call them. `ToolChoice::None` keeps the definitions but never offers
/// them to the model, so the output is plain text; a request without tools is [`ToolCalling::disabled`].
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCalling {
    pub tools: Vec<FunctionTool>,
    pub choice: ToolChoice,
    /// False ends decoding once the first call is complete.
    pub parallel: bool,
}

impl ToolCalling {
    /// No tools, no calls.
    pub fn disabled() -> ToolCalling {
        ToolCalling {
            tools: Vec::new(),
            choice: ToolChoice::None,
            parallel: true,
        }
    }

    /// True when the model's output may contain calls and must be parsed.
    pub fn parses_output(&self) -> bool {
        self.choice != ToolChoice::None
    }

    /// Functions the model may call in this response.
    pub fn callable(&self) -> Vec<&FunctionTool> {
        match &self.choice {
            ToolChoice::None => Vec::new(),
            ToolChoice::Function(name) => self.tools.iter().filter(|tool| tool.name() == name).collect(),
            ToolChoice::Auto | ToolChoice::Required => self.tools.iter().collect(),
        }
    }

    /// The definitions for the prompt's tool block; empty renders the template's no-tools branch.
    pub fn prompt_tools(&self) -> Vec<ToolDefinition> {
        if self.choice == ToolChoice::None {
            return Vec::new();
        }
        self.tools.iter().map(|tool| tool.definition.clone()).collect()
    }

    /// The options of the output pipeline; `None` when the output is plain text.
    pub fn pipeline_options(&self) -> Option<ToolOptions> {
        self.parses_output().then(|| ToolOptions {
            tools: self.tools.iter().map(|tool| tool.definition.clone()).collect(),
            choice: self.choice.clone(),
            parallel: self.parallel,
        })
    }

    /// Validates the request's `tools`, `tool_choice` and `parallel_tool_calls`; an absent value is `None`.
    pub fn from_request(
        tools: Option<&JsonValue>,
        tool_choice: Option<&JsonValue>,
        parallel_tool_calls: Option<bool>,
    ) -> Result<ToolCalling, ApiError> {
        let tool_choice = tool_choice.filter(|choice| **choice != JsonValue::Null);
        let Some(tools) = tools.filter(|tools| **tools != JsonValue::Null) else {
            return Self::without_tools(tool_choice);
        };
        let JsonValue::Array(definitions) = tools else {
            return Err(ApiError::invalid_request(
                "'tools' must be an array of tool definitions.",
                Some("tools"),
            ));
        };
        let mut functions = Vec::with_capacity(definitions.len());
        let mut names = HashSet::new();
        for (index, definition) in definitions.iter().enumerate() {
            let tool = function_tool(definition, &format!("tools[{index}]"))?;
            if !names.insert(tool.name().to_owned()) {
                return Err(ApiError::invalid_request(
                    format!("Tool names must be unique; '{}' is defined twice.", tool.name()),
                    Some(&format!("tools[{index}].function.name")),
                ));
            }
            functions.push(tool);
        }
        if functions.is_empty() {
            return Self::without_tools(tool_choice);
        }
        let parallel = parallel_tool_calls != Some(false);
        let choice = match tool_choice {
            None => ToolChoice::Auto,
            Some(JsonValue::String(mode)) => match mode.as_str() {
                "none" => ToolChoice::None,
                "auto" => ToolChoice::Auto,
                "required" => ToolChoice::Required,
                _ => return Err(bad_choice()),
            },
            Some(named @ JsonValue::Object(_)) => ToolChoice::Function(named_function(named, &names)?),
            Some(_) => return Err(bad_choice()),
        };
        Ok(ToolCalling {
            tools: functions,
            choice,
            parallel,
        })
    }

    /// Without tools only a choice that requests no behavior is meaningful.
    fn without_tools(tool_choice: Option<&JsonValue>) -> Result<ToolCalling, ApiError> {
        match tool_choice {
            None => Ok(ToolCalling::disabled()),
            Some(JsonValue::String(mode)) if mode == "none" || mode == "auto" => Ok(ToolCalling::disabled()),
            Some(_) => Err(ApiError::invalid_request(
                "'tool_choice' is only allowed when 'tools' are specified.",
                Some("tool_choice"),
            )),
        }
    }
}

fn bad_choice() -> ApiError {
    ApiError::invalid_request(
        "'tool_choice' must be 'none', 'auto', 'required', or a function choice.",
        Some("tool_choice"),
    )
}

/// A member of an object that is present and not null.
pub(crate) fn member<'a>(object: &'a JsonValue, key: &str) -> Option<&'a JsonValue> {
    object.get(key).filter(|value| **value != JsonValue::Null)
}

/// The first member of `object` whose name is not in `known` is an unrecognized argument at `path`.
pub(crate) fn check_keys(object: &JsonValue, known: &[&str], path: &str) -> Result<(), ApiError> {
    if let JsonValue::Object(members) = object {
        for (key, _) in members {
            if !known.contains(&key.as_str()) {
                return Err(ApiError::unrecognized_argument(&format!("{path}.{key}")));
            }
        }
    }
    Ok(())
}

fn named_function(choice: &JsonValue, names: &HashSet<String>) -> Result<String, ApiError> {
    let Some(JsonValue::String(kind)) = member(choice, "type") else {
        return Err(ApiError::invalid_request(
            "'tool_choice.type' is required.",
            Some("tool_choice.type"),
        ));
    };
    if kind != "function" {
        return Err(ApiError::unsupported_parameter("tool_choice.type"));
    }
    check_keys(choice, &["type", "function"], "tool_choice")?;
    let Some(function @ JsonValue::Object(_)) = choice.get("function") else {
        return Err(ApiError::invalid_request(
            "'tool_choice.function' is required.",
            Some("tool_choice.function"),
        ));
    };
    check_keys(function, &["name"], "tool_choice.function")?;
    let Some(JsonValue::String(name)) = member(function, "name") else {
        return Err(ApiError::invalid_request(
            "'tool_choice.function.name' is required.",
            Some("tool_choice.function.name"),
        ));
    };
    if !names.contains(name) {
        return Err(ApiError::invalid_request(
            format!("'tool_choice' names function '{name}', which is not in 'tools'."),
            Some("tool_choice"),
        ));
    }
    Ok(name.clone())
}

fn function_tool(definition: &JsonValue, path: &str) -> Result<FunctionTool, ApiError> {
    if !matches!(definition, JsonValue::Object(_)) {
        return Err(ApiError::invalid_request("Each tool must be an object.", Some(path)));
    }
    let Some(JsonValue::String(kind)) = member(definition, "type") else {
        return Err(ApiError::invalid_request(
            "Tool 'type' is required.",
            Some(&format!("{path}.type")),
        ));
    };
    if kind != "function" {
        return Err(ApiError::unsupported_parameter(&format!("{path}.type")));
    }
    check_keys(definition, &["type", "function"], path)?;
    let function_path = format!("{path}.function");
    let Some(function @ JsonValue::Object(_)) = definition.get("function") else {
        return Err(ApiError::invalid_request(
            "Function tools require a 'function' object.",
            Some(&function_path),
        ));
    };
    check_keys(
        function,
        &["name", "description", "parameters", "strict"],
        &function_path,
    )?;
    let name = match member(function, "name") {
        Some(JsonValue::String(name)) if is_function_name(name) => name.clone(),
        _ => {
            return Err(ApiError::invalid_request(
                "Function names must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes.",
                Some(&format!("{function_path}.name")),
            ));
        }
    };
    let description = match member(function, "description") {
        None => None,
        Some(JsonValue::String(text)) => Some(text.clone()),
        Some(_) => {
            return Err(ApiError::invalid_request(
                "Function 'description' must be a string.",
                Some(&format!("{function_path}.description")),
            ));
        }
    };
    let strict = match member(function, "strict") {
        None => false,
        Some(JsonValue::Bool(strict)) => *strict,
        Some(_) => {
            return Err(ApiError::invalid_request(
                "Function 'strict' must be a boolean.",
                Some(&format!("{function_path}.strict")),
            ));
        }
    };
    let parameters_param = format!("{function_path}.parameters");
    let parameters = parameters(member(function, "parameters"), &parameters_param)?.cloned();
    Ok(FunctionTool {
        definition: ToolDefinition {
            name,
            description,
            parameters,
        },
        strict,
        parameters_param,
    })
}

/// Checks only what the call format depends on; the rest of the JSON Schema passes through verbatim. `path` names
/// the schema's request field.
pub fn parameters<'a>(value: Option<&'a JsonValue>, path: &str) -> Result<Option<&'a JsonValue>, ApiError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !matches!(value, JsonValue::Object(_)) {
        return Err(ApiError::invalid_request(
            "Function 'parameters' must be a JSON Schema object.",
            Some(path),
        ));
    }
    if let Some(kind) = member(value, "type")
        && *kind != JsonValue::String("object".to_owned())
    {
        return Err(ApiError::invalid_request(
            "Function 'parameters' must describe an object; tool calls pass named arguments.",
            Some(&format!("{path}.type")),
        ));
    }
    let properties = member(value, "properties");
    if let Some(properties) = properties {
        let JsonValue::Object(named) = properties else {
            return Err(ApiError::invalid_request(
                "'properties' must be an object.",
                Some(&format!("{path}.properties")),
            ));
        };
        if named.iter().any(|(name, _)| !is_parameter_name(name)) {
            return Err(ApiError::invalid_request(
                "Parameter names must be non-empty and free of '<', '>' and line breaks.",
                Some(&format!("{path}.properties")),
            ));
        }
    }
    if let Some(required) = member(value, "required") {
        let names = match required {
            JsonValue::Array(items) => items
                .iter()
                .map(|item| match item {
                    JsonValue::String(name) => Some(name.as_str()),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>(),
            _ => None,
        };
        let Some(names) = names else {
            return Err(ApiError::invalid_request(
                "'required' must be an array of strings.",
                Some(&format!("{path}.required")),
            ));
        };
        let closed = matches!(member(value, "additionalProperties"), Some(JsonValue::Bool(false)));
        let mut seen = HashSet::new();
        for name in names {
            if !is_parameter_name(name) || !seen.insert(name) {
                return Err(ApiError::invalid_request(
                    "Required parameter names must be unique, non-empty and free of '<', '>' and line breaks.",
                    Some(&format!("{path}.required")),
                ));
            }
            if closed && !properties.is_some_and(|declared| declared.get(name).is_some()) {
                return Err(ApiError::invalid_request(
                    "Required parameters must be declared when additional properties are forbidden.",
                    Some(&format!("{path}.required")),
                ));
            }
        }
    }
    Ok(Some(value))
}

/// OpenAI's function-name rule, which every surface's tools follow: 1-64 of `a-z`, `A-Z`, `0-9`, `_` and `-`.
pub fn is_function_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

/// A parameter is written as `<parameter=NAME>`, so its name cannot hold the tag's delimiters.
pub fn is_parameter_name(name: &str) -> bool {
    !name.is_empty() && !name.chars().any(|c| matches!(c, '<' | '>' | '\n' | '\r'))
}
