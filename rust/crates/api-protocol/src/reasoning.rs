//! A request's reasoning controls, resolved into the checkpoint template's thinking variables.
//!
//! The template knows thinking on or off and, when on, three efforts: `low` (instructions to keep the thinking
//! brief), `medium` (no instructions) and `xhigh` (instructions to think carefully, the default). The API's
//! `reasoning_effort` names them `low`, `medium` and `high` (`xhigh` is accepted as the same level), and `none`
//! turns thinking off; `minimal` has no counterpart and is refused. The template's own variables are also accepted
//! as `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`, `preserve_thinking`). A request that sets
//! neither gets the template's default.

use crate::error::ApiError;
use crate::json::JsonValue;
use crate::template::{Effort, TemplateOptions};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Level {
    None,
    Low,
    Medium,
    High,
}

/// Resolves the request's `reasoning_effort` and `chat_template_kwargs`; an absent value is `None`.
pub fn thinking(effort: Option<&JsonValue>, template_kwargs: Option<&JsonValue>) -> Result<TemplateOptions, ApiError> {
    let kwargs = match template_kwargs.filter(|value| **value != JsonValue::Null) {
        None => None,
        Some(kwargs @ JsonValue::Object(members)) => {
            for (key, _) in members {
                if !matches!(
                    key.as_str(),
                    "enable_thinking" | "reasoning_effort" | "preserve_thinking"
                ) {
                    return Err(ApiError::unsupported_parameter(&format!("chat_template_kwargs.{key}")));
                }
            }
            Some(kwargs)
        }
        Some(_) => {
            return Err(ApiError::invalid_request(
                "'chat_template_kwargs' must be an object.",
                Some("chat_template_kwargs"),
            ));
        }
    };
    let kwarg = |key: &str| {
        kwargs
            .and_then(|kwargs| kwargs.get(key))
            .filter(|value| **value != JsonValue::Null)
    };
    let enable = flag(kwarg("enable_thinking"), "chat_template_kwargs.enable_thinking")?;
    let preserve = flag(kwarg("preserve_thinking"), "chat_template_kwargs.preserve_thinking")?;
    let level = effort
        .filter(|value| **value != JsonValue::Null)
        .map(|value| level_of(value, "reasoning_effort"))
        .transpose()?;
    let template_level = kwarg("reasoning_effort")
        .map(|value| level_of(value, "chat_template_kwargs.reasoning_effort"))
        .transpose()?;
    if template_level == Some(Level::None) {
        return Err(ApiError::invalid_request(
            "'chat_template_kwargs.reasoning_effort' must be low, medium or xhigh; use enable_thinking to turn \
             thinking off.",
            Some("chat_template_kwargs.reasoning_effort"),
        ));
    }
    if let (Some(level), Some(template_level)) = (level, template_level)
        && level != template_level
    {
        return Err(ApiError::invalid_request(
            "'reasoning_effort' and 'chat_template_kwargs.reasoning_effort' conflict; send only one.",
            Some("reasoning_effort"),
        ));
    }
    let level = level.or(template_level);
    if let (Some(enable), Some(level)) = (enable, level)
        && enable != (level != Level::None)
    {
        return Err(ApiError::invalid_request(
            "'chat_template_kwargs.enable_thinking' conflicts with 'reasoning_effort'.",
            Some("reasoning_effort"),
        ));
    }
    let enabled = enable.unwrap_or(level != Some(Level::None));
    let resolved = match level {
        None | Some(Level::None) | Some(Level::High) => Effort::XHigh,
        Some(Level::Low) => Effort::Low,
        Some(Level::Medium) => Effort::Medium,
    };
    Ok(TemplateOptions {
        effort: enabled.then_some(resolved),
        preserve_thinking: preserve.unwrap_or(true),
    })
}

fn level_of(value: &JsonValue, param: &str) -> Result<Level, ApiError> {
    if let JsonValue::String(name) = value {
        match name.as_str() {
            "none" => return Ok(Level::None),
            "low" => return Ok(Level::Low),
            "medium" => return Ok(Level::Medium),
            "high" | "xhigh" => return Ok(Level::High),
            "minimal" => {
                return Err(ApiError::invalid_request(
                    format!(
                        "'{param}' 'minimal' is not supported: this model's reasoning levels are none, low, medium \
                         and high."
                    ),
                    Some(param),
                ));
            }
            _ => {}
        }
    }
    Err(ApiError::invalid_request(
        format!("'{param}' must be one of none, low, medium, high (or xhigh)."),
        Some(param),
    ))
}

fn flag(value: Option<&JsonValue>, param: &str) -> Result<Option<bool>, ApiError> {
    match value {
        None => Ok(None),
        Some(JsonValue::Bool(flag)) => Ok(Some(*flag)),
        Some(_) => Err(ApiError::invalid_request(
            format!("'{param}' must be a boolean."),
            Some(param),
        )),
    }
}
