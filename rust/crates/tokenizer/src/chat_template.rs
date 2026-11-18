//! The chat template a tokenizer section carries.

use euhedral_inference_edrl::TokenizerSection;
use serde::Deserialize;
use serde_json::Value;

use crate::error::TokenizerError;
use crate::files;

const CHAT_TEMPLATE_JINJA: &str = "chat_template.jinja";
const TOKENIZER_CONFIG: &str = "tokenizer_config.json";

/// The template's field in `tokenizer_config.json`.
#[derive(Debug, Deserialize)]
struct TemplateConfig {
    chat_template: Option<Value>,
}

/// The source of the chat template of `section`: the text of its `chat_template.jinja` entry when that holds
/// any, otherwise the `chat_template` string of `tokenizer_config.json`. An empty entry stands for a checkpoint
/// without the file. Fails when neither gives a template.
pub fn chat_template(section: &TokenizerSection) -> Result<String, TokenizerError> {
    let jinja = section.text(CHAT_TEMPLATE_JINJA).filter(|text| !text.is_empty());
    if let Some(source) = jinja {
        return Ok(source.to_owned());
    }
    let config = section
        .file(TOKENIZER_CONFIG)
        .ok_or(TokenizerError::MissingFile { file: TOKENIZER_CONFIG })?;
    let config: TemplateConfig = files::parse(TOKENIZER_CONFIG, config)?;
    match config.chat_template {
        Some(Value::String(source)) => Ok(source),
        _ => Err(TokenizerError::NoChatTemplate),
    }
}
