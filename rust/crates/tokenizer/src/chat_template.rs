//! The chat template a tokenizer section carries.

use euhedral_inference_edrl::TokenizerSection;

use crate::error::TokenizerError;

const CHAT_TEMPLATE_JINJA: &str = "chat_template.jinja";

/// The source of the chat template of `section`: its `chat_template.jinja` entry, as it is. The artifact's writer
/// stores the template it chose there (the checkpoint's `.jinja` file, else the `chat_template` of
/// `tokenizer_config.json`), and a section always has the entry, so the config is not read here. An empty entry is
/// returned empty; the template check refuses it.
pub fn chat_template(section: &TokenizerSection) -> Result<&str, TokenizerError> {
    section.text(CHAT_TEMPLATE_JINJA).ok_or(TokenizerError::MissingFile {
        file: CHAT_TEMPLATE_JINJA,
    })
}
