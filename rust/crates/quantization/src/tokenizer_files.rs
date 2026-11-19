//! The tokenizer files of a checkpoint as an artifact carries them, and the checks that come before they are written.

use euhedral_inference_api_protocol::template::check_template_source;
use euhedral_inference_edrl::{
    FOOTER_BYTES, TokenizerFooter, TokenizerSection, TokenizerTexts, encode_tokenizer_section,
};
use euhedral_inference_tokenizer::{Tokenizer, chat_template};
use serde_json::{Map, Value};

use crate::package_error::PackageError;

/// The `generation_config.json` keys the special-token file keeps, in the order it lists them.
pub const SPECIAL_TOKEN_KEYS: [&str; 3] = ["bos_token_id", "eos_token_id", "pad_token_id"];

/// The four texts of a tokenizer section: a checkpoint's `tokenizer.json` and `tokenizer_config.json`, its chat
/// template, and its special-token ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerFiles {
    tokenizer: String,
    tokenizer_config: String,
    chat_template: String,
    special_tokens: String,
}

impl TokenizerFiles {
    /// The files of a checkpoint from the bytes of what it holds.
    ///
    /// `chat_template` is the checkpoint's `chat_template.jinja` when it has one; otherwise the template is the
    /// `chat_template` string of `tokenizer_config.json`. `generation_config` is `generation_config.json`: the
    /// special-token text keeps the ids of [`SPECIAL_TOKEN_KEYS`] it sets, as written, in that order and without spaces.
    pub fn from_checkpoint(
        tokenizer: Vec<u8>,
        tokenizer_config: Vec<u8>,
        chat_template: Option<Vec<u8>>,
        generation_config: &[u8],
    ) -> Result<TokenizerFiles, PackageError> {
        let tokenizer = text("tokenizer.json", tokenizer)?;
        let tokenizer_config = text("tokenizer_config.json", tokenizer_config)?;
        let chat_template = match chat_template {
            Some(bytes) => text("chat_template.jinja", bytes)?,
            None => {
                let config: Value = serde_json::from_str(&tokenizer_config).map_err(|source| PackageError::Json {
                    file: "tokenizer_config.json",
                    source,
                })?;
                match config.get("chat_template") {
                    Some(Value::String(template)) => template.clone(),
                    _ => return Err(PackageError::NoChatTemplate),
                }
            }
        };
        let generation: Value = serde_json::from_slice(generation_config).map_err(|source| PackageError::Json {
            file: "generation_config.json",
            source,
        })?;
        let mut special = Map::new();
        for key in SPECIAL_TOKEN_KEYS {
            if let Some(value) = generation.get(key) {
                special.insert(key.to_owned(), value.clone());
            }
        }
        Ok(TokenizerFiles {
            tokenizer,
            tokenizer_config,
            chat_template,
            special_tokens: Value::Object(special).to_string(),
        })
    }

    /// The texts, in the order the section stores them.
    pub fn texts(&self) -> TokenizerTexts<'_> {
        TokenizerTexts {
            tokenizer: &self.tokenizer,
            tokenizer_config: &self.tokenizer_config,
            chat_template: &self.chat_template,
            special_tokens: &self.special_tokens,
        }
    }

    /// The section holding the files, body then footer, after the checks that must pass before any byte is written:
    /// the section reads back, the tokenizer loads from it, and its chat template is the one the server renders.
    pub fn encode_checked(&self) -> Result<Vec<u8>, PackageError> {
        let bytes = encode_tokenizer_section(&self.texts())?;
        let (body, footer) = bytes.split_at(bytes.len() - FOOTER_BYTES);
        let footer: [u8; FOOTER_BYTES] = footer.try_into().map_err(|_| PackageError::Section)?;
        let footer = TokenizerFooter::parse(&footer)?.ok_or(PackageError::Section)?;
        let section = TokenizerSection::parse(body.to_vec(), &footer)?;
        Tokenizer::from_section(&section).map_err(|error| PackageError::Tokenizer(error.to_string()))?;
        let source = chat_template(&section).map_err(|error| PackageError::Tokenizer(error.to_string()))?;
        check_template_source(source).map_err(|error| PackageError::Template(error.to_string()))?;
        Ok(bytes)
    }
}

fn text(file: &'static str, bytes: Vec<u8>) -> Result<String, PackageError> {
    String::from_utf8(bytes).map_err(|_| PackageError::NotText { file })
}
