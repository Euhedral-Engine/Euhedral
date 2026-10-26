//! The parts of the section's JSON files the tokenizer reads itself. The library reads `tokenizer.json` whole; these
//! types skip the vocabulary and the merges and keep what is checked and what the decoder needs.

use serde::Deserialize;

use crate::error::TokenizerError;

/// `tokenizer.json`, without its vocabulary and merges.
#[derive(Debug, Deserialize)]
pub(crate) struct TokenizerJson {
    pub(crate) normalizer: Option<Component>,
    pub(crate) pre_tokenizer: Option<PreTokenizerJson>,
    pub(crate) model: ModelJson,
    pub(crate) added_tokens: Vec<AddedTokenJson>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Component {
    #[serde(rename = "type")]
    pub(crate) kind: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelJson {
    #[serde(rename = "type")]
    pub(crate) kind: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PreTokenizerJson {
    #[serde(rename = "type")]
    pub(crate) kind: String,
    #[serde(default)]
    pub(crate) pretokenizers: Vec<PreTokenizerPart>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PreTokenizerPart {
    #[serde(rename = "type")]
    pub(crate) kind: String,
    pub(crate) pattern: Option<Pattern>,
    pub(crate) behavior: Option<String>,
    pub(crate) invert: Option<bool>,
    pub(crate) add_prefix_space: Option<bool>,
    pub(crate) use_regex: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Pattern {
    #[serde(rename = "Regex")]
    pub(crate) regex: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AddedTokenJson {
    pub(crate) id: i64,
    pub(crate) content: String,
    #[serde(default)]
    pub(crate) special: bool,
}

/// `tokenizer_config.json`.
#[derive(Debug, Deserialize)]
pub(crate) struct ConfigJson {
    pub(crate) eos_token: Option<String>,
    pub(crate) bos_token: Option<String>,
    #[serde(default)]
    pub(crate) add_bos_token: bool,
    #[serde(default)]
    pub(crate) add_eos_token: bool,
}

/// `special_tokens.json`: the generation config's stop and start IDs.
#[derive(Debug, Deserialize)]
pub(crate) struct SpecialTokensJson {
    pub(crate) eos_token_id: Option<OneOrMany>,
    pub(crate) bos_token_id: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum OneOrMany {
    One(u32),
    Many(Vec<u32>),
}

impl OneOrMany {
    pub(crate) fn into_vec(self) -> Vec<u32> {
        match self {
            OneOrMany::One(id) => vec![id],
            OneOrMany::Many(ids) => ids,
        }
    }
}

pub(crate) fn parse<'a, T: Deserialize<'a>>(file: &'static str, bytes: &'a [u8]) -> Result<T, TokenizerError> {
    serde_json::from_slice(bytes).map_err(|source| TokenizerError::Json { file, source })
}

impl TokenizerJson {
    /// Checks the structure the encoder relies on: byte-level BPE behind NFC normalization and one isolated split.
    pub(crate) fn check_structure(&self) -> Result<(), TokenizerError> {
        if self.model.kind != "BPE" {
            return Err(TokenizerError::unsupported(format!(
                "model type {}, not BPE",
                self.model.kind
            )));
        }
        if self.normalizer.as_ref().map(|normalizer| normalizer.kind.as_str()) != Some("NFC") {
            return Err(TokenizerError::unsupported("the normalizer must be NFC"));
        }
        let Some(pre_tokenizer) = self.pre_tokenizer.as_ref().filter(|part| part.kind == "Sequence") else {
            return Err(TokenizerError::unsupported("the pre-tokenizer must be a sequence"));
        };
        let [split, byte_level] = pre_tokenizer.pretokenizers.as_slice() else {
            return Err(TokenizerError::unsupported(
                "the pre-tokenizer must be a split followed by byte-level encoding",
            ));
        };
        let has_expression = split.pattern.as_ref().is_some_and(|pattern| pattern.regex.is_some());
        if split.kind != "Split" || !has_expression {
            return Err(TokenizerError::unsupported(
                "the pre-tokenizer must start with a split expression",
            ));
        }
        if split.behavior.as_deref() != Some("Isolated") || split.invert != Some(false) {
            return Err(TokenizerError::unsupported("the split must isolate its matches"));
        }
        // The library applies its own split expression when `use_regex` is on, which the artifact's expression replaces.
        if byte_level.kind != "ByteLevel"
            || byte_level.use_regex != Some(false)
            || byte_level.add_prefix_space == Some(true)
        {
            return Err(TokenizerError::unsupported(
                "byte-level encoding must not add a prefix space or apply its own expression",
            ));
        }
        Ok(())
    }
}
