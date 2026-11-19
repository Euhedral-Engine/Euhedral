//! Deterministic benchmark prompts, byte for byte the ones the Java benchmark builds.
//!
//! A prompt is a text of a stated token budget plus the IDs the session encodes for it. Two generators exist:
//! [`words`] (`euhedral-words-v1`, raw text) and [`chat`] (`euhedral-chat-v2`, chat-templated instructions that quote
//! a frozen document). `sha256` is the digest of the UTF-8 text, as the Java benchmark reports it in `promptSha256`.

pub mod chat;
pub mod splittable_random;
pub mod words;

use euhedral_inference_tokenizer::{Tokenizer, TokenizerError};
use sha2::{Digest, Sha256};

/// One prompt: its text, the IDs a session encodes for it (model special tokens included) and the text's SHA-256.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    pub text: String,
    pub ids: Vec<u32>,
    /// Lower-case hex, as Java's `promptSha256`.
    pub sha256: String,
}

/// A generated text and the number of tokens it encodes to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Built {
    pub text: String,
    pub tokens: usize,
}

/// Why a prompt could not be built.
#[derive(Debug, thiserror::Error)]
pub enum PromptError {
    #[error("the target token count must be positive")]
    ZeroTarget,
    #[error("target {target} is below the shortest prompt ({shortest} tokens)")]
    TargetTooSmall { target: usize, shortest: usize },
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
}

/// Which generator builds the prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCorpus {
    /// `euhedral-chat-v2`: [`chat::TASKS`] prompts per target; iteration `i` runs prompt `i` modulo that.
    Chat,
    /// `euhedral-words-v1`: one prompt per target and seed.
    Words,
}

impl PromptCorpus {
    /// The generator's name, as the benchmark's result rows record it.
    pub fn generator(self) -> &'static str {
        match self {
            PromptCorpus::Chat => chat::GENERATOR,
            PromptCorpus::Words => words::GENERATOR,
        }
    }

    /// How many distinct prompts one target has.
    pub fn size(self) -> usize {
        match self {
            PromptCorpus::Chat => chat::TASKS,
            PromptCorpus::Words => 1,
        }
    }

    /// The prompt measured iteration `iteration` runs, of at most `target_tokens` tokens: element `iteration` modulo
    /// [`size`](Self::size) of [`prompts`](Self::prompts). That builds the whole set, so a caller that runs many
    /// iterations of one target builds it once with `prompts` and indexes it itself.
    pub fn prompt(
        self,
        target_tokens: usize,
        iteration: usize,
        seed: u64,
        tok: &Tokenizer,
    ) -> Result<Prompt, PromptError> {
        let mut prompts = self.prompts(target_tokens, seed, tok)?;
        let index = iteration % prompts.len();
        Ok(prompts.swap_remove(index))
    }

    /// Every prompt of one target, in iteration order. `seed` picks the words (the chat corpus is fixed and ignores
    /// it). A chat target is refused when any task's standalone prompt is above it, where Java's
    /// `Prerequisites.check` refuses it, whichever iteration would have run that task.
    pub fn prompts(self, target_tokens: usize, seed: u64, tok: &Tokenizer) -> Result<Vec<Prompt>, PromptError> {
        let mut count = counter(tok);
        let built = match self {
            PromptCorpus::Chat => chat::build_all(target_tokens, &mut count)?,
            PromptCorpus::Words => vec![words::build(target_tokens, seed, &mut count)?],
        };
        built.into_iter().map(|built| finish(built, tok)).collect()
    }
}

/// The token count a session takes: the encoding with model special tokens.
fn counter(tok: &Tokenizer) -> impl FnMut(&str) -> Result<usize, TokenizerError> {
    |text| tok.encode_with_model_special_tokens(text).map(|ids| ids.len())
}

fn finish(built: Built, tok: &Tokenizer) -> Result<Prompt, PromptError> {
    let ids = tok.encode_with_model_special_tokens(&built.text)?;
    let sha256 = sha256_hex(&built.text);
    Ok(Prompt {
        text: built.text,
        ids,
        sha256,
    })
}

/// The SHA-256 of `text` as UTF-8, in lower-case hex.
pub fn sha256_hex(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}
