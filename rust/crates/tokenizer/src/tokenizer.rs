use std::collections::HashMap;
use std::fmt;

use euhedral_inference_edrl::TokenizerSection;

use crate::byte_level::ByteLevel;
use crate::error::TokenizerError;
use crate::files::{self, ConfigJson, SpecialTokensJson, TokenizerJson};
use crate::stream::{Piece, TokenStream};

const TOKENIZER_JSON: &str = "tokenizer.json";
const TOKENIZER_CONFIG: &str = "tokenizer_config.json";
const SPECIAL_TOKENS: &str = "special_tokens.json";

/// The Qwen byte-level BPE tokenizer an artifact packages.
///
/// Built once from the artifact's [`TokenizerSection`] and shared. Encoding takes no state, so it may run from any
/// number of frames at once. Added tokens (`<|im_start|>`, `<think>`, ...) found in a text encode to their own IDs;
/// everything between them is NFC-normalized, split by the checkpoint's expression and merged by BPE.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    pieces: Vec<Piece>,
    control_ids: HashMap<String, u32>,
    eos_id: u32,
    bos_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    generation_eos_ids: Vec<u32>,
    generation_bos_id: Option<u32>,
}

impl fmt::Debug for Tokenizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tokenizer")
            .field("tokens", &self.pieces.len())
            .field("eos_id", &self.eos_id)
            .field("generation_eos_ids", &self.generation_eos_ids)
            .finish_non_exhaustive()
    }
}

impl Tokenizer {
    /// Builds the tokenizer from the section's files, refusing anything that is not byte-level BPE behind NFC
    /// normalization and one isolated split, or whose special tokens are missing from the vocabulary.
    pub fn from_section(section: &TokenizerSection) -> Result<Tokenizer, TokenizerError> {
        let tokenizer_json = file(section, TOKENIZER_JSON)?;
        let config: ConfigJson = files::parse(TOKENIZER_CONFIG, file(section, TOKENIZER_CONFIG)?)?;
        let special: SpecialTokensJson = files::parse(SPECIAL_TOKENS, file(section, SPECIAL_TOKENS)?)?;

        let description: TokenizerJson = files::parse(TOKENIZER_JSON, tokenizer_json)?;
        description.check_structure()?;
        let inner = tokenizers::Tokenizer::from_bytes(tokenizer_json)
            .map_err(|failure| TokenizerError::Library(failure.to_string()))?;

        let Vocabulary {
            pieces,
            control_ids,
            special_ids,
        } = build_vocabulary(&inner, &description)?;
        let eos_id = resolve_control(&config.eos_token, &control_ids, "EOS")?
            .filter(|id| special_ids.contains(id))
            .ok_or_else(|| TokenizerError::unsupported("the EOS token is not a special added token"))?;
        let bos_id = resolve_control(&config.bos_token, &control_ids, "BOS")?;
        if config.add_bos_token && bos_id.is_none() {
            return Err(TokenizerError::unsupported("the BOS token is required and absent"));
        }

        let mut generation_eos_ids = special
            .eos_token_id
            .map(files::OneOrMany::into_vec)
            .filter(|ids| !ids.is_empty())
            .ok_or_else(|| TokenizerError::unsupported("special_tokens.json has no EOS token IDs"))?;
        generation_eos_ids.sort_unstable();
        generation_eos_ids.dedup();
        for id in generation_eos_ids.iter().copied().chain(special.bos_token_id) {
            if !is_present(&pieces, id) {
                return Err(TokenizerError::unsupported(format!(
                    "special_tokens.json names token {id}, which the vocabulary lacks"
                )));
            }
        }

        Ok(Tokenizer {
            inner,
            pieces,
            control_ids,
            eos_id,
            bos_id,
            add_bos: config.add_bos_token,
            add_eos: config.add_eos_token,
            generation_eos_ids,
            generation_bos_id: special.bos_token_id,
        })
    }

    /// The token IDs of `text`, with no BOS or EOS added.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode_fast(text, false)
            .map_err(|failure| TokenizerError::Library(failure.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// The token IDs of `text` with the BOS and EOS tokens that `tokenizer_config.json` enables: the IDs a session
    /// encodes for a prompt.
    pub fn encode_with_model_special_tokens(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let plain = self.encode(text)?;
        let mut ids = Vec::with_capacity(plain.len() + 2);
        if let Some(bos) = self.bos_id.filter(|_| self.add_bos) {
            ids.push(bos);
        }
        ids.extend(plain);
        if self.add_eos {
            ids.push(self.eos_id);
        }
        Ok(ids)
    }

    /// A stream that decodes generated IDs to text, holding back an incomplete UTF-8 suffix.
    pub fn decode_stream(&self) -> TokenStream<'_> {
        TokenStream::new(&self.pieces)
    }

    /// The text of `ids`, with every control token (EOS included) spelled out.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut stream = self.decode_stream();
        let mut text = stream.push_all(ids)?;
        text.push_str(&stream.finish());
        Ok(text)
    }

    /// The IDs that stop generation (the generation config's EOS IDs), ascending.
    pub fn eos_ids(&self) -> &[u32] {
        &self.generation_eos_ids
    }

    /// The EOS token `tokenizer_config.json` declares.
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    /// The generation config's BOS ID, which encoding never inserts.
    pub fn generation_bos_id(&self) -> Option<u32> {
        self.generation_bos_id
    }

    /// The ID of an added token (special or not), such as `<|im_start|>` or `<think>`.
    pub fn control_token_id(&self, token: &str) -> Option<u32> {
        self.control_ids.get(token).copied()
    }

    /// The size of the ID space: one past the largest ID.
    pub fn vocabulary_size(&self) -> usize {
        self.pieces.len()
    }
}

fn file<'a>(section: &'a TokenizerSection, name: &'static str) -> Result<&'a [u8], TokenizerError> {
    section.file(name).ok_or(TokenizerError::MissingFile { file: name })
}

fn is_present(pieces: &[Piece], id: u32) -> bool {
    usize::try_from(id)
        .ok()
        .and_then(|index| pieces.get(index))
        .is_some_and(|piece| !matches!(piece, Piece::Absent))
}

fn resolve_control(
    token: &Option<String>,
    control_ids: &HashMap<String, u32>,
    what: &str,
) -> Result<Option<u32>, TokenizerError> {
    let Some(token) = token else { return Ok(None) };
    control_ids
        .get(token)
        .copied()
        .map(Some)
        .ok_or_else(|| TokenizerError::unsupported(format!("the {what} token {token} is not an added token")))
}

/// What each ID decodes to, the added tokens by spelling, and the IDs of the special ones.
struct Vocabulary {
    pieces: Vec<Piece>,
    control_ids: HashMap<String, u32>,
    special_ids: Vec<u32>,
}

fn build_vocabulary(inner: &tokenizers::Tokenizer, description: &TokenizerJson) -> Result<Vocabulary, TokenizerError> {
    let alphabet = ByteLevel::new();
    let vocabulary = inner.get_vocab(false);
    let mut added = Vec::with_capacity(description.added_tokens.len());
    for token in &description.added_tokens {
        let id = u32::try_from(token.id)
            .ok()
            .filter(|_| !token.content.is_empty())
            .ok_or_else(|| TokenizerError::unsupported("invalid added-token metadata"))?;
        added.push((id, token));
    }
    let largest = vocabulary
        .values()
        .copied()
        .chain(added.iter().map(|(id, _)| *id))
        .max()
        .ok_or_else(|| TokenizerError::unsupported("the vocabulary is empty"))?;
    let size = usize::try_from(largest)
        .ok()
        .and_then(|largest| largest.checked_add(1))
        .ok_or_else(|| TokenizerError::unsupported("the vocabulary has an out-of-range ID"))?;

    let mut pieces = vec![Piece::Absent; size];
    for (token, id) in &vocabulary {
        let bytes = alphabet
            .token_bytes(token)
            .ok_or_else(|| TokenizerError::unsupported(format!("vocabulary token {token:?} is not byte-level BPE")))?;
        let slot = slot(&mut pieces, *id)?;
        if !matches!(slot, Piece::Absent) {
            return Err(TokenizerError::unsupported(format!("token ID {id} is used twice")));
        }
        *slot = Piece::Bytes(bytes.into_boxed_slice());
    }
    let mut control_ids = HashMap::with_capacity(added.len());
    let mut special_ids = Vec::new();
    for (id, token) in added {
        if control_ids.insert(token.content.clone(), id).is_some() {
            return Err(TokenizerError::unsupported(format!(
                "added token {:?} is listed twice",
                token.content
            )));
        }
        let slot = slot(&mut pieces, id)?;
        if !matches!(slot, Piece::Absent) {
            return Err(TokenizerError::unsupported(format!(
                "added-token ID {id} collides with the vocabulary"
            )));
        }
        *slot = Piece::Control(token.content.as_str().into());
        if token.special {
            special_ids.push(id);
        }
    }
    Ok(Vocabulary {
        pieces,
        control_ids,
        special_ids,
    })
}

fn slot(pieces: &mut [Piece], id: u32) -> Result<&mut Piece, TokenizerError> {
    usize::try_from(id)
        .ok()
        .and_then(|index| pieces.get_mut(index))
        .ok_or_else(|| TokenizerError::unsupported("a token ID is out of range"))
}
