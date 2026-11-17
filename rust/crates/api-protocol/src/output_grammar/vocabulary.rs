//! The vocabulary a grammar is checked and enforced over, as the tokenizer gives it.

use euhedral_inference_tokenizer::Tokenizer;

/// The byte `0xFF` that starts the bytes of a token a grammar may only name.
const NAMED: u8 = 0xFF;

/// The table [`GrammarChecker::new`](super::GrammarChecker::new) takes for a logits row of `vocabulary_size`
/// entries: token `id` is its exact bytes when it is an ordinary token (a token that ends inside a UTF-8 sequence
/// keeps its partial bytes); a control token is `0xFF` followed by its spelling; and an ID the vocabulary leaves
/// unused, or one past it, is `0xFF` followed by `<[unused ID]>`, so every entry is non-empty and a grammar matches
/// such a token only by naming it.
pub fn token_bytes(tokenizer: &Tokenizer, vocabulary_size: usize) -> Vec<Vec<u8>> {
    (0..vocabulary_size)
        .map(|index| {
            let id = u32::try_from(index).unwrap_or(u32::MAX);
            match tokenizer.token_bytes(id) {
                Some(bytes) if !bytes.is_empty() => bytes.to_vec(),
                Some(_) => vec![NAMED],
                None => match tokenizer.control_token_text(id) {
                    Some(name) => named(name),
                    None => named(&format!("<[unused {index}]>")),
                },
            }
        })
        .collect()
}

/// The IDs that end a grammar: the tokenizer's EOS first (the main end token), then the generation's EOS IDs
/// ascending, each once.
pub fn end_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    let mut ids = vec![tokenizer.eos_id()];
    for id in tokenizer.eos_ids() {
        if !ids.contains(id) {
            ids.push(*id);
        }
    }
    ids
}

fn named(name: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(name.len() + 1);
    bytes.push(NAMED);
    bytes.extend_from_slice(name.as_bytes());
    bytes
}
