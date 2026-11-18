//! The answer's grammar as the engine's token constraint: the checker over the model's vocabulary, shared by every
//! request, and the adapter that lets a matcher choose the tokens of one generation.
//!
//! Compiling a grammar is CPU work. It runs where the backend is asked: the grammar and schema checks in the planner's
//! own frame on a worker, before the prompt is encoded, which leaves the compiled grammar in the checker's cache; the
//! constraint a generation starts from is then a copy of it. A grammar the cache has dropped meanwhile is compiled
//! again, in the frame that opens the generation. Neither runs on a poll path of the server's own.
#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "the checker's cache is shared by the workers that plan and open requests, behind a lock held only for CPU \
              work (a compile or a copy), never across a wait or a call out"
)]

use std::sync::{Mutex, MutexGuard, PoisonError};

use euhedral_inference_api_protocol::output_grammar::{DEFAULT_CACHED_GRAMMARS, GrammarChecker, GrammarMatcher};
use euhedral_inference_engine::generation::{ConstraintError, TokenConstraint};
use euhedral_inference_engine::sampling::TokenFilter;

use super::Codec;
use crate::backend::BackendError;

/// The checker over the model's vocabulary, or why there is none.
pub(super) struct Grammars {
    checker: Result<Mutex<GrammarChecker>, String>,
}

impl Grammars {
    /// The checker over the codec's vocabulary as wide as the engine's logits row.
    pub(super) fn new(codec: &dyn Codec, vocabulary_size: usize) -> Grammars {
        let bytes = codec.token_bytes(vocabulary_size);
        let checker = GrammarChecker::with_cache_size(&bytes, &codec.end_ids(), DEFAULT_CACHED_GRAMMARS)
            .map(Mutex::new)
            .map_err(|error| error.to_string());
        Grammars { checker }
    }

    fn checker(&self) -> Result<MutexGuard<'_, GrammarChecker>, BackendError> {
        match &self.checker {
            Ok(checker) => Ok(checker.lock().unwrap_or_else(PoisonError::into_inner)),
            Err(reason) => Err(BackendError::Failed(format!("no grammar can be enforced: {reason}"))),
        }
    }

    /// Whether the Lark grammar compiles; a refusal is llguidance's.
    pub(super) fn check_grammar(&self, grammar: &str) -> Result<(), BackendError> {
        self.checker()?
            .check_grammar_text(grammar)
            .map_err(BackendError::Unenforceable)
    }

    /// Whether the JSON Schema (as llguidance reads it) compiles on its own.
    pub(super) fn check_json_schema(&self, schema: &str) -> Result<(), BackendError> {
        self.checker()?
            .check_schema_text(schema)
            .map_err(BackendError::Unenforceable)
    }

    /// A constraint at the start of the grammar.
    pub(super) fn constraint(&self, grammar: &str) -> Result<Box<dyn TokenConstraint>, BackendError> {
        let matcher = self
            .checker()?
            .constraint(grammar)
            .map_err(|error| BackendError::Failed(error.message))?;
        Ok(Box::new(MatcherConstraint(matcher)))
    }
}

/// A grammar matcher as a token constraint: `prepare` computes the step's mask, which `allows` and `mask_disallowed`
/// read, and `accept` consumes the token.
pub(super) struct MatcherConstraint(pub(super) GrammarMatcher);

impl TokenFilter for MatcherConstraint {
    fn allows(&self, id: u32) -> bool {
        self.0.mask().is_some_and(|words| bit(words, id))
    }

    fn mask_disallowed(&self, row: &mut [f32]) {
        let Some(words) = self.0.mask() else {
            // No mask is no token: a step that was not prepared selects nothing.
            row.fill(f32::NEG_INFINITY);
            return;
        };
        for (word, chunk) in words.iter().zip(row.chunks_mut(32)) {
            if *word == u32::MAX && chunk.len() == 32 {
                continue;
            }
            for (bit, logit) in chunk.iter_mut().enumerate() {
                if word & (1 << bit) == 0 {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
        // Ids past the mask's last word are not in the vocabulary.
        let covered = words.len() * 32;
        if let Some(rest) = row.get_mut(covered..) {
            rest.fill(f32::NEG_INFINITY);
        }
    }
}

impl TokenConstraint for MatcherConstraint {
    fn prepare(&mut self) -> Result<(), ConstraintError> {
        self.0
            .compute_mask()
            .map(|_| ())
            .map_err(|error| ConstraintError::new(error.to_string()))
    }

    fn accept(&mut self, id: u32) -> Result<(), ConstraintError> {
        self.0
            .consume(id)
            .map_err(|error| ConstraintError::new(error.to_string()))
    }
}

fn bit(words: &[u32], id: u32) -> bool {
    words
        .get((id >> 5) as usize)
        .is_some_and(|word| word & (1 << (id & 31)) != 0)
}
