//! A generation's position in a compiled grammar, and the bounded cache of compiled grammars it starts from.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use llguidance::Matcher;
use llguidance::panic_utils::catch_unwind;
use thiserror::Error;
use toktrie::SimpleVob;

use super::{GrammarChecker, summary};
use crate::error::ApiError;

/// Compiled grammars a checker keeps for reuse unless told otherwise.
pub const DEFAULT_CACHED_GRAMMARS: usize = 32;

/// Why a matcher refused a step. After any of these but [`MatcherError::Violation`] the matcher is unusable.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MatcherError {
    /// The token is not allowed at this position; the matcher is unchanged.
    #[error("token {0} violates the grammar")]
    Violation(u32),
    /// llguidance failed (or panicked) while computing a mask or consuming a token.
    #[error("llguidance failed: {0}")]
    Failed(String),
}

/// One generation's position in a grammar: the token mask for the next step, computed once when the step first asks
/// and reused until a token is consumed.
///
/// Used by one generation step after another, never concurrently, so it holds no lock. It is `Send`, so a frame
/// carries it between workers. Once llguidance has failed (including by panic) every later call reports that
/// failure and [`GrammarMatcher::is_usable`] is false.
pub struct GrammarMatcher {
    matcher: Matcher,
    size: u32,
    end_ids: Arc<[u32]>,
    mask: Option<SimpleVob>,
    failure: Option<String>,
}

impl GrammarMatcher {
    fn new(matcher: Matcher, size: u32, end_ids: Arc<[u32]>) -> GrammarMatcher {
        GrammarMatcher {
            matcher,
            size,
            end_ids,
            mask: None,
            failure: None,
        }
    }

    /// The tokens allowed next, one bit per token id (bit `id % 32` of word `id / 32`), the end ids included
    /// exactly when the grammar may end. Bits at or above the vocabulary's size are never set.
    pub fn compute_mask(&mut self) -> Result<&[u32], MatcherError> {
        self.ensure_mask()?;
        Ok(self.mask.as_ref().map_or(&[][..], SimpleVob::as_slice))
    }

    /// Whether `id` is allowed next; an id at or above the vocabulary's size never is.
    pub fn allows(&mut self, id: u32) -> Result<bool, MatcherError> {
        if id >= self.size {
            return Ok(false);
        }
        let words = self.compute_mask()?;
        Ok(words
            .get((id >> 5) as usize)
            .is_some_and(|word| word & (1 << (id & 31)) != 0))
    }

    /// Commits `id`. An end id ends the generation: it is allowed once the grammar may end and consumes nothing.
    pub fn consume(&mut self, id: u32) -> Result<(), MatcherError> {
        if !self.allows(id)? {
            return Err(MatcherError::Violation(id));
        }
        if self.end_ids.contains(&id) {
            return Ok(());
        }
        self.mask = None;
        let consumed = self.matcher.consume_token(id);
        consumed.map_err(|error| self.fail(error.to_string()))
    }

    /// Whether the text so far is a whole sentence of the grammar; false once the matcher is unusable.
    pub fn is_accepting(&mut self) -> bool {
        if self.failure.is_some() {
            return false;
        }
        match self.matcher.is_accepting() {
            Ok(accepting) => accepting,
            Err(error) => {
                self.fail(error.to_string());
                false
            }
        }
    }

    /// The mask computed for this step by [`GrammarMatcher::compute_mask`], read without computing; `None` before
    /// it is computed, after a [`GrammarMatcher::consume`] and after a failure.
    pub fn mask(&self) -> Option<&[u32]> {
        self.mask.as_ref().map(SimpleVob::as_slice)
    }

    /// False once llguidance has failed; the matcher then refuses every step.
    pub fn is_usable(&self) -> bool {
        self.failure.is_none()
    }

    fn ensure_mask(&mut self) -> Result<(), MatcherError> {
        if let Some(failure) = &self.failure {
            return Err(MatcherError::Failed(failure.clone()));
        }
        if self.mask.is_none() {
            // The end ids are in the mask when the grammar may end, and alone once it has stopped.
            match self.matcher.compute_mask_or_eos() {
                Ok(mask) => self.mask = Some(mask),
                Err(error) => return Err(self.fail(error.to_string())),
            }
        }
        Ok(())
    }

    fn fail(&mut self, message: String) -> MatcherError {
        self.mask = None;
        self.failure = Some(message.clone());
        MatcherError::Failed(message)
    }
}

impl Debug for GrammarMatcher {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("GrammarMatcher")
            .field("usable", &self.is_usable())
            .finish_non_exhaustive()
    }
}

impl GrammarChecker {
    /// A matcher at the start of the Lark grammar `grammar`, owned by the caller. The compiled grammar is kept
    /// (least recently used first out, within the cache's bound) so that a client repeating its tools or schema
    /// compiles them once; each call starts from its own copy, independent of every other.
    ///
    /// The cache is the checker's own state, polled by whoever owns the checker one request at a time: no lock, no
    /// wait. Compiling costs enough that the owner does it as its own frame, never inline in a request handler; a
    /// cached grammar only copies. A grammar llguidance refuses is a 400 without a parameter and is not cached.
    pub fn constraint(&mut self, grammar: &str) -> Result<GrammarMatcher, ApiError> {
        let matcher = if self.compiled.touch(grammar) {
            let template = self.compiled.latest().ok_or_else(ApiError::server_error)?;
            // A panic copying is a server fault, never the request's.
            catch_unwind(AssertUnwindSafe(|| Ok(template.deep_clone()))).map_err(|_| ApiError::server_error())?
        } else {
            let compiled = self.matcher(grammar, "lark");
            if let Some(message) = compiled.get_error() {
                return Err(ApiError::invalid_request(
                    format!("The requested output cannot be enforced: {}", summary(&message)),
                    None,
                ));
            }
            if self.compiled.is_disabled() {
                // Nothing is kept: the compiled grammar is this generation's own.
                compiled
            } else {
                let copy = catch_unwind(AssertUnwindSafe(|| Ok(compiled.deep_clone())))
                    .map_err(|_| ApiError::server_error())?;
                self.compiled.insert(grammar, compiled);
                copy
            }
        };
        Ok(GrammarMatcher::new(matcher, self.size, Arc::clone(&self.end_ids)))
    }
}

/// The compiled grammars of a checker, keyed by grammar text, least recently used first. The last entry is the one
/// just used or inserted.
pub(super) struct CompiledGrammars {
    capacity: usize,
    entries: Vec<(Box<str>, Matcher)>,
}

impl CompiledGrammars {
    pub(super) fn new(capacity: usize) -> CompiledGrammars {
        CompiledGrammars {
            capacity,
            entries: Vec::new(),
        }
    }

    fn is_disabled(&self) -> bool {
        self.capacity == 0
    }

    /// Makes `grammar` the most recent entry; false when it is not kept.
    fn touch(&mut self, grammar: &str) -> bool {
        let Some(index) = self.entries.iter().position(|(key, _)| &**key == grammar) else {
            return false;
        };
        let entry = self.entries.remove(index);
        self.entries.push(entry);
        true
    }

    /// Keeps `compiled` as the most recent entry, dropping the least recent when the bound is reached.
    fn insert(&mut self, grammar: &str, compiled: Matcher) {
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        self.entries.push((grammar.into(), compiled));
    }

    fn latest(&self) -> Option<&Matcher> {
        self.entries.last().map(|(_, matcher)| matcher)
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(super) fn holds(&self, grammar: &str) -> bool {
        self.entries.iter().any(|(key, _)| &**key == grammar)
    }
}

#[cfg(test)]
mod tests;
