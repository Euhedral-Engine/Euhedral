//! A generation's position in a compiled grammar, and the bounded cache of compiled grammars it starts from.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use llguidance::Matcher;
use llguidance::panic_utils::catch_unwind;
use thiserror::Error;
use toktrie::SimpleVob;

use super::{GrammarChecker, OutputGrammar, summary};
use crate::error::ApiError;
use crate::json::JsonValue;

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
    /// A grammar already checked by [`GrammarChecker::check_grammar`] is cached and compiles no more.
    pub fn constraint(&mut self, grammar: &str) -> Result<GrammarMatcher, ApiError> {
        match self.cloned(Kind::Lark, grammar)? {
            Ok(matcher) => Ok(GrammarMatcher::new(matcher, self.size, Arc::clone(&self.end_ids))),
            Err(refused) => Err(grammar_refusal(&refused)),
        }
    }

    /// A matcher at the start of the JSON Schema `schema`, owned by the caller, sharing the cache of
    /// [`GrammarChecker::constraint`] under its own keys: a schema and a grammar of the same text never meet. A
    /// refusal is an unenforceable schema at `param` and is not cached.
    pub fn json_schema_constraint(&mut self, schema: &JsonValue, param: &str) -> Result<GrammarMatcher, ApiError> {
        match self.cloned(Kind::JsonSchema, &OutputGrammar::schema_text(schema))? {
            Ok(matcher) => Ok(GrammarMatcher::new(matcher, self.size, Arc::clone(&self.end_ids))),
            Err(refused) => Err(schema_refusal(&refused, param)),
        }
    }

    /// Whether `text` of `kind` compiles: found in the cache, else compiled now and kept unless the cache is
    /// disabled. A refusal is llguidance's, in its summary, and is not kept.
    pub(super) fn check(&mut self, kind: Kind, text: &str) -> Result<(), String> {
        if self.compiled.touch(kind, text) {
            return Ok(());
        }
        let compiled = self.compile(kind, text)?;
        if !self.compiled.is_disabled() {
            self.compiled.insert(kind, text, compiled);
        }
        Ok(())
    }

    /// A copy of the compiled grammar of `kind`, from the cache or compiled now and kept.
    fn cloned(&mut self, kind: Kind, text: &str) -> Result<Result<Matcher, String>, ApiError> {
        if self.compiled.touch(kind, text) {
            let template = self.compiled.latest().ok_or_else(ApiError::server_error)?;
            // A panic copying is a server fault, never the request's.
            return catch_unwind(AssertUnwindSafe(|| Ok(template.deep_clone())))
                .map(Ok)
                .map_err(|_| ApiError::server_error());
        }
        let compiled = match self.compile(kind, text) {
            Ok(compiled) => compiled,
            Err(refused) => return Ok(Err(refused)),
        };
        if self.compiled.is_disabled() {
            // Nothing is kept: the compiled grammar is this generation's own.
            return Ok(Ok(compiled));
        }
        let copy =
            catch_unwind(AssertUnwindSafe(|| Ok(compiled.deep_clone()))).map_err(|_| ApiError::server_error())?;
        self.compiled.insert(kind, text, compiled);
        Ok(Ok(copy))
    }

    /// Compiles `text` of `kind`: the matcher, or llguidance's refusal in its summary.
    fn compile(&mut self, kind: Kind, text: &str) -> Result<Matcher, String> {
        #[cfg(test)]
        {
            self.compiled.compilations += 1;
        }
        let compiled = self.matcher(text, kind.tag());
        match compiled.get_error() {
            None => Ok(compiled),
            Some(message) => Err(summary(&message)),
        }
    }
}

/// What is compiled, which is part of a cache key: a JSON schema and a grammar of the same text are two entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Lark,
    JsonSchema,
}

impl Kind {
    /// The type tag llguidance is given with the text.
    fn tag(self) -> &'static str {
        match self {
            Kind::Lark => "lark",
            Kind::JsonSchema => "json_schema",
        }
    }
}

pub(super) fn grammar_refusal(refused: &str) -> ApiError {
    ApiError::invalid_request(format!("The requested output cannot be enforced: {refused}"), None)
}

pub(super) fn schema_refusal(refused: &str, param: &str) -> ApiError {
    ApiError::unenforceable_schema(format!("The JSON Schema cannot be enforced: {refused}"), Some(param))
}

/// The compiled grammars of a checker, keyed by kind and text, least recently used first. The last entry is the one
/// just used or inserted.
pub(super) struct CompiledGrammars {
    capacity: usize,
    entries: Vec<(Kind, Box<str>, Matcher)>,
    /// How many times llguidance compiled a grammar, hit or refused or kept.
    #[cfg(test)]
    pub(super) compilations: usize,
}

impl CompiledGrammars {
    pub(super) fn new(capacity: usize) -> CompiledGrammars {
        CompiledGrammars {
            capacity,
            entries: Vec::new(),
            #[cfg(test)]
            compilations: 0,
        }
    }

    fn is_disabled(&self) -> bool {
        self.capacity == 0
    }

    /// Makes the entry the most recent; false when it is not kept.
    fn touch(&mut self, kind: Kind, text: &str) -> bool {
        let Some(index) = self.entries.iter().position(|(k, key, _)| *k == kind && &**key == text) else {
            return false;
        };
        let entry = self.entries.remove(index);
        self.entries.push(entry);
        true
    }

    /// Keeps `compiled` as the most recent entry, dropping the least recent when the bound is reached.
    fn insert(&mut self, kind: Kind, text: &str, compiled: Matcher) {
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        self.entries.push((kind, text.into(), compiled));
    }

    fn latest(&self) -> Option<&Matcher> {
        self.entries.last().map(|(_, _, matcher)| matcher)
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(super) fn holds(&self, grammar: &str) -> bool {
        self.holds_kind(Kind::Lark, grammar)
    }

    #[cfg(test)]
    pub(super) fn holds_kind(&self, kind: Kind, text: &str) -> bool {
        self.entries.iter().any(|(k, key, _)| *k == kind && &**key == text)
    }
}

#[cfg(test)]
mod tests;
