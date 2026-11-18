//! The answer's grammar as the engine's token constraint: the checker over the model's vocabulary, shared by every
//! request, and the adapter that lets a matcher choose the tokens of one generation.
//!
//! Compiling a grammar is CPU work and runs with no lock held: under the lock a worker looks the grammar up in the
//! checker's cache and copies it; on a miss it compiles on its own (the compiler shares the vocabulary's factory),
//! then takes the lock again to keep the result, or finds that another worker kept the same grammar meanwhile and
//! uses that. The checks run in the planner's own frame on a worker, before the prompt is encoded, and leave the
//! grammar in the cache; the constraint a generation starts from is then a copy of it. A grammar the cache has
//! dropped meanwhile is compiled again, in the frame that opens the generation. Neither runs on a poll path of the
//! server's own.

#[allow(
    clippy::disallowed_types,
    reason = "the cache is shared by workers; the lock covers lookups and copies only, never a compile"
)]
use std::sync::Mutex;
use std::sync::{MutexGuard, PoisonError};

use euhedral_inference_api_protocol::ApiError;
use euhedral_inference_api_protocol::output_grammar::{
    DEFAULT_CACHED_GRAMMARS, GrammarChecker, GrammarCompiler, GrammarKind, GrammarMatcher,
};
use euhedral_inference_engine::generation::{ConstraintError, TokenConstraint};
use euhedral_inference_engine::sampling::TokenFilter;

use super::Codec;
use crate::backend::BackendError;

/// A test's hook, called with the text of a grammar about to be compiled.
#[cfg(test)]
type CompileHook = Box<dyn Fn(&str) + Send + Sync>;

/// The checker over the model's vocabulary and the compiler that works beside it.
pub(super) struct Grammars {
    /// The cache of compiled grammars, shared by the workers that plan and open requests. It is held for a lookup, a
    /// copy or an insertion, never across a compile or a call out.
    #[allow(
        clippy::disallowed_types,
        reason = "the cache is shared by workers; the lock covers lookups and copies only, never a compile"
    )]
    cache: Mutex<GrammarChecker>,
    compiler: GrammarCompiler,
    /// Called with the text of each grammar about to be compiled, outside the lock.
    #[cfg(test)]
    before_compile: Option<CompileHook>,
}

impl Grammars {
    /// The checker over the codec's vocabulary as wide as the engine's logits row, or why llguidance refuses it.
    #[allow(
        clippy::disallowed_types,
        clippy::disallowed_methods,
        reason = "the cache is shared by workers; the lock covers lookups and copies only, never a compile"
    )]
    pub(super) fn new(codec: &dyn Codec, vocabulary_size: usize) -> Result<Grammars, String> {
        let bytes = codec.token_bytes(vocabulary_size);
        let checker = GrammarChecker::with_cache_size(&bytes, &codec.end_ids(), DEFAULT_CACHED_GRAMMARS)
            .map_err(|error| error.to_string())?;
        Ok(Grammars {
            compiler: checker.compiler(),
            cache: Mutex::new(checker),
            #[cfg(test)]
            before_compile: None,
        })
    }

    #[allow(
        clippy::disallowed_methods,
        reason = "the cache is shared by workers; the lock covers lookups and copies only, never a compile"
    )]
    fn cache(&self) -> MutexGuard<'_, GrammarChecker> {
        self.cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Compiles `text` unless it is cached, and keeps it. The refusal is llguidance's; a fault is the server's.
    fn ensure(&self, kind: GrammarKind, text: &str) -> Result<Result<GrammarMatcher, String>, ApiError> {
        if let Some(matcher) = self.cache().cached(kind, text)? {
            return Ok(Ok(matcher));
        }
        #[cfg(test)]
        if let Some(hook) = &self.before_compile {
            hook(text);
        }
        match self.compiler.compile(kind, text) {
            Ok(compiled) => self.cache().keep(compiled).map(Ok),
            Err(refused) => Ok(Err(refused)),
        }
    }

    /// Whether the Lark grammar compiles; a refusal is llguidance's.
    pub(super) fn check_grammar(&self, grammar: &str) -> Result<(), BackendError> {
        self.check(GrammarKind::Lark, grammar)
    }

    /// Whether the JSON Schema (as llguidance reads it) compiles on its own.
    pub(super) fn check_json_schema(&self, schema: &str) -> Result<(), BackendError> {
        self.check(GrammarKind::JsonSchema, schema)
    }

    fn check(&self, kind: GrammarKind, text: &str) -> Result<(), BackendError> {
        if self.cache().is_cached(kind, text) {
            return Ok(());
        }
        match self.ensure(kind, text) {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(refused)) => Err(BackendError::Unenforceable(refused)),
            Err(error) => Err(BackendError::Failed(error.message)),
        }
    }

    /// A constraint at the start of the grammar.
    pub(super) fn constraint(&self, grammar: &str) -> Result<Box<dyn TokenConstraint>, BackendError> {
        match self.ensure(GrammarKind::Lark, grammar) {
            Ok(Ok(matcher)) => Ok(Box::new(MatcherConstraint(matcher))),
            Ok(Err(refused)) => Err(BackendError::Unenforceable(refused)),
            Err(error) => Err(BackendError::Failed(error.message)),
        }
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

#[cfg(test)]
#[allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    clippy::absolute_paths,
    clippy::expect_used,
    reason = "test fixture: the test thread is not a lattice worker; it waits on a channel with a bound"
)]
mod tests {
    use std::sync::Arc;
    use std::sync::mpsc::{Sender, channel};
    use std::thread::spawn;
    use std::time::Duration;

    use super::*;
    use crate::engine_backend::TextStream;

    const SIZE: usize = 130;
    const STOP: u32 = 1;
    const WAIT: Duration = Duration::from_secs(20);

    /// Printable ASCII as one token each; the rest are named only.
    struct Ascii;

    impl Codec for Ascii {
        fn encode(&self, _text: &str) -> Result<Vec<u32>, String> {
            Err("not used".to_owned())
        }

        fn stream(&self) -> Box<dyn TextStream> {
            unreachable!("the grammar tests decode nothing")
        }

        fn stop_tokens(&self) -> Vec<u32> {
            vec![STOP]
        }

        fn think_end(&self) -> Option<u32> {
            None
        }

        fn token_bytes(&self, size: usize) -> Vec<Vec<u8>> {
            (0..size)
                .map(|index| match u8::try_from(index) {
                    Ok(byte) if (0x20..0x7F).contains(&byte) => vec![byte],
                    _ => [&[0xFF][..], format!("<token {index}>").as_bytes()].concat(),
                })
                .collect()
        }

        fn end_ids(&self) -> Vec<u32> {
            vec![STOP]
        }
    }

    fn grammars() -> Grammars {
        Grammars::new(&Ascii, SIZE).expect("a vocabulary")
    }

    #[test]
    fn a_vocabulary_without_an_end_token_is_refused() {
        struct NoEnd;
        impl Codec for NoEnd {
            fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
                Ascii.encode(text)
            }
            fn stream(&self) -> Box<dyn TextStream> {
                Ascii.stream()
            }
            fn stop_tokens(&self) -> Vec<u32> {
                Vec::new()
            }
            fn think_end(&self) -> Option<u32> {
                None
            }
            fn token_bytes(&self, size: usize) -> Vec<Vec<u8>> {
                Ascii.token_bytes(size)
            }
            fn end_ids(&self) -> Vec<u32> {
                Vec::new()
            }
        }
        let reason = Grammars::new(&NoEnd, SIZE).err().expect("refused");
        assert!(reason.contains("end token"), "{reason}");
    }

    #[test]
    fn an_unprepared_matcher_allows_no_token() {
        let constraint = grammars().constraint("start: \"ab\"").expect("a constraint");
        assert!(!constraint.allows(u32::from('a')));
        let mut row = vec![0.0; SIZE];
        constraint.mask_disallowed(&mut row);
        assert!(row.iter().all(|logit| *logit == f32::NEG_INFINITY));
    }

    #[test]
    fn a_prepared_matcher_masks_the_row_and_what_lies_past_the_vocabulary() {
        let mut constraint = grammars().constraint("start: \"ab\"").expect("a constraint");
        constraint.prepare().expect("a mask");
        let mut row = vec![0.0; SIZE + 40];
        constraint.mask_disallowed(&mut row);
        for (id, logit) in row.iter().enumerate() {
            let allowed = id == usize::from(b'a');
            assert_eq!(*logit == 0.0, allowed, "token {id}");
        }
    }

    #[test]
    fn a_token_the_grammar_does_not_allow_is_refused_and_the_matcher_goes_on() {
        let mut constraint = grammars().constraint("start: \"ab\"").expect("a constraint");
        constraint.prepare().expect("a mask");
        let refused = constraint.accept(u32::from('z')).expect_err("z is not in the grammar");
        assert!(!refused.to_string().is_empty());
        constraint.prepare().expect("the step is still open");
        assert!(constraint.allows(u32::from('a')));
    }

    #[test]
    fn a_failed_matcher_reports_the_failure_at_every_step() {
        let mut constraint = MatcherConstraint(GrammarMatcher::failed("llguidance broke"));
        let prepared = constraint.prepare().expect_err("no mask");
        assert!(prepared.to_string().contains("llguidance broke"), "{prepared}");
        let accepted = constraint.accept(u32::from('a')).expect_err("no token");
        assert!(accepted.to_string().contains("llguidance broke"), "{accepted}");
        assert!(!constraint.allows(u32::from('a')));
        let mut row = vec![0.0; SIZE];
        constraint.mask_disallowed(&mut row);
        assert!(row.iter().all(|logit| *logit == f32::NEG_INFINITY));
    }

    #[test]
    fn the_cache_is_free_while_a_grammar_compiles() {
        let (compiling, started) = channel::<()>();
        let (release, released) = channel::<()>();
        let released = Mutex::new(released);
        let compiling: Sender<()> = compiling;
        let mut shared = grammars();
        shared.before_compile = Some(Box::new(move |text| {
            if text == "start: \"slow\"" {
                compiling.send(()).ok();
                released.lock().expect("the receiver").recv_timeout(WAIT).ok();
            }
        }));
        let shared = Arc::new(shared);
        let slow = {
            let shared = shared.clone();
            spawn(move || shared.check_grammar("start: \"slow\""))
        };
        started.recv_timeout(WAIT).expect("the first request is compiling");
        assert!(
            shared.cache.try_lock().is_ok(),
            "the cache's lock is held during a compile"
        );
        let (done, finished) = channel();
        let second = {
            let shared = shared.clone();
            spawn(move || done.send(shared.check_grammar("start: \"quick\"")).ok())
        };
        let answer = finished
            .recv_timeout(WAIT)
            .expect("a second request is not held up by the first");
        assert!(answer.is_ok());
        release.send(()).ok();
        assert!(slow.join().expect("the first request").is_ok());
        second.join().expect("the second request");
        assert!(
            shared.check_grammar("start: \"slow\"").is_ok(),
            "the first grammar was kept"
        );
    }
}
