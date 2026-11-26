//! The scripted backend follows the contract the generating routes will rely on.
#![allow(
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixture: the test thread is not a lattice worker; it waits on a channel with a bound"
)]

mod common;

use std::sync::mpsc::{Sender, channel};

use common::{HANG_GUARD, lattice};
use euhedral_inference_api_server::scripted::{COMPLETION_ID_BASE, PROMPT_ID_BASE, Script, ScriptedBackend};
use euhedral_inference_api_server::{
    Backend, BackendError, EncodedPrompt, GenerationConfig, GenerationResult, OutputSpec, TextListener,
};

enum Event {
    Text(String),
    Done(Result<GenerationResult, BackendError>),
}

struct Collect(Sender<Event>);

impl TextListener for Collect {
    fn text(&mut self, text: &str) {
        self.0.send(Event::Text(text.to_owned())).ok();
    }

    fn finished(&mut self, result: Result<GenerationResult, BackendError>) {
        self.0.send(Event::Done(result)).ok();
    }
}

fn prompt(text: &str) -> EncodedPrompt {
    EncodedPrompt {
        text: text.to_owned(),
        token_ids: ScriptedBackend::prompt_tokens(text),
    }
}

#[test]
fn a_prompt_is_encoded_on_a_worker_one_token_per_character() {
    let lattice = lattice(Some(2));
    let backend = ScriptedBackend::on(&lattice);
    let (tx, rx) = channel();
    backend.encode_prompt("abc".to_owned(), Box::new(move |encoded| drop(tx.send(encoded))));
    let encoded = rx.recv_timeout(HANG_GUARD).expect("an encoding").expect("encoded");
    assert_eq!(
        encoded.token_ids,
        vec![PROMPT_ID_BASE, PROMPT_ID_BASE + 1, PROMPT_ID_BASE + 2]
    );
    assert_eq!(encoded.token_count(), 3);
    let (tx, rx) = channel();
    backend.count_tokens("hello".to_owned(), Box::new(move |count| drop(tx.send(count))));
    assert_eq!(rx.recv_timeout(HANG_GUARD).expect("a count"), Ok(5));
    assert_eq!(backend.encoded(), 2);
    backend.release();
    lattice.close();
}

#[test]
fn a_generation_emits_its_chunks_in_order_then_a_stop_token() {
    let lattice = lattice(Some(2));
    let backend = ScriptedBackend::on(&lattice);
    let generation = backend
        .open_generation(GenerationConfig::greedy(1), OutputSpec::text())
        .expect("a generation");
    let (tx, rx) = channel();
    generation.generate(prompt("Hi"), 10, Box::new(Collect(tx)));
    let mut chunks = Vec::new();
    let result = loop {
        match rx.recv_timeout(HANG_GUARD).expect("an event") {
            Event::Text(text) => chunks.push(text),
            Event::Done(result) => break result.expect("a result"),
        }
    };
    assert_eq!(chunks, ["Hello", ", ", "world"]);
    assert_eq!(result.completion_tokens, 4);
    assert!(result.stop_token_reached);
    assert_eq!(
        result.token_ids,
        (0..4).map(|n| COMPLETION_ID_BASE + n).collect::<Vec<_>>()
    );
    generation.close();
    let recorded = backend.generations();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].started(), Some(("Hi".to_owned(), 10)));
    assert_eq!(recorded[0].close_count(), 1);
    backend.release();
    lattice.close();
}

#[test]
fn the_budget_ends_a_generation_without_a_stop_token() {
    let lattice = lattice(Some(2));
    let backend = ScriptedBackend::on(&lattice);
    let generation = backend
        .open_generation(GenerationConfig::greedy(1), OutputSpec::text())
        .expect("a generation");
    let (tx, rx) = channel();
    generation.generate(prompt("Hi"), 2, Box::new(Collect(tx)));
    let result = loop {
        if let Event::Done(result) = rx.recv_timeout(HANG_GUARD).expect("an event") {
            break result.expect("a result");
        }
    };
    assert_eq!((result.completion_tokens, result.stop_token_reached), (2, false));
    backend.release();
    lattice.close();
}

#[test]
fn cancel_stops_an_endless_generation_at_its_next_step() {
    let lattice = lattice(Some(2));
    let backend = ScriptedBackend::on(&lattice);
    backend.set_script(Script::Endless);
    let generation = backend
        .open_generation(GenerationConfig::greedy(1), OutputSpec::text())
        .expect("a generation");
    let (tx, rx) = channel();
    generation.generate(prompt("Hi"), usize::MAX, Box::new(Collect(tx)));
    assert!(matches!(rx.recv_timeout(HANG_GUARD).expect("an event"), Event::Text(_)));
    generation.cancel();
    assert!(generation.is_cancelled());
    let result = loop {
        if let Event::Done(result) = rx.recv_timeout(HANG_GUARD).expect("an event") {
            break result.expect("a result");
        }
    };
    assert!(!result.stop_token_reached);
    assert!(result.completion_tokens >= 1);
    backend.release();
    lattice.close();
}

#[test]
fn an_unavailable_backend_opens_no_generation() {
    let lattice = lattice(Some(2));
    let backend = ScriptedBackend::on(&lattice);
    backend.set_available(false);
    assert!(!backend.is_available());
    let refused = backend.open_generation(GenerationConfig::greedy(1), OutputSpec::text());
    assert!(matches!(refused, Err(BackendError::Unavailable(_))));
    assert!(backend.generations().is_empty());
    backend.release();
    lattice.close();
}

#[test]
fn work_the_lattice_refuses_still_reports_exactly_once() {
    let lattice = lattice(Some(2));
    let backend = ScriptedBackend::on(&lattice);
    let generation = backend
        .open_generation(GenerationConfig::greedy(1), OutputSpec::text())
        .expect("a generation");
    backend.refuse_work();
    let (tx, rx) = channel();
    backend.encode_prompt("abc".to_owned(), Box::new(move |encoded| drop(tx.send(encoded))));
    assert!(matches!(
        rx.recv_timeout(HANG_GUARD).expect("a report"),
        Err(BackendError::Unavailable(_))
    ));
    assert!(rx.try_recv().is_err(), "reported twice");
    let (tx, rx) = channel();
    generation.generate(prompt("abc"), 3, Box::new(Collect(tx)));
    assert!(matches!(
        rx.recv_timeout(HANG_GUARD).expect("a report"),
        Event::Done(Err(BackendError::Unavailable(_)))
    ));
    assert!(rx.try_recv().is_err(), "reported twice");
    generation.close();
    lattice.close();
}
