#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "the fake shares its record with the test through a mutex: the test reads it between calls"
)]

//! The engine's own contract over a fake stepper: every accepted request is answered, and a close never frees what a
//! running request holds.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use euhedral_data_structures::queues::MpscQueue;

use super::*;
use crate::flash::generate::Step;
use crate::generation::{GenerationListener, GenerationOutcome, QuantumEvent, Speculation};
use crate::sampling::SamplingConfig;

/// A stepper that refuses to start, or starts and never retires, and records that it was dropped.
struct Fake {
    refuse: bool,
    dropped: Arc<AtomicBool>,
}

impl Stepper for Fake {
    fn begin(&mut self, _step: &Step) -> Result<(), String> {
        if self.refuse { Err("no lane".to_owned()) } else { Ok(()) }
    }

    fn selected(&self) -> Option<u32> {
        None
    }

    fn row(&self) -> Option<&[u8]> {
        None
    }

    fn commit(&mut self, _next_context: &[i32]) -> Result<(), String> {
        Ok(())
    }

    fn discard(&mut self) {}
}

impl Drop for Fake {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

struct Told(Arc<Mutex<Vec<String>>>);

impl GenerationListener for Told {
    fn prefill_done(&mut self, _at: Instant) {}
    fn token(&mut self, _id: u32, _at: Instant) {}
    fn quantum(&mut self, _event: QuantumEvent) {}
    fn finished(&mut self, outcome: GenerationOutcome) {
        self.0.lock().unwrap().push(format!("{:?}", outcome.reason));
    }
}

fn engine(refuse: bool) -> (FlashEngine<Fake>, Arc<AtomicBool>, MpscQueue<Message>) {
    let queue = MpscQueue::<Message>::new(4);
    let dropped = Arc::new(AtomicBool::new(false));
    let limits = Limits {
        vocabulary: 8,
        capacity: 64,
        chunk: 8,
    };
    let stepper = Fake {
        refuse,
        dropped: Arc::clone(&dropped),
    };
    let spawn: Spawn = Box::new(|work| {
        work();
        true
    });
    (FlashEngine::over(stepper, limits, &queue, spawn), dropped, queue)
}

fn request(told: &Arc<Mutex<Vec<String>>>) -> GenerationRequest {
    GenerationRequest {
        prompt: vec![1, 2],
        max_tokens: 1,
        stop_tokens: Vec::new(),
        speculation: Speculation::Off,
        sampling: SamplingConfig::greedy(0),
        constraint: None,
        listener: Box::new(Told(Arc::clone(told))),
    }
}

#[test]
fn a_first_quantum_that_cannot_start_is_told_to_the_listener() {
    let (engine, _, _queue) = engine(true);
    let told = Arc::new(Mutex::new(Vec::new()));
    assert!(engine.generate(request(&told), CancelHandle::default()).is_ok());
    let told = told.lock().unwrap().clone();
    assert_eq!(told.len(), 1, "the listener was not told once: {told:?}");
    assert!(
        told[0].contains("no lane"),
        "the outcome does not name the refusal: {told:?}"
    );
}

#[test]
fn an_idle_engine_closes_and_frees_its_stepper() {
    let (engine, dropped, _queue) = engine(false);
    assert!(engine.close());
    assert!(dropped.load(Ordering::SeqCst), "the stepper was not freed");
}

#[test]
fn an_engine_with_a_request_in_flight_keeps_its_stepper_when_closed() {
    let (engine, dropped, _queue) = engine(false);
    let told = Arc::new(Mutex::new(Vec::new()));
    assert!(engine.generate(request(&told), CancelHandle::default()).is_ok());
    assert!(!engine.close(), "a close with a request in flight reported success");
    assert!(
        !dropped.load(Ordering::SeqCst),
        "the stepper was freed under the running quantum"
    );
}
