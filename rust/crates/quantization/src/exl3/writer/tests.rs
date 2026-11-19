use std::sync::Arc;

use euhedral_inference_engine::Handoff;

use super::{EndOf, Parked};
use crate::checkpoint::CheckpointError;

type Ended = Arc<Handoff<(String, String)>>;

/// A callback that hands over the writer and the outcome of the call it ends. A second call is a debug panic.
fn recording() -> (Ended, EndOf<String>) {
    let ended: Ended = Handoff::new();
    let record = Arc::clone(&ended);
    let then: EndOf<String> = Arc::new(move |writer, result| {
        let outcome = result.map_or_else(|error| error.to_string(), |()| "ok".to_owned());
        record.send((writer, outcome));
    });
    (ended, then)
}

// Mutant caught: a guard whose drop does not end the call, so a step that is dropped unrun leaves the caller waiting
// for ever.
#[test]
fn a_step_dropped_unrun_ends_the_call_with_the_writer() {
    let (ended, then) = recording();
    let parked = Parked::new("writer".to_owned(), then, ());
    let step = Arc::clone(&parked);
    drop(parked);
    assert!(ended.take().is_none(), "the step still holds the call");
    drop(step);
    let closed = CheckpointError::LakeClosed.to_string();
    assert_eq!(ended.take(), Some(("writer".to_owned(), closed)));
}

// Mutant caught: a step that runs and still has the guard end the call a second time when it is dropped.
#[test]
fn a_step_that_ran_does_not_end_the_call_again() {
    let (ended, then) = recording();
    let parked = Parked::new("writer".to_owned(), then, 7);
    let step = Arc::clone(&parked);
    let (writer, then, extra) = step.take().unwrap().go();
    assert_eq!(extra, 7);
    then(writer, Ok(()));
    drop(step);
    drop(parked);
    assert_eq!(ended.take(), Some(("writer".to_owned(), "ok".to_owned())));
}

// Mutant caught: a refused step that lets the drop report "closed" in place of the refusal's own error, or that
// ends the call twice.
#[test]
fn a_refused_step_reports_its_own_error_once() {
    let (ended, then) = recording();
    let parked = Parked::new("writer".to_owned(), then, ());
    let step = Arc::clone(&parked);
    // The callee drops the step before it returns its refusal.
    drop(step);
    parked.refused(CheckpointError::Invalid("refused".to_owned()));
    drop(parked);
    assert_eq!(ended.take(), Some(("writer".to_owned(), "refused".to_owned())));
}

// Mutant caught: a step that takes the call and fails (here, unwinds) before it ends it, which loses the call.
#[test]
fn a_step_that_unwinds_after_taking_the_call_still_ends_it() {
    let (ended, then) = recording();
    let parked = Parked::new("writer".to_owned(), then, ());
    let step = Arc::clone(&parked);
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _taken = step.take().unwrap();
        panic!("a step failed");
    }));
    assert!(unwound.is_err());
    drop((step, parked));
    let (writer, outcome) = ended.take().unwrap();
    assert_eq!(writer, "writer");
    assert!(outcome.contains("failed before it ended"), "{outcome}");
}

// Mutant caught: `end` that also lets the guard end the call again, or that gives back a different writer.
#[test]
fn a_step_that_ends_the_call_ends_it_once() {
    let (ended, then) = recording();
    let parked = Parked::new("writer".to_owned(), then, 3);
    let mut taken = parked.take().unwrap();
    let (writer, extra) = taken.parts();
    writer.push('!');
    assert_eq!(*extra, 3);
    taken.end(Ok(()));
    drop(parked);
    assert_eq!(ended.take(), Some(("writer!".to_owned(), "ok".to_owned())));
}
