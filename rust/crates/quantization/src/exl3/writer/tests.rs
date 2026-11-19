use std::sync::{Arc, Mutex};

use super::{EndOf, Parked};
use crate::checkpoint::CheckpointError;

type Ended = Arc<Mutex<Vec<(String, String)>>>;

/// A callback that records the writer and the outcome of each call it ends.
fn recording() -> (Ended, EndOf<String>) {
    let ended: Ended = Arc::default();
    let record = Arc::clone(&ended);
    let then: EndOf<String> = Arc::new(move |writer, result| {
        let outcome = result.map_or_else(|error| error.to_string(), |()| "ok".to_owned());
        record.lock().unwrap().push((writer, outcome));
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
    assert!(ended.lock().unwrap().is_empty(), "the step still holds the call");
    drop(step);
    let closed = CheckpointError::LakeClosed.to_string();
    assert_eq!(*ended.lock().unwrap(), vec![("writer".to_owned(), closed)]);
}

// Mutant caught: a step that runs and still has the guard end the call a second time when it is dropped.
#[test]
fn a_step_that_ran_does_not_end_the_call_again() {
    let (ended, then) = recording();
    let parked = Parked::new("writer".to_owned(), then, 7);
    let step = Arc::clone(&parked);
    let (writer, then, extra) = step.take().unwrap();
    assert_eq!(extra, 7);
    then(writer, Ok(()));
    drop(step);
    drop(parked);
    assert_eq!(*ended.lock().unwrap(), vec![("writer".to_owned(), "ok".to_owned())]);
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
    assert_eq!(
        *ended.lock().unwrap(),
        vec![("writer".to_owned(), "refused".to_owned())]
    );
}
