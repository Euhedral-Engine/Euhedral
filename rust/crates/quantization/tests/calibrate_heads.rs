//! The heads' sequence, over fakes: the order the parts are rounded and sealed in, what `--skip-mtp` changes, a
//! failed part leaving the resume file, and the resume file dropped only after the last part is listed; and the
//! ledger's part and drop records.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test: the fake records in a mutex and a failed unwrap is the failure"
)]

use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use euhedral_inference_quantization::calibrate::heads::{DraftHessian, Heads, run};
use euhedral_inference_quantization::calibrate::manifest::{Ledger, Manifest, Part, RESUME_FILE};
use euhedral_inference_quantization::calibrate::{CalibrateError, Then};

const HANG: Duration = Duration::from_secs(20);

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Observe,
    Mtp,
    OutputHead,
    DraftHead(DraftHessian),
    Seal(Part),
    Drop,
}

/// A fake device: logs the calls, fails one on request, and keeps the resume point a later run reads.
struct Fake {
    events: Mutex<Vec<Event>>,
    fail: Option<Event>,
    /// The resume file is present until the drop.
    resume_kept: Mutex<bool>,
}

impl Fake {
    fn new(fail: Option<Event>) -> Arc<Fake> {
        Arc::new(Fake {
            events: Mutex::new(Vec::new()),
            fail,
            resume_kept: Mutex::new(true),
        })
    }

    fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }

    fn call<T>(&self, event: Event, value: T, then: Then<T>) {
        self.events.lock().unwrap().push(event.clone());
        if self.fail.as_ref() == Some(&event) {
            then(Err(CalibrateError::Failed(format!("{event:?} failed"))));
        } else {
            then(Ok(value));
        }
    }
}

impl Heads for Fake {
    fn observe(&self, then: Then<()>) {
        self.call(Event::Observe, (), then);
    }

    fn mtp(&self, then: Then<Vec<String>>) {
        self.call(Event::Mtp, vec!["mtp.npz".to_owned()], then);
    }

    fn output_head(&self, then: Then<Vec<String>>) {
        self.call(Event::OutputHead, vec!["output_head.npz".to_owned()], then);
    }

    fn draft_head(&self, hessian: DraftHessian, then: Then<Vec<String>>) {
        self.call(Event::DraftHead(hessian), vec!["draft_head.npz".to_owned()], then);
    }

    fn seal(&self, part: Part, _files: Vec<String>, then: Then<()>) {
        self.call(Event::Seal(part), (), then);
    }

    fn drop_checkpoint(&self, then: Then<()>) {
        *self.resume_kept.lock().unwrap() = false;
        self.call(Event::Drop, (), then);
    }
}

fn heads(fake: &Arc<Fake>, skip_mtp: bool) -> Result<(), CalibrateError> {
    let (sender, receiver) = channel();
    run(
        Arc::clone(fake),
        skip_mtp,
        Box::new(move |result| sender.send(result).unwrap()),
    );
    receiver.recv_timeout(HANG).expect("the heads never ended")
}

#[test]
fn the_parts_are_rounded_in_javas_order_each_sealed_before_the_next_and_the_draft_head_takes_the_mtp_hessian() {
    let fake = Fake::new(None);
    heads(&fake, false).unwrap();
    assert_eq!(
        fake.events(),
        vec![
            Event::Observe,
            Event::Mtp,
            Event::Seal(Part::Mtp),
            Event::OutputHead,
            Event::Seal(Part::OutputHead),
            Event::DraftHead(DraftHessian::Mtp),
            Event::Seal(Part::DraftHead),
            Event::Drop,
        ]
    );
}

#[test]
fn skipping_the_mtp_block_skips_it_and_rounds_the_draft_head_under_the_output_heads_hessian() {
    let fake = Fake::new(None);
    heads(&fake, true).unwrap();
    assert_eq!(
        fake.events(),
        vec![
            Event::Observe,
            Event::OutputHead,
            Event::Seal(Part::OutputHead),
            Event::DraftHead(DraftHessian::OutputHead),
            Event::Seal(Part::DraftHead),
            Event::Drop,
        ]
    );
}

#[test]
fn a_part_that_fails_ends_the_sequence_with_its_error_and_the_resume_file_stays() {
    let all = [
        Event::Observe,
        Event::Mtp,
        Event::Seal(Part::Mtp),
        Event::OutputHead,
        Event::Seal(Part::OutputHead),
        Event::DraftHead(DraftHessian::Mtp),
        Event::Seal(Part::DraftHead),
    ];
    for (at, failing) in all.iter().enumerate() {
        let fake = Fake::new(Some(failing.clone()));
        let error = heads(&fake, false).unwrap_err();
        assert!(error.to_string().contains("failed"), "{error}");
        assert_eq!(fake.events(), all[..=at].to_vec(), "nothing runs after {failing:?}");
        assert!(*fake.resume_kept.lock().unwrap(), "{failing:?} left the resume file");
    }
}

#[test]
fn a_failed_drop_is_the_runs_failure_after_every_part_was_sealed() {
    let fake = Fake::new(Some(Event::Drop));
    assert!(heads(&fake, false).is_err());
    assert_eq!(fake.events().last(), Some(&Event::Drop));
}

// ---- the ledger ----

fn saved(
    ledger: &Arc<Ledger>,
    record: impl FnOnce(&Arc<Ledger>, Box<dyn FnOnce(Vec<u8>, Then<()>) + Send>, Then<()>),
) -> Manifest {
    let written = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&written);
    let (sender, receiver) = channel();
    record(
        ledger,
        Box::new(move |bytes, then| {
            *sink.lock().unwrap() = Some(Manifest::parse(&bytes).unwrap());
            then(Ok(()));
        }),
        Box::new(move |result| sender.send(result).unwrap()),
    );
    receiver.recv_timeout(HANG).expect("the record never ended").unwrap();
    written.lock().unwrap().take().expect("a manifest was written")
}

#[test]
fn a_part_lists_its_files_once_and_sets_only_its_own_flag() {
    let ledger = Arc::new(Ledger::new(Manifest::new("run", 4)));
    let mtp = saved(&ledger, |ledger, write, then| {
        ledger.sealed(Part::Mtp, vec!["a.npz".to_owned(), "a.npz".to_owned()], write, then);
    });
    assert_eq!(
        (mtp.mtp_done, mtp.heads_done, mtp.files.clone()),
        (true, false, vec!["a.npz".to_owned()])
    );
    let output = saved(&ledger, |ledger, write, then| {
        ledger.sealed(Part::OutputHead, vec!["b.npz".to_owned()], write, then);
    });
    assert_eq!(
        (output.mtp_done, output.heads_done),
        (true, false),
        "the output head sets no flag"
    );
    let draft = saved(&ledger, |ledger, write, then| {
        ledger.sealed(Part::DraftHead, vec!["c.npz".to_owned()], write, then);
    });
    assert!(draft.heads_done);
    assert_eq!(draft.files, vec!["a.npz", "b.npz", "c.npz"]);
}

#[test]
fn dropping_the_resume_file_unlists_only_it() {
    let ledger = Arc::new(Ledger::new(Manifest::new("run", 4)));
    assert!(ledger.list([RESUME_FILE.to_owned(), "a.npz".to_owned()]));
    let dropped = saved(&ledger, |ledger, write, then| ledger.resume_file_dropped(write, then));
    assert_eq!(dropped.files, vec!["a.npz"]);
}
