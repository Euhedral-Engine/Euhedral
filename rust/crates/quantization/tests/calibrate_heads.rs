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

use std::mem::take;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use euhedral_inference_engine::Handoff;
use euhedral_inference_quantization::calibrate::head_state::{
    HeadState, at_pass_length, drop_resume_file, release_failed,
};
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
fn skipping_the_mtp_block_records_the_skip_and_rounds_the_draft_head_under_the_output_heads_hessian() {
    let fake = Fake::new(None);
    heads(&fake, true).unwrap();
    assert_eq!(
        fake.events(),
        vec![
            Event::Observe,
            Event::Seal(Part::MtpSkipped),
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

#[test]
fn a_run_that_skipped_the_mtp_block_is_complete_without_it_and_one_that_did_not_is_not() {
    let all_but_the_mtp_block = |skipped: bool| {
        let mut manifest = Manifest::new("run", 4).after(4);
        manifest.heads_done = true;
        manifest.mtp_skipped = skipped;
        manifest
    };
    assert!(
        !all_but_the_mtp_block(false).is_complete(),
        "the MTP block is missing and was not skipped"
    );
    assert!(all_but_the_mtp_block(true).is_complete());
    let mut mtp = all_but_the_mtp_block(false);
    mtp.mtp_done = true;
    assert!(mtp.is_complete());
    let mut early = all_but_the_mtp_block(true).after(3);
    early.heads_done = true;
    assert!(
        !early.is_complete(),
        "a skipped MTP block does not stand in for a layer"
    );
}

#[test]
fn sealing_the_skip_lists_no_file_and_sets_only_the_skip() {
    let ledger = Arc::new(Ledger::new(Manifest::new("run", 4)));
    let skipped = saved(&ledger, |ledger, write, then| {
        ledger.sealed(Part::MtpSkipped, Vec::new(), write, then);
    });
    assert_eq!(
        (skipped.mtp_skipped, skipped.mtp_done, skipped.heads_done, skipped.files),
        (true, false, false, Vec::<String>::new())
    );
}

#[test]
fn a_manifest_of_a_run_that_did_not_record_a_skip_reads_as_not_skipped() {
    let mut value: serde_json::Value = serde_json::from_slice(&Manifest::new("run", 4).to_bytes().unwrap()).unwrap();
    value.as_object_mut().unwrap().remove("mtp_skipped");
    let manifest = Manifest::parse(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(!manifest.mtp_skipped);
}

// ---- the heads' state ----

/// A buffer that counts the times it is dropped: a plain drop of a device buffer waits for the device.
struct Counted(Arc<AtomicUsize>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn state_holding(drops: &Arc<AtomicUsize>, buffers: usize) -> HeadState<Counted> {
    let mut state = HeadState::default();
    let mut made = (0..buffers).map(|_| Counted(Arc::clone(drops)));
    state.norm = made.next();
    state.hessian = made.next();
    state.mtp.table = made.next();
    state.mtp.decoded = made.next();
    state.gathered.extend(made);
    state
}

#[test]
fn every_buffer_a_failed_call_left_goes_to_the_stream_ordered_free_and_none_is_dropped() {
    let drops = Arc::new(AtomicUsize::new(0));
    let held = Handoff::empty();
    held.send(state_holding(&drops, 7));
    let mut freed = Vec::new();
    release_failed(&held, |buffers| freed = buffers);
    assert_eq!(
        freed.len(),
        7,
        "the MTP block's buffers and the gathered rows are among them"
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0, "a plain drop waits for the device");
    let mut state = held.take().expect("the state is put back");
    assert!(state.every().is_empty(), "nothing is left to free twice");
    drop(freed);
    assert_eq!(drops.load(Ordering::SeqCst), 7);
}

#[test]
fn a_state_with_nothing_in_it_frees_nothing_and_one_in_use_is_left_to_its_call() {
    let held = Handoff::empty();
    held.send(HeadState::<Counted>::default());
    release_failed(&held, |_| panic!("nothing was held"));
    assert!(held.take().is_some());
    let in_use = Handoff::<HeadState<Counted>>::empty();
    release_failed(&in_use, |_| panic!("the state was in use"));
}

// ---- the pass length of the MTP layer ----

#[test]
fn the_pass_length_is_shortened_for_the_layer_and_restored_whether_it_ran_or_failed() {
    for outcome in [Ok(()), Err(CalibrateError::Failed("the layer failed".to_owned()))] {
        let length = Arc::new(AtomicUsize::new(10));
        let seen = Arc::new(AtomicUsize::new(0));
        let (sender, receiver) = channel();
        let (during, restored) = (Arc::clone(&length), Arc::clone(&length));
        let looked = Arc::clone(&seen);
        let failing = outcome.is_err();
        at_pass_length(
            &length,
            9,
            move |ended| {
                looked.store(during.load(Ordering::SeqCst), Ordering::SeqCst);
                ended(outcome);
            },
            Box::new(move |result| {
                assert_eq!(
                    restored.load(Ordering::SeqCst),
                    10,
                    "restored before the next call runs"
                );
                sender.send(result).unwrap();
            }),
        );
        let result = receiver
            .recv_timeout(HANG)
            .expect("the layer's end never reached the caller");
        assert_eq!(result.is_err(), failing);
        assert_eq!(seen.load(Ordering::SeqCst), 9, "the layer ran at the shorter length");
        assert_eq!(length.load(Ordering::SeqCst), 10);
    }
}

// ---- the resume file's drop ----

#[derive(Debug, PartialEq, Eq)]
enum Step {
    Saved(Vec<String>),
    Removed,
}

/// Drops the resume file of a ledger that lists it, with a save and a removal that fail on request; the steps in the
/// order they happened, the result, and the files the ledger lists after.
fn dropped(fail_save: bool, fail_remove: bool) -> (Vec<Step>, Result<(), CalibrateError>, Vec<String>) {
    let ledger = Arc::new(Ledger::new(Manifest::new("run", 4)));
    assert!(ledger.list([RESUME_FILE.to_owned(), "a.npz".to_owned()]));
    let steps = Arc::new(Mutex::new(Vec::new()));
    let (saving, removing) = (Arc::clone(&steps), Arc::clone(&steps));
    let (sender, receiver) = channel();
    drop_resume_file(
        &ledger,
        move |bytes, then| {
            saving
                .lock()
                .unwrap()
                .push(Step::Saved(Manifest::parse(&bytes).unwrap().files));
            then(if fail_save {
                Err(CalibrateError::Failed("the save failed".to_owned()))
            } else {
                Ok(())
            });
        },
        move |then| {
            removing.lock().unwrap().push(Step::Removed);
            then(if fail_remove {
                Err(CalibrateError::Failed("the removal failed".to_owned()))
            } else {
                Ok(())
            });
        },
        Box::new(move |result| sender.send(result).unwrap()),
    );
    let result = receiver.recv_timeout(HANG).expect("the drop never ended");
    let steps = take(&mut *steps.lock().unwrap());
    (steps, result, ledger.manifest().unwrap().files)
}

#[test]
fn the_resume_file_is_unlisted_on_disk_before_it_is_removed() {
    let (steps, result, listed) = dropped(false, false);
    result.unwrap();
    assert_eq!(steps, vec![Step::Saved(vec!["a.npz".to_owned()]), Step::Removed]);
    assert_eq!(listed, vec!["a.npz"]);
}

#[test]
fn a_failed_save_removes_nothing_and_leaves_the_file_listed() {
    let (steps, result, listed) = dropped(true, false);
    assert!(result.unwrap_err().to_string().contains("the save failed"));
    assert_eq!(
        steps,
        vec![Step::Saved(vec!["a.npz".to_owned()])],
        "no removal after a failed save"
    );
    assert_eq!(
        listed,
        vec![RESUME_FILE, "a.npz"],
        "the manifest the run holds still lists the file"
    );
}

#[test]
fn a_failed_removal_is_the_failure_and_the_file_stays_unlisted() {
    let (steps, result, listed) = dropped(false, true);
    assert!(result.unwrap_err().to_string().contains("the removal failed"));
    assert_eq!(steps.len(), 2);
    assert_eq!(listed, vec!["a.npz"], "a manifest never lists a file that may be gone");
}
