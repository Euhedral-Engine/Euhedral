//! The heads' sequence of a trellis run, over a fake: the writer's MTP block first, the heads under the output head's
//! Hessian, the checkpoint's closing files before the draft head is sealed, and a part that fails leaving the resume
//! file and sealing nothing after it.

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

use euhedral_inference_quantization::calibrate::heads::{DraftHessian, Heads, TrellisHeads, run_trellis};
use euhedral_inference_quantization::calibrate::manifest::Part;
use euhedral_inference_quantization::calibrate::{CalibrateError, Then};

const HANG: Duration = Duration::from_secs(20);

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Mtp,
    Observe,
    OutputHead,
    DraftHead(DraftHessian),
    Close,
    Seal(Part, Vec<String>),
    Drop,
}

struct Fake {
    events: Mutex<Vec<Event>>,
    /// The call that fails, by its name in `Debug` up to the first parenthesis.
    fail: Option<&'static str>,
    resume_kept: Mutex<bool>,
}

impl Fake {
    fn new(fail: Option<&'static str>) -> Arc<Fake> {
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
        let name = format!("{event:?}");
        let name = name.split('(').next().unwrap().to_owned();
        self.events.lock().unwrap().push(event);
        match self.fail {
            Some(failing) if failing == name => then(Err(CalibrateError::Failed(format!("{name} failed")))),
            _ => then(Ok(value)),
        }
    }
}

fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| name.to_string()).collect()
}

impl Heads for Fake {
    fn observe(&self, then: Then<()>) {
        self.call(Event::Observe, (), then);
    }

    fn mtp(&self, then: Then<Vec<String>>) {
        self.call(Event::Mtp, names(&["mtp.safetensors"]), then);
    }

    fn output_head(&self, then: Then<Vec<String>>) {
        self.call(Event::OutputHead, names(&["heads/output_head.npz"]), then);
    }

    fn draft_head(&self, hessian: DraftHessian, then: Then<Vec<String>>) {
        self.call(Event::DraftHead(hessian), names(&["heads/draft_head.npz"]), then);
    }

    fn seal(&self, part: Part, files: Vec<String>, then: Then<()>) {
        self.call(Event::Seal(part, files), (), then);
    }

    fn drop_checkpoint(&self, then: Then<()>) {
        *self.resume_kept.lock().unwrap() = false;
        self.call(Event::Drop, (), then);
    }
}

impl TrellisHeads for Fake {
    fn close_checkpoint(&self, then: Then<Vec<String>>) {
        self.call(Event::Close, names(&["lm_head.safetensors", "config.json"]), then);
    }
}

fn heads(fake: &Arc<Fake>) -> Result<(), CalibrateError> {
    let (sender, receiver) = channel();
    run_trellis(Arc::clone(fake), Box::new(move |result| sender.send(result).unwrap()));
    receiver.recv_timeout(HANG).expect("the heads never ended")
}

fn whole() -> Vec<Event> {
    vec![
        Event::Mtp,
        Event::Seal(Part::Mtp, names(&["mtp.safetensors"])),
        Event::Observe,
        Event::OutputHead,
        Event::Seal(Part::OutputHead, names(&["heads/output_head.npz"])),
        Event::DraftHead(DraftHessian::OutputHead),
        Event::Close,
        Event::Seal(
            Part::DraftHead,
            names(&["heads/draft_head.npz", "lm_head.safetensors", "config.json"]),
        ),
        Event::Drop,
    ]
}

#[test]
fn the_writers_mtp_comes_first_and_the_draft_head_is_sealed_only_after_the_checkpoint_is_closed() {
    let fake = Fake::new(None);
    heads(&fake).unwrap();
    assert_eq!(fake.events(), whole());
}

#[test]
fn a_part_that_fails_ends_the_sequence_and_nothing_after_it_runs_and_the_resume_file_stays() {
    let events = whole();
    for failing in ["Mtp", "Observe", "OutputHead", "DraftHead", "Close", "Drop"] {
        let fake = Fake::new(Some(failing));
        let error = heads(&fake).unwrap_err();
        assert!(error.to_string().contains(failing), "{error}");
        let at = events
            .iter()
            .position(|event| format!("{event:?}").starts_with(failing))
            .unwrap();
        assert_eq!(fake.events(), events[..=at].to_vec(), "after {failing}");
        assert_eq!(*fake.resume_kept.lock().unwrap(), failing != "Drop", "{failing}");
    }
}

#[test]
fn a_failed_seal_stops_the_sequence_where_it_is() {
    let fake = Fake::new(Some("Seal"));
    assert!(heads(&fake).is_err());
    assert_eq!(fake.events(), whole()[..2].to_vec(), "the first seal ends the run");
}
